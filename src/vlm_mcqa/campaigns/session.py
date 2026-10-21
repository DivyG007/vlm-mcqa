"""Local-only explicitly gated model/session factory. No heavy import on import."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .verification import verify_inputs


def _align_qwen_rotary_inputs(module, args, kwargs):
    """Route generated multimodal positions to the Qwen RoPE buffer device.

    Accelerate cannot route tensors that Qwen creates *inside* the outer VLM
    forward.  With an explicit multi-GPU map, ``position_ids`` can therefore
    reach the shared language-model rotary module on CPU even though its
    non-persistent ``inv_freq`` buffer and hidden states are on CUDA.  Moving
    only the integer positions across that boundary preserves their values and
    makes the otherwise valid explicit placement executable.
    """
    args = tuple(args)
    kwargs = dict(kwargs)
    hidden = args[0] if args else kwargs.get("x")
    if hidden is None:
        raise RuntimeError("Qwen rotary boundary did not receive hidden states")
    target = hidden.device
    if module.inv_freq.device != target:
        raise RuntimeError(
            f"Qwen rotary hidden/buffer placement mismatch: {target} != {module.inv_freq.device}"
        )
    if len(args) >= 2:
        position_ids = args[1]
        if position_ids.device != target:
            args = (args[0], position_ids.to(target), *args[2:])
    elif "position_ids" in kwargs and kwargs["position_ids"].device != target:
        kwargs["position_ids"] = kwargs["position_ids"].to(target)
    return args, kwargs


def create_session(config: Mapping[str, Any], out_dir: str | Path, *, execute: bool = False,
                   profile: Mapping[str, Any] | None = None, verified: Mapping[str, Any] | None = None):
    if execute is not True:
        raise PermissionError("model/session creation requires explicit execution authorization")
    proof = verify_inputs(config)
    if verified is not None and proof != verified:
        raise ValueError("inputs changed since campaign preflight")
    out_dir = Path(out_dir).resolve()
    protected = [Path(config["output_dir"]).resolve().parent,
                 Path(config["data"]["directory"]).resolve(), Path(config["model"]["checkpoint_dir"]).resolve()]
    if out_dir.exists() or any(p == out_dir or out_dir in p.parents for p in protected):
        raise ValueError("Session output must be a fresh child directory, outside immutable inputs")
    for p in protected[1:]:
        if p in out_dir.parents:
            raise ValueError("Session output lies in immutable inputs")
    import dataclasses
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from ..core7.adapters import VLMAdapter, load_model_specs
    from ..core7.session import Session

    spec = load_model_specs(Path(config["model"]["models_config"]))[config["model"]["key"]]
    deployment = config["deployment"]
    spec = dataclasses.replace(spec, dtype=deployment["dtype"], device_map=deployment["device_map"],
                               attn_implementation=deployment.get("attn_implementation", spec.attn_implementation))
    if spec.lm_backbone == "gemma2" and spec.attn_implementation != "eager":
        raise ValueError("Gemma2 softcapped attention requires the preserved eager implementation")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[spec.dtype]
    kwargs = {"local_files_only": True, "device_map": spec.device_map,
              "attn_implementation": spec.attn_implementation, "low_cpu_mem_usage": True}
    model_path = config["model"]["checkpoint_dir"]
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, **spec.processor_kwargs)
    adapter = VLMAdapter(model, processor, spec)
    # Campaign-local boundary repair: normalize where the actual norm lives,
    # then explicitly route to unembedding. Existing Core-7 source is preserved.
    from types import MethodType
    def device_safe_readout(handle, hidden, *, apply_softcap=True):
        parameter = next(handle.norm.parameters())
        normed = handle.norm(hidden.to(device=parameter.device, dtype=parameter.dtype))
        return handle.unembed_normed(normed, apply_softcap=apply_softcap)
    def device_safe_projection(handle, update, final_resid):
        weight = handle.lm_head.weight
        parameter = next(handle.norm.parameters())
        u = update.to(device=weight.device, dtype=torch.float32)
        x = final_resid.to(device=weight.device, dtype=torch.float32)
        rms_u = u.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        eps = float(getattr(handle.norm, "variance_epsilon", getattr(handle.norm, "eps", 0.0)))
        scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        normed = handle.norm((u / rms_u).to(device=parameter.device, dtype=parameter.dtype))
        scaled = normed.to(device=u.device, dtype=torch.float32) * rms_u / scale
        return (scaled.to(weight.dtype) @ weight.T).float()
    adapter.handle.readout = MethodType(device_safe_readout, adapter.handle)
    adapter.handle.direct_projection = MethodType(device_safe_projection, adapter.handle)
    adapter.handle.info["campaign_norm_boundary"] = "norm_parameter_device_then_unembedding_device"
    if spec.family == "qwen_vl":
        rotary = getattr(adapter.handle.decoder, "rotary_emb", None)
        if rotary is None or not hasattr(rotary, "inv_freq"):
            raise ValueError("Qwen decoder lacks the expected shared rotary embedding")
        hook = rotary.register_forward_pre_hook(_align_qwen_rotary_inputs, with_kwargs=True)
        adapter._campaign_device_hooks = [hook]
        adapter.handle.info["campaign_qwen_rope_boundary"] = "position_ids_to_inv_freq_device"
    placement = adapter.describe().get("hf_device_map") or {"": str(adapter.device)}
    if any(str(device) in {"cpu", "disk", "meta"} for device in placement.values()):
        raise ValueError("configured policy forbids CPU/disk/meta model offload")
    selected_profile = dict(profile or config.get("session_profile", {}))
    selected_profile["batch_size"] = deployment["batch_size"]
    session = Session(adapter, Path(config["data"]["directory"]), out_dir, selected_profile,
                      seed=config["seed"], prepared_cache=deployment.get("prepared_cache", 24),
                      clean_cache=deployment.get("clean_cache", 512))
    return session, proof
