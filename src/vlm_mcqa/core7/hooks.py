"""Hook sites and batched capture/patch interventions on a decoder stack.

Sites (``Site.kind``):

- ``resid_pre``  input hidden state of decoder layer ``L``;
- ``resid_post`` output hidden state of decoder layer ``L``;
- ``attn``       the attention update actually added to the residual stream
                 (``self_attn`` output, or ``post_attention_layernorm`` output
                 for Gemma2-style sandwich norms);
- ``mlp``        the MLP update added to the residual stream (``mlp`` output, or
                 ``post_feedforward_layernorm`` output for sandwich norms);
- ``head``       concatenated per-head outputs entering ``self_attn.o_proj``;
                 a head is a ``head_dim`` channel slice (exact pre-projection
                 patch).

Every intervention names batch rows, token positions and values; donors are
never modified in place. Each hooked site must fire exactly once per forward.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch

SITE_KINDS = ("resid_pre", "resid_post", "attn", "mlp", "head")


@dataclass(frozen=True, order=True)
class Site:
    kind: str
    layer: int

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.layer}"


@dataclass
class Intervention:
    """Replace (or add to) ``values`` at ``positions`` for batch ``rows``.

    ``values`` has shape ``[len(positions), width]`` (or ``[width]`` for one
    position). For ``head`` sites, ``heads`` selects channel slices and
    ``values`` holds the full concatenated head vector; only the selected
    slices are written.
    """

    site: Site
    positions: Sequence[int]
    values: torch.Tensor
    rows: Sequence[int] | None = None
    heads: Sequence[int] | None = None
    mode: str = "replace"


@dataclass
class DecoderHandle:
    """Resolved decoder modules of a (vision-)language model."""

    model: Any
    decoder: Any
    decoder_path: str
    layers: Any
    norm: Any
    lm_head: Any
    num_heads: int
    head_dim: int
    sandwich_norm: bool
    softcap: float | None = None
    info: dict = field(default_factory=dict)

    def module_for(self, site: Site) -> tuple[Any, bool]:
        """Return ``(module, is_pre_hook)`` for a site."""
        layer = self.layers[site.layer]
        if site.kind == "resid_pre":
            return layer, True
        if site.kind == "resid_post":
            return layer, False
        if site.kind == "attn":
            return (layer.post_attention_layernorm if self.sandwich_norm else layer.self_attn), False
        if site.kind == "mlp":
            return (layer.post_feedforward_layernorm if self.sandwich_norm else layer.mlp), False
        if site.kind == "head":
            return layer.self_attn.o_proj, True
        raise ValueError(site.kind)

    def readout(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logit lens: final norm, unembedding and (if used) final softcap."""
        weight = self.lm_head.weight
        normed = self.norm(hidden.to(device=weight.device, dtype=weight.dtype))
        return self.unembed_normed(normed)

    def unembed_normed(self, normed: torch.Tensor) -> torch.Tensor:
        weight = self.lm_head.weight
        logits = self.lm_head(normed.to(device=weight.device, dtype=weight.dtype)).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def direct_projection(self, update: torch.Tensor, final_resid: torch.Tensor) -> torch.Tensor:
        """Direct logit attribution of an additive residual update.

        Uses the final residual's RMS as a frozen normalization scale, so that
        contributions of all updates add up to the final (pre-softcap) logits:
        ``W_U (g * u) / rms(x_final)`` computed as ``norm(u / rms(u)) * rms(u) / rms(x_final)``
        (unit-RMS input keeps the norm's epsilon negligible). Valid for
        RMSNorm-family final norms (all supported families).
        """
        weight = self.lm_head.weight
        u = update.to(device=weight.device, dtype=torch.float32)
        x = final_resid.to(device=weight.device, dtype=torch.float32)
        rms = lambda v: v.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)  # noqa: E731
        eps = float(getattr(self.norm, "variance_epsilon", getattr(self.norm, "eps", 0.0)))
        final_scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()  # same epsilon as the real final norm
        scaled = self.norm((u / rms(u)).to(weight.dtype)).float() * rms(u) / final_scale
        return (scaled.to(weight.dtype) @ weight.T).float()


def resolve_decoder(model: Any) -> DecoderHandle:
    """Find the decoder stack, final norm and unembedding of an HF model."""
    candidates = (
        "model.language_model", "language_model.model", "language_model",
        "model.language_model.model", "model", "model.model",
    )
    decoder, path = None, None
    for candidate in candidates:
        try:
            module = model.get_submodule(candidate)
        except AttributeError:
            continue
        if hasattr(module, "layers") and hasattr(module, "norm"):
            decoder, path = module, candidate
            break
    if decoder is None:
        raise ValueError("unsupported layout: no submodule with .layers and .norm")
    layers = decoder.layers
    first = layers[0]
    for name in ("self_attn", "mlp"):
        if not hasattr(first, name):
            raise ValueError(f"decoder layer lacks {name}")
    config = getattr(decoder, "config", None) or model.config
    num_heads = int(getattr(config, "num_attention_heads"))
    in_features = first.self_attn.o_proj.in_features
    if in_features % num_heads:
        raise ValueError("o_proj input is not divisible into query heads")
    sandwich = hasattr(first, "pre_feedforward_layernorm") and hasattr(first, "post_feedforward_layernorm")
    lm_head = model.get_output_embeddings()
    softcap = getattr(config, "final_logit_softcapping", None)
    info = {
        "decoder_path": path,
        "text_model_type": getattr(config, "model_type", "unknown"),
        "num_layers": len(layers),
        "num_heads": num_heads,
        "head_dim": in_features // num_heads,
        "hidden_size": int(getattr(config, "hidden_size")),
        "sandwich_norm": sandwich,
        "softcap_candidate": softcap,
    }
    return DecoderHandle(model, decoder, path, layers, decoder.norm, lm_head, num_heads,
                         in_features // num_heads, sandwich, None, info)


def _first(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"expected tensor or tensor-first tuple, got {type(value)}")


def _replace_first(value: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return tensor
    if isinstance(value, tuple):
        return (tensor, *value[1:])
    if isinstance(value, list):
        return [tensor, *value[1:]]
    raise TypeError(type(value))


def _apply(tensor: torch.Tensor, items: Sequence[Intervention], head_dim: int) -> torch.Tensor:
    out = tensor.clone()
    for item in items:
        rows = list(range(out.shape[0])) if item.rows is None else list(item.rows)
        pos = list(item.positions)
        value = item.values.to(device=out.device, dtype=out.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] != len(pos):
            raise ValueError(f"{item.site.key}: {value.shape[0]} values for {len(pos)} positions")
        index_rows = torch.tensor(rows, device=out.device)[:, None]
        index_pos = torch.tensor(pos, device=out.device)[None, :]
        if item.heads is None:
            new = value.unsqueeze(0).expand(len(rows), -1, -1)
            if item.mode == "add":
                out[index_rows, index_pos] = out[index_rows, index_pos] + new
            else:
                out[index_rows, index_pos] = new
            continue
        for head in item.heads:
            channels = slice(head * head_dim, (head + 1) * head_dim)
            block = out[index_rows, index_pos]
            block[..., channels] = value[None, :, channels].expand(len(rows), -1, -1)
            out[index_rows, index_pos] = block
    return out


@contextmanager
def hooked(
    handle: DecoderHandle,
    *,
    capture: Sequence[Site] = (),
    capture_positions: Sequence[int] | None = None,
    interventions: Sequence[Intervention] = (),
) -> Iterator[dict[str, torch.Tensor]]:
    """Capture sites (after any intervention at the same site) and patch sites.

    Captured tensors are ``[batch, len(capture_positions), width]`` on CPU in
    float32 (all positions when ``capture_positions`` is ``None``).
    """
    by_site: dict[Site, list[Intervention]] = {}
    for item in interventions:
        by_site.setdefault(item.site, []).append(item)
    sites = sorted(set(capture) | set(by_site))
    captured: dict[str, torch.Tensor] = {}
    calls = {site: 0 for site in sites}
    handles = []

    def transform(site: Site, tensor: torch.Tensor) -> torch.Tensor:
        calls[site] += 1
        if site in by_site:
            tensor = _apply(tensor, by_site[site], handle.head_dim)
        if site in capture:
            chosen = tensor if capture_positions is None else tensor[:, list(capture_positions)]
            captured[site.key] = chosen.detach().float().cpu()
        return tensor

    try:
        for site in sites:
            module, pre = handle.module_for(site)
            if pre:
                def pre_hook(_m, args, kwargs, _site=site):
                    if args:
                        return (transform(_site, args[0]), *args[1:]), kwargs
                    return args, {**kwargs, "hidden_states": transform(_site, kwargs["hidden_states"])}

                handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            else:
                def post_hook(_m, _args, output, _site=site):
                    return _replace_first(output, transform(_site, _first(output)))

                handles.append(module.register_forward_hook(post_hook))
        yield captured
        bad = {s.key: n for s, n in calls.items() if n != 1}
        if bad:
            raise RuntimeError(f"each hooked site must fire once per forward: {bad}")
    finally:
        for h in handles:
            h.remove()


def all_sites(num_layers: int, kinds: Sequence[str] = SITE_KINDS) -> list[Site]:
    return [Site(kind, layer) for kind in kinds for layer in range(num_layers)]
