"""Model adapters for Qwen-VL, PaliGemma, LLaVA-OneVision and InternVL.

All four families load through native Transformers classes
(``AutoModelForImageTextToText``; no ``trust_remote_code``). The vision tower is
treated as a black box. ``prepare`` runs one full VLM forward and records the
exact arguments the VLM passes to its language decoder (image-filled input
embeddings, multimodal RoPE ids, prefix/sliding masks). ``decoder_forward``
replays only the language decoder from those arguments, optionally expanded to
a batch of independently patched copies.

Nothing here downloads or loads weights at import time; weights load only in
``VLMAdapter.load``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .hooks import DecoderHandle, Intervention, Site, hooked, resolve_decoder

FAMILIES = ("qwen_vl", "paligemma", "llava_onevision", "internvl")
DEFAULT_MODELS = Path(__file__).resolve().parents[3] / "configs/core7/models.json"


@dataclass
class ModelSpec:
    key: str
    family: str
    hf_id: str
    params_b: float
    lm_backbone: str
    revision: str = "main"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device_map: str = "cuda"
    processor_kwargs: dict = field(default_factory=dict)
    prompt_prefix: str = ""
    chat_template_kwargs: dict = field(default_factory=dict)
    gated: bool = False
    notes: str = ""


def load_model_specs(path: Path | None = None) -> dict[str, ModelSpec]:
    data = json.loads(Path(path or DEFAULT_MODELS).read_text())
    specs = {}
    for key, value in data["models"].items():
        spec = ModelSpec(key=key, **value)
        if spec.family not in FAMILIES:
            raise ValueError(f"{key}: unknown family {spec.family}")
        specs[key] = spec
    return specs


@dataclass
class Prepared:
    variant_id: str
    input_ids: list[int]
    decoder_args: tuple
    decoder_kwargs: dict
    final_position: int
    clean_logits: torch.Tensor  # [V] float32 CPU, from the full VLM forward
    groups: dict[str, list[int]]
    label_ids: dict[str, int | None]
    content_ids: dict[str, int | None]
    rendered_prompt: str

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)


def _detach(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_detach(v) for v in value)
    return value


def expand_batch(value: Any, batch: int, key: str = "") -> Any:
    """Expand batch-one decoder arguments to ``batch`` identical rows."""
    if batch == 1:
        return value
    if torch.is_tensor(value):
        if key == "position_ids" and value.ndim == 3:  # multimodal RoPE: [sections, batch, T]
            return value.expand(-1, batch, -1).contiguous()
        if value.ndim >= 1 and value.shape[0] == 1:
            return value.expand(batch, *value.shape[1:]).contiguous()
        return value
    if isinstance(value, dict):
        return {k: expand_batch(v, batch, k if k in ("position_ids",) else "") for k, v in value.items()}
    return value


def continuation_tokens(tokenizer: Any, ids: Sequence[int], strings: Sequence[str]) -> dict[str, list[int]]:
    """Tokens each string adds when it directly continues the actual prompt.

    A prompt tail that round-trips through the tokenizer is decoded, each string
    is appended and re-encoded, and the prompt tokens must stay unchanged. An
    empty list means the string cannot be scored as a clean continuation.
    """
    for width in (12, 8, 6, 4, 3, 2, 1):
        tail = list(ids[-width:])
        text = tokenizer.decode(tail, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if list(tokenizer.encode(text, add_special_tokens=False)) == tail:
            break
    else:
        raise ValueError("no prompt tail round-trips through the tokenizer")
    out: dict[str, list[int]] = {}
    for string in strings:
        encoded = list(tokenizer.encode(text + string, add_special_tokens=False))
        out[string] = encoded[len(tail):] if encoded[: len(tail)] == tail else []
    return out


def _find(ids: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    width = len(needle)
    for i in range(start, len(ids) - width + 1):
        if list(ids[i:i + width]) == list(needle):
            return i
    return -1


def locate(tokenizer: Any, ids: Sequence[int], text: str, start: int) -> list[int]:
    """Token positions of ``text`` in ``ids`` (context-robust subsequence search)."""
    variants, merged_tail = [], []
    for prefix in ("", " ", "\n"):
        enc = list(tokenizer.encode(prefix + text, add_special_tokens=False))
        variants.append((enc, 0, 0))
        if len(enc) > 1:
            variants.append((enc[1:], 1, 0))  # first token may merge with preceding text
        if len(enc) > 2:
            # last token may merge with following text (e.g. Qwen's "?\n")
            merged_tail.append((enc[:-1], 0, 1))
        if len(enc) > 3:
            merged_tail.append((enc[1:-1], 1, 1))
    for enc, dropped, tail in variants + merged_tail:
        if not enc:
            continue
        at = _find(ids, enc, start)
        if at >= 0:
            return list(range(max(at - dropped, 0), min(at + len(enc) + tail, len(ids))))
    return []


def paligemma_mcq_prompt(prompt: str, prefix: str) -> str:
    """Keep the full MCQ inside PaliGemma's single VQA task line.

    The common seven-line template puts the first newline immediately after
    the question. PaliGemma's documented ``answer en {question}\n`` syntax
    makes that a poor place for the answer choices. The processor appends the
    final newline and image prefix; no model-specific chat template is used.
    """
    lines = prompt.splitlines()
    if len(lines) != 7 or lines[-2:] != ["Respond with only the option label.", "Answer:"]:
        raise ValueError("PaliGemma expected the seven-line MCQ prompt")
    return prefix + lines[0] + " Options: " + " ".join(lines[1:5]) + " Respond with only the option label."


class VLMAdapter:
    def __init__(self, model: Any, processor: Any, spec: ModelSpec, handle: DecoderHandle | None = None):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.spec = spec
        self.handle = handle or resolve_decoder(model)
        self.image_token_ids = self._image_token_ids()

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, spec: ModelSpec, *, cache_dir: str | None = None) -> "VLMAdapter":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[spec.dtype]
        kwargs = dict(revision=spec.revision, attn_implementation=spec.attn_implementation,
                      device_map=spec.device_map, cache_dir=cache_dir, low_cpu_mem_usage=True)
        try:
            model = AutoModelForImageTextToText.from_pretrained(spec.hf_id, dtype=dtype, **kwargs)
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            model = AutoModelForImageTextToText.from_pretrained(spec.hf_id, torch_dtype=dtype, **kwargs)
        processor = AutoProcessor.from_pretrained(spec.hf_id, revision=spec.revision, cache_dir=cache_dir,
                                                  **spec.processor_kwargs)
        return cls(model, processor, spec)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @property
    def num_layers(self) -> int:
        return len(self.handle.layers)

    def describe(self) -> dict[str, Any]:
        raw_map = getattr(self.model, "hf_device_map", None)
        device_map = ({str(module): str(device) for module, device in raw_map.items()}
                      if isinstance(raw_map, Mapping) else None)
        return {
            **self.handle.info,
            "model_key": self.spec.key,
            "hf_id": self.spec.hf_id,
            "family": self.spec.family,
            "model_class": type(self.model).__name__,
            "dtype": self.spec.dtype,
            "attn_implementation": self.spec.attn_implementation,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "softcap": self.handle.softcap,
            "hf_device_map": device_map,
            "device_map_devices": sorted(set(device_map.values())) if device_map else [str(self.device)],
        }

    def _image_token_ids(self) -> set[int]:
        found = set()
        for obj in (self.model.config, getattr(self.model.config, "text_config", None), self.processor):
            for name in ("image_token_id", "image_token_index", "image_token"):
                value = getattr(obj, name, None) if obj is not None else None
                if isinstance(value, str):
                    value = self.tokenizer.convert_tokens_to_ids(value)
                if isinstance(value, int) and value >= 0 and value != self.tokenizer.unk_token_id:
                    found.add(value)
        return found

    # ------------------------------------------------------------------ inputs

    def render(self, prompt: str) -> str:
        if self.spec.family == "paligemma":
            return paligemma_mcq_prompt(prompt, self.spec.prompt_prefix)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **self.spec.chat_template_kwargs)

    def encode(self, image: Any, prompt: str) -> tuple[dict[str, Any], str]:
        rendered = self.render(prompt)
        if self.spec.family == "paligemma":
            inputs = self.processor(text=rendered, images=image, return_tensors="pt")
        else:
            inputs = self.processor(text=[rendered], images=[image], return_tensors="pt")
        dtype = self.model.get_input_embeddings().weight.dtype
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(self.device, dtype=dtype) if value.is_floating_point() else value.to(self.device)
            moved[key] = value
        if moved["input_ids"].shape[0] != 1:
            raise ValueError("batch-one preparation only")
        return moved, rendered

    def token_groups(self, ids: Sequence[int], variant: Mapping[str, Any]) -> dict[str, list[int]]:
        image = [i for i, t in enumerate(ids) if t in self.image_token_ids]
        start = (max(image) + 1) if image else 0
        groups = {"image": image, "final": [len(ids) - 1]}
        question = locate(self.tokenizer, ids, variant["question"], start)
        groups["question"] = question
        cursor = question[-1] + 1 if question else start
        labels, contents = [], []
        for label, content in zip(variant["labels"], variant["option_contents"], strict=True):
            line = locate(self.tokenizer, ids, f"{label}. {content}", cursor)
            if not line:
                continue
            cursor = line[-1] + 1
            line_ids = [ids[i] for i in line]
            content_local = locate(self.tokenizer, line_ids, content, 0)
            contents.extend(line[i] for i in content_local)
            label_local = [i for i in range(len(line)) if self.tokenizer.decode([line_ids[i]]).strip() == label]
            labels.extend(line[i] for i in label_local[:1])
        groups["options"] = sorted(set(contents))
        groups["labels"] = sorted(set(labels))
        return groups

    # ------------------------------------------------------------------ forwards

    @torch.inference_mode()
    def prepare(self, variant: Mapping[str, Any], image: Any, *, extra_strings: Sequence[str] = ()) -> Prepared:
        inputs, rendered = self.encode(image, variant["prompt"])
        return self.prepare_inputs(variant, inputs, rendered, extra_strings=extra_strings)

    @torch.inference_mode()
    def prepare_inputs(self, variant: Mapping[str, Any], inputs: Mapping[str, Any], rendered: str,
                       *, extra_strings: Sequence[str] = ()) -> Prepared:
        """Full VLM forward on processor outputs, recording the decoder's arguments."""
        store: dict[str, Any] = {}

        def grab(_module, args, kwargs):
            store["args"] = _detach(tuple(args))
            store["kwargs"] = _detach(dict(kwargs))

        hook = self.handle.decoder.register_forward_pre_hook(grab, with_kwargs=True)
        try:
            output = self.model(**inputs, use_cache=False, return_dict=True)
        finally:
            hook.remove()
        if "kwargs" not in store:
            raise RuntimeError("decoder was not called by the VLM forward")
        ids = inputs["input_ids"][0].tolist()
        if output.logits.shape[1] != len(ids):
            raise RuntimeError("logit length differs from input length; image tokens not expanded?")
        kwargs = dict(store["kwargs"])
        kwargs.update(use_cache=False, past_key_values=None, output_hidden_states=False,
                      output_attentions=False, return_dict=True)
        symbols = list(dict.fromkeys([*variant["labels"], "A", "B", "C", "D", *extra_strings]))
        label_ids = {k: (v[0] if len(v) == 1 else None)
                     for k, v in continuation_tokens(self.tokenizer, ids, symbols).items()}
        content_ids = {k: (v[0] if v else None)
                       for k, v in continuation_tokens(self.tokenizer, ids, list(variant["option_contents"])).items()}
        return Prepared(
            variant_id=variant["variant_id"],
            input_ids=ids,
            decoder_args=store["args"],
            decoder_kwargs=kwargs,
            final_position=len(ids) - 1,
            clean_logits=output.logits[0, -1].float().cpu(),
            groups=self.token_groups(ids, variant),
            label_ids=label_ids,
            content_ids=content_ids,
            rendered_prompt=rendered,
        )

    @torch.inference_mode()
    def decoder_forward(
        self,
        prepared: Prepared,
        *,
        batch: int = 1,
        interventions: Sequence[Intervention] = (),
        capture: Sequence[Site] = (),
        capture_positions: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Replay the language decoder; returns final-token logits ``[batch, V]``."""
        args = tuple(expand_batch(a, batch) for a in prepared.decoder_args)
        kwargs = {k: expand_batch(v, batch, k) for k, v in prepared.decoder_kwargs.items()}
        with hooked(self.handle, capture=capture, capture_positions=capture_positions,
                    interventions=interventions) as captured:
            output = self.handle.decoder(*args, **kwargs)
        hidden = output.last_hidden_state[:, prepared.final_position]
        return self.handle.unembed_normed(hidden), captured

    @torch.inference_mode()
    def generate(self, image: Any, prompt: str, max_new_tokens: int = 6) -> str:
        inputs, _ = self.encode(image, prompt)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
