"""Per-stage affine translators and the lens readout.

Follows the official tuned-lens parameterization: one ``nn.Linear(d, d, bias)``
per stage, zero-initialized and applied residually (``h + A h + b``), so an
untrained translator reproduces the Logit Lens exactly. The readout reuses the
model's own final norm, unembedding and softcap via ``DecoderHandle.readout``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


class TunedLens(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, *, bias: bool = True):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        layers = []
        for _ in range(self.num_layers):  # stages 0..num_layers-1; the final stage is identity
            layer = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=torch.float32)
            nn.init.zeros_(layer.weight)
            if bias:
                nn.init.zeros_(layer.bias)
            layers.append(layer)
        self.translators = nn.ModuleList(layers)

    def transform(self, hidden: torch.Tensor, stage: int) -> torch.Tensor:
        if stage == self.num_layers:
            return hidden
        if not 0 <= stage < self.num_layers:
            raise IndexError(f"stage {stage} outside 0..{self.num_layers}")
        layer = self.translators[stage]
        h = hidden.to(device=layer.weight.device, dtype=torch.float32)
        return h + layer(h)

    def stage_state(self, stage: int) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.translators[stage].state_dict().items()}

    def load_stage(self, stage: int, state: dict[str, torch.Tensor]) -> None:
        self.translators[stage].load_state_dict(state)


def lens_logits(handle: Any, lens: TunedLens | None, hidden: torch.Tensor, stage: int) -> torch.Tensor:
    """Logit Lens when ``lens`` is None, otherwise Tuned Lens logits for one stage."""
    if lens is not None:
        hidden = lens.transform(hidden, stage)
    return handle.readout(hidden)


def stage_file(directory: Path, stage: int) -> Path:
    return Path(directory) / f"translator_stage_{stage:02d}.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lens(directory: Path, num_layers: int, hidden_size: int) -> tuple[TunedLens, dict[str, str]]:
    """Assemble a lens from per-stage files; every stage must be present."""
    lens = TunedLens(num_layers, hidden_size)
    hashes = {}
    for stage in range(num_layers):
        path = stage_file(directory, stage)
        if not path.exists():
            raise FileNotFoundError(f"missing trained translator for stage {stage}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        lens.load_stage(stage, payload["state"])
        hashes[str(stage)] = file_sha256(path)
    return lens, hashes
