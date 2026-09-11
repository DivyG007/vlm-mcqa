"""Small, explicit hook primitives for Qwen2.5-VL causal interventions.

The functions in this module deliberately avoid model-specific monkey-patching.
They operate on public PyTorch hook boundaries and preserve tuple outputs. The
experiment runner records the exact boundary and token indices for every row.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


def output_tensor(output: Any) -> Any:
    """Return the activation tensor from a tensor or tuple module output."""
    return output[0] if isinstance(output, tuple) else output


def replace_output_tensor(output: Any, tensor: Any) -> Any:
    """Replace the activation tensor while preserving any auxiliary outputs."""
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    return tensor


def find_subsequence(sequence: Sequence[int], needle: Sequence[int]) -> list[int]:
    """Return every start index at which ``needle`` occurs in ``sequence``."""
    if not needle or len(needle) > len(sequence):
        return []
    width = len(needle)
    return [
        index
        for index in range(len(sequence) - width + 1)
        if list(sequence[index : index + width]) == list(needle)
    ]


def bbox_to_grid_indices(
    *,
    bbox: Sequence[int],
    image_size: int,
    grid_height: int,
    grid_width: int,
    image_token_indices: Sequence[int],
    expansion: int = 0,
) -> list[int]:
    """Map a pixel box to merged visual tokens using grid-cell centres.

    Qwen exposes the pre-merge grid through ``image_grid_thw``. The caller is
    responsible for dividing it by the configured spatial merge size and for
    passing the corresponding contiguous image-token positions.
    """
    if image_size <= 0 or grid_height <= 0 or grid_width <= 0:
        raise ValueError("image and grid dimensions must be positive")
    if len(image_token_indices) != grid_height * grid_width:
        raise ValueError(
            "image token count does not match merged grid: "
            f"{len(image_token_indices)} != {grid_height}*{grid_width}"
        )
    left, top, right, bottom = [float(value) for value in bbox]
    selected: list[int] = []
    for row in range(grid_height):
        for column in range(grid_width):
            x = (column + 0.5) * image_size / grid_width
            y = (row + 0.5) * image_size / grid_height
            if left <= x <= right and top <= y <= bottom:
                selected.append(row * grid_width + column)
    if expansion:
        expanded: set[int] = set()
        for flat_index in selected:
            row, column = divmod(flat_index, grid_width)
            for rr in range(max(0, row - expansion), min(grid_height, row + expansion + 1)):
                for cc in range(
                    max(0, column - expansion), min(grid_width, column + expansion + 1)
                ):
                    expanded.add(rr * grid_width + cc)
        selected = sorted(expanded)
    return [int(image_token_indices[index]) for index in selected]


def complement_indices(indices: Sequence[int], universe: Sequence[int]) -> list[int]:
    excluded = set(indices)
    return [int(index) for index in universe if index not in excluded]


def evenly_spaced_control_region(
    *, target_count: int, universe: Sequence[int], excluded: Sequence[int]
) -> list[int]:
    """Choose a deterministic same-size region outside ``excluded``."""
    available = [int(index) for index in universe if index not in set(excluded)]
    if target_count > len(available):
        raise ValueError("not enough control positions outside excluded region")
    if target_count == 0:
        return []
    if target_count == 1:
        return [available[len(available) // 2]]
    step = (len(available) - 1) / (target_count - 1)
    return [available[round(index * step)] for index in range(target_count)]


@dataclass(frozen=True)
class HookSite:
    """A named module boundary used by the experiment ledger."""

    key: str
    module: Any
    boundary: str


@contextmanager
def capture_outputs(
    sites: Iterable[HookSite], *, to_cpu: bool = True
) -> Iterator[dict[str, Any]]:
    """Capture module outputs without changing the forward pass."""
    cache: dict[str, Any] = {}
    handles = []

    for site in sites:
        def hook(_module: Any, _inputs: Any, output: Any, *, _key: str = site.key) -> None:
            tensor = output_tensor(output).detach().clone()
            cache[_key] = tensor.cpu() if to_cpu else tensor

        handles.append(site.module.register_forward_hook(hook))
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_inputs(
    sites: Iterable[HookSite], *, to_cpu: bool = True
) -> Iterator[dict[str, Any]]:
    """Capture the first positional input to a module."""
    cache: dict[str, Any] = {}
    handles = []
    for site in sites:
        def hook(_module: Any, inputs: Any, *, _key: str = site.key) -> None:
            tensor = inputs[0].detach().clone()
            cache[_key] = tensor.cpu() if to_cpu else tensor

        handles.append(site.module.register_forward_pre_hook(hook))
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_outputs(
    patches: Mapping[str, tuple[HookSite, Any, Sequence[int], Sequence[int]]]
) -> Iterator[None]:
    """Replace selected output token positions with cached donor values.

    Mapping values are ``(site, donor_tensor, recipient_positions,
    donor_positions)``. Donor tensors may live on CPU and are moved lazily.
    """
    handles = []
    for key, (site, donor, recipient_positions, donor_positions) in patches.items():
        if key != site.key:
            raise ValueError(f"patch key {key!r} does not match site key {site.key!r}")
        recipient = [int(index) for index in recipient_positions]
        donor_pos = [int(index) for index in donor_positions]
        if len(recipient) != len(donor_pos):
            raise ValueError("recipient and donor token groups must have equal size")

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            _donor: Any = donor,
            _recipient: list[int] = recipient,
            _donor_pos: list[int] = donor_pos,
        ) -> Any:
            tensor = output_tensor(output)
            patched = tensor.clone()
            donor_value = _donor.to(device=tensor.device, dtype=tensor.dtype)
            patched[:, _recipient, :] = donor_value[:, _donor_pos, :]
            return replace_output_tensor(output, patched)

        handles.append(site.module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def add_to_outputs(
    additions: Mapping[str, tuple[HookSite, Any, Sequence[int], float]]
) -> Iterator[None]:
    """Add scaled vectors to selected output token positions."""
    handles = []
    for key, (site, vector, positions, scale) in additions.items():
        if key != site.key:
            raise ValueError(f"addition key {key!r} does not match site key {site.key!r}")
        token_positions = [int(index) for index in positions]

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            _vector: Any = vector,
            _positions: list[int] = token_positions,
            _scale: float = float(scale),
        ) -> Any:
            tensor = output_tensor(output)
            patched = tensor.clone()
            value = _vector.to(device=tensor.device, dtype=tensor.dtype)
            if value.ndim == 1:
                value = value.view(1, 1, -1)
            elif value.ndim == 2:
                value = value.unsqueeze(0)
            patched[:, _positions, :] += _scale * value
            return replace_output_tensor(output, patched)

        handles.append(site.module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_head_inputs(
    patches: Mapping[str, tuple[HookSite, Any, Sequence[int], Sequence[int], Sequence[int], int]]
) -> Iterator[None]:
    """Patch head channels immediately before an attention output projection.

    Values are ``(site, donor_input, heads, recipient_tokens, donor_tokens,
    head_dim)``. The first positional input to ``o_proj`` has concatenated head
    channels, so replacing one slice is an exact pre-projection head patch.
    """
    handles = []
    for key, value in patches.items():
        site, donor, heads, recipient_tokens, donor_tokens, head_dim = value
        if key != site.key:
            raise ValueError(f"head patch key {key!r} does not match site key {site.key!r}")
        recipient = [int(index) for index in recipient_tokens]
        donor_pos = [int(index) for index in donor_tokens]
        head_ids = [int(head) for head in heads]

        def hook(
            _module: Any,
            inputs: Any,
            *,
            _donor: Any = donor,
            _heads: list[int] = head_ids,
            _recipient: list[int] = recipient,
            _donor_pos: list[int] = donor_pos,
            _head_dim: int = int(head_dim),
        ) -> tuple[Any, ...]:
            tensor = inputs[0]
            patched = tensor.clone()
            donor_value = _donor.to(device=tensor.device, dtype=tensor.dtype)
            for head in _heads:
                channel = slice(head * _head_dim, (head + 1) * _head_dim)
                patched[:, _recipient, channel] = donor_value[:, _donor_pos, channel]
            return (patched, *inputs[1:])

        handles.append(site.module.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def block_attention_edges(
    sites: Iterable[HookSite],
    *,
    query_positions: Sequence[int],
    key_positions: Sequence[int],
) -> Iterator[None]:
    """Mask selected query-to-key attention edges at given attention modules."""
    query = [int(index) for index in query_positions]
    keys = [int(index) for index in key_positions]
    handles = []
    for site in sites:
        def hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            _query: list[int] = query,
            _keys: list[int] = keys,
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            mask = kwargs.get("attention_mask")
            if mask is None:
                raise RuntimeError("attention edge ablation requires an explicit attention mask")
            if mask.ndim != 4:
                raise RuntimeError(f"unexpected attention mask shape: {tuple(mask.shape)}")
            patched_mask = mask.clone()
            blocked_value = (
                False
                if patched_mask.dtype == getattr(__import__("torch"), "bool")
                else __import__("torch").finfo(patched_mask.dtype).min
            )
            for query_index in _query:
                patched_mask[:, :, query_index, _keys] = blocked_value
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = patched_mask
            return args, kwargs

        handles.append(site.module.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def path_patch_outputs(
    *,
    sender_site: HookSite,
    sender_donor: Any,
    sender_recipient_positions: Sequence[int],
    sender_donor_positions: Sequence[int],
    receiver_site: HookSite,
    receiver_clean: Any,
    receiver_positions: Sequence[int],
) -> Iterator[None]:
    """Keep only a sender-to-receiver-token causal path alive.

    The sender tokens are replaced with donor values. At the later receiver
    boundary, every token except ``receiver_positions`` is reset to the clean
    recipient activation. Consequently, downstream effects can continue only
    through the named receiver tokens. This is token-path patching, not a claim
    to isolate every internal edge inside the intervening layers.
    """
    sender_recipient = [int(index) for index in sender_recipient_positions]
    sender_donor_pos = [int(index) for index in sender_donor_positions]
    receiver_keep = {int(index) for index in receiver_positions}
    if len(sender_recipient) != len(sender_donor_pos):
        raise ValueError("sender recipient and donor token groups must have equal size")

    def sender_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output_tensor(output)
        patched = tensor.clone()
        donor_value = sender_donor.to(device=tensor.device, dtype=tensor.dtype)
        patched[:, sender_recipient, :] = donor_value[:, sender_donor_pos, :]
        return replace_output_tensor(output, patched)

    def receiver_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output_tensor(output)
        patched = tensor.clone()
        clean_value = receiver_clean.to(device=tensor.device, dtype=tensor.dtype)
        reset = [index for index in range(tensor.shape[1]) if index not in receiver_keep]
        patched[:, reset, :] = clean_value[:, reset, :]
        return replace_output_tensor(output, patched)

    handles = [
        sender_site.module.register_forward_hook(sender_hook),
        receiver_site.module.register_forward_hook(receiver_hook),
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def bootstrap_interval(
    values: Sequence[float], *, seed: int = 42, samples: int = 2000
) -> tuple[float, float]:
    """Pure-Python percentile bootstrap interval for a mean."""
    import random

    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    width = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(width)] for _ in range(width)]
        means.append(sum(draw) / width)
    means.sort()
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def normalized_recovery(
    source_difference: float, target_difference: float, patched_difference: float
) -> float:
    denominator = target_difference - source_difference
    if abs(denominator) < 1e-8:
        return float("nan")
    return (patched_difference - source_difference) / denominator
