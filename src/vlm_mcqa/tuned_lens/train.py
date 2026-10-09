"""Fit translators one stage at a time from the hidden-state cache.

Optimizer and loss follow the official tuned-lens defaults (``train_loop.py``,
``ingredients.py``): SGD with Nesterov momentum 0.9, lr = lr_scale * (1 - 0.9),
weight decay 1e-3, no warmup, linear decay to zero, global-norm clipping 1.0,
loss KL(final || lens) with the frozen model's final distribution as target.
The per-step token budget is a frozen project setting (the paper's 2**18 tokens
per step exceeds this corpus); see ``configs/tuned_lens/train.json``.

Resumable: every evaluation writes ``translator_stage_XX.partial.pt``; a
finished stage writes ``translator_stage_XX.pt`` (best validation-KL weights)
and is skipped on restart.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from .translator import TunedLens, file_sha256, stage_file


def kl_and_agreement(handle: Any, lens: TunedLens | None, stage: int, hidden: torch.Tensor,
                     final: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token KL(final || lens) and top-1 agreement for one microbatch."""
    with torch.no_grad():
        target = F.log_softmax(handle.readout(final).float(), dim=-1)
    h = lens.transform(hidden, stage) if lens is not None else hidden
    pred = F.log_softmax(handle.readout(h).float(), dim=-1)
    kl = (target.exp() * (target - pred)).sum(-1)
    agree = (pred.argmax(-1) == target.argmax(-1)).float()
    return kl, agree


@torch.no_grad()
def evaluate_stage(handle: Any, lens: TunedLens | None, stage: int, hidden: torch.Tensor,
                   final: torch.Tensor, microbatch: int) -> dict[str, float]:
    kls, agrees = [], []
    for start in range(0, hidden.shape[0], microbatch):
        kl, agree = kl_and_agreement(handle, lens, stage, hidden[start:start + microbatch],
                                     final[start:start + microbatch])
        kls.append(kl.double().cpu())
        agrees.append(agree.double().cpu())
    kl, agree = torch.cat(kls), torch.cat(agrees)
    return {"kl": float(kl.mean()), "top1_agreement": float(agree.mean()), "tokens": int(kl.numel())}


def train_stage(handle: Any, lens: TunedLens, stage: int, train: tuple[torch.Tensor, torch.Tensor],
                val: tuple[torch.Tensor, torch.Tensor], config: Mapping[str, Any], out_dir: Path,
                *, tracker: Any = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = stage_file(out_dir, stage)
    partial_path = final_path.with_name(final_path.stem + ".partial.pt")
    if final_path.exists():
        payload = torch.load(final_path, map_location="cpu", weights_only=True)
        lens.load_stage(stage, payload["state"])
        log(f"[train] stage {stage} already complete; skipped")
        return payload["summary"]
    device = lens.translators[stage].weight.device
    h_train, f_train = (t.to(device) for t in train)
    h_val, f_val = (t.to(device) for t in val)
    steps, per_step, micro = int(config["steps"]), int(config["tokens_per_step"]), int(config["microbatch_tokens"])
    momentum = float(config["momentum"])
    params = list(lens.translators[stage].parameters())
    optimizer = torch.optim.SGD(params, lr=float(config["lr_scale"]) * (1 - momentum), momentum=momentum,
                                nesterov=True, weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / steps))
    generator = torch.Generator().manual_seed(int(config["seed"]) * 1000 + stage)
    n_train = h_train.shape[0]
    order, cursor, step = torch.randperm(n_train, generator=generator), 0, 0
    history: list[dict[str, Any]] = []
    logit_val = evaluate_stage(handle, None, stage, h_val, f_val, micro)
    best = {"kl": math.inf}
    if partial_path.exists():
        state = torch.load(partial_path, map_location="cpu", weights_only=False)
        lens.load_stage(stage, state["state"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        order, cursor, step = state["order"], state["cursor"], state["step"]
        history, best = state["history"], state["best"]
        log(f"[train] stage {stage} resumed at step {step}")
    started = time.time()

    def evaluate_and_checkpoint() -> None:
        nonlocal best
        metrics = evaluate_stage(handle, lens, stage, h_val, f_val, micro)
        record = {"step": step, "val_kl": metrics["kl"], "val_top1": metrics["top1_agreement"],
                  "lr": scheduler.get_last_lr()[0]}
        history.append(record)
        if metrics["kl"] < best["kl"]:
            best = {"kl": metrics["kl"], "top1_agreement": metrics["top1_agreement"], "step": step,
                    "state": lens.stage_state(stage)}
        if tracker is not None:
            tracker.log({f"stage_{stage:02d}/val_kl": metrics["kl"], f"stage_{stage:02d}/val_top1": metrics["top1_agreement"],
                         f"stage_{stage:02d}/step": step})
        tmp = partial_path.with_name(partial_path.name + ".tmp")
        torch.save({"state": lens.stage_state(stage), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
                    "order": order, "cursor": cursor, "step": step, "history": history, "best": best}, tmp)
        os.replace(tmp, partial_path)

    if step == 0 and not history:
        evaluate_and_checkpoint()  # step-0 translator is the Logit Lens
    while step < steps:
        optimizer.zero_grad(set_to_none=True)
        picked = []
        while sum(len(p) for p in picked) < per_step:
            need = per_step - sum(len(p) for p in picked)
            take = order[cursor:cursor + need]
            picked.append(take)
            cursor += len(take)
            if cursor >= n_train:
                order, cursor = torch.randperm(n_train, generator=generator), 0
        index = torch.cat(picked).to(device)
        loss_total = 0.0
        for start in range(0, len(index), micro):
            part = index[start:start + micro]
            kl, _ = kl_and_agreement(handle, lens, stage, h_train[part], f_train[part])
            loss = kl.sum() / len(index)
            loss.backward()
            loss_total += float(loss)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(config["grad_clip"])))
        optimizer.step()
        scheduler.step()
        step += 1
        if tracker is not None and step % int(config.get("log_every", 10)) == 0:
            tracker.log({f"stage_{stage:02d}/train_kl": loss_total, f"stage_{stage:02d}/grad_norm": grad_norm,
                         f"stage_{stage:02d}/step": step})
        if step % int(config["eval_every"]) == 0 or step == steps:
            evaluate_and_checkpoint()
    if not math.isfinite(best["kl"]):
        raise RuntimeError(f"stage {stage}: no finite validation KL")
    lens.load_stage(stage, best["state"])
    summary = {"stage": stage, "logit_lens_val_kl": logit_val["kl"], "logit_lens_val_top1": logit_val["top1_agreement"],
               "tuned_val_kl": best["kl"], "tuned_val_top1": best["top1_agreement"], "best_step": best["step"],
               "train_tokens": int(n_train), "val_tokens": int(h_val.shape[0]), "steps": steps,
               "seconds": time.time() - started, "history": history,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0}
    tmp = final_path.with_name(final_path.name + ".tmp")
    torch.save({"state": best["state"], "summary": summary}, tmp)
    os.replace(tmp, final_path)
    summary["sha256"] = file_sha256(final_path)
    partial_path.unlink(missing_ok=True)
    log(f"[train] stage {stage}: val KL logit {logit_val['kl']:.4f} -> tuned {best['kl']:.4f} "
        f"(best step {best['step']}) top1 {logit_val['top1_agreement']:.3f} -> {best['top1_agreement']:.3f} "
        f"{summary['seconds'] / 60:.1f} min")
    if tracker is not None:
        tracker.log({"stage": stage, "summary/logit_val_kl": logit_val["kl"], "summary/tuned_val_kl": best["kl"],
                     "summary/logit_val_top1": logit_val["top1_agreement"],
                     "summary/tuned_val_top1": best["top1_agreement"]})
    return summary
