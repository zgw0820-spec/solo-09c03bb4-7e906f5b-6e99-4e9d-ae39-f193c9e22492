"""Training engine: micro-batch OOM drill with a full-batch reference.

Both paths share the same initial parameters and the same sample order.  The
micro-batch path splits each update's ``B`` samples into contiguous
micro-batches, accumulates gradients scaled by ``n / B`` so every sample has
equal weight, and performs exactly one SGD + momentum step per update.

The drill can cap how many samples a single backward pass may handle from a
chosen backward number onward.  A micro-batch above the cap performs a real
partial backward for the first ``cap`` samples and then raises
:class:`SimulatedOOM`; the partial gradient writes are rolled back by the
engine, which halves the micro-batch (never going above ``cap``) and retries.
A single sample that still exceeds the cap aborts the run at the last
completed update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from . import data

DEFAULT_SEED = 1234


class SimulatedOOM(RuntimeError):
    """Raised when a backward pass exceeds the drill's sample cap."""

    def __init__(self, size: int, cap: int, written: int, norm: float) -> None:
        super().__init__(
            f"{size} samples exceed cap {cap} after writing {written} sample gradients"
        )
        self.size = size
        self.cap = cap
        self.written = written
        self.partial_grad_norm = norm


@dataclass(frozen=True)
class TrainConfig:
    samples_per_update: int = 8
    initial_micro_batch: int = 8
    updates: int = 6
    lr: float = 0.2
    momentum: float = 0.9
    seed: int = DEFAULT_SEED
    oom_start_backward: int = 3  # 0 disables the drill
    oom_cap: int = 2

    def validated(self) -> "TrainConfig":
        if not 1 <= self.samples_per_update <= data.N_SAMPLES:
            raise ValueError("每次更新的样本数必须在 1 到数据集样本数之间")
        if not 1 <= self.initial_micro_batch <= self.samples_per_update:
            raise ValueError("初始微批大小必须在 1 到每次更新样本数之间")
        if not 1 <= self.updates <= 20:
            raise ValueError("更新次数必须在 1 到 20 之间")
        if not 0.0 < self.lr <= 10.0:
            raise ValueError("学习率必须大于 0 且不超过 10")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("动量必须在 [0, 1) 之间")
        if self.seed < 0:
            raise ValueError("随机种子不能为负数")
        if self.oom_start_backward < 0:
            raise ValueError("收紧起始反传序号不能为负数")
        if not 0 <= self.oom_cap <= data.N_SAMPLES:
            raise ValueError("微批样本上限必须在 0 到数据集样本数之间")
        return self


def initial_model(seed: int) -> nn.Linear:
    """Deterministic linear classifier initialisation (2 -> 3)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = nn.Linear(data.FEATURES, data.CLASSES)
    with torch.no_grad():
        model.weight.copy_(
            torch.randn(
                (data.CLASSES, data.FEATURES), generator=generator, dtype=torch.float32
            )
            * 0.5
        )
        model.bias.zero_()
    return model


def _make_optimizer(model: nn.Linear, config: TrainConfig) -> torch.optim.SGD:
    return torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum
    )


def _snapshot(model: nn.Linear, optimizer: torch.optim.SGD) -> dict[str, Any]:
    params = [p.detach().clone() for p in model.parameters()]
    momenta = []
    for p in model.parameters():
        buffer = optimizer.state.get(p, {}).get("momentum_buffer")
        momenta.append(
            torch.zeros_like(p) if buffer is None else buffer.detach().clone()
        )
    return {"params": params, "momentum": momenta}


def _flat(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _max_abs_diff(a: dict[str, Any], b: dict[str, Any], key: str) -> float:
    return float((_flat(a[key]) - _flat(b[key])).abs().max().item())


def _grad_snapshot(model: nn.Linear) -> list[torch.Tensor | None]:
    return [
        None if p.grad is None else p.grad.detach().clone()
        for p in model.parameters()
    ]


def _restore_grads(model: nn.Linear, snapshot: list[torch.Tensor | None]) -> None:
    for p, grad in zip(model.parameters(), snapshot):
        if grad is None:
            p.grad = None
        elif p.grad is None:
            p.grad = grad.clone()
        else:
            p.grad.copy_(grad)


def _micro_loss(model: nn.Linear, indices: list[int], total: int) -> torch.Tensor:
    """Mean cross-entropy over a chunk, scaled so each sample weighs 1/total."""
    logits = model(data.X[indices])
    return F.cross_entropy(logits, data.Y[indices]) * (len(indices) / total)


def _full_batch_loss(model: nn.Linear, indices: list[int]) -> torch.Tensor:
    logits = model(data.X[indices])
    return F.cross_entropy(logits, data.Y[indices])


def reference_training(
    config: TrainConfig, order: list[int]
) -> list[dict[str, Any]]:
    """Plain full-batch training: one backward and one step per update."""
    model = initial_model(config.seed)
    optimizer = _make_optimizer(model, config)
    snapshots = []
    for update_index in range(config.updates):
        indices = data.batch_indices(order, update_index, config.samples_per_update)
        optimizer.zero_grad(set_to_none=True)
        _full_batch_loss(model, indices).backward()
        optimizer.step()
        snapshots.append(_snapshot(model, optimizer))
    return snapshots


def _run_micro_backward(
    model: nn.Linear, chunk: list[int], total: int, cap: int | None
) -> float:
    """Backward one micro-batch, raising SimulatedOOM past the cap.

    Returns the accumulated gradient norm after the chunk is added.  When the
    drill cap applies, only the first ``cap`` samples are allowed to write
    gradients before the simulated OOM, so the caller can observe that a real
    partial backward happened.
    """
    if cap is None or len(chunk) <= cap:
        _micro_loss(model, chunk, total).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        return float(torch.linalg.vector_norm(_flat(grads)).item()) if grads else 0.0

    # Oversized: allow the first `cap` samples to really write gradients, then
    # fail.  The caller rolls these writes back.
    written = min(cap, len(chunk))
    norm = 0.0
    if written:
        _micro_loss(model, chunk[:written], total).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            norm = float(torch.linalg.vector_norm(_flat(grads)).item())
    raise SimulatedOOM(size=len(chunk), cap=cap, written=written, norm=norm)


def micro_batch_training(
    config: TrainConfig, order: list[int]
) -> dict[str, Any]:
    """Drill training with shrink-and-retry, one optimizer step per update."""
    model = initial_model(config.seed)
    optimizer = _make_optimizer(model, config)
    drill_active = config.oom_start_backward > 0
    backward_number = 0
    shrink_count = 0
    updates: list[dict[str, Any]] = []
    stopped: dict[str, Any] | None = None

    for update_index in range(config.updates):
        indices = data.batch_indices(
            order, update_index, config.samples_per_update
        )
        optimizer.zero_grad(set_to_none=True)
        size = config.initial_micro_batch
        position = 0
        attempts: list[dict[str, Any]] = []
        aborted = False

        while position < len(indices):
            chunk_size = min(size, len(indices) - position)
            chunk = indices[position : position + chunk_size]
            cap_applies = (
                drill_active and backward_number + 1 >= config.oom_start_backward
            )
            cap = config.oom_cap if cap_applies else None
            backward_number += 1
            before = _grad_snapshot(model)
            try:
                grad_norm = _run_micro_backward(
                    model, chunk, config.samples_per_update, cap
                )
            except SimulatedOOM as oom:
                _restore_grads(model, before)
                record = {
                    "backward": backward_number,
                    "size": chunk_size,
                    "written": oom.written,
                    "partial_grad_norm": oom.partial_grad_norm,
                    "cap": config.oom_cap,
                }
                if chunk_size == 1:
                    attempts.append(
                        {**record, "status": "single_sample_failed"}
                    )
                    if update_index == 0:
                        message = (
                            f"单样本反传仍超过微批上限 {config.oom_cap}，"
                            "第 1 次更新即中止，没有任何完整更新。"
                        )
                    else:
                        message = (
                            f"单样本反传仍超过微批上限 {config.oom_cap}，"
                            f"已停在最后一次完整更新（第 {update_index} 次）。"
                        )
                    stopped = {
                        "update": update_index + 1,
                        "backward": backward_number,
                        "message": message,
                        "completed_updates": update_index,
                        "sample_ids": indices,
                        "attempts": attempts,
                    }
                    aborted = True
                    break

                # Halve (rounding up), never above the cap.  The sample order
                # and the set of samples in the update are untouched.
                next_size = max(1, min(config.oom_cap, (chunk_size + 1) // 2))
                shrink_count += 1
                attempts.append(
                    {
                        **record,
                        "status": "oom",
                        "next_size": next_size,
                    }
                )
                size = next_size
                continue

            attempts.append(
                {
                    "backward": backward_number,
                    "size": chunk_size,
                    "status": "ok",
                    "grad_norm": grad_norm,
                    "cap": cap,
                }
            )
            position += chunk_size

        if aborted:
            break

        optimizer.step()
        updates.append(
            {
                "update": update_index + 1,
                "sample_ids": indices,
                "attempts": attempts,
                "optimizer_steps": 1,
                "snapshot": _snapshot(model, optimizer),
            }
        )

    reference = reference_training(config, order)

    rows = []
    for record in updates:
        index = record["update"] - 1
        ref = reference[index]
        rows.append(
            {
                "update": record["update"],
                "sample_ids": record["sample_ids"],
                "attempts": record["attempts"],
                "optimizer_steps": record["optimizer_steps"],
                "param_max_abs_diff": _max_abs_diff(
                    record["snapshot"], ref, "params"
                ),
                "momentum_max_abs_diff": _max_abs_diff(
                    record["snapshot"], ref, "momentum"
                ),
            }
        )

    return {
        "completed_updates": len(rows),
        "total_updates": config.updates,
        "backward_attempts": backward_number,
        "shrink_count": shrink_count,
        "stopped": stopped,
        "updates": rows,
        "max_param_diff": max(
            (row["param_max_abs_diff"] for row in rows), default=0.0
        ),
        "max_momentum_diff": max(
            (row["momentum_max_abs_diff"] for row in rows), default=0.0
        ),
    }


def run(config: TrainConfig) -> dict[str, Any]:
    config = config.validated()
    order = data.build_sample_order(config.seed)
    result = micro_batch_training(config, order)
    result["config"] = {
        "samples_per_update": config.samples_per_update,
        "initial_micro_batch": config.initial_micro_batch,
        "updates": config.updates,
        "lr": config.lr,
        "momentum": config.momentum,
        "seed": config.seed,
        "oom_start_backward": config.oom_start_backward,
        "oom_cap": config.oom_cap,
        "drill_active": config.oom_start_backward > 0,
    }
    return result
