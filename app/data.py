"""Built-in dataset for the linear classifier demo.

The dataset is generated deterministically at import time so every run sees the
same 96 samples.  Sample *order* is controlled separately by the run seed.
"""

from __future__ import annotations

import torch

N_SAMPLES = 96
FEATURES = 2
CLASSES = 3
PER_CLASS = N_SAMPLES // CLASSES
DATASET_SEED = 7

_CLASS_NAMES = ("类别 0", "类别 1", "类别 2")


def build_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features, labels) for 3 Gaussian blobs of 32 points each."""
    generator = torch.Generator(device="cpu").manual_seed(DATASET_SEED)
    centers = torch.tensor(
        [[-2.0, -1.2], [2.0, -1.0], [0.0, 2.0]], dtype=torch.float32
    )
    features = []
    labels = []
    for class_index in range(CLASSES):
        blob = centers[class_index] + 0.7 * torch.randn(
            (PER_CLASS, FEATURES), generator=generator
        )
        features.append(blob)
        labels.append(
            torch.full((PER_CLASS,), class_index, dtype=torch.long)
        )
    features_t = torch.cat(features, dim=0)
    labels_t = torch.cat(labels, dim=0)
    permutation = torch.randperm(N_SAMPLES, generator=generator)
    return features_t[permutation], labels_t[permutation]


X, Y = build_dataset()


def build_sample_order(seed: int) -> list[int]:
    """A fixed cyclic permutation of the dataset used by both training paths."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    return torch.randperm(N_SAMPLES, generator=generator).tolist()


def batch_indices(order: list[int], update_index: int, batch_size: int) -> list[int]:
    """Samples of update ``update_index`` (0-based), wrapping around epochs.

    Shrinking/retrying a micro-batch never changes this list; it always defines
    the original samples and their original order for the update.
    """
    start = (update_index * batch_size) % N_SAMPLES
    return [order[(start + i) % N_SAMPLES] for i in range(batch_size)]


def class_name(label: int) -> str:
    return _CLASS_NAMES[label]


def dataset_rows() -> list[dict]:
    return [
        {
            "id": i,
            "x1": float(X[i, 0]),
            "x2": float(X[i, 1]),
            "label": int(Y[i]),
            "class_name": class_name(int(Y[i])),
        }
        for i in range(N_SAMPLES)
    ]


def class_counts() -> list[int]:
    return [int((Y == c).sum()) for c in range(CLASSES)]


def scatter_points(width: int = 520, height: int = 280) -> list[dict]:
    """Project the 2-D samples into SVG coordinates for the page preview."""
    pad = 26
    x_min, x_max = float(X[:, 0].min()), float(X[:, 0].max())
    y_min, y_max = float(X[:, 1].min()), float(X[:, 1].max())
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    points = []
    for i in range(N_SAMPLES):
        cx = pad + (float(X[i, 0]) - x_min) / span_x * (width - 2 * pad)
        cy = height - pad - (float(X[i, 1]) - y_min) / span_y * (height - 2 * pad)
        points.append(
            {
                "id": i,
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "label": int(Y[i]),
                "class_name": class_name(int(Y[i])),
            }
        )
    return points
