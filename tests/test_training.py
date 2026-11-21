"""Dependency-free checks for the training engine and HTTP layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import data, training
from app.main import app

TOL = 1e-6


def test_dataset_shape_and_balance():
    assert data.X.shape == (data.N_SAMPLES, data.FEATURES)
    assert data.Y.shape == (data.N_SAMPLES,)
    assert data.class_counts() == [data.PER_CLASS] * data.CLASSES


def test_oom_writes_partial_gradient_then_rolls_back(monkeypatch):
    """An OOM micro-batch writes ``cap`` samples' gradients, then discards them."""
    config = training.TrainConfig(
        samples_per_update=8, initial_micro_batch=8, updates=1,
        oom_start_backward=1, oom_cap=2,
    )
    seen = {}
    real_restore = training._restore_grads

    def spy(model, snapshot):
        grads = training._grad_snapshot(model)
        seen["written_norm"] = float(
            sum(0.0 if g is None else g.abs().sum().item() for g in grads)
        )
        seen["snapshot_norm"] = float(
            sum(0.0 if g is None else g.abs().sum().item() for g in snapshot)
        )
        return real_restore(model, snapshot)

    monkeypatch.setattr(training, "_restore_grads", spy)
    result = training.run(config)
    assert result["stopped"] is None
    # Gradients of the two allowed samples were really written...
    assert seen["written_norm"] > 0.0
    # ...and the pre-attempt gradient state (empty) was restored.
    assert seen["snapshot_norm"] == 0.0
    oom = result["updates"][0]["attempts"][0]
    assert oom["status"] == "oom"
    assert oom["written"] == 2
    assert oom["partial_grad_norm"] > 0.0


@pytest.mark.parametrize(
    "config",
    [
        training.TrainConfig(),
        training.TrainConfig(oom_start_backward=0, initial_micro_batch=3),
        training.TrainConfig(
            samples_per_update=9, initial_micro_batch=4,
            oom_start_backward=2, oom_cap=3, updates=5,
        ),
        training.TrainConfig(
            samples_per_update=5, initial_micro_batch=5,
            oom_start_backward=1, oom_cap=1, updates=4,
        ),
        training.TrainConfig(
            samples_per_update=96, initial_micro_batch=32,
            oom_start_backward=2, oom_cap=7, updates=3,
        ),
    ],
)
def test_matches_full_batch_reference(config):
    result = training.run(config)
    assert result["stopped"] is None
    assert result["completed_updates"] == config.updates
    assert result["max_param_diff"] <= TOL
    assert result["max_momentum_diff"] <= TOL
    for row in result["updates"]:
        expected = data.batch_indices(
            data.build_sample_order(config.seed), row["update"] - 1,
            config.samples_per_update,
        )
        assert row["sample_ids"] == expected
        assert row["optimizer_steps"] == 1
        assert sum(
            a["size"] for a in row["attempts"] if a["status"] == "ok"
        ) == config.samples_per_update


def test_shrink_keeps_original_order_and_equal_weight():
    config = training.TrainConfig(
        samples_per_update=8, initial_micro_batch=8,
        oom_start_backward=1, oom_cap=2, updates=2,
    )
    result = training.micro_batch_training(
        config, data.build_sample_order(config.seed)
    )
    for row in result["updates"]:
        sizes = [
            a["size"] for a in row["attempts"]
            if a["status"] == "ok"
        ]
        assert sizes == [2, 2, 2, 2]
        ooms = [a for a in row["attempts"] if a["status"] == "oom"]
        assert ooms[0]["size"] == 8 and ooms[0]["written"] == 2
        assert ooms[0]["partial_grad_norm"] > 0.0
        assert ooms[0]["next_size"] == 2


def test_oom_with_existing_gradient_keeps_earlier_chunks(monkeypatch):
    """Rolling back an OOM chunk must not disturb gradients written earlier.

    Update 1 is started with chunks of 4 samples: backward #1 succeeds (below
    the start threshold) and #2 is oversized, so the rollback happens while a
    non-empty gradient already exists.  The completed update must still match
    the full-batch reference, and the rollback must restore exactly the
    gradient state from before the failed chunk.
    """
    config = training.TrainConfig(
        samples_per_update=8, initial_micro_batch=4,
        oom_start_backward=2, oom_cap=2, updates=2,
    )
    order = data.build_sample_order(config.seed)
    indices = data.batch_indices(order, 0, 8)

    expected = training.initial_model(config.seed)
    training._micro_loss(expected, indices[:4], 8).backward()
    expected_grads = [p.grad.clone() for p in expected.parameters()]

    captured = {}
    real_restore = training._restore_grads

    def spy(model, snapshot):
        captured.setdefault("before", snapshot)
        return real_restore(model, snapshot)

    monkeypatch.setattr(training, "_restore_grads", spy)
    result = training.micro_batch_training(config, order)
    monkeypatch.undo()

    before = captured["before"]
    assert before is not None and all(g is not None for g in before)
    for got, want in zip(before, expected_grads):
        assert (got - want).abs().max().item() < 1e-8

    attempts = result["updates"][0]["attempts"]
    assert [a["status"] for a in attempts] == ["ok", "oom", "ok", "ok"]
    assert attempts[1]["size"] == 4 and attempts[1]["next_size"] == 2
    assert attempts[1]["partial_grad_norm"] > 0.0
    assert result["max_param_diff"] <= TOL
    assert result["max_momentum_diff"] <= TOL


def test_cap_zero_stops_at_last_complete_update():
    config = training.TrainConfig(
        samples_per_update=8, initial_micro_batch=8,
        oom_start_backward=3, oom_cap=0, updates=6,
    )
    result = training.run(config)
    assert result["completed_updates"] == 2
    assert result["stopped"]["update"] == 3
    # #3 shrinks from 8 samples straight to 1, then #4 (single sample) fails.
    assert result["stopped"]["backward"] == 4
    assert result["backward_attempts"] == 4
    assert result["shrink_count"] == 1
    assert "单样本" in result["stopped"]["message"]
    assert result["updates"][-1]["update"] == 2
    # The stopped update never ran an optimizer step.
    assert len(result["updates"]) == 2


def test_single_sample_failure_after_shrinking():
    config = training.TrainConfig(
        samples_per_update=4, initial_micro_batch=4,
        oom_start_backward=2, oom_cap=0, updates=3,
    )
    result = training.run(config)
    # Update 1 uses backward #1, which is below the start threshold.
    assert result["completed_updates"] == 1
    assert result["stopped"]["update"] == 2
    # #2 is the 4-sample chunk that shrinks to 1, #3 is the failing single sample.
    assert result["shrink_count"] == 1
    assert result["backward_attempts"] == 3
    (row,) = result["updates"]
    assert [a["status"] for a in row["attempts"]] == ["ok"]


def test_failure_on_first_update_reports_no_complete_update():
    config = training.TrainConfig(
        samples_per_update=4, initial_micro_batch=4,
        oom_start_backward=1, oom_cap=0, updates=3,
    )
    result = training.run(config)
    assert result["completed_updates"] == 0
    assert result["updates"] == []
    assert result["stopped"]["update"] == 1
    assert "没有任何完整更新" in result["stopped"]["message"]
    # Backward #1 was oversized (written=0) and shrank to 1; #2 failed.
    assert [a["status"] for a in result["stopped"]["attempts"]] == [
        "oom", "single_sample_failed"
    ]
    assert result["stopped"]["attempts"][0]["written"] == 0


def test_drill_off_matches_plain_full_batch():
    config = training.TrainConfig(
        samples_per_update=12, initial_micro_batch=4,
        oom_start_backward=0, updates=4,
    )
    result = training.run(config)
    assert result["shrink_count"] == 0
    assert result["max_param_diff"] <= TOL
    assert result["max_momentum_diff"] <= TOL


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        training.TrainConfig(samples_per_update=0).validated()
    with pytest.raises(ValueError):
        training.TrainConfig(samples_per_update=4, initial_micro_batch=5).validated()
    with pytest.raises(ValueError):
        training.TrainConfig(momentum=1.0).validated()
    with pytest.raises(ValueError):
        training.TrainConfig(oom_cap=97).validated()


def test_api_dataset_and_train():
    client = TestClient(app)
    ds = client.get("/api/dataset").json()
    assert ds["n_samples"] == data.N_SAMPLES
    assert len(ds["points"]) == data.N_SAMPLES
    assert len(ds["rows"]) == data.N_SAMPLES

    resp = client.post(
        "/api/train",
        json={
            "samples_per_update": 8,
            "initial_micro_batch": 8,
            "updates": 3,
            "oom_start_backward": 2,
            "oom_cap": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["completed_updates"] == 3
    assert body["max_param_diff"] <= TOL


def test_api_rejects_micro_batch_larger_than_update():
    client = TestClient(app)
    resp = client.post(
        "/api/train",
        json={"samples_per_update": 4, "initial_micro_batch": 8},
    )
    assert resp.status_code == 422


def test_index_served():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "微批显存演练" in resp.text
