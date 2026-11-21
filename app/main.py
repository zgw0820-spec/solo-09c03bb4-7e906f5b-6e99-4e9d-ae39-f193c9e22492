"""FastAPI application: training page plus JSON API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data, training

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="微批显存演练", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TrainRequest(BaseModel):
    samples_per_update: int = Field(default=8, ge=1, le=data.N_SAMPLES)
    initial_micro_batch: int = Field(default=8, ge=1, le=data.N_SAMPLES)
    updates: int = Field(default=6, ge=1, le=20)
    lr: float = Field(default=0.2, gt=0.0, le=10.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    seed: int = Field(default=training.DEFAULT_SEED, ge=0)
    oom_start_backward: int = Field(default=3, ge=0)
    oom_cap: int = Field(default=2, ge=0, le=data.N_SAMPLES)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/dataset")
def dataset() -> dict:
    return {
        "n_samples": data.N_SAMPLES,
        "features": data.FEATURES,
        "classes": data.CLASSES,
        "class_names": [data.class_name(c) for c in range(data.CLASSES)],
        "class_counts": data.class_counts(),
        "points": data.scatter_points(),
        "rows": data.dataset_rows(),
        "defaults": {
            "samples_per_update": 8,
            "initial_micro_batch": 8,
            "updates": 6,
            "lr": 0.2,
            "momentum": 0.9,
            "seed": training.DEFAULT_SEED,
            "oom_start_backward": 3,
            "oom_cap": 2,
        },
    }


@app.post("/api/train")
def train(request: TrainRequest) -> dict:
    config = training.TrainConfig(**request.model_dump())
    try:
        config.validated()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if config.initial_micro_batch > config.samples_per_update:
        raise HTTPException(
            status_code=422, detail="初始微批大小不能超过每次更新的样本数"
        )
    return training.run(config)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
