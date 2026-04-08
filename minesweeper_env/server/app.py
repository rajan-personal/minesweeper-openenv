"""FastAPI application for the Minesweeper Environment."""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from minesweeper_env.environment import MinesweeperEnvironment, TASKS
from minesweeper_env.models import MinesweeperAction

app = FastAPI(title="Minesweeper OpenEnv", version="1.0.0")

env = MinesweeperEnvironment(
    task_id=os.environ.get("DEFAULT_TASK", "easy"),
    max_episode_steps=int(os.environ.get("MAX_EPISODE_STEPS", 200)),
)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: MinesweeperAction


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": {
            tid: {
                "description": cfg["description"],
                "width": cfg["width"],
                "height": cfg["height"],
                "mines": cfg["mines"],
            }
            for tid, cfg in TASKS.items()
        }
    }


@app.post("/reset")
def reset(request: ResetRequest = None):
    task_id = request.task_id if request else None
    obs = env.reset(task_id=task_id)
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
    }


@app.post("/step")
def step(request: StepRequest):
    obs = env.step(request.action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state")
def get_state():
    return env.state.model_dump()


def run():
    import uvicorn
    uvicorn.run(
        "minesweeper_env.server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
