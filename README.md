---
title: Minesweeper OpenEnv
emoji: "\U0001F4A3"
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - minesweeper
  - puzzle
  - logic
  - deduction
---

# Minesweeper OpenEnv

A Minesweeper environment for AI agents following the OpenEnv specification. The agent must use logical deduction from number clues to safely reveal all non-mine cells.

## Environment Overview

| Property | Description |
|----------|-------------|
| **Action Space** | `reveal(x, y)` or `flag(x, y)` on grid cells |
| **Observation Space** | 2D grid with cell states: `?` (hidden), `F` (flagged), `0-8` (revealed), `M` (mine) |
| **Reward** | `0.0 - 1.0` — fraction of safe cells successfully revealed |
| **Episode End** | All safe cells revealed (win) or mine hit (game over) |

## Tasks (3 Difficulty Levels)

| Task | Grid Size | Mines | Description |
|------|-----------|-------|-------------|
| `easy` | 5x5 | 3 | Beginner — small grid, few mines |
| `medium` | 8x8 | 10 | Intermediate — requires multi-step deduction |
| `hard` | 10x10 | 20 | Expert — dense minefield, advanced logic needed |

## API Endpoints

### `POST /reset`
Reset the environment. Optionally pass `{"task_id": "easy"|"medium"|"hard"}`.

### `POST /step`
Submit an action: `{"action": {"action_type": "reveal", "x": 3, "y": 2}}`

### `GET /state`
Get current episode state (episode_id, step_count, task_id).

### `GET /tasks`
List all available tasks with descriptions.

### `GET /health`
Health check endpoint.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face API key |

## Running Locally

```bash
# Start the server
pip install -r requirements.txt
uvicorn minesweeper_env.server.app:app --host 0.0.0.0 --port 8000

# Run inference (in another terminal)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-token"
python inference.py
```

## Running with Docker

```bash
docker build -t minesweeper-openenv .
docker run -p 8000:8000 minesweeper-openenv
```

## How the Agent Works

The LLM agent:
1. Receives the current grid state with number clues
2. Reasons about which cells are safe vs mines using Minesweeper logic
3. Reveals safe cells or flags suspected mines
4. Repeats until all safe cells are revealed or a mine is hit

The reward function provides partial credit (0.0-1.0) based on the fraction of safe cells revealed, encouraging progressive exploration.

## Setup & Deployment

Deployed on Hugging Face Spaces using Docker SDK. The `inference.py` script connects to the running environment server and uses an LLM to play all three difficulty levels.
