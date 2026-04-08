#!/usr/bin/env python3
"""
Minesweeper OpenEnv - Inference Script

Uses an LLM (via OpenAI-compatible client) to play Minesweeper
across 3 difficulty levels: easy, medium, hard.

Required environment variables:
    API_BASE_URL  - LLM API endpoint
    MODEL_NAME    - Model identifier (e.g. "meta-llama/Llama-3-70b-chat-hf")
    HF_TOKEN      - Hugging Face API key
"""

import json
import os
import sys
import time
import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.minimax.io/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "minimax-m2.7-highspeed")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
MAX_STEPS_PER_TASK = {"easy": 40, "medium": 80, "hard": 120}

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
)

TASKS = ["easy", "medium", "hard"]


# ── Helpers ────────────────────────────────────────────────────────────

def format_grid(grid):
    """Format grid for LLM prompt with coordinates."""
    width = len(grid[0]) if grid else 0
    header = "  " + " ".join(f"{c}" for c in range(width))
    lines = [header]
    for r, row in enumerate(grid):
        lines.append(f"{r} " + " ".join(row))
    return "\n".join(lines)


def find_safe_cells(grid):
    """Use constraint logic to find provably safe cells and definite mines."""
    height = len(grid)
    width = len(grid[0]) if grid else 0
    safe = set()
    mines = set()

    for r in range(height):
        for c in range(width):
            if grid[r][c] not in "012345678":
                continue
            num = int(grid[r][c])
            hidden = []
            flagged = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if grid[nr][nc] == "?":
                            hidden.append((nc, nr))  # x, y
                        elif grid[nr][nc] == "F":
                            flagged += 1
            remaining_mines = num - flagged
            if remaining_mines == 0 and hidden:
                safe.update(hidden)
            elif remaining_mines == len(hidden) and hidden:
                mines.update(hidden)
    return safe, mines


def build_prompt(grid, width, height, mines_total, flags_placed, cells_revealed, total_safe, step_num):
    grid_str = format_grid(grid)

    # Find logically safe cells to hint the LLM
    safe_cells, mine_cells = find_safe_cells(grid)
    hints = ""
    if safe_cells:
        safe_list = sorted(safe_cells)[:5]
        hints += f"\nLogically SAFE cells (all neighbors of satisfied numbers): {safe_list}"
    if mine_cells:
        mine_list = sorted(mine_cells)[:5]
        hints += f"\nLogically CERTAIN mines (flag these): {mine_list}"

    return f"""You are an expert Minesweeper player on a {width}x{height} grid with {mines_total} mines.

Board (x=column, y=row, 0-indexed):
{grid_str}

Stats: {cells_revealed}/{total_safe} safe cells revealed | {flags_placed} flags | Step {step_num}
{hints}

Strategy:
1. If a number cell has ALL its adjacent mines accounted for (flagged), ALL remaining hidden neighbors are SAFE.
2. If a number cell's hidden neighbor count equals remaining mines, ALL those hidden cells are MINES — flag them.
3. NEVER reveal a cell adjacent to an unsatisfied number unless you've proven it's safe.
4. Prefer revealing cells that are adjacent to "0" cells or fully-satisfied number cells.
5. If no certain deduction exists, pick the hidden cell farthest from any known mine.

Pick ONE action. Respond with ONLY this JSON:
{{"action_type": "reveal", "x": <column>, "y": <row>}}
or
{{"action_type": "flag", "x": <column>, "y": <row>}}"""


def parse_llm_response(response_text):
    """Extract action JSON from LLM response."""
    text = response_text.strip()
    # Try to find JSON in the response
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from: {text}")


def env_reset(task_id):
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action):
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Main inference loop ────────────────────────────────────────────────

def run_task(task_id):
    """Run a single task and return the score."""
    max_steps = MAX_STEPS_PER_TASK[task_id]

    # Reset environment
    result = env_reset(task_id)
    obs = result["observation"]

    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "width": obs["width"],
        "height": obs["height"],
        "mines": obs["mines_total"],
        "total_safe_cells": obs["total_safe_cells"],
    }))
    sys.stdout.flush()

    step_num = 0
    while not obs["done"] and step_num < max_steps:
        # Build prompt and call LLM
        prompt = build_prompt(
            grid=obs["grid"],
            width=obs["width"],
            height=obs["height"],
            mines_total=obs["mines_total"],
            flags_placed=obs["flags_placed"],
            cells_revealed=obs["cells_revealed"],
            total_safe=obs["total_safe_cells"],
            step_num=step_num,
        )

        # Use constraint solver first — only call LLM when no certain move exists
        safe_cells, mine_cells = find_safe_cells(obs["grid"])

        if mine_cells:
            # Flag a known mine first
            x, y = sorted(mine_cells)[0]
            action = {"action_type": "flag", "x": x, "y": y}
        elif safe_cells:
            # Reveal a provably safe cell
            x, y = sorted(safe_cells)[0]
            action = {"action_type": "reveal", "x": x, "y": y}
        else:
            # No certain deduction — ask the LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert Minesweeper player. Respond with only valid JSON actions."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=200,
                )
                llm_text = response.choices[0].message.content
                action = parse_llm_response(llm_text)
            except Exception as e:
                # Fallback: reveal a hidden cell far from numbers
                hidden_cells = [
                    (c, r)
                    for r, row in enumerate(obs["grid"])
                    for c, cell in enumerate(row)
                    if cell == "?"
                ]
                if hidden_cells:
                    x, y = hidden_cells[len(hidden_cells) // 2]
                    action = {"action_type": "reveal", "x": x, "y": y}
                else:
                    break

        # Validate action fields
        action["action_type"] = str(action.get("action_type", "reveal"))
        action["x"] = int(action.get("x", 0))
        action["y"] = int(action.get("y", 0))

        # Take step
        result = env_step(action)
        obs = result["observation"]
        step_num += 1

        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "step": step_num,
            "action": action,
            "reward": result["reward"],
            "cells_revealed": obs["cells_revealed"],
            "total_safe_cells": obs["total_safe_cells"],
            "score": obs["score"],
            "done": obs["done"],
            "won": obs["won"],
            "game_over": obs["game_over"],
        }))
        sys.stdout.flush()

    final_score = obs["score"]
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "final_score": final_score,
        "cells_revealed": obs["cells_revealed"],
        "total_safe_cells": obs["total_safe_cells"],
        "won": obs["won"],
        "game_over": obs["game_over"],
        "total_steps": step_num,
    }))
    sys.stdout.flush()

    return final_score


def main():
    print(f"Minesweeper OpenEnv Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")
    print(f"Environment: {ENV_URL}")
    print("-" * 50)
    sys.stdout.flush()

    # Wait for server to be ready
    for attempt in range(30):
        try:
            resp = requests.get(f"{ENV_URL}/health", timeout=5)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)
    else:
        print("ERROR: Environment server not reachable")
        sys.exit(1)

    scores = {}
    for task_id in TASKS:
        print(f"\n{'='*50}")
        print(f"Task: {task_id}")
        print(f"{'='*50}")
        sys.stdout.flush()

        score = run_task(task_id)
        scores[task_id] = score
        print(f"\nTask '{task_id}' score: {score:.4f}")
        sys.stdout.flush()

    # Final summary
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    for tid, sc in scores.items():
        print(f"  {tid}: {sc:.4f}")
    print(f"  Average: {avg_score:.4f}")
    print(f"{'='*50}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
