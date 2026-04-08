"""Minesweeper Environment Implementation."""

import random
import uuid
from typing import List, Tuple, Set, Dict, Any

from .models import MinesweeperAction, MinesweeperObservation, MinesweeperState


TASKS = {
    "easy": {
        "width": 5,
        "height": 5,
        "mines": 3,
        "description": "5x5 grid with 3 mines - beginner level",
    },
    "medium": {
        "width": 8,
        "height": 8,
        "mines": 10,
        "description": "8x8 grid with 10 mines - intermediate level",
    },
    "hard": {
        "width": 10,
        "height": 10,
        "mines": 20,
        "description": "10x10 grid with 20 mines - expert level",
    },
}


class MinesweeperEnvironment:
    """Minesweeper Environment following OpenEnv spec.

    An agent must reveal all safe cells without hitting a mine.
    The environment provides number clues indicating adjacent mine counts.

    Actions:
        reveal(x, y) - Reveal a hidden cell
        flag(x, y)   - Toggle flag on a hidden cell

    Reward:
        0.0-1.0 based on fraction of safe cells revealed.
        1.0 = all safe cells revealed (win).
        On mine hit, reward stays at current progress.
    """

    def __init__(self, task_id: str = "easy", max_episode_steps: int = 200):
        self.task_id = task_id
        self.max_episode_steps = max_episode_steps
        self._configure(task_id)
        self._state = MinesweeperState()
        self._init_board()

    def _configure(self, task_id: str):
        config = TASKS.get(task_id, TASKS["easy"])
        self.width = config["width"]
        self.height = config["height"]
        self.num_mines = config["mines"]

    def _init_board(self):
        self.board: List[List[int]] = [[0] * self.width for _ in range(self.height)]
        self.revealed: List[List[bool]] = [[False] * self.width for _ in range(self.height)]
        self.flagged: List[List[bool]] = [[False] * self.width for _ in range(self.height)]
        self.mines: Set[Tuple[int, int]] = set()
        self.game_over = False
        self.won = False
        self.cells_revealed = 0
        self.total_safe = self.width * self.height - self.num_mines
        self._first_reveal = True

    def reset(self, task_id: str = None) -> MinesweeperObservation:
        if task_id:
            self.task_id = task_id
            self._configure(task_id)
        self._state = MinesweeperState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=self.task_id,
        )
        self._init_board()
        # Place mines now (will be re-placed on first reveal to avoid instant death)
        self._place_mines()
        self._calculate_numbers()
        return self._get_observation()

    def _place_mines(self, exclude: Tuple[int, int] = None):
        """Place mines randomly, optionally excluding a cell and its neighbors."""
        self.mines.clear()
        self.board = [[0] * self.width for _ in range(self.height)]

        excluded = set()
        if exclude:
            ey, ex = exclude
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = ey + dr, ex + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        excluded.add((nr, nc))

        positions = [
            (r, c) for r in range(self.height) for c in range(self.width)
            if (r, c) not in excluded
        ]

        rng = random.Random()
        chosen = rng.sample(positions, min(self.num_mines, len(positions)))
        self.mines = set(chosen)
        for r, c in self.mines:
            self.board[r][c] = -1

    def _calculate_numbers(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.board[r][c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width and self.board[nr][nc] == -1:
                            count += 1
                self.board[r][c] = count

    def step(self, action: MinesweeperAction) -> MinesweeperObservation:
        if self.game_over or self.won:
            return self._get_observation()

        x, y = action.x, action.y
        if not (0 <= y < self.height and 0 <= x < self.width):
            self._state.step_count += 1
            return self._get_observation()

        self._state.step_count += 1

        if action.action_type == "flag":
            if not self.revealed[y][x]:
                self.flagged[y][x] = not self.flagged[y][x]

        elif action.action_type == "reveal":
            if self.flagged[y][x] or self.revealed[y][x]:
                return self._get_observation()

            # On first reveal, re-place mines to guarantee safe start
            if self._first_reveal:
                self._first_reveal = False
                self._place_mines(exclude=(y, x))
                self._calculate_numbers()

            if self.board[y][x] == -1:
                # Hit a mine
                self.game_over = True
                for r, c in self.mines:
                    self.revealed[r][c] = True
            else:
                self._flood_reveal(y, x)
                if self.cells_revealed >= self.total_safe:
                    self.won = True

        if self._state.step_count >= self.max_episode_steps:
            self.game_over = True

        return self._get_observation()

    def _flood_reveal(self, r: int, c: int):
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return
        if self.revealed[r][c] or self.flagged[r][c]:
            return
        if self.board[r][c] == -1:
            return

        self.revealed[r][c] = True
        self.cells_revealed += 1

        if self.board[r][c] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._flood_reveal(r + dr, c + dc)

    def _get_observation(self) -> MinesweeperObservation:
        grid = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if self.revealed[r][c]:
                    if self.board[r][c] == -1:
                        row.append("M")
                    else:
                        row.append(str(self.board[r][c]))
                elif self.flagged[r][c]:
                    row.append("F")
                else:
                    row.append("?")
            grid.append(row)

        flags = sum(
            1 for r in range(self.height) for c in range(self.width)
            if self.flagged[r][c]
        )
        score = self.cells_revealed / self.total_safe if self.total_safe > 0 else 0.0
        done = self.game_over or self.won

        return MinesweeperObservation(
            grid=grid,
            width=self.width,
            height=self.height,
            mines_total=self.num_mines,
            flags_placed=flags,
            cells_revealed=self.cells_revealed,
            total_safe_cells=self.total_safe,
            game_over=self.game_over,
            won=self.won,
            score=round(score, 4),
            done=done,
            reward=round(score, 4),
            metadata={"task_id": self.task_id, "step_count": self._state.step_count},
        )

    @property
    def state(self) -> MinesweeperState:
        return self._state

    def get_tasks(self) -> Dict[str, Any]:
        return TASKS
