"""Data models for the Minesweeper Environment."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class MinesweeperAction(BaseModel):
    """Action for the Minesweeper environment.

    action_type: "reveal" to uncover a cell, "flag" to toggle a flag
    x: column index (0-based)
    y: row index (0-based)
    """
    action_type: str  # "reveal" or "flag"
    x: int
    y: int


class MinesweeperObservation(BaseModel):
    """Observation from the Minesweeper environment.

    Grid cell values:
        "?" = hidden cell
        "F" = flagged cell
        "0"-"8" = revealed cell with adjacent mine count
        "M" = revealed mine (game over)
    """
    grid: List[List[str]]
    width: int
    height: int
    mines_total: int
    flags_placed: int
    cells_revealed: int
    total_safe_cells: int
    game_over: bool = False
    won: bool = False
    score: float = 0.0
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = {}


class MinesweeperState(BaseModel):
    """State for the Minesweeper environment."""
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = "easy"
