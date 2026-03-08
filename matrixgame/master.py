"""Matrix Game: collaborative grid rearrangement.

Two LLMs take turns moving objects one cell at a time on a 4×N grid.
The goal is to replicate the target pattern (row 4) in row 3.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

from clemcore.backends import Model, CustomResponseModel
from clemcore.clemgame import GameSpec, Player, GameBenchmark
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, \
    METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE

from utils.board import Board, DIRECTIONS
from utils.astar import astar_solve

logger = logging.getLogger(__name__)

# ── Response parsing ──────────────────────────────────────────────────

RESPONSE_PATTERN = re.compile(
    r"reason:\s*(?P<reason>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
    re.IGNORECASE | re.DOTALL,
)


def parse_player_response(response: str) -> Optional[Dict[str, str]]:
    """Extract reason, object, direction from a player response.
    Returns dict with keys reason/object/direction, or None on failure."""
    match = RESPONSE_PATTERN.search(response)
    if not match:
        return None
    return {
        "reason": match.group("reason").strip(),
        "object": match.group("object").strip().upper(),
        "direction": match.group("direction").strip().lower(),
    }


# ── Player classes ────────────────────────────────────────────────────

class MatrixPlayer(Player):
    def __init__(self, model: Model, role_name: str):
        super().__init__(model)
        self.role_name = role_name

    def _custom_response(self, context):
        return "reason: custom player\nmove: A down"


class AStarPlayer(Player):
    """Programmatic player that uses A* to find the best next move.

    Re-plans from the current board state on every turn so that it
    adapts to moves made by a (possibly non-optimal) LLM partner.
    """

    def __init__(self, role_name: str, board: Board, goal_row: int, target_row: int):
        super().__init__(CustomResponseModel())
        self.role_name = role_name
        self.board = board
        self.goal_row = goal_row
        self.target_row = target_row

    def _custom_response(self, context):
        plan = astar_solve(self.board, self.goal_row, self.target_row)
        if plan:
            obj, direction = plan[0]
            return f"reason: A* planned move\nmove: {obj} {direction}"
        return "reason: A* could not find a solution\nmove: A down"


# ── Game master ───────────────────────────────────────────────────────

TURN_MOVES = "Turn Moves"
MOVE_COUNT = "Move Count"


class MatrixGameMaster(DialogueGameMaster):

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        self.request_counts: int = 0
        self.parsed_request_counts: int = 0
        self.violated_request_counts: int = 0
        self.move_log: List[Dict] = []

    # ── setup ─────────────────────────────────────────────────────────

    def _on_setup(self, **game_instance):
        self.num_columns = game_instance["num_columns"]
        self.num_rows = game_instance["num_rows"]
        self.target_row = game_instance["target_row"]
        self.goal_row = game_instance["goal_row"]
        self.max_retries = game_instance.get("max_retries", 2)
        self.strict = game_instance.get("strict", False)
        self.max_turns = game_instance.get("max_turns", 50)

        # Increase max output tokens to avoid response truncation
        for model in self.player_models:
            model.set_gen_arg("max_tokens", 4096)

        # Build board from instance data
        self.board = Board(self.num_columns, self.num_rows, self.target_row)
        for obj, (r, c) in game_instance["start_positions"].items():
            self.board.place_object(obj, r, c)
        target_pattern = game_instance["target_pattern"]  # list of N entries
        self.board.set_target_row(target_pattern)

        # State
        self.success = False
        self.aborted = False
        self.reprompt_attempts = 0
        self.current_parsed: Optional[Dict[str, str]] = None
        self.valid_response = False
        self.seen_states: Dict[str, int] = {self.board.state_key(): 1}

        prompt_template = game_instance["player_prompt"]
        prompt = prompt_template.replace("$NUM_COLUMNS$", str(self.num_columns))
        board_text = self.board.render()
        initial_context = f"{prompt}\n\nCurrent board:\n{board_text}"

        # Players (added in order → turn order)
        model_a = self.player_models[0]
        model_b = self.player_models[1] if len(self.player_models) > 1 else self.player_models[0]

        self.player_a = self._make_player(model_a, "Player A")
        self.player_b = self._make_player(model_b, "Player B")
        self.add_player(self.player_a, initial_context=initial_context)
        self.add_player(self.player_b, initial_context=initial_context)

    def _make_player(self, model: Model, role_name: str) -> Player:
        """Return an AStarPlayer for programmatic models, MatrixPlayer otherwise."""
        if isinstance(model, CustomResponseModel):
            return AStarPlayer(role_name, self.board, self.goal_row, self.target_row)
        return MatrixPlayer(model, role_name)

    # ── game loop control ─────────────────────────────────────────────

    def _does_game_proceed(self) -> bool:
        if self.success or self.aborted:
            return False
        if len(self.move_log) >= self.max_turns:
            self.aborted = True
            self.log_to_self("abort", f"Max turns ({self.max_turns}) reached")
            return False
        return True

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        parsed = parse_player_response(response)
        if parsed is None:
            self.violated_request_counts += 1
            self.valid_response = False
            self.log_to_self("invalid format", f"{player.role_name}: could not parse response")
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid responses")
            else:
                reprompt = (
                    "Your response could not be parsed. Please respond exactly in this format:\n"
                    "reason: <your reasoning>\n"
                    "move: <OBJECT> <DIRECTION>\n\n"
                    f"Current board:\n{self.board.render()}"
                )
                self.set_context_for(player, reprompt)
            return False

        # Validate the move on the board
        obj = parsed["object"]
        direction = parsed["direction"]
        error = self.board.validate_move(obj, direction)
        if error:
            self.violated_request_counts += 1
            self.valid_response = False
            self.log_to_self("invalid move", f"{player.role_name}: {error}")

            # Strict mode: any illegal move immediately ends the game
            if self.strict:
                self.aborted = True
                self.log_to_self("abort", f"Strict mode: game ended due to illegal move by {player.role_name}")
                return False

            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid moves")
            else:
                reprompt = (
                    f"Invalid move: {error}\n"
                    "Please try again with a valid move.\n\n"
                    "reason: <your reasoning>\n"
                    "move: <OBJECT> <DIRECTION>\n\n"
                    f"Current board:\n{self.board.render()}"
                )
                self.set_context_for(player, reprompt)
            return False

        self.parsed_request_counts += 1
        self.current_parsed = parsed
        self.valid_response = True
        self.reprompt_attempts = 0
        return True

    def _parse_response(self, player: Player, response: str) -> str:
        """Return the parsed move as a string for logging."""
        if self.current_parsed:
            return f"{self.current_parsed['object']} {self.current_parsed['direction']}"
        return response

    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        obj = self.current_parsed["object"]
        direction = self.current_parsed["direction"]
        reason = self.current_parsed["reason"]

        # Apply the move
        self.board.apply_move(obj, direction)
        self.move_log.append({
            "player": player.role_name,
            "object": obj,
            "direction": direction,
            "reason": reason,
        })

        self.log_to_self("move", f"{player.role_name} moved {obj} {direction}")
        self.log_to_self("reason", reason)

        # Check for repeated board state (cycle detection)
        state = self.board.state_key()
        self.seen_states[state] = self.seen_states.get(state, 0) + 1
        if self.seen_states[state] >= 3:
            self.aborted = True
            self.log_to_self("abort", "Board state seen 3 times — players are stuck in a cycle")
            return

        # Check win
        if self.board.goal_reached(self.goal_row):
            self.success = True
            self.log_to_self("success", "Row 3 matches the target pattern!")
            return

        # Send updated board to the other player
        other = self.player_b if player == self.player_a else self.player_a
        board_text = self.board.render()
        context = (
            f"{player.role_name} moved {obj} {direction}.\n\n"
            f"Current board:\n{board_text}\n\n"
            "Your turn. Respond with:\n"
            "reason: <your reasoning>\n"
            "move: <OBJECT> <DIRECTION>"
        )
        self.set_context_for(other, context)

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_SUCCESS, int(self.success))
        self.log_key(METRIC_LOSE, int(not self.success and not self.aborted))
        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        self.log_key(MOVE_COUNT, len(self.move_log))
        self.log_key(TURN_MOVES, self.move_log)


# ── Scorer ────────────────────────────────────────────────────────────

class MatrixGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        aborted = episode_interactions.get(METRIC_ABORTED, 0)
        success = episode_interactions.get(METRIC_SUCCESS, 0)
        move_count = episode_interactions.get(MOVE_COUNT, 0)
        min_manhattan = self.game_instance.get("min_manhattan_distance", move_count)

        # Standard framework metrics (required by clem eval)
        self.log_episode_score(METRIC_ABORTED, aborted)
        self.log_episode_score(METRIC_SUCCESS, success)
        self.log_episode_score(METRIC_LOSE, int(not success and not aborted))

        if aborted:
            self.log_episode_score(BENCH_SCORE, np.nan)
        elif success:
            # Score relative to theoretical minimum: 100 if optimal, scales down
            score = min(100, round(min_manhattan / max(move_count, 1) * 100, 2))
            self.log_episode_score(BENCH_SCORE, score)
        else:
            self.log_episode_score(BENCH_SCORE, 0)

        self.log_episode_score(MOVE_COUNT, move_count)


# ── Benchmark entry point ─────────────────────────────────────────────

class MatrixGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> MatrixGameMaster:
        return MatrixGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> MatrixGameScorer:
        return MatrixGameScorer(self.game_name, experiment, game_instance)
