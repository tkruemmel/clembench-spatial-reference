"""Matrix Game (covered base): collaborative grid rearrangement on an 8×8 grid.

Two players take turns moving objects one cell at a time toward their target
positions. Player A is typically an LLM; Player B is a programmatic A* agent
that re-plans from the current board state on every turn.

Visibility modes:
  use_masking=False  — both players see all objects (Experiment 1 baseline).
  use_masking=True   — each player sees only their own objects' labels;
                       the opponent's objects appear as 'X' (Experiment 2+).
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

from clemcore.backends import Model, CustomResponseModel
from clemcore.clemgame import GameSpec, Player, GameBenchmark
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import (
    METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE,
    METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_VIOLATED,
    METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE,
)

from utils.board import Board, DIRECTIONS
from utils.astar import astar_solve

logger = logging.getLogger(__name__)

RESPONSE_PATTERN = re.compile(
    r"reason:\s*(?P<reason>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
    re.IGNORECASE | re.DOTALL,
)

TURN_MOVES = "Turn Moves"
MOVE_COUNT = "Move Count"


# ── Player classes ─────────────────────────────────────────────────────

def parse_player_response(response: str) -> Optional[Dict[str, str]]:
    match = RESPONSE_PATTERN.search(response)
    if not match:
        return None
    return {
        "reason": match.group("reason").strip(),
        "object": match.group("object").strip().upper(),
        "direction": match.group("direction").strip().lower(),
    }


class MatrixPlayer(Player):
    def __init__(self, model: Model, role_name: str, own_objects: Set[str]):
        super().__init__(model)
        self.role_name = role_name
        self.own_objects = own_objects

    def _custom_response(self, context):
        first = sorted(self.own_objects)[0]
        return f"reason: custom player\nmove: {first} down"


class AStarPlayer(Player):
    """Programmatic partner: re-plans from the current board state each turn."""

    def __init__(self, role_name: str, board: Board, own_objects: Set[str]):
        super().__init__(CustomResponseModel())
        self.role_name = role_name
        self.board = board
        self.own_objects = own_objects

    def _custom_response(self, context):
        plan = astar_solve(self.board, player_objects=self.own_objects)
        if plan:
            obj, direction = plan[0]
            return f"reason: A* planned move\nmove: {obj} {direction}"
        first = sorted(self.own_objects)[0]
        return f"reason: A* could not find a solution\nmove: {first} down"


# ── Game master ────────────────────────────────────────────────────────

class MatrixGameMaster(DialogueGameMaster):

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        self.request_counts = 0
        self.parsed_request_counts = 0
        self.violated_request_counts = 0
        self.move_log: List[Dict] = []

    # ── setup ──────────────────────────────────────────────────────────

    def _on_setup(self, **game_instance):
        self.grid_size = game_instance["grid_size"]
        self.max_retries = game_instance.get("max_retries", 2)
        self.strict = game_instance.get("strict", False)
        self.max_turns = game_instance.get("max_turns", 50)
        self.use_masking: bool = game_instance.get("use_masking", False)

        for model in self.player_models:
            model.set_gen_arg("max_tokens", 4096)

        # Build board
        walls = {tuple(w) for w in game_instance.get("walls", [])}
        self.board = Board(self.grid_size, walls)

        for obj, (r, c) in game_instance["start_positions"].items():
            self.board.place_object(obj, r, c)
        for obj, (r, c) in game_instance["target_positions"].items():
            self.board.set_target(obj, r, c)

        self.player_a_objects: Set[str] = set(game_instance["player_a_objects"])
        self.player_b_objects: Set[str] = set(game_instance["player_b_objects"])

        self.success = False
        self.aborted = False
        self.reprompt_attempts = 0
        self.current_parsed: Optional[Dict[str, str]] = None
        self.seen_states: Dict[str, int] = {self.board.state_key(): 1}

        model_a = self.player_models[0]
        model_b = self.player_models[1] if len(self.player_models) > 1 else self.player_models[0]

        self.player_a = self._make_player(model_a, "Player A", self.player_a_objects)
        self.player_b = self._make_player(model_b, "Player B", self.player_b_objects)

        prompt_template = game_instance["player_prompt"]
        objs_a = ", ".join(sorted(self.player_a_objects))
        objs_b = ", ".join(sorted(self.player_b_objects))

        ctx_a = self._build_context(prompt_template, objs_a, self.player_a)
        ctx_b = self._build_context(prompt_template, objs_b, self.player_b)

        self.add_player(self.player_a, initial_context=ctx_a)
        self.add_player(self.player_b, initial_context=ctx_b)

    def _make_player(self, model: Model, role_name: str, own_objects: Set[str]) -> Player:
        if isinstance(model, CustomResponseModel):
            return AStarPlayer(role_name, self.board, own_objects)
        return MatrixPlayer(model, role_name, own_objects)

    def _build_context(self, prompt_template: str, own_objects_str: str, player: Player) -> str:
        prompt = prompt_template.replace("$YOUR_OBJECTS$", own_objects_str)
        prompt = prompt.replace("$GRID_SIZE$", str(self.grid_size))
        board_view = self._render_for(player)
        goal_view = self.board.render_targets()
        return (
            f"{prompt}\n\n"
            f"CURRENT BOARD:\n{board_view}\n\n"
            f"GOAL BOARD (where each object must end up):\n{goal_view}"
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _objects_for(self, player: Player) -> Set[str]:
        return self.player_a_objects if player == self.player_a else self.player_b_objects

    def _render_for(self, player: Player) -> str:
        if self.use_masking:
            return self.board.render_for_player(self._objects_for(player))
        return self.board.render()

    # ── game loop ──────────────────────────────────────────────────────

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
            self.log_to_self("invalid format", f"{player.role_name}: could not parse response")
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid responses")
            else:
                self.set_context_for(player, (
                    "Your response could not be parsed. Please respond exactly in this format:\n"
                    "reason: <your reasoning>\n"
                    "move: <OBJECT> <DIRECTION>\n\n"
                    f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                    f"GOAL BOARD:\n{self.board.render_targets()}"
                ))
            return False

        obj = parsed["object"]
        direction = parsed["direction"]
        allowed = self._objects_for(player)
        error = self.board.validate_move(obj, direction, allowed_objects=allowed)
        if error:
            self.violated_request_counts += 1
            self.log_to_self("invalid move", f"{player.role_name}: {error}")
            if self.strict:
                self.aborted = True
                self.log_to_self("abort", f"Strict: illegal move by {player.role_name}")
                return False
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid moves")
            else:
                self.set_context_for(player, (
                    f"Invalid move: {error}\nPlease try again.\n\n"
                    "reason: <your reasoning>\n"
                    "move: <OBJECT> <DIRECTION>\n\n"
                    f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                    f"GOAL BOARD:\n{self.board.render_targets()}"
                ))
            return False

        self.parsed_request_counts += 1
        self.current_parsed = parsed
        self.reprompt_attempts = 0
        return True

    def _parse_response(self, player: Player, response: str) -> str:
        if self.current_parsed:
            return f"{self.current_parsed['object']} {self.current_parsed['direction']}"
        return response

    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        obj = self.current_parsed["object"]
        direction = self.current_parsed["direction"]
        reason = self.current_parsed["reason"]

        self.board.apply_move(obj, direction)
        self.move_log.append({
            "player": player.role_name,
            "object": obj,
            "direction": direction,
            "reason": reason,
        })
        self.log_to_self("move", f"{player.role_name} moved {obj} {direction}")
        self.log_to_self("reason", reason)

        # Cycle detection
        state = self.board.state_key()
        self.seen_states[state] = self.seen_states.get(state, 0) + 1
        if self.seen_states[state] >= 3:
            self.aborted = True
            self.log_to_self("abort", "Board state repeated 3 times — players stuck in cycle")
            return

        if self.board.is_solved():
            self.success = True
            self.log_to_self("success", "All objects have reached their target positions!")
            return

        other = self.player_b if player == self.player_a else self.player_a
        other_objects = self._objects_for(other)
        if self.use_masking and obj not in other_objects:
            move_desc = f"{player.role_name} moved a blocked object {direction}."
        else:
            move_desc = f"{player.role_name} moved {obj} {direction}."

        self.set_context_for(other, (
            f"{move_desc}\n\n"
            f"CURRENT BOARD:\n{self._render_for(other)}\n\n"
            f"GOAL BOARD:\n{self.board.render_targets()}\n\n"
            "Your turn. Respond with:\n"
            "reason: <your reasoning>\n"
            "move: <OBJECT> <DIRECTION>"
        ))

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_SUCCESS, int(self.success))
        self.log_key(METRIC_LOSE, int(not self.success and not self.aborted))
        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        self.log_key(MOVE_COUNT, len(self.move_log))
        self.log_key(TURN_MOVES, self.move_log)


# ── Scorer ─────────────────────────────────────────────────────────────

class MatrixGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        aborted = episode_interactions.get(METRIC_ABORTED, 0)
        success = episode_interactions.get(METRIC_SUCCESS, 0)
        move_count = episode_interactions.get(MOVE_COUNT, 0)
        optimal = self.game_instance.get("optimal_moves", move_count)

        self.log_episode_score(METRIC_ABORTED, aborted)
        self.log_episode_score(METRIC_SUCCESS, success)
        self.log_episode_score(METRIC_LOSE, int(not success and not aborted))
        self.log_episode_score(MOVE_COUNT, move_count)

        if aborted:
            self.log_episode_score(BENCH_SCORE, np.nan)
        elif success:
            # 100 if optimal, scales down proportionally
            score = min(100, round(optimal / max(move_count, 1) * 100, 2))
            self.log_episode_score(BENCH_SCORE, score)
        else:
            self.log_episode_score(BENCH_SCORE, 0)


# ── Benchmark entry point ──────────────────────────────────────────────

class MatrixGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> MatrixGameMaster:
        return MatrixGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> MatrixGameScorer:
        return MatrixGameScorer(self.game_name, experiment, game_instance)
