"""Matrix Game (covered solo): single-LLM grid rearrangement.

Full-view only — the model sees all 6 objects and owns all 6.

Optional 'thinking' axis adds a free-form `message:` line per turn that the
master logs but does not interpret.

Turn structure (silent):
  reason: ...
  move: OBJECT to Rr,Cc (DIRECTION)

Turn structure (thinking):
  message: ...
  reason: ...
  move: OBJECT to Rr,Cc (DIRECTION)

The episode ends automatically when all owned objects are at their target
positions (success), when max_turns is reached (lose), or when parse/validate
failures exhaust max_retries (abort).
"""

import re
import logging
from typing import Dict, List, Optional, Set

import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, Player, GameBenchmark
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import (
    METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE,
    METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_VIOLATED,
    METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE,
)

from utils.board import Board, DIRECTIONS

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

TURN_MOVES = "Turn Moves"
MOVE_COUNT = "Move Count"

# ── Response patterns ──────────────────────────────────────────────────

# Used when thinking is off.
# Anchored with \A to reject responses that prepend other lines (e.g. message:).
MOVE_ONLY_PATTERN = re.compile(
    r"\A\s*reason:\s*(?P<reason>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+to\s+R(?P<target_row>\d+),?C(?P<target_col>\d+)\s*\((?P<direction>up|down|left|right)\)",
    re.IGNORECASE | re.DOTALL,
)

# Used when thinking is on.
MESSAGE_MOVE_PATTERN = re.compile(
    r"\A\s*message:\s*(?P<message>.+?)\s*\nreason:\s*(?P<reason>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+to\s+R(?P<target_row>\d+),?C(?P<target_col>\d+)\s*\((?P<direction>up|down|left|right)\)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_response(response: str, with_message: bool) -> Optional[Dict[str, str]]:
    """Parse a model response into a move dict, or None if malformed.

    The two patterns are anchored with \\A so a stray 'message:' line in silent
    mode (and a missing 'message:' line in thinking mode) both fail to parse.
    """
    pattern = MESSAGE_MOVE_PATTERN if with_message else MOVE_ONLY_PATTERN
    match = pattern.match(response)
    if not match:
        return None
    result: Dict[str, str] = {
        "reason": match.group("reason").strip(),
        "object": match.group("object").strip().upper(),
        "direction": match.group("direction").strip().lower(),
        "target": f"R{match.group('target_row')},C{match.group('target_col')}",
    }
    if with_message:
        result["message"] = match.group("message").strip()
    return result


# ── Player ─────────────────────────────────────────────────────────────

class SoloMatrixPlayer(Player):
    def __init__(self, model: Model, role_name: str, own_objects: Set[str]):
        super().__init__(model)
        self.role_name = role_name
        self.own_objects = own_objects

    def _custom_response(self, context):
        # Used by clemcore's mock player; emit a deterministic valid-format move.
        first = sorted(self.own_objects)[0]
        return f"reason: custom player\nmove: {first} to R1,C1 (down)"


# ── Rendering helper ───────────────────────────────────────────────────

def _render_view(board: Board, compact: bool) -> str:
    """Render the current board state."""
    if compact:
        return board.render_compact()
    return board.render()


def _render_targets_view(board: Board, compact: bool) -> str:
    """Render the goal board (where each object must end up)."""
    if compact:
        return board.render_targets_compact()
    return board.render_targets()


# ── Validation ─────────────────────────────────────────────────────────

def _validate_move(board: Board, player: SoloMatrixPlayer, parsed: Dict[str, str]) -> Optional[str]:
    """Return an error string if the move is invalid, else None.

    Wraps Board.validate_move with the player's allowed-objects set.
    """
    return board.validate_move(
        parsed["object"],
        parsed["direction"],
        allowed_objects=player.own_objects,
    )


# ── Game master ────────────────────────────────────────────────────────

class SoloMatrixGameMaster(DialogueGameMaster):

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
        optimal = game_instance.get("optimal_moves", 1)
        self.max_turns = game_instance.get("max_turns", max(4 * optimal, 20))
        self.thinking: bool = game_instance["thinking"]
        self.with_message: bool = self.thinking
        self.compact_board: bool = game_instance.get("compact_board", False)

        for model in self.player_models:
            model.set_gen_arg("max_tokens", 16384)

        walls = {tuple(w) for w in game_instance.get("walls", [])}
        self.board = Board(self.grid_size, walls)

        for obj, (r, c) in game_instance["start_positions"].items():
            self.board.place_object(obj, r, c)
        for obj, (r, c) in game_instance["target_positions"].items():
            self.board.set_target(obj, r, c)

        self.objects: Set[str] = set(game_instance["objects"])

        self.success = False
        self.aborted = False
        self.reprompt_attempts = 0
        self.reprompt_pending: bool = False
        self.current_parsed: Optional[Dict[str, str]] = None
        self.seen_states: Dict[str, int] = {self.board.state_key(): 1}

        model = self.player_models[0]
        self.player = SoloMatrixPlayer(model, "Player", self.objects)

        prompt_template = game_instance["player_prompt"]
        objs_str = ", ".join(sorted(self.objects))
        ctx = self._build_initial_context(prompt_template, objs_str)
        self.add_player(self.player, initial_context=ctx)

    def _build_initial_context(self, template: str, own_objects_str: str) -> str:
        prompt = template.replace("$YOUR_OBJECTS$", own_objects_str)
        prompt = prompt.replace("$GRID_SIZE$", str(self.grid_size))
        board_view = _render_view(self.board, self.compact_board)
        goal_view = _render_targets_view(self.board, self.compact_board)
        return (
            f"{prompt}\n\n"
            f"CURRENT BOARD:\n{board_view}\n\n"
            f"GOAL BOARD (where each object must end up):\n{goal_view}"
        )

    def _format_reminder(self) -> str:
        if self.with_message:
            return (
                "message: <a brief note to yourself>\n"
                "reason: <your reasoning>\n"
                "move: <OBJECT> to R<row>,C<col> (<DIRECTION>)"
            )
        return (
            "reason: <your reasoning>\n"
            "move: <OBJECT> to R<row>,C<col> (<DIRECTION>)"
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _all_objects_at_target(self) -> bool:
        return all(
            self.board.object_positions.get(obj) == self.board.target_positions.get(obj)
            for obj in self.objects
        )

    def _render_board(self) -> str:
        return _render_view(self.board, self.compact_board)

    def _render_goal(self) -> str:
        return _render_targets_view(self.board, self.compact_board)

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
        parsed = _parse_response(response, self.with_message)

        if parsed is None:
            self.violated_request_counts += 1
            self.log_to_self("invalid_format", f"{player.role_name}: could not parse response")
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid responses")
            else:
                self.reprompt_pending = True
                self.set_context_for(player, (
                    "Your response could not be parsed. Please respond exactly in this format:\n"
                    f"{self._format_reminder()}\n\n"
                    f"CURRENT BOARD:\n{self._render_board()}\n\n"
                    f"GOAL BOARD:\n{self._render_goal()}"
                ))
            return False

        # Validate the move
        error = _validate_move(self.board, self.player, parsed)
        if error:
            self.violated_request_counts += 1
            self.log_to_self("invalid_move", f"{player.role_name}: {error}")
            if self.strict:
                self.aborted = True
                self.log_to_self("abort", "Strict: illegal move")
                return False
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid moves")
            else:
                self.reprompt_pending = True
                self.set_context_for(player, (
                    f"Invalid move: {error}\nPlease try again.\n\n"
                    f"{self._format_reminder()}\n\n"
                    f"CURRENT BOARD:\n{self._render_board()}\n\n"
                    f"GOAL BOARD:\n{self._render_goal()}"
                ))
            return False

        self.parsed_request_counts += 1
        self.current_parsed = parsed
        self.reprompt_attempts = 0
        self.reprompt_pending = False
        return True

    def _parse_response(self, player: Player, response: str) -> str:
        if self.current_parsed:
            target = self.current_parsed.get("target", "?")
            return f"{self.current_parsed['object']} to {target} ({self.current_parsed['direction']})"
        return response

    def _should_pass_turn(self) -> bool:
        return not self.reprompt_pending

    def _next_player(self) -> Player:
        return self.player

    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        reason = self.current_parsed["reason"]
        message = self.current_parsed.get("message")
        obj = self.current_parsed["object"]
        direction = self.current_parsed["direction"]

        self.board.apply_move(obj, direction)
        self.move_log.append({
            "object": obj,
            "direction": direction,
            "target": self.current_parsed.get("target", ""),
            "reason": reason,
            "message": message,
        })
        self.log_to_self("move", f"{player.role_name} moved {obj} {direction}")
        self.log_to_self("reason", reason)
        if message:
            self.log_to_self("message", f"{player.role_name}: {message}")

        state = self.board.state_key()
        self.seen_states[state] = self.seen_states.get(state, 0) + 1
        if self.seen_states[state] >= 3:
            self.aborted = True
            self.log_to_self("abort", "Board state repeated 3 times — model is in a cycle")
            return

        if self._all_objects_at_target():
            self.success = True
            self.log_to_self("success", "All objects have reached their targets")
            return

        # Build next-turn context
        self.set_context_for(player, (
            f"You moved {obj} {direction}.\n\n"
            f"CURRENT BOARD:\n{self._render_board()}\n\n"
            f"GOAL BOARD:\n{self._render_goal()}\n\n"
            f"Your turn. Respond with:\n{self._format_reminder()}"
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

class SoloMatrixGameScorer(GameScorer):
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
            score = min(100, round(optimal / max(move_count, 1) * 100, 2))
            self.log_episode_score(BENCH_SCORE, score)
        else:
            self.log_episode_score(BENCH_SCORE, 0)


# ── Benchmark entry point ──────────────────────────────────────────────

class SoloMatrixGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> SoloMatrixGameMaster:
        return SoloMatrixGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> SoloMatrixGameScorer:
        return SoloMatrixGameScorer(self.game_name, experiment, game_instance)
