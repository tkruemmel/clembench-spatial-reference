"""Matrix Game (covered comm): two-LLM collaborative grid rearrangement with communication.

Three communication protocol variants (plus a no-communication control):
  none       — no inter-agent messages; move-only response format.
  structured — fixed template: PLAN: move X from (Rr,Cc) to (Rr,Cc) — REQUEST: ...
  freeform   — unrestricted natural language, max 100 tokens per message.
  hybrid     — natural language with a shared coordinate system in the system prompt.

Turn structure for none:
  reason: ...
  move: OBJECT DIRECTION

Turn structure for structured / freeform / hybrid:
  message: ...
  reason: ...
  move: OBJECT DIRECTION

The game master relays each player's message to the other player before they respond.
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

COMM_NONE = "none"
COMM_STRUCTURED = "structured"
COMM_FREEFORM = "freeform"
COMM_HYBRID = "hybrid"

TURN_MOVES = "Turn Moves"
MOVE_COUNT = "Move Count"

# ── Response patterns ──────────────────────────────────────────────────

# Used when comm_protocol == "none"
MOVE_ONLY_PATTERN = re.compile(
    r"reason:\s*(?P<reason>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
    re.IGNORECASE | re.DOTALL,
)

# Used when comm_protocol is structured / freeform / hybrid
COMM_MOVE_PATTERN = re.compile(
    r"message:\s*(?P<message>.+?)\s*\nreason:\s*(?P<reason>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
    re.IGNORECASE | re.DOTALL,
)

# Lenient check for C1 structured message format (warn but don't abort on mismatch)
STRUCTURED_MSG_PATTERN = re.compile(
    r"PLAN:\s*move\s+\S+\s+from\s*\(R\d+,C\d+\)\s+to\s*\(R\d+,C\d+\)\s*[—\-]+\s*REQUEST:\s*.+",
    re.IGNORECASE,
)


def _parse_response(response: str, with_message: bool) -> Optional[Dict[str, str]]:
    pattern = COMM_MOVE_PATTERN if with_message else MOVE_ONLY_PATTERN
    match = pattern.search(response)
    if not match:
        return None
    result = {
        "reason": match.group("reason").strip(),
        "object": match.group("object").strip().upper(),
        "direction": match.group("direction").strip().lower(),
    }
    if with_message:
        result["message"] = match.group("message").strip()
    return result


# ── Player ─────────────────────────────────────────────────────────────

class MatrixPlayer(Player):
    def __init__(self, model: Model, role_name: str, own_objects: Set[str]):
        super().__init__(model)
        self.role_name = role_name
        self.own_objects = own_objects

    def _custom_response(self, context):
        first = sorted(self.own_objects)[0]
        return f"reason: custom player\nmove: {first} down"


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
        optimal = game_instance.get("optimal_moves", 1)
        self.max_turns = game_instance.get("max_turns", max(50, 5 * optimal))
        self.use_masking: bool = game_instance.get("use_masking", True)
        self.comm_protocol: str = game_instance.get("comm_protocol", COMM_NONE)
        self.with_message: bool = self.comm_protocol != COMM_NONE

        for model in self.player_models:
            model.set_gen_arg("max_tokens", 4096)

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

        self.player_a = MatrixPlayer(model_a, "Player A", self.player_a_objects)
        self.player_b = MatrixPlayer(model_b, "Player B", self.player_b_objects)

        prompt_template = game_instance["player_prompt"]
        objs_a = ", ".join(sorted(self.player_a_objects))
        objs_b = ", ".join(sorted(self.player_b_objects))

        ctx_a = self._build_initial_context(prompt_template, objs_a, self.player_a)
        ctx_b = self._build_initial_context(prompt_template, objs_b, self.player_b)

        self.add_player(self.player_a, initial_context=ctx_a)
        self.add_player(self.player_b, initial_context=ctx_b)

    def _build_initial_context(self, template: str, own_objects_str: str, player: Player) -> str:
        prompt = template.replace("$YOUR_OBJECTS$", own_objects_str)
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

    def _format_reminder(self) -> str:
        if self.with_message:
            return (
                "message: <your message to your partner>\n"
                "reason: <your reasoning>\n"
                "move: <OBJECT> <DIRECTION>"
            )
        return (
            "reason: <your reasoning>\n"
            "move: <OBJECT> <DIRECTION>"
        )

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
                self.set_context_for(player, (
                    "Your response could not be parsed. Please respond exactly in this format:\n"
                    f"{self._format_reminder()}\n\n"
                    f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                    f"GOAL BOARD:\n{self.board.render_targets()}"
                ))
            return False

        # C1: warn if structured message format not followed (but don't abort)
        if self.comm_protocol == COMM_STRUCTURED:
            msg = parsed.get("message", "")
            if not STRUCTURED_MSG_PATTERN.search(msg):
                self.log_to_self(
                    "malformed_message",
                    f"{player.role_name}: structured format not followed — got: {msg[:80]}"
                )

        obj = parsed["object"]
        direction = parsed["direction"]
        allowed = self._objects_for(player)
        error = self.board.validate_move(obj, direction, allowed_objects=allowed)

        if error:
            self.violated_request_counts += 1
            self.log_to_self("invalid_move", f"{player.role_name}: {error}")
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
                    f"{self._format_reminder()}\n\n"
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
        message = self.current_parsed.get("message")

        self.board.apply_move(obj, direction)
        self.move_log.append({
            "player": player.role_name,
            "object": obj,
            "direction": direction,
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
            self.log_to_self("abort", "Board state repeated 3 times — players stuck in cycle")
            return

        if self.board.is_solved():
            self.success = True
            self.log_to_self("success", "All objects have reached their target positions!")
            return

        other = self.player_b if player == self.player_a else self.player_a
        other_objects = self._objects_for(other)

        # Move description: hide object identity if other player can't see it
        if self.use_masking and obj not in other_objects:
            move_desc = f"{player.role_name} moved a blocked object {direction}."
        else:
            move_desc = f"{player.role_name} moved {obj} {direction}."

        # Prepend relayed message if communication is enabled
        if message and self.with_message:
            comm_line = f"{player.role_name} says: {message}\n\n"
        else:
            comm_line = ""

        self.set_context_for(other, (
            f"{comm_line}"
            f"{move_desc}\n\n"
            f"CURRENT BOARD:\n{self._render_for(other)}\n\n"
            f"GOAL BOARD:\n{self.board.render_targets()}\n\n"
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
