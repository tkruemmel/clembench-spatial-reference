"""Matrix Game (covered comm): two-LLM collaborative grid rearrangement with communication.

Three communication protocol variants (plus a no-communication control):
  none       — no inter-agent messages; move-only response format.
  structured — fixed template: PLAN: move X from (Rr,Cc) to (Rr,Cc) — REQUEST: ...
  freeform   — unrestricted natural language; the prompt asks for under 100 words (not enforced).
  hybrid     — free-form messaging plus an explicit shared coordinate frame in the prompt.

Turn structure for none:
  move: OBJECT to Rr,Cc (DIRECTION)   (or: done: <explanation> if all own objects are at targets)

Turn structure for structured / freeform / hybrid:
  message: ...
  move: OBJECT to Rr,Cc (DIRECTION)   (or: done: <explanation> if all own objects are at targets)

The game master relays each player's message to the other player before they respond.
A player who declares "done" (with all their objects at targets) stops receiving turns.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, Player, GameBenchmark
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import (
    METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_PLAYED,
    METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_VIOLATED,
    METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE,
)

from utils.board import Board, DIRECTIONS, BLOCKED_MARKER
from utils.metrics import oscillation_stats, deadlock_flag, path_ratios, message_lengths

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

COMM_NONE = "none"
COMM_STRUCTURED = "structured"
COMM_FREEFORM = "freeform"
COMM_HYBRID = "hybrid"

TURN_MOVES = "Turn Moves"
MOVE_COUNT = "Move Count"
MOVE_ATTEMPTS = "Move Attempts"
ILLEGAL_ATTEMPTS = "Illegal Attempts"
BLOCKED_ATTEMPTS = "Blocked Attempts"

# ── Response patterns ──────────────────────────────────────────────────

# Matches a move line whose object is the masked marker (the partner's object),
# in either the bare form `move: X ...` or the position-qualified form models
# tend to emit, `move: X@R3C4 ...`. Used to give a targeted semantic redirect
# instead of a generic parse error.
MASKED_OBJECT_MOVE_PATTERN = re.compile(
    rf"move:\s*{re.escape(BLOCKED_MARKER)}(?:@R\d+,?C\d+)?\b",
    re.IGNORECASE,
)

# Channel-off (comm_protocol == "none"): the response IS the move line.
MOVE_ONLY_PATTERN = re.compile(
    r"\A\s*move:\s*(?P<object>[A-Z])\s+to\s+R(?P<target_row>\d+),?C(?P<target_col>\d+)\s*\((?P<direction>up|down|left|right)\)",
    re.IGNORECASE,
)

# Channel on (structured / freeform / hybrid): message precedes the move.
# NOTE: no re.DOTALL — the message capture must not span the newline before
# "move:", otherwise a stray extra line inserted between message and move
# would be silently absorbed into the message instead of failing to parse.
COMM_MOVE_PATTERN = re.compile(
    r"\A\s*message:\s*(?P<message>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+to\s+R(?P<target_row>\d+),?C(?P<target_col>\d+)\s*\((?P<direction>up|down|left|right)\)",
    re.IGNORECASE,
)

# Done declarations — "done:" replaces the move line.
DONE_ONLY_PATTERN = re.compile(
    r"\A\s*done:\s*(?P<done_reason>.+)",
    re.IGNORECASE | re.DOTALL,
)
# NOTE: no re.DOTALL here either, for the same reason as COMM_MOVE_PATTERN
# above — the message capture must stop at the newline before "done:" so a
# stray extra line in between fails the anchored parse instead of being
# absorbed into the message. As a consequence done_reason only captures the
# first line after "done:"; that text is logged only, never parsed further,
# so this is fine.
COMM_DONE_PATTERN = re.compile(
    r"\A\s*message:\s*(?P<message>.+?)\s*\ndone:\s*(?P<done_reason>.+)",
    re.IGNORECASE,
)

# Structured-protocol message schema, enforced (thesis: the protocol
# *prescribes* the schema). Anchored: the message must BE the schema.
STRUCTURED_MSG_PATTERN = re.compile(
    r"\A\s*PLAN:\s*move\s+\S+\s+from\s*\(R\d+,C\d+\)\s+to\s*\(R\d+,C\d+\)\s*[—\-]+\s*REQUEST:\s*.+\Z",
    re.IGNORECASE | re.DOTALL,
)


def _parse_response(response: str, with_message: bool) -> Optional[Dict[str, str]]:
    # Try move pattern first
    move_pattern = COMM_MOVE_PATTERN if with_message else MOVE_ONLY_PATTERN
    match = move_pattern.match(response)
    if match:
        result = {
            "object": match.group("object").strip().upper(),
            "direction": match.group("direction").strip().lower(),
            "target": f"R{match.group('target_row')},C{match.group('target_col')}",
        }
        if with_message:
            result["message"] = match.group("message").strip()
        return result

    # Try done pattern
    done_pattern = COMM_DONE_PATTERN if with_message else DONE_ONLY_PATTERN
    match = done_pattern.match(response)
    if match:
        result: Dict[str, str] = {
            "is_done": True,
        }
        if with_message:
            result["message"] = match.group("message").strip()
        return result

    return None


def _declared_cell(parsed: Dict[str, str]) -> Optional[Tuple[int, int]]:
    """0-based (row, col) of the declared 'RrowCcol' target, or None if absent."""
    m = re.match(r"R(\d+),?C(\d+)", parsed.get("target", ""))
    if not m:
        return None
    return int(m.group(1)) - 1, int(m.group(2)) - 1


# ── Player ─────────────────────────────────────────────────────────────

class MatrixPlayer(Player):
    def __init__(self, model: Model, role_name: str, own_objects: Set[str]):
        super().__init__(model)
        self.role_name = role_name
        self.own_objects = own_objects

    def _custom_response(self, context):
        first = sorted(self.own_objects)[0]
        return f"move: {first} to R1,C1 (down)"


# ── Game master ────────────────────────────────────────────────────────

class MatrixGameMaster(DialogueGameMaster):

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        self.request_counts = 0
        self.move_attempts = 0
        self.illegal_attempts = 0
        self.blocked_attempts = 0
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
        self.masking_hints: bool = game_instance.get("masking_hints", True)
        self.comm_protocol: str = game_instance.get("comm_protocol", COMM_NONE)
        self.with_message: bool = self.comm_protocol != COMM_NONE
        self.compact_board: bool = game_instance.get("compact_board", False)

        for model in self.player_models:
            model.set_gen_arg("max_tokens", 16384)

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
        self.lost = False
        self.reprompt_attempts = 0
        self.reprompt_pending: bool = False
        self.current_parsed: Optional[Dict[str, str]] = None
        self.seen_states: Dict[str, int] = {self.board.state_key(): 1}
        self.player_a_done: bool = False
        self.player_b_done: bool = False

        model_a = self.player_models[0]
        model_b = self.player_models[1] if len(self.player_models) > 1 else self.player_models[0]

        self.player_a = MatrixPlayer(model_a, "Player A", self.player_a_objects)
        self.player_b = MatrixPlayer(model_b, "Player B", self.player_b_objects)

        prompt_template = game_instance["player_prompt"]
        objs_a = ", ".join(sorted(self.player_a_objects))
        objs_b = ", ".join(sorted(self.player_b_objects))

        ctx_a = self._build_initial_context(prompt_template, objs_a, self.player_a)
        ctx_b = self._build_initial_context(prompt_template, objs_b, self.player_b)

        self._prompt_template = prompt_template
        self._player_b_received_intro = False

        self.add_player(self.player_a, initial_context=ctx_a)
        self.add_player(self.player_b, initial_context=ctx_b)

    def _build_initial_context(self, template: str, own_objects_str: str, player: Player) -> str:
        prompt = template.replace("$YOUR_OBJECTS$", own_objects_str)
        prompt = prompt.replace("$GRID_SIZE$", str(self.grid_size))
        if "DONE OPTION" not in prompt:
            prompt += self._done_instructions_text()
        board_view = self._render_for(player)
        goal_view = self._render_targets_for(player)
        return (
            f"{prompt}\n\n"
            f"CURRENT BOARD:\n{board_view}\n\n"
            f"GOAL BOARD (where YOUR objects must end up):\n{goal_view}"
        )

    def _done_instructions_text(self) -> str:
        if self.with_message:
            return (
                "\n\nDONE OPTION:\n"
                "Once ALL your objects are at their target positions, you may declare done "
                "instead of making a move:\n"
                "message: done\n"
                "done: all my objects are placed\n"
                "IMPORTANT: Only declare done if ALL your objects are truly at their targets. "
                "An incorrect done declaration ends the game as a failure."
            )
        return (
            "\n\nDONE OPTION:\n"
            "Once ALL your objects are at their target positions, you may declare done "
            "instead of making a move:\n"
            "done: all my objects are placed\n"
            "IMPORTANT: Only declare done if ALL your objects are truly at their targets. "
            "An incorrect done declaration ends the game as a failure."
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _objects_for(self, player: Player) -> Set[str]:
        return self.player_a_objects if player == self.player_a else self.player_b_objects

    def _render_for(self, player: Player) -> str:
        visible = self._objects_for(player) if self.use_masking else None
        if self.compact_board:
            return self.board.render_compact(visible_objects=visible)
        if visible is not None:
            return self.board.render_for_player(visible)
        return self.board.render()

    def _render_targets_for(self, player: Player) -> str:
        visible = self._objects_for(player) if self.use_masking else None
        if self.compact_board:
            return self.board.render_targets_compact(visible_objects=visible)
        if visible is not None:
            return self.board.render_targets_for_player(visible)
        return self.board.render_targets()

    def _masked_move_message(self, player: Player) -> str:
        """Targeted redirect for a player that tried to move a masked ('X')
        object — i.e. its partner's object, which it cannot control."""
        own = ", ".join(sorted(self._objects_for(player)))
        return (
            f"'{BLOCKED_MARKER}' marks your partner's objects — you cannot move them. "
            f"You can only move YOUR objects: {own}. Choose one of yours."
        )

    def _is_done_player(self, player: Player) -> bool:
        return (player == self.player_a and self.player_a_done) or \
               (player == self.player_b and self.player_b_done)

    def _own_objects_at_target(self, player: Player) -> bool:
        return all(
            self.board.object_positions.get(obj) == self.board.target_positions.get(obj)
            for obj in self._objects_for(player)
        )

    def _format_reminder(self) -> str:
        if self.with_message:
            return (
                "message: <your message to your partner>\n"
                "move: <OBJECT> to R<row>,C<col> (<DIRECTION>)\n\n"
                "OR, if ALL your objects are at their target positions:\n"
                "message: done\n"
                "done: all my objects are placed"
            )
        return (
            "move: <OBJECT> to R<row>,C<col> (<DIRECTION>)\n\n"
            "OR, if ALL your objects are at their target positions:\n"
            "done: all my objects are placed"
        )

    # ── game loop ──────────────────────────────────────────────────────

    def _does_game_proceed(self) -> bool:
        if self.success or self.aborted or self.lost:
            return False
        if len(self.move_log) >= self.max_turns:
            self.lost = True
            self.log_to_self("lose", f"Max turns ({self.max_turns}) reached without success")
            return False
        return True

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        parsed = _parse_response(response, self.with_message)

        if parsed is None:
            self.violated_request_counts += 1
            # A common failure: the player tries to move a masked partner object
            # (e.g. `move: X@R3C4 ...`), which never matches the move grammar.
            # Redirect it semantically instead of with a generic format error.
            masked_attempt = bool(self.use_masking and MASKED_OBJECT_MOVE_PATTERN.search(response))
            if masked_attempt:
                self.log_to_self("invalid_move", f"{player.role_name}: attempted to move a masked partner object")
            else:
                self.log_to_self("invalid_format", f"{player.role_name}: could not parse response")
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid responses")
            else:
                self.reprompt_pending = True
                guidance = (
                    self._masked_move_message(player)
                    if (masked_attempt and self.masking_hints)
                    else "Your response could not be parsed. Please respond exactly in this format:"
                )
                self.set_context_for(player, (
                    f"{guidance}\n"
                    f"{self._format_reminder()}\n\n"
                    f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                    f"GOAL BOARD:\n{self._render_targets_for(player)}"
                ))
            return False

        if parsed.get("is_done"):
            if not self._own_objects_at_target(player):
                self.violated_request_counts += 1
                self.log_to_self(
                    "invalid_done",
                    f"{player.role_name}: declared done but not all their objects are at target positions",
                )
                self.aborted = True
                self.log_to_self(
                    "abort",
                    f"{player.role_name} declared done before all their objects reached their targets",
                )
                return False
            self.parsed_request_counts += 1
            self.current_parsed = parsed
            self.reprompt_attempts = 0
            return True

        # Structured protocol: the message MUST match the PLAN/REQUEST schema.
        # A malformed message is a violation like any parse failure: reprompt,
        # consume the retry budget, abort on exhaustion. Done declarations are
        # exempt (their message is the "done" marker).
        if self.comm_protocol == COMM_STRUCTURED:
            msg = parsed.get("message", "")
            if not STRUCTURED_MSG_PATTERN.match(msg):
                self.violated_request_counts += 1
                self.log_to_self(
                    "malformed_message",
                    f"{player.role_name}: structured format not followed — got: {msg[:80]}",
                )
                self.reprompt_attempts += 1
                if self.reprompt_attempts > self.max_retries:
                    self.aborted = True
                    self.log_to_self("abort", "Too many malformed structured messages")
                else:
                    self.reprompt_pending = True
                    self.set_context_for(player, (
                        "Your message did not follow the required structured format. "
                        "Use exactly:\n"
                        'message: PLAN: move [OBJECT] from (Rr,Cc) to (Rr,Cc) — REQUEST: [request or "none"]\n\n'
                        f"{self._format_reminder()}\n\n"
                        f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                        f"GOAL BOARD:\n{self._render_targets_for(player)}"
                    ))
                return False

        self.move_attempts += 1

        obj = parsed["object"]
        direction = parsed["direction"]
        allowed = self._objects_for(player)
        error = self.board.validate_move(
            obj, direction, allowed_objects=allowed, declared_target=_declared_cell(parsed)
        )

        if error:
            code, err_msg = error
            if code == "occupied":
                self.blocked_attempts += 1
            else:
                self.illegal_attempts += 1
            self.violated_request_counts += 1
            self.log_to_self("invalid_move", f"{player.role_name}: [{code}] {err_msg}")
            if self.strict:
                self.aborted = True
                self.log_to_self("abort", f"Strict: illegal move by {player.role_name}")
                return False
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid moves")
            else:
                self.reprompt_pending = True
                # `move: X ...` parses as object "X" but X is a masked partner
                # object; give the same semantic redirect as the unparsed form.
                masked_attempt = self.use_masking and obj == BLOCKED_MARKER
                guidance = (
                    self._masked_move_message(player)
                    if (masked_attempt and self.masking_hints)
                    else f"Invalid move: {err_msg}\nPlease try again."
                )
                self.set_context_for(player, (
                    f"{guidance}\n\n"
                    f"{self._format_reminder()}\n\n"
                    f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                    f"GOAL BOARD:\n{self._render_targets_for(player)}"
                ))
            return False

        self.parsed_request_counts += 1
        self.current_parsed = parsed
        self.reprompt_attempts = 0
        self.reprompt_pending = False
        return True

    def _parse_response(self, player: Player, response: str) -> str:
        if self.current_parsed:
            if self.current_parsed.get("is_done"):
                return "done"
            target = self.current_parsed.get("target", "?")
            return f"{self.current_parsed['object']} to {target} ({self.current_parsed['direction']})"
        return response

    def _should_pass_turn(self) -> bool:
        return not self.reprompt_pending

    def _next_player(self) -> Player:
        players = self.get_players()
        n = len(players)
        for _ in range(n):
            self._current_player_idx = (self._current_player_idx + 1) % n
            candidate = players[self._current_player_idx]
            if not self._is_done_player(candidate):
                return candidate
        # All players done — advance anyway (game will end on next _does_game_proceed check)
        self._current_player_idx = (self._current_player_idx + 1) % n
        return players[self._current_player_idx]

    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        message = self.current_parsed.get("message")

        if self.current_parsed.get("is_done"):
            if player == self.player_a:
                self.player_a_done = True
            else:
                self.player_b_done = True

            self.log_to_self("player_done", f"{player.role_name} declared done")
            if message:
                self.log_to_self("message", f"{player.role_name}: {message}")

            if self.board.is_solved():
                self.success = True
                self.log_to_self("success", "All objects have reached their target positions!")
                return

            other = self.player_b if player == self.player_a else self.player_a
            if message and self.with_message:
                comm_line = f"{player.role_name} says: {message}\n\n"
            else:
                comm_line = ""
            self.set_context_for(other, (
                f"{comm_line}"
                f"{player.role_name} has finished — all their objects are at their target positions. "
                f"Continue alone until all objects are placed.\n\n"
                f"CURRENT BOARD:\n{self._render_for(other)}\n\n"
                f"GOAL BOARD:\n{self._render_targets_for(other)}\n\n"
                f"Your turn. Respond with:\n{self._format_reminder()}"
            ))
            return

        obj = self.current_parsed["object"]
        direction = self.current_parsed["direction"]

        from_row, from_col = self.board.object_positions[obj]
        dr, dc = DIRECTIONS[direction]
        self.board.apply_move(obj, direction)
        self.move_log.append({
            "player": player.role_name,
            "object": obj,
            "direction": direction,
            "target": self.current_parsed.get("target", ""),
            "from": [from_row, from_col],
            "to": [from_row + dr, from_col + dc],
            "message": message,
        })
        self.log_to_self("move", f"{player.role_name} moved {obj} {direction}")
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
        other_done = self._is_done_player(other)

        # Move description: hide object identity if other player can't see it
        if self.use_masking and obj not in self._objects_for(other):
            move_desc = f"{player.role_name} moved a blocked object {direction}."
        else:
            move_desc = f"{player.role_name} moved {obj} {direction}."

        if message and self.with_message:
            comm_line = f"{player.role_name} says: {message}\n\n"
        else:
            comm_line = ""

        if other_done:
            # Partner finished; current player continues alone
            self.set_context_for(player, (
                f"Your partner has finished. Continue placing your remaining objects alone.\n\n"
                f"You moved {obj} {direction}.\n\n"
                f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                f"GOAL BOARD:\n{self._render_targets_for(player)}\n\n"
                f"Your turn. Respond with:\n{self._format_reminder()}"
            ))
        elif not self._player_b_received_intro and other == self.player_b:
            self._player_b_received_intro = True
            objs_b = ", ".join(sorted(self.player_b_objects))
            self.set_context_for(other, self._build_initial_context(self._prompt_template, objs_b, other))
        else:
            self.set_context_for(other, (
                f"{comm_line}"
                f"{move_desc}\n\n"
                f"CURRENT BOARD:\n{self._render_for(other)}\n\n"
                f"GOAL BOARD:\n{self._render_targets_for(other)}\n\n"
                f"Your turn. Respond with:\n{self._format_reminder()}"
            ))

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_SUCCESS, int(self.success))
        self.log_key(METRIC_LOSE, int(self.lost))
        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        self.log_key(MOVE_COUNT, len(self.move_log))
        self.log_key(MOVE_ATTEMPTS, self.move_attempts)
        self.log_key(ILLEGAL_ATTEMPTS, self.illegal_attempts)
        self.log_key(BLOCKED_ATTEMPTS, self.blocked_attempts)
        self.log_key(TURN_MOVES, self.move_log)
        self.log_key("Player A Done", int(self.player_a_done))
        self.log_key("Player B Done", int(self.player_b_done))


# ── Scorer ─────────────────────────────────────────────────────────────

class MatrixGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        aborted = episode_interactions.get(METRIC_ABORTED, 0)
        success = episode_interactions.get(METRIC_SUCCESS, 0)
        move_count = episode_interactions.get(MOVE_COUNT, 0)
        move_log = episode_interactions.get(TURN_MOVES, []) or []
        attempts = episode_interactions.get(MOVE_ATTEMPTS, 0)
        illegal = episode_interactions.get(ILLEGAL_ATTEMPTS, 0)
        blocked = episode_interactions.get(BLOCKED_ATTEMPTS, 0)
        optimal = self.game_instance.get("optimal_moves", move_count)
        starts = self.game_instance["start_positions"]
        targets = self.game_instance["target_positions"]

        self.log_episode_score(METRIC_ABORTED, aborted)
        self.log_episode_score(METRIC_SUCCESS, success)
        lost = episode_interactions.get(METRIC_LOSE, int(not success and not aborted))
        self.log_episode_score(METRIC_LOSE, lost)
        self.log_episode_score(METRIC_PLAYED, 1 - aborted)
        self.log_episode_score(MOVE_COUNT, move_count)

        if aborted:
            self.log_episode_score(BENCH_SCORE, np.nan)
        elif success:
            score = min(100, round(optimal / max(move_count, 1) * 100, 2))
            self.log_episode_score(BENCH_SCORE, score)
        else:
            self.log_episode_score(BENCH_SCORE, 0)

        # Spatial-efficiency family
        self.log_episode_score(
            "Move Efficiency",
            round(optimal / max(move_count, 1), 4) if success else np.nan,
        )
        self.log_episode_score(
            "Illegal Move Rate", round(illegal / attempts, 4) if attempts else 0
        )
        self.log_episode_score(
            "Blocked Move Rate", round(blocked / attempts, 4) if attempts else 0
        )
        ratios = path_ratios(move_log, starts, targets)
        if ratios:
            vals = list(ratios.values())
            self.log_episode_score("Path Ratio Mean", round(sum(vals) / len(vals), 4))
            self.log_episode_score("Path Ratio Max", round(max(vals), 4))

        # Failure-mode family
        osc_count, osc_flag = oscillation_stats(move_log, k=4)
        self.log_episode_score("Oscillation Count", osc_count)
        self.log_episode_score("Oscillation Flag", osc_flag)
        self.log_episode_score(
            "Deadlock Flag", deadlock_flag(move_log, starts, targets, k=6)
        )

        # Communication family (message: is the solo scratchpad / team channel)
        lengths = message_lengths(move_log)
        self.log_episode_score("Message Count", len(lengths))
        if lengths:
            self.log_episode_score("Message Length Mean", round(sum(lengths) / len(lengths), 2))
            self.log_episode_score("Message Length Max", max(lengths))


# ── Benchmark entry point ──────────────────────────────────────────────

class MatrixGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> MatrixGameMaster:
        return MatrixGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> MatrixGameScorer:
        return MatrixGameScorer(self.game_name, experiment, game_instance)
