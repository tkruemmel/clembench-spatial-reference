"""Matrix Game (covered comm): two-LLM collaborative grid rearrangement with communication.

Three communication protocol variants (plus a no-communication control):
  none       — no inter-agent messages; move-only response format.
  structured — fixed template: PLAN: move X from (Rr,Cc) to (Rr,Cc) — REQUEST: ...; PLAN and REQUEST each <=10 words, enforced (reprompt/abort).
  freeform   — unrestricted natural language; the prompt asks for at most 100 words (not enforced).
  hybrid     — free-form prose (<=100 words) plus a required PLAN/REQUEST clause (each part <=10 words); message may span multiple lines; schema+caps enforced (reprompt/abort) like structured.

Turn structure for none:
  move: OBJECT DIRECTION   (or: done: <explanation> if all own objects are at targets)

Turn structure for structured / freeform / hybrid:
  message: ...
  move: OBJECT DIRECTION   (or: done: <explanation> if all own objects are at targets)

The game master relays each player's message to the other player before they respond.
A player who declares "done" (with all their objects at targets) stops receiving turns.
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

# Matches a move line whose object is the masked marker (the partner's object).
# Used to give a targeted semantic redirect instead of a generic parse error.
MASKED_OBJECT_MOVE_PATTERN = re.compile(
    rf"move:\s*{re.escape(BLOCKED_MARKER)}\b",
    re.IGNORECASE,
)

# Channel-off (comm_protocol == "none"): the response IS the move line.
MOVE_ONLY_PATTERN = re.compile(
    r"\A\s*move:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)\b",
    re.IGNORECASE,
)

# Channel on (structured / freeform / hybrid): message precedes the move.
# re.DOTALL: the message MAY span multiple lines (e.g. free-form prose on its
# own line(s) and the PLAN clause on another). The lazy `.+?` still stops at the
# first `\nmove:`, which remains the required terminator, so the move line is
# always parsed. Tradeoff: a stray line between message and move is absorbed
# into the message rather than failing the parse — accepted to allow multi-line
# messages.
COMM_MOVE_PATTERN = re.compile(
    r"\A\s*message:\s*(?P<message>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)\b",
    re.IGNORECASE | re.DOTALL,
)

# Done declarations — "done:" replaces the move line.
DONE_ONLY_PATTERN = re.compile(
    r"\A\s*done:\s*(?P<done_reason>.+)",
    re.IGNORECASE | re.DOTALL,
)
# re.DOTALL: like COMM_MOVE_PATTERN, the message may span multiple lines; the
# lazy `.+?` stops at the first `\ndone:`. done_reason is logged only, never
# parsed further.
COMM_DONE_PATTERN = re.compile(
    r"\A\s*message:\s*(?P<message>.+?)\s*\ndone:\s*(?P<done_reason>.+)",
    re.IGNORECASE | re.DOTALL,
)

# Word caps (inclusive) enforced on the schema message parts.
PROSE_MAX_WORDS = 100    # hybrid free-form prose part
CLAUSE_MAX_WORDS = 10    # PLAN content and REQUEST content, each

# Structured-protocol message schema, enforced (thesis: the protocol
# *prescribes* the schema). Anchored full-match; named groups feed the word
# caps. The PLAN clause may sit on its own line (re.DOTALL).
STRUCTURED_MSG_PATTERN = re.compile(
    r"\A\s*PLAN:\s*(?P<plan>move\s+\S+\s+from\s*\(R\d+,C\d+\)\s+to\s*\(R\d+,C\d+\))\s*[—\-]+\s*REQUEST:\s*(?P<request>.+?)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

# Hybrid-protocol message schema, enforced (thesis: hybrid *combines* freeform
# and structured). Free natural-language prose, then the same PLAN/REQUEST
# clause the structured protocol prescribes. Anchored full-match with named
# groups (prose / plan / request); prose needs >=1 non-space char, so a
# clause-only message (no prose) is rejected — that would just be structured.
HYBRID_MSG_PATTERN = re.compile(
    r"\A\s*(?P<prose>\S.*?)\s*PLAN:\s*(?P<plan>move\s+\S+\s+from\s*\(R\d+,C\d+\)\s+to\s*\(R\d+,C\d+\))\s*[—\-]+\s*REQUEST:\s*(?P<request>.+?)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

# Reminders shown when the schema structure is wrong (word-cap reminders are
# built inline in _schema_violation).
_STRUCTURED_REMINDER = (
    'Use exactly:\n'
    'message: PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc) — REQUEST: <request or "none">\n'
    "The first (Rr,Cc) is the object's current cell. The second is its goal cell."
)
_HYBRID_REMINDER = (
    'Use exactly:\n'
    'message: <natural language> PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc) — REQUEST: <request or "none">\n'
    "The first (Rr,Cc) is the object's current cell. The second is its goal cell."
)


def _clause_word_violation(match) -> Optional[str]:
    """Word-cap check shared by structured and hybrid: PLAN and REQUEST content
    must each be at most CLAUSE_MAX_WORDS words. Returns a reminder or None."""
    if len(match.group("plan").split()) > CLAUSE_MAX_WORDS:
        return f"Keep the PLAN clause to at most {CLAUSE_MAX_WORDS} words."
    if len(match.group("request").split()) > CLAUSE_MAX_WORDS:
        return f'Keep the REQUEST to at most {CLAUSE_MAX_WORDS} words (or "none").'
    return None


def _schema_violation(protocol: str, message: str) -> Optional[str]:
    """Return a specific reprompt reminder if `message` violates the protocol's
    schema (structure or word caps), else None. The reminder names the exact
    problem — missing PLAN, missing REQUEST, wrong PLAN form, missing prose
    (hybrid), or a word-cap overrun — rather than only restating the template.
    Only structured/hybrid have schemas."""
    if protocol not in (COMM_STRUCTURED, COMM_HYBRID):
        return None
    pattern = STRUCTURED_MSG_PATTERN if protocol == COMM_STRUCTURED else HYBRID_MSG_PATTERN
    reminder = _STRUCTURED_REMINDER if protocol == COMM_STRUCTURED else _HYBRID_REMINDER
    m = pattern.match(message)
    if m:
        # Structure is correct; only the word caps can still be violated.
        if protocol == COMM_HYBRID and len(m.group("prose").split()) > PROSE_MAX_WORDS:
            return (f"Your PLAN/REQUEST clause is fine, but the natural-language part is too "
                    f"long. Keep it to at most {PROSE_MAX_WORDS} words.")
        return _clause_word_violation(m)
    # Structure did not match — name the specific missing/wrong part.
    has_plan = re.search(r"PLAN\s*:", message, re.IGNORECASE) is not None
    has_request = re.search(r"REQUEST\s*:", message, re.IGNORECASE) is not None
    if not has_plan and not has_request:
        prefix = ("Your message is missing the required PLAN/REQUEST clause. "
                  "Keep your natural-language prose, then append the clause. "
                  if protocol == COMM_HYBRID
                  else "Your message is missing the required PLAN/REQUEST clause. ")
        return prefix + reminder
    if has_plan and not has_request:
        return ('Your message has a PLAN but no REQUEST. Append '
                '— REQUEST: <request or "none"> after the PLAN clause. ' + reminder)
    if has_request and not has_plan:
        return ("Your message has a REQUEST but no PLAN. Add "
                "PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc) before it. " + reminder)
    if protocol == COMM_HYBRID and not re.split(r"PLAN\s*:", message, flags=re.IGNORECASE)[0].strip():
        return ("Hybrid messages need natural-language prose BEFORE the PLAN clause. Add a "
                "sentence, then the PLAN/REQUEST clause. " + reminder)
    return ("Your PLAN clause is not in the required form. It must read exactly "
            "PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc). " + reminder)


def _parse_response(response: str, with_message: bool) -> Optional[Dict[str, str]]:
    # Try move pattern first
    move_pattern = COMM_MOVE_PATTERN if with_message else MOVE_ONLY_PATTERN
    match = move_pattern.match(response)
    if match:
        result = {
            "object": match.group("object").strip().upper(),
            "direction": match.group("direction").strip().lower(),
        }
        if with_message:
            result["message"] = match.group("message").strip()
        return result

    # Try done pattern. Under a messaging protocol the canonical form is
    # "message: done" + "done: ...", but a bare "done: ..." line is accepted too:
    # a done declaration ends the player's participation and its message part
    # carries no information, so rejecting the bare form only produced format
    # errors for players who followed the DONE OPTION text literally.
    done_pattern = COMM_DONE_PATTERN if with_message else DONE_ONLY_PATTERN
    match = done_pattern.match(response)
    if match:
        result: Dict[str, str] = {
            "is_done": True,
        }
        if with_message:
            result["message"] = match.group("message").strip()
        return result
    if with_message:
        match = DONE_ONLY_PATTERN.match(response)
        if match:
            return {"is_done": True}

    return None


# ── Player ─────────────────────────────────────────────────────────────

class MatrixPlayer(Player):
    def __init__(self, model: Model, role_name: str, own_objects: Set[str]):
        super().__init__(model)
        self.role_name = role_name
        self.own_objects = own_objects
        # Pinned rules/instructions, set once at setup. When present, the player's
        # perspective is Markovian: the model sees the fixed preamble plus only the
        # current turn's context (appended by the caller), never the whole transcript.
        # The board fully re-describes the state each turn, so no history is needed,
        # and the per-turn prompt stays bounded regardless of episode length.
        self.preamble: Optional[str] = None

    def get_perspective(self):
        if self.preamble is not None:
            return [{"role": "user", "content": self.preamble}]
        return self._messages

    def _custom_response(self, context):
        first = sorted(self.own_objects)[0]
        return f"move: {first} down"


# ── Game master ────────────────────────────────────────────────────────

class MatrixGameMaster(DialogueGameMaster):

    # Seat names; the final arm overrides them.
    ROLE_NAMES = ("Player A", "Player B")

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
        self.max_turns = game_instance.get("max_turns", max(4 * optimal, 20))
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
        # Each player's most recent message, relayed back to everyone (including
        # the sender) in the Markovian per-turn context.
        self.last_message: Dict[str, Optional[str]] = {r: None for r in self.ROLE_NAMES}

        model_a = self.player_models[0]
        model_b = self.player_models[1] if len(self.player_models) > 1 else self.player_models[0]

        self.player_a = MatrixPlayer(model_a, self.ROLE_NAMES[0], self.player_a_objects)
        self.player_b = MatrixPlayer(model_b, self.ROLE_NAMES[1], self.player_b_objects)

        prompt_template = game_instance["player_prompt"]
        objs_a = ", ".join(sorted(self.player_a_objects))
        objs_b = ", ".join(sorted(self.player_b_objects))

        preamble_a, first_a = self._build_initial_context(prompt_template, objs_a, self.player_a)
        preamble_b, first_b = self._build_initial_context(prompt_template, objs_b, self.player_b)
        self.player_a.preamble = preamble_a
        self.player_b.preamble = preamble_b

        self._prompt_template = prompt_template

        self.add_player(self.player_a, initial_context=first_a)
        self.add_player(self.player_b, initial_context=first_b)

    def _build_initial_context(self, template: str, own_objects_str: str, player: Player) -> tuple:
        prompt = template.replace("$YOUR_OBJECTS$", own_objects_str)
        prompt = prompt.replace("$GRID_SIZE$", str(self.grid_size))
        if "DONE OPTION" not in prompt:
            prompt += self._done_instructions_text()
        board_view = self._render_for(player)
        goal_view = self._render_targets_for(player)
        # Return the fixed rules preamble and the first turn's context separately.
        # The preamble is pinned on the player (Markovian perspective); the context
        # mirrors the per-turn contexts built by _turn_context.
        first_ctx = (
            f"CURRENT BOARD:\n{board_view}\n\n"
            f"GOAL BOARD (where YOUR objects must end up):\n{goal_view}"
        )
        return prompt, first_ctx

    def _turn_context(self, recipient: Player, notice: str = "") -> str:
        """Markovian per-turn context: every player's last message (including the
        recipient's own), an optional master notice, current and goal boards, and
        the response-format reminder. No accumulated history."""
        message_lines: List[str] = []
        for p in (self.player_a, self.player_b):
            msg = self.last_message.get(p.role_name)
            if msg:
                label = f"{p.role_name} (you)" if p == recipient else p.role_name
                message_lines.append(f"{label} says: {msg}")
        comm_block = ("\n".join(message_lines) + "\n\n") if message_lines else ""
        notice_block = (f"{notice}\n\n") if notice else ""
        return (
            f"{comm_block}"
            f"{notice_block}"
            f"{self._state_block(recipient)}\n\n"
            f"Your turn. Respond with:\n{self._format_reminder()}"
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

    def _state_block(self, player: Player) -> str:
        """Current and goal boards: the tail of every turn and retry context."""
        return (
            f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
            f"GOAL BOARD:\n{self._render_targets_for(player)}"
        )

    def _masked_move_message(self, player: Player) -> str:
        """Targeted redirect for a player that tried to move a masked ('X')
        object — i.e. its partner's object, which it cannot control."""
        own = ", ".join(sorted(self._objects_for(player)))
        return (
            f"'{BLOCKED_MARKER}' marks your teammate's objects. You cannot move them. "
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
                "message: <your message to your teammate>\n"
                "move: <OBJECT> <DIRECTION>\n\n"
                "OR, if ALL your objects are at their target positions:\n"
                "message: done\n"
                "done: all my objects are placed"
            )
        return (
            "move: <OBJECT> <DIRECTION>\n\n"
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
        if self.player_a_done and self.player_b_done:
            # Structurally unreachable (the last done triggers success when the
            # board is solved); defensive stop so the game can never cycle
            # through done players.
            self.lost = True
            self.log_to_self("lose", "All players declared done but the board is not solved")
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
                    f"{self._state_block(player)}"
                ))
            return False

        if parsed.get("is_done"):
            if not self._own_objects_at_target(player):
                self.violated_request_counts += 1
                self.log_to_self(
                    "invalid_done",
                    f"{player.role_name}: declared done but not all their objects are at target positions",
                )
                self.lost = True
                self.log_to_self(
                    "lose",
                    f"{player.role_name} declared done before all their objects reached their targets",
                )
                return False
            self.parsed_request_counts += 1
            self.current_parsed = parsed
            self.reprompt_attempts = 0
            # A valid done ends this player's participation; clear any pending
            # reprompt so the turn passes instead of re-asking the done player.
            self.reprompt_pending = False
            return True

        # Structured & hybrid protocols prescribe a message schema (structure +
        # word caps); enforce it. A violation is treated like any parse failure:
        # reprompt, consume the retry budget, abort on exhaustion. Done
        # declarations are exempt (their message is the "done" marker, above).
        msg = parsed.get("message", "")
        reminder = _schema_violation(self.comm_protocol, msg)
        if reminder is not None:
            self.violated_request_counts += 1
            self.log_to_self(
                "malformed_message",
                f"{player.role_name}: {self.comm_protocol} message rules not followed — got: {msg[:80]}",
            )
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", f"Too many malformed {self.comm_protocol} messages")
            else:
                self.reprompt_pending = True
                lines = [
                    f"Your message did not follow the required {self.comm_protocol} message rules.",
                    reminder,
                ]
                echo = msg.strip()
                if echo:
                    if len(echo) > 200:
                        echo = echo[:200] + "…"
                    lines.append(f'You wrote: "{echo}"')
                self.set_context_for(player, (
                    "\n".join(lines) + "\n\n"
                    f"{self._format_reminder()}\n\n"
                    f"{self._state_block(player)}"
                ))
            return False

        self.move_attempts += 1

        obj = parsed["object"]
        direction = parsed["direction"]
        allowed = self._objects_for(player)
        error = self.board.validate_move(
            obj, direction, allowed_objects=allowed
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
                masked_attempt = self.use_masking and obj == BLOCKED_MARKER
                if masked_attempt and self.masking_hints:
                    guidance = self._masked_move_message(player)
                else:
                    guidance = f"Invalid move: {err_msg}\nPlease try again."
                self.set_context_for(player, (
                    f"{guidance}\n\n"
                    f"{self._format_reminder()}\n\n"
                    f"{self._state_block(player)}"
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
            return f"{self.current_parsed['object']} {self.current_parsed['direction']}"
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

        # Cache this player's most recent message for relay (to everyone, including
        # itself). For protocols without a message this stays None.
        self.last_message[player.role_name] = message if (message and self.with_message) else None

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
            self.set_context_for(other, self._turn_context(
                other,
                notice=(f"{player.role_name} has finished. All their objects are at their target positions. "
                        f"Continue alone until all objects are placed."),
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
            self.lost = True
            self.log_to_self("lose", "Board state repeated 3 times — players stuck in cycle")
            return

        if self.board.is_solved():
            self.success = True
            self.log_to_self("success", "All objects have reached their target positions!")
            return

        other = self.player_b if player == self.player_a else self.player_a
        other_done = self._is_done_player(other)

        if other_done:
            # Partner finished; current player continues alone.
            self.set_context_for(player, self._turn_context(
                player,
                notice="Your teammate has finished. Continue placing your remaining objects alone.",
            ))
        else:
            # Relay to the partner: their own + the mover's latest message, plus the
            # refreshed board. (The rules preamble is pinned on each player, so a
            # separate intro message is no longer needed.)
            self.set_context_for(other, self._turn_context(other))

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
        self.log_key(f"{self.ROLE_NAMES[0]} Done", int(self.player_a_done))
        self.log_key(f"{self.ROLE_NAMES[1]} Done", int(self.player_b_done))


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
        # Two players by default; the key is absent from dyad instances.
        num_players = self.game_instance.get("num_players", 2)
        self.log_episode_score(
            "Deadlock Flag", deadlock_flag(move_log, starts, targets, num_players)
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
