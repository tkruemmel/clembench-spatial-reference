"""Matrixgame: move six objects from their start cells to their target cells.

  matrixgame_solo  one player sees and moves all six objects
  matrixgame_dyad  two players, three objects each
  matrixgame_3p    three players, two objects each
  matrixgame_6p    six players, one object each

Every turn one player moves one of its own objects one cell. In the team games
a player sees only its own objects and its own targets (teammates' objects are
X) and may declare `done` once all its objects are home: it then takes no more
turns and its objects stay put. Under a messaging protocol each turn also
carries a message to the teammates:

  none        move only
  structured  PLAN: move X from (Rr,Cc) to (Rr,Cc) — REQUEST: ..., each part <= 10 words
  hybrid      free prose (<= 100 words) followed by the structured clause
  freeform    free prose; the prompt asks for at most 100 words (not enforced)

The structured and hybrid schemas are enforced like the move grammar: a
violation is re-asked and counts against the retry budget.

The player keeps the rules as a fixed preamble and gets a fresh context every
turn: the MESSAGE BOARD and MOVE BOARD over the last rounds (utils/history.py),
the current board and the goal board. The episode is won when every object is
on its target, and lost on a `done` with an object off target, on the third
occurrence of any board state, or at the turn limit. It is aborted when a
player exceeds the retry budget.
"""

import logging
import re
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

from utils import history
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

MAX_TOKENS = 16384

# ── Response patterns ──────────────────────────────────────────────────

# Matches a move line whose object is the masked marker (a teammate's object).
# Used to give a targeted semantic redirect instead of a generic parse error.
MASKED_OBJECT_MOVE_PATTERN = re.compile(
    rf"move:\s*{re.escape(BLOCKED_MARKER)}\b",
    re.IGNORECASE,
)

# Solo and `none`: the response IS the move line.
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

# Structured-protocol message schema. Anchored full-match; named groups feed
# the word caps. The PLAN clause may sit on its own line (re.DOTALL).
STRUCTURED_MSG_PATTERN = re.compile(
    r"\A\s*PLAN:\s*(?P<plan>move\s+\S+\s+from\s*\(R\d+,C\d+\)\s+to\s*\(R\d+,C\d+\))\s*[—\-]+\s*REQUEST:\s*(?P<request>.+?)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

# Hybrid-protocol message schema: free prose, then the structured clause.
# Prose needs >=1 non-space char, so a clause-only message (no prose) is
# rejected — that would just be structured.
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
    """A move dict, a done dict ({"is_done": True}) or None if malformed."""
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

    # Under a messaging protocol the canonical form is "message: done" +
    # "done: ...", but a bare "done: ..." line is accepted too: a done
    # declaration ends the player's participation and its message part carries
    # no information.
    done_pattern = COMM_DONE_PATTERN if with_message else DONE_ONLY_PATTERN
    match = done_pattern.match(response)
    if match:
        result: Dict[str, str] = {"is_done": True}
        if with_message:
            result["message"] = match.group("message").strip()
        return result
    if with_message and DONE_ONLY_PATTERN.match(response):
        return {"is_done": True}
    return None


def _preamble(template: str, own_objects: Set[str], grid_size: int, role: str, n_others: int) -> str:
    return (template.replace("$YOUR_OBJECTS$", ", ".join(sorted(own_objects)))
            .replace("$GRID_SIZE$", str(grid_size))
            .replace("$YOUR_ROLE$", role)
            .replace("$N_OTHER_PLAYERS$", str(n_others)))


# ── Player ─────────────────────────────────────────────────────────────

class MatrixPlayer(Player):
    def __init__(self, model: Model, role_name: str, own_objects: Set[str]):
        super().__init__(model)
        self.role_name = role_name
        self.own_objects = own_objects
        # The rules, pinned at setup. The model sees this preamble plus only the
        # current turn's context, never the whole transcript.
        self.preamble = ""

    def get_perspective(self):
        return [{"role": "user", "content": self.preamble}]

    def _custom_response(self, context):
        # Used by clemcore's mock player; emit a deterministic valid-format move.
        first = sorted(self.own_objects)[0]
        return f"move: {first} down"


# ── Shared game-master behaviour ───────────────────────────────────────

class _MatrixGameMaster(DialogueGameMaster):

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        self.request_counts = 0
        self.move_attempts = 0
        self.illegal_attempts = 0
        self.blocked_attempts = 0
        self.parsed_request_counts = 0
        self.violated_request_counts = 0
        self.move_log: List[Dict] = []

    def _setup_board(self, game_instance: Dict) -> None:
        self.grid_size = game_instance["grid_size"]
        self.max_retries = game_instance["max_retries"]
        self.max_turns = game_instance["max_turns"]
        for model in self.player_models:
            model.set_gen_arg("max_tokens", MAX_TOKENS)
        self.board = Board(self.grid_size, {tuple(w) for w in game_instance["walls"]})
        for obj, (r, c) in game_instance["start_positions"].items():
            self.board.place_object(obj, r, c)
        for obj, (r, c) in game_instance["target_positions"].items():
            self.board.set_target(obj, r, c)
        self.record: List[Dict] = []
        self.success = False
        self.aborted = False
        self.lost = False
        self.reprompt_attempts = 0
        self.reprompt_pending: bool = False
        self.current_parsed: Optional[Dict[str, str]] = None
        self.seen_states: Dict[str, int] = {self.board.state_key(): 1}

    def _does_game_proceed(self) -> bool:
        if self.success or self.aborted or self.lost:
            return False
        if len(self.move_log) >= self.max_turns:
            self.lost = True
            self.log_to_self("lose", f"Max turns ({self.max_turns}) reached without success")
            return False
        return True

    def _retry(self, player: Player, abort_reason: str, context: str) -> None:
        """Count a rejected response against the retry budget: re-ask, or abort."""
        self.reprompt_attempts += 1
        if self.reprompt_attempts > self.max_retries:
            self.aborted = True
            self.log_to_self("abort", abort_reason)
        else:
            self.reprompt_pending = True
            self.set_context_for(player, context)

    def _reject_move(self, player: Player, error, guidance: str) -> bool:
        code, err_msg = error
        if code == "occupied":
            self.blocked_attempts += 1
        else:
            self.illegal_attempts += 1
        self.violated_request_counts += 1
        self.log_to_self("invalid_move", f"{player.role_name}: [{code}] {err_msg}")
        self._retry(player, "Too many invalid moves",
                    f"{guidance}\n\n{self._format_reminder()}\n\n{self._state_block(player)}")
        return False

    def _accept(self, parsed: Dict) -> bool:
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

    def _apply_move(self, player: Player, message: Optional[str], with_player: bool) -> bool:
        """Apply the accepted move; True when the episode ended on it."""
        obj = self.current_parsed["object"]
        direction = self.current_parsed["direction"]
        from_row, from_col = self.board.object_positions[obj]
        dr, dc = DIRECTIONS[direction]
        self.board.apply_move(obj, direction)
        entry = {"player": player.role_name} if with_player else {}
        entry.update({
            "object": obj,
            "direction": direction,
            "from": [from_row, from_col],
            "to": [from_row + dr, from_col + dc],
            "message": message,
        })
        self.move_log.append(entry)
        self.log_to_self("move", f"{player.role_name} moved {obj} {direction}")
        if message:
            self.log_to_self("message", f"{player.role_name}: {message}")
        state = self.board.state_key()
        self.seen_states[state] = self.seen_states.get(state, 0) + 1
        return self.seen_states[state] >= 3

    def _turn_context(self, player: Player) -> str:
        return f"{self._state_block(player)}\n\nYour turn. Respond with:\n{self._format_reminder()}"

    def _log_counts(self) -> None:
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


# ── Solo ───────────────────────────────────────────────────────────────

class SoloMatrixGameMaster(_MatrixGameMaster):

    def _on_setup(self, **game_instance):
        self._setup_board(game_instance)
        self.objects: Set[str] = set(game_instance["objects"])
        self.player = MatrixPlayer(self.player_models[0], "Player", self.objects)
        self.player.preamble = _preamble(game_instance["player_prompt"], self.objects, self.grid_size, "Player", 0)
        self.add_player(self.player, initial_context=(
            f"CURRENT BOARD:\n{self.board.render()}\n\n"
            f"GOAL BOARD (where each object must end up):\n{self.board.render_targets()}"
        ))

    def _format_reminder(self) -> str:
        return "move: <OBJECT> <DIRECTION>"

    def _state_block(self, player: Player = None) -> str:
        """MOVE BOARD, current and goal boards: the tail of every turn and retry context."""
        return (
            history.boards(self.record, self.player.role_name, "")
            + f"CURRENT BOARD:\n{self.board.render()}\n\n"
            f"GOAL BOARD:\n{self.board.render_targets()}"
        )

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        match = MOVE_ONLY_PATTERN.match(response)
        if match is None:
            self.violated_request_counts += 1
            self.log_to_self("invalid_format", f"{player.role_name}: could not parse response")
            self._retry(player, "Too many invalid responses", (
                "Your response could not be parsed. Please respond exactly in this format:\n"
                f"{self._format_reminder()}\n\n"
                f"{self._state_block()}"
            ))
            return False
        parsed = {"object": match.group("object").strip().upper(),
                  "direction": match.group("direction").strip().lower()}
        self.move_attempts += 1
        error = self.board.validate_move(parsed["object"], parsed["direction"], allowed_objects=player.own_objects)
        if error:
            return self._reject_move(player, error, f"Invalid move: {error[1]}\nPlease try again.")
        return self._accept(parsed)

    def _next_player(self) -> Player:
        return self.player

    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        # Recorded before the move is applied, so the context set below includes it.
        history.capture(self.record, player.role_name, self.current_parsed, self.board, DIRECTIONS)
        if self._apply_move(player, None, with_player=False):
            self.lost = True
            self.log_to_self("lose", "Board state repeated 3 times — model is in a cycle")
            return
        if self.board.is_solved():
            self.success = True
            self.log_to_self("success", "All objects have reached their targets")
            return
        self.set_context_for(player, self._turn_context(player))

    def _on_after_game(self):
        self._log_counts()


# ── Dyad, 3p, 6p ───────────────────────────────────────────────────────

class TeamMatrixGameMaster(_MatrixGameMaster):

    def _on_setup(self, **game_instance):
        self._setup_board(game_instance)
        self.comm_protocol: str = game_instance["comm_protocol"]
        self.with_message: bool = self.comm_protocol != COMM_NONE
        self.player_objects: List[Set[str]] = [set(group) for group in game_instance["player_objects"]]
        self.num_players = len(self.player_objects)
        self.player_done: List[bool] = [False] * self.num_players
        self.role_names = history.SEATS[:self.num_players]
        # The dyad was played with the teammate's lines of a round in seeded
        # random order. A dyad round holds one teammate turn, so this can only
        # put a done declaration before its message (once in the 240 played
        # episodes); the 3p and 6p games list every line in the order sent.
        self.seed = str(game_instance["game_id"])
        self.shuffle = self.num_players == 2
        # The dyad's prompts address one teammate, the 3p/6p prompts several.
        dyad = self.num_players == 2
        self.mates = "teammate" if dyad else "teammates"
        self.mates_poss = "teammate's" if dyad else "teammates'"
        self.mate_log = "partner" if dyad else "teammate"
        goal_of = "YOUR objects" if dyad else "each object"

        self._player_to_index: Dict[MatrixPlayer, int] = {}
        for i, role in enumerate(self.role_names):
            model = self.player_models[i] if i < len(self.player_models) else self.player_models[0]
            player = MatrixPlayer(model, role, self.player_objects[i])
            self._player_to_index[player] = i
            player.preamble = _preamble(game_instance["player_prompt"], self.player_objects[i], self.grid_size,
                                        role, self.num_players - 1)
            self.add_player(player, initial_context=(
                f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                f"GOAL BOARD (where {goal_of} must end up):\n{self._render_targets_for(player)}"
            ))

    # ── helpers ────────────────────────────────────────────────────────

    def _objects_for(self, player: Player) -> Set[str]:
        return self.player_objects[self._player_to_index[player]]

    def _render_for(self, player: Player) -> str:
        return self.board.render(self._objects_for(player))

    def _render_targets_for(self, player: Player) -> str:
        return self.board.render_targets(self._objects_for(player))

    def _state_block(self, player: Player) -> str:
        """MESSAGE and MOVE BOARD, current and goal boards: the tail of every turn and retry context."""
        return (
            history.boards(self.record, player.role_name, self.seed, self.with_message, self.shuffle)
            + f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
            f"GOAL BOARD:\n{self._render_targets_for(player)}"
        )

    def _masked_move_message(self, player: Player) -> str:
        """Targeted redirect for a player that tried to move a masked ('X')
        object — a teammate's object, which it cannot control."""
        own = ", ".join(sorted(self._objects_for(player)))
        return (
            f"'{BLOCKED_MARKER}' marks your {self.mates_poss} objects. You cannot move them. "
            f"You can only move YOUR objects: {own}. Choose one of yours."
        )

    def _is_done_player(self, player: Player) -> bool:
        return self.player_done[self._player_to_index[player]]

    def _own_objects_at_target(self, player: Player) -> bool:
        return all(
            self.board.object_positions.get(obj) == self.board.target_positions.get(obj)
            for obj in self._objects_for(player)
        )

    def _format_reminder(self) -> str:
        if self.with_message:
            return (
                f"message: <your message to your {self.mates}>\n"
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
        if not super()._does_game_proceed():
            return False
        if all(self.player_done):
            # Unreachable (the last done solves the board); a defensive stop so
            # the game can never cycle through finished players.
            self.lost = True
            self.log_to_self("lose", "All players declared done but the board is not solved")
            return False
        return True

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        parsed = _parse_response(response, self.with_message)

        if parsed is None:
            self.violated_request_counts += 1
            # A common failure: the player tries to move a masked teammate object
            # (e.g. `move: X ...`), which never matches the move grammar.
            # Redirect it semantically instead of with a generic format error.
            masked_attempt = bool(MASKED_OBJECT_MOVE_PATTERN.search(response))
            if masked_attempt:
                self.log_to_self("invalid_move", f"{player.role_name}: attempted to move a masked {self.mate_log} object")
            else:
                self.log_to_self("invalid_format", f"{player.role_name}: could not parse response")
            guidance = (self._masked_move_message(player) if masked_attempt
                        else "Your response could not be parsed. Please respond exactly in this format:")
            self._retry(player, "Too many invalid responses",
                        f"{guidance}\n{self._format_reminder()}\n\n{self._state_block(player)}")
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
            return self._accept(parsed)

        # Structured & hybrid prescribe a message schema (structure + word
        # caps); a violation is treated like any parse failure. Done
        # declarations are exempt (their message is the "done" marker, above).
        msg = parsed.get("message", "")
        reminder = _schema_violation(self.comm_protocol, msg)
        if reminder is not None:
            self.violated_request_counts += 1
            self.log_to_self(
                "malformed_message",
                f"{player.role_name}: {self.comm_protocol} message rules not followed — got: {msg[:80]}",
            )
            lines = [f"Your message did not follow the required {self.comm_protocol} message rules.", reminder]
            echo = msg.strip()
            if echo:
                if len(echo) > 200:
                    echo = echo[:200] + "…"
                lines.append(f'You wrote: "{echo}"')
            self._retry(player, f"Too many malformed {self.comm_protocol} messages",
                        "\n".join(lines) + "\n\n"
                        f"{self._format_reminder()}\n\n"
                        f"{self._state_block(player)}")
            return False

        self.move_attempts += 1
        obj = parsed["object"]
        error = self.board.validate_move(obj, parsed["direction"], allowed_objects=self._objects_for(player))
        if error:
            guidance = (self._masked_move_message(player) if obj == BLOCKED_MARKER
                        else f"Invalid move: {error[1]}\nPlease try again.")
            return self._reject_move(player, error, guidance)
        return self._accept(parsed)

    def _next_player(self) -> Player:
        players = self.get_players()
        n = len(players)
        for _ in range(n):
            self._current_player_idx = (self._current_player_idx + 1) % n
            candidate = players[self._current_player_idx]
            if not self._is_done_player(candidate):
                return candidate
        # All players done — unreachable, since the board is solved before the
        # last player can declare done. Defensive fallback: advance anyway.
        self._current_player_idx = (self._current_player_idx + 1) % n
        return players[self._current_player_idx]

    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        # Recorded before the turn is applied, so the contexts set below include it.
        history.capture(self.record, player.role_name, self.current_parsed, self.board, DIRECTIONS)
        message = self.current_parsed.get("message")
        player_idx = self._player_to_index[player]

        if self.current_parsed.get("is_done"):
            self.player_done[player_idx] = True
            self.log_to_self("player_done", f"{player.role_name} declared done")
            if message:
                self.log_to_self("message", f"{player.role_name}: {message}")
        elif self._apply_move(player, message, with_player=True):
            self.lost = True
            self.log_to_self("lose", "Board state repeated 3 times — players stuck in cycle")
            return

        if self.board.is_solved():
            self.success = True
            self.log_to_self("success", "All objects have reached their target positions!")
            return

        # Every active teammate gets the refreshed context. When none is left,
        # the mover keeps the turn and gets it instead. A finish is announced
        # only by the MESSAGE BOARD's "declared done" line.
        others = [other for i, other in enumerate(self.get_players())
                  if i != player_idx and not self.player_done[i]]
        if not others and not self.current_parsed.get("is_done"):
            others = [player]
        for other in others:
            self.set_context_for(other, self._turn_context(other))

    def _on_after_game(self):
        self._log_counts()
        for role, done in zip(self.role_names, self.player_done):
            self.log_key(f"{role} Done", int(done))


# ── Scorer ─────────────────────────────────────────────────────────────

class MatrixGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict, num_players: int):
        super().__init__(game_name, experiment, game_instance)
        self.num_players = num_players

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

        # Spatial efficiency
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

        # Failure modes
        osc_count, osc_flag = oscillation_stats(move_log, k=4)
        self.log_episode_score("Oscillation Count", osc_count)
        self.log_episode_score("Oscillation Flag", osc_flag)
        self.log_episode_score(
            "Deadlock Flag", deadlock_flag(move_log, starts, targets, self.num_players)
        )

        # Communication
        lengths = message_lengths(move_log)
        self.log_episode_score("Message Count", len(lengths))
        if lengths:
            self.log_episode_score("Message Length Mean", round(sum(lengths) / len(lengths), 2))
            self.log_episode_score("Message Length Max", max(lengths))


# ── Benchmark entry point ──────────────────────────────────────────────

class MatrixGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> DialogueGameMaster:
        master = SoloMatrixGameMaster if self.game_spec.players == 1 else TeamMatrixGameMaster
        return master(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> MatrixGameScorer:
        return MatrixGameScorer(self.game_name, experiment, game_instance, self.game_spec.players)
