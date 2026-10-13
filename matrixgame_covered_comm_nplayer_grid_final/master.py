"""matrixgame_covered_comm_nplayer_grid_final: final arm of the grid 3p/6p games
(matrixgame_covered_comm_3p_grid_final and matrixgame_covered_comm_6p_grid_final).

Same boards, ownership, protocols and A_k budgets as
matrixgame_covered_comm_nplayer_grid_updated_prompt; the instances are a copy of
its in/instances_{3p,6p}.json with only player_prompt swapped (see
prompt_final/). The master differs in two ways. Every turn and retry context
carries the last ROUNDS rounds as a MESSAGE BOARD (every message with its
sender, teammates' messages shuffled within a round) and a MOVE BOARD (every
move in the order made, no player named, teammates' objects as X, the
recipient's own moves in full), and the teammates that have finished; the
boards replace the latest-message block. And the seats are named by colour
(history.SEATS), so a name gives away neither a seat's place in the turn order
nor an object letter.

clemcore loads this file by path and requires exactly one GameBenchmark subclass
in the module namespace, so the base master is imported as a module.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMPACT_PKG = os.path.join(_REPO_ROOT, "matrixgame_covered_comm_nplayer")

# The compact master imports its helpers as `utils.*`, resolved from its own
# directory; clemcore only puts *this* directory on sys.path while loading.
for _path in (_REPO_ROOT, _COMPACT_PKG):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matrixgame_covered_comm_nplayer.master as base  # noqa: E402
from matrixgame_covered_shared import history  # noqa: E402

class FinalNPlayerMatrixGameMaster(base.NPlayerMatrixGameMaster):

    ROLE_NAMES = history.SEATS

    def _on_setup(self, **game_instance):
        self.record = []
        self.seed = str(game_instance["game_id"])
        super()._on_setup(**game_instance)

    def _on_valid_player_response(self, player, parsed_response):
        # Before the base applies the turn, so the contexts it sets include it.
        history.capture(self.record, player.role_name, self.current_parsed, self.board, base.DIRECTIONS)
        super()._on_valid_player_response(player, parsed_response)

    def _state_block(self, player):
        # The base announces a finish only once; a done player's objects are
        # fixed for the rest of the episode, so the list stays in view.
        finished = [p.role_name for i, p in enumerate(self.get_players()) if self.player_done[i]]
        names = " and ".join(filter(None, [", ".join(finished[:-1]), finished[-1]])) if finished else ""
        status = (f"{names} {'have' if len(finished) > 1 else 'has'} finished. "
                  "Their objects will not move again.\n\n" if finished else "")
        return (
            history.boards(self.record, player.role_name, self.seed)
            + status
            + super()._state_block(player)
        )

    def _build_broadcast_context(self, recipient, move_desc=""):
        move_block = f"{move_desc}\n\n" if move_desc else ""
        return (
            f"{move_block}{self._state_block(recipient)}\n\n"
            f"Your turn. Respond with:\n{self._format_reminder()}"
        )


class FinalNPlayerMatrixGameBenchmark(base.NPlayerMatrixGameBenchmark):

    def create_game_master(self, experiment, player_models):
        return FinalNPlayerMatrixGameMaster(self.game_spec, experiment, player_models)
