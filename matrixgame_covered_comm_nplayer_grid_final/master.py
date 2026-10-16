"""matrixgame_covered_comm_nplayer_grid_final: final arm of the grid 3p/6p games
(matrixgame_covered_comm_3p_grid_final and matrixgame_covered_comm_6p_grid_final).

Same boards, ownership, protocols and A_k budgets as
matrixgame_covered_comm_nplayer_grid_updated_prompt; the instances are a copy of
its in/instances_{3p,6p}.json with only player_prompt swapped (see
prompt_final/). The master differs in three ways. Every turn and retry context
carries the last ROUNDS rounds as a MESSAGE BOARD (every message with its
sender and who declared done, teammates' messages shuffled within a round)
and a MOVE BOARD (every move in the order made, no player named, teammates'
objects as X, the recipient's own moves in full); the boards replace the
latest-message block. The MESSAGE BOARD's "declared done" line is the only
finish signal (none under `none`): the base's finished notices are dropped. The seats are named by colour (history.SEATS), so a name
gives away neither a seat's place in the turn order nor an object letter. And
the grid's column labels sit over their cells (history.aligned).

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

    def _render_for(self, player):
        return history.aligned(super()._render_for(player))

    def _render_targets_for(self, player):
        return history.aligned(super()._render_targets_for(player))

    def _state_block(self, player):
        return (
            history.boards(self.record, player.role_name, self.seed, self.with_message)
            + super()._state_block(player)
        )

    def _build_broadcast_context(self, recipient, move_desc=""):
        # The base's only notices announce a finish. The final arm drops them
        # under every protocol; the MESSAGE BOARD's "declared done" line is
        # the only finish signal, so under `none` there is none.
        return (
            f"{self._state_block(recipient)}\n\n"
            f"Your turn. Respond with:\n{self._format_reminder()}"
        )


class FinalNPlayerMatrixGameBenchmark(base.NPlayerMatrixGameBenchmark):

    def create_game_master(self, experiment, player_models):
        return FinalNPlayerMatrixGameMaster(self.game_spec, experiment, player_models)
