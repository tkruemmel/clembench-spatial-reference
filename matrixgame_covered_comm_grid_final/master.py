"""matrixgame_covered_comm_grid_final: final arm of the grid dyad.

Same boards, protocols and A_k budgets as
matrixgame_covered_comm_grid_updated_prompt; the instances are a copy of its
in/instances.json with player_prompt swapped (see prompt_final/), and at S3
the objects re-split so that B and D belong to different players, as at 3p/6p
(prompt_final/split_s3.py, with budgets on the new A_k). The
master differs in three ways. Every turn and retry context carries the last
ROUNDS rounds as a MESSAGE BOARD (every message with its sender, and who
declared done, which replaces the finished notices; under `none` nothing does)
and a MOVE BOARD (every move with no player named, teammates' objects as X,
the recipient's own moves in full); the boards replace the latest-message
block. The seats are named by colour (history.SEATS), as in the 3p/6p games.
And the grid's column labels sit over their cells (history.aligned).

clemcore loads this file by path and requires exactly one GameBenchmark subclass
in the module namespace, so the base master is imported as a module.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMPACT_PKG = os.path.join(_REPO_ROOT, "matrixgame_covered_comm")

# The compact master imports its helpers as `utils.*`, resolved from its own
# directory; clemcore only puts *this* directory on sys.path while loading.
for _path in (_REPO_ROOT, _COMPACT_PKG):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matrixgame_covered_comm.master as base  # noqa: E402
from matrixgame_covered_shared import history  # noqa: E402

class FinalMatrixGameMaster(base.MatrixGameMaster):

    ROLE_NAMES = history.SEATS[:2]

    def _on_setup(self, **game_instance):
        self.record = []
        self.seed = str(game_instance["game_id"])
        super()._on_setup(**game_instance)

    def _on_valid_player_response(self, player, parsed_response):
        # Before the base applies the turn, so the contexts it sets include it.
        history.capture(self.record, player.role_name, self.current_parsed, self.board, base.DIRECTIONS)
        super()._on_valid_player_response(player, parsed_response)

    def _state_block(self, player):
        return (
            history.boards(self.record, player.role_name, self.seed, self.with_message)
            + super()._state_block(player)
        )

    def _render_for(self, player):
        return history.aligned(super()._render_for(player))

    def _render_targets_for(self, player):
        return history.aligned(super()._render_targets_for(player))

    def _turn_context(self, recipient, notice=""):
        # The base's only notices announce a finish. The final arm drops them
        # under every protocol; the MESSAGE BOARD's "declared done" line is
        # the only finish signal, so under `none` there is none.
        return (
            f"{self._state_block(recipient)}\n\n"
            f"Your turn. Respond with:\n{self._format_reminder()}"
        )


class FinalMatrixGameBenchmark(base.MatrixGameBenchmark):

    def create_game_master(self, experiment, player_models):
        return FinalMatrixGameMaster(self.game_spec, experiment, player_models)
