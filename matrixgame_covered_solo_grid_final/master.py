"""matrixgame_covered_solo_grid_final: final arm of the grid solo game.

Same boards as matrixgame_covered_solo_grid; the instances are a copy of its
silent experiments (the final arm has no thinking variant) with player_prompt
swapped (see prompt_final/) and the budget set from the attainable solo
optimum, as in the team arms. The master differs in one way: every turn and
retry context carries the player's last ROUNDS moves as a MOVE BOARD, in the
team arms' format.

clemcore loads this file by path and requires exactly one GameBenchmark subclass
in the module namespace, so the base master is imported as a module.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMPACT_PKG = os.path.join(_REPO_ROOT, "matrixgame_covered_solo")

# The compact master imports its helpers as `utils.*`, resolved from its own
# directory; clemcore only puts *this* directory on sys.path while loading.
for _path in (_REPO_ROOT, _COMPACT_PKG):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matrixgame_covered_solo.master as base  # noqa: E402
from matrixgame_covered_shared import history  # noqa: E402

class FinalSoloMatrixGameMaster(base.SoloMatrixGameMaster):

    def _on_setup(self, **game_instance):
        self.record = []
        super()._on_setup(**game_instance)

    def _on_valid_player_response(self, player, parsed_response):
        # Before the base applies the move, so the context it sets includes it.
        history.capture(self.record, player.role_name, self.current_parsed, self.board, base.DIRECTIONS)
        super()._on_valid_player_response(player, parsed_response)

    def _state_block(self):
        return (
            history.boards(self.record, self.player.role_name, "")
            + super()._state_block()
        )

    def _turn_context(self):
        return f"{self._state_block()}\n\nYour turn. Respond with:\n{self._format_reminder()}"


class FinalSoloMatrixGameBenchmark(base.SoloMatrixGameBenchmark):

    def create_game_master(self, experiment, player_models):
        return FinalSoloMatrixGameMaster(self.game_spec, experiment, player_models)
