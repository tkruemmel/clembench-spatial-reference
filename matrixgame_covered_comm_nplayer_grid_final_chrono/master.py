"""matrixgame_covered_comm_nplayer_grid_final_chrono: the final 3p/6p games
(matrixgame_covered_comm_{3p,6p}_grid_final) with teammates' messages on the
MESSAGE BOARD listed in the order sent instead of shuffled within a round.

Everything else is the final arm: the master subclasses its master, and the
instances are its instances minus the sentence announcing the shuffle
(prompt_final/chrono.py). The dyad needs no variant, since its blocks never
hold more than one teammate message.

clemcore loads this file by path and requires exactly one GameBenchmark subclass
in the module namespace, so the final master is imported as a module.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matrixgame_covered_comm_nplayer_grid_final.master as final  # noqa: E402
from matrixgame_covered_shared import history  # noqa: E402


class ChronoNPlayerMatrixGameMaster(final.FinalNPlayerMatrixGameMaster):

    def _state_block(self, player):
        return (
            history.boards(self.record, player.role_name, self.seed, self.with_message, shuffle=False)
            + final.base.NPlayerMatrixGameMaster._state_block(self, player)
        )


class ChronoNPlayerMatrixGameBenchmark(final.FinalNPlayerMatrixGameBenchmark):

    def create_game_master(self, experiment, player_models):
        return ChronoNPlayerMatrixGameMaster(self.game_spec, experiment, player_models)
