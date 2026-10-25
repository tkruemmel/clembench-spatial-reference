"""The chrono variant is the final 3p/6p arm with teammates' messages in the
order sent: same instances minus the shuffle sentence, same master minus the
shuffle."""

import importlib.util
import json
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
FINAL = PKG.parent / "matrixgame_covered_comm_nplayer_grid_final"
SENTENCE = " Within a round, your teammates' messages are listed in random order."

_spec = importlib.util.spec_from_file_location("chrono_nplayer_master", PKG / "master.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
ROUNDS = M.history.ROUNDS


def test_instances_are_the_final_ones_without_the_shuffle_sentence():
    for f in ("instances_3p.json", "instances_6p.json"):
        final = (FINAL / "in" / f).read_text(encoding="utf-8")
        chrono = (PKG / "in" / f).read_text(encoding="utf-8")
        assert chrono == final.replace(SENTENCE, "") and "random order" not in chrono, f


class _Dummy:
    name = "dummy"

    def set_gen_arg(self, *_):
        pass


def test_messages_are_listed_in_the_order_sent():
    from clemcore.clemgame.registry import GameSpec
    objs = "ABCDEF"
    m = M.ChronoNPlayerMatrixGameMaster(GameSpec(game_name="g", game_path=str(PKG), players=3),
                                        {"name": "x"}, [_Dummy() for _ in range(3)])
    m._on_setup(game_id=1, grid_size=8, walls=[], objects=list(objs), num_players=3,
                player_objects=[["A", "B"], ["C", "D"], ["E", "F"]],
                start_positions={o: [0, i] for i, o in enumerate(objs)},
                target_positions={o: [7, i] for i, o in enumerate(objs)},
                optimal_moves=42, max_turns=100, max_retries=2, strict=False, use_masking=True,
                spatial_level="S1", comm_protocol="freeform", compact_board=False,
                player_prompt="PROMPT $YOUR_ROLE$ $YOUR_OBJECTS$ $GRID_SIZE$ $N_OTHER_PLAYERS$")
    players, owned = m.get_players(), ["AB", "CD", "EF"]
    for t in range(3 * ROUNDS + 1):  # objects only move down: no board recurs
        response = f"message: turn {t + 1}\nmove: {owned[t % 3][(t // 3) % 2]} down"
        assert m._validate_player_response(players[t % 3], response)
        m._on_valid_player_response(players[t % 3], m._parse_response(players[t % 3], response))
    messages = m.get_context_for(players[1])["content"].split("\n\n")[0].splitlines()[1:]
    # Orange's window opens at its own turn 2 and runs through turn 13.
    assert [int(l.split()[-1][:-1]) for l in messages] == list(range(2, 3 * ROUNDS + 2))
    assert "said" in messages[0] and "(you)" in messages[0]
