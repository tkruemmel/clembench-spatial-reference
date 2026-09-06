"""Instance generator for matrixgame_covered_comm.

Boards come from matrixgame_covered_shared.boardgen (single source of truth
for S1-S4 board generation, shared across _solo, _comm, and _comm_nplayer so
the physical puzzle distribution is identical by construction). Ownership
(the dyad's per-level player_a/player_b split) is layered on top here from
board["dyad_split"].

Spatial levels
──────────────
S1  No walls, region-partitioned placement. No dependency chains.
    Player assignment: first half / second half of objects.

S2  4-6 wall segments (~1.5× Manhattan ratio). Length-2 dependency chain
    [A←B←C] with cross-agent ownership: A,C→Player A; B→Player B.
    Two remaining free objects split between players.

S3  Dead-end corridor + length-3 chain [A←B←C←D] with cross-agent
    alternation. Secondary length-1 chain [E←F]. ~2× Manhattan ratio.
    Player A: {D, B, E}; Player B: {C, A, F}.

S4  Same map as S3 (dead-end corridor + ~2× Manhattan walls, 2-connected).
    INNER (Player B) starts at the corridor dead end; OUTER (Player A) starts
    at the adjacent interior cell targeting the dead end. OUTER must retreat
    out of the corridor to let INNER exit, then re-enter — forced non-monotonic
    trajectory. Two additional length-1 chains.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

from matrixgame_covered_shared.boardgen import generate_board, instance_rng


class MatrixGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        variant = kwargs.get("variant", "exp2_s2_none")
        variant_config = self.load_json("resources/config")[variant]
        common_config = self.load_json("resources/common_config")

        grid_size = common_config["grid_size"]
        max_retries = common_config["max_retries"]
        strict = common_config.get("strict", False)
        all_objects = common_config["objects"]

        num_objects = variant_config["num_objects"]
        spatial_level = variant_config["spatial_level"]
        comm_protocol = variant_config["comm_protocol"]
        use_masking = variant_config.get("use_masking", True)
        compact_board = variant_config.get("compact_board", False)

        objects = all_objects[:num_objects]
        prompt_template = self.load_template(
            f"resources/initial_prompts/en/player_prompt_{comm_protocol}"
        )

        num_instances = kwargs.get("num_instances", 8)

        # Rendering suffix. This package ships compact-only variants; the grid
        # twin (matrixgame_covered_comm_grid) reuses this generator with
        # compact_board=false and its own templates.
        experiment_name = f"{spatial_level}_{comm_protocol}"
        experiment_name += "_compact" if compact_board else "_grid"

        experiment = self.add_experiment(experiment_name)
        experiment["common_config"] = common_config
        experiment["variant_config"] = variant_config

        for idx in range(num_instances):
            rng = instance_rng(seed, spatial_level, idx)
            board = generate_board(spatial_level, grid_size, objects, rng)
            player_a_objects, player_b_objects = board["dyad_split"]

            game_instance = self.add_game_instance(experiment, idx + 1)
            game_instance["grid_size"] = grid_size
            game_instance["walls"] = board["walls"]
            game_instance["objects"] = objects
            game_instance["player_a_objects"] = player_a_objects
            game_instance["player_b_objects"] = player_b_objects
            game_instance["start_positions"] = board["start_positions"]
            game_instance["target_positions"] = board["target_positions"]
            game_instance["optimal_moves"] = board["optimal_moves"]
            game_instance["max_turns"] = max(4 * board["optimal_moves"], 20)
            game_instance["max_retries"] = max_retries
            game_instance["strict"] = strict
            game_instance["use_masking"] = use_masking
            game_instance["masking_hints"] = common_config.get("masking_hints", True)
            game_instance["spatial_level"] = spatial_level
            game_instance["comm_protocol"] = comm_protocol
            game_instance["compact_board"] = compact_board
            game_instance["player_prompt"] = prompt_template
            if board.get("device"):
                game_instance["device"] = board["device"]


# ── Entry point ────────────────────────────────────────────────────────
#
# All variants are compact-board only.
# Exp2 covers S2; Exp3 covers S1/S3/S4.
#
# Usage:
#   python instancegenerator.py                          # generate everything (Exp2 + Exp3)
#   python instancegenerator.py exp2                     # Exp2 only (S2)
#   python instancegenerator.py exp3                     # Exp3 only (S1/S3/S4)
#   python instancegenerator.py exp3 --n 10              # override instance count
#   python instancegenerator.py exp3 --levels s1 s3      # only levels S1, S3
#   python instancegenerator.py --levels s2              # only S2 (Exp2 only)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiments",
        nargs="*",
        # Note: `choices` is intentionally omitted here (pre-existing bug,
        # unrelated to the boardgen rewiring): argparse re-validates the
        # positional's *default* against `choices` when nargs='*' and no
        # value is given, so `choices=["exp2","exp3"]` combined with
        # `default=["exp2","exp3"]` always raised "invalid choice:
        # \"['exp2','exp3']\"" on this Python version whenever the CLI was
        # invoked without positional args. Validated manually below instead.
        default=["exp2", "exp3"],
        help="Which experiments to generate (default: both)",
    )
    parser.add_argument(
        "--n", type=int, default=8,
        help="Number of instances per condition (default: 8)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["s1", "s2", "s3", "s4"],
        default=["s1", "s2", "s3", "s4"],
        metavar="LEVEL",
        help="Spatial levels to include (default: all). E.g. --levels s1 s2 s3",
    )
    args = parser.parse_args()
    invalid = set(args.experiments) - {"exp2", "exp3"}
    if invalid:
        parser.error(f"argument experiments: invalid choice(s): {sorted(invalid)} (choose from exp2, exp3)")
    selected_levels = set(args.levels)

    gen = MatrixGameInstanceGenerator()
    N = args.n

    if "exp2" in args.experiments and "s2" in selected_levels:
        # Same seed (42) → identical boards across conditions, isolating protocol effect.
        for protocol in ["none", "structured", "freeform", "hybrid"]:
            variant = f"exp2_s2_{protocol}_compact"
            gen.generate(filename="instances.json", seed=42,
                         variant=variant, num_instances=N)
            print(f"Generated: {variant} (n={N})")

    if "exp3" in args.experiments:
        # All four spatial levels × all three communication protocols.
        # Same seed (42) → identical boards across protocols per level,
        # isolating spatial difficulty from protocol choice.
        for level in ("s1", "s3", "s4"):
            if level not in selected_levels:
                continue
            for protocol in ["none", "structured", "freeform", "hybrid"]:
                variant = f"exp3_{level}_{protocol}_compact"
                gen.generate(filename="instances.json", seed=42,
                             variant=variant, num_instances=N)
                print(f"Generated: {variant} (n={N})")
