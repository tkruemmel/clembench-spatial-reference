"""Instance generator for matrixgame_covered_solo (full-view only).

Boards come from matrixgame_covered_shared.boardgen (single source of truth
for S1-S4 board generation, shared across _solo, _comm, and _comm_nplayer so
the physical puzzle distribution is identical by construction). The single
solo player owns all 6 objects; there is no masked-view variant in this game
(the masked variant was dropped during brainstorming because _comm's
S2/S3/S4 levels build cross-agent dependency chains that become unreachable
when half the chain is static).

Per-instance fields:
- objects, start_positions, target_positions, walls (from generate_board)
- optimal_moves: per-object BFS sum (lower bound; matches _comm's
  optimal_moves formula)
- max_turns = max(4 * optimal_moves, 20)
- thinking, spatial_level, compact_board (from config.json)
- player_prompt: the loaded template string for the (thinking) variant
- device: present for S3/S4 boards (corridor/dead-end geometry)
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

from matrixgame_covered_shared.boardgen import generate_board, instance_rng


class SoloMatrixGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        common_config = self.load_json("resources/common_config")
        configs = self.load_json("resources/config")

        grid_size = common_config["grid_size"]
        max_retries = common_config["max_retries"]
        strict = common_config.get("strict", False)
        all_objects = common_config["objects"]

        num_instances = kwargs.get("num_instances", 8)
        compact_only = kwargs.get("compact_only", False)

        if compact_only:
            configs = {
                k: v for k, v in configs.items() if v.get("compact_board", False)
            }
            if not configs:
                raise RuntimeError(
                    "--compact-only selected but no compact variants in config.json"
                )

        for variant_key, variant_config in configs.items():
            thinking = variant_config["thinking"]
            spatial_level = variant_config["spatial_level"]
            compact_board = variant_config.get("compact_board", False)
            num_objects = variant_config["num_objects"]

            objects = all_objects[:num_objects]
            template_name = f"player_prompt_{'thinking' if thinking else 'silent'}"
            prompt_template = self.load_template(
                f"resources/initial_prompts/en/{template_name}"
            )

            experiment = self.add_experiment(variant_config["name"])
            experiment["common_config"] = common_config
            experiment["variant_config"] = variant_config

            for idx in range(num_instances):
                # Same (seed, level, idx) => identical board in every variant,
                # package, protocol, and team size.
                rng = instance_rng(seed, spatial_level, idx)
                board = generate_board(spatial_level, grid_size, objects, rng)

                game_instance = self.add_game_instance(experiment, idx + 1)
                game_instance["grid_size"] = grid_size
                game_instance["walls"] = board["walls"]
                game_instance["objects"] = objects
                game_instance["start_positions"] = board["start_positions"]
                game_instance["target_positions"] = board["target_positions"]
                game_instance["optimal_moves"] = board["optimal_moves"]
                game_instance["max_turns"] = max(4 * board["optimal_moves"], 20)
                game_instance["max_retries"] = max_retries
                game_instance["strict"] = strict
                game_instance["thinking"] = thinking
                game_instance["spatial_level"] = spatial_level
                game_instance["compact_board"] = compact_board
                game_instance["player_prompt"] = prompt_template
                if board.get("device"):
                    game_instance["device"] = board["device"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate solo instances.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-instances", type=int, default=8)
    parser.add_argument("--out", default="instances.json")
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="Only generate variants with compact_board=true in config.json",
    )
    args = parser.parse_args()

    gen = SoloMatrixGameInstanceGenerator()
    gen.generate(
        filename=args.out,
        seed=args.seed,
        num_instances=args.num_instances,
        compact_only=args.compact_only,
    )
    print(f"Wrote instances to {args.out}")
