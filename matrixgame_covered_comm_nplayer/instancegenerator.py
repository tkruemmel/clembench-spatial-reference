"""Instance generator for matrixgame_covered_comm_nplayer.

Boards come from matrixgame_covered_shared.boardgen (single source of truth
for S1-S4 board generation, shared across _solo, _comm, and _comm_nplayer so
the physical puzzle distribution is identical by construction). The dyad
split returned by the shared generator is discarded here — ownership is a
round-robin partition over the requested number of players.

Two modes:
  3p — 3 players, each owning 2 of 6 objects.
  6p — 6 players, each owning 1 of 6 objects.

Ownership: `round_robin_partition` assigns objects[j] -> player j % num_players
(interleaving, not contiguous chunks) so that consecutive dependency-chain
members and the S4 swap pair land on different agents at 3 players.

The compact board representation is the only supported rendering mode.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

from matrixgame_covered_shared.boardgen import generate_board, instance_rng


def round_robin_partition(objects, num_players):
    """objects[j] -> player j % num_players.

    Interleaving (not contiguous chunks) keeps consecutive dependency-chain
    members and the S4 swap pair on different agents at 3 players.
    """
    return [
        [objects[j] for j in range(len(objects)) if j % num_players == i]
        for i in range(num_players)
    ]


class MatrixGameNPlayerInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        all_variant_configs = self.load_json("resources/config")
        common_config = self.load_json("resources/common_config")

        grid_size = common_config["grid_size"]
        max_retries = common_config["max_retries"]
        strict = common_config.get("strict", False)
        all_objects = common_config["objects"]

        num_instances = kwargs.get("num_instances", 8)

        for variant_key, variant_config in all_variant_configs.items():
            num_players = variant_config["num_players"]
            spatial_level = variant_config["spatial_level"]
            comm_protocol = variant_config["comm_protocol"]
            use_masking = variant_config.get("use_masking", True)
            compact_board = variant_config.get("compact_board", True)

            objects = all_objects[:]
            assert len(objects) % num_players == 0, (
                f"{variant_key}: {len(objects)} objects not divisible by {num_players} players"
            )
            player_objects = round_robin_partition(objects, num_players)

            prompt_template = self.load_template(
                f"resources/initial_prompts/en/player_prompt_{comm_protocol}"
            )

            experiment_name = f"{spatial_level}_{comm_protocol}_{num_players}p"
            experiment = self.add_experiment(experiment_name)
            experiment["common_config"] = common_config
            experiment["variant_config"] = variant_config

            for idx in range(num_instances):
                rng = instance_rng(seed, spatial_level, idx)
                board = generate_board(spatial_level, grid_size, objects, rng)

                game_instance = self.add_game_instance(experiment, idx + 1)
                game_instance["grid_size"] = grid_size
                game_instance["walls"] = board["walls"]
                game_instance["objects"] = objects
                game_instance["num_players"] = num_players
                game_instance["player_objects"] = player_objects
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
# Usage:
#   python instancegenerator.py                          # generate everything

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=8,
        help="Number of instances per condition (default: 8)",
    )
    args = parser.parse_args()

    gen = MatrixGameNPlayerInstanceGenerator()
    gen.generate(filename="instances.json", seed=42, num_instances=args.n)
