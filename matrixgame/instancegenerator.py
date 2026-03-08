"""Instance generator for matrixgame.

Generates game instances with randomised starting positions (row 1) and
target patterns (row 4) across difficulty levels (fewer vs more objects /
columns).
"""

import os
import random
from typing import List, Optional

from clemcore.clemgame import GameInstanceGenerator


class MatrixGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        variant = kwargs.get("variant", "matrixgame")
        variant_config = self.load_json("resources/config")[variant]
        common_config = self.load_json("resources/common_config")

        num_columns = variant_config["num_columns"]
        num_objects = variant_config["num_objects"]
        num_rows = common_config["num_rows"]
        target_row = common_config["target_row"]
        goal_row = common_config["goal_row"]
        max_retries = common_config["max_retries"]
        strict = common_config.get("strict", False)
        start_row = common_config["start_row"]
        all_objects = common_config["objects"][:num_objects]

        prompt_template = self.load_template("resources/initial_prompts/en/player_prompt")

        num_instances = kwargs.get("num_instances", 10)

        experiment_name = f"{variant_config['name']}_{num_columns}cols_{num_objects}obj"
        experiment = self.add_experiment(experiment_name)
        experiment["common_config"] = common_config
        experiment["variant_config"] = variant_config

        rng = random.Random(seed)

        for idx in range(num_instances):
            game_instance = self.add_game_instance(experiment, idx + 1)

            # Random starting positions in row 0
            start_columns = rng.sample(range(num_columns), num_objects)
            start_positions = {}
            for obj, col in zip(all_objects, start_columns):
                start_positions[obj] = [0, col]  # row 0

            # Random target pattern in the target row (different ordering)
            target_columns = rng.sample(range(num_columns), num_objects)
            target_pattern: List[Optional[str]] = [None] * num_columns
            for obj, col in zip(all_objects, target_columns):
                target_pattern[col] = obj

            game_instance["num_columns"] = num_columns
            game_instance["num_rows"] = num_rows
            game_instance["target_row"] = target_row
            game_instance["goal_row"] = goal_row
            game_instance["max_retries"] = max_retries
            game_instance["strict"] = strict

            # Compute max_turns as 4x minimum total Manhattan distance
            min_total_distance = 0
            for obj, start_col in zip(all_objects, start_columns):
                target_col = target_columns[all_objects.index(obj)]
                row_dist = abs(goal_row - start_row)
                col_dist = abs(target_col - start_col)
                min_total_distance += row_dist + col_dist
            game_instance["min_manhattan_distance"] = min_total_distance
            game_instance["max_turns"] = 4 * min_total_distance

            game_instance["objects"] = all_objects
            game_instance["start_positions"] = start_positions
            game_instance["target_pattern"] = target_pattern
            game_instance["player_prompt"] = prompt_template


if __name__ == "__main__":
    MatrixGameInstanceGenerator().generate(
        filename="instances.json", seed=42, variant="matrixgame", num_instances=10
    )
