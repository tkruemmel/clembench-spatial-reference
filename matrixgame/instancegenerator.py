"""Instances of the four Matrixgame games: in/instances_{solo,dyad,3p,6p}.json.

Every game plays the same 15 boards per level (utils/boardgen.py, seed 42).
Solo is played without messages; the team games once under each protocol.
Ownership: the dyad takes the board's own split, 3p and 6p deal the objects
round-robin (objects[j] -> player j % n), which keeps consecutive chain
members on different players.

The turn budget is four times the optimal move count of the board under the
game's rules, max(4 * A, 20). A is the attainable optimum an A* search over
the real turn order, ownership and `done` rules finds (utils/solver.py), read
from resources/optimal_moves.json (`python3 matrixgame/utils/solver.py --table`
recomputes it).

    python3 matrixgame/instancegenerator.py
"""

import os

from clemcore.clemgame import GameInstanceGenerator

from utils.boardgen import LEVELS, generate_board

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
BOARDS_PER_LEVEL = 15
GRID_SIZE = 8
OBJECTS = ["A", "B", "C", "D", "E", "F"]
MAX_RETRIES = 2
GAMES = {"solo": 1, "dyad": 2, "3p": 3, "6p": 6}
PROTOCOLS = ("none", "structured", "hybrid", "freeform")
# S2 board 10 comes from stream index 16: at index 9 the 6p player owning B
# has no legal move on its first turn, so no 6p play-through exists in which
# no player is enclosed (utils/solver.py finds none).
REDRAWN = {("S2", 9): 16}


def round_robin(objects, num_players):
    return [[o for j, o in enumerate(objects) if j % num_players == i] for i in range(num_players)]


class MatrixGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(HERE)
        self.optimal_moves = self.load_json("resources/optimal_moves")
        self.boards = {
            (level, idx): generate_board(level, GRID_SIZE, OBJECTS, SEED, REDRAWN.get((level, idx), idx))
            for level in LEVELS for idx in range(BOARDS_PER_LEVEL)
        }

    def on_generate(self, seed: int, game: str):
        self.instances = dict(experiments=list())
        num_players = GAMES[game]
        protocols = [None] if game == "solo" else PROTOCOLS
        prompts = "solo" if game == "solo" else "dyad" if game == "dyad" else "nplayer"
        for level in LEVELS:
            for protocol in protocols:
                name = f"resources/initial_prompts/en/{prompts}/player_prompt"
                prompt = self.load_template(name if protocol is None else f"{name}_{protocol}")
                experiment = self.add_experiment(level if protocol is None else f"{level}_{protocol}")
                for idx in range(BOARDS_PER_LEVEL):
                    board = self.boards[(level, idx)]
                    optimal = self.optimal_moves[game][level][idx]
                    instance = self.add_game_instance(experiment, idx + 1)
                    instance["grid_size"] = GRID_SIZE
                    instance["walls"] = board["walls"]
                    instance["objects"] = OBJECTS
                    if num_players == 2:
                        instance["player_objects"] = list(board["dyad_split"])
                    elif num_players > 2:
                        instance["player_objects"] = round_robin(OBJECTS, num_players)
                    instance["start_positions"] = board["start_positions"]
                    instance["target_positions"] = board["target_positions"]
                    instance["optimal_moves"] = optimal
                    instance["max_turns"] = max(4 * optimal, 20)
                    instance["max_retries"] = MAX_RETRIES
                    instance["spatial_level"] = level
                    if protocol is not None:
                        instance["comm_protocol"] = protocol
                    instance["player_prompt"] = prompt
                    if board["device"]:
                        instance["device"] = board["device"]


if __name__ == "__main__":
    generator = MatrixGameInstanceGenerator()
    for game in GAMES:
        print(generator.generate(filename=f"instances_{game}.json", seed=SEED, game=game))
