"""Checks that this package reproduces the episodes in results/.

The instance generator must rebuild in/ exactly, and every recorded episode,
replayed through master.py with the responses the players gave, must yield
the same prompts, the same logged events and outcome keys, and the same
scores. Instances come from in/, so this also checks the generator against
the boards that were played.

    python3 -m pytest matrixgame/test_matrixgame.py        # from the repository root
"""

import glob
import json
import math
import os
import sys
import unittest

from clemcore.backends import Model, ModelSpec
from clemcore.backends.utils import ensure_alternating_roles
from clemcore.clemgame import GameBenchmark
from clemcore.clemgame.recorder import GameInteractionsRecorder
from clemcore.clemgame.registry import GameSpec

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("MATRIXGAME_RESULTS", os.path.join(os.path.dirname(HERE), "results"))
SPECS = {s["game_name"]: s for s in json.load(open(os.path.join(HERE, "clemgame.json")))}

sys.path.insert(0, HERE)
from utils.history import ROUNDS  # noqa: E402


class ReplayModel(Model):
    """Returns one player's recorded responses in order and checks each prompt
    as the backend sent it (consecutive user messages joined)."""

    def __init__(self, prompts, responses):
        super().__init__(ModelSpec(model_name="replay"))
        self.prompts, self.responses = list(prompts), list(responses)

    def generate_response(self, messages):
        prompt, expected = ensure_alternating_roles(messages), self.prompts.pop(0)
        if prompt != expected:
            raise AssertionError(f"prompt differs:\n{_first_diff(prompt[-1]['content'], expected[-1]['content'])}")
        return prompt, {}, self.responses.pop(0)


def _first_diff(a, b):
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"replayed: ...{a[max(0, i - 80):i + 80]!r}\nrecorded: ...{b[max(0, i - 80):i + 80]!r}"


def _events(turns):
    return [[{k: v for k, v in e.items() if k != "timestamp"} for e in t] for t in turns]


def _same(x, y):
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    return x == y


def instances(game):
    data = json.load(open(os.path.join(HERE, "in", f"instances_{game}.json")))
    return {(e["name"], i["game_id"]): (e, i) for e in data["experiments"] for i in e["game_instances"]}


def episodes(game):
    pattern = os.path.join(RESULTS, "*", f"matrixgame_{game}", "*", "instance_*", "interactions.json")
    return sorted(glob.glob(pattern))


def replay(benchmark, game, path, insts):
    """Replay one recorded episode; returns the replayed interactions."""
    episode_dir = os.path.dirname(path)
    recorded = json.load(open(path))
    experiment_name = os.path.basename(os.path.dirname(episode_dir))
    game_id = int(os.path.basename(episode_dir).split("_")[1])
    experiment, instance = insts[(experiment_name, game_id)]
    models = []
    for n in range(1, SPECS[f"matrixgame_{game}"]["players"] + 1):
        calls = json.load(open(os.path.join(episode_dir, f"player_{n}.requests.json")))["calls"]
        said = [e["action"]["content"] for t in recorded["turns"] for e in t
                if e["from"] == f"Player {n}" and e["action"]["type"] == "get message"]
        models.append(ReplayModel([c["call"]["manipulated_prompt_obj"] for c in calls], said))
    master = benchmark.create_game_master(experiment, models)
    master.setup(**instance)
    recorder = GameInteractionsRecorder(benchmark.game_name, experiment_name, game_id, "replay", None)
    for player in master.get_players():
        recorder.log_player(player.name, player.game_role, player.model.name)
    master.register(recorder)
    master.before_game()
    done = False
    while not done:
        player = master.current_player
        done, _ = master.step(player(master.get_context_for(player)))
    assert all(not m.prompts and not m.responses for m in models), "recorded calls left over"
    return recorded, recorder.interactions


class MatrixGameTest(unittest.TestCase):

    def test_generator_rebuilds_in(self):
        from instancegenerator import GAMES, SEED, MatrixGameInstanceGenerator
        generator = MatrixGameInstanceGenerator()
        for game in GAMES:
            with self.subTest(game=game):
                generator.on_generate(SEED, game=game)
                stored = json.load(open(os.path.join(HERE, "in", f"instances_{game}.json")))
                self.assertEqual(json.loads(json.dumps(generator.instances)), stored)

    def test_prompts_state_rounds(self):
        for path in glob.glob(os.path.join(HERE, "resources", "initial_prompts", "en", "*", "*.template")):
            with self.subTest(template=os.path.relpath(path, HERE)):
                text = open(path).read()
                self.assertIn(f"last {ROUNDS} rounds", text)

    def test_recorded_episodes_replay(self):
        for game_name, spec in SPECS.items():
            game = game_name.split("_", 1)[1]
            paths = episodes(game)
            if not paths:
                self.skipTest(f"no recorded episodes of {game_name} under {RESULTS}")
            insts = instances(game)
            with GameBenchmark.load_from_spec(GameSpec.from_dict({**spec, "game_path": HERE})) as benchmark:
                for path in paths:
                    with self.subTest(episode=os.path.relpath(path, RESULTS)):
                        recorded, replayed = replay(benchmark, game, path, insts)
                        self.assertEqual(_events(replayed["turns"]), _events(recorded["turns"]))
                        for key in set(recorded) - {"meta", "players", "player_models", "turns"}:
                            self.assertEqual(replayed.get(key), recorded[key], key)
                        episode_dir = os.path.dirname(path)
                        scorer = benchmark.create_game_scorer(
                            json.load(open(os.path.join(os.path.dirname(episode_dir), "experiment.json"))),
                            json.load(open(os.path.join(episode_dir, "instance.json"))))
                        scorer.compute_scores(recorded)
                        stored = json.load(open(os.path.join(episode_dir, "scores.json")))
                        self.assertTrue(_same(json.loads(json.dumps(scorer.scores)), stored), "scores differ")


if __name__ == "__main__":
    unittest.main()
