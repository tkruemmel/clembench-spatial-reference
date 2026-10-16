"""What would the final-arm record have shown, replayed on the played updated-prompt episodes?

For every turn, the acting player sees the last W = n_players*4 events. Per teammate object:
- attributed: it moved inside the window, so the record names its owner (at 6p, by the
  Player-letter == object-letter assignment, also its letter);
- toward: each recorded teammate move, does it reduce Manhattan distance to the hidden target?
  chance = share of the four directions that would.

Usage: python3 analysis/thesis/record_leak.py results_qwen38_grid_updated_prompt_n15_{dyad,3p,6p}
"""
import collections, glob, json, os, sys

D = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def owners(inst):
    if "player_objects" in inst:
        return {o: f"Player {chr(65 + i)}" for i, objs in enumerate(inst["player_objects"]) for o in objs}
    return {**{o: "Player A" for o in inst["player_a_objects"]}, **{o: "Player B" for o in inst["player_b_objects"]}}


def events(inter):
    out = []
    for t in inter["turns"]:
        for e in t:
            a = e["action"]
            if e["from"] == "GM" and e["to"] == "GM" and a["type"] in ("move", "player_done"):
                w = a["content"].split()
                out.append((f"{w[0]} {w[1]}", w[3] if a["type"] == "move" else None, w[4] if a["type"] == "move" else None))
    return out


def run(root):
    S = collections.defaultdict(lambda: collections.Counter())
    for f in glob.glob(f"{root}/**/interactions.json", recursive=True):
        inst = json.load(open(os.path.join(os.path.dirname(f), "instance.json")))
        own = owners(inst)
        n = len(set(own.values()))
        W = n * 4
        proto = inst["comm_protocol"]
        tgt = {o: tuple(p) for o, p in inst["target_positions"].items()}
        pos = {o: tuple(p) for o, p in inst["start_positions"].items()}
        ev = events(json.load(open(f)))
        hist = []  # (player, obj, src, dst)
        for player, obj, d in ev:
            win = hist[-W:]
            mates = [o for o in own if own[o] != player]
            live = [o for o in mates if pos[o] != tgt[o]]
            moved = {o for (_, o, _, _) in win if o}
            k = S[proto]
            k["views"] += 1
            k["mate_objs"] += len(mates)
            k["mate_attr"] += len(moved & set(mates))
            k["live_objs"] += len(live)
            k["live_attr"] += len(moved & set(live))
            if mates and set(mates) <= moved:
                k["views_all_attr"] += 1
            if obj:
                src = pos[obj]
                dst = (src[0] + D[d][0], src[1] + D[d][1])
                before = abs(tgt[obj][0] - src[0]) + abs(tgt[obj][1] - src[1])
                after = abs(tgt[obj][0] - dst[0]) + abs(tgt[obj][1] - dst[1])
                k["moves"] += 1
                k["toward"] += after < before
                k["chance"] += sum(
                    abs(tgt[obj][0] - src[0] - dr) + abs(tgt[obj][1] - src[1] - dc) < before for dr, dc in D.values()) / 4
                pos[obj] = dst
            hist.append((player, obj, src if obj else None, dst if obj else None))
    return S


if __name__ == "__main__":
    for root in sys.argv[1:]:
        S = run(root)
        tot = collections.Counter()
        for p in ("none", "structured", "hybrid", "freeform"):
            k = S[p]
            tot.update(k)
        for p, k in [*((p, S[p]) for p in ("none", "freeform")), ("ALL", tot)]:
            print(f"{os.path.basename(root):45s} {p:9s} views={k['views']:6d} "
                  f"attr_all={k['mate_attr']/k['mate_objs']:.2f} attr_live={k['live_attr']/max(k['live_objs'],1):.2f} "
                  f"all_mates_attr={k['views_all_attr']/k['views']:.2f} "
                  f"toward={k['toward']/k['moves']:.2f} (chance {k['chance']/k['moves']:.2f})")
