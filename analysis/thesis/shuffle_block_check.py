"""Brute-force check of the structural claims about one viewer block (exclusive occupancy,
one move per teammate, each teammate moves one of its own objects one cell).
Claims: (1) to-cells in a block are distinct; (2) the unordered arrow set plus the
post-block board reconstructs the pre-block board uniquely; (3) to(b)==from(a) implies
a moved before b, so chained precedence survives shuffling; (4) chained pairs fully
attribute both moves in a 2-teammate block (3p) once the cyclic order is known;
(5) in 3p, two objects of one owner never move in the same block."""
import random

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def run(seed, n=5, owners=((0, 1), (2, 3)), blocks=200):
    rng = random.Random(seed)
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], sum(map(len, owners)))
    pos = dict(enumerate(cells))
    chained = attributed = 0
    for _ in range(blocks):
        before, arrows = dict(pos), []          # arrows in true temporal order: (owner, obj, src, dst)
        for k, objs in enumerate(owners):
            occ = set(pos.values())
            opts = [(o, (pos[o][0] + dr, pos[o][1] + dc)) for o in objs for dr, dc in DIRS]
            opts = [(o, d) for o, d in opts if 0 <= d[0] < n and 0 <= d[1] < n and d not in occ]
            if not opts:
                continue                       # stuck: no legal move (the game would need 'done' or abort)
            o, d = rng.choice(opts)
            arrows.append((k, o, pos[o], d)); pos[o] = d
        tos = [a[3] for a in arrows]
        assert len(set(tos)) == len(tos)                                   # (1)
        shown = {(a[2], a[3]) for a in arrows}                             # what a shuffled display reveals
        cell_to_obj = {c: o for o, c in pos.items()}
        rebuilt = dict(pos)
        for src, dst in shown:
            rebuilt[cell_to_obj[dst]] = src
        assert rebuilt == before                                           # (2)
        for i, a in enumerate(arrows):
            for j, b in enumerate(arrows):
                if b[3] == a[2]:
                    assert i < j                                          # (3)
                    chained += 1
                    if len(arrows) == 2:                                   # (4) order known -> owners known
                        assert (a[0], b[0]) == (0, 1); attributed += 1
        moved = [a[1] for a in arrows]
        for objs in owners:
            assert sum(o in objs for o in moved) <= 1                      # (5)
    return chained, attributed

if __name__ == "__main__":
    tot = [0, 0]
    for s in range(300):
        c, a = run(s)
        tot[0] += c; tot[1] += a
    print("all claims hold; chained pairs seen:", tot[0], "fully attributed 2-move blocks:", tot[1])
