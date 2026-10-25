# Matrixgame: cooperative object rearrangement on a grid

Six objects, A to F, stand on an 8×8 grid with walls. Every object has a target
cell, and the game is won when every object stands on its target. On each turn
one player moves one of its own objects one cell up, down, left or right.
Objects cannot enter walls, leave the grid or share a cell.

## Games

| game | players | objects per player |
|---|---|---|
| `matrixgame_solo` | 1 | 6 |
| `matrixgame_dyad` | 2 | 3 |
| `matrixgame_3p` | 3 | 2 |
| `matrixgame_6p` | 6 | 1 |

In solo the player sees the whole board and the whole goal board. In the team
games a player sees its own objects by letter and its teammates' objects as `X`,
and its goal board shows only its own targets. Players take turns in a fixed
order (Player Green, Orange, Blue, Yellow, Red, Purple). A player may declare
`done` once all its own objects are on their targets; it then takes no more
turns and its objects stay where they are.

The team games are played under four protocols:

| protocol | each turn carries |
|---|---|
| `none` | the move only |
| `structured` | `PLAN: move X from (Rr,Cc) to (Rr,Cc) — REQUEST: ...`, each part at most 10 words |
| `hybrid` | free text (at most 100 words) followed by the structured clause |
| `freeform` | free text; the prompt asks for at most 100 words |

The structured and hybrid formats are enforced like the move format.

## Levels

Every game plays the same 15 boards at each of four levels (60 in all):

- **S1** free placement, no walls, no object starts on another's target.
- **S2** walls that lengthen the paths, and the chain A←B←C: B starts on A's
  target and C on B's, so C must move before B can land, and B before A.
- **S3** a dead-end corridor of four cells. C starts on its entrance, D's target
  is its dead end, so the chain A←B←C←D has to be resolved through the corridor.
  The pair E←F lies outside.
- **S4** a dead-end corridor whose two inner objects must swap. A starts next to
  the dead end and targets it, B starts on the dead end, so A has to leave the
  corridor and come back. They are the head of the chain A←B←C←D; E←F lies in
  open space.

## Turns and the end of an episode

The player keeps the rules as a fixed prompt and gets a fresh context every
turn. It shows a MESSAGE BOARD (under a messaging protocol) and a MOVE BOARD
over the last 4 rounds, then the current board and the goal board. Messages keep
their sender. Moves name no player; a teammate's move reads
`X moved from (R1,C4) to (R2,C4)`. A `done` appears only on the message board,
as `Player Orange declared done`.

The episode is

- won when every object is on its target,
- lost when a player declares `done` with an object off target, when the board
  repeats a configuration for the third time, or when the moves reach four
  times the optimal move count (at least 20),
- aborted when a player answers out of format, or with an illegal move, more
  than twice in a row.

## Scores

The Quality Score of a won episode is `100 × min(1, A / moves)`, where A is the
optimal move count of the board under the game's rules; a lost episode scores 0
and an aborted one has none. Beside it each episode logs move efficiency, the
illegal and blocked move rates, per-object path ratios, oscillation, a deadlock
flag and message counts and lengths (`master.py`, `utils/metrics.py`).

## Instances

```
python3 matrixgame/instancegenerator.py      # in/instances_{solo,dyad,3p,6p}.json
```

`utils/boardgen.py` draws the boards from seed 42; a board depends only on its
level and index, so all four games play the same boards. Each candidate is
checked for reachability and, from S2 on, for a free space that stays connected
when any one cell is blocked (at S3 and S4, the space outside the corridor). The dyad takes the split the board generator returns, 3p and 6p
deal the objects round-robin.

The optimal move count A comes from an A* search under the real turn order,
ownership and `done` rules, in which no player is ever left without a legal move
(`utils/solver.py`). It differs between the games, since teammates have to move
out of each other's way. The results are stored in `resources/optimal_moves.json`:

```
cc -O3 -o matrixgame/utils/solver_c matrixgame/utils/solver.c
python3 matrixgame/utils/solver.py --table           # recompute resources/optimal_moves.json
python3 matrixgame/utils/solver.py 6p S3 10 --path   # one board, with a replay-checked path
```

## Tests

```
python3 -m pytest matrixgame/test_matrixgame.py
```

checks that the generator rebuilds `in/`, and replays every episode in
`../results` through the master with the responses the players gave, comparing
every prompt, logged event and score with the record.
