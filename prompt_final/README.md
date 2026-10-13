# prompt_final

Edit surface for the final grid arm (`*_grid_final` packages). `dyad/` and
`nplayer/` are the `prompt_updated/` templates, `solo/` the
`matrixgame_covered_solo_grid` templates, each with one addition: a WHAT YOU
SEE EACH TURN section stating the record the master keeps (`ROUNDS` in `matrixgame_covered_shared/history.py`; change the
number in both places, the packages' tests check they agree). Solo plays only
the silent variant; the thinking experiments are dropped (`--only silent`
below). The cycle
rule stays unstated, as in the played arms: it checks whether play makes
progress and is not something to play to.

The instances are copies of already played files with only `player_prompt`
swapped, so boards and A_k budgets are unchanged; solo additionally takes its
budget from the attainable solo optimum, as the team arms do:

```
A=prompt_updated/apply.py
python3 $A --src matrixgame_covered_comm_grid_updated_prompt/in/instances.json \
    --prompts prompt_final/dyad --out matrixgame_covered_comm_grid_final/in/instances.json
for n in 3 6; do python3 $A \
    --src matrixgame_covered_comm_nplayer_grid_updated_prompt/in/instances_${n}p.json \
    --prompts prompt_final/nplayer --out matrixgame_covered_comm_nplayer_grid_final/in/instances_${n}p.json; done
python3 $A --src matrixgame_covered_solo_grid/in/instances.json --prompts prompt_final/solo \
    --budget analysis/efficiency_baseline/attainable_solo.jsonl --only silent \
    --out matrixgame_covered_solo_grid_final/in/instances.json
```

One change goes beyond the prompts. At S3 the dyad re-splits the objects so
that B (target on the corridor entrance) and D (target on the dead end) belong
to different players, as they do at 3p and 6p. The played split gave both to
one player, so a `done` could never seal the corridor in the dyad
(`thesis/design_review_2026-10-12.md`, K2). The new split is [D,A,E] / [C,B,F]:
only the link B<-C stays with one player. The new partition needs its own A_k:

```
F=matrixgame_covered_comm_grid_final/in/instances.json
python3 prompt_final/split_s3.py $F
python3 analysis/efficiency_baseline/stuckfree.py --suite dyad --c --all --instances $F \
    --save analysis/efficiency_baseline/attainable_dyad_final.jsonl
python3 prompt_final/split_s3.py $F --budget analysis/efficiency_baseline/attainable_dyad_final.jsonl
```

The final dyad arm's A_k therefore comes from `attainable_dyad_final.jsonl`, not
from `attainable_dyad.jsonl`. They differ on S3 boards 3, 6, 7, 13 and 14.

S4 plays the S4x boards (`matrixgame_covered_crossed_shared/boards.py`): same
corridor and swap, but the swap pair is the head link of the deep chain
A<-B<-C<-D and E<-F sits in open space. The level keeps the name S4. Every
rebuild of a final file with apply.py must be followed by

```
python3 prompt_final/s4x.py
```

which copies the S4x board, dyad split and budget from the crossed packages
into each S4 instance (solo budget from `joint_optimum_crossed.jsonl`). It
touches only S4, so it commutes with the split_s3.py steps; a second run is a
no-op. The packages' tests fail if it is forgotten.

Runner suites: `solo_grid_final`, `dyad_grid_final`, `3p_grid_final`, `6p_grid_final`.
