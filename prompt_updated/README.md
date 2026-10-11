# prompt_updated

Edit surface for the second grid prompt arm. `dyad/` and `nplayer/` hold copies of
the templates currently shipped by `matrixgame_covered_comm_grid` and
`matrixgame_covered_comm_nplayer_grid`. Edit them here; nothing reads them at run
time.

`apply.py` then writes a copy of an existing instances file with only
`player_prompt` swapped, so the new arm plays byte-identical boards. With
`--budget` it also replaces the generator's per-object BFS sum with the A*
attainable optimum A_k, in both `optimal_moves` (the scorer's efficiency
denominator) and `max_turns` (`max(4 * A_k, 20)`):

```
python3 prompt_updated/apply.py \
    --src matrixgame_covered_comm_grid/in/instances.json \
    --prompts prompt_updated/dyad \
    --budget analysis/efficiency_baseline/attainable_dyad.jsonl \
    --out  matrixgame_covered_comm_grid_<tag>/in/instances.json
```

It refuses to write if a `$PLACEHOLDER$` was renamed or dropped, if anything but
the prompt and budget changed, if a jsonl row is not `exact`, or if a new budget
would come out tighter than the stored one (which would mean the optimum was
computed on a different board). Keep `dyad/` and `nplayer/` identical modulo
plurals and the `$YOUR_ROLE$` / `$N_OTHER_PLAYERS$` lines.
