# F6 tier audit (2026-10-03/04)

These scripts check the cross-model material behind finding F6 (§8.5, §9.6.4,
§10.1–10.3, §10.5). They are read-only on the episode trees. Run each one from
the repo root as `python3 <path>`. They read the three grid campaigns through
`analysis/thesis/tables_comm.py`, `qual_corpus.py`, `aborts.json` and
`episodes.json`.

The items they feed are C1–C5 in `thesis/todo.md`. Outputs are printed, not
cached; the one exception is `claims/claims.py`, which writes `claims.json`
(~10 MB) next to itself.

| Script | What it reproduces | todo |
|---|---|---|
| `claims/claimfix.py` | Own- and teammate-claim accuracy per model and prose protocol, first with the shipped `CLAIM` regex, then with the object letter made case-sensitive (the article "a" was being read as object A). Nemotron free-form goes 29.9% → 37.3%; its hybrid figure falls to 3 claims. | C1 |
| `claims/claims.py` | Logs every claim: letter case, matched span, move index, suite. Used for the per-suite split, the destination filter and the before-move-10 check. | C1 |
| `claims/echo.py` | For each teammate claim, whether the same (letter, cell) was stated earlier and by whom. Gives the stale-cell figure: about 78% of the weak model's wrong teammate claims name a cell the object used to occupy. | C1 |
| `rejection/retry.py` | Every rejected move in 2,520 episodes, classified by what the next attempt does. Strong tier, legal alternative open: 101/102 revised and accepted, 0 repeats. Nemotron: 1,027/2,566 identical repeats. Also the first retry (the response right after a player's first rejection): Nemotron 571/1,646. | C2 |
| `rejection/retry_suite.py` | The same, split by suite. Nemotron repeat rate 21% solo, 37% dyad, 42% 3p, 53% 6p; strong tier 0% in every suite. | C2 |
| `rejection/rep4.py` | Aborts closed by a back-to-back identical repeat, under four definitions: 545–577 of Nemotron's 749. Also the abort composition by final-run verdicts. | C2 |
| `format/fmt.py` | Message length by protocol and verdict. Accepted Nemotron hybrid messages average about 65 words, not 557. | C3 |
| `format/mal.py` | Why hybrid messages were rejected as malformed: 230 of 284 break the 10-word REQUEST cap. Uses `_schema_violation` from `matrixgame_covered_comm/master.py`. | C3 |
| `tiers/f6.py` | Success concordance between the strong models (732/47/28/33). Splits DeepSeek's Quality lead into efficiency (+1.45) and discordant successes (+0.79). Per-board success correlation r = 0.87. | C3 |
| `tiers/aborts.py` | Accepted moves before each abort: Nemotron median 2, and 206 of 749 aborts never got a move accepted. | C4 |
| `tiers/weak.py` | The weak-tier numbers no other script prints: successes by game and level, protocol McNemars, the strict-subset expectation, team failures on solo-solved boards, the silent baseline, how episodes end, illegal/blocked rates, claim staleness, format aborts by level, the S4 #8/#13 traps, enclosure aborts by model. Needs `claims/claims.py` first. | C1–C4 |
| `tiers/cells.py` | Per-cell success for all three models: the weak model matches neither strong model in any of 56 cells. | C4 |

One number was taken from the audit transcript and has no script here: the
Nemotron McNemar contrasts between protocols in C3. Recompute it before quoting
it.

## Why Nemotron breaks the rules (2026-10-04)

Two follow-up studies. Run them from the repo root. Each one writes its cache
next to itself, and the caches are not copied here.

| Script | What it shows |
|---|---|
| `nemotron_legality/attempts.py` | Run this first. Replays every move attempt in the 2,520 grid episodes into `attempts.json` (~96 MB) and matches the game master's verdict on all of them. |
| `nemotron_legality/greedy.py` | Rejection causes by suite: wall, off-grid, own object in the way, teammate's object in the way. How often the moved object goes toward its target. Repeats after a rejection, and whether a legal toward-target move existed. |
| `nemotron_legality/blind.py` | The cleanest legality test. Take a fresh choice where an object has two directions toward its target and exactly one is blocked; a model ignoring obstacles picks the blocked one 50% of the time. Nemotron does so 39.2% of the time (520/1,325); Qwen3.8 0.1% and DeepSeek 0.1%. |
| `nemotron_legality/knows.py` | Whether the response text mentions the obstacle or the rejection, for repeated moves against changed moves. |
| `nemotron_legality/claims.py` | Whether the stated position and destination match the board and the direction. No up/down or row/column confusion was found. |
| `nemotron_legality/s1solo.py` | Why Nemotron goes 0/15 on solo S1 silent: wandering and undoing its own moves. |
| `nemotron_reasoning/extract.py` | Run this first. Joins the raw Bedrock responses (`player_N.requests.json`) to the game events and writes `calls.jsonl`. It records whether an inline `</think>` trace was present and the output length. |
| `nemotron_reasoning/tokens.py` | Output tokens per call, all three models, by game: median and mean, and the strong/weak ratios under both. |
| `nemotron_reasoning/analyze.py [1-7]` | Legality of turns with a trace against turns without, both in the same episode and in the same turn (draft against final answer). Legality by output length. Solo silent against thinking. Compact against grid. |

All scripts read the team suites of `ARM` (`updated` by default, `original`
for regression; solo is always the original campaign), and every cache they
write carries the arm in its name (`attempts.{ARM}.json`, `claims.{ARM}.json`,
`calls.{ARM}.jsonl`). The compact roots, and so section 5 of `analyze.py`,
exist only for `ARM=original`. The numbers quoted above are original-arm values.
