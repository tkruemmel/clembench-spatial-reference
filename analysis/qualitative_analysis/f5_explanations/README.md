# F5 explanation analyses (2026-10-03/04)

These scripts produced the numbers in `thesis/f5_explanations.md`. They were copied
here from a session scratchpad. The `.pkl` caches were left out (about 50 MB);
the scripts rebuild them on first run. All three folders build on
`../seal_averted.py` and `../averted.py` and read `../episodes.json`.

Run each script from inside its own folder: the cut scripts read and write
their caches relative to the working directory.

## cut/ — how prose teams avoid the cut fork (§2.1 of the notes, C1/C3)

`order.py`, `doors.py` and `msgs.py` are libraries. The `runN.py` scripts are
steps, and five of them write caches that later steps read:

| step | writes | read by |
|---|---|---|
| `run1.py` | `s3rows.pkl` (S3 door relations, order outcome) | run2, run3, run7, run8, run12 |
| `run4.py` | `waitS3.pkl` (holder waiting decisions) | run10, run11, run12 |
| `run5.py` | `dyn.pkl` (dynamic doors, all levels) | run6, run11 |
| `run7.py` | `s3tagged.pkl` (message tags) | run9 |
| `run8.py` | `s3ord.pkl` (order statements in context) | run9, run11, run13 |

Run them in the order run1, run4, run5, run7, run8, then the rest. The combined
output is in `results.txt`.

## enclosure/ — sensitivity to the enclosure hazard (E2 and its retraction, §2.1–2.3)

- `plies.py` replays every 6p episode and builds `plies.pkl` (16 MB) on first
  import. It checks its pinch flags against `seal_averted.scan`.
- `analysis.py`, `robust.py`, `decomp.py`, `inway.py`, `q4.py`, `placebo.py`:
  the first round (baselines, conditional logit M0/M1, graded sensitivity,
  plan-following). Outputs are in `out.txt`, `robust_out.txt`,
  `decomp_out.txt`, `decomp_full.txt` and `placebo_out.txt`.
  - The done-X placebo in `placebo.py` is uninformative in the `none` arm. The
    notes (§2.1) explain why.
- `followup.py` … `followup4.py`, `dsplit.py`: the turn-order tests.
  - `followup3.py` is the matched-look contrast (fatal vs freed in time).
  - `dsplit.py` is the distance-by-relievability split.
  - Outputs are in `followup_out.txt` and `followup3_out.txt`.
  - `followup4.py` and `dsplit.py` print to stdout.

## verbal/ — what agents say about teammates (E3, A2)

`verbal.py` holds the lexicon and the replay, and builds `cache.pkl` (30 MB) on
first run. `t1.py` gives the rates by reference type; `t2.py` and `t2b.py` give
the board-conditioned mentions and seals; `t3.py` and `t3b.py` give the waiting
language. Outputs are in `t1_out.txt` and `t2b_out.txt`.
