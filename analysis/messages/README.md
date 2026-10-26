# analysis/messages: the message analysis

The analysis behind the thesis's Communication section and Appendix B, run on
the DeepSeek-V4.1-Flash episodes in `results/`. The § numbers in the code refer
to the analysis design notes written alongside the thesis.

```
python3 -m analysis.messages.run test --models dsv41 --out test_dsv41   # every output into out/test_dsv41/
python3 analysis/thesis/probe_targets.py                                 # the listener probe's target split
python3 -m pytest analysis/messages/tests -q
```

`out/` is rebuilt by these commands. The language-model answers they score are
cached: the E5 extraction replies in `extract/cache/`, the listener probe's in
`probe/cache/`, so a rerun calls no API.

## E5 and the author check (spec §9.6, extension E5)

```
python3 -m analysis.messages.extract.run test --dry-run --model gpt-5.6-luna      # what is not cached yet, no calls
python3 -m analysis.messages.coding.sampler --pools                               # sample sizes, writes nothing
open analysis/messages/coding/sample/tool.html                                    # the labelling tool
python3 -m analysis.messages.coding.importer analysis/messages/coding/sample/labels_5f798ba8.json
python3 -m analysis.messages.coding.metrics                                       # the §9.6 decision
python3 -m analysis.messages.coding.dest analysis/messages/coding/dest/labels_0e97b9ba.json   # destination check
```

`coding/sample/` holds the author check of the teammate-position records and
`coding/dest/` the check of announced destinations, each with its items, the
exported labels and the scored metrics. `key.json` in each holds tier,
extractor and truth; it is not opened before labelling.

## Scopes (spec §2)

| scope | what | use |
|---|---|---|
| `calib` | S1 episodes | grammar coverage and frame composition only, never estimates |
| `test` | S2–S4 episodes | the reported estimates |

Scope is decided from the path, so no S2–S4 file is opened outside `test`.

## Modules

| module | contents |
|---|---|
| `corpus.py` | parser, owner map, replay, visibility, exposures, distances, hazards, self-checks |
| `grammar.py` | G-* grammars, frames, audits, normalisation, the §4.1 word lists |
| `stats.py` | intervals, bootstrap, sign-flip, exact conditional OR, clogit, Holm, §8 cell and contrast helpers |
| `measures/` | §5, one file per subsection group |
| `failure.py` | §6 |
| `vocab.py` | §7 |
| `curve.py` | §9.3 |
| `cases.py` | §10 |
| `coding/` | §9.6 author check: sampler, HTML tool, import, metrics |
| `extract/` | E5: language-model extraction of teammate-position statements (extension §4) |
| `tables.py`, `run.py` | outputs into `out/<scope>/` |
| `data/` | VerbNet class files (commit in `grammar.VN_COMMIT`), `wordlists.json` |

## Data model (corpus.py)

`load(scope, models=MODELS, suites=..., protos=..., check=True)` returns a list of episode dicts:

- `ep["meta"]`: `id` (`arm|model|suite|level|proto|board`), `arm`, `model` (`dsv41`), `tier` (`strong`), `suite` (`dyad`/`3p`/`6p`), `n`, `level` (`S1`–`S4`), `board` (int game id), `proto` (`none`/`structured`/`hybrid`/`freeform`), `success`, `aborted`, `lose`, `path`.
- `ep["seats"]` (turn order), `ep["owner"]` (object -> seat), `ep["objs"]` (seat -> sorted objects), `ep["grid"]`, `ep["walls"]` (frozenset of (r, c)), `ep["start"]`, `ep["targets"]` (object -> (r, c)), `ep["device"]` (S3/S4: `entrance`, `corridor`, `dead_end`, S4 `inner_adjacent`), `ep["done_ply"]` (seat -> ply), `ep["final_pos"]`, `ep["calls"]` (every call with its logged context `ctx`).
- `ep["turns"]`: accepted turns in order. Each has `ply`, `seat`, `si` (seat index), `own_turn`, `obj`, `dir`, `src`, `dst`, `done`, `msg`, `board` (B_t⁻, object -> (r, c)), `after` (B_t⁺), `retry`, `attempts`, `call`, `is_msg` (a message unit), `slots` (structured/hybrid: `prose`, `plan`, `plan_obj`, `plan_from`, `plan_to`, `request`, `request_none`), `dup` (exact repeat within the seat), `vis` (plies of messages shown in the context of the accepting call).

Cells are zero-based `(r, c)` tuples everywhere; `corpus.cell(p)` prints `(Rr,Cc)`.

Helpers: `messages(eps)` -> [(ep, turn)]; `visible(ep, idx, retry)`; `move_board(ep, idx)` (final arm: `(ply, own, src, dst)` shown at idx); `done_visible(ep, idx)` (seats whose finish the context of idx signals); `exposures(ep)` -> {(message ply, listener seat): first decision ply showing it}; `dist_map(ep, target, blocked)` (BFS distance to target; `blocked` cells act as walls); `goalward(ep, obj, src, d)`; `legal(ep, board, objs)`; `step(board, obj, d)`; `hazards(ep)` (§6 decision records).

## Grammar API (grammar.py)

`analyse(text, arm, proto, slots, P=DEFAULT)` returns a dict:
- `loc`: G-loc candidates, `dict(region, clause, letter, cell, rule ('fwd'|'rev'), frames (set), text)`. A candidate is kept by `kept(frames, 'strict'|'lenient')`. The frame names are `grammar.FRAMES`.
- `frm`: G-from candidates (`L ... from (R,C)`) in every region, with frames {`past`, `future`, `negation`, `question`, `think`}; kept iff the frame set is empty. Prose G-from is `region == "prose"`; the PLAN slot also yields one, but PLAN-from proper is in `slots`.
- `moves`: letter–direction pairs, `dict(region, clause, letter, dir, frame)`, frame in {present, future, sequence, conditional, past, negated, question}.
- `cells`, `lines`, `seats`, `letters`, `dirs`: lists of `(region, clause, value, token index)`.
- `clause_info`: per clause `id, region, term, letters, cells, goal_noun, my_its, goal_noun_audit, text`.
- `think`, `tokens` (per region), `n_clauses`.

Regions are `prose`, `plan` and `request` (structured messages have no prose; freeform is all prose). Every measure is reported separately for slots and prose (spec §5).

`analyse_all(eps, P)` -> {(episode id, ply): analysis}, about 20 s on the development arm. `target_mentions(an, own_targets)`, `audits(text, an, P)`, `normalise(an, own, regions)`.

`DEFAULT` is the primary specification: `comma=False, gap=4, lookback=3, frames='strict', article=True, compass=False, move_scope='clause'`. The curve varies these (spec §9.3).

## Statistics (stats.py)

`cell_estimate(values, levels)` (spec §8: level-stratified bootstrap when pooled, Clopper–Pearson or t within one level), `paired_contrast(rows, value, key, a, b, unit)` (board-paired sign-flip), `signflip`, `strat_bootstrap`, `wilson`, `clopper_pearson`, `t_interval`, `exact_conditional_or`, `clogit`, `holm`, `perm_pvalue`, `permute_within`, `mde_or`.

## Conventions

- A module's entry point is `run(scope, eps, A)`, with `A = grammar.analyse_all(eps)`. It returns `{table name: [row dicts]}`, and `run.py` writes each table to `out/<scope>/<name>.csv`. Rows are flat and carry their grouping keys (`model`, `tier`, `suite`, `level`, `proto`, `board`, `episode`).
- Episode-level rows are the unit (spec §1.6). Message-level rates deduplicate exact repeats within a seat.
- `none` episodes enter only as baselines.
- Every p-value outside D2 is exploratory. Holm within a named family is a multiplicity guard only.
- No hand-picked word lists: words come only from `grammar.lists()`, the game's prompt and code, or the vocabulary ranking.
- Seeds are fixed (0 unless the spec says otherwise).
- No new dependencies (numpy, scipy, pandas, matplotlib, nltk are installed).
