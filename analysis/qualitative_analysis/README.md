# Reproducing Section 5.1–5.6 of `results_three_models_grid.md` without an LLM

`qual_corpus.py` recomputes the qualitative-analysis section from the raw
`interactions.json` files of the three grid campaigns. No model is called at any
point: every number comes from regular-expression feature detection plus a
deterministic board replay that checks each positional claim against the board
state at the moment it was uttered.

```
python3 qual_corpus.py --refresh          # extract corpus (~20 s) and print all tables
python3 qual_corpus.py --verify           # also check every value the document asserts
```

The extraction caches to `corpus.json` (one record per broadcast message, 66 584
of them). `--refresh` re-reads the 2 160 episode directories.

## What reproduces

All 301 values the document asserts from this corpus are reproduced exactly: 192
in Sections 5.1 to 5.6, and 109 in Table 12 of Part II, the dyad-only
restriction of the 5.2 and 5.3 tables that answers RQ2's mechanism question.

| Part | Status |
|---|---|
| 5.1 corpus size, episode count, the two sanity checks, S2 #10 share | exact |
| 5.2 the nine-column protocol table, all 81 cells | exact |
| 5.2 `REQUEST: none` rates, and the temporary-shift counts | exact |
| 5.3 the grounding table, all 24 cells | exact |
| 5.3 question rates, and the `move: <X> stay` pseudo-action counts | exact |
| 5.4 the compliance table, requests per episode, the Mann-Whitney result | exact |
| 5.4 DeepSeek structured compliance by outcome, with the Mann-Whitney | exact |
| 5.5 the success-vs-failure table including all seven p-values (Fisher exact) | exact |
| 5.5 the naive pooled comparison and the within-structured comparison | exact |
| Part II Table 12: the dyad-only protocol table, all 78 cells | exact |
| Part II Table 12: dyad grounding claims and accuracies, and the dyad `temporarily` counts | exact |
| 5.6 every count in the S3 #14 worked episode | exact |

Every check is an exact string test, a count, or a board replay. There are no
approximate lexicons and no thresholds chosen after seeing the answer.

## The S2 #10 replacement, folded in on 21 Sep

The corpus was re-extracted after board S2 #10 was regenerated and its 42
episodes replayed. It now covers the same sixty boards as every other figure in
the document, so the scope caveat 5.1 used to carry — corpus on the full board
set, episode figures on the 826-episode analysis set — is retired. 109 of the
asserted values moved and were corrected in both the document and `DOC`; none
by enough to change a claim. The largest single movements are the corpus size
(66 007 to 66 584), the replaced board's own contribution (560 messages to
1 137, because its 6p episodes are now playable), and three p-values in 5.5
that move within the same order of magnitude or one below it. The pre-replacement
cache is kept as `corpus.826.json`.

## Three corrections made to the results document

- Section 5.5 gave the Fisher p for "names a teammate's object" as 1.2e-08. The
  exact value is 1.148e-08, which rounds to 1.1e-08. Corrected.
- Section 5.6 misquoted Player D, writing "if any done teammate **could**
  temporarily shift, I need one open." The message reads "**can** temporarily
  shift, I need one open **to reach R1C2**." Corrected, and the third line of that
  block reattributed from E to D, which is where the quoted wording comes from.
- Section 5.2 called the Qwen hybrid excerpt "a comparable board". It is the same
  board: start positions, target positions and walls are identical to the DeepSeek
  structured episode beside it. Corrected to "the same board".

## What no script can produce

The section's quoted transcript excerpts were selected for what they illustrate,
and the prose that interprets the tables — that a structured message is a state
announcement while a prose message is a joint plan over time, that the extra
requesting in failed episodes reads as distress signalling rather than
coordination failure — is argument, not measurement. A script can verify every
number those claims rest on; it cannot choose the examples or make the argument.

---

# Inventory: what in 5.1–5.6 is not script-reproducible

Two kinds of content. The transcript excerpts are verifiable as quotations (I
traced every one to its source episode below) but a script cannot *choose* them.
The interpretive claims rest on numbers the script confirms, but the reading
itself is argument.

## A. Quoted transcript excerpts, with provenance

All six blocks are verbatim as the document now stands. Provenance was recovered
by normalising quote marks and dashes and searching the corpus.

| Doc line | Excerpt | Source episode | Speakers / turn | Outcome |
|---|---|---|---|---|
| 235–238 | Four `PLAN:` messages, the dependency chain | `results_dsv41_grid_n15_6p/.../S3_structured_6p_grid/instance_00002` | A, B, C, D at turn 0 | cycle loss, 60 moves |
| 242–243 | Qwen hybrid pair, "time your pass" | `results_qwen38_grid_n15_6p/.../S3_hybrid_6p_grid/instance_00002` | B, D at turn 0 | success, 75 moves |
| 264 | `move: B stay` then "Wait — 'stay' is not a valid direction" | `results_nemotron_fixthink_grid_n15_6p/.../S1_hybrid_6p_grid/instance_00012` | B at turn 6 | abort, too many invalid moves |
| 303 | "I can wait in C7 if needed" | `results_qwen38_grid_n15_6p/.../S3_hybrid_6p_grid/instance_00002` | A at turn 0 | success, 75 moves |
| 303 | "I'll wait at R4C4 until R5C3 clears" | `results_dsv41_grid_n15_6p/.../S3_freeform_6p_grid/instance_00014` | E, message 97 | cycle loss, 97 moves |
| 311–313 | The three-message deadlock diagnosis | `results_dsv41_grid_n15_6p/.../S3_freeform_6p_grid/instance_00014` | E msg 99, D msg 100, D msg 98 | cycle loss, 97 moves |

Two claims about the excerpts check out and one is understated:

- "six consecutive messages" (line 233) is exact. Turn 0 of that episode carries
  six `PLAN:` messages, A through F; the document quotes the first four.
- "Hybrid (Qwen) on a **comparable** board" (line 240, now corrected) understated the design.
  The two episodes are the same board: `start_positions`, `target_positions` and
  `walls` are identical between DeepSeek S3 structured #2 and Qwen S3 hybrid #2.
  The comparison is exactly matched, not merely comparable.
- Only the S3 #14 free-form episode is a cycle loss among the prose examples; the
  Qwen hybrid excerpt comes from a *successful* episode, which the document does
  not say and which strengthens the contrast it draws.
- Both defects in that quoted block have been fixed in the document, as recorded
  above. "If B cannot move, we're deadlocked" is used by both D and E in the
  episode; the version ending in a full stop, as quoted, is D's.

## B. Interpretive claims

| Doc line | Claim | What the script can confirm |
|---|---|---|
| 229 | "A structured message is a state announcement; a prose message is a joint plan over time." | The nine feature rates the reading generalises from. |
| 240 | "The messages describe a dependency chain that only a serialisation can resolve, in a format with no way to express one." | That the four messages exist and each names an occupied target. Not that serialisation is the only resolution. |
| 245 | "The prose adds exactly three things the schema cannot express: subordination, temporal instruction, transient occupancy." | Nothing. The taxonomy is read off two messages. |
| 260 | Grounding is "the cleanest available operationalisation of the tier gap, and it is not a spatial-reasoning measure". | The accuracy table. Not that it isolates modelling others from spatial skill. |
| 262 | Nemotron "deliberates in the channel rather than communicating a result". | The question rate and the nineteen pseudo-action messages. Not the causal story about length. |
| 295 | Extra requesting in failures reads as "distress signals rather than coordination failures". | All seven rates and p-values. The document already flags this as correlational. |
| 303 | "The models plan with a primitive the rules do not provide." | The nineteen `move: <X> stay` messages and the temporary-shift counts. Not the inference about why. |
| 315 | "The agents identify the deadlock and its cause after the fact. The capability they lack is identifying it beforehand." | Every count in the S3 #14 episode, and that it ends in a cycle loss. Not the counterfactual. |

Calling an excerpt "representative" (line 233) is itself a judgement no script
makes.

---

# Appendix: verbatim evidence for each annotation feature

Every quote below is word for word from the corpus. Provenance reads
`model / suite / level / protocol / instance / turn / speaker`, which locates the
message uniquely under

```
results_<model>_grid_n15_<suite>/<model-dir>/matrixgame_covered_comm[_<suite>]_grid/<level>_<protocol>[_<suite>]_grid/instance_XXXXX/interactions.json
```

`turn` is the index of the turn block in `interactions.json`, not a per-message
counter. A single block can hold many messages, so the speaker disambiguates
within it. Quotes are excerpts from longer messages; a leading or trailing `…`
marks the cut.
The lexicons are deliberately shallow, so several features fire on text that does
not carry the intended meaning. Where that materially affects a reported number I
give the false positive too.

## `COORD` — coordinate mentions

Feeds the `coords` column of 5.2 and the coordinate-density row of 5.5.

> …I see that C needs to go to (R3,C4), B to (R7,C1), and A to (R8,C1).…
> — nemotron / dyad / S1 / freeform / instance_00001 / turn 0 / Player A

## `CLAIM` — positional claims, and the grounding table of 5.3

Matches `<OBJ> at (Rr,Cc)`. Each hit is checked against the board state at the
moment of utterance, and classed by whether the object belongs to the speaker.
All four verdicts below were computed by replay.

Own object, correct:

> …D will hold at (R3,C6) until E clears (R3,C8)/(R3,C7), then D takes (R3,C7)-(R3,C8).…
> — qwen38 / dyad / S1 / freeform / instance_00001 / turn 3 / Player B
> Claim: D at (R3,C6). Board: (R3,C6).

Own object, wrong:

> …I'll hold D at R3C7 if needed.…
> — qwen38 / dyad / S1 / freeform / instance_00001 / turn 0 / Player B
> Claim: D at (R3,C7). Board: (R4,C4).

Teammate's object, correct:

> …D at (R4,C6) doesn't block my C route.…
> — qwen38 / dyad / S1 / freeform / instance_00001 / turn 2 / Player A

Teammate's object, wrong:

> …Please let A hold at (R3,C3) until E has passed (R3,C2), then A can continue.…
> — qwen38 / dyad / S1 / freeform / instance_00003 / turn 3 / Player B
> Claim: A at (R3,C3). Board: (R3,C7).

The last example also shows why 5.1 warns that the measure counts statements of
*intended* position as errors. Player B is proposing where A should hold, not
asserting where A is, and the replay scores it wrong.

## `names_other` — reference to a teammate's object

No lexicon of its own: any bare letter A–F in the message that the ownership map
assigns to another player. The teammate-correct example above is a hit, since D
belongs to Player B and the speaker is Player A.

## `tgt_disclosed` — target disclosure

Also derived rather than lexical: a `COORD` hit that equals one of the speaker's
own target cells, read from the episode's `instance.json`. This is why the rate
is near-identical across protocols. The structured `PLAN:` clause forces the
destination into every message.

## `REQ` — politeness markers

Feeds the `request` column of 5.2 and the request row of 5.5.

> …please keep (R5,C1) and (R4,C1) clear as agreed.…
> — qwen38 / dyad / S3 / freeform / instance_00012 / turn 10 / Player A

> …C, can you clear R1C5 when B needs it?…
> — dsv41 / 6p / S3 / freeform / instance_00013 / turn 0 / Player D

> …I'll start moving D left toward column 1, and I kindly ask you to move your object from (R6,C1) when possible to clear the path for D to proceed up to row 2.…
> — nemotron / dyad / S3 / hybrid / instance_00001 / turn 12 / Player A

## `MOVEREQ` — directed move requests, the compliance table of 5.4

Stricter than `REQ`. It requires a move verb, a named object or occupied cell,
and a direction, and the object must belong to another player.

> …Please move D left through (R1,C6)/(R1,C5) when ready; I’ll keep (R2,C5)/(R3,C5) clear.…
> — qwen38 / dyad / S3 / freeform / instance_00007 / turn 2 / Player B

The ownership test matters. Without it the same pattern matches a speaker
narrating their own move, as in "For now, I'll move C up to open space", which is
not a request at all.

## `WAIT` — self-reported immobility

Feeds the `blocked/waiting` column of 5.2 and the distress row of 5.5.

> I’m blocked left and up, so I must stage right to (R8,C7), vacating (R8,C6).…
> — qwen38 / 6p / S3 / hybrid / instance_00002 / turn 4 / Player B

> I’m stuck behind your X in the C4 dead-end; I must move up to (R6,C4).…
> — qwen38 / 6p / S4 / hybrid / instance_00001 / turn 0 / Player A

> …So F cannot move left yet.…
> — nemotron / dyad / S1 / freeform / instance_00001 / turn 1 / Player B

The pattern is unanchored, so it also fires on a route described as blocked
rather than a player reporting itself stuck:

> …However, C's path to (R3,C4) is blocked by X at (R4,C4).…
> — nemotron / dyad / S1 / freeform / instance_00001 / turn 1 / Player A

## `ACK` — acknowledgement

Feeds the `ack` column of 5.2, and one of only two features higher in successful
dialogue.

> …F, thanks for staying placed.…
> — qwen38 / 6p / S3 / hybrid / instance_00003 / turn 5 / Player A

> …I’ll hold E at (R8,C7) for now, as requested.…
> — qwen38 / dyad / S4 / hybrid / instance_00010 / turn 26 / Player B

> Got it: F is at R8C4.…
> — dsv41 / dyad / S2 / freeform / instance_00002 / turn 4 / Player A

False positive, `noted` as a reporting verb about a third party rather than an
acknowledgement:

> …D's path down column 5 may be blocked by X at (R6,C5) — I see teammate B noted column 5 obstructions affecting E's path, so I'll avoid moving into column 5 unnecessarily.…
> — nemotron / 3p / S1 / freeform / instance_00011 / turn 4 / Player A

## `SEQ` — sequencing markers

Feeds the `seq` column of 5.2, the 78 to 86 percent against 2 percent contrast
that separates prose from the schema.

> I’ll continue A along row 1 toward (R1,C6), then down to (R2,C6).…
> — qwen38 / dyad / S2 / hybrid / instance_00001 / turn 20 / Player A

> …B, please take row 8 first and finish at (R7,C2).…
> — qwen38 / 6p / S2 / freeform / instance_00007 / turn 0 / Player D

> …Instead, I'll move A down to (R8,C2) to start its descent, then left next turn.…
> — nemotron / dyad / S1 / freeform / instance_00001 / turn 8 / Player A

## `CORR` — corridor and dead-end talk

Feeds the corridor row of 5.5, which turns out not to distinguish outcomes.

> …I won’t use (R5,C1) or block the C1 corridor.…
> — qwen38 / dyad / S2 / hybrid / instance_00014 / turn 5 / Player B

> …going up to (R1,C7) is a dead end.…
> — dsv41 / 6p / S4 / hybrid / instance_00015 / turn 2 / Player B

> I'm moving D down to clear (R3,C2) and start D's route to the C5 pocket.…
> — qwen38 / dyad / S3 / freeform / instance_00003 / turn 0 / Player A

## `THINK` — reasoning-trace leakage

The first sanity check of 5.1. One message in 66 584 matches, and it is the
borderline Nemotron case the document mentions. No `<think>` tag appears
anywhere; the hit is a conversational phrase.

> …Let me think: maybe we need to move objects out of the way.…
> — nemotron / dyad / S3 / freeform / instance_00008 / turn 4 / Player A

## `CJK` — non-Latin script

The second sanity check of 5.1. Zero matches in the corpus, so there is nothing
to quote. Chinese-language leakage does not reach the channel.

## `REQNONE` — the schema's unused request slot

Shows that the structured protocol's single coordination slot is filled with
`none` in 54 percent of messages for both strong models.

> …PLAN: move A from (R4,C8) to (R2,C5) — REQUEST: none
> — qwen38 / dyad / S2 / hybrid / instance_00005 / turn 24 / Player A

## `QUESTION` — questions in the broadcast

A bare question mark. Reported for 5.3, where Nemotron reaches 12 percent against
0.2 to 0.3 percent for the strong models.

> …C's path left is blocked by ## at (R2,C5) and (R2,C4)? Actually (R2,C4) is .…
> — nemotron / 3p / S4 / freeform / instance_00011 / turn 1 / Player C

## `STAY` — the pseudo-action the rules do not offer

Exact string test for `move: <X> stay` or `move: <X> wait` in the action slot.
Nineteen messages in 66 584 match, all of them Nemotron, and eighteen dispute the
action inside the same message. This replaces the earlier self-correction rate in
5.3: it measures the same deliberation with no lexicon and no judgement.

> …PLAN: move B from (R4,C8) to (R4,C8) — REQUEST: none / move: B stay / Wait — "stay" is not a valid direction. The rules require moving exactly one object exactly one cell in one of the four directions: up, down, left, or right.…
> — nemotron / 6p / S1 / hybrid / instance_00012 / turn 6 / Player B

> …Since "stay" is not a valid direction (must be up, down, left, right), and I cannot move A without blocking or being blocked, I must choose a direction that doesn't worsen…
> — nemotron / 6p / S4 / freeform / instance_00004 / turn 1 / Player A

The strong models announce waits in prose but never try to play one:

> …I'll wait at (R4,C3) until it clears.…
> — dsv41 / 3p / S3 / hybrid / instance_00012 / turn 1 / Player C

> …I can wait in C7 if needed…
> — qwen38 / 6p / S3 / hybrid / instance_00002 / turn 0 / Player A

## `TEMP` — the plea for a temporary shift

Exact match on `temporary` or `temporarily`, with a second test for whether the
message also names the return. This replaces the earlier hold-position and
turn-yielding rates in 5.2. The trigger word is unambiguous: nobody asks for a
temporary move except to get past something that will not move on its own.

| | messages | say "temporarily" | also name the return |
|---|---|---|---|
| strong-tier prose | 40 225 | 514 | 52 |
| structured, all models | 22 544 | 8 | 0 |

Prose, with the return:

> …B, please shift X R5C2→R5C1 next turn, even though done, so I can pass R5C4→R5C3→R5C2 toward R1C2.…
> — dsv41 / 6p / S3 / freeform / instance_00014 / turn 10 / Player D (message 88)

> …C’s done object at (R7,C6) blocks the direct westward row, so I’ll go via R8,C7–C5, then up to (R6,C5).…
> — dsv41 / 6p / S2 / hybrid / instance_00001 / turn 6 / Player F

All eight structured attempts ask for the shift and have nowhere to put the
return, because `REQUEST` holds one cell:

> PLAN: move D from (R8,C2) to (R8,C1) — REQUEST: please move (R8,C3) to (R7,C3) temporarily
> — qwen38 / 6p / S3 / structured / instance_00002 / turn 1 / Player D

> PLAN: move C from (R6,C1) to (R4,C2) — REQUEST: A please clear (R4,C2) after temporary use
> — dsv41 / 3p / S2 / structured / instance_00007 / turn 2 / Player C

## The S3 #14 worked episode

Section 5.6 now rests on one episode read end to end rather than on a
done-blocking lexicon: DeepSeek, 6p, S3, free-form, board #14, 101 messages,
ending in a cycle loss at 97 moves. Every count below is an exact test.

Player B says six times that it will wait at (R5,C1) until D has crossed (R5,C2):

> …I’ll wait at (R5,C1) before entering my goal (R5,C2) if D still needs to cross through it.…
> — dsv41 / 6p / S3 / freeform / instance_00014 / turn 1 / Player B (message 8 of 101)

> B moving up to R5C1 as planned. I'll hold there until D/E signal R5C2/R5C3 are clear or no longer needed, then step into my goal R5C2.…
> — dsv41 / 6p / S3 / freeform / instance_00014 / turn 8 / Player B (message 44)

At message 50 it takes the cell anyway, stating the conflict it is creating:

> B moves right to goal R5C2. D: I know you need R5C2 to reach R1C2; I'll step left to R5C1 when you reach R5C3, so signal me.…
> — dsv41 / 6p / S3 / freeform / instance_00014 / turn 9 / Player B (message 50)

At message 56 it declares `done`, which makes that offer unkeepable. Over the 45
remaining messages D raises (R5,C2) nineteen times, asking B to move in eighteen
of them, and E raises it eighteen times. B never speaks again.

> …B, since you're done at R5C2, could you step to R5C1 briefly when I arrive (~10 turns) and then return?…
> — dsv41 / 6p / S3 / freeform / instance_00014 / turn 10 / Player D (message 58)

> …I still see R5C2 and R7C5 blocked by done objects; if any done teammate can temporarily shift, I need one open to reach R1C2.…
> — dsv41 / 6p / S3 / freeform / instance_00014 / turn 10 / Player D (message 100)

The turn index is the same for all of these because the interaction file groups
the episode into few turn blocks; the message index within the episode, which the
script reports, is what orders them.
