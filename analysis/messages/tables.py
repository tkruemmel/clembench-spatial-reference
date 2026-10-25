"""Tables, figures and the validity-evidence table (spec §9.7, §13).

`run(scope, tables)` adds the validity-evidence table (one row per measure:
systematized concept, indicator, evidence supplied and lacking, the claims it
may support; D2's status from the §9.6 author check if its metrics exist) and
writes a LaTeX rendering of the headline tables into out/<scope>/tex/.

Evidence kinds: protocol (fixed by the game's grammar or canonical cells, the
replay decides), content (definition and sourced word lists), criterion (the
§9.6 author check), coverage (§9.2 audits), robustness (§9.3 curve),
hypothesis (a known-groups difference the measure is expected to show).
"""
import json
import re
import os

from . import corpus

TEX = ("d2", "frame_composition", "audit_rates", "cross_arm", "curve_summary", "validity_evidence")

ROWS = [
    # §5.1 form
    dict(measure="Words per message, slot presence, PLAN fields, REQUEST filled or none", section="5.1",
         concept="the form of a message under the protocol schema",
         indicator="the game's schema regexes; PLAN-to goal or waypoint by replay",
         supplied="protocol", lacking="none needed (manifest)",
         claims="format compliance; descriptive protocol differences"),
    dict(measure="Think tags in the message", section="5.1", concept="reasoning-format tags broadcast as text",
         indicator="<think> or </think> in the text", supplied="protocol",
         lacking="none needed (manifest)", claims="tag frequency only, not reasoning"),
    # §5.2 private information
    dict(measure="Own letter paired with its true cell", section="5.2",
         concept="a speaker puts its object's position in the channel",
         indicator="G-loc or G-from about an own letter, true at B_t- or B_t+", supplied="content, coverage, robustness",
         lacking="criterion", claims="exploratory, how often the pairing appears"),
    dict(measure="Target stated / own target cell named", section="5.2",
         concept="disclosure of a private target (linkage criterion)",
         indicator="G-target strict / lenient", supplied="content, coverage",
         lacking="criterion", claims="'target stated' is called disclosure; the lenient form is a cell mention only"),
    dict(measure="Own move described", section="5.2", concept="a self-description of the move sent with the message",
         indicator="G-move, present frame, naming the move sent", supplied="content, protocol (the move line)",
         lacking="criterion", claims="consistency of description and move; not an announcement"),
    dict(measure="Later move stated", section="5.2", concept="a statement of the speaker's next moves",
         indicator="PLAN (primary); G-move in a future or sequence frame", supplied="protocol (PLAN), content (prose)",
         lacking="criterion for prose", claims="PLAN-based keeping rates; prose rates exploratory"),
    dict(measure="Cell named before entry", section="5.2", concept="a cell named ahead of the speaker's own entry",
         indicator="G-cell, entry within W rounds, against permuted message times", supplied="protocol (canonical cells)",
         lacking="—", claims="exploratory lead-time distribution"),
    dict(measure="Disclosure curve D(r) / co-presence curve V(t)", section="5.2",
         concept="speaker-side disclosure / listener-side availability of the team's hidden facts",
         indicator="strict statements by round / facts present in the decider's visible context",
         supplied="protocol (visibility model checked against the logged contexts)",
         lacking="criterion for prose-detected facts",
         claims="only V(t) supports statements about what a player had available"),
    # §5.3 reference
    dict(measure="Board-reference classes; device cell or line named", section="5.3",
         concept="what a named cell holds at the moment of the message", indicator="G-cell / G-line classified at B_t-",
         supplied="protocol", lacking="none needed (manifest)", claims="reference targets against a chance baseline"),
    dict(measure="Listener-resolvable reference", section="5.3",
         concept="a named letter the listener can bind to a cell (audience design)",
         indicator="own letter, same-clause true pairing, or a binding in the listener's visible context",
         supplied="protocol, content", lacking="criterion", claims="a rule-relative share of unresolvable mentions"),
    dict(measure="Owners resolvable by the frozen rules", section="5.3",
         concept="teammates' objects whose owner follows from the visible context",
         indicator="PLAN; PLAN plus prose; plus the MOVE BOARD linking rule (test arm)",
         supplied="protocol; synthetic-context unit tests", lacking="—", claims="a lower bound"),
    # §5.4 accuracy
    dict(measure="Own-position accuracy", section="5.4", concept="accuracy of a speaker's statements about its objects",
         indicator="PLAN-from true at B_t- or B_t+ (primary); prose pairs with frame composition",
         supplied="protocol (PLAN-from)", lacking="criterion for prose pairs",
         claims="PLAN-from accuracy; prose descriptive"),
    dict(measure="D2, statements about teammates' objects", section="5.4",
         concept="an indicative statement that teammate object L is now at cell c",
         indicator="G-loc teammate pairs (strict frames), plus the model-extracted records the grammar misses if they pass the author check; true at B_t-",
         supplied="content, coverage, robustness, criterion (the author check, precision only), hypothesis (tier difference)",
         lacking="recall; inter-coder reliability (one coder)",
         claims="confirmatory tier contrast if the author check passes; otherwise exploratory under the manifest name"),
    # §5.5 alignment
    dict(measure="Move alignment O1/O2", section="5.5",
         concept="acceptance-phase evidence, the listener making the named move",
         indicator="letter-direction or REQUEST opportunity at first visibility; board-matched silent baseline; "
                   "negative-control exposure and outcome", supplied="protocol (moves), content (G-move)",
         lacking="causal identification; criterion for prose opportunities",
         claims="associational; 'demonstration' only in the presenter-observable tier"),
    dict(measure="REQUEST exchange chain", section="5.5", concept="presentation, evidence and closure of a REQUEST",
         indicator="REQUEST cell held by a teammate; next turns classified by replay", supplied="protocol",
         lacking="—", claims="descriptive against the silent baseline"),
    dict(measure="Teammate-target cell left free", section="5.5",
         concept="a listener avoids an unfilled teammate target", indicator="legal entry opportunities, split by a stated target",
         supplied="protocol", lacking="causal identification", claims="associational"),
    # §5.6, §5.7
    dict(measure="Talk about finished teammates", section="5.6",
         concept="mentions of a finished teammate's object or cell", indicator="G-cell / G-letter, done line visible or not",
         supplied="protocol", lacking="precedent", claims="descriptive"),
    dict(measure="Owner-side correction", section="5.7",
         concept="an owner restates its object's true cell after a false teammate statement",
         indicator="false visible G-loc about T's letter, then T's next message", supplied="content",
         lacking="criterion", claims="exploratory"),
    # §6, §7, E5
    dict(measure="Hazard features a–e", section="6",
         concept="what was in the actor's view at a fatal decision", indicator="visible-context rules at the call",
         supplied="protocol (risk sets by replay), content", lacking="power for forks",
         claims="'not in view in a form the rules detect' or 'in view, and taken', with compatible odds ratios"),
    dict(measure="Vocabulary ranking", section="7", concept="words that separate two groups",
         indicator="weighted log-odds, permutation-studentized", supplied="robustness (family-wise threshold)",
         lacking="any functional reading", claims="describes vocabulary, not functions"),
    # extension E1-E5
    dict(measure="REQUEST classes, reference accuracy, relevance, recall, compliance", section="ext. E1",
         concept="what a REQUEST slot refers to and whether it is followed",
         indicator="G-cell in a filled REQUEST classed at B_t-; relevance on a shortest wall-only path of the same "
                   "PLAN; blocker spells; vacated or kept clear against unrequested cells, rolled-off requests "
                   "and the board-matched silent game", supplied="protocol",
         lacking="causal identification (compliance)",
         claims="reference accuracy and relevance exact; compliance associational"),
    dict(measure="PLAN execution, validity, persistence, conflict, uptake", section="ext. E2",
         concept="whether the PLAN slot drives the speaker's own move and fits the teammates' plans",
         indicator="PLAN object and progress at the same turn; PLAN-to classes and plan episodes by replay; "
                   "vertex conflicts by a rule checker; avoidance against rolled-off plans", supplied="protocol",
         lacking="causal identification (uptake)", claims="say-do and validity exact; uptake associational"),
    dict(measure="Listener interpretation probe", section="ext. E3",
         concept="the listener's elicited reading of a named object or request",
         indicator="the logged prompt re-sent with a fixed-grammar question, scored by exact match to the replay; "
                   "message-deletion arm", supplied="protocol (exact scoring), intervention (deletion arm)",
         lacking="mutual belief; in-play processing",
         claims="the understanding half of grounding, one step, observer-verified; a wrong answer is a lower bound"),
    dict(measure="Referent swap", section="ext. E4",
         concept="one-step listener sensitivity to the REQUEST referent",
         indicator="the REQUEST cell rewritten to the cell of the listener's other object; the move re-sampled "
                   "against the split-half noise floor", supplied="intervention (causal, one step)",
         lacking="any link to episode success", claims="causal at one step and at the prompt level only"),
    dict(measure="Model-extracted teammate-position records", section="ext. E5",
         concept="statements of a teammate object's current cell that the grammar misses",
         indicator="language-model records passing the acceptance rules, scored by the replay",
         supplied="acceptance rules; extractor agreement on canonical forms; prompt-wording spread; "
                  "criterion (the author check, precision only)",
         lacking="recall", claims="enter D2 only if the model part passes the author check; otherwise exploratory"),
]


def _d2_status():
    """The §9.6 decision, if the author check has been scored (coding/metrics.py)."""
    p = os.path.join(corpus.HERE, "coding", "sample", "metrics.json")
    return json.load(open(p)).get("decision") if os.path.exists(p) else "author check not yet run"


def validity_rows():
    out = []
    for r in ROWS:
        r = dict(r)
        if r["measure"].startswith("D2"):
            r["status"] = _d2_status()
        out.append(r)
    return out


TEXMAP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
          "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}",
          ">": r"\textgreater{}", "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''",
          "×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "−": "$-$", "⁻": "$^-$", "⁺": "$^+$", "…": r"\dots{}",
          "é": r"\'e", "ö": r'\"o', "ü": r'\"u', "ä": r'\"a', "§": r"\S{}", "α": r"$\alpha$", "·": r"$\cdot$"}


def tex_escape(s):
    """Plain text -> ASCII LaTeX (special characters escaped, the few
    non-ASCII characters of the outputs spelled in LaTeX, any other dropped)."""
    return "".join(TEXMAP.get(ch, ch if ord(ch) < 128 else "") for ch in str(s))


def tex_prose(s):
    """A validity-table phrase as LaTeX: escaped, boards as math, quoted terms in TeX quotes."""
    s = tex_escape(s).replace("B\\_t-", "$B_t^-$").replace("B\\_t+", "$B_t^+$")
    s = s.replace("\\textless{}think\\textgreater{}", "\\texttt{\\textless{}think\\textgreater{}}")
    s = s.replace("\\textless{}/think\\textgreater{}", "\\texttt{\\textless{}/think\\textgreater{}}")
    return re.sub(r"(?<![A-Za-z])'([^']+)'(?![A-Za-z])", r"``\1''", s)


VALIDITY_COLS = (("measure", "Measure", 0.13), ("concept", "Systematized concept", 0.15),
                 ("indicator", "Indicator", 0.22), ("supplied", "Evidence supplied", 0.15),
                 ("lacking", "Evidence lacking", 0.11), ("claims", "Claims it may support", 0.17))


def validity_tex(rows, caption, label):
    """The validity-evidence table (spec §9.7) as one LaTeX float in the thesis style."""
    spec = "".join(rf">{{\raggedright\arraybackslash}}p{{{w}\linewidth}}" for _, _, w in VALIDITY_COLS)
    out = [r"\begin{table}[tbp]", r"  \centering", r"  \scriptsize", r"  \setlength{\tabcolsep}{3pt}",
           rf"  \caption{{{caption}}}", rf"  \label{{{label}}}", rf"  \begin{{tabular}}{{@{{}}{spec}@{{}}}}",
           r"    \toprule", "    " + " & ".join(h for _, h, _ in VALIDITY_COLS) + r" \\", r"    \midrule"]
    for i, r in enumerate(rows):
        cells = [tex_prose(r.get(k, "")) for k, _, _ in VALIDITY_COLS]
        if r.get("status"):
            cells[-1] += f" ({tex_prose(r['status'])})"
        out.append("    " + " & ".join(cells) + r" \\" + (r" \addlinespace[2pt]" if i < len(rows) - 1 else ""))
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def _tex(rows, caption):
    if not rows:
        return ""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    esc = lambda s: str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else esc(v)
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize", rf"\caption{{{esc(caption)}}}",
             r"\begin{tabular}{" + "l" * len(keys) + "}", r"\toprule", " & ".join(esc(k) for k in keys) + r" \\", r"\midrule"]
    lines += [" & ".join(fmt(r.get(k, "")) for k in keys) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def run(scope, tables, out_dir=None):
    out = dict(validity_evidence=validity_rows())
    d = os.path.join(corpus.HERE, "out", out_dir or scope, "tex")
    os.makedirs(d, exist_ok=True)
    for name in TEX:
        rows = out.get(name) or tables.get(name)
        if rows:
            with open(os.path.join(d, f"{name}.tex"), "w") as f:
                f.write(_tex(rows, name.replace("_", " ")))
    return out
