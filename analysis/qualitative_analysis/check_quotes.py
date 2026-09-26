"""Audit every quotation in a markdown file against the extracted corpus.

Works on this directory's README.md (default) and on Appendix C of
results_three_models_grid.md, which uses the same provenance format.

Each appendix entry is a block of `> ` lines followed by a `> — provenance` line.
This re-reads the corpus cache and confirms that each quoted string occurs
verbatim in the message the provenance names. Ellipses mark elisions at the ends
of a quote and are stripped before matching; nothing inside a quote is elided.

Usage:  python3 check_quotes.py  [--cache corpus.json] [--readme README.md]
"""
import argparse, json, os, re, sys

PROV = re.compile(r"^([\w.-]+) / (\w+) / (S\d) / (\w+) / instance_(\d+) / turn (\d+) / Player (\w+)")
ANNOTATION = ("Claim:", "Board:")
# the results document uses display names; the corpus uses arm keys
ALIAS = {"Nemotron": "nemotron", "Qwen3.8": "qwen38", "DeepSeek-V4.1": "dsv41"}


def blocks(path, start=None):
    """Yield (quote_lines, provenance) for every attributed quote block.

    A block is one or more consecutive `> ` lines immediately followed by a
    `> \u2014 provenance` line. Anything else resets the accumulator, so quoted
    passages without provenance are skipped rather than merged into the next one.
    """
    out, cur, live = [], [], start is None
    for ln in open(path).read().split("\n"):
        if not live:
            live = ln.startswith(start)
            continue
        if ln.startswith("> \u2014 "):
            if cur:
                out.append((cur[:], ln[4:].strip()))
            cur = []
        elif ln.startswith("> "):
            body = ln[2:]
            cur.append(body) if not body.startswith(ANNOTATION) else None
        else:
            cur = []
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--cache", default=os.path.join(here, "corpus.json"))
    ap.add_argument("--readme", default=os.path.join(here, "README.md"),
                    help="file to audit; pass the results document to audit Appendix C")
    ap.add_argument("--start", default=None,
                    help="only audit from the first line with this prefix onward")
    a = ap.parse_args()
    if not os.path.exists(a.cache):
        sys.exit("corpus.json missing; run: python3 qual_corpus.py --refresh")
    rows = json.load(open(a.cache))["rows"]
    idx = {}
    for r in rows:
        idx.setdefault((r["model"], r["suite"], r["lvl"], r["proto"],
                        r["gid"], r["turn"], r["spk"]), []).append(r["text"])
    ok = bad = 0
    for quotes, prov in blocks(a.readme, a.start):
        m = PROV.match(prov)
        if not m:
            print(f"  UNPARSED PROVENANCE  {prov}")
            bad += 1
            continue
        key = (ALIAS.get(m.group(1), m.group(1)), m.group(2), m.group(3), m.group(4),
               int(m.group(5)), int(m.group(6)), m.group(7))
        txts = idx.get(key)
        if not txts:
            print(f"  NO SUCH MESSAGE      {prov}")
            bad += 1
            continue
        joined = " ".join(re.sub(r"\s+", " ", t) for t in txts)
        for q in quotes:
            needle = q.strip().strip("…").strip()
            # one quote renders the message's own newlines as " / "
            if needle in joined or needle.replace(" / ", " ") in joined:
                ok += 1
            else:
                print(f"  NOT VERBATIM         {prov}\n     {needle[:110]}")
                bad += 1
    print(f"\n  {ok} quotations verified verbatim, {bad} failed.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
