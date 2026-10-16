"""Split generated LaTeX into one file per float: thesis/tables/<label>.tex.

The chapters \\input each float where it is discussed, so a regenerated table
lands in place without hand-copying.  Producers (ARM=updated, the default):

    python3 analysis/thesis/tables_comm.py         > thesis/numbers/A_outputs/tables_comm.tex
    python3 analysis/thesis/confirmatory.py        > thesis/numbers/A_outputs/confirmatory.tex
    python3 analysis/thesis/dialogue.py --tex      # writes thesis/numbers/E_tab_dialogue.tex
    python3 analysis/thesis/stall.py --tex         # writes thesis/numbers/E_tab_stall_level.tex
    python3 analysis/thesis/hard_boards.py --tex   > thesis/numbers/D_tab_hard_boards.tex
    python3 analysis/thesis/crossed.py --tex       > thesis/numbers/G_tab_crossed.tex
    (tab:competence is curated from competence.py's keyed output: thesis/numbers/H_tab_competence.tex)

    python3 analysis/thesis/split_floats.py thesis/numbers/A_outputs/*.tex thesis/numbers/*_tab_*.tex

Each table environment goes to tables/<label without "tab:">.tex.  What follows
the Appendix B heading in tables_comm.tex is a section, not a float, and goes
whole to tables/lexicon.tex.
"""
import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thesis", "tables")
FLOAT = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.S)
LEXICON = r"\section*{B\quad The annotation lexicon}"


def split(path):
    text = open(path).read()
    head = f"% Generated from {os.path.basename(path)} by split_floats.py -- do not hand-edit.\n"
    written = []
    if LEXICON in text:
        text, lex = text.split(LEXICON, 1)
        open(os.path.join(OUT, "lexicon.tex"), "w").write(head + LEXICON + lex)
        written.append("lexicon")
    for m in FLOAT.finditer(text):
        labels = re.findall(r"\\label\{tab:([^}]+)\}", m.group(0))
        assert len(labels) == 1, (path, labels)
        open(os.path.join(OUT, labels[0] + ".tex"), "w").write(head + m.group(0) + "\n")
        written.append(labels[0])
    assert written, f"no float in {path}"
    return written


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    seen = {}
    for p in sys.argv[1:]:
        for lab in split(p):
            assert lab not in seen, f"tab:{lab} in both {seen[lab]} and {p}"
            seen[lab] = p
    print(f"{len(seen)} files -> {os.path.normpath(OUT)}: " + ", ".join(sorted(seen)))
