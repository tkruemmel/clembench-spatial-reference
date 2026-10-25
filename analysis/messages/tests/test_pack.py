"""The thesis pack: builds from the current out/dev into a tmp dir, every .tex
compiles alone in a minimal wrapper (tectonic), captions carry no commit hash,
and missing CSVs or columns are skipped and listed rather than raised."""
import glob
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from analysis.messages import pack

TECTONIC = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
DEV = os.path.join(pack.OUT, "dev")
WRAPPER = r"""\documentclass[12pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,graphicx,enumitem}
\usepackage[dvipsnames]{xcolor}
\usepackage[round]{natbib}
\usepackage[hidelinks]{hyperref}
\graphicspath{{%(figs)s/}}
\newcommand{\thesisfig}[2][\linewidth]{\includegraphics[width=#1]{#2}}
\providecommand{\rw}[1]{(#1)}
\begin{document}
\input{%(file)s}
\end{document}
"""


def captions(text):
    """The argument of every \\caption{...}, braces balanced."""
    out = []
    for m in re.finditer(r"\\caption\{", text):
        i, depth = m.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out.append(text[m.end():i - 1])
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not os.path.exists(os.path.join(DEV, "d2.csv")):
        pytest.skip("no development outputs in out/dev")
    dst = str(tmp_path_factory.mktemp("pack") / "thesis")
    return pack.build("dev", dst=dst), dst


def _compile(args):
    tex, figs, work = args
    os.makedirs(work, exist_ok=True)
    w = os.path.join(work, "w.tex")
    with open(w, "w") as f:
        f.write(WRAPPER % dict(figs=figs, file=tex))
    r = subprocess.run([TECTONIC, "-X", "compile", "--untrusted", "w.tex"], cwd=work, capture_output=True, text=True,
                       timeout=300)
    return tex, r.returncode, (r.stdout + r.stderr)[-3000:]


@pytest.mark.skipif(not os.path.exists(TECTONIC), reason="tectonic not installed")
def test_every_tex_compiles_alone(built, tmp_path):
    pk, dst = built
    texs = sorted(glob.glob(os.path.join(dst, "**", "*.tex"), recursive=True))
    assert any("/tables/" in t for t in texs) and any("/figures/" in t for t in texs)
    assert os.path.exists(os.path.join(dst, "appendix_grammar.tex"))
    jobs = [(t, os.path.join(dst, "figures"), str(tmp_path / f"c{i}")) for i, t in enumerate(texs)]
    with ThreadPoolExecutor(4) as ex:
        bad = [(t, log) for t, rc, log in ex.map(_compile, jobs) if rc]
    assert not bad, "\n\n".join(f"{t}\n{log}" for t, log in bad)


def test_text_rules(built):
    pk, dst = built
    for p in glob.glob(os.path.join(dst, "**", "*"), recursive=True):
        if not p.endswith((".tex", ".md")):
            continue
        text = open(p).read()
        assert not pack.HASH40.search(text), p
        if p.endswith(".tex"):
            assert text.isascii(), p
            for c in captions(text):
                assert not pack.HASH40.search(c), (p, c)
                prose = re.sub(r"\\texttt\{[^}]*\}", "", c)
                assert ": " not in prose, (p, prose[max(0, prose.index(": ") - 60):prose.index(": ") + 20])
                assert "analysis/" not in c and ".py" not in c, (p, c)
                assert "\\\\" not in c, (p, "forced line break in a caption")
    for name in ("digest.md", "changes.md", "MANIFEST.md"):
        assert os.path.exists(os.path.join(dst, name)), name


def test_missing_inputs_are_skipped(tmp_path):
    src = tmp_path / "out"
    src.mkdir()
    if os.path.exists(os.path.join(DEV, "d2_splits.csv")):
        shutil.copy(os.path.join(DEV, "d2_splits.csv"), src)
    with open(src / "d2.csv", "w") as f:  # a d2 table without its p column
        f.write("version,n_strong,n_weak,strong,weak,diff,strong_lo,strong_hi,weak_lo,weak_hi\n"
                "grammar,10,5,0.9,0.4,0.5,0.8,0.95,0.2,0.6\n")
    pk = pack.build("dev", src=str(src), dst=str(tmp_path / "thesis"), calib=str(tmp_path / "nocalib"))
    skipped = dict(pk.skipped)
    assert "d2" in skipped and "lacks columns p" in skipped["d2"]
    assert any(f.endswith("d2.csv") and c == ["p"] for f, c in pk.missing)
    assert "cross_arm" in skipped and "alignment" in skipped
    man = open(tmp_path / "thesis" / "MANIFEST.md").read()
    assert "lacks columns p" in man
    assert os.path.exists(tmp_path / "thesis" / "appendix_grammar.tex")


@pytest.mark.skipif(not os.path.exists(TECTONIC), reason="tectonic not installed")
def test_extension_slots_render_when_present(tmp_path):
    """E1/E2, E3/E4 and E5 floats and the D2 union row appear once their CSVs exist, and compile."""
    src = tmp_path / "out"
    src.mkdir()
    head = "version,name,status,primary,n_strong,n_weak,strong,weak,p,diff,strong_lo,strong_hi,weak_lo,weak_hi\n"
    (src / "d2.csv").write_text(head + "grammar,x,pending §9.6,False,10,5,0.9,0.4,0.01,0.5,0.8,0.95,0.2,0.6\n"
                                       "union,x,pending §9.6,True,12,6,0.88,0.45,0.02,0.43,0.8,0.95,0.2,0.6\n")
    base = "model,prompt,inputs_eligible,messages_annotated,"
    (src / "e5_agreement.csv").write_text(
        base + "tier,n_grammar,n_model,n_both,grammar_only,model_only,disagree_share,n_canonical,canonical_found,canonical_recall\n"
        "m,primary,10,10,strong,100,90,80,20,10,0.25,50,45,0.9\n")
    (src / "e5_model_only.csv").write_text(base + "tier,n,true,note\nm,primary,10,10,weak,7,5,x\n")
    (src / "e5_rejects.csv").write_text(base + "reason,n,example\nm,primary,10,10,span not in message,3,x\n")
    (src / "rp_request_classes.csv").write_text("model,suite,proto,cls,n,share\nqwen38,dyad,hybrid,V,40,0.5\n")
    (src / "probe_e3_summary.csv").write_text("model,arm,n,p_correct\ndsv41,original,100,0.71\n")
    dst = tmp_path / "thesis"
    pk = pack.build("dev", src=str(src), dst=str(dst), calib=str(tmp_path / "nocalib"))
    for name in ("e5", "rp_request_classes", "probe_e3_summary", "d2"):
        assert os.path.exists(dst / "tables" / f"{name}.tex"), (name, pk.skipped)
    assert "Union with model" in (dst / "tables" / "d2.tex").read_text()
    jobs = [(str(dst / "tables" / f"{n}.tex"), str(dst / "figures"), str(tmp_path / f"c{n}"))
            for n in ("e5", "rp_request_classes", "probe_e3_summary", "d2")]
    bad = [(t, log) for t, rc, log in map(_compile, jobs) if rc]
    assert not bad, "\n\n".join(f"{t}\n{log}" for t, log in bad)
