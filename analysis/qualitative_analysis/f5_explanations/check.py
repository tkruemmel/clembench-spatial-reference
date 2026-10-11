"""Pinned numbers behind ch9_f5_additions.tex and current_version_discussion.tex l.69-81, per arm.

Runs the producers in README order with ARM set, reads each number off the printed
line and asserts it. The id column is the row id in thesis/numbers/C_order.md.
Then checks that no f5 cache for this arm is older than episodes.{ARM}.json.

    ARM=updated  python3 check.py     # about 30 min (followup*.py and dsplit.py refit the choice models)
    ARM=original python3 check.py

The step scripts rebuild their caches when run, except verbal/cache.{ARM}.pkl, which
verbal.load() builds only when it is missing; the age check catches a stale one.
Numbers read by hand (C13, C39, C41, C43's "about ten") have no line here.
"""
import glob, os, re, subprocess, sys

ARM = os.environ.get("ARM", "updated")
F5 = os.path.dirname(os.path.abspath(__file__))
QA = os.path.dirname(F5)

# name -> (folder, command); run once each, in this order (later steps read earlier caches)
R = {n: (f, c) for n, f, c in [
    ("doors", "cut", "doors.py"), ("run1", "cut", "run1.py"), ("run4", "cut", "run4.py"), ("run5", "cut", "run5.py"),
    ("run7", "cut", "run7.py"), ("run8", "cut", "run8.py"), ("run9", "cut", "run9.py"), ("run11", "cut", "run11.py"),
    ("run12", "cut", "run12.py"), ("plies", "enclosure", "plies.py"), ("followup", "enclosure", "followup.py"),
    ("followup3", "enclosure", "followup3.py"), ("dsplit", "enclosure", "dsplit.py"), ("verbal", "verbal", "verbal.py"),
    ("t1", "verbal", "t1.py"), ("t2", "verbal", "t2.py"), ("t3b", "verbal", "t3b.py"),
    ("sa_prose", "..", "seal_averted.py --tier strong"),
    ("sa_rigid", "..", "seal_averted.py --tier strong --proto none structured"),
    ("sa_none", "..", "seal_averted.py --tier strong --proto none"),
    ("sa_structured", "..", "seal_averted.py --tier strong --proto structured"),
    ("sa_freeform", "..", "seal_averted.py --tier strong --proto freeform"),
    ("sa_hybrid", "..", "seal_averted.py --tier strong --proto hybrid"),
    ("spells", "..", "seal_averted.py --spells --tier strong"), ("selftrap", "..", "selftrap.py"), ("forced2", "..", "forced2.py")]}

D = r"(\d+/\d+)"
CHOICE = r"a choice existed\s+: (\d+)\n\s+sealing move played\s+: \d+\n\s+declined\s+: (\d+)"
DCELL = r"  d=%s\s+fatal\s+" + D + r"\s+\(\s*\d+%%\) exp\s+(\d+)%%\s+p=(\S+)"
# id, run, regex (groups joined by " "), {arm: expected}
CHECKS = [
    ("C01", R["doors"], r"^boards S1: 15, corridor with a target inside (\d+).*\n^boards S2: 15, corridor with a target inside (\d+).*\n"
                        r"^boards S3: 15, corridor with a target inside (\d+).*\n^boards S4: 15, corridor with a target inside (\d+)",
     {"original": "0 0 15 15", "updated": "0 0 15 15"}),
    ("C02", R["doors"], r"^boards S3: 15, .* target on the entrance (\d+), door pair holder on the entrance (\d+)\n"
                        r"^boards S4: 15, .* target on the entrance (\d+), door pair holder on the entrance (\d+)",
     {"original": "15 15 0 0", "updated": "15 15 0 0"}),
    ("C03", R["sa_none"], CHOICE, {"original": "67 39", "updated": "85 52"}),
    ("C03", R["sa_structured"], CHOICE, {"original": "39 23", "updated": "48 25"}),
    ("C03", R["sa_freeform"], CHOICE, {"original": "35 19", "updated": "43 26"}),
    ("C03", R["sa_hybrid"], CHOICE, {"original": "38 24", "updated": "45 32"}),
    ("C04", R["run11"], r"^S3 v first \(B arrived\)\s+prose " + D + r"\s+rigid " + D + r"\s+p=(\S+)",
     {"original": "106/116 60/110 1.9e-10", "updated": "98/116 54/112 5.2e-09"}),
    ("C05", R["run1"], r"^door, B arrived, v first: none " + D + r"\s+structured " + D + r"\s+freeform " + D + r"\s+hybrid " + D,
     {"original": "24/53 36/57 53/56 53/60", "updated": "23/56 31/56 49/59 49/57"}),
    ("C06", R["run11"], r"^S3 reach .*all eps prose " + D + r"\s+rigid " + D + r"\s+p=(\S+)",
     {"original": "5/120 44/120 1.5e-10", "updated": "16/120 48/120 4.3e-06"}),
    ("C07", R["run1"], r"^door, at the fork \(rigid\): (\d+), declared at once \(cut\) (\d+), gave way (\d+)",
     {"original": "44 42 2", "updated": "48 41 6"}),
    ("C07", R["run1"], r"^door, at the fork \(prose\): (\d+), declared at once \(cut\) (\d+), gave way (\d+)",
     {"original": "5 4 1", "updated": "16 7 8"}),
    ("C08", R["run4"], r"^6p holder near door, D pending: prose stepped elsewhere " + D + r", rigid stepped elsewhere " + D,
     {"original": "56/130 12/85", "updated": "66/164 14/105"}),
    ("C09", R["run11"], r"^S3 B waits near home \(eps>=1 wait\), D pending 6p h non\s+prose " + D + r"\s+rigid " + D + r"\s+p=(\S+)",
     {"original": "23/31 4/36 1.9e-07", "updated": "26/38 5/39 6.1e-07"}),
    ("C10", R["run4"], r"^6p holder near door, D already passed: prose stepped elsewhere " + D + r", rigid stepped elsewhere " + D,
     {"original": "2/99 1/55", "updated": "0/91 0/46"}),
    ("C11", R["run11"], r"^all arrivals landing on a live door\s+prose " + D + r"\s+rigid " + D + r"\s+p=(\S+)",
     {"original": "16/2947 109/2835 2.2e-19", "updated": "28/2927 128/2829 7.9e-18"}),
    ("C12", R["run9"], r"^structured: D's target disclosed before min\(D pass, B arrival\) " + D,
     {"original": "53/57", "updated": "52/56"}),
    ("C14", R["run9"], r"^door-order statement made \(B arrived\): prose " + D + r" structured " + D + r" Fisher p=(\S+)",
     {"original": "112/116 32/57 9.9e-11", "updated": "114/116 29/56 8e-14"}),
    ("C15", R["run9"], r"^v first given a statement: prose " + D + r" structured " + D + r" Fisher p=(\S+)",
     {"original": "104/112 28/32 0.47", "updated": "98/114 17/29 0.0027"}),
    ("C16", R["run11"], r"^structured without order stmt " + D + r" vs none " + D + r" p=(\S+)",
     {"original": "8/25 24/53 0.33", "updated": "14/27 23/56 0.48"}),
    ("C17", R["run11"], r"^structured: v first with door-order stmt " + D + r" vs without " + D + r" p=(\S+)",
     {"original": "28/32 8/25 2.2e-05", "updated": "17/29 14/27 0.79"}),
    ("C18", R["run12"], r"structured \('never said', 'h first'\) (\d+)\n.*structured \('never said', 'v first'\) \d+\n"
                        r".*structured \('said earlier, not visible', 'h first'\) (\d+)",
     {"original": "17 4", "updated": "13 9"}),
    ("C20", R["run8"], r"^first door-order statement by the corridor object's player \(prose\): " + D + r"\n"
                       r"^first door-order statement by the corridor object's player \(structured\): " + D,
     {"original": "60/112 26/32", "updated": "60/114 22/29"}),
    ("C22", R["run12"], r"^6p statement in view \(prose\): " + D + r" decisions; waited with " + D + r", without " + D,
     {"original": "104/130 49/104 7/26", "updated": "112/164 52/112 14/52"}),
    ("C22", R["run12"], r"^6p statement in view \(structured\): " + D + r" decisions; waited with " + D + r", without " + D,
     {"original": "12/53 6/12 4/41", "updated": "18/67 6/18 8/49"}),
    ("C24", R["spells"], r"prose  freed (\d+) of (\d+)\s+victim's own turn came first\s+(\d+)",
     {"original": "19 30 9", "updated": "22 33 11"}),
    ("C25", R["t3b"], r"^strong: .*\n.*names the cell the sealer then entered: " + D + r", same sentence names the sealer: " + D,
     {"original": "29/30 27/30", "updated": "28/33 21/33"}),
    ("C27", R["t3b"], r"^strong: .*\n.*\n.*took it from >=2 legal moves to one: " + D + r" \(left it with one, whatever before: " + D,
     {"original": "15/30 23/30", "updated": "19/33 25/33"}),
    ("C28", R["t3b"], r"^strong: victim last msg plans a return/pass-by " + D + r" \(\d+%\) vs baseline \d+/\d+ \(([\d.]+)%\), binom p=(\S+);",
     {"original": "19/30 22.5 2e-06", "updated": "21/33 21.8 2.7e-07"}),
    ("C30", R["spells"], r"rigid  freed (\d+) of (\d+)\s+victim's own turn came first\s+(\d+)",
     {"original": "25 48 23", "updated": "43 62 19"}),
    ("C31", R["followup"], r"relievable=False: (\d+/\d+) \| relievable=True: (\d+/\d+)",
     {"original": "0/34 44/44", "updated": "6/36 59/59"}),
    ("C32", R["followup3"], r"^  msg arms   relievable\s+" + D + r" \(\s*\d+%\) exp (\d+)%.*\n\s+unrelievable: turn order only\s+" + D + r" \(\s*\d+%\) exp (\d+)% p=(\S+)",
     {"original": "27/44 57 12/56 44 5.9e-06", "updated": "32/60 49 14/58 41 0.00056"}),
    ("C33", R["followup3"], r"^  msg arms  relievable \(n=\d+\).*\n\s+turn-order-only vs relievable: ratio ([\d.]+) \[([\d.]+), ([\d.]+)\]",
     {"original": "0.12 0.03 0.46", "updated": "0.16 0.03 0.72"}),
    ("C34", R["followup3"], r"^  none       relievable\s+" + D + r" \(\s*\d+%\) exp (\d+)%.*\n\s+unrelievable: turn order only\s+" + D + r" \(\s*\d+%\) exp (\d+)% p=(\S+)",
     {"original": "15/32 61 9/22 41 1", "updated": "18/28 66 7/31 34 0.052"}),
    ("C34", R["followup3"], r"^  none      relievable \(n=\d+\).*\n\s+turn-order-only vs relievable: ratio ([\d.]+) \[([\d.]+), ([\d.]+)\]",
     {"original": "2.32 0.62 8.61", "updated": "0.29 0.07 1.16"}),
    ("C35", R["followup"], r"^  all .*\n(?:.*\n){2}\s+k1 x relievable\s+OR ([\d.]+) \[([\d.]+), ([\d.]+)\] z=(\S+)",
     {"original": "2.17 0.75 6.30 1.42", "updated": "7.92 1.86 33.68 2.80"}),
    ("C37", R["dsplit"], r"^msg arms\n" + DCELL % "1" + r"\n.*\n" + DCELL % "2" + r"\n.*\n" + DCELL % "3" + r"\n.*\n" + DCELL % "4-5",
     {"original": "4/21 50 0.0011 3/17 46 0.0073 11/19 60 1 1/11 12 1",
      "updated": "7/22 47 0.11 4/21 31 0.27 5/17 54 0.0098 5/16 32 1"}),
    ("C37", R["dsplit"], r"^none\n" + DCELL % "1" + r"\n.*\n" + DCELL % "2" + r"\n.*\n" + DCELL % "3",
     {"original": "5/8 65 1 2/9 38 0.31 4/9 50 1", "updated": "4/14 52 0.02 2/11 46 0.027 5/16 69 4.5e-06"}),
    ("C38", R["dsplit"], r"^  d=3 fatal seals played (\d+), episode aborted (\d+), episodes (\d+); prose (\d+)",
     {"original": "11 10 11 7", "updated": "5 5 5 5"}),
    ("C40", R["t1"], r"^strong 6p   msgs=\s*(\d+)(?:.*\n){4}\s+route_mate: \d+ \(([\d.]+)/1k.*\n\s+req: \d+ \(([\d.]+)/1k.*\n\s+mob_mate: (\d+) \(([\d.]+)/1k",
     {"original": "14793 442.8 636.8 28 1.9", "updated": "14636 463.3 652.8 22 1.5"}),
    ("C44", R["t2"], r"^strong 6p   rblk spells=.*own spoke (\d+):.*cell\+req (\d+)",
     {"original": "791 430", "updated": "821 439"}),
    ("C45", R["t2"], r"^strong 6p   prec spells=.*own spoke (\d+): mob (\d+)",
     {"original": "1275 92", "updated": "1309 106"}),
    ("C46", R["t3b"], r"^strong: .*\n.*mentions its own restricted mobility: " + D,
     {"original": "3/30", "updated": "3/33"}),
    ("C46", R["t2"], r"^strong 6p   prec .*sealed next ply (\d+): others T\+mob (\d+)",
     {"original": "30 1", "updated": "33 0"}),
    ("C47", R["forced2"], r"declared DONE: (\d+) / (\d+)", {"original": "13 39", "updated": "9 32"}),
    ("C49", R["selftrap"], r"\(n=(\d+)\):\n\s+(\d+)  a teammate sealing the last exit", {"original": "39 39", "updated": "32 32"}),
    ("C50", R["sa_prose"], CHOICE, {"original": "73 43", "updated": "88 58"}),
    ("C50", R["sa_rigid"], CHOICE, {"original": "106 62", "updated": "133 77"}),
    ("C50", R["plies"], r"strong check ok (.*)$",
     {"original": "[(73, 43, 46, 20), (106, 62, 72, 32)]", "updated": "[(88, 58, 41, 16), (133, 77, 87, 39)]"}),
    ("C69", R["dsplit"], r"^turn horizon, msg arms, fatal pinches: seal played d<=2 " + D + r" vs d=3 " + D + r"\s+Fisher p=(\S+)",
     {"original": "7/38 11/19 0.0054", "updated": "11/43 5/17 0.76"}),
]


def run(folder, cmd):
    p = subprocess.run([sys.executable] + cmd.split(), cwd=os.path.normpath(os.path.join(F5, folder)),
                       env=dict(os.environ, ARM=ARM), capture_output=True, text=True)
    assert p.returncode == 0, (folder, cmd, p.stderr[-2000:])
    return p.stdout


if __name__ == "__main__":
    out = {}
    for r in R.values():
        print("running", *r, flush=True)
        out[r] = run(*r)
    bad = 0
    for cid, r, rx, exp in CHECKS:
        m = re.search(rx, out[r], re.M)
        got = " ".join(m.groups()) if m else None
        bad += got != exp[ARM]
        print("%s %s %-28s got %s | want %s" % ("ok " if got == exp[ARM] else "BAD", cid, r[1][:28], got, exp[ARM]))
    episodes = os.path.join(QA, f"episodes.{ARM}.json")
    stale = [f for f in sorted(glob.glob(os.path.join(F5, "*", f"*.{ARM}.pkl")))
             if os.path.getmtime(f) < os.path.getmtime(episodes)]
    print("caches older than episodes.%s.json: %s" % (ARM, stale or "none"))
    assert not bad and not stale, "%d checks failed" % bad
