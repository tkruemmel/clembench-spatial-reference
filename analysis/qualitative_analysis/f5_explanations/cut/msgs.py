import collections, json, pickle, re
from order import *
SENT = re.compile(r"[^.;!?\n]+")
PASS = re.compile(r"\b(pass\w*|through|enter\w*|cross\w*|via|thread\w*|descend\w*|climb\w*|into|inside|reach\w*|go(?:es)? (?:up|down|left|right)|route\w*|path)\b", re.I)
CORR = re.compile(r"\b(corridor|dead[- ]?end|entrance|chokepoint|bottleneck|doorway|single[- ]file|narrow|tunnel|pocket|cul-de-sac)\b", re.I)
ORDER = re.compile(r"\b(until|till|once|after|before|first|wait\w*|hold(?:ing)? (?:off|back)|let\b|delay\w*|later|ahead of|behind|then (?:I|B)\b)", re.I)

def ref(o):
    return re.compile(r"(?<![A-Za-z(,])%s(?![A-Za-z0-9])(?!\s*,\s*C)" % o)

def tag_episode(rec, ins, ev, h, v, win_end):
    B = Board(ins)
    E, T = B.tgt[h], B.tgt[v]
    rxE, rxT = re.compile(cell_rx(E), re.I), re.compile(cell_rx(T), re.I)
    corr = ins.get("device", {}).get("corridor", [])
    cols = {c[1] for c in corr}; rws = {c[0] for c in corr}
    line = (re.compile(r"\bC%d\b(?!\s*\))" % (list(cols)[0] + 1)) if len(cols) == 1
            else re.compile(r"\bR%d\b(?!\s*,)" % (list(rws)[0] + 1)))
    rh, rv = ref(h), ref(v)
    ho, vo = B.owner[h], B.owner[v]
    hits = collections.defaultdict(list)
    for e, ply, pos, done in replay(ins, ev):
        if e[0] != "msg" or ply > win_end:
            continue
        spk, text = e[1], e[2]
        plan = re.search(r"PLAN:\s*move\s+([A-F])\s+from\s+\(R(\d),\s*C(\d)\)\s+to\s+\(R(\d),\s*C(\d)\)", text)
        if plan and plan.group(1) == v and (int(plan.group(4)) - 1, int(plan.group(5)) - 1) == T:
            hits["d_plan"].append((ply, spk, text))
        if plan and plan.group(1) == h and (int(plan.group(4)) - 1, int(plan.group(5)) - 1) == E:
            hits["h_goal_plan"].append((ply, spk, text))
        for s in SENT.findall(text):
            mv = bool(rv.search(s)) or (spk == vo and re.search(r"\b(I|I'll|I’ll|my)\b", s) and vo == v)
            mh = bool(rh.search(s)) or (spk == ho and re.search(r"\b(I|I'll|I’ll|my)\b", s))
            if rxE.search(s):
                hits["a_door_cell"].append((ply, spk, s))
            if CORR.search(s):
                hits["a2_corr_word"].append((ply, spk, s))
            if mv and (rxE.search(s) or CORR.search(s) or line.search(s)) and PASS.search(s):
                hits["b_v_route_door"].append((ply, spk, s))
            if mv and rxE.search(s) and PASS.search(s):
                hits["b1_v_route_doorcell"].append((ply, spk, s))
            if mv and mh and ORDER.search(s) and (rv.search(s)) and (rh.search(s) or spk == ho or rxE.search(s)):
                hits["c_order_hv"].append((ply, spk, s))
            if rv.search(s) and ORDER.search(s):
                hits["c0_order_any_v"].append((ply, spk, s))
            if rxT.search(s) and (rv.search(s) or spk == vo):
                hits["d_v_target"].append((ply, spk, s))
    return hits
