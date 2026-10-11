"""Every number the method chapter takes from the played episodes.

Covers models_section.tex (dates, decoding, output tokens, </think>, cost),
current_version_task_experiment_design.tex and metrics_chapter.tex (budget,
turn-limit losses, false `done`, replay check, detour ratios of the boards in
play).  Reported episodes only: solo from the original campaign, the team
suites from the arm ARM selects (default updated, the reported one).

    ARM=updated  python3 analysis/thesis/method_numbers.py
    ARM=original python3 analysis/thesis/method_numbers.py   # reproduces the old text's values

Times are UTC.  Qwen ran from a laptop that logs local time; its offset is
read from the server's `created` epoch.  DeepSeek (Hetzner) and Nemotron (ECS)
log UTC.  Episode start is the earliest call start where the backend records
one (Qwen `call_start`, DeepSeek `created`), else the first logged event, which
is one call late (Nemotron).

Prices (fetched 2026-10-11, list price, per 1M tokens):
  DeepSeek deepseek-flash  cache hit 0.003 / miss 0.15 / out 0.60 off-peak,
                           double at peak (01-04 and 06-10 UTC Mon-Fri, not
                           on Chinese public holidays) - api-docs.deepseek.com
  Bedrock Nemotron 3 Super in 0.15 / out 0.65 (US, Ireland, Milan; Frankfurt,
                           where the runs went, is not listed)
"""
import collections
import datetime as dt
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tables_comm as T                                      # noqa: E402
sys.path.insert(0, os.path.join(T.ROOT, "analysis/efficiency_baseline"))
import instance_audit as IA                                  # noqa: E402

GAMES = ("solo",) + T.SUITES
SHORT = {"Qwen3.8": "Qwen", "DeepSeek-V4.1-Flash": "DeepSeek", "Nemotron": "Nemotron"}
DS_PRICE = {"hit": 0.003, "miss": 0.15, "out": 0.60}          # USD per 1M, off-peak
NEMO_PRICE = {"in": 0.15, "out": 0.65}                        # USD per 1M
CN_HOLIDAYS = {dt.date(2026, 10, d) for d in range(1, 8)}     # National Day week
DS_MEASURED_TEAM = 148.40   # balance drawn by the 720 updated team episodes (campaign notes)
DS_MEASURED_ORIG = 77.50    # balance 86.09 -> 8.59 over the 840-episode original campaign


def utc(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).replace(tzinfo=None)


def ts(s):
    return dt.datetime.fromisoformat(s.replace(" ", "T"))


def call_row(c):
    r = c["raw_response_obj"]
    if "output" in r:                                          # Bedrock Converse
        text = "".join(p.get("text", "") for p in r["output"]["message"]["content"])
        u = r.get("usage") or {}
        return dict(out=u.get("outputTokens"), inp=u.get("inputTokens"), reason=None,
                    hit=None, miss=None, stop=r.get("stopReason"), model=None,
                    think="</think>" in text, chars=len(text), created=None, start=None,
                    ts=c["timestamp"])
    ch = r["choices"][0]
    u = r.get("usage") or {}
    text = ch["message"].get("content") or ""
    return dict(out=u.get("completion_tokens"), inp=u.get("prompt_tokens"),
                reason=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                hit=u.get("prompt_cache_hit_tokens"), miss=u.get("prompt_cache_miss_tokens"),
                stop=ch.get("finish_reason"), model=r.get("model"), think="</think>" in text,
                chars=len(text), created=r.get("created"),
                start=(r.get("clem_player") or {}).get("call_start"), ts=c["timestamp"])


def scan(args):
    model, game, d = args
    inst = json.load(open(os.path.join(d, "instance.json")))
    es = json.load(open(os.path.join(d, "scores.json")))["episode scores"]
    ev, stamps, end = collections.Counter(), [], ""
    for turn in json.load(open(os.path.join(d, "interactions.json")))["turns"]:
        for e in turn:
            ty = e.get("action", {}).get("type")
            ev[ty] += 1
            stamps.append(e["timestamp"])
            if ty in ("success", "lose", "abort"):
                end = str(e["action"].get("content"))
    calls = [call_row(c["call"])
             for f in sorted(glob.glob(os.path.join(d, "player_*.requests.json")))
             for c in json.load(open(f))["calls"]]
    exp = d.split(os.sep)[-2]
    level = exp.split("_")[1] if game == "solo" else exp.split("_")[0]
    board = json.dumps([inst["grid_size"], sorted(inst["walls"]),
                        sorted(inst["start_positions"].items()),
                        sorted(inst["target_positions"].items())])
    home = T.objects_home(d) if es["Success"] else None
    bfs = sum(IA.bfs_dist(tuple(inst["start_positions"][o]), tuple(inst["target_positions"][o]),
                          inst["grid_size"], {tuple(w) for w in inst["walls"]})
              for o in inst["objects"])
    return dict(model=model, game=game, exp=exp, level=level, gid=inst["game_id"], path=d,
                proto=inst.get("comm_protocol") or ("thinking" if inst.get("thinking") else "silent"),
                words=len(inst["player_prompt"].split()), success=es["Success"],
                rules=(inst.get("strict"), inst.get("masking_hints"), inst.get("max_retries")),
                parts=json.dumps(inst.get("player_objects") or [inst.get("player_a_objects"),
                                                                 inst.get("player_b_objects")]), aborted=es["Aborted"], moves=es["Move Count"],
                optimal=inst["optimal_moves"], max_turns=inst["max_turns"], bfs=bfs,
                end=end, ev=ev, first=min(stamps), last=max(stamps), calls=calls, home=home,
                board=hashlib.sha1(board.encode()).hexdigest()[:12],
                inst=hashlib.sha1(json.dumps(inst, sort_keys=True).encode()).hexdigest()[:12],
                detour=IA.detour_ratios(inst["start_positions"], inst["target_positions"],
                                        inst["grid_size"], {tuple(w) for w in inst["walls"]}))


def load():
    jobs = [(m, g, os.path.dirname(f))
            for m, pref in T.ARMS.items() for g in GAMES
            for f in sorted(glob.glob(os.path.join(T.ROOT, T.tree(pref, g), "*", "*", "*",
                                                   "instance_*", "interactions.json")))]
    with ProcessPoolExecutor() as ex:
        return list(ex.map(scan, jobs, chunksize=8))


def q(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "n/a"
    qs = st.quantiles(xs, n=4) if len(xs) > 1 else [xs[0]] * 3
    return f"median {st.median(xs):,.0f} [IQR {qs[0]:,.0f}-{qs[2]:,.0f}] (n={len(xs):,}, max {xs[-1]:,})"


def tz_offset(eps):
    """Hours the tree's local clock runs ahead of UTC, from server epochs."""
    diffs = [round((ts(c["start"] or c["ts"]) - utc(c["created"]))
                   .total_seconds() / 3600)
             for e in eps for c in e["calls"] if c["created"]]
    return st.median(diffs) if diffs else 0


def span(eps, off):
    """(start, end) per episode in UTC."""
    out = []
    for e in eps:
        starts = [ts(e["first"])]
        starts += [ts(c["start"]) for c in e["calls"] if c["start"]]
        starts += [utc(c["created"]) + dt.timedelta(hours=off)
                   for c in e["calls"] if c["created"] and not c["start"]]
        h = dt.timedelta(hours=off)
        out.append((min(starts) - h, ts(e["last"]) - h))
    return out


def union_hours(iv):
    tot, cur = 0.0, None
    for a, b in sorted(iv):
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            if cur:
                tot += (cur[1] - cur[0]).total_seconds()
            cur = [a, b]
    return (tot + (cur[1] - cur[0]).total_seconds()) / 3600 if cur else 0


def max_overlap(iv):
    pts = sorted([(a, 1) for a, _ in iv] + [(b, -1) for _, b in iv], key=lambda p: (p[0], p[1]))
    run = best = 0
    for _, s in pts:
        run += s
        best = max(best, run)
    return best


def peak(utc):
    return (utc.weekday() < 5 and utc.date() not in CN_HOLIDAYS
            and (1 <= utc.hour < 4 or 6 <= utc.hour < 10))


def ds_cost(calls):
    usd = peak_calls = 0.0
    for c in calls:
        k = 2 if peak(utc(c["created"])) else 1
        peak_calls += k == 2
        usd += k * ((c["hit"] or 0) * DS_PRICE["hit"] + (c["miss"] or 0) * DS_PRICE["miss"]
                    + (c["out"] or 0) * DS_PRICE["out"]) / 1e6
    return usd, peak_calls


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def main():
    eps = load()
    by = collections.defaultdict(list)
    for e in eps:
        by[(e["model"], e["game"])].append(e)
    opt = T.load_optima()
    print(f"ARM={T.ARM} (team suites); solo is always the original campaign")

    section("1. Data set")
    for m in T.ARMS:
        print(f"{SHORT[m]:9s}", {g: len(by[(m, g)]) for g in GAMES},
              "total", sum(len(by[(m, g)]) for g in GAMES))
    boards = {(e["level"], e["gid"]) for e in eps}
    print(f"episodes {len(eps):,}; distinct boards (level, id) {len(boards)}")
    fp = collections.defaultdict(set)
    for e in eps:
        fp[(e["level"], e["gid"])].add(e["board"])
    print("boards whose physical puzzle differs between any two episodes:",
          sorted(k for k, v in fp.items() if len(v) > 1) or "none")
    same = collections.defaultdict(set)
    for e in eps:
        same[(e["game"], e["exp"], e["gid"])].add(e["inst"])
    diff = [k for k, v in same.items() if len(v) > 1]
    print(f"instance files identical across the three models within a suite: "
          f"{len(same) - len(diff)}/{len(same)}")
    assert all(sum(len(by[(m, g)]) for g in GAMES) == 840 for m in T.ARMS), "840 per model"
    assert len(boards) == 60 and not diff

    section("2. Campaign dates and wall-clock (UTC)")
    for m in T.ARMS:
        for g in GAMES:
            es_ = by[(m, g)]
            off = tz_offset(es_)
            iv = span(es_, off)
            print(f"{SHORT[m]:9s} {g:5s} first {min(a for a, _ in iv):%Y-%m-%d %H:%M}  "
                  f"last {max(b for _, b in iv):%Y-%m-%d %H:%M}  "
                  f"span {(max(b for _, b in iv) - min(a for a, _ in iv)).total_seconds() / 3600:6.1f} h  "
                  f"busy {union_hours(iv):6.1f} h  episode-h {sum((b - a).total_seconds() for a, b in iv) / 3600:7.1f}  "
                  f"max concurrent {max_overlap(iv):2d}  (log clock UTC+{off})")
        solo = by[(m, "solo")]
        off = tz_offset(solo)
        s10 = [iv for e, iv in zip(solo, span(solo, off)) if (e["level"], e["gid"]) == ("S2", 10)]
        rest = [iv for e, iv in zip(solo, span(solo, off)) if (e["level"], e["gid"]) != ("S2", 10)]
        team = [iv for g in T.SUITES for e, iv in zip(by[(m, g)], span(by[(m, g)], tz_offset(by[(m, g)])))
                if (e["level"], e["gid"]) != ("S2", 10)]
        print(f"  {SHORT[m]}: solo without S2#10 {min(a for a, _ in rest):%d %b %H:%M}-{max(b for _, b in rest):%d %b %H:%M}"
              f" (span {(max(b for _, b in rest) - min(a for a, _ in rest)).total_seconds() / 3600:.1f} h,"
              f" busy {union_hours(rest):.1f} h);"
              f" solo S2#10 ({len(s10)} eps) {min(a for a, _ in s10):%d %b %H:%M}-{max(b for _, b in s10):%d %b %H:%M};"
              f" team without S2#10 {min(a for a, _ in team):%d %b %H:%M}-{max(b for _, b in team):%d %b %H:%M}"
              f" (span {(max(b for _, b in team) - min(a for a, _ in team)).total_seconds() / 3600:.1f} h,"
              f" busy {union_hours(team):.1f} h)")

    section("3. Request settings recorded and their evidence in the responses")
    for m, pref in T.ARMS.items():
        for g in GAMES:
            for f in sorted(glob.glob(os.path.join(T.ROOT, T.tree(pref, g), "*", "run.json"))):
                r = json.load(open(f))
                for pm in r["player_models"].values():
                    print(f"{SHORT[m]:9s} {g:5s} run.json model_id={pm['model_spec']['model_id']} "
                          f"gen_args={pm['gen_args']} model_config={pm['model_spec'].get('model_config')}")
    print("(clemcore's openai backend and the repo's openai_compatible backend drop max_tokens for "
          "reasoning_model; the Bedrock backend omits maxTokens under uncapped_output.)")
    for m in T.ARMS:
        for g in GAMES:
            cs = [c for e in by[(m, g)] for c in e["calls"]]
            outs = [c["out"] for c in cs if c["out"] is not None]
            print(f"{SHORT[m]:9s} {g:5s} calls {len(cs):6,}  max out {max(outs):6,}  >300: {sum(o > 300 for o in outs):6,} "
                  f"({100 * sum(o > 300 for o in outs) / len(outs):4.1f}%)  >=16384: {sum(o >= 16384 for o in outs):4,}  "
                  f"stop {dict(collections.Counter(c['stop'] for c in cs))}  "
                  f"model field {dict(collections.Counter(c['model'] for c in cs))}  "
                  f"no usage logged {sum(c['out'] is None for c in cs)}")

    section("4. Output tokens per call (completion incl. reasoning)")
    for m in T.ARMS:
        cs_all = []
        for g in GAMES:
            cs = [c for e in by[(m, g)] for c in e["calls"]]
            cs_all += cs
            print(f"{SHORT[m]:9s} {g:5s} {q(c['out'] for c in cs)}")
        print(f"{SHORT[m]:9s} ALL   {q(c['out'] for c in cs_all)}  total out {sum(c['out'] or 0 for c in cs_all):,}"
              f"  total in {sum(c['inp'] or 0 for c in cs_all):,}")
        nepi = sum(len(by[(m, g)]) for g in GAMES)
        print(f"          calls {len(cs_all):,} over {nepi} episodes; calls/episode "
              f"{q(len(e['calls']) for g in GAMES for e in by[(m, g)])}")
        n_ev = sum(e["ev"]["get message"] for g in GAMES for e in by[(m, g)])
        print(f"          logged responses (get message events) {n_ev:,}; calls in requests files {len(cs_all):,}")
        if any(c["reason"] is not None for c in cs_all):
            r = [c for c in cs_all if c["reason"] is not None]
            share = sum(c["reason"] for c in r) / sum(c["out"] for c in r)
            per = [c["reason"] / c["out"] for c in r if c["out"]]
            vis = sum(c["out"] - c["reason"] for c in r)
            print(f"          reasoning tokens {q(c['reason'] for c in r)}; hidden share of all output "
                  f"{100 * share:.1f}% (per call median {100 * st.median(per):.1f}%); "
                  f"visible chars per visible token {sum(c['chars'] for c in r) / vis:.2f}")
    # Qwen logs no reasoning count: estimate its visible tokens with DeepSeek's chars/token.
    dsr = [c for g in GAMES for e in by[("DeepSeek-V4.1-Flash", g)] for c in e["calls"] if c["reason"] is not None]
    cpt = sum(c["chars"] for c in dsr) / sum(c["out"] - c["reason"] for c in dsr)
    qc = [c for g in GAMES for e in by[("Qwen3.8", g)] for c in e["calls"] if c["out"] is not None]
    print(f"Qwen hidden share, ESTIMATED with DeepSeek's {cpt:.2f} chars/token: "
          f"{100 * (1 - sum(c['chars'] for c in qc) / cpt / sum(c['out'] for c in qc)):.1f}%")
    ratio = {m: st.median([c["out"] for g in GAMES for e in by[(m, g)] for c in e["calls"]
                           if c["out"] is not None]) for m in T.ARMS}
    print("median ratio to Nemotron:", {SHORT[m]: round(ratio[m] / ratio["Nemotron"], 1) for m in T.ARMS})

    section("5. Nemotron inline </think>")
    tot = th = 0
    for g in GAMES:
        cs = [c for e in by[("Nemotron", g)] for c in e["calls"]]
        n = sum(c["think"] for c in cs)
        tot, th = tot + len(cs), th + n
        print(f"{g:5s} {n:5,}/{len(cs):6,} = {100 * n / len(cs):4.1f}%")
    print(f"ALL   {th:5,}/{tot:6,} = {100 * th / tot:4.1f}%")
    cs = [c for g in GAMES for e in by[("Nemotron", g)] for c in e["calls"]]
    print(f"output tokens with </think>: {q(c['out'] for c in cs if c['think'])}")
    print(f"output tokens without:       {q(c['out'] for c in cs if not c['think'])}")
    print("strong models' answers containing </think>:",
          {SHORT[m]: sum(c["think"] for g in GAMES for e in by[(m, g)] for c in e["calls"])
           for m in T.STRONG})

    section("6. Game-level reprompts, aborts and false done")
    kinds = ("invalid_move", "invalid_format", "malformed_message", "invalid_done")
    for m in T.ARMS:
        for g in GAMES:
            es_ = by[(m, g)]
            evc = sum((e["ev"] for e in es_), collections.Counter())
            ends = collections.Counter(e["end"].split(" — ")[0] if "declared done" not in e["end"]
                                       else "false done" for e in es_ if not e["success"])
            print(f"{SHORT[m]:9s} {g:5s} " + "  ".join(f"{k} {evc[k]:5,}" for k in kinds)
                  + f"  | non-success endings {dict(ends)}")
    fd = [e for e in eps if e["ev"]["invalid_done"]]
    print(f"episodes with a false done: {len(fd)}; their outcome",
          dict(collections.Counter(("abort" if e["aborted"] else "success" if e["success"] else "lose")
                                   for e in fd)),
          "by model", dict(collections.Counter(SHORT[e["model"]] for e in fd)))

    section("7. Move budget")
    bad = [e for e in eps if e["max_turns"] != max(4 * e["optimal"], 20)]
    print(f"max_turns == max(4*optimal_moves, 20): {len(eps) - len(bad)}/{len(eps)}")
    team = [e for e in eps if e["game"] != "solo"]
    solo = [e for e in eps if e["game"] == "solo"]
    print(f"team optimal_moves == A_k (attainable optimum of the suite): "
          f"{sum(e['optimal'] == opt[(e['game'], e['level'], e['gid'])] for e in team)}/{len(team)}")
    print(f"solo optimal_moves == per-object BFS bound: {sum(e['optimal'] == e['bfs'] for e in solo)}/{len(solo)}; "
          f"== J: {sum(e['optimal'] == opt[(e['level'], e['gid'])] for e in solo)}/{len(solo)}")
    tl = [e for e in eps if "Max turns" in e["end"]]
    print(f"episodes ending at the move budget (turn-limit losses): {len(tl)}")
    for m in T.ARMS:
        for g in GAMES:
            fr = [e["moves"] / e["max_turns"] for e in by[(m, g)]]
            print(f"{SHORT[m]:9s} {g:5s} moves/max_turns max {max(fr):.2f}  "
                  f">=0.50: {sum(f >= .5 for f in fr):3d}  >=0.75: {sum(f >= .75 for f in fr):3d}")
    den = {e["path"]: (opt[(e["level"], e["gid"])] if e["game"] == "solo"
                       else opt[(e["game"], e["level"], e["gid"])]) for e in eps}
    suc = [e for e in eps if e["success"]]
    print(f"successes {len(suc)}: below their denominator (A_k; J for solo) "
          f"{sum(e['moves'] < den[e['path']] for e in suc)}, exactly at it: team "
          f"{sum(e['moves'] == den[e['path']] for e in suc if e['game'] != 'solo')}, solo "
          f"{sum(e['moves'] == den[e['path']] for e in suc if e['game'] == 'solo')}")
    for m in T.ARMS:
        cyc = collections.Counter(e["game"] for e in by_model(eps, m) if "repeated 3 times" in e["end"])
        print(f"{SHORT[m]:9s} cycle losses by suite {dict(cyc)}")

    section("8. Replay of successful episodes")
    ok = [e for e in suc if e["home"][0] == e["home"][1]]
    print(f"successes replaying to all objects home: {len(ok)}/{len(suc)}")
    assert len(ok) == len(suc)

    section("9. Cost")
    for g in GAMES:
        cs = [c for e in by[("DeepSeek-V4.1-Flash", g)] for c in e["calls"]]
        usd, pk = ds_cost(cs)
        print(f"DeepSeek {g:5s} ESTIMATED {usd:7.2f} USD ({usd / len(by[('DeepSeek-V4.1-Flash', g)]):.3f}/ep; "
              f"{100 * pk / len(cs):.0f}% of calls at peak price)  in {sum(c['inp'] for c in cs):,} "
              f"(cache hit {sum(c['hit'] or 0 for c in cs):,})  out {sum(c['out'] or 0 for c in cs):,}")
    team_usd = sum(ds_cost([c for e in by[("DeepSeek-V4.1-Flash", g)] for c in e["calls"]])[0] for g in T.SUITES)
    solo_usd = ds_cost([c for e in by[("DeepSeek-V4.1-Flash", "solo")] for c in e["calls"]])[0]
    ref = DS_MEASURED_TEAM if T.ARM == "updated" else DS_MEASURED_ORIG
    print(f"DeepSeek team suites ESTIMATED {team_usd:.2f} USD vs MEASURED {ref:.2f} "
          f"({'720 updated team episodes' if T.ARM == 'updated' else '840-episode original campaign incl. solo'}); "
          f"solo ESTIMATED {solo_usd:.2f}")
    if T.ARM == "original":
        spent = topups = 0.0
        prev = None
        for ln in open(os.path.join(T.ROOT, "campaign/supervisor_dsv41.log")):
            m = re.search(r"\] \d+/840 done.*?bal ([\d.]+) \$", ln)
            if m:
                b = float(m.group(1))
                if prev is not None:
                    spent, topups = spent + max(prev - b, 0), topups + (b > prev + 0.5)
                prev = b
        print(f"original campaign: balance drawn between supervisor samples {spent:.2f} USD, "
              f"{int(topups)} top-up(s) in between (the 10-min sample gap around each is not counted)")
    for g in GAMES:
        cs = [c for e in by[("Nemotron", g)] for c in e["calls"]]
        usd = sum((c["inp"] or 0) * NEMO_PRICE["in"] + (c["out"] or 0) * NEMO_PRICE["out"] for c in cs) / 1e6
        print(f"Nemotron {g:5s} ESTIMATED {usd:6.2f} USD  in {sum(c['inp'] or 0 for c in cs):,}  out {sum(c['out'] or 0 for c in cs):,}")
    cs = [c for g in GAMES for e in by[("Nemotron", g)] for c in e["calls"]]
    print(f"Nemotron ALL ESTIMATED "
          f"{sum((c['inp'] or 0) * NEMO_PRICE['in'] + (c['out'] or 0) * NEMO_PRICE['out'] for c in cs) / 1e6:.2f} USD")

    section("10. Detour ratio of the 60 boards in play (per object, BFS / Manhattan)")
    seen, dr = set(), collections.defaultdict(list)
    for e in eps:
        if (e["level"], e["gid"]) not in seen and e["detour"]:
            seen.add((e["level"], e["gid"]))
            dr[e["level"]] += e["detour"]
    for lv in sorted(dr):
        print(f"{lv}: n={len(dr[lv])} mean {st.mean(dr[lv]):.2f} median {st.median(dr[lv]):.2f}")

    section("11. Prompt length in words (player_prompt of the played instance files)")
    for g in GAMES:
        w = collections.defaultdict(set)
        for e in eps:
            if e["game"] == g:
                w[e["proto"]].add(e["words"])
        print(f"{g:5s}", {k: sorted(v) for k, v in sorted(w.items())})
    print("(strict, masking_hints, max_retries) over all episodes:",
          dict(collections.Counter((e["game"],) + e["rules"] for e in eps)))
    for g in T.SUITES:
        parts = collections.defaultdict(set)
        for e in eps:
            if e["game"] == g:
                parts[e["level"]].add(e["parts"])
        print(f"partition {g:4s}", {lv: sorted(v) for lv, v in sorted(parts.items())})


def by_model(eps, m):
    return (e for e in eps if e["model"] == m)


if __name__ == "__main__":
    main()
