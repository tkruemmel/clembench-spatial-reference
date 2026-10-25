"""Message-analysis figures of the results outline (thesis/results_outline.tex), one model's test arm.

    res_msg_o1           instructions followed: in view against left the board, instructed against other moves
    res_msg_baselines    REQUEST and PLAN outcomes against their baselines
    res_msg_probe        listener probe by team size, one line per protocol
    res_msg_probe_target the same for the teammates' target cells, on the same scale
    res_msg_persistence  how a run of messages with one PLAN ends, per game and protocol
    res_msg_boardref     what the named cells hold, relative to their share of the board

Every number comes from the loaders of analysis/thesis/results_messages.py over
analysis/messages/out/<dir>/, the CSVs the outline's message tables are printed
from, so the figures and the tables cannot drift apart. Sizes, fonts and the
protocol styles are results_final_figs.py's.

    python3 figures/results_messages_figs.py [--dir test_dsv41] [--model dsv41]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import results_final_figs as F
from style import CUE, FAINT, GRID, INK, MOVE, MUTED, OWNER, PAPER, WALL

sys.path.insert(0, os.path.join(F.REPO_ROOT, "analysis", "thesis"))
import results_messages as rm  # noqa: E402

PROTOS = ("structured", "hybrid", "freeform")
SLOTS = ("structured", "hybrid")


def rows_of(protos):
    """(game, protocol) conditions with their y positions, a gap between games."""
    conds, ys, y = [], [], 0.0
    for s in rm.TEAMS:
        if conds:
            y += 0.6
        for p in protos:
            conds.append((s, p))
            ys.append(y)
            y += 1
    return conds, ys


def xaxis(ax, lim=(-3, 103), ticks=(0, 25, 50, 75, 100)):  # -3: a marker at 0 stays whole
    ax.set_xlim(*lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=F.FS_TICK, color=MUTED)
    ax.tick_params(length=0, pad=3)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(FAINT)
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def ylabels(ax, conds, ys):
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{F.GAME_NAME[s]}, {p}" for s, p in conds], fontsize=F.FS_TICK, color=INK)
    ax.invert_yaxis()


def dumbbell(ax, conds, ys, pairs):
    """Comparison (open) joined to observed (filled), in the protocol's colour and marker."""
    for (s, p), y, pair in zip(conds, ys, pairs):
        if pair is None:
            continue
        obs, comp = pair
        colour, marker = F.PROTO_STYLE[p]
        ax.plot([comp, obs], [y, y], color=colour, linewidth=F.LW, zorder=2)
        ax.plot(comp, y, marker=marker, markersize=F.MS, markerfacecolor=PAPER, markeredgecolor=colour,
                markeredgewidth=1.3, linestyle="none", zorder=3)
        ax.plot(obs, y, marker=marker, markersize=F.MS, color=colour, markeredgecolor=PAPER,
                markeredgewidth=0.8, linestyle="none", zorder=4)


def pair_legend(fig, observed, comparison, x=0.6):
    handles = [Line2D([], [], marker="o", markersize=F.MS, color=INK, linestyle="none", label=observed),
               Line2D([], [], marker="o", markersize=F.MS, markerfacecolor=PAPER, markeredgecolor=INK,
                      markeredgewidth=1.3, linestyle="none", label=comparison)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(x, 1.0), ncol=2, frameon=False,
               fontsize=F.FS_LEGEND, labelcolor=INK)


def paired(table, s, p, comparison):
    """(observed, comparison) in % from an alignment_contrasts or rp_baselines row, or None."""
    hit = [r for r in rm.rows(table) if r["model"] == rm.M and r["suite"] == s and r["proto"] == p
           and r["comparison"] == comparison and r["mean_observed"]]
    return (100 * float(hit[0]["mean_observed"]), 100 * float(hit[0]["mean_comparison"])) if hit else None


def o1(out):
    import matplotlib.pyplot as plt
    conds, ys = rows_of(PROTOS)
    fig, axes = plt.subplots(1, 2, figsize=(F.TEXT_W, 3.6), sharey=True)
    for ax, (comp, head) in zip(axes, (("direction_specificity", "against an equally good direction"),
                                       ("silent_baseline", "against the none game"))):
        dumbbell(ax, conds, ys, [paired("alignment_contrasts", s, p, comp) for s, p in conds])
        xaxis(ax)
        F.title(ax, head)
    ylabels(axes[0], conds, ys)
    fig.supxlabel("moves made (%)", fontsize=F.FS_LABEL, color=INK, y=0.015)
    pair_legend(fig, "instructed move", "comparison named above the panel")
    fig.subplots_adjust(left=0.22, right=0.985, top=0.80, bottom=0.13, wspace=0.14)
    return F.publish(fig, out, "res_msg_o1")


def baselines(out):
    import matplotlib.pyplot as plt
    conds, ys = rows_of(SLOTS)
    fig, axes = plt.subplots(2, 2, figsize=(F.TEXT_W, 4.6), sharey=True)
    panels = (("V_silent_baseline", "vacated: no messages"),
              ("V_unnamed_onpath", "vacated: unnamed object"),
              ("K_unnamed_onpath", "kept clear: unnamed cell"),
              ("uptake_silent_baseline", "keep clear: no messages"))
    for ax, (comp, head) in zip(axes.flat, panels):
        dumbbell(ax, conds, ys, [paired("rp_baselines", s, p, comp) for s, p in conds])
        xaxis(ax)
        F.title(ax, head)
    ylabels(axes[0, 0], conds, ys)  # shared y: the left column carries the labels
    axes[1, 0].tick_params(axis="y", labelsize=F.FS_TICK, labelcolor=INK)
    fig.supxlabel("share of requests or decisions (%)", fontsize=F.FS_LABEL, color=INK, y=0.012)
    pair_legend(fig, "with the request or PLAN", "baseline")
    fig.subplots_adjust(left=0.22, right=0.985, top=0.88, bottom=0.09, wspace=0.14, hspace=0.42)
    return F.publish(fig, out, "res_msg_baselines")


def probe_mean(s, p, col, name="probe_e3_decisions"):
    v = [float(r[col]) for r in rm.rows(name)
         if r["model"] == rm.M and r["suite"] == s and r["proto"] == p and r[col] != ""]
    return 100 * statistics.mean(v)


# (CSV, panels, stem): the current cells, and the target cells from analysis/thesis/probe_targets.py
PROBES = (("probe_e3_decisions", (("correct_orig", "answered with the current cell"),
                                  ("named_share", "named in a message in view")), "res_msg_probe"),
          ("probe_e3g_targets", (("correct", "answered with the target cell"),
                                 ("in_view", "target cell named in a message in view")), "res_msg_probe_target"))


def probe(out, name, panels, stem):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.9), sharey=True)
    for ax, (col, head) in zip(axes, panels):
        for k, p in enumerate(PROTOS):  # dodged, so equal values sit side by side
            colour, marker = F.PROTO_STYLE[p]
            F.line(ax, [probe_mean(s, p, col, name) for s in rm.TEAMS], colour, marker, dx=(k - 1) * 0.08)
        F.frame(ax, ["dyad", "3p", "6p"], (35, 102), range(40, 101, 10))  # one scale for both probe figures
        F.title(ax, head)
    axes[0].set_ylabel("teammates' objects (%)", fontsize=F.FS_LABEL, color=INK, labelpad=4)
    handles = [Line2D([], [], color=c, marker=m, markersize=F.MS, markeredgecolor=PAPER, linewidth=F.LW, label=p)
               for p, (c, m) in F.PROTO_STYLE.items() if p in PROTOS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, frameon=False,
               fontsize=F.FS_LEGEND, handlelength=2.2, handletextpad=0.5, columnspacing=1.4, labelcolor=INK)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.79, bottom=0.10, wspace=0.10)
    return F.publish(fig, out, stem)


ENDS = (("run_reached", "reached", MOVE), ("run_unforced", "revised, unforced", OWNER["B"]),
        ("run_forced", "revised, forced", CUE), ("run_stale", "kept, unreachable", WALL))


def persistence(out):
    import matplotlib.pyplot as plt
    conds, ys = rows_of(SLOTS)
    fig, ax = plt.subplots(figsize=(F.TEXT_W, 2.9))
    rp = rm.rows("rp_cells")
    for (s, p), y in zip(conds, ys):
        left = 0.0
        for metric, _, colour in ENDS:
            v = 100 * rm.cell(rp, model=rm.M, suite=s, proto=p, level="all", metric=metric)[0]
            ax.barh(y, v, left=left, height=0.72, color=colour, edgecolor=PAPER, linewidth=0.6, zorder=2)
            if v >= 6:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=F.FS_VALUE,
                        color=F.ink_on(to_rgb(colour)))
            left += v
        if left < 99.95:  # runs still open when the episode ends or turns fatal
            ax.barh(y, 100 - left, left=left, height=0.72, color=PAPER, edgecolor=MUTED, linewidth=0.8, zorder=2)
    xaxis(ax, (0, 100.5))
    ax.grid(False)
    ylabels(ax, conds, ys)
    ax.set_xlabel("runs of messages with one PLAN (%)", fontsize=F.FS_LABEL, color=INK, labelpad=4)
    handles = [Patch(facecolor=c, edgecolor=PAPER, label=lab) for _, lab, c in ENDS]
    handles.append(Patch(facecolor=PAPER, edgecolor=MUTED, label="open"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=5, frameon=False,
               fontsize=F.FS_LEGEND, handlelength=1.0, handletextpad=0.35, columnspacing=0.8, labelcolor=INK)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.86, bottom=0.17)
    return F.publish(fig, out, "res_msg_persistence")


CLASSES = (("own_obj", "own object"), ("own_target", "own target"), ("mate_active", "teammate's object"),
           ("mate_target", "teammate's target"), ("mate_finished", "finished object"),
           ("device", "corridor or entrance"), ("empty", "empty cell"))
PARTS = ((("structured", "slots"), "structured, slots", False), (("hybrid", "slots"), "hybrid, slots", False),
         (("hybrid", "prose"), "hybrid, prose", True), (("freeform", "prose"), "freeform, prose", True))


def boardref(out):
    """Share of named cells over the class's share of the board; a cell never named (ratio 0) is not drawn."""
    import matplotlib.pyplot as plt
    pr = [r for r in rm.rows("pr_cells") if r["table"] == "pr_boardref" and r["model"] == rm.M]
    fig, axes = plt.subplots(1, 3, figsize=(F.TEXT_W, 3.3), sharey=True)
    offsets = (-0.27, -0.09, 0.09, 0.27)
    for ax, s in zip(axes, rm.TEAMS):
        ax.axvline(1, color=MUTED, linewidth=1.1, linestyle=(0, (3, 2)), zorder=1)
        for i, (key, _) in enumerate(CLASSES):
            for off, ((p, part), _, filled) in zip(offsets, PARTS):
                hit = [r for r in pr if r["measure"] == key and r["suite"] == s and r["proto"] == p
                       and r["part"] == part]
                if not hit or not hit[0]["baseline"] or float(hit[0]["est"]) == 0:
                    continue
                ratio = float(hit[0]["est"]) / float(hit[0]["baseline"])
                colour, marker = F.PROTO_STYLE[p]
                ax.plot(ratio, i + off, marker=marker, markersize=F.MS + (1.6 if marker == "^" and filled else 0.8 if marker == "^" else 0),
                        linestyle="none", zorder=3, markerfacecolor=colour if filled else PAPER,
                        markeredgecolor=colour if not filled else PAPER, markeredgewidth=1.2 if not filled else 0.4)
        ax.set_xscale("log")
        xaxis(ax, (0.03, 30), (0.1, 1, 10))
        ax.set_xticklabels(["0.1", "1", "10"], fontsize=F.FS_TICK, color=MUTED)
        ax.minorticks_off()
        F.title(ax, F.GAME_NAME[s])
    axes[0].set_yticks(range(len(CLASSES)))
    axes[0].set_yticklabels([lab for _, lab in CLASSES], fontsize=F.FS_TICK, color=INK)
    axes[0].set_ylim(len(CLASSES) - 0.5, -0.5)
    fig.supxlabel("share of named cells over share of the board (log scale)", fontsize=F.FS_LABEL, color=INK,
                  y=0.015)
    handles = []
    for (p, _), lab, filled in PARTS:
        colour, marker = F.PROTO_STYLE[p]
        handles.append(Line2D([], [], marker=marker, markersize=F.MS + (1.6 if marker == "^" and filled else 0.8 if marker == "^" else 0),
                              linestyle="none", label=lab,
                              markerfacecolor=colour if filled else PAPER,
                              markeredgecolor=colour if not filled else PAPER,
                              markeredgewidth=1.2 if not filled else 0.4))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 1.0), ncol=4, frameon=False,
               fontsize=F.FS_LEGEND, handletextpad=0.3, columnspacing=1.1, labelcolor=INK)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.80, bottom=0.15, wspace=0.08)
    return F.publish(fig, out, "res_msg_boardref")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="test_dsv41")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--out", default=F.OUT_DIR)
    args = ap.parse_args()
    rm.D, rm.M = args.dir, args.model
    if (args.dir, args.model) == ("test_dsv41", "dsv41"):
        # pinned to tab:res-msg-o1, tab:res-msg-probe and tab:res-msg-probe-target, so changed outputs fail here
        assert round(paired("alignment_contrasts", "3p", "freeform", "nc_exposure")[0], 1) == 51.9
        assert round(probe_mean("dyad", "structured", "correct_orig"), 1) == 71.9
        assert round(probe_mean("3p", "structured", "correct", "probe_e3g_targets"), 1) == 46.1
    probes = [path for a in PROBES for path in probe(args.out, *a)]
    for path in o1(args.out) + baselines(args.out) + probes + persistence(args.out) + boardref(args.out):
        print(f"wrote {os.path.relpath(path, F.REPO_ROOT)}")


if __name__ == "__main__":
    main()
