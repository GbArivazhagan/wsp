"""Figures for the three-step story (2x2 diagnostic -> H1 loop failure -> H2 number words -> H3 thinking language).
Reads the result files; nothing is hard-coded.   venvs/hf/bin/python src/figures_story.py
Writes report/figures/fig_story.png (4 panels, used in the research note) and report/figures/fig_numword.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT  # noqa: E402
from figures import C, OUT  # noqa: E402  (same palette and rcParams as the other figures)
from grade_math import grade_numeric  # noqa: E402
from summarize import per_problem_acc  # noqa: E402

NW = ROOT / "numword" / "results" / "test"
SW = ROOT / "swthink" / "results"
RNG_SEED, NBOOT = 0, 5000


def pooled(paths, lang="sv"):
    """Per-problem accuracy averaged over several runs (seeds) of the same condition."""
    accs = [per_problem_acc(p, lang) for p in paths]
    keys = sorted(set.intersection(*[set(a) for a in accs]))
    return {k: float(np.mean([a[k] for a in accs])) for k in keys}


def nw_acc(path):
    acc = {}
    for l in open(path):
        r = json.loads(l)
        acc.setdefault(r["id"], []).append(grade_numeric(r["response"], r["answer"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def mean_ci(acc):
    v = np.array(list(acc.values()))
    rng = np.random.default_rng(RNG_SEED)
    b = v[rng.integers(0, len(v), (NBOOT, len(v)))].mean(1)
    return 100 * v.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def diff_ci(a, b):
    ks = sorted(set(a) & set(b))
    d = np.array([b[k] - a[k] for k in ks])
    rng = np.random.default_rng(RNG_SEED)
    bs = d[rng.integers(0, len(d), (NBOOT, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def bars(ax, xs, vals, colors, width, labels=True):
    for x, (m, lo, hi), c in zip(xs, vals, colors):
        ax.bar(x, m, width=width, color=c, edgecolor=C["surface"], linewidth=1.5, zorder=3)
        ax.plot([x, x], [lo, hi], color=C["ink2"], lw=1.0, zorder=4)
        if labels:
            ax.text(x, hi + 1.2, f"{m:.1f}", ha="center", va="bottom", fontsize=7, color=C["ink"])


def panel_2x2(ax):
    X = RESULTS / "x2"
    cells = {("en", "en"): mean_ci(per_problem_acc(X / "base_P-en_R-none.jsonl", "en")),
             ("en", "sv"): mean_ci(per_problem_acc(X / "base_P-en_R-svU.jsonl", "en")),
             ("sv", "en"): mean_ci(per_problem_acc(X / "base_P-sv_R-enPrefill.jsonl", "sv")),
             ("sv", "sv"): mean_ci(per_problem_acc(X / "base_P-sv_R-none.jsonl", "sv"))}
    w = 0.36
    for i, p in enumerate(("en", "sv")):
        bars(ax, [i - w / 2, i + w / 2], [cells[(p, "en")], cells[(p, "sv")]], [C["en"], C["sv"]], w - 0.03)
    ax.set_xticks([0, 1], ["English problem", "Swedish problem"], fontsize=7.5)
    ax.set_ylabel("GSM8K TEST accuracy (%)")
    ax.set_ylim(0, 70)
    ax.set_title("(a) The gap, no thinking", loc="left", fontsize=8.5, color=C["ink"])
    ax.grid(axis="x", visible=False)


def panel_loops(ax):
    D = RESULTS / "expD"
    rows = [("unedited", D / "gsm_think16k_base.jsonl", C["muted"]),
            ("p50c\ntwin", D / "gsm_think16k_rnd.jsonl", C["muted"]),
            ("budget\nforcing 8k", D / "gsm_think16k_base_bf8192.jsonl", C["en"]),
            ("p50c\nedit", D / "gsm_think16k_p50.jsonl", C["en"])]
    vals = [mean_ci(per_problem_acc(p, "sv")) for _, p, _ in rows]
    bars(ax, range(len(rows)), vals, [c for *_, c in rows], 0.62)
    # the same 200 problems in English: looping is not specific to Swedish
    en = [mean_ci(per_problem_acc(p, "en"))[0] for _, p, _ in rows]
    ax.scatter(range(len(rows)), en, marker="D", s=22, color=C["ink"], edgecolor=C["surface"], linewidth=1.0, zorder=5)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(0, 80)
    ax.set_title("(b) H1: loop removal, thinking on", loc="left", fontsize=8.5, color=C["ink"])
    ax.grid(axis="x", visible=False)
    ax.text(0.5, 73, "loops not controlled", ha="center", fontsize=6.8, color=C["ink2"])
    ax.text(2.5, 73, "loops controlled", ha="center", fontsize=6.8, color=C["ink2"])


def nw_rows():
    base = {c: nw_acc(NW / f"base_{c}_NUM.jsonl") for c in ("svsv", "sven")}
    rows = [("digits instead of words", "digits"), ("B · paper-style rank-1", "B"), ("random twin of B", "Btwin"),
            ("C · gradient rank-1", "C"), ("random twin of C", "Ctwin"), ("D · LoRA r16", "D"),
            ("C2 · rank-1, self-distilled*", "C2"), ("D2 · LoRA r16, self-distilled*", "D2")]
    out = []
    for lab, t in rows:
        d = {}
        for c in ("svsv", "sven"):
            f = NW / f"{t}_{c}_NUM.jsonl"
            if f.exists():
                d[c] = diff_ci(base[c], nw_acc(f))
        out.append((lab, t, d))
    return out


def forest(ax, rows, xlim, fs=7.5):
    y = np.arange(len(rows))[::-1]
    for yi, (lab, t, d) in zip(y, rows):
        for c, off, col in (("svsv", 0.17, C["sv"]), ("sven", -0.17, C["en"])):
            if c not in d:
                continue
            m, lo, hi = d[c]
            twin = "twin" in t
            ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [yi + off] * 2, color=col, lw=2, alpha=0.55 if twin else 1,
                    solid_capstyle="round", zorder=3)
            ax.scatter([max(m, xlim[0])], [yi + off], s=26, color=col if not twin else C["surface"], edgecolor=col,
                       linewidth=1.4, zorder=4)
    ax.axvline(0, color=C["ink2"], lw=0.8, zorder=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=fs, color=C["ink"])
    ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


def panel_numword(ax):
    keep = {"digits", "B", "Btwin", "C2", "D2"}
    rows = [r for r in nw_rows() if r[1] in keep]
    forest(ax, rows, (-8, 15), fs=7)
    ax.set_xlabel("Δ vs unedited (points, 95% CI)", fontsize=7.5)
    ax.set_title("(c) H2: number-word edits (59 problems)", loc="left", fontsize=8.5, color=C["ink"])


def panel_think(ax):
    groups = [("p50c edit\n(16k tokens)", "p50", ""), ("budget forcing 8k\n(unedited)", "base", "_bf8192")]
    w = 0.36
    for i, (lab, m, suf) in enumerate(groups):
        en = pooled([SW / f"{m}_enthink_s{s}{suf}.jsonl" for s in (0, 1)])
        sv = pooled([SW / f"{m}_svthink_s{s}{suf}.jsonl" for s in (0, 1)])
        bars(ax, [i - w / 2, i + w / 2], [mean_ci(en), mean_ci(sv)], [C["en"], C["sv"]], w - 0.03)
        d = diff_ci(en, sv)
        ax.text(i, 80, f"Δ {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]", ha="center", fontsize=6.8, color=C["ink2"])
    ax.set_xticks([0, 1], [g[0] for g in groups], fontsize=7.5)
    ax.set_ylim(0, 88)
    ax.set_title("(d) H3: thinking language (prefill, 2 seeds)", loc="left", fontsize=8.5,
                 color=C["ink"])
    ax.grid(axis="x", visible=False)


def fig_story():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6), gridspec_kw={"width_ratios": [1, 1.12]})
    panel_2x2(axs[0, 0])
    panel_loops(axs[0, 1])
    panel_numword(axs[1, 0])
    panel_think(axs[1, 1])
    axs[1, 1].set_ylabel("accuracy (%)")
    axs[0, 1].set_ylabel("accuracy (%)")
    fig.legend(handles=[Patch(color=C["en"], label="reasoning / thinking in English"),
                        Patch(color=C["sv"], label="reasoning / thinking in Swedish"),
                        Patch(color=C["muted"], label="thinking loops not controlled"),
                        Line2D([], [], color=C["ink"], marker="D", lw=0, markersize=4.5,
                               label="(b) same problems in English"),
                        Line2D([], [], color=C["muted"], marker="o", markerfacecolor=C["surface"], lw=0,
                               label="(c) hollow: random twin")],
               loc="lower center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handletextpad=0.4)
    fig.text(0.01, 0.075, "Qwen3.5-0.8B, GSM8K TEST: 200 problems ((c): the 59 with number words), translated to Swedish. "
             "Bars and lines: 95% bootstrap CI over problems. * exploratory.", fontsize=6.3, color=C["muted"])
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_story.png", dpi=200)
    plt.close(fig)


def fig_numword():
    from matplotlib.lines import Line2D
    rows = nw_rows()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    forest(ax, rows, (-28, 16))
    for yi, (lab, t, d) in zip(np.arange(len(rows))[::-1], rows):
        for c, off in (("svsv", 0.17), ("sven", -0.17)):
            if c in d:
                ax.text(15.6, yi + off, f"{d[c][0]:+.1f}", va="center", ha="right", fontsize=6.5, color=C["ink2"])
    ax.set_xlabel("Δ vs unedited on the 59 number-word TEST problems (points, 95% CI)")
    ax.set_title("H2: number-word edits vs the free alternative (rewrite words as digits)", loc="left", fontsize=9,
                 color=C["ink"])
    fig.legend(handles=[Line2D([], [], color=C["sv"], marker="o", lw=2, label="Swedish reasoning (default)"),
                        Line2D([], [], color=C["en"], marker="o", lw=2, label="English reasoning (answer prefill)"),
                        Line2D([], [], color=C["muted"], marker="o", markerfacecolor=C["surface"], lw=0,
                               label="hollow: norm-matched random twin")],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, 0.005, "C and D collapse to one-line answers (format copied from the synthetic training data). "
             "* exploratory.", fontsize=6.3, color=C["muted"])
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUT / "fig_numword.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_story()
    fig_numword()
    print("wrote", OUT / "fig_story.png", OUT / "fig_numword.png")
