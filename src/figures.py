"""Figures for the report (static PNG, light surface). Reads whatever result files exist.
python src/figures.py
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
from summarize import per_problem_acc  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C = {"sv": "#2a78d6", "en": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6",
     "magenta": "#e87ba4", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
     "axis": "#c3c2b7", "surface": "#fcfcfb"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.facecolor": C["surface"],
                     "figure.facecolor": C["surface"], "savefig.facecolor": C["surface"], "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6})


def mean_ci(path, lang, n_boot=2000):
    if not Path(path).exists():
        return None
    try:
        acc = per_problem_acc(path, lang)
    except Exception:
        return None
    if not acc:
        return None
    v = np.array(list(acc.values()))
    rng = np.random.default_rng(0)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fig_main(conds, fname, title):
    """conds: list of (label, path_sv, path_en). Grouped horizontal bars, Swedish vs English, 95% CI."""
    rows = []
    for lab, ps, pe in conds:
        s, e = mean_ci(ps, "sv") if ps else None, mean_ci(pe, "en") if pe else None
        if s or e:
            rows.append((lab, s, e))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    h = 0.36
    for k, (lang, off) in enumerate((("sv", h / 2), ("en", -h / 2))):
        for yi, (lab, s, e) in zip(y, rows):
            v = s if lang == "sv" else e
            if v is None:
                continue
            ax.barh(yi + off, v[0] * 100, height=h - 0.04, color=C[lang], label=None)
            ax.errorbar(v[0] * 100, yi + off, xerr=[[100 * (v[0] - v[1])], [100 * (v[2] - v[0])]], fmt="none",
                        ecolor=C["ink2"], elinewidth=0.8, capsize=0)
            ax.text(v[2] * 100 + 1, yi + off, f"{v[0] * 100:.1f}", va="center", fontsize=7.5, color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=C["ink"])
    ax.set_xlabel("GSM8K accuracy, same 200 TEST problems (%; whiskers: 95% CI)")
    ax.set_xlim(0, 80)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C["sv"], label="problem in Swedish"), Patch(color=C["en"], label="problem in English")],
              loc="lower center", ncol=2, frameon=False)
    ax.set_title(title, loc="left", color=C["ink"], fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)


def fig_sweep():
    p = RESULTS / "msweep" / "table.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    fam = {"derail": C["violet"], "rlang": C["aqua"], "inalign": C["yellow"]}
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for r in rows:
        k = r["cfg"].split("_")[1]
        if k == "base":
            ax.scatter(r["acc_en"] * 100, r["acc_sv"] * 100, s=90, marker="*", color=C["ink"], zorder=5)
            ax.annotate("unedited", (r["acc_en"] * 100, r["acc_sv"] * 100), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=C["ink"])
            continue
        ax.scatter(r["acc_en"] * 100, r["acc_sv"] * 100, s=34, color=fam.get(k, C["muted"]), edgecolor=C["surface"],
                   linewidth=1.5, zorder=4)
    for k, c in fam.items():
        ax.scatter([], [], s=34, color=c, label={"derail": "derail (paper's recipe)", "rlang": "reasoning-language",
                                                  "inalign": "input alignment"}[k])
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_xlabel("English DEV accuracy (%)")
    ax.set_ylabel("Swedish DEV accuracy (%)")
    ax.set_title("Every edit in the DEV grid (100 problems × 2 samples)", loc="left", color=C["ink"], fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep_dev.png", dpi=200)
    plt.close(fig)


def fig_gender():
    p = RESULTS / "gender" / "sweep.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    names = {"early": "layers 4,6", "mid": "layers 10,12", "p50c": "layers 15,19", "late": "layers 20,22",
             "mid_late_8": "8 layers 8–22"}
    cols = [C["sv"], C["en"], C["aqua"], C["yellow"], C["violet"]]
    for (k, lab), c in zip(names.items(), cols):
        g = sorted([x for x in d["grid_dev"] if x["layers"] == k], key=lambda x: x["alpha"])
        ax.plot([x["alpha"] for x in g], [x["auc"] for x in g], marker="o", ms=4, lw=1.6, color=c, label=lab)
    ax.axhline(d["base"]["dev"]["auc"], color=C["ink2"], lw=1, ls="--")
    ax.text(3.2, d["base"]["dev"]["auc"] + 0.003, "unedited model", fontsize=8, color=C["ink2"])
    ax.set_xlabel("edit strength α   (< 0: amplify the en/ett direction;  > 0: suppress / reverse it)")
    ax.set_ylabel("DEV AUC (neuter vs common)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("en/ett agreement (DEV nouns): amplifying ≈ no gain, reversing = damage", loc="left",
                 color=C["ink"], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gender_dose.png", dpi=200)
    plt.close(fig)


def fig_delta(groups, fname, noise):
    """Forest plot of paired Δ (Swedish TEST accuracy vs the unedited model, same problems) with 95% CIs.
    groups: [(group title, [(label, ref, path, kind)])], kind in {"int", "ctrl", "route"}; noise: half-width of the
    shaded band (the seed-to-seed difference of the unedited model)."""
    from summarize import paired
    rows = []
    for g, items in groups:
        got = []
        for lab, ref, p, kind in items:
            if Path(ref).exists() and Path(p).exists():
                try:
                    got.append((lab, paired(ref, p, "sv"), kind))
                except Exception:
                    pass
        if got:
            rows.append((g, None, "head"))
            rows += got
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.4, 0.24 * len(rows) + 1.5))
    ax.axvspan(-noise, noise, color=C["grid"], alpha=0.55, lw=0, zorder=0)
    ax.axvline(0, color=C["ink2"], lw=0.8, zorder=1)
    y = np.arange(len(rows))[::-1]
    col = {"int": C["sv"], "ctrl": C["muted"], "route": C["en"]}
    labels = []
    for yi, (lab, d, kind) in zip(y, rows):
        if kind == "head":
            labels.append(lab)
            continue
        labels.append("   " + lab)
        m, lo, hi = 100 * d["diff"], 100 * d["ci95"][0], 100 * d["ci95"][1]
        ax.plot([lo, hi], [yi, yi], color=col[kind], lw=2, solid_capstyle="round", zorder=3)
        ax.scatter([m], [yi], s=34, color=col[kind], edgecolor=C["surface"], linewidth=1.5, zorder=4)
        ax.text(max(hi, 0) + 0.6, yi, f"{m:+.1f}", va="center", fontsize=7.5, color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=C["ink"], fontsize=8)
    for t, (_, _, kind) in zip(ax.get_yticklabels(), rows):
        if kind == "head":
            t.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ vs unedited, same mode (points, 95% CI)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([], [], color=C["sv"], marker="o", lw=2, label="intervention, answers stay Swedish"),
                       Line2D([], [], color=C["en"], marker="o", lw=2, label="intervention, answers switch to English"),
                       Line2D([], [], color=C["muted"], marker="o", lw=2, label="control (random twin / re-seed)"),
                       Patch(color=C["grid"], label=f"±{noise:.1f}: seed-to-seed difference, unedited model")],
              loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=7.5)
    ax.set_title("Swedish GSM8K: Δ accuracy vs the unedited model in the same mode",
                 loc="right", color=C["ink"], fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)


def delta_groups():
    X, D, T = RESULTS / "x2", RESULTS / "expD", RESULTS / "mtest"
    R = ROOT / "rank_study" / "results"
    bs, bs1 = X / "base_P-sv_R-none.jsonl", D / "gsm_nothink_n4_base.jsonl"
    sel = json.loads((T / "selected.json").read_text()) if (T / "selected.json").exists() else {}
    short = {"edit: derail (paper's recipe)": "derail (paper's recipe)", "edit: reasoning-language": "reasoning-language",
             "edit: input alignment": "input alignment"}
    wsp = []
    for lab, n in sel.items():
        wsp.append((f"{short.get(lab, lab)}, DEV-selected", bs, T / f"{n}.jsonl", "int"))
        wsp.append((f"{short.get(lab, lab)} — random twin", bs, T / f"{n.replace('msw_', 'mrnd_')}.jsonl", "ctrl"))
    wsp.append(("reasoning-language L15–19 α=6 (excluded by rule)", bs, T / "msw_rlang_L15-19_a6.jsonl", "route"))
    wsp.append(("its random twin", bs, T / "mrnd_rlang_L15-19_a6.jsonl", "ctrl"))
    return [
        ("Noise floor", [("unedited, other sampling seed", bs, bs1, "ctrl")]),
        ("Inference only (no weight change)", [
            ("presence penalty 0", bs, X / "base_P-sv_R-none_pp0.jsonl", "int"),
            ("greedy decoding", bs, X / "base_P-sv_R-none_greedy.jsonl", "int"),
            ("'reason in English' in the user turn", bs, X / "base_P-sv_R-enU.jsonl", "int"),
            ("self-translate, then solve in English", bs, X / "base_P-sv_selftranslate.jsonl", "route"),
            ("English answer prefill", bs, X / "base_P-sv_R-enPrefill.jsonl", "route")]),
        ("The paper's p50c edit (vs same-seed unedited)", [
            ("p50c", bs1, D / "gsm_nothink_n4_p50.jsonl", "int"),
            ("p50c random twin", bs1, D / "gsm_nothink_n4_rnd.jsonl", "ctrl")]),
        ("WSP-style edits (directions from activations)", wsp),
        ("Thinking mode on (16k cap, 1 sample; vs unedited, thinking on)", [
            ("p50c (the paper's loop edit)", D / "gsm_think16k_base.jsonl", D / "gsm_think16k_p50.jsonl", "int"),
            ("p50c random twin", D / "gsm_think16k_base.jsonl", D / "gsm_think16k_rnd.jsonl", "ctrl"),
            ("budget forcing, 4k thinking tokens", D / "gsm_think16k_base.jsonl", D / "gsm_think16k_base_bf4096.jsonl", "int"),
            ("budget forcing, 8k thinking tokens", D / "gsm_think16k_base.jsonl", D / "gsm_think16k_base_bf8192.jsonl", "int")]),
        ("Gradient-trained edits, same RFT data (rank study)", [
            ("rank 1, down_proj L15", bs, R / "test_r1d15x10.jsonl", "int"),
            ("rank 1, down_proj L15, training seed 1", bs, R / "test_r1d15x10s1.jsonl", "int"),
            ("rank 8, down_proj L15", bs, R / "test_r8d15.jsonl", "int"),
            ("rank 64, down_proj L15", bs, R / "test_r64d15.jsonl", "int"),
            ("rank 64, down_proj L15, lr 2e-3", bs, R / "test_r64d15x10.jsonl", "int"),
            ("rank 1, every down_proj", bs, R / "test_r1alldown.jsonl", "int"),
            ("rank 16, all linear (LoRA)", bs, R / "test_r16all.jsonl", "int"),
            ("rank 16, all linear, training seed 1", bs, R / "test_r16alls1.jsonl", "int")]),
    ]


if __name__ == "__main__":
    fig_delta(delta_groups(), "fig_delta_sv.png", noise=3.6)
    X, D, T = RESULTS / "x2", RESULTS / "expD", RESULTS / "mtest"
    conds = [
        ("unedited, default decoding", X / "base_P-sv_R-none.jsonl", X / "base_P-en_R-none.jsonl"),
        ("unedited, presence penalty 0", X / "base_P-sv_R-none_pp0.jsonl", X / "base_P-en_R-none_pp0.jsonl"),
        ("unedited, greedy", X / "base_P-sv_R-none_greedy.jsonl", X / "base_P-en_R-none_greedy.jsonl"),
        ("Swedish problem, 'reason in English'", X / "base_P-sv_R-enU.jsonl", None),
        ("English problem, 'reason in Swedish'", None, X / "base_P-en_R-svU.jsonl"),
        ("paper's p50c edit", D / "gsm_nothink_n4_p50.jsonl", D / "gsm_nothink_n4_p50.jsonl"),
        ("p50c random control", D / "gsm_nothink_n4_rnd.jsonl", D / "gsm_nothink_n4_rnd.jsonl"),
    ]
    sel = json.loads((RESULTS / "mtest" / "selected.json").read_text()) if (RESULTS / "mtest" / "selected.json").exists() else {}
    for lab, name in sel.items():
        conds.append((lab, T / f"{name}.jsonl", T / f"{name}.jsonl"))
        conds.append((lab + " — random twin", T / f"{name.replace('msw_', 'mrnd_')}.jsonl", T / f"{name.replace('msw_', 'mrnd_')}.jsonl"))
    R = ROOT / "rank_study" / "results"
    conds.append(("rank-1 edit, gradient-trained (rank study)", R / "test_r1d15x10.jsonl", R / "test_r1d15x10.jsonl"))
    conds.append(("LoRA r16 all linear (rank study)", R / "test_r16all.jsonl", R / "test_r16all.jsonl"))
    fig_main(conds, "fig_main_gsm.png", "Swedish vs English GSM8K: inference changes, the paper's edit, our edits, fine-tuning")
    think = [("unedited, thinking on (16k)", D / "gsm_think16k_base.jsonl", D / "gsm_think16k_base.jsonl"),
             ("p50c, thinking on (16k)", D / "gsm_think16k_p50.jsonl", D / "gsm_think16k_p50.jsonl"),
             ("p50c random ctrl, thinking on", D / "gsm_think16k_rnd.jsonl", D / "gsm_think16k_rnd.jsonl"),
             ("unedited, thinking on, budget forcing 4k", D / "gsm_think16k_base_bf4096.jsonl", D / "gsm_think16k_base_bf4096.jsonl"),
             ("unedited, thinking on, budget forcing 8k", D / "gsm_think16k_base_bf8192.jsonl", D / "gsm_think16k_base_bf8192.jsonl"),
             ("unedited, thinking off (4 samples)", D / "gsm_nothink_n4_base.jsonl", D / "gsm_nothink_n4_base.jsonl"),
             ("p50c, thinking off (4 samples)", D / "gsm_nothink_n4_p50.jsonl", D / "gsm_nothink_n4_p50.jsonl")]
    fig_main(think, "fig_expD_thinking.png", "Loop fixes (p50c edit, budget forcing), thinking on vs off")
    fig_sweep()
    fig_gender()
    print("figures in", OUT)
