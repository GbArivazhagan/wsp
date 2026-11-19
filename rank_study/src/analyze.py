"""Edit-rank study: tables for DEV / TEST and the cosine analysis.

python rank_study/src/analyze.py dev            # every rank_study/results/dev_*.jsonl vs dev_base.jsonl
python rank_study/src/analyze.py test           # every rank_study/results/test_*.jsonl vs the x2 base files
python rank_study/src/analyze.py cos            # cosines of learned single-matrix dW directions vs known directions
"""
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import load_jsonl  # noqa: E402
from summarize import paired  # noqa: E402

RS = ROOT / "rank_study"
RES = RS / "results"
BASE_SV = ROOT / "results" / "x2" / "base_P-sv_R-none.jsonl"
BASE_EN = ROOT / "results" / "x2" / "base_P-en_R-none.jsonl"


def lang_share(path, lang, want="sv"):
    rows = [r for r in load_jsonl(path) if r["lang"] == lang]
    c = collections.Counter(r["resp_lang"] for r in rows)
    return c[want] / max(1, len(rows)), sum(r["finish"] == "length" for r in rows) / max(1, len(rows))


def finished_acc(path, lang):
    from grade_math import grade_numeric
    rows = [r for r in load_jsonl(path) if r["lang"] == lang]
    fin = [r for r in rows if r["finish"] != "length"]
    return sum(grade_numeric(r["response"], r["answer"]) for r in fin) / max(1, len(fin))


def train_info(name):
    f = RES / "train" / f"rank_{name}.train.json"
    return json.loads(f.read_text()) if f.exists() else {}


def fmt(p):
    lo, hi = p["ci95"]
    return f"{100 * p['diff']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def dev():
    base = RES / "dev_base.jsonl"
    out = []
    print("| DEV run | sv acc | Δ vs base [95% CI] | sv answers | cut | train loss (last ep) |")
    print("|---|---|---|---|---|---|")
    for f in sorted(glob.glob(str(RES / "dev_*.jsonl"))):
        name = Path(f).stem[4:]
        p = paired(str(base), f, "sv")
        sh, cut = lang_share(f, "sv")
        ti = train_info(name)
        el = ti.get("epoch_loss", [None])[-1]
        print(f"| {name} | {100 * p['acc_b']:.1f} | {fmt(p)} | {100 * sh:.0f}% | {100 * cut:.1f}% | {el if el is None else round(el, 4)} |")
        out.append({"name": name, **p, "sv_share": sh, "cut": cut, "train": {k: ti.get(k) for k in ("lr", "epochs", "r", "target", "data", "epoch_loss")}})
    (RES / "dev_summary.json").write_text(json.dumps(out, indent=1))


def test():
    out = []
    print("| condition | trainable params | ‖ΔW‖_F | train loss (last ep) | sv acc | sv Δ [95% CI] | sv answers (sv prompts) | en acc | en Δ [95% CI] | sv answers (en prompts) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(str(RES / "test_*.jsonl"))):
        name = Path(f).stem[5:]
        langs = {r["lang"] for r in load_jsonl(f)}
        NA = {"acc_b": float("nan"), "diff": float("nan"), "ci95": (float("nan"), float("nan"))}
        ps = paired(str(BASE_SV), f, "sv") if "sv" in langs else NA
        pe = paired(str(BASE_EN), f, "en") if "en" in langs else NA
        sh_sv, cut_sv = lang_share(f, "sv")
        sh_en, cut_en = lang_share(f, "en")
        ti = train_info(name)
        npar = ti.get("trainable_params", 0)
        fro = ti.get("dW_fro_total", 0.0)
        el = (ti.get("epoch_loss") or [float("nan")])[-1]
        print(f"| {name} | {npar:,} | {fro:.3f} | {el:.4f} | {100 * ps['acc_b']:.1f} | {fmt(ps)} | {100 * sh_sv:.1f}% | "
              f"{100 * pe['acc_b']:.1f} | {fmt(pe)} | {100 * sh_en:.1f}% |")
        fa = finished_acc(f, "sv") if "sv" in langs else float("nan")
        out.append({"name": name, "sv": ps, "en": pe, "sv_acc_finished": fa, "sv_share_svprompts": sh_sv, "sv_share_enprompts": sh_en,
                    "cut_sv": cut_sv, "cut_en": cut_en, "trainable_params": npar, "dW_fro": fro,
                    "train": {k: ti.get(k) for k in ("lr", "epochs", "r", "target", "data", "seed", "n_examples", "epoch_loss", "dW_rel_mean")}})
    b_sv, _ = lang_share(BASE_SV, "sv")
    b_en, _ = lang_share(BASE_EN, "en")
    print("\n| condition | sv cut (length) | sv acc on finished | en cut |")
    print("|---|---|---|---|")
    print(f"| base | {100 * lang_share(BASE_SV, 'sv')[1]:.1f}% | {100 * finished_acc(BASE_SV, 'sv'):.1f} | {100 * lang_share(BASE_EN, 'en')[1]:.1f}% |")
    for o in out:
        print(f"| {o['name']} | {100 * o['cut_sv']:.1f}% | {100 * o['sv_acc_finished']:.1f} | {100 * o['cut_en']:.1f}% |")
    if out:
        print(f"\nbase: sv answers on sv prompts {100 * b_sv:.1f}%, sv answers on en prompts {100 * b_en:.1f}% "
              f"(base acc: sv {100 * paired(str(BASE_SV), str(BASE_SV), 'sv')['acc_a']:.1f}, en {100 * paired(str(BASE_EN), str(BASE_EN), 'en')['acc_a']:.1f})")
    (RES / "test_summary.json").write_text(json.dumps(out, indent=1))


def cos():
    import numpy as np
    import torch
    known = {}
    for n in ("derail", "rlang", "inalign"):
        d = torch.load(ROOT / "models" / "dirs" / f"{n}.pt", weights_only=False)
        known[n] = d["u"][15].float()
        known[n + "@L14"] = d["u"][14].float()
    known["p50c_loop"] = torch.tensor(np.load(ROOT / "models" / "p50c_u_layer15.npy")).float()
    for k in known:
        known[k] = known[k] / known[k].norm()
    out = {}
    svds = {Path(f).name.split(".")[0].replace("rank_", ""): torch.load(f) for f in sorted(glob.glob(str(RES / "train" / "*.svd.pt")))}
    print("| edit | σ1/‖ΔW‖ | " + " | ".join(f"|cos(u1, {k})|" for k in known) + " | " +
          " | ".join(f"‖P_top-k {k}‖" for k in ("derail", "rlang", "inalign", "p50c_loop")) + " |")
    print("|---" * (2 + len(known) + 4) + "|")
    for name, s in svds.items():
        U, S = s["U"].float(), s["S"].float()
        u1 = U[:, 0]
        c = {k: float((u1 @ v).abs()) for k, v in known.items()}
        proj = {k: float((U.T @ known[k]).norm()) for k in ("derail", "rlang", "inalign", "p50c_loop")}
        frac = float(S[0] / S.norm())
        out[name] = {"cos_u1": c, "proj_topk": proj, "k": U.shape[1], "sigma": S.tolist()[:8]}
        print(f"| {name} | {frac:.2f} | " + " | ".join(f"{v:.3f}" for v in c.values()) + " | " +
              " | ".join(f"{v:.3f}" for v in proj.values()) + " |")
    # cross-cosines between the learned rank-1 directions
    names = list(svds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = svds[names[i]]["U"][:, 0].float(), svds[names[j]]["U"][:, 0].float()
            print(f"|cos(u1[{names[i]}], u1[{names[j]}])| = {float((a @ b).abs()):.3f}")
    rng = torch.Generator().manual_seed(0)
    r = torch.randn(2000, 1024, generator=rng)
    r = r / r.norm(dim=1, keepdim=True)
    print(f"random-direction reference: mean |cos| = {float((r @ known['rlang']).abs().mean()):.3f}, "
          f"99th pct = {float((r @ known['rlang']).abs().quantile(0.99)):.3f}; "
          f"‖P_top-k‖ for random unit vector ≈ sqrt(k/1024)")
    (RES / "cos_summary.json").write_text(json.dumps(out, indent=1))


def decide(r1="r1d15", high=("r16all", "r64d15", "r64d15x10", "r1alldown", "r8d15"), pc="pc_r1"):
    """Apply the pre-registered decision rule to rank_study/results/test_summary.json (run 'test' first).
    'high-rank' conditions = every condition other than R1-single (R1-all-down is rank 1 per matrix but 24 matrices)."""
    S = {r["name"]: r for r in json.loads((RES / "test_summary.json").read_text())}
    g1 = S[r1]["sv"]["diff"]
    hs = {h: S[h]["sv"]["diff"] for h in high if h in S}
    best = max(hs, key=hs.get)
    gb = hs[best]
    ratio = g1 / gb if gb > 0 else float("nan")
    pcs = S[pc]["sv_share_enprompts"] if pc in S else float("nan")
    print(f"R1-single gain {100 * g1:+.1f} pts; best high-rank = {best} {100 * gb:+.1f} pts "
          f"(CI {100 * S[best]['sv']['ci95'][0]:+.1f}, {100 * S[best]['sv']['ci95'][1]:+.1f}); ratio = {ratio:.2f}")
    print(f"positive control: Swedish answers on English TEST problems = {100 * pcs:.1f}%")
    print(f"R1-single Swedish answers on Swedish problems = {100 * S[r1]['sv_share_svprompts']:.1f}%")
    if S[best]["sv"]["ci95"][0] <= 0:
        v = "INCONCLUSIVE (best high-rank gain CI includes 0)"
    elif ratio <= 1 / 3 and pcs >= 0.90:
        v = "SUPPORTED"
    elif ratio >= 2 / 3 and S[r1]["sv_share_svprompts"] >= 0.95:
        v = "REFUTED"
    else:
        v = "INCONCLUSIVE / PARTIAL"
    print("verdict (seed 0, as pre-registered):", v)
    # robustness: average over the two seeds where a seed-1 replicate exists
    def avg(n):
        xs = [S[k]["sv"]["diff"] for k in (n, n + "s1") if k in S]
        return sum(xs) / len(xs), len(xs)
    g1a, k1 = avg(r1)
    hsa = {h: avg(h) for h in high if h in S}
    besta = max(hsa, key=lambda h: hsa[h][0])
    print(f"seed-averaged: R1-single {100 * g1a:+.1f} ({k1} seeds); best high-rank {besta} {100 * hsa[besta][0]:+.1f} "
          f"({hsa[besta][1]} seeds); ratio {g1a / hsa[besta][0] if hsa[besta][0] > 0 else float('nan'):.2f}")


def contrast(a="r1d15", b="r16all", n_boot=5000, seed=0):
    """Joint paired bootstrap over the 200 TEST problems: gain(a), gain(b), gain(b) - gain(a), and gain(a)/gain(b)."""
    import random
    from summarize import per_problem_acc
    B0 = per_problem_acc(str(BASE_SV), "sv")
    A = per_problem_acc(str(RES / f"test_{a}.jsonl"), "sv")
    Bb = per_problem_acc(str(RES / f"test_{b}.jsonl"), "sv")
    keys = sorted(set(B0) & set(A) & set(Bb))
    da = [A[k] - B0[k] for k in keys]
    db = [Bb[k] - B0[k] for k in keys]
    rng = random.Random(seed)
    diffs, ratios = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(len(keys)) for _ in keys]
        ga, gb = sum(da[i] for i in idx) / len(idx), sum(db[i] for i in idx) / len(idx)
        diffs.append(gb - ga)
        ratios.append(ga / gb if gb > 0 else float("nan"))
    diffs.sort()
    ok = sorted(r for r in ratios if r == r)
    q = lambda v, p: v[int(p * (len(v) - 1))]
    ga, gb = sum(da) / len(da), sum(db) / len(db)
    out = {"a": a, "b": b, "gain_a": ga, "gain_b": gb, "diff_b_minus_a": gb - ga,
           "diff_ci95": (q(diffs, 0.025), q(diffs, 0.975)), "ratio": ga / gb if gb > 0 else None,
           "ratio_ci95": (q(ok, 0.025), q(ok, 0.975)) if ok else None,
           "p_ratio_le_1/3": sum(r <= 1 / 3 for r in ok) / max(1, len(ok)),
           "p_ratio_ge_2/3": sum(r >= 2 / 3 for r in ok) / max(1, len(ok)), "boot_gain_b_le_0": sum(r != r for r in ratios) / n_boot}
    print(json.dumps(out, indent=1))
    return out


BASE_S1 = ROOT / "results" / "expD" / "gsm_nothink_n4_base.jsonl"   # same settings as the x2 base, sampling seed 1


def acc_rows(paths, lang):
    """per-problem accuracy pooled over all samples of several files"""
    from grade_math import grade_numeric
    acc = collections.defaultdict(list)
    for pth in paths:
        for r in load_jsonl(pth):
            if r["lang"] == lang:
                acc[r["id"].replace("gsm_sv_", "").replace("gsm_en_", "")].append(grade_numeric(r["response"], r["answer"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def boot(d, n_boot=5000, seed=0):
    import random
    rng = random.Random(seed)
    b = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), (b[int(0.025 * n_boot)], b[int(0.975 * n_boot)])


def test8():
    """Gains vs the pooled 8-sample base (seed 0 + seed 1 base runs, 4 samples each), as asked by the main track."""
    out = {}
    print("| condition | sv Δ vs 8-sample base [95% CI] | en Δ vs 8-sample base [95% CI] |")
    print("|---|---|---|")
    B = {l: acc_rows([BASE_SV if l == "sv" else BASE_EN, BASE_S1], l) for l in ("sv", "en")}
    print(f"| base (8-sample pooled) | acc {100 * sum(B['sv'].values()) / len(B['sv']):.1f} | acc {100 * sum(B['en'].values()) / len(B['en']):.1f} |")
    for l, ref in (("sv", BASE_SV), ("en", BASE_EN)):
        a0, a1 = acc_rows([ref], l), acc_rows([BASE_S1], l)
        m, ci = boot([a1[k] - a0[k] for k in a0])
        print(f"| (noise floor) base seed 1 − base seed 0, {l} | {100 * m:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] | |")
    for f in sorted(glob.glob(str(RES / "test_*.jsonl"))):
        name = Path(f).stem[5:]
        row = {}
        for l in ("sv", "en"):
            c = acc_rows([f], l)
            if not c:
                row[l] = None
                continue
            m, ci = boot([c[k] - B[l][k] for k in c])
            row[l] = {"acc": sum(c.values()) / len(c), "diff": m, "ci95": ci}
        out[name] = row
        f2 = lambda r: "–" if r is None else f"{100 * r['diff']:+.1f} [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}]"
        print(f"| {name} | {f2(row['sv'])} | {f2(row['en'])} |")
    (RES / "test8_summary.json").write_text(json.dumps(out, indent=1))
    # decision rule read against the 8-sample base
    r1 = "r1d15x10"
    hs = {h: out[h]["sv"]["diff"] for h in ("r16all", "r64d15", "r64d15x10", "r1alldown", "r8d15") if h in out and out[h]["sv"]}
    if r1 in out and hs:
        best = max(hs, key=hs.get)
        g1, gb = out[r1]["sv"]["diff"], hs[best]
        print(f"\n8-sample base: R1-single {100 * g1:+.1f}, best high-rank {best} {100 * gb:+.1f} "
              f"(CI {100 * out[best]['sv']['ci95'][0]:+.1f}, {100 * out[best]['sv']['ci95'][1]:+.1f}), ratio {g1 / gb if gb > 0 else float('nan'):.2f}")


def contrast_seeds(n_boot=5000, seed=0):
    """Seed-pooled contrast: R1-single (lr 2e-3, seeds 0+1 -> 8 samples/problem) vs R16-all-linear (seeds 0+1),
    against the seed-0 base (4 samples) and the pooled 8-sample base. Joint bootstrap over the 200 TEST problems."""
    import random
    A = acc_rows([RES / "test_r1d15x10.jsonl", RES / "test_r1d15x10s1.jsonl"], "sv")
    Bc = acc_rows([RES / "test_r16all.jsonl", RES / "test_r16alls1.jsonl"], "sv")
    for bname, bfiles in (("seed-0 base", [BASE_SV]), ("8-sample base", [BASE_SV, BASE_S1])):
        B0 = acc_rows(bfiles, "sv")
        keys = sorted(B0)
        da, db = [A[k] - B0[k] for k in keys], [Bc[k] - B0[k] for k in keys]
        rng = random.Random(seed)
        rs, ds, gas, gbs = [], [], [], []
        for _ in range(n_boot):
            idx = [rng.randrange(len(keys)) for _ in keys]
            ga, gb = sum(da[i] for i in idx) / len(idx), sum(db[i] for i in idx) / len(idx)
            gas.append(ga); gbs.append(gb); ds.append(gb - ga)
            if gb > 0:
                rs.append(ga / gb)
        q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
        ga, gb = sum(da) / len(da), sum(db) / len(db)
        print(f"[{bname}] R1-single (2 seeds) {100 * ga:+.1f} [{100 * q(gas, .025):+.1f}, {100 * q(gas, .975):+.1f}]; "
              f"R16-all (2 seeds) {100 * gb:+.1f} [{100 * q(gbs, .025):+.1f}, {100 * q(gbs, .975):+.1f}]; "
              f"R16 - R1 {100 * (gb - ga):+.1f} [{100 * q(ds, .025):+.1f}, {100 * q(ds, .975):+.1f}]; ratio {ga / gb:.2f} "
              f"[{q(rs, .025):.2f}, {q(rs, .975):.2f}], P(ratio<=1/3)={sum(r <= 1 / 3 for r in rs) / len(rs):.3f}, "
              f"P(ratio>=2/3)={sum(r >= 2 / 3 for r in rs) / len(rs):.3f}")


def pooled(n_boot=5000, seed=0):
    """Secondary (not pre-registered): per-problem paired differences on DEV (100) and TEST (200) pooled -> 300 problems."""
    import random
    from summarize import per_problem_acc
    bd = per_problem_acc(str(RES / "dev_base.jsonl"), "sv")
    bt = per_problem_acc(str(BASE_SV), "sv")
    print("| condition | pooled Δ (300 problems) [95% CI] | DEV Δ | TEST Δ |")
    print("|---|---|---|---|")
    out = {}
    for f in sorted(glob.glob(str(RES / "test_*.jsonl"))):
        name = Path(f).stem[5:]
        fd = RES / f"dev_{name}.jsonl"
        if not fd.exists():
            continue
        cd, ct = per_problem_acc(str(fd), "sv"), per_problem_acc(f, "sv")
        if not ct:
            continue
        d = [cd[k] - bd[k] for k in sorted(set(cd) & set(bd))] + [ct[k] - bt[k] for k in sorted(set(ct) & set(bt))]
        rng = random.Random(seed)
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
        m = sum(d) / len(d)
        dd = sum(cd[k] - bd[k] for k in cd) / len(cd)
        dt = sum(ct[k] - bt[k] for k in ct) / len(ct)
        out[name] = {"pooled": m, "ci95": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]), "dev": dd, "test": dt, "n": len(d)}
        print(f"| {name} | {100 * m:+.1f} [{100 * out[name]['ci95'][0]:+.1f}, {100 * out[name]['ci95'][1]:+.1f}] | {100 * dd:+.1f} | {100 * dt:+.1f} |")
    (RES / "pooled_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"contrast_seeds": contrast_seeds, "test8": test8, "pooled": pooled, "dev": dev, "test": test, "cos": cos, "decide": lambda: decide(*sys.argv[2:3]),
     "contrast": lambda: contrast(*sys.argv[2:4])}[sys.argv[1]]()
