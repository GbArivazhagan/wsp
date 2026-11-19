"""Markdown tables for the report, computed from result files (paired bootstrap CIs over the 200 TEST problems).
python src/make_tables.py > report/tables.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_jsonl  # noqa: E402
from summarize import paired, per_problem_acc  # noqa: E402

X, D, T, LP = RESULTS / "x2", RESULTS / "expD", RESULTS / "mtest", RESULTS / "logprob"


def ok(p):
    try:
        return Path(p).exists() and bool(load_jsonl(p))
    except Exception:
        return False


def lang_share(p, lang):
    rows = [r for r in load_jsonl(p) if r["lang"] == lang]
    return sum(r["resp_lang"] == lang for r in rows) / max(1, len(rows))


def row(label, ref, path, lang, note=""):
    if not (ok(ref) and ok(path)):
        return None
    d = paired(ref, path, lang)
    return (f"| {label} | {lang} | {100 * d['acc_b']:.1f} | {100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] | "
            f"{100 * lang_share(path, lang):.0f} % | {note} |")


def t_inference():
    print("\n**Table 1. Inference-time conditions, unedited model (TEST, 200 problems × 4 samples; Δ vs the same problems in the default setting).**\n")
    print("| condition | problem | acc % | Δ vs default [95% CI] | answers in problem language | note |")
    print("|----------------------------------|-----|-----|---------------|---------|--------------|")
    bs, be = X / "base_P-sv_R-none.jsonl", X / "base_P-en_R-none.jsonl"
    for r in [
        row("default (non-thinking, presence 1.5)", bs, bs, "sv"),
        row("default (non-thinking, presence 1.5)", be, be, "en"),
        row("default, other sampling seed (null)", bs, D / "gsm_nothink_n4_base.jsonl", "sv", "same model and settings"),
        row("default, other sampling seed (null)", be, D / "gsm_nothink_n4_base.jsonl", "en", "same model and settings"),
        row("presence penalty 0", bs, X / "base_P-sv_R-none_pp0.jsonl", "sv"),
        row("presence penalty 0", be, X / "base_P-en_R-none_pp0.jsonl", "en"),
        row("greedy (1 sample)", bs, X / "base_P-sv_R-none_greedy.jsonl", "sv"),
        row("greedy (1 sample)", be, X / "base_P-en_R-none_greedy.jsonl", "en"),
        row("'reason in Swedish' (user turn)", be, X / "base_P-en_R-svU.jsonl", "en", "answers switch to Swedish"),
        row("'reason in English' (user turn)", bs, X / "base_P-sv_R-enU.jsonl", "sv", "only 14 % switch"),
        row("English answer prefill", bs, X / "base_P-sv_R-enPrefill.jsonl", "sv", "forces English reasoning"),
        row("self-translate, then solve in English", bs, X / "base_P-sv_selftranslate.jsonl", "sv", "no weight change"),
    ]:
        if r:
            print(r)


def t_paper():
    print("\n**Table 2. The paper's p50c edit (loop removal) and its norm-matched random twin, on Swedish and English GSM8K (TEST).**\n")
    print("| model / mode | problem | acc % | Δ vs unedited same mode [95% CI] | answers in problem language | note |")
    print("|----------------------------------|-----|-----|---------------|---------|--------------|")
    for mode, base, files in [("non-thinking (4 samples)", D / "gsm_nothink_n4_base.jsonl",
                               [("p50c", D / "gsm_nothink_n4_p50.jsonl"), ("random twin", D / "gsm_nothink_n4_rnd.jsonl")]),
                              ("thinking, 16k cap (1 sample)", D / "gsm_think16k_base.jsonl",
                               [("p50c", D / "gsm_think16k_p50.jsonl"), ("random twin", D / "gsm_think16k_rnd.jsonl"),
                                ("budget forcing at 4k thinking tokens (no weight change)", D / "gsm_think16k_base_bf4096.jsonl"),
                                ("budget forcing at 8k thinking tokens (no weight change)", D / "gsm_think16k_base_bf8192.jsonl")])]:
        for lang in ("sv", "en"):
            r = row(f"unedited, {mode}", base, base, lang)
            if r:
                print(r)
            for lab, f in files:
                note = "share counts cut-off outputs (English thinking only)" if "thinking," in mode and lang == "sv" else ""
                r = row(f"{lab}, {mode}", base, f, lang, note)
                if r:
                    print(r)


def t_edits():
    sel = RESULTS / "mtest" / "selected.json"
    if not sel.exists():
        return
    sel = json.loads(sel.read_text())
    print("\n**Table 3. DEV-selected WSP-style edits for the Swedish gap and their norm-matched random twins (TEST; Δ vs unedited).**\n")
    print("| model | problem | acc % | Δ vs unedited [95% CI] | answers in problem language | note |")
    print("|----------------------------------|-----|-----|---------------|---------|--------------|")
    bs, be = X / "base_P-sv_R-none.jsonl", X / "base_P-en_R-none.jsonl"
    items = [(lab, T / f"{n}.jsonl", T / f"{n.replace('msw_', 'mrnd_')}.jsonl", n) for lab, n in sel.items()]
    for lab, f, frnd, n in items:
        for lang, b in (("sv", bs), ("en", be)):
            for l2, ff in ((lab + f" ({n})", f), (lab + " — random twin", frnd)):
                r = row(l2, b, ff, lang)
                if r:
                    print(r)
    for lang, b in (("sv", bs), ("en", be)):
        for l2, ff in (("exploratory: rlang L15-19 α=6 (not eligible: 0 % Swedish answers on DEV)", T / "msw_rlang_L15-19_a6.jsonl"),
                       ("exploratory — random twin", T / "mrnd_rlang_L15-19_a6.jsonl")):
            r = row(l2, b, ff, lang)
            if r:
                print(r)


RANK = [("r1d15", "rank 1, down_proj L15, lr 2e-4"), ("r1d15x10", "rank 1, down_proj L15, lr 2e-3 (DEV-selected R1)"),
        ("r1d15x10s1", "rank 1, down_proj L15, lr 2e-3, training seed 1"), ("r8d15", "rank 8, down_proj L15"),
        ("r64d15", "rank 64, down_proj L15"), ("r64d15x10", "rank 64, down_proj L15, lr 2e-3"),
        ("r1alldown", "rank 1, every down_proj"), ("r16all", "rank 16, all linear layers (RFT-LoRA)"),
        ("r16alls1", "rank 16, all linear layers, training seed 1")]


def t_rank():
    R = RESULTS.parent / "rank_study" / "results"
    if not R.exists():
        return
    print("\n**Table 6. Edit-rank study: gradient-trained edits on the same RFT data (own correct Swedish TRAIN solutions); TEST, Δ vs unedited.**\n")
    print("| edit | problem | acc % | Δ vs unedited [95% CI] | answers in problem language | note |")
    print("|----------------------------------|-----|-----|---------------|---------|--------------|")
    bs, be = X / "base_P-sv_R-none.jsonl", X / "base_P-en_R-none.jsonl"
    for k, lab in RANK:
        for lang, b in (("sv", bs), ("en", be)):
            r = row(lab, b, R / f"test_{k}.jsonl", lang)
            if r:
                print(r)
    r = row("positive control: rank 1, down_proj L15, trained to answer English problems in Swedish", be,
            R / "test_pc_r1.jsonl", "en", "share column = answers in English")
    if r:
        print(r)


def t_logprob():
    files = sorted(LP.glob("*.json"))
    if not files:
        return
    print("\n**Table 4. Regression checks (log-prob): Belebele accuracy, FLORES perplexity (nats/token), en/ett agreement on unseen nouns.**\n")
    print("| model | Belebele sv | Belebele en | ppl sv | ppl en | en/ett acc | en/ett AUC |")
    print("|--------------------------|-------|-------|-------|-------|-------|-------|")
    for f in files:
        d = json.loads(f.read_text())
        g = d.get("gender_test", {})
        fmt = lambda k: f"{d[k]:.3f}" if k in d else "–"
        print(f"| {f.stem} | {fmt('belebele_sv')} | {fmt('belebele_en')} | {fmt('ppl_sv')} | {fmt('ppl_en')} | "
              f"{g.get('acc', float('nan')):.3f} | {g.get('auc', float('nan')):.3f} |")


def t_gender():
    p = RESULTS / "gender" / "sweep.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    b = json.loads((RESULTS / "gender" / "bootstrap.json").read_text()) if (RESULTS / "gender" / "bootstrap.json").exists() else {}
    print("\n**Table 5. Capability probe: Swedish en/ett agreement on 946 unseen TEST nouns (3 templates each).**\n")
    print("| condition | acc | AUC | ΔAUC vs base [95% CI] | note |")
    print("|----------------------------------------|------|------|------------------|----------------|")
    print(f"| unedited | {d['base']['test']['acc']:.3f} | {d['base']['test']['auc']:.3f} | – | best global-bias accuracy {d['base']['test']['acc_best_shift']:.3f} |")
    print(f"| few-shot prompt | {d['fewshot']['test']['acc']:.3f} | {d['fewshot']['test']['auc']:.3f} | – | 8 article–noun examples |")
    bd = d["best_dev"]
    for k, lab in (("acts", f"edit, activation direction (DEV-best: {bd['layers']}, α={bd['alpha']}; {d['n_feedback_evals']} DEV evals)"),
                   ("random", "same edit, random direction"), ("weights", "same edit, weights-only direction (embedding en−ett)"),
                   ("weights_apriori_p50c_a-2", "weights-only, a-priori config (no feedback)")):
        t = d["test"][k]["test"]
        bb = b.get({"acts": "qwen35_gender_best_acts", "random": "qwen35_gender_best_random"}.get(k, ""), {})
        ci = f"{bb['d_auc']:+.4f} [{bb['d_auc_ci95'][0]:+.4f}, {bb['d_auc_ci95'][1]:+.4f}]" if bb else "–"
        print(f"| {lab} | {t['acc']:.3f} | {t['auc']:.3f} | {ci} | |")
    lp = LP / "qwen35_gender_lora.json"
    if lp.exists():
        g = json.loads(lp.read_text())["gender_test"]
        print(f"| LoRA fine-tuned on 1,901 TRAIN nouns | {g['acc']:.3f} | {g['auc']:.3f} | – | gradient baseline |")


if __name__ == "__main__":
    t_inference()
    t_paper()
    t_edits()
    t_rank()
    t_logprob()
    t_gender()
