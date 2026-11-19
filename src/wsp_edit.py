"""WSP-style targeted weight edits (the paper's p50c form) for two Swedish targets.

  W <- W - alpha * outer(u, mask_k * (u @ W))       on residual writers (mlp.down_proj, attn out) of chosen layers

u is the unit 'bad' direction (what we want the model to write less of):
  task=lang   : u = English-response minus Swedish-response direction
  task=gender : u = (common-article minus neuter-article) direction at the noun-prediction position; alpha < 0
                *amplifies* the article's gender signal instead (see --sign)
Where u comes from (the 'information ladder', central to the paper's claim):
  --source weights : zero forward passes. u from the tied embedding / unembedding rows of language-specific
                     tokens (lang) or of the article tokens (gender). Columns chosen by |u . W_col| alone.
  --source acts    : teacher-forced forward passes on DEV contrast pairs (no outcome feedback). u = diff in
                     means of the residual stream at the layer output; columns chosen by mean|act_j| * |u.W_col|.
  --source random  : a random unit direction (seeded), columns chosen like 'weights'. Untargeted control.
Outcome feedback (choosing layers / alpha / k by looking at generations on DEV prompts) is done by
src/select_edit.py, which counts every generation pass it uses.

Writes models/<out>/ (vLLM-loadable) and models/<out>/EDIT_INFO.json.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, load_jsonl, prompt_split  # noqa: E402
import hf_tools as H  # noqa: E402

SV_WORDS = ("och att det som är på för med inte har ett till kan eller också när hur vad vilka därför mycket "
            "från under efter mellan utan bara något några andra sedan många skulle måste kunde både även här där "
            "detta dessa vår deras hennes honom själv varför vilken alltid aldrig ofta kanske eftersom medan").split()
EN_WORDS = ("and that the which is on for with not has a to can or also when how what which therefore very "
            "from under after between without only something some other then many would must could both even here "
            "there this these our their her him self why which always never often maybe because while").split()


def token_ids(tok, words):
    ids = set()
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            t = tok(v, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(t[0])
    return sorted(ids)


def direction_weights(model, tok, task):
    E = model.get_output_embeddings().weight.float()
    if task == "lang":
        sv, en = token_ids(tok, SV_WORDS), token_ids(tok, EN_WORDS)
        sv, en = sorted(set(sv) - set(en)), sorted(set(en) - set(sv))
        u = E[en].mean(0) - E[sv].mean(0)
        meta = {"n_sv_tokens": len(sv), "n_en_tokens": len(en)}
    else:
        en_t, ett_t = token_ids(tok, ["en"]), token_ids(tok, ["ett"])
        u = E[en_t].mean(0) - E[ett_t].mean(0)
        meta = {"en_tokens": en_t, "ett_tokens": ett_t}
    return u / u.norm(), meta


def contrast_items(tok, task, model_name, n_max=400):
    """(prefix, continuation) pairs for the two sides of the contrast, from DEV data only."""
    if task == "lang":
        gen = load_jsonl(DATA / "contrast" / f"{model_name}_lang_contrast.jsonl")
        A = [(H.chat(tok, r["prompt"]), r["resp_en"]) for r in gen][:n_max]   # bad side: English answer
        B = [(H.chat(tok, r["prompt"]), r["resp_sv"]) for r in gen][:n_max]   # good side: Swedish answer
        return A, B, "response"
    # Real contexts from UD Swedish-Talbanken TRAIN sentences: the prefix up to and including an indefinite
    # article 'en' / 'ett' (DET) whose head is a noun. A = common-article contexts, B = neuter-article contexts.
    A, B = [], []
    sent = []
    for line in open(DATA / "ud" / "sv_talbanken-ud-train.conllu", encoding="utf8"):
        if line.startswith("#"):
            continue
        if not line.strip():
            toks = [c for c in sent if c[0].isdigit()]
            for k, c in enumerate(toks):
                if c[3] == "DET" and c[1].lower() in ("en", "ett") and toks[int(c[6]) - 1][3] == "NOUN" if c[6].isdigit() and int(c[6]) > 0 else False:
                    prefix = " ".join(t[1] for t in toks[:k + 1])
                    (A if c[1].lower() == "en" else B).append((prefix, None))
            sent = []
            continue
        sent.append(line.rstrip("\n").split("\t"))
    import random as _r
    _r.Random(0).shuffle(A)
    _r.Random(1).shuffle(B)
    n = min(len(A), len(B), n_max)
    return A[:n], B[:n], "last_prefix"


@torch.no_grad()
def mlp_act_means(model, tok, items, layer_ids, which):
    """Mean |activation| of each MLP hidden unit (input of down_proj) over the contrast positions."""
    L = H.layers(model)
    store = {i: [] for i in layer_ids}
    hooks = []
    for i in layer_ids:
        def pre(mod, inp, i=i):
            store[i].append(inp[0].detach().abs().float().mean(dim=(0, 1)).cpu())
        hooks.append(L[i].mlp.down_proj.register_forward_pre_hook(pre))
    H.residual_means(model, tok, items, which=which)
    for h in hooks:
        h.remove()
    return {i: torch.stack(v).mean(0) for i, v in store.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--task", choices=["lang", "gender"], required=True)
    ap.add_argument("--source", choices=["weights", "acts", "random"], required=True)
    ap.add_argument("--layers", default="15,19")
    ap.add_argument("--dir_layer", type=int, default=-1, help="acts: layer output used for u (default: each edited layer)")
    ap.add_argument("--writers", default="mlp", help="mlp | attn | mlp,attn")
    ap.add_argument("--topk", type=int, default=64, help="columns per matrix; 0 = all columns")
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_dir_only", action="store_true", help="only compute and save u (no checkpoint)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    model, tok, p = H.load(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lids = [int(x) for x in a.layers.split(",")]
    info = {k: v for k, v in vars(a).items()}
    n_fwd_items = 0
    us, actw = {}, {}
    if a.source == "weights":
        u, meta = direction_weights(model, tok, a.task)
        us = {i: u for i in lids}
        info["dir_meta"] = meta
    elif a.source == "random":
        d = model.get_output_embeddings().weight.shape[1]
        u = torch.randn(d, generator=torch.Generator().manual_seed(a.seed))
        us = {i: (u / u.norm()).to(H.DEV) for i in lids}
    else:
        A, B, which = contrast_items(tok, a.task, a.model)
        n_fwd_items = len(A) + len(B)
        RA = H.residual_means(model, tok, A, which=which)   # [L+1, n, d]
        RB = H.residual_means(model, tok, B, which=which)
        for i in lids:
            j = (a.dir_layer if a.dir_layer >= 0 else i) + 1    # hidden_states[i+1] = output of layer i
            u = RA[j].mean(0) - RB[j].mean(0)
            us[i] = (u / u.norm()).to(H.DEV)
            # how separable are the two sides along u (held-in, descriptive)
            pa, pb = RA[j] @ us[i].cpu(), RB[j] @ us[i].cpu()
            info.setdefault("dir_sep_dprime", {})[i] = float((pa.mean() - pb.mean()) / torch.cat([pa, pb]).std())
        actw = mlp_act_means(model, tok, A + B, lids, which) if "mlp" in a.writers else {}
    torch.save({i: v.cpu() for i, v in us.items()}, MODELS / f"{a.out}_u.pt")
    if a.save_dir_only:
        print(json.dumps(info, indent=1, default=str))
        return
    edited, changed = [], 0
    for i in lids:
        for kind, lin in H.writers(H.layers(model)[i]).items():
            if kind not in a.writers.split(","):
                continue
            W = lin.weight.data
            score = (us[i].float() @ W.float()).abs()
            if kind == "mlp" and i in actw:
                score = score * actw[i].to(score)
            cols = None if a.topk == 0 else torch.topk(score, a.topk).indices.cpu()
            H.rank1_edit_(W, us[i], a.alpha, cols)
            edited.append(H.param_name(model, lin))
            changed += W.shape[0] * (W.shape[1] if cols is None else len(cols))
            info.setdefault("columns", {})[f"{i}.{kind}"] = None if cols is None else cols.tolist()
    info.update(edited=edited, changed_weights=changed, forward_items_used=n_fwd_items,
                total_params=sum(x.numel() for x in model.parameters()))
    out = H.export_ckpt(model, a.model, a.out, edited, info)
    print(out, "changed", changed, "weights;", "forward items", n_fwd_items)


if __name__ == "__main__":
    main()
