"""HF-transformers utilities: load Qwen3.5 text model, capture residual activations, rank-one weight edits,
activation steering hooks, log-prob scoring, and export of edited checkpoints for vLLM.

Edit formula (the paper's p50c form):  W <- W - alpha * outer(u, mask * (u @ W))
  alpha = 1 removes the u-component written by the selected columns (directional ablation),
  alpha = 2 reflects it, alpha > 2 reverses and amplifies it (p50c used alpha = 6).
  A negative alpha amplifies the u-component instead.
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, model_path  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(name="qwen3.5-0.8b", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = model_path(name)
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=dtype).to(DEV).eval()
    return model, tok, p


def layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and not isinstance(getattr(m, attr), torch.nn.ModuleList):
            m = getattr(m, attr)
    if hasattr(m, "layers"):
        return m.layers
    return m.model.layers


def writers(layer):
    """Matrices that write into the residual stream in one decoder layer: {kind: nn.Linear}."""
    out = {"mlp": layer.mlp.down_proj}
    if hasattr(layer, "self_attn"):
        out["attn"] = layer.self_attn.o_proj
    elif hasattr(layer, "linear_attn"):
        out["attn"] = layer.linear_attn.out_proj
    return out


def param_name(model, module):
    for n, m in model.named_modules():
        if m is module:
            return n + ".weight"
    raise KeyError


def chat(tok, user, system=None, think=False, assistant=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    return prefix if assistant is None else (prefix, assistant)


@torch.no_grad()
def residual_means(model, tok, items, batch=16, max_len=512, which="response"):
    """items: list of (prefix, continuation). Returns tensor [n_layers+1, n_items, d] of residual-stream means
    over the continuation tokens (which='response'), or the last prefix token (which='last_prefix')."""
    out = []
    tok.padding_side = "right"
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ids, spans = [], []
        for pre, cont in chunk:
            a = tok(pre, add_special_tokens=False)["input_ids"]
            b = tok(cont, add_special_tokens=False)["input_ids"] if cont else []
            x = (a + b)[:max_len]
            ids.append(x)
            spans.append((len(a) - 1, len(a) - 1) if which == "last_prefix" else (len(a), min(len(x), len(a) + 64) - 1))
        L = max(map(len, ids))
        inp = torch.full((len(ids), L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros_like(inp)
        for j, x in enumerate(ids):
            inp[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        hs = model(input_ids=inp.to(DEV), attention_mask=att.to(DEV), output_hidden_states=True).hidden_states
        H = torch.stack(hs, 0).float()  # [L+1, B, T, d]
        for j, (s, e) in enumerate(spans):
            out.append(H[:, j, s:e + 1].mean(1).cpu())
    return torch.stack(out, 1)


def rank1_edit_(W: torch.Tensor, u: torch.Tensor, alpha: float, cols=None):
    """In place: W <- W - alpha * outer(u, mask*(u@W)). W: [d_model, d_in]; u: unit [d_model]."""
    Wf = W.float()
    u = (u / u.norm()).to(Wf)
    proj = u @ Wf                                   # [d_in]
    if cols is not None:
        mask = torch.zeros_like(proj)
        mask[torch.as_tensor(cols, device=proj.device)] = 1
        proj = proj * mask
    W.copy_((Wf - alpha * torch.outer(u, proj)).to(W.dtype))


class Steer:
    """Context manager: add vec to the residual stream output of the given decoder layers (all positions)."""

    def __init__(self, model, layer_ids, vec):
        self.h, self.model, self.ids, self.vec = [], model, layer_ids, vec

    def __enter__(self):
        L = layers(self.model)
        for i in self.ids:
            def hook(mod, inp, out, v=self.vec):
                if isinstance(out, tuple):
                    return (out[0] + v.to(out[0]),) + tuple(out[1:])
                return out + v.to(out)
            self.h.append(L[i].register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()


@torch.no_grad()
def cont_logprobs(model, tok, items, batch=32, max_len=1024):
    """items: list of (prefix, continuation). Returns list of (sum logprob, n_tokens) of continuation."""
    res = []
    tok.padding_side = "right"
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ids, spans = [], []
        for pre, cont in chunk:
            a = tok(pre, add_special_tokens=False)["input_ids"] if pre else []
            b = tok(cont, add_special_tokens=False)["input_ids"]
            x = (a + b)[-max_len:]
            ids.append(x)
            spans.append(len(x) - len(b))
        L = max(map(len, ids))
        inp = torch.full((len(ids), L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros_like(inp)
        for j, x in enumerate(ids):
            inp[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        # only project the positions we score onto the (248k) vocabulary: avoids B*T*V float logits
        hid = model.model(input_ids=inp.to(DEV), attention_mask=att.to(DEV)).last_hidden_state
        head = model.get_output_embeddings()
        for j, x in enumerate(ids):
            s = max(spans[j], 1)
            h = hid[j, s - 1:len(x) - 1]
            tgt = torch.tensor(x[s:], device=DEV)
            lp = torch.log_softmax(head(h).float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            res.append((lp.sum().item(), len(x) - s))
    return res


def export_ckpt(model, base_name, out_name, edited_param_names, info=None):
    """Write a vLLM-loadable checkpoint dir: symlink all base files, rewrite the safetensors shard with the
    edited tensors replaced (bf16)."""
    from safetensors.torch import load_file, save_file
    base = Path(model_path(base_name))
    out = MODELS / out_name
    out.mkdir(parents=True, exist_ok=True)
    shards = [f for f in base.iterdir() if f.name.endswith(".safetensors")]
    sd_model = dict(model.named_parameters())
    for f in base.iterdir():
        if not f.name.endswith(".safetensors"):
            dst = out / f.name
            if not dst.exists():
                os.symlink(os.path.realpath(f), dst)
    done = set()
    for sh in shards:
        t = load_file(str(sh))
        changed = False
        for pn in edited_param_names:
            key = pn
            if key not in t:
                cands = [k for k in t if k.endswith(re.sub(r"^model\.", "", pn).split("language_model.")[-1])]
                cands = [k for k in cands if "visual" not in k and "mtp" not in k]
                if len(cands) != 1:
                    continue
                key = cands[0]
            t[key] = sd_model[pn].detach().to(torch.bfloat16).cpu().contiguous()
            done.add(pn)
            changed = True
        if changed:
            save_file(t, str(out / sh.name), metadata={"format": "pt"})
        else:
            dst = out / sh.name
            if not dst.exists():
                os.symlink(os.path.realpath(sh), dst)
    missing = set(edited_param_names) - done
    assert not missing, missing
    if info is not None:
        (out / "EDIT_INFO.json").write_text(json.dumps(info, indent=1))
    return str(out)
