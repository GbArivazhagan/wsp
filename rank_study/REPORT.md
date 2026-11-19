# Edit-rank study: is the Swedish math gap a low-rank defect? (Qwen3.5-0.8B, GSM8K)

"We wrote the pre-registration before any training
(`preregistration.txt`) and logged every later change in `logs/decisions.log`.*

**Hypothesis (H_rank).** The Swedish gap is not a low-rank defect. The best rank-1 edit of the paper's edit family
(`mlp.down_proj`, layer 15), found by gradient descent with full supervision, recovers only a small part of what
higher-rank edits recover. A failure-type behaviour (response-language routing) is fixable at rank 1.

**Decision rule** (`preregistration.txt`, written 18:58, before any training).
- *Supported* if TEST gain(R1-single) ≤ ⅓ · gain(best high-rank) and the positive control gives ≥ 90 % Swedish answers.
- *Refuted* if gain(R1-single) ≥ ⅔ · gain(best high-rank) with ≥ 95 % Swedish answers.
- Otherwise inconclusive. Also inconclusive if the best high-rank gain's 95 % CI includes 0.
- Protocol: one shared lr, epochs and batch for all ranks; R1-single may also use 10× lr if that is better on DEV.

**Supervision and training.**
- RFT: the model's own correct Swedish solutions on TRAIN problems 0–199. There are 12 samples per problem
  (4 existing + 8 new) and at most 2 are kept per problem, giving 242 examples from 137 problems.
- Pilot gate: R16-all-linear gained +5.2 [0.0, 10.5] on DEV (`gsmtr_sv:dev`, 100 problems × 4 samples, base
  22.2 %). That is ≥ 3, so RFT was kept.
- A gold pilot (Swedish step-by-step solutions translated by Claude from `solution_en`) gained +2.0 on DEV and was
  not used.
- LoRA (peft): alpha = 2r, lr 2e-4, 3 epochs, batch 8, max length 1536, seed 0. The adapter is merged and exported
  as bf16; ≥ 99.9 % of ΔW survives the bf16 rounding.
- R1-single lr check: 2e-3 gave DEV +6.8, against +4.2 for 2e-4, so the 2e-3 edit is used in the decision. At
  2e-3 the LoRA A factor really moves away from its random start (overlap with its initial value 0.12).

**Results** (TEST: 200 problems × 4 samples, T 0.7, 4096 tokens; paired bootstrap 95 % CI).
- Δ₀ = gain vs the seed-0 base (25.9 %). Δ₈ = gain vs the pooled 8-sample base (24.1 %).
- Two base runs that differ only in sampling seed: seed 1 − seed 0 = −3.6 [−6.9, −0.4].

| condition | trainable params | ‖ΔW‖_F | train loss | sv acc | sv Δ₀ | sv Δ₈ | % sv answers | en Δ₀ |
|---|---|---|---|---|---|---|---|---|
| R1-single, lr 2e-4 | 4,608 | 0.48 | 0.437 | 28.7 | +2.9 [−0.6, 6.5] | +4.7 [1.4, 8.0] | 99.8 | +2.4 [−1.1, 5.9] |
| **R1-single, lr 2e-3 (DEV-selected)** | 4,608 | 1.58 | 0.424 | 30.4 | **+4.5 [0.9, 8.1]** | **+6.3 [2.8, 9.8]** | 100 | +0.8 [−2.9, 4.2] |
| — same, seed 1 | 4,608 | 1.63 | 0.424 | 27.9 | +2.0 [−1.9, 5.9] | +3.8 [0.2, 7.5] | 99.8 | +4.0 [0.4, 7.8] |
| R8-single | 36,864 | 0.66 | 0.426 | 28.1 | +2.2 [−1.5, 6.0] | +4.1 [0.2, 7.8] | 99.6 | +3.0 [−0.8, 6.6] |
| R64-single | 294,912 | 1.17 | 0.407 | 26.8 | +0.9 [−3.0, 4.8] | +2.7 [−0.7, 6.1] | 99.8 | +1.6 [−1.9, 4.9] |
| R64-single, lr 2e-3 (added, not pre-registered) | 294,912 | 9.12 | 0.349 | 22.9 | −3.0 [−6.9, 0.9] | −1.2 [−4.9, 2.5] | 100 | −3.5 [−7.6, 0.8] |
| R1-all-down (24 × rank 1) | 110,592 | 0.68 | 0.401 | 29.4 | +3.5 [−0.4, 7.1] | +5.3 [1.8, 8.7] | 98.8 | +3.0 [−0.5, 6.5] |
| R16-all-linear | 10,822,656 | 6.17 | 0.181 | 28.5 | +2.6 [−1.5, 6.8] | +4.4 [0.7, 8.2] | 99.8 | +1.0 [−2.9, 4.8] |
| — same, seed 1 | 10,822,656 | 6.17 | 0.182 | 31.4 | +5.5 [1.6, 9.1] | +7.3 [3.8, 10.8] | 99.8 | +1.2 [−2.5, 5.1] |

Seed-pooled (2 seeds each, joint bootstrap over problems):

| reference | R1-single gain | R16-all-linear gain | R1 / R16 | P(ratio ≤ ⅓) |
|---|---|---|---|---|
| seed-0 base | +3.2 | +4.1 | 0.80 [−0.03, 1.87] | 0.09 |
| 8-sample base | +5.1 | +5.9 | 0.86 [0.44, 1.42] | 0.009 |

**Positive control.**
- R1-single, trained on plain English prompts paired with the model's own Swedish answers.
- On English TEST problems with no instruction: 98.0 % Swedish answers (base 0 %).
- English accuracy 42.4 % (−14.2 [−18.5, −10.0]). This matches the earlier prompting result of 35.1 % for English
  problems reasoned in Swedish, so routing is fixable at rank 1.

**What the gains are.**
- Truncated (length-cut) Swedish answers: base 7.9 %, R1-single 3.8 % (seed 0) and 6.0 % (seed 1).
- Accuracy on finished answers: base 28.1 %, R1-single 31.6 / 29.7 %, R16-all-linear 30.4 / 33.9 %.
- Held-out NLL of the base model's DEV solutions:
  - Every single-matrix edit lowers NLL by about the same amount for correct and incorrect solutions
    (sharpening). The correct-vs-incorrect margin stays at 0.144–0.146.
  - Only R16-all-linear widens the margin, to 0.165–0.167.
- Greedy decoding of the unedited model gives +6.6 [0.4, 13.0].

**Cosine analysis** (layer-15 output direction u1 of ΔW; for random unit vectors, mean |cos| is 0.025 and the
99th percentile is 0.085).
- R1-single u1 vs our activation-contrast directions: derail 0.18, rlang 0.015, inalign 0.03.
- R1-single u1 vs the paper's loop direction (p50c): 0.02.
- The direction is a stable optimum: seed 0 vs seed 1 gives 0.977.
- R8 and R64 put their top singular direction on the same vector (0.94–0.96). R64's ΔW has 55 % of its energy
  in σ1.
- The routing control's u1 shares 0.55 with the math direction but only 0.08 with rlang, so even the routing fix
  does not use the activation-contrast language direction.

**Verdict.**
- Formally, under the pre-registered seed-0 reference: *inconclusive*, because the best high-rank CI includes 0.
  Against the pooled 8-sample base: *refuted* (ratio 1.19, 100 % Swedish answers). Seed-pooled, rank 1 recovers
  about 80–86 % of the LoRA-16-all gain.
- Taken together, H_rank is not supported. For this supervision a single rank-1 edit of the paper's family is about
  as good as 10.8M LoRA parameters. Extra rank only memorizes: train loss falls to 0.18 vs 0.42 without a TEST
  gain.
- So the earlier failure of WSP-style edits was a **search** problem: the activation-contrast directions are about
  orthogonal to the edit gradient descent finds.
- But what can be fixed is small and failure-type: fewer loops and truncations, plus sharpening, about what greedy
  decoding gives. About 25 points of the gap stay open at every rank.
- **Implication for SE.LLMA:** WSP-style rank-1 edits are plausible cheap patches for failure modes (loops, answer
  language). They are not a route to Swedish reasoning ability. With about 240 self-generated examples, the binding
  constraint is supervision, not parameter budget.

**Limitations.**
- Single-run 95 % CIs are about ±3.7 points, and the training-seed effect is about 3 points; only R1-single and
  R16-all-linear have 2 seeds.
- R1-single's lr was chosen on DEV; the higher-rank conditions got no lower-lr or fewer-epoch variant. R64 at 10×
  lr overfits.
- RFT data is small and skewed toward easier problems (63 of 200 TRAIN problems had no correct sample). DEV is
  noisy: 22.2 % here vs 29.0 % in an earlier run.
- LoRA rank 1 finds a local optimum (though stable across seeds). Only layer 15 was tested for single-matrix
  edits. "All-linear" means all 186 linear layers of the 24 blocks.
- Held-out NLL uses base-model samples. Gold supervision was piloted only.
- The positive-control model was not evaluated on Swedish problems: that DEV job ran out of GPU memory on a 12 GB GPU.

**Files.**
- Scripts: `rank_study/src/` (`lora_rank.py`, `analyze.py`, `heldout_nll.py`, `build_gold.py`, `add_jobs.py`)
- Job queue and runners: `rank_study/queue.txt`, `rank_study/run_queue*.sh`
- All generations and summaries: `rank_study/results/*.jsonl` and `*_summary.json`; every table in
  `rank_study/results/tables.txt` (rebuild with `rank_study/make_tables.sh`)
- The timestamped decisions log: `rank_study/logs/decisions.log`
- Kept checkpoints: `models/rank_r1d15` and `models/rank_r1d15x10`; all others were deleted after evaluation.
