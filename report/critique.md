# Part 1 — Critical reading of "Sparks of Weight-Space Programming" (Takehi et al., Nov 2026)

Sources: the blog post and its released logs (`github.com/RikiyaT/lm_understand-logs`, cloned to `data/wsp_logs/`).
We rebuilt the Experiment 8 edits bit-for-bit from the public `Qwen/Qwen3.5-0.8B` (rev `2fc0636`) with the
authors' `apply_edit.py` (every tensor hash matches), so the statements about the edit below are checked, not inferred.

## Strongest evidence

1. **Experiment 3 (copying heads from middle blocks, no forward passes).** Ground truth is measured independently,
   the model is forbidden to run the network, and its head-level answers match exactly in all three models. This
   is real weight-reading. Caveats: it rediscovers a published procedure (QK/OV composition, Elhage et al. 2021) on
   a 4-layer, 7M-parameter character model, and the language guess has ~0.5–0.6 confidence among 3 given options.
2. **Experiment 5 (MinAtar policies written with zero environment interaction).** No data, rollouts or tools, and
   the networks still beat PPO-100M. This is a clean existence proof that an LLM can *compile* a hand-designed
   policy (e.g., "predict where the ball lands, move the paddle there") into a small MLP. It shows weight *writing*
   from a program, not reading of learned weights.
3. **Transparency.** Every turn, tool call, edit and evaluation is released, including the runs that failed.
   That is what makes the critique below possible.

## Most important limitations

1. **The headline LLM result (Exp 8) comes from outcome feedback, not from reasoning about weights.** The authors'
   own results table (upstream `exp8_math/eval/RESULTS_excerpt.md`; not copied into `data/wsp_logs/`) lists runs the post omits:

   | Qwen3.5-0.8B, thinking on | MATH-500 | GSM8K |
   |---|---|---|
   | 0 passes (prompt v2) | 7.9 | 13.8 — *no edit made* |
   | 5 passes | 7.9 | 13.8 — *no edit submitted* |
   | 10 passes, prompt v4 | 7.9 | 13.8 — *no edit submitted* |
   | 10 / 25 / 50 passes (prompt v3, in the post) | 12.3 / 14.1 / **48.1** | 25.7 / 25.1 / **67.1** |

   With zero forward passes (pure weight reading), nothing happened. The large gain comes from one run with
   50 passes. One pass is up to 512 full generations at 32,768 tokens, so that run had up to ~25,600 graded
   rollouts, paired original-vs-edited comparisons, and 20 wall-clock hours. Its own notebook
   (`data/wsp_logs/exp8_math/runs/p50c/work_notes/experiment_history.md`) reads as a classic hill-climb:
   - ablate one neuron (+6 pp);
   - remove the direction with α=1 (+24 pp);
   - α=2 (+25 pp);
   - top-64 columns with α=6 (+44 pp), each step judged on fresh practice generations.

   This is LLM-guided black-box optimisation with a validation set, and should be compared with other
   optimisers given the same evaluation budget.
2. **The mechanism is standard activation-based directional editing.** The direction `u` is a contrast of
   residual activations on "productive" vs "repetitive" teacher-forced contexts (passes 2–3), i.e. diff-in-means
   steering / directional ablation (Turner et al. 2023; Arditi et al. 2024), baked into weights. We verified the
   final edit is exactly `W ← W − 6·u(uᵀW)` on 64 columns of `mlp.down_proj` in layers 15 and 19. Per layer it is
   rank 1 (σ₂/σ₁ ≈ 6·10⁻⁴), with α recovered as 6.000. The "0.016 % of weights" framing understates it: the two
   matrices change by 10.6 % and 8.1 % in Frobenius norm, and the component along `u` is reversed and amplified 5×.
3. **The baseline is a non-default, known-broken inference mode.** Qwen3.5-0.8B runs non-thinking by default (its
   chat template only opens a think block if `enable_thinking=True`). The authors' own profile file notes that with
   thinking on, "92 % of MATH-500 answers hit 32,768 tokens", and that the model card warns about loops. The base
   score of 7.9 % mostly measures truncation (93 % cut). The post doesn't report the default non-thinking model on
   the same benchmarks. The FAQ argues thinking-off "does not fix the weights", which is true. But for a
   practitioner, the right control for "better at math" is the cheapest configuration that avoids the failure.
   We measure it (Exp D).
4. **By the authors' own numbers, the improvement is failure removal, not new capability.** Astra's confirmation
   report: 576 of 579 rescued answers were originally truncated, and "aggregate accuracy on originally finished
   outputs was unchanged". On Gemma 3 270M, which rarely loops, no edit helped (6.8 → 7.0). The IFEval gain
   (21 → 62) is consistent with the same mechanism: IFEval is scored on the text after `</think>`, which a looping
   model never emits.
5. **Reading the weights of a real LM failed where it was tested cleanly.** In Exp 7 (sycophancy, Gemma 3 270M),
   4/4 weights-only attempts failed. Success needed attention patterns from forward passes on texts Astra chose.
6. **Small n, single runs, selection.** Exps 1–2 have n = 4 and n = 3 networks. In Exp 1 the game is plausibly
   leaked by where first-layer weights concentrate, which is a simple statistic. Exp 8 has one run per budget,
   and the trajectory is non-monotonic (IFEval 49.0 → 39.9 → 62.3). Prompt versions were iterated, and other
   versions are omitted from the post. The editor (GPT-6 Astra) is proprietary, so the editing itself can't be
   replicated; only its outputs can.

## Which conclusions are justified

- **Justified.** Frontier LLMs can:
  - apply known interpretability analyses to small networks' weights;
  - compile explicit algorithms into small MLPs without feedback;
  - given forward-pass access and a large evaluation budget, find a low-rank edit that removes a gross,
    mode-specific failure (looping) in a 0.8B model, with a sensible norm-matched random control.
- **Not justified.**
  - That LLMs "natively read" LLM-scale weights well enough to improve capabilities. The 0- and 5-pass runs
    produced nothing.
  - That the edit makes the model "better at math" or "generalizes to instruction following" beyond removing loops.
  - That this offers "zero-data capability steering" or "dramatically reduced optimization cost", since the
    successful run used ~25k long generations.
  - Any claim of an advantage over standard steering, directional ablation, or a small LoRA at equal budget,
    because none were compared.

## What this means for SE.LLMA (motivating our hypothesis)

The distinction that matters for a Swedish-model team is **failure vs capability**. If WSP-style edits mainly
remove failures that block an ability the model already has, they are a debugging tool, and should be compared
against prompting and decoding changes that remove the same failure. They won't substitute for Swedish data and
training, which is what SE.LLMA is built around.
