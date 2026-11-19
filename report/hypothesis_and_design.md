# Part 2–3 — Hypothesis and design (written 2026-11-18 ~16:15, before the main edit results)

## Observation that motivated the question (Exp 0 / D, non-thinking default mode)

Qwen3.5-0.8B, the paper's own Exp 8 model, on the same 200 GSM8K test problems:
- **57.5 %** when the problem is in English;
- **25.0 %** when it is in Swedish (our translation).

Its Swedish answers are always in Swedish (100 %, Exp 0 scan). Reading the failures shows the problems were
*understood*: relations are restated correctly. The reasoning then derails (e.g. "365 days in a week",
dropped steps). The math ability exists, but it is not used on Swedish input.

## Narrow question

> Is a small multilingual model's Swedish math gap a *removable failure* that a targeted, WSP-style weight edit
> can fix (as the paper's loop edit did), or a *capability gap*? How does such an edit compare with
> inference-time changes and with conventional fine-tuning?

**Why it matters for SE.LLMA.** SE.LLMA adapts existing open models to Swedish. If cross-lingual gaps were
failures that a direction-level edit could remove, a cheap post-hoc tool would exist. If they're capability
gaps, Swedish data and training are needed, and WSP-style editing is at best a diagnostic.

**Hypothesis H.** Targeted edits of the paper's form (rank-1, top-64 columns, direction from activation
contrasts, strength chosen with dev feedback) close only the part of the gap that is a routing failure. Any gain
they give is matched by an inference-time change, and they don't add Swedish reasoning ability.

**What would support H.**
1. The paper's own recipe ("derail" direction: wrong vs right Swedish solutions) gives no gain on held-out
   Swedish problems beyond a norm-matched random edit, as with the authors' Gemma follow-up.
2. Any edit that does raise Swedish accuracy does it by switching the reasoning language to English. The 2×2
   prompting cell "Swedish problem, English reasoning" achieves the same, so it's routing, not capability.
3. Conventional fine-tuning on the model's own correct Swedish solutions (RFT-LoRA) moves Swedish accuracy at
   least as much as any edit.

**What would challenge H.** An edit derived from TRAIN problems, selected on DEV problems, that raises
Swedish TEST accuracy:
- significantly above the random-edit control,
- while answers stay in Swedish,
- with no English, Belebele or perplexity regression,
- and beyond the best prompting cell.

That would be a genuine weight-space improvement in Swedish.

## Data separation

- **TRAIN:** 200 GSM8K-train problems, translated. Used for directions and RFT.
- **DEV:** 100 more GSM8K-train problems. Used for grid selection; every DEV evaluation is counted as feedback.
- **TEST:** 200 GSM8K-test problems, translated. Each condition is evaluated once, with 4 samples per problem
  and paired bootstrap CIs.

## Conditions on TEST

- Unchanged base, in default non-thinking mode and in thinking mode.
- Inference/prompting: the 2×2 of problem language × reasoning language, set via system prompt.
- The paper's p50c edit (loop removal), plus its norm-matched random control.
- WSP-style edits from TRAIN contrasts, each in its best DEV configuration:
  - **derail** (the paper's recipe);
  - **rlang** (Swedish vs English reasoning);
  - **inalign** (Swedish vs English problem text).
- A norm-matched random-direction control at the selected configuration.
- Conventional fine-tuning: RFT-LoRA on TRAIN.

## Regressions

- English GSM8K, on the same problems.
- Belebele zero-shot in sv, en, da and nb.
- Perplexity on FLORES passages, sv and en.
- Language of the answer.

## Side probe (capability control)

Swedish en/ett gender agreement on unseen nouns. The base model is already optimally calibrated (accuracy
equals the best accuracy reachable by a global bias shift), so any gain must come from per-noun knowledge
(AUC). We test the same edit family, a random control, few-shot prompting, and LoRA on train nouns.

## Selection rule (pre-registered 2026-11-18 17:55, before the DEV grid was complete)

For each edit family (derail, rlang, inalign), the config sent to TEST is the one with the highest Swedish DEV
accuracy among configs that keep ≥ 95 % of Swedish answers in Swedish. Ties go to the higher English DEV
accuracy. The selected config goes to TEST **whether or not it beats the unedited model on DEV**, so the TEST
number also shows what selection on a 100-problem DEV set is worth. Each selected edit is paired with a
norm-matched random-direction twin (same layers, same columns, same α, same ‖ΔW‖_F).

Qwen3.5 ignores a *system* prompt asking for English when the problem is Swedish (100 % Swedish answers).
  We added reasoning-language cells with the instruction in the *user* turn, an English answer prefill, and a
  self-translate pipeline.
