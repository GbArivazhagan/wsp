# Detailed report

This part gives the details behind the research note: setup, every result with its controls, examples, and the
exact commands. The generated tables are in Appendix A (`tables.md`), the paper critique in Appendix C, and the
three pre-registrations in Appendix D. Every number below comes from a file in `results/`, `numword/results/`,
`swthink/results/` or `rank_study/results/`.

## 1. Setup shared by all three studies

- **Model.** `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17` (Apache-2.0). It is the model
  of the paper's Experiment 8.
  - Architecture: hybrid linear attention, with full attention in every 4th layer, hidden size 1024, 24 layers.
  - Non-thinking is the chat template's default. Thinking needs `enable_thinking=True`.
- **Decoding.**
  - Default (non-thinking) mode: T 0.7, top-p 0.8, top-k 20, presence penalty 1.5, at most 4,096 tokens, 4 samples
    per problem (8 on the 59 number-word problems of H2).
  - Thinking mode: T 0.6, top-p 0.95, top-k 20, presence penalty 1.5, at most 16,384 tokens, 1 sample per problem
    per seed.
  - The seed is 0 unless a `.config.json` next to the result says otherwise. Every result file has one, recording
    the model path and revision, the decoding settings and the seed.
- **Data.**
  - *Swedish GSM8K.* An LLM (Claude) translated GSM8K into Swedish, and a script checked that every number in the
    English problem also appears in the Swedish one (`data/gsm8k/`).
  - TRAIN: 200 GSM8K-train problems, used to build edits and as training data.
  - DEV: 100 more train problems, used to select edits. Every DEV evaluation counts as feedback.
  - TEST: a random sample of 200 GSM8K-test problems (numpy seed 0). TEST is never used to build or select an
    edit, and each condition is run on it once. The one exception is p50c in the default mode: it was first run
    with 1 sample and then with 4 (§8). TEST outputs did motivate H2 and H3 (§3).
  - H2 also uses synthetic template problems (480 train, 200 held-out) that share no text with GSM8K.
- **Grading.** Exact match on the final number, handling Swedish decimal commas and thousands separators
  (`src/grade_math.py`); math-verify for MATH-500. No LLM was used to grade.
- **Statistics.**
  - Paired bootstrap 95 % CIs over problems (5,000 resamples, seed 0).
  - **Noise floor:** the unedited model re-run with only the sampling seed changed scores 22.2 % instead of 25.9 %
    on Swedish TEST, Δ −3.6 [−6.9, −0.4]. The bootstrap over problems therefore understates run-to-run noise, so we
    read single-run differences under ~4 points as noise even when the CI excludes 0. Dozens of TEST comparisons
    are reported without multiplicity correction.
- **Pre-registration.** Each hypothesis has a design file written before its runs, and later changes are logged
  in it. The files are `report/hypothesis_and_design.md` (H1), `numword/preregistration.md` (H2),
  `swthink/preregistration.md` (H3) and `rank_study/preregistration.txt` (the edit-rank side study).
- **Compute.** Four single-GPU workstations: three with 12 GB cards and one RTX 3090. vLLM 0.17.0 was used for
  generation and transformers 5.20 / peft 0.21 / torch 2.10 for edits and training. No API was used for the model under test.

## 2. H1: do edits of the paper's form close the Swedish gap?

> **H1.** WSP-style targeted edits (rank-1, top-64 columns, directions from activation contrasts, strength chosen
> with DEV feedback) remove *failures* but do not add *capability*. On the Swedish gap they won't beat a
> norm-matched random edit on held-out problems, and any real gain will be matched by an inference-time change.

*Would challenge H1:* a TRAIN-derived, DEV-selected edit that raises Swedish TEST accuracy above its random twin,
keeps answers in Swedish, causes no English, Belebele or perplexity regression, and beats the best prompting cell.

Our edits sit between the paper's two modes. Their directions come from activation contrasts (forward passes, no
outcome feedback), and their strength is chosen with DEV feedback. Neither is pure weight reading. 

**Edits.** Each edit has the paper's form, `W ← W − α·u(uᵀW)` on the top-64 `down_proj` columns of the chosen layers.
The direction `u` comes from one of three activation contrasts:
- *derail* (the paper's recipe): wrong vs right Swedish solutions;
- *rlang*: Swedish vs English reasoning on the same Swedish problem;
- *inalign*: Swedish vs English problem text.

The grid is 3 kinds × 3 layer sets × α ∈ {1, 2, 6}, giving 27 DEV evaluations. A pre-registered rule picked one
configuration per kind: ≥ 95 % Swedish answers, then the highest Swedish DEV accuracy. The chosen configuration went
to TEST whether or not it beat the unedited model on DEV. Each selected edit has a **norm-matched random twin**: the
same columns and the same ‖ΔW‖, with a random direction.

### 2.1 Default (non-thinking) mode

| DEV-selected edit (TEST, 200 × 4) | Swedish Δ | its random twin | English Δ |
|---|---|---|---|
| derail, L15–19, α = 1 (paper's recipe) | −3.1 [−6.6, +0.2] | −3.1 [−6.1, −0.1] | −0.8 [−3.8, +2.2] |
| rlang, L10–14, α = 2 | −0.4 [−3.9, +3.1] | −0.6 [−3.8, +2.6] | +1.9 [−1.4, +5.2] |
| inalign, L6–18 (every 4th), α = 1 | −2.0 [−5.5, +1.4] | −1.9 [−5.0, +1.2] | +0.6 [−2.8, +3.8] |
| the paper's p50c edit | +0.0 [−3.5, +3.5] | (twin +1.9) | +0.0 [−3.4, +3.4] |

- None of these edits moves Belebele or FLORES perplexity by more than 0.05 (Table 4).
- The one edit that raised Swedish accuracy was excluded by the rule: rlang, L15–19, α = 6. Its TEST run was an
  exploratory addition, not in the pre-registration.
  - It gains +6.1 [+1.9, +10.6], against −2.2 for its twin.
  - But 99.9 % of its answers are in English, so it is the "English answer prefill" cell (+9.9) moved into the
    weights.
  - It damages Swedish: FLORES perplexity 3.47 → 8.29, en/ett AUC 0.87 → 0.53 (chance), Belebele 0.357 → 0.332.
- On MATH-500 in the default mode, the unedited model already scores **45.6 %** and p50c scores 42.6 %. The paper
  reports 7.9 % for the unedited model and 48.1 % after its edit, both in thinking mode.

### 2.2 Can search not find the edit, or does no good rank-1 edit exist? (edit-rank study)

A separate study trained the *best* rank-1 edit of
the same matrix by gradient descent, with full supervision: rejection-sampling fine-tuning on 242 of the model's own
correct Swedish TRAIN solutions. It compared that edit with higher ranks trained on the same data.

| edit trained on the same data (TEST) | trainable params | Swedish Δ, training seed 0 / 1 | English Δ |
|---|---|---|---|
| rank 1, `down_proj` L15 | 4.6 k | +4.5 [+0.9, +8.1] / +2.0 [−1.9, +5.9] | +0.8 |
| rank 8, `down_proj` L15 | 37 k | +2.2 [−1.5, +6.0] | +3.0 |
| rank 64, `down_proj` L15 | 295 k | +0.9 [−3.0, +4.8] | +1.6 |
| rank 1 in every `down_proj` | 111 k | +3.5 [−0.4, +7.1] | +3.0 |
| LoRA r16, all linear layers | 10.8 M | +2.6 [−1.5, +6.8] / +5.5 [+1.6, +9.1] | +1.0 |
| *positive control:* rank 1 trained to answer English problems in Swedish | 4.6 k | – | 98 % Swedish answers |

- **Rank doesn't matter.** Pooled over seeds, rank 1 recovers 0.86 [0.44, 1.42] of the LoRA gain against the
  pooled 8-sample base. Against the pre-registered seed-0 base the ratio is 0.80 [−0.03, 1.87], which is formally
  inconclusive. Either way, the study's own hypothesis (rank 1 recovers at most ⅓) is not supported. Higher rank
  only memorises: training loss falls to 0.18, with no TEST gain.
- **The gain is small.** Against the seed-0 base the ranks gain +1 to +6 points, about what greedy decoding gives
  (+6.6). Part of the gain comes from fewer truncated answers (7.9 % → 3.8 %). The gradient-trained edits have no
  random twin.
- **Reading activations misses this edit.** The gradient direction is stable across seeds (cosine 0.98). It is
  nearly orthogonal to every activation-contrast direction (|cos| 0.02–0.19; the random-direction 99th percentile
  is 0.085) and to the paper's loop direction (0.02).

### 2.3 Capability probe: en/ett agreement on 946 unseen nouns

This probe tests whether an edit can add knowledge, with the same edit family, a random twin and few-shot prompting.

| | AUC on unseen nouns |
|---|---|
| unedited | 0.866 (accuracy 82.2 %) |
| DEV-best activation edit (35 DEV evaluations) | +0.008 [+0.003, +0.012] |
| its random twin | +0.0006 |
| few-shot prompting | 0.855 |
| LoRA on training nouns | 0.918 (+0.05) |

The unedited model is already at the best accuracy a global en/ett bias shift could reach. Reversing the direction
the way p50c does (α = 6) drops the DEV-noun AUC from 0.88 to 0.72–0.83: an edit along it can *remove* the ability, but amplifying it adds
almost nothing ([`figures/fig_gender_dose.png`](figures/fig_gender_dose.png), Appendix B).

### 2.4 Thinking mode, the paper's setting

Settings: thinking on, a 16k-token cap, 1 sample per problem. We added budget forcing after the first thinking
results as an inference-time control (logged in the design file). It cuts the unedited model's thinking at a fixed
budget, closes it with `</think>`, and lets the model answer (Muennighoff et al., 2025). Both budgets were fixed in
advance.

| thinking on, TEST | Swedish | English | cut off at 16k (sv / en) |
|---|---|---|---|
| unedited | 33.5 % | 10.5 % | 58 % / 92 % |
| **p50c** | **57.0 %** (+23.5 [+16.5, +31.0]) | 60.5 % (+50.0) | 6 % / 26 % |
| p50c random twin | 28.5 % (−5.0 [−12.0, +2.0]) | 8.5 % | 67 % / 94 % |
| budget forcing at 4k thinking tokens (no edit) | 50.0 % (+16.5 [+11.0, +22.5]) | 51.5 % | – |
| budget forcing at 8k thinking tokens (no edit) | 52.0 % (+18.5 [+13.0, +24.0]) | 54.5 % | – |

- **p50c removes the loops in Swedish too.** This replicates the paper's mechanism in a new language and task, and
  the random twin does nothing. Looping is not Swedish-specific: the unedited model loops even more in English.
- **The gain is all loop removal.**
  - On the 84 Swedish problems that the unedited model finished, p50c goes from 78.6 to 72.6 %.
  - On the 116 it cut off, p50c goes from 0.9 to 45.7 %.
  - In English: 100 → 94 % on the 17 finished problems, 2.2 → 57.4 % on the 183 cut off.
- The model thinks in English in 200 of 200 traces, and answers in Swedish in 91–98 % of them.
- **Budget forcing gets ~80 % of p50c's gain, and the two can't be told apart.**
  - p50c's remaining advantage is +5.0 [−1.0, +11.5] over the 8k budget.
  - Part of it comes from forced answers that loop again: 29–32 % of all Swedish outputs (41–47 % of the forced ones)
    hit the 1,024-token answer cap.
  - Both budgets are reported. Thinking mode generates ~8× the tokens of the default mode (mean 7.2k vs 0.9k per
    answer).
- **With loops controlled, the Swedish problems are almost as easy as the English ones.** p50c scores 57.0 vs 60.5,
  Δ −3.5 [−11.5, +4.0], against about −30 points in the default mode.

### 2.5 An LLM as the weight programmer

Claude code received the weights, the tokenizer and Python, but **0 forward passes** (`wsp/runs/zero`).
- Like the paper's own 0- and 5-pass runs, it concluded that no edit was justified and submitted none.
- It noted one thing it could not do from the weights alone: separate a Swedish-vs-English direction from token
  frequency and sub-word structure.
- It also proposed a testable idea: that the presence penalty corrupts Swedish sub-words. Setting it to 0 gave
  +1.2 [−2.2, +4.8], so the idea did not hold.

**Verdict on H1: supported.**
- No TRAIN-built, DEV-selected edit of the paper's form beats its random twin on held-out Swedish problems while
  keeping Swedish answers.
- The edits that do change accuracy either remove a failure (p50c removes looping) or switch the route (rlang α = 6
  answers in English).
- An inference-time change with no training (budget forcing, an English prefill) gets most of the same gain.
- H1 is a null result for our own edits. The noise floor puts the smallest effect we could have detected at about
  4 points.
- Our TRAIN-built edits were evaluated only in the default mode, where there is no loop to remove.

## 3. From H1 to H2: what the default-mode gap is made of

**The 2×2 of problem language × reasoning language** (unedited model, TEST, 200 × 4; Figure 1a):

| | reasoning in English | reasoning in Swedish |
|---|---|---|
| **problem in English** | 56.6 % (default) | 35.1 % ("reason in Swedish" in the user turn) |
| **problem in Swedish** | 35.8 % (English answer prefill) | 25.9 % (default) |

Reading Swedish and reasoning in Swedish each cost about 21 points. Their combined effect is close to
multiplicative: the multiplicative prediction is 22.2 %, against 25.9 % observed (+3.7 [−0.6, +8.0]). So there is
no extra Swedish-Swedish interaction. The other inference-time controls on Swedish problems:
- self-translate, +7.2 [+1.8, +12.5];
- greedy decoding, +6.6 [+0.4, +13.0];
- presence penalty 0, +1.2 [−2.2, +4.8].

Qwen3.5 ignores a *system* prompt that asks for English reasoning (100 % of answers stay Swedish). With the
instruction in the user turn, only 14 % switch, which is why the English cells use a user-turn instruction or an
answer prefill.

**In the default mode, the gap is not looping.** Truncation at the token cap is the same in both languages
(7.9 % Swedish, 7.7 % English). Among finished answers, 28 % are correct in Swedish against 61 % in English. So
there is no loop for a p50c-style edit to remove here.

**What reading Swedish costs: number words.** Reading the failed English-reasoning outputs on Swedish problems, the
most visible errors are misread numbers written as words. These were TEST outputs. H2's number-word subset was
therefore motivated by TEST, although no edit was built or selected on TEST. The tokenizer splits many Swedish number words into
pieces that are other words: " s"+"ju" (7), " n"+"io" (9), "åt"+"ta" (8), " n"+"oll" (0). Examples:

- *Synthetic problem, default mode.* "Stina köper **elva** paket muttrar. Varje paket innehåller fem muttrar."
  The model restates "Stina köper elva paket", then writes "10 paket" and answers 50 (gold 55). With digits
  ("11 paket") it is correct.
- *GSM8K TEST `gsm_sv_077`, English reasoning.* For "**Sju** av knapparna hade två hål…", the model writes: *'ju av
  knapparna' (all) have two holes. So, there are 7 knaps in this group*. It glosses the word wrongly but still uses
  the right number, 7. This sample still fails later, for an unrelated reason (it answers 98; the gold answer is
  70). A visible misreading does not always change the numbers used.
- *GSM8K TEST `gsm_sv_182`.* "**Åtta** mer än fyra gånger antalet kaffemuggar … är 40" becomes "x > 4x". The eight
  is lost, but the digit version ("8 mer än 4 gånger …") also fails in 8 of 8 samples. The real failure is the
  relation "more than", not the number word.

These examples are why H2 was worth a controlled test: the misreadings are visible, but the outputs alone can't
tell how much they cost.

## 4. H2: can a targeted edit fix Swedish number words?

> **H2.** The number-word misreading is a narrow failure that a small targeted weight edit, built only from
> synthetic number-word data (never from GSM8K), can fix, raising accuracy on held-out Swedish GSM8K problems that
> contain number words.

*Decision rule (primary metric: accuracy on the number-word problems, Swedish-reasoning cell).*
- *Supports H2:* an edit gains ≥ 5 points with its CI against its random twin excluding 0. In addition, the other
  problems may not drop more than 3 points, FLORES perplexity may not rise by 5 % or more, answers must stay ≥ 95 %
  Swedish, and the gain must be at least half of what digits give.
- *Challenges H2:* no edit beats its twin.

**Design.**
- *TEST split.* A fixed normaliser splits the 200 TEST problems into **NUM** (59 problems whose Swedish text
  contains a cardinal number word; "en/ett" are excluded) and **NONUM** (141). NUM gets 8 samples per problem and
  NONUM 4.
- *Two cells.* Swedish reasoning (the default), and English reasoning (an English answer prefill).
- *The free alternative.* The same problems with the number words rewritten as digits.
- *Synthetic data.* 480 TRAIN and 200 held-out template problems, each in a word form and a digit form. Names and
  objects are disjoint between the two sets, and two of the ten templates appear only in the held-out set.

**The edits:**
- **B**, the paper's recipe. The direction is the mean residual difference at the number position, word minus digit
  form, built from 1,027 synthetic pairs. It is applied in the two layers with the largest separation within the
  pre-registered range 6–18 (14 and 18),
  on the top-64 `down_proj` columns. The strength α ∈ {1, 2, 4} was chosen on DEV-NUM: α = 1 scored 18.3 %,
  α = 2 17.7 % and α = 4 7.9 %.
- **C**, a gradient-trained rank-1 LoRA on layer-15 `down_proj`.
- **D**, a LoRA r16 on all linear layers.
- C and D were trained on the synthetic solutions with lr 2e-4, 3 epochs, batch 8.
- **Random twins.** B and C each get a norm-matched twin (same matrices, same ‖ΔW‖_F per layer). The first
  B-twins were not norm-matched; they were replaced before any twin was evaluated, as logged in the
  pre-registration.
- *Dropped before running:* an embedding-row edit. The failing words span several tokens that share rows with
  other words, and the embeddings are tied to the output layer.
- *Exploratory, added after seeing C and D:* **C2** and **D2**. C and D had copied the one-line format of the
  synthetic solutions. C2 and D2 have the same recipes but are trained on the unedited model's own correct
  solutions of the digit version, used as targets for the word version.

**Results** (Figure 1c, [`figures/fig_numword.png`](figures/fig_numword.png) in Appendix B; full tables in
`numword/results.md`):

| Δ on NUM (59 problems × 8) vs unedited | Swedish reasoning | vs its twin | English reasoning | vs its twin |
|---|---|---|---|---|
| unedited (accuracy) | 21.6 % | | 29.9 % | |
| **digits instead of words** | **+0.4 [−3.4, +4.0]** | | **+7.4 [+1.7, +13.3]** | |
| B, paper-style rank-1 | +2.8 [−1.3, +6.8] | +2.3 [−1.5, +5.9] | +4.2 [−0.8, +9.3] | +0.8 [−3.6, +5.5] |
| B's random twin | +0.4 [−2.8, +3.6] | | +3.4 [−1.7, +8.3] | |
| C, gradient rank-1 | −14.2 (format collapse) | −10.2 [−15.9, −5.1] | +1.5 [−3.2, +5.7] | −3.0 [−8.1, +1.5] |
| C's random twin | −4.0 [−8.5, +0.2] | | +4.4 [+1.1, +8.1] | |
| D, LoRA r16 | −15.0 (format collapse) | | −20.6 | |
| C2, rank-1, self-distilled (exploratory) | +1.7 [−3.0, +6.1] | | −0.8 [−4.7, +3.0] | |
| D2, LoRA r16, self-distilled (exploratory) | +4.4 [−1.3, +10.2] | | **+6.4 [+0.6, +12.1]** | |

| synthetic held-out problems (greedy) | words | digits | words, unseen templates |
|---|---|---|---|
| unedited | 8.0 % | 55.0 % | 0.0 % |
| B / B's twin | 8.5 / 9.0 % | 53.5 / 52.5 % | 0.0 / 2.6 % |
| C / C2 | 11.0 / 9.5 % | 74.0 / 55.5 % | 2.6 / 5.3 % |
| D / D2 | **95.0** / 67.0 % | 96.0 / 73.0 % | 94.7 / 36.8 % |

| regression checks | FLORES sv perplexity | en/ett AUC | Belebele-sv | English GSM8K Δ (50 × 4) |
|---|---|---|---|---|
| unedited | 3.470 | 0.866 | 35.7 % | (53.5 %) |
| B | 3.472 (+0.1 %) | 0.866 | 35.1 % | +3.5 [−1.5, +9.0] |
| C | 3.549 (+2.3 %) | 0.874 | 35.3 % | −5.5 [−13.5, +3.0] |
| D | 4.193 (+20.8 %) | 0.822 | 35.9 % | −26.0 [−36.5, −16.0] |
| C2 | 3.498 (+0.8 %) | 0.861 | 35.7 % | not run |
| D2 | 3.924 (**+13.1 %**) | **0.823** | 36.1 % | −0.5 [−7.5, +6.5] |

1. **The misreading is real, but it matters only on the English-reasoning route.** On the synthetic problems the
   model solves 8 % with words and 55 % with digits. On GSM8K, digits gain nothing with Swedish reasoning and
   +7.4 with English reasoning. Number words are not a main cause of the Swedish gap.
2. **Rank-1 edits don't fix it.** B, C and C2 leave synthetic number reading at 8–11 %, and none beats its random
   twin on GSM8K. In the English-reasoning cell the twins alone gain +3.4 and +4.4, so 3–4 points there are within
   the noise of perturbing the weights.
3. **A LoRA learns the number words, but at a cost to Swedish.** D reads 95 % of the synthetic problems, including
   unseen templates, but copies the one-line format and collapses on GSM8K: its median answer is 66–78 tokens (C:
   94), against 485–568 for the unedited model. D2 keeps the format and gains +6.4 with English reasoning, but it has
   no twin, and random twins alone gain +3.4 to +4.4 in that cell. It also raises Swedish perplexity by 13 % (the
   limit was 5 %) and lowers en/ett AUC to 0.823.
4. **Number words are a small part of the gap.** Digits help only on the 59 NUM problems and only with English
   reasoning, which is about 2 points over all 200 TEST problems.

**Verdict on H2: challenged** by the pre-registered rule. No edit beats its random twin on NUM in the primary cell.
The misreading is a lexical gap: mapping multi-token Swedish number words to values. Rank-1 edits did not reach it.
A LoRA did, but at a cost to Swedish. The cheap fix is to rewrite number words as digits when the model reasons in
English.

## 5. H3: does thinking in Swedish match thinking in English?

The 2×2 shows that reasoning in Swedish costs ~21 points in the default mode. In thinking mode the model already
reasons in English, and once looping is controlled the Swedish problems are almost as easy as the English ones
(§2.4). H3 asks whether the model can also reason *in Swedish* when looping is controlled. It was formulated
alongside H2 and run in parallel with it.

> **H3 (prediction stated before running).** With looping controlled, thinking in Swedish will be worse than
> thinking in English by several points.

*Decision rule.*
- "Matches English" if the lower CI bound of Δ(sv-think − en-think) is above −5 points.
- "Worse" if the upper bound is below 0.
- If fewer than 80 % of the Swedish traces are mostly Swedish, the result counts as a statement about the prefill,
  not about Swedish thinking.

**Design.**
- *Problems.* The same 200 Swedish TEST problems, in thinking mode.
- *Setting the thinking language.* Qwen3.5 ignores language instructions, so the think block is started with a
  prefill:
  - Swedish: "Okej, jag löser det här steg för steg och tänker på svenska."
  - English: "Okay, I'll solve this step by step and think in English."
  - The English prefill controls for the mere presence of a prefill.
- *Loop control:*
  - p50c with 16k tokens (primary);
  - the unedited model with budget forcing at 8k (secondary). These runs were generated to 8k instead of 16k
    because of a deadline. That change was logged before they started, and it does not affect budget forcing at 8k.
- *Seeds.* Two seeds with 1 sample each, pooled per problem.

| loop control | thinking in | accuracy | traces mostly Swedish | think closed by itself | answers in Swedish |
|---|---|---|---|---|---|
| p50c, 16k | English | **61.5 %** | 1 % | 100 % | 74 % |
| p50c, 16k | Swedish | **38.8 %** | 100 % | 95 % | 99 % |
| unedited + budget forcing 8k | English | 50.0 % | 1 % | 43 % | 80 % |
| unedited + budget forcing 8k | Swedish | 34.8 % | 100 % | 36 % | 100 % |

- With p50c: Δ(sv − en) = **−22.8 [−28.2, −17.2]**.
- With budget forcing: Δ = **−15.2 [−20.8, −9.8]**.
- Both upper bounds are below 0, and the Swedish prefill works (100 % compliance).

**Confounds: why this does not yet show a capability gap.**
- *Swedish thinking is off the model's default.* Without a prefill the model thinks in English (200 of 200 traces),
  so the English prefill is close to normal behaviour and the Swedish one is not. We have no control that forces
  another language the model knows well.
- *The prefill itself moves accuracy.* With p50c, seed 0, accuracy is 62.0 % with the English prefill against 57.0 %
  without any prefill.
- *Swedish traces are shorter and loop more.*
  - Their median length is 3.9k characters, against 9.7k for English traces.
  - They hit the 16k cap in 58 of 400 outputs, against 14 of 400. Some loop after the think block closes, which the
    "closed by itself" column does not show.
  - This looks partly like a failure: forced, unfamiliar thinking stops early or loops.
- *Answer language.* With English thinking, 20–26 % of answers are in English. Accuracy is similar either way
  (62.6 % for Swedish answers, 60.5 % for English ones), so this does not drive the result.

**Examples** (p50c, seed 0; problems that English thinking solves in both seeds and Swedish thinking fails in both):
- `gsm_sv_124`, "En farsot smittar **tio** personer. Varje dag smittar varje smittad person sex andra…" (gold 3,430).
  - English trace: *"In a group of people, 10 infected initially. Each day, each infected person infects 6
    others."*
  - Swedish trace: *"Antal smittade i första dagen: S₀ = 0 (ingen smittad)"*, i.e. "number infected on the first
    day: S₀ = 0 (no one infected)". It loses the starting count.
- `gsm_sv_077` (buttons with two and four holes). The English trace translates the problem and computes 7·2 + 14·4.
  The Swedish trace invents categories ("Knappar som hade två hål (t.ex. 'fråga')") and sets up "en
  kombinationstest".

**Verdict on H3: supported** (the prediction held). Forced Swedish thinking costs 15–23 points even with loops
controlled. Whether this is a failure that a cheap fix could remove or a capability gap is open (§7).

## 6. Interpretation across the three studies

| part of the Swedish gap | size | what changes it | did a targeted edit help? |
|---|---|---|---|
| reading a Swedish problem (default mode) | ~21 points; almost 0 in thinking mode (57.0 vs 60.5) | reasoning in English: thinking mode, prefill, self-translate | the rlang α = 6 edit switched to English (+6.1) but damaged Swedish |
| … of which number words | ~2 points overall (+7.4 on 59 problems, English reasoning only) | digits (no training); a LoRA r16 that costs perplexity | no (rank-1 ≈ twin) |
| reasoning in Swedish | ~21 points in the default mode; 15–23 when Swedish thinking is forced | not tested | not tried |
| looping in thinking mode (not Swedish-specific) | +23.5 Swedish / +50 English when removed | p50c, or budget forcing (+18.5, no training) | yes, but budget forcing matches it within noise |

- **What edits of the paper's form did:**
  - removed a failure (p50c on loops);
  - switched routes at rank 1, in either direction: 99.9 % English answers with rlang α = 6, and 98 % Swedish answers
    in the rank-study positive control;
  - removed an ability when pushed (en/ett DEV AUC 0.88 → 0.72).
- **What they didn't do:** in H1 and H2, no edit beat its random twin by adding ability. Abilities were added only by
  LoRAs (number words, en/ett agreement), and at a cost to Swedish perplexity.
- **Directions read from activations missed even the useful small edit.** That edit is nearly orthogonal to them and
  was found by gradient. The paper's own working edit was likewise tuned on outcomes, not read from the weights.
- **The paper's MATH-500 result fits this pattern.** It removed a failure that only exists in the non-default
  thinking mode. In the default mode, the unedited model already scores 45.6 %, close to the edited 48.1 %.

## 7. Recommendation for SE.LLMA

1. **Don't invest in WSP-style editing to add Swedish capability.** Our TRAIN-built edits gave no held-out gain over
   random twins, and the best rank-1 edit was only as good as greedy decoding.
2. **Use rank-1 edits as cheap patches for failure modes and routing.**
   - Shown here: loops and answer language.
   - Untested, but a natural candidate: Scandinavian language confusion (in our model scan, Norwegian prompts
     produced some Danish answers).
   - Find the edits by gradient or outcome feedback, not by reading activations, and pair each with a norm-matched
     random twin and regression checks.
   - A rank-1 LoRA on one matrix has 4.6 k parameters and trains in minutes.
3. **Cheap win for small multilingual reasoning models: thinking on, with a capped thinking budget.**
   - The model thinks in English and answers in Swedish.
   - This takes Qwen3.5-0.8B from 26 % to 50–52 % on Swedish GSM8K with no weight change (one run per budget), at
     ~8× the generated tokens. The loop edit gives 57 %, a difference we cannot resolve.
   - When the model reasons in English, rewriting number words as digits adds ~7 points on the affected problems.
4. **Most informative next experiment: decide failure vs capability for Swedish reasoning.** This is the
   assignment's own question, applied to the gap that remains, and it costs less than a GPU-day.
   - (i) Add a control that forces a third language the model knows well (e.g. Chinese) to separate "Swedish" from
     "forced, unfamiliar".
   - (ii) Extend Swedish thinking to English length, s1-style ("Vänta,").
   - (iii) Train a gradient rank-1 edit, a LoRA and a norm-matched twin on ~300 translated self-traces, with
     perplexity and en/ett checks.
   - If rank 1 closes the gap, it is a failure that SE.LLMA can patch cheaply. If only LoRA (or nothing) closes it,
     the investment is Swedish reasoning data.
   - Repeating the diagnostic on 2–3 larger models (Qwen3.5-2B/4B, EuroLLM) would show whether the gap shrinks with
     scale.
5. **For the paper's own claim,** the informative test is an LLM agent against gradient search and random search at
   an equal budget of graded generations.
6. **Evaluation hygiene for every Swedish claim:**
   - report the free baselines (prefill, self-translate, greedy, budget forcing, digits) next to any trained gain;
   - use ≥ 2 seeds or ≥ 8 samples per problem, since two identical runs differed by 3.6 points;
   - give every edit a random twin.

## 8. Limitations

- **Scope:**
  - The study uses one 0.8B model; Qwen3.5-2B failed to load in vLLM 0.17.
  - The assignment targets ~4 hours. The core is H1: the 2×2, plus the paper's edit vs its twin vs budget forcing. H2, H3 are the main hypothesis that the author tested in this study.
- **Data:** the Swedish GSM8K is an LLM translation. A script checked the numbers, and the translations were
  spot-checked but not professionally reviewed. The synthetic problems of H2 are simple templates.
- **Resolution:**
  - 200 TEST problems resolve about ±3.5 points; in the primary cell the 59 NUM problems resolve ±3–5 (±5–6 with
    English reasoning); thinking-mode runs resolve ±5.5.
  - The bootstrap over problems understates run-to-run noise (a re-seed moved the unedited model by 3.6 points), and
    dozens of comparisons are uncorrected.
  - The English GSM8K check in H2 uses 50 problems (±7).
- **Use of TEST:**
  - TEST outputs motivated H2 and H3, although no edit was built or selected on TEST.
  - p50c in the default mode was run on TEST twice. The first run had 1 sample, seed 0: Swedish −3.5, English −7.5.
    The second had 4 samples, seed 1: +0.0 in both languages, the run reported here.
- **Pre-registration timing:** the H1 selection rule was written after 6 of the 27 DEV evaluations had finished.
- **Post-hoc parts:**
  - the rlang α = 6 TEST run (H1);
  - the C2/D2 edits (H2);
  - one rank-64 run at 10× learning rate (rank study);
  - budget forcing, added after the first thinking results (both budgets were fixed before running).
  All are marked as such.
- **Design gaps:**
  - Our TRAIN-built edits were evaluated only in the default mode, where there is no loop to remove.
  - The gradient-trained edits of the rank study have no random twin.
  - H3 has no control that forces another language.
- **Untuned baselines:** budget forcing was not tuned. 29–32 % of all forced outputs (41–47 % of the forced answers)
  hit the 1,024-token answer cap.
- **Directions:** the paper's "derail" direction separates right from wrong solutions only weakly (d′ ≤ 0.9), so its
  null result partly reflects a weak direction. The gradient study (§2.2) addresses this.

## 9. Reproducing the experiments

[`REPRODUCE.md`](../REPRODUCE.md) gives the full pipeline in the order it was run, and the setup is in `README.md`.
Every number in this report can be re-derived from the included outputs without a GPU:

```bash
bash scripts/unpack_results.sh                       # outputs are stored as *.jsonl.gz
source scripts/env_hf.sh                             # sets PY (transformers/peft venv) and HF_HOME
$PY src/make_tables.py > report/tables.md            # H1 tables (Appendix A)
$PY numword/src/analyze.py --md numword/results.md   # H2
venvs/vllm/bin/python swthink/analyze.py --md swthink/results.md   # H3
$PY src/figures.py && $PY src/figures_story.py       # figures
bash report/build_pdf.sh                             # REPORT.pdf and RESEARCH_NOTE.pdf
```
