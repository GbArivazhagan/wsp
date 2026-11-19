# Weight-space edits for Swedish: they removed a failure, not the Swedish gap

*SE.LLMA candidate assignment, November 2026 · Qwen3.5-0.8B (rev `2fc0636`), the paper's Experiment 8 model · code,
outputs and configs in this repository · details, examples, critique and pre-registrations in [`REPORT.pdf`](REPORT.pdf).*

**The paper, critically.** *Sparks of Weight-Space Programming* is convincing on small networks. It reads copying
heads from the weights with no forward passes, and compiles game policies into MLPs. Its language-model headline
(Qwen3.5-0.8B, MATH-500 7.9 → 48.1 %) is weaker than presented:
- With 0 or 5 forward passes, no edit was made.
- The large gain came from one 50-pass run (up to ~25,600 graded generations). Its direction was an activation
  contrast, and its strength and columns were hill-climbed on graded outcomes.
- The final edit is plain rank-1 directional editing in two layers; we rebuilt it bit-exact.
- 576 of the 579 answers it rescued had been cut off by looping in the non-default thinking mode.

So the gain came from **forward passes and outcome feedback, not from reading weights**, and it **removed a failure
rather than adding an ability** ([critique](report/critique.md)). By *failure* we mean the model has the ability
but something blocks it, such as loops or a wrong route. By *capability gap* we mean that closing it needs new
knowledge, i.e. training data.

## Question and hypotheses

Qwen3.5-0.8B solves **56.6 %** of 200 GSM8K test problems in English but **25.9 %** of the same problems in Swedish.
- H1 tests the paper's idea on this gap.
- A 2×2 diagnostic then splits the gap into a cost of *reading* Swedish and a cost of *reasoning* in Swedish.
- H2 follows up a concrete reading failure, and H3 the reasoning language.

Each hypothesis was pre-registered before its runs.

| | hypothesis | challenged if | verdict |
|--|----------------------------------------------|--------------------------------|---------|
| **H1** | Edits of the paper's form remove failures, not add capability: no TRAIN-built, DEV-selected edit beats its norm-matched random twin, and an inference change matches any gain | an edit beats its twin, keeps Swedish answers, has no regression and beats the best prompt | **supported** |
| **H2** | A targeted edit built from synthetic data fixes the misreading of Swedish number words | no edit beats its twin on number-word problems | **challenged** |
| **H3** | With looping controlled, thinking in Swedish is worse than in English | Swedish within 5 points | **supported** |

**Design.** *Data:* GSM8K translated by an LLM, with every number checked by script; TRAIN 200 problems, DEV 100,
TEST 200 (a random sample of GSM8K test, never used to build or select an edit). *Controls and alternatives:* the
unedited model, a norm-matched random twin for every edit (same matrix, same ‖ΔW‖), decoding and prompting changes,
gradient-trained rank-1 edits and LoRA. *Regression checks:* English GSM8K, Belebele-sv, Swedish perplexity, en/ett
agreement on unseen nouns. *Statistics:* paired bootstrap 95 % CIs over problems.

## Results

![Figure 1: (a) the default-mode gap; (b) H1 in thinking mode, loop removal by the paper's edit or by budget forcing, with the same problems in English as diamonds; (c) H2, number-word edits vs writing numbers as digits; (d) H3, thinking in Swedish vs English, set by a prefill.](report/figures/fig_story.png)

**H1: the edits removed a failure, and a decoding change matched them within noise.**
- *Default mode:* the three DEV-selected edits score −3.1, −0.4 and −2.0 against the unedited model, and their random
  twins −3.1, −0.6 and −1.9. The paper's edit (p50c) scores +0.0 [−3.5, +3.5]. An exploratory edit gained +6.1, but
  only by switching 99.9 % of answers to English, and it doubled Swedish perplexity. A gradient-trained rank-1 edit
  (+2.0 / +4.5) does as well as LoRA (+2.6 / +5.5) or greedy decoding (+6.6), and is nearly orthogonal to the
  activation-contrast directions.
- *Thinking mode* (Fig. 1b). The model thinks in English and loops: 58 % of Swedish outputs and 92 % of English ones
  are cut off at 16k tokens.
  - p50c lifts Swedish from 33.5 to **57.0 %** (+23.5 [+16.5, +31.0]) and English from 10.5 to 60.5 %. Its twin
    scores 28.5 %.
  - The gain is all loop removal. On problems the unedited model finished, p50c goes 78.6 → 72.6 %; on those it cut
    off, 0.9 → 45.7 %.
  - Budget forcing (stop thinking at a fixed 4k or 8k tokens, then answer; no weight change) reaches 50.0 / 52.0 %.
    The edit's extra +5.0 [−1.0, +11.5] is not significant.
  - With loops controlled, Swedish nearly matches English: 57.0 vs 60.5 %.

**From H1 to H2 and H3.** Without thinking, reading a Swedish problem costs ~21 points, and so does reasoning in
Swedish (Fig. 1a). The outputs show misread number words, which the tokenizer splits into other words (" s"+"ju",
"åt"+"ta"). For example, "Stina köper *elva* paket" (eleven packets) becomes "10 paket". On synthetic problems the
model solves 8 % when the numbers are written as words, against 55 % with digits.

**H2: number words are a small part of the gap, and no targeted edit fixes them** (Fig. 1c, the 59 TEST problems
with number words).
- Writing the numbers as digits gains +0.4 [−3.4, +4.0] with Swedish reasoning and +7.4 [+1.7, +13.3] with English
  reasoning, about 2 points over all 200 problems.
- The paper-style edit beats its random twin by +2.3 [−1.5, +5.9], and rank-1 edits leave synthetic number reading
  at 8–11 %.
- A LoRA r16 learns the number words (95 %), but collapses on GSM8K by copying the training format.
- An exploratory self-distilled LoRA gains +6.4 [+0.6, +12.1] with English reasoning. In that same cell, random twins
  alone gain +3.4 to +4.4, and this LoRA raises Swedish perplexity by 13 % (the pre-registered limit was 5 %).

**H3: forced Swedish thinking costs 15–23 points.** A prefill sets the thinking language, and 100 % of the Swedish
traces stay Swedish. Each condition was run with two seeds.
- With p50c: 38.8 % against 61.5 % for English thinking, Δ −22.8 [−28.2, −17.2].
- With budget forcing: 34.8 % against 50.0 %, Δ −15.2 [−20.8, −9.8].
- The Swedish traces are 2.5× shorter, and they hit the cap more often (58 vs 14 of 400 outputs). So "cannot reason
  in Swedish" is not yet separated from "forced, unfamiliar thinking stops early".

## Interpretation and recommendation

**Without thinking,** reading and reasoning in Swedish each cost ~21 points, and number words explain only ~2 of
them. **With thinking,** the model reasons in English, and the gap almost closes once looping is controlled. Looping
is a failure that is not specific to Swedish. Edits of the paper's form removed that failure and switched the answer
language; none added ability. Only LoRAs did (number words, en/ett), and at a cost to Swedish perplexity. The open
question is whether the cost of Swedish thinking is a failure or a capability gap.

For SE.LLMA:
1. **Don't invest in WSP-style editing to add Swedish capability.** Use rank-1 edits as cheap patches for failure
   modes, such as the loops and answer language shown here. Find them by gradient or outcome feedback, not by reading
   activations, and test each against a random twin and regressions.
2. **Take the cheap win.** Thinking with a capped budget (English thinking, Swedish answers) takes Swedish GSM8K from
   26 % to 50–52 % with no training, at ~8× the generated tokens.
3. **Next experiment:** decide failure vs capability for Swedish thinking.
   - Force a third language as a control, and extend Swedish thinking to English length (s1-style "Vänta,").
   - Then train a rank-1 edit, a LoRA and a random twin on ~300 translated self-traces.
   - If rank 1 closes the gap, it is a cheap patch. If only LoRA does, Swedish reasoning data is the investment.

**Limitations.**
- One 0.8B model, and LLM-translated data.
- The CIs (±3.5 on TEST, ±3–6 on number-word problems, ±5.5 in thinking mode) understate run-to-run noise:
  re-seeding the unedited model moved it by 3.6 points. Dozens of comparisons are uncorrected.
- Our TRAIN-built edits ran only in the default mode, where there is no loop to remove.
- TEST outputs motivated H2 and H3.
- An earlier 1-sample run of p50c in the default mode gave −3.5 for Swedish and −7.5 for English.
- **Scope:** the assignment targets ~4 hours. The core is H1 (the 2×2, and the paper's edit vs its twin vs budget
  forcing). The rest are extensions that AI agents ran in parallel over ~12 hours of wall-clock time on four GPUs
  ([`AI_USAGE.md`](AI_USAGE.md)).
