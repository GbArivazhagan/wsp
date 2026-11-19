# Swedish thinking study: pre-registration (2026-11-19 ~17:50 CET, before any run)

## Question
In thinking mode, Qwen3.5-0.8B thinks in English (200/200 traces) and answers in Swedish. With looping controlled
(the paper's p50c edit, or budget forcing at 8k thinking tokens) it reaches 57.0 % / 52.0 % on the Swedish GSM8K TEST
set, against 60.5 % in English. In the default (non-thinking) mode, reasoning in Swedish costs ~21 points.
**Does thinking in Swedish reach the accuracy of thinking in English, once looping is controlled?** If yes, a Swedish
model can reason in Swedish at no cost in this setting; if no, the English-thinking route is worth a measurable gain.

## Prediction (stated before running)
Swedish thinking will be worse than English thinking (the default-mode Swedish-reasoning cost carries over), by
several points.

## Design
- Problems: the 200 Swedish GSM8K TEST problems (same as all earlier runs).
- Thinking language set by a prefill at the start of the think block (Qwen3.5 ignores language instructions):
  sv: "Okej, jag löser det här steg för steg och tänker på svenska.\n\n"
  en: "Okay, I'll solve this step by step and think in English.\n\n"
  The English prefill is the control for "a prefill is present".
- Models: unedited Qwen3.5-0.8B (rev 2fc0636), and the paper's p50c edit (loops removed by the weights).
- Sampling as the main study's thinking runs: T 0.6, top_p 0.95, top_k 20, presence 1.5, 16,384 tokens; seeds 0 and 1
  (1 sample each; both pooled per problem).
- Loop control for the unedited model: budget forcing at 8,192 thinking tokens (src/budget_force.py; the budget was
  fixed in the main study).

## Measures
- Accuracy; share of think traces mostly in Swedish (lingua on the think text); answer language; truncation/looping rate.

## Analysis and decision rule (primary: p50c, sv-think vs en-think; secondary: unedited + budget forcing 8k)
Paired bootstrap over problems (5000, seed 0) of Δ = acc(sv-think) − acc(en-think), both seeds pooled.
- "Swedish thinking matches English": lower CI bound > −5 points.
- "Swedish thinking is worse": upper CI bound < 0.
- Otherwise inconclusive.
- Compliance: if fewer than 80 % of the sv-think traces are mostly Swedish, the result is reported as a result about
  the prefill, not about Swedish thinking.

Change (17:45, before any unedited-model run started; reason: the user's 21:00 deadline): the unedited model is
generated with max_tokens 8,192 instead of 16,384 and then budget-forced at 8,192. For budget forcing at 8k this is
the same as cutting a 16k generation, because thinking past 8,192 tokens is discarded. The unforced 16k accuracy of the
unedited model with a prefill is therefore not measured. The p50c runs keep 16,384. Seed-1 runs are done if time allows.
