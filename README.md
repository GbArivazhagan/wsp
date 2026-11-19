# Weight-space edits for Swedish: they removed a failure, not the Swedish gap

**Candidate assignment:** Weight-Space Programming for SE.LLMA · **Date:** November 2026 ·
**Model:** `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`

Can the weight edits from
[*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) help build strong
Swedish models? We tested this on the language model of the paper's Experiment 8. It solves 57 % of 200 GSM8K test
problems in English but only 26 % of the same problems in Swedish.

We ran three pre-registered hypotheses:
- **H1** tests the paper's idea on this gap.
- A 2×2 diagnostic then splits the gap into a cost of reading Swedish and a cost of reasoning in Swedish.
- **H2** follows up a reading failure (Swedish number words), and **H3** the reasoning language.

Every edit was compared with four things, each on problems never used to build or select the edit:
- the unedited model;
- a norm-matched random edit of the same size;
- inference-time alternatives;
- conventional fine-tuning.

| | hypothesis | verdict | key result |
|---|---|---|---|
| H1 | Edits of the paper's form remove failures but don't add capability | supported | No TRAIN-built, DEV-selected edit beats its random twin. The paper's edit helps only in thinking mode, by removing loops (+23.5 on Swedish); budget forcing, with no weight change, gets +18.5, and the two can't be told apart |
| H2 | A targeted edit can fix the misreading of Swedish number words | challenged | Rank-1 edits ≈ random twins. Writing numbers as digits helps only with English reasoning, about 2 points overall |
| H3 | With loops controlled, thinking in Swedish is still worse than in English | supported | −15 to −23 points, though forced Swedish traces are shorter and loop more (failure vs capability is still open) |

> **What this is, and what it is not.**
> - We did **not** replicate the paper's method, an LLM agent hill-climbing an edit with thousands of graded
>   generations.
> - We tested *edits of the same form*: rank-1 directional edits on `down_proj` columns.
>   - The directions come from activation contrasts, and the strength is chosen on a DEV split.
>   - We also rebuilt the paper's own edit bit-exact.
>   - Our LLM weight-programmer agent ran with 0 forward passes only.
> - **Scope:** The core experiment is H1: the 2×2, and the paper's edit vs its random
>   twin vs budget forcing. Everything else (H2, H3, an edit-rank side study, an en/ett probe, a 7-model language
>   scan) forms the hypothesis that was tested in this work. 

## Where to read

| File | Content |
|---|---|
| [`RESEARCH_NOTE.md`](RESEARCH_NOTE.md) | The short note: critique, hypotheses, experiment, results, interpretation, recommendation. |
| [`REPORT.pdf`](REPORT.pdf) | The note, then the detailed report with examples, all tables and figures, the critique and the three pre-registrations. |
| [`report/report.md`](report/report.md) | Markdown source of the detailed report. |
| [`report/critique.md`](report/critique.md) | The critical reading of the paper and its released logs. |
| [`report/hypothesis_and_design.md`](report/hypothesis_and_design.md), [`numword/preregistration.md`](numword/preregistration.md), [`swthink/preregistration.md`](swthink/preregistration.md) | Pre-registrations of H1, H2 and H3, written before the runs, with later changes logged inside. |
| [`report/tables.md`](report/tables.md), [`numword/results.md`](numword/results.md), [`swthink/results.md`](swthink/results.md) | Generated result tables for H1, H2 and H3. |
| [`rank_study/REPORT.md`](rank_study/REPORT.md) | H1 side study: gradient-trained edits of rank 1 to 64 against LoRA, with its own pre-registration and decisions log. |
| [`REPRODUCE.md`](REPRODUCE.md) | How to re-derive every number without a GPU, and the full GPU pipeline in the order it was run. |
| `results/`, `numword/results/`, `swthink/results/`, `rank_study/results/` | The model outputs behind every reported number, gzipped (`*.jsonl.gz`, one line per sample: prompt, response, gold answer, finish reason, response language), each with a `.config.json` (see below). Unpack with `bash scripts/unpack_results.sh`. |
| [`wsp/`](wsp/) | The Claude-as-weight-programmer harness and its full run (`runs/zero`: context, code, answer). |
| [`data/`](data/) | Swedish GSM8K (TEST 200, TRAIN/DEV 300), MATH-500, en/ett nouns, scan prompts, and the paper's p50c edit with the authors' rebuild code. |
| [`AI_USAGE.md`](AI_USAGE.md) | Which AI tools did what. |

**Configs, revisions and seeds.** Every output file has a config recording the model, decoding settings, number of
samples and seed. The seed is 0 unless the config says otherwise.
- *Base models:* the config gives the Hugging Face repo and revision.
- *Edited checkpoints:* the config gives the checkpoint name. All edited checkpoints are built from
  `Qwen/Qwen3.5-0.8B` at the revision above (see *Files not in the repository*).
- *Self-translate:* one config, `results/x2/base_P-sv_selftranslate.config.json`, was reconstructed from the code
  afterwards and says so.

## Code

| Path | What it does |
|---|---|
| `src/gen_vllm.py` | Generation with vLLM. Every run writes its config. |
| `src/grade_math.py` | Exact-match grading of the final number (Swedish number formats handled), and math-verify for MATH-500. |
| `src/langid.py` | sv/en/da/nb/nn language ID of answers and thinking traces (lingua). |
| `src/math_edit.py`, `src/math_sweep.py`, `src/select_configs.py` | H1 edits of the paper's form, the DEV grid, and the pre-registered selection rule. |
| `scripts/build_paper_edit.sh`, `src/build_paper_controls.py` | The paper's edits, rebuilt bit-exact with the authors' `apply_edit.py`, and the norm-matched random twin. |
| `src/budget_force.py` | Budget forcing: cut thinking at B tokens, close it, and let the model answer. |
| `src/eval_logprob.py` | Regression checks: Belebele, en/ett agreement, Swedish perplexity on FLORES passages. |
| `src/gender_sweep.py`, `src/lora_train.py` | The en/ett capability probe and its LoRA comparator. |
| `src/summarize.py`, `src/make_tables.py`, `src/figures.py`, `src/figures_story.py` | Paired bootstrap CIs, tables and figures. |
| `numword/src/` | H2: number normaliser, synthetic problems, edit B and its twin, LoRA edits C/D/C2/D2, analysis. |
| `swthink/run.sh`, `swthink/analyze.py` | H3: thinking-language prefills and analysis. |
| `scripts/`, `numword/scripts/`, `*/queue/` | The run scripts and job queues used for all GPU runs. |

## Setup

```bash
# Linux, CUDA 12.8 wheels, Python 3.10. Two environments: vLLM for generation, transformers/peft for edits and training.
python3.10 -m venv venvs/vllm && venvs/vllm/bin/pip install -r requirements/vllm.txt
python3.10 -m venv venvs/hf   && venvs/hf/bin/pip install -r requirements/hf.txt "math-verify[antlr4_13_2]==0.9.0"
export HF_HOME=~/.cache/huggingface      # models and Belebele are read from here
venvs/hf/bin/python scripts/download.py  # all models at their pinned revisions (src/common.py), Belebele
```

`requirements/*.txt` are `pip freeze` lock files of the two environments we used. One 12 GB GPU is enough for every
step; we ran three 12 GB GPUs and one RTX 3090 in parallel.

## Reproduce

Every reported number can be re-derived from the included outputs on a CPU in a few minutes:

```bash
bash scripts/unpack_results.sh                                     # *.jsonl.gz -> *.jsonl
source scripts/env_hf.sh                                           # sets PY and HF_HOME, cds to the repo root
$PY src/make_tables.py > report/tables.md                          # H1
$PY numword/src/analyze.py --md numword/results.md                 # H2
venvs/vllm/bin/python swthink/analyze.py --md swthink/results.md   # H3
$PY src/figures.py && $PY src/figures_story.py && bash report/build_pdf.sh
```

To regenerate the outputs on a GPU, follow [`REPRODUCE.md`](REPRODUCE.md). It lists the steps in the order they
were run, starting with the main experiment:

```bash
bash scripts/build_paper_edit.sh     # the paper's edit, rebuilt bit-exact, and its random twin
bash scripts/run_2x2.sh              # the 2x2: problem language x reasoning language
bash scripts/think_worker.sh 0.55    # thinking mode, 16k cap: unedited, p50c, random twin
```

The run scripts skip outputs that already exist. Move the included outputs aside first; `REPRODUCE.md` explains how.

## Files not in the repository

- **Model weights.** The base model is downloaded at the pinned revision. The edited checkpoints are excluded for
  size: 28 GB in `models/`, about 1.7 GB each. Each is rebuilt by a script with seed 0:

  | checkpoints | rebuilt by |
  |---|---|
  | `qwen35_paper_p50c`, `qwen35_paper_p50c_randctrl` | `scripts/build_paper_edit.sh` |
  | `msw_*`, `mrnd_*` (H1 edits and their twins) | `src/math_edit.py` / `src/math_sweep.py`, via `scripts/run_math_stage1.sh` and `run_math_stage2.sh` |
  | `qwen35_gender_*` | `src/gender_sweep.py`, `src/lora_train.py` |
  | `rank_*` (edit-rank study) | `rank_study/src/`, via `rank_study/run_queue2.sh` |
  | `nw_*` (H2 edits and their twins) | `numword/src/build_edit_B.py`, `build_Btwin_matched.py`, `build_Ctwin.py`, `lora_syn.py` |

- **Environments and caches:** `venvs/`, the Hugging Face cache, and the job queues' lock directories.
- **Kept out to keep the repository small, available from the author on request:**
  - execution logs;
  - the DEV-grid and TRAIN-pool generations of H1 (the DEV table is in `results/msweep/table.md`);
  - the edit-rank study's DEV runs;
  - the raw thinking runs of H3 before budget forcing;
  - the GSM8K translation batches and their assembly scripts;
  - our copy of UD Swedish-Talbanken, which is public.

## Data and licences

| Data | Licence | Notes |
|---|---|---|
| `Qwen/Qwen3.5-0.8B` | Apache-2.0 | The model under test. |
| Qwen2.5-0.5B/1.5B-Instruct, Qwen3-0.6B, SmolLM2-1.7B-Instruct, OLMo-2-1B-Instruct, EuroLLM-1.7B-Instruct | Apache-2.0 per their model cards | Used only in the language scan, at the revisions in `src/common.py`. |
| GSM8K, © 2021 OpenAI | MIT | `data/gsm8k/`, including our Swedish translations. These are LLM translations: every number was checked by script, and they were spot-checked but not professionally reviewed. |
| MATH-500 | MIT | `data/math500/` |
| Belebele | CC BY-SA 4.0 | Downloaded at revision `7899cdf`; its FLORES passages are used for perplexity. |
| UD Swedish-Talbanken | CC BY-SA 4.0 | Not included (public at universaldependencies.org). It is needed only by `src/build_gender_pairs.py`. |
| `data/gender/` | CC BY-SA 4.0 | en/ett nouns and contexts derived from UD Swedish-Talbanken. |
| Synthetic prompts and number-word problems | — | Generated for this study by script (`data/prompts/`, `numword/data/`). |
| `data/wsp_logs/` | Logs and data CC BY 4.0, scripts MIT | An unmodified subset of the paper's released logs (`github.com/RikiyaT/lm_understand-logs`, commit `4d09ba6`): the p50c edit, its notes, and `apply_edit.py`. See `LICENSE` and `SOURCE.md` there. |

## AI usage

| Tool and model | Used for |
|---|---|
| Claude Code with Claude Opus 5.5 | Coding and research under the author's direction: the H1 design, all code, running the GPU jobs, figures, and drafts of the note, the report and this README.|

The author set the scope, formulated H2 and H3, and reviewed the design, the results and the conclusions. No AI
model graded any result: all accuracies come from exact-match graders and log-probabilities. Details are in
[`AI_USAGE.md`](AI_USAGE.md).

## Reference

Takehi et al. (2026). *Sparks of Weight-Space Programming.* https://rikiyat.github.io/blog/weight-space-programming/
Released logs: https://github.com/RikiyaT/lm_understand-logs
