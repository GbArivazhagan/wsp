# Reproducing the study

There are two levels:
- **Level 1** needs no GPU. It re-derives every reported number from the outputs included in this repository.
- **Level 2** regenerates those outputs on a GPU, in the order the runs were made.

Setup (environments, model download) is in [`README.md`](README.md#setup).

## Level 1: re-derive every number (CPU only, a few minutes)

```bash
bash scripts/unpack_results.sh                                     # outputs are stored as *.jsonl.gz
source scripts/env_hf.sh                                           # sets PY, HF_HOME; cd to the repo root
$PY src/make_tables.py > report/tables.md                          # H1 tables (REPORT.pdf, Appendix A)
$PY numword/src/analyze.py --md numword/results.md                 # H2 tables
venvs/vllm/bin/python swthink/analyze.py --md swthink/results.md   # H3 table
venvs/vllm/bin/python src/summarize.py expD                        # thinking on/off and MATH-500 summary
$PY src/figures.py && $PY src/figures_story.py                     # report/figures/*.png
bash report/build_pdf.sh                                           # REPORT.pdf (pandoc + xelatex)
```

MATH-500 grading needs `math-verify`. Without it the grader stops with an error, because a silent string-equality
fallback would under-count (39.4 % instead of 45.6 %).

## Level 2: regenerate the outputs (GPU)

**Every run script skips an output that already exists.** To regenerate, start from empty result folders. For
example, run `mv results results_shipped && mkdir results` (and the same for `numword/results`, `swthink/results`).
Runtimes are for one 12 GB GPU: the thinking-mode runs take 1–2 h each, and most other steps take minutes. Run every
command from the repository root after `source scripts/env_vllm.sh`.

### H1: edits of the paper's form on the Swedish gap

| step | command | produces |
|---|---|---|
| 1 | `venvs/hf/bin/python scripts/download.py` | Qwen3.5-0.8B and the scan models at the pinned revisions, plus Belebele |
| 2 | `bash scripts/build_paper_edit.sh` | `models/qwen35_paper_p50c` (bit-exact, hash-checked) and its random twin |
| 3 | `venvs/hf/bin/python src/eval_logprob.py --model qwen3.5-0.8b --out results/logprob/qwen35_base.json` | regression checks for the unedited model |
| 4 | `bash scripts/run_scan.sh` | the language scan over small models (`results/scan/`) |
| 5 | `bash scripts/run_expD.sh` | p50c and its twin with thinking off (1 sample), and MATH-500 (`results/expD/`) |
| 6 | `bash scripts/run_expD_n4.sh` | p50c and its twin with thinking off, 4 samples, seed 1: the reported default-mode comparison |
| 7 | `bash scripts/run_2x2.sh` and `bash scripts/run_routing_cells.sh` | the 2×2, the English-answer prefill and self-translate (`results/x2/`) |
| 8 | `bash scripts/run_decoding.sh` | decoding controls (presence penalty 0, greedy), user-turn language cells, and the TRAIN data for the *rlang* direction |
| 9 | `bash scripts/run_math_stage1.sh`, then `bash scripts/run_math_stage1b.sh` | the TRAIN pools, the three directions and the 27-config DEV grid (`results/msweep/`) |
| 10 | `$PY src/math_sweep.py table > results/msweep/table.md && $PY src/select_configs.py` | the pre-registered selection (`results/mtest/selected.json`) |
| 11 | `SEL="msw_derail_L15-19_a1 msw_rlang_L10-14_a2 msw_inalign_L6-18x4_a1" bash scripts/run_math_stage2.sh` | TEST runs of the three selected edits and their twins, with regression checks |
| 12 | `SEL=msw_rlang_L15-19_a6 bash scripts/run_math_stage2.sh` | the exploratory α = 6 edit, which was not in the pre-registration |
| 13 | `bash scripts/think_worker.sh 0.55` (one per GPU) | thinking mode with a 16k cap: unedited, p50c and its twin (`results/expD/gsm_think16k_*`) |
| 14 | `for B in 4096 8192; do $PY src/budget_force.py --src results/expD/gsm_think16k_base.jsonl --budget $B --out results/expD/gsm_think16k_base_bf$B.jsonl; done` | budget forcing |

**en/ett capability probe.** All of these run in the hf environment (`source scripts/env_hf.sh`):
- `$PY src/gender_sweep.py`: directions, the DEV grid, and TEST for the best edit and its twin (`results/gender/`).
- `$PY src/lora_train.py --data gender --r 16 --lr 1e-4 --epochs 2 --bs 32 --out qwen35_gender_lora`, then
  `$PY src/eval_logprob.py --model qwen35_gender_lora --tasks gender_test,belebele_sv,ppl_sv,ppl_en --out results/logprob/qwen35_gender_lora.json`.
- `$PY src/gender_bootstrap.py qwen35_gender_best_acts qwen35_gender_best_random qwen35_gender_lora`.

The noun pairs in `data/gender/` are included. Rebuilding them with `src/build_gender_pairs.py` needs the UD
Swedish-Talbanken `.conllu` files in `data/ud/`.

**Edit-rank side study.** `bash rank_study/run_queue2.sh` executes `rank_study/queue.txt` (see
`rank_study/preregistration.txt`). It stops starting jobs after 22:35 local time, which was our time cap. Delete
that check to run it at other times.

**LLM weight programmer.** The code is run through `wsp/run.sh <workdir> <script>`, with the budget in
`<workdir>/budget.json`. The full run is in `wsp/runs/zero/`.

### H2: number words

| step | command | produces |
|---|---|---|
| 1 | `$PY numword/src/make_synth.py` and `$PY numword/src/normalize_numbers.py` | the synthetic problems, the digit-rewritten TEST set and the NUM/NONUM split (`numword/data/`, included) |
| 2 | `bash numword/scripts/step1_build_B.sh > numword/logs/step1_build_B.log` | edit B for α ∈ {1, 2, 4} |
| 3 | `bash numword/scripts/step2_twins.sh > numword/logs/step2_twins.log` | the unedited model on the synthetic problems, and the norm-matched twins of B |
| 4 | `bash numword/scripts/step3_train_CD.sh` | edits C and D, C's twin, and the DEV runs for choosing α |
| 5 | `bash numword/queue/worker.sh 1 0.45` (one per GPU) | everything in `numword/queue/jobs.txt`: selection, TEST cells, synthetic probes, regressions, C2/D2 |
| 6 | `venvs/hf/bin/python numword/src/analyze.py --md numword/results.md` | the tables |

Steps 3 and 5 wait for the lines "B built" and "twins rebuilt" in those logs, so keep the redirections in steps 2
and 3 (create `numword/logs/` first).

### H3: thinking language

| step | command | produces |
|---|---|---|
| 1 | `bash swthink/queue/worker.sh 1 0.45` (one per GPU) | the 8 runs in `swthink/queue/jobs.txt`: p50c and the unedited model (budget-forced at 8k), each with Swedish and English thinking, seeds 0 and 1 |
| 2 | `venvs/vllm/bin/python swthink/analyze.py --md swthink/results.md` | the table |