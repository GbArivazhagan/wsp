
**Table 1. Inference-time conditions, unedited model (TEST, 200 problems × 4 samples; Δ vs the same problems in the default setting).**

| condition | problem | acc % | Δ vs default [95% CI] | answers in problem language | note |
|----------------------------------|-----|-----|---------------|---------|--------------|
| default (non-thinking, presence 1.5) | sv | 25.9 | +0.0 [+0.0, +0.0] | 100 % |  |
| default (non-thinking, presence 1.5) | en | 56.6 | +0.0 [+0.0, +0.0] | 100 % |  |
| default, other sampling seed (null) | sv | 22.2 | -3.6 [-6.9, -0.4] | 100 % | same model and settings |
| default, other sampling seed (null) | en | 57.5 | +0.9 [-2.0, +3.6] | 100 % | same model and settings |
| presence penalty 0 | sv | 27.1 | +1.2 [-2.2, +4.8] | 100 % |  |
| presence penalty 0 | en | 55.2 | -1.4 [-5.0, +2.4] | 100 % |  |
| greedy (1 sample) | sv | 32.5 | +6.6 [+0.4, +13.0] | 100 % |  |
| greedy (1 sample) | en | 59.5 | +2.9 [-2.4, +8.1] | 100 % |  |
| 'reason in Swedish' (user turn) | en | 35.1 | -21.5 [-26.0, -17.1] | 0 % | answers switch to Swedish |
| 'reason in English' (user turn) | sv | 28.6 | +2.8 [-1.4, +6.6] | 86 % | only 14 % switch |
| English answer prefill | sv | 35.8 | +9.9 [+5.1, +14.5] | 1 % | forces English reasoning |
| self-translate, then solve in English | sv | 33.1 | +7.2 [+1.8, +12.5] | 4 % | no weight change |

**Table 2. The paper's p50c edit (loop removal) and its norm-matched random twin, on Swedish and English GSM8K (TEST).**

| model / mode | problem | acc % | Δ vs unedited same mode [95% CI] | answers in problem language | note |
|----------------------------------|-----|-----|---------------|---------|--------------|
| unedited, non-thinking (4 samples) | sv | 22.2 | +0.0 [+0.0, +0.0] | 100 % |  |
| p50c, non-thinking (4 samples) | sv | 22.2 | +0.0 [-3.5, +3.5] | 100 % |  |
| random twin, non-thinking (4 samples) | sv | 24.1 | +1.9 [-1.5, +5.4] | 100 % |  |
| unedited, non-thinking (4 samples) | en | 57.5 | +0.0 [+0.0, +0.0] | 100 % |  |
| p50c, non-thinking (4 samples) | en | 57.5 | +0.0 [-3.4, +3.4] | 100 % |  |
| random twin, non-thinking (4 samples) | en | 57.6 | +0.1 [-2.9, +3.2] | 100 % |  |
| unedited, thinking, 16k cap (1 sample) | sv | 33.5 | +0.0 [+0.0, +0.0] | 44 % |  |
| p50c, thinking, 16k cap (1 sample) | sv | 57.0 | +23.5 [+16.5, +31.0] | 90 % | share counts cut-off outputs (English thinking only) |
| random twin, thinking, 16k cap (1 sample) | sv | 28.5 | -5.0 [-12.0, +2.0] | 34 % | share counts cut-off outputs (English thinking only) |
| budget forcing at 4k thinking tokens (no weight change), thinking, 16k cap (1 sample) | sv | 50.0 | +16.5 [+11.0, +22.5] | 96 % | share counts cut-off outputs (English thinking only) |
| budget forcing at 8k thinking tokens (no weight change), thinking, 16k cap (1 sample) | sv | 52.0 | +18.5 [+13.0, +24.0] | 97 % | share counts cut-off outputs (English thinking only) |
| unedited, thinking, 16k cap (1 sample) | en | 10.5 | +0.0 [+0.0, +0.0] | 100 % |  |
| p50c, thinking, 16k cap (1 sample) | en | 60.5 | +50.0 [+43.0, +57.0] | 100 % |  |
| random twin, thinking, 16k cap (1 sample) | en | 8.5 | -2.0 [-7.0, +3.0] | 100 % |  |
| budget forcing at 4k thinking tokens (no weight change), thinking, 16k cap (1 sample) | en | 51.5 | +41.0 [+34.0, +48.0] | 100 % |  |
| budget forcing at 8k thinking tokens (no weight change), thinking, 16k cap (1 sample) | en | 54.5 | +44.0 [+37.0, +51.0] | 100 % |  |

**Table 3. DEV-selected WSP-style edits for the Swedish gap and their norm-matched random twins (TEST; Δ vs unedited).**

| model | problem | acc % | Δ vs unedited [95% CI] | answers in problem language | note |
|----------------------------------|-----|-----|---------------|---------|--------------|
| edit: derail (paper's recipe) (msw_derail_L15-19_a1) | sv | 22.8 | -3.1 [-6.6, +0.2] | 100 % |  |
| edit: derail (paper's recipe) — random twin | sv | 22.8 | -3.1 [-6.1, -0.1] | 100 % |  |
| edit: derail (paper's recipe) (msw_derail_L15-19_a1) | en | 55.9 | -0.8 [-3.8, +2.2] | 100 % |  |
| edit: derail (paper's recipe) — random twin | en | 58.5 | +1.9 [-1.1, +4.9] | 100 % |  |
| edit: reasoning-language (msw_rlang_L10-14_a2) | sv | 25.5 | -0.4 [-3.9, +3.1] | 100 % |  |
| edit: reasoning-language — random twin | sv | 25.2 | -0.6 [-3.8, +2.6] | 100 % |  |
| edit: reasoning-language (msw_rlang_L10-14_a2) | en | 58.5 | +1.9 [-1.4, +5.2] | 100 % |  |
| edit: reasoning-language — random twin | en | 56.0 | -0.6 [-3.9, +2.5] | 100 % |  |
| edit: input alignment (msw_inalign_L6-18x4_a1) | sv | 23.9 | -2.0 [-5.5, +1.4] | 99 % |  |
| edit: input alignment — random twin | sv | 24.0 | -1.9 [-5.0, +1.2] | 100 % |  |
| edit: input alignment (msw_inalign_L6-18x4_a1) | en | 57.2 | +0.6 [-2.8, +3.8] | 100 % |  |
| edit: input alignment — random twin | en | 56.2 | -0.4 [-3.6, +2.8] | 100 % |  |
| exploratory: rlang L15-19 α=6 (not eligible: 0 % Swedish answers on DEV) | sv | 32.0 | +6.1 [+1.9, +10.6] | 0 % |  |
| exploratory — random twin | sv | 23.6 | -2.2 [-5.5, +0.9] | 100 % |  |
| exploratory: rlang L15-19 α=6 (not eligible: 0 % Swedish answers on DEV) | en | 56.0 | -0.6 [-4.2, +2.9] | 100 % |  |
| exploratory — random twin | en | 57.6 | +1.0 [-2.4, +4.5] | 100 % |  |

**Table 6. Edit-rank study: gradient-trained edits on the same RFT data (own correct Swedish TRAIN solutions); TEST, Δ vs unedited.**

| edit | problem | acc % | Δ vs unedited [95% CI] | answers in problem language | note |
|----------------------------------|-----|-----|---------------|---------|--------------|
| rank 1, down_proj L15, lr 2e-4 | sv | 28.7 | +2.9 [-0.6, +6.5] | 100 % |  |
| rank 1, down_proj L15, lr 2e-4 | en | 59.0 | +2.4 [-1.1, +5.9] | 100 % |  |
| rank 1, down_proj L15, lr 2e-3 (DEV-selected R1) | sv | 30.4 | +4.5 [+0.9, +8.1] | 100 % |  |
| rank 1, down_proj L15, lr 2e-3 (DEV-selected R1) | en | 57.4 | +0.8 [-2.9, +4.2] | 100 % |  |
| rank 1, down_proj L15, lr 2e-3, training seed 1 | sv | 27.9 | +2.0 [-1.9, +5.9] | 100 % |  |
| rank 1, down_proj L15, lr 2e-3, training seed 1 | en | 60.6 | +4.0 [+0.4, +7.8] | 100 % |  |
| rank 8, down_proj L15 | sv | 28.1 | +2.2 [-1.5, +6.0] | 100 % |  |
| rank 8, down_proj L15 | en | 59.6 | +3.0 [-0.8, +6.6] | 100 % |  |
| rank 64, down_proj L15 | sv | 26.8 | +0.9 [-3.0, +4.8] | 100 % |  |
| rank 64, down_proj L15 | en | 58.2 | +1.6 [-1.9, +4.9] | 100 % |  |
| rank 64, down_proj L15, lr 2e-3 | sv | 22.9 | -3.0 [-6.9, +0.9] | 100 % |  |
| rank 64, down_proj L15, lr 2e-3 | en | 53.1 | -3.5 [-7.6, +0.8] | 100 % |  |
| rank 1, every down_proj | sv | 29.4 | +3.5 [-0.4, +7.1] | 99 % |  |
| rank 1, every down_proj | en | 59.6 | +3.0 [-0.5, +6.5] | 100 % |  |
| rank 16, all linear layers (RFT-LoRA) | sv | 28.5 | +2.6 [-1.5, +6.8] | 100 % |  |
| rank 16, all linear layers (RFT-LoRA) | en | 57.6 | +1.0 [-2.9, +4.8] | 100 % |  |
| rank 16, all linear layers, training seed 1 | sv | 31.4 | +5.5 [+1.6, +9.1] | 100 % |  |
| rank 16, all linear layers, training seed 1 | en | 57.9 | +1.2 [-2.5, +5.1] | 100 % |  |
| positive control: rank 1, down_proj L15, trained to answer English problems in Swedish | en | 42.4 | -14.2 [-18.5, -10.0] | 2 % | share column = answers in English |

**Table 4. Regression checks (log-prob): Belebele accuracy, FLORES perplexity (nats/token), en/ett agreement on unseen nouns.**

| model | Belebele sv | Belebele en | ppl sv | ppl en | en/ett acc | en/ett AUC |
|--------------------------|-------|-------|-------|-------|-------|-------|
| mrnd_derail_L15-19_a1 | 0.356 | 0.370 | 3.470 | 3.174 | 0.819 | 0.866 |
| mrnd_inalign_L6-18x4_a1 | 0.354 | 0.373 | 3.473 | 3.176 | 0.817 | 0.865 |
| mrnd_rlang_L10-14_a2 | 0.354 | 0.370 | 3.470 | 3.174 | 0.818 | 0.865 |
| mrnd_rlang_L15-19_a6 | 0.366 | 0.369 | 3.501 | 3.183 | 0.812 | 0.858 |
| msw_derail_L15-19_a1 | 0.358 | 0.369 | 3.470 | 3.175 | 0.818 | 0.866 |
| msw_inalign_L6-18x4_a1 | 0.354 | 0.368 | 3.493 | 3.187 | 0.805 | 0.854 |
| msw_rlang_L10-14_a2 | 0.353 | 0.371 | 3.485 | 3.173 | 0.815 | 0.861 |
| msw_rlang_L15-19_a6 | 0.332 | 0.367 | 8.287 | 3.186 | 0.688 | 0.527 |
| qwen35_base | 0.357 | 0.369 | 3.470 | 3.174 | 0.822 | 0.866 |
| qwen35_gender_lora | 0.362 | – | 4.682 | 3.435 | 0.866 | 0.918 |
| qwen35_paper_p50c | 0.351 | 0.358 | 3.519 | 3.186 | 0.820 | 0.871 |
| qwen35_paper_p50c_randctrl | 0.358 | 0.371 | 3.489 | 3.184 | 0.821 | 0.869 |

**Table 5. Capability probe: Swedish en/ett agreement on 946 unseen TEST nouns (3 templates each).**

| condition | acc | AUC | ΔAUC vs base [95% CI] | note |
|----------------------------------------|------|------|------------------|----------------|
| unedited | 0.822 | 0.866 | – | best global-bias accuracy 0.822 |
| few-shot prompt | 0.808 | 0.855 | – | 8 article–noun examples |
| edit, activation direction (DEV-best: mid, α=-4; 35 DEV evals) | 0.824 | 0.874 | +0.0076 [+0.0032, +0.0120] | |
| same edit, random direction | 0.821 | 0.867 | +0.0006 [-0.0006, +0.0018] | |
| same edit, weights-only direction (embedding en−ett) | 0.820 | 0.867 | – | |
| weights-only, a-priori config (no feedback) | 0.818 | 0.866 | – | |
| LoRA fine-tuned on 1,901 TRAIN nouns | 0.866 | 0.918 | – | gradient baseline |
