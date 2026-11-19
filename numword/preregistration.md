# Number-word study: pre-registration (written 2026-11-19 ~17:10 CET, before any run of this study)

## Question
Yesterday's 2x2 showed that reading a Swedish problem costs ~21 points, and the cost is about twice as large on
problems whose Swedish text contains number words (−30 vs −14 points, observational). The model's own reasoning
shows misreadings such as "sju" -> "ju", "tio" -> "tío", and "Åtta mer än fyra gånger" losing the eight.
Tokenizer check (done before this file): many Swedish number words are split into sub-words that are other words
(" s"+"ju", " n"+"io", "åt"+"ta", " n"+"oll"); embeddings are tied to the output head.

**H_num:** the number-word misreading is a narrow failure that a small targeted weight edit, built only from
synthetic number-word data (never from GSM8K), can fix, raising accuracy on held-out Swedish GSM8K problems that
contain number words.

## Data
- TEST: the 200 Swedish GSM8K TEST problems, split by the digit normaliser (`src/normalize_numbers.py`) into
  NUM (text contains a Swedish cardinal number word, 'en'/'ett' excluded) and NONUM (the rest). The split rule is
  mechanical and fixed here. It was motivated by yesterday's look at TEST outputs; this is stated in the write-up.
- DEV: GSM8K-train Swedish DEV pool (ids 200-299), NUM subset, for choosing the strength of edit B only.
- SYN: synthetic Swedish word problems with number words (templates, names and objects not from GSM8K):
  SYN-train (edit construction / training) and SYN-test (held out; direct target metric).

## Conditions (unedited Qwen3.5-0.8B rev 2fc0636 unless stated)
- base; digits (base model, Swedish problems with number words rewritten as digits: the free alternative)
- B: paper-style rank-1 directional edit: u = mean residual at the number position for word minus digit
  versions (SYN-train), two layers with the largest separation in 6..18, top-64 down_proj columns, alpha in
  {1, 2, 4} chosen on DEV-NUM (rule below); B-twin: random unit direction, same layers/columns/alpha.
- C: gradient rank-1 LoRA on layer-15 down_proj trained on SYN-train (fixed recipe, no selection);
  C-twin: random rank-1 dW with the same Frobenius norm on the same matrix.
- D: conventional comparator: LoRA r16 on all linear layers, same data and recipe.
- Dropped before running: embedding-row edit (A). The words that fail are multi-token and share sub-word rows
  with other words, and the embeddings are tied, so a row edit cannot target them.

## Evaluation
- Cells: Swedish problem with default (Swedish) reasoning, and Swedish problem with the English answer prefill.
  NUM: 8 samples per problem; NONUM: 4 samples. Sampling as before (T 0.7, top_p 0.8, top_k 20, presence 1.5,
  4096 tokens, seed 0).
- Regressions: FLORES Swedish perplexity, en/ett agreement AUC, Swedish answer share, NONUM accuracy,
  50 English GSM8K TEST problems x 4 samples (edits only).
- Statistics: paired bootstrap 95 % CI over problems (5000 resamples, seed 0).

## Selection rule for B (DEV)
Among alpha in {1, 2, 4}: answers >= 95 % Swedish, then highest DEV-NUM accuracy in the default cell; ties -> smaller alpha.

## Decision rule (primary metric: NUM accuracy, default Swedish-reasoning cell)
- Supports H_num: an edit gains >= +5 points over base on NUM with its CI vs its random twin excluding 0, NONUM
  does not drop by more than 3 points, FLORES perplexity rises < 5 %, answers stay >= 95 % Swedish, and the gain
  is >= half of the digits gain on NUM.
- Partial: gain over twin significant but smaller than half of the digits gain, or a regression.
- Challenges H_num: no edit beats its twin on NUM; or gains come only by switching answer language; or NUM and
  NONUM rise equally (general effect, not number reading).

Split sizes from the normaliser (recorded ~17:10, before any model run): TEST NUM 59, NONUM 141; DEV-NUM 41.
(Yesterday's rough count of 86 also counted 'dubbelt', 'hälften' and the like; the pre-registered rule counts cardinals only.)

Correction (17:30, before any twin was evaluated): B-twins are norm-matched per layer (||dW_twin||_F = ||dW_B||_F),
as in the main study. The first twins (same alpha) changed the weights ~3x less than B and were replaced
(numword/src/build_Btwin_matched.py).

Exploratory addition (18:15, after seeing C and D results; NOT pre-registered): C and D copied the one-line format of
the synthetic solutions (median 67-94 tokens vs 485 for the unedited model) and lost step-by-step reasoning on GSM8K.
To separate number reading from format, C2/D2 use self-distilled targets: the unedited model's own correct solutions of
the DIGIT version of each SYN-train problem (sampling as TEST, 2 samples), used as targets for the WORD version.
Same recipes as C/D otherwise. Reported as exploratory.
