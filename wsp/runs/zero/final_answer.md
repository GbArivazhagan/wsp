# Final answer: zero-pass condition (run `zero`)

## Decision: NO EDIT
I did not call `save_edit`. `runs/zero/edit/` is empty, so the "edited" model is the base Qwen3.5-0.8B.
Forward passes used: 0 of 0. Scripts run: 7 of 40 (archived in `runs/zero/code/000-006`).

## What I looked at
1. **Architecture and code** (`harness.py`, `agent_gen.py`, HF `modeling_qwen3_5.py`). The model has no linear biases. Its RMSNorm is `x̂·(1+w)`, so I fold the factor `1+w` into any direction analysis. Decoding is T=0.7, top_p 0.8, top_k 20, **presence_penalty 1.5**.
2. **Tokenisation (run 000).** Swedish problems take 1.74 tokens per word and English ones 1.32. Swedish content words split into rare pieces: " bananer" is " ban"+"aner" (id 206422), " högar" is " hö"+"gar", " klasar" is " klas"+"ar". The English words are single tokens (" bananas", " monkeys").
3. **Example outputs (run 001).** 64-72 % of output tokens had already been emitted earlier in the same answer, so the presence penalty hits most tokens. The corrupted forms in the WRONG answers ("bananter" = " ban"+"anter", "klassar", "högare") match what that penalty would cause: the correct continuation piece has already been used, so a fresh piece wins and the word changes meaning (klasar "bunches" becomes klassar "classes"; högar "piles" becomes högare "higher"). This is a hypothesis I could not test. If true, part of the Swedish deficit comes from decoding acting on low-confidence Swedish word pieces, and a weight edit can only counter it by sharpening logits. Sharpening amounts to a temperature hack, is not Swedish-specific and goes against the spirit of rule 4. I did not do it.
4. **Searching for a Swedish-language direction in the tied embedding (runs 002-005).** I tried four estimates:
   - Swedish-only vs English-only token types. Held-out AUC 0.86, but the direction turned out to be "frequent English word vs rare fragment". Its top tokens are spam and punctuation junk.
   - The same, frequency-matched by token-id bins. AUC within bins 0.85-0.99, but the Swedish side is sub-word fragments with no leading space ("ud", "ur", "ren").
   - Word-initial tokens only. AUC 0.77. The Swedish side is short ambiguous starts (" de", " en", " se"), and Spanish and German probe words score as high as Swedish ones.
   - 77 hand-made single-token Swedish/English **translation pairs** (och/and, varje/each, …). The direction is coherent: held-out pair-sign accuracy 0.986, mean-difference coherence 0.40. But projected over the vocabulary it is mainly "frequent English tokens vs everything rare": Danish/Norwegian −0.03 and Swedish −0.04 sit far from English −0.35, and German, Spanish and French fall in between.

   Conclusion: from weights alone, language identity in the embedding space cannot be separated from frequency, training amount and how a word is split into tokens.
5. **Per-layer read and write gains of the translation-pair direction (run 006).** This compares ‖W·diag(1+g)·d‖ with random unit directions, for every input and output matrix. Layers 1-12 read and write it at random level (gain 0.89-1.10). Only layer 0 (1.42) and layers 16-23 read it more strongly (up to 1.56 in layer 23's MLP), and those layers set the output language and the next-token choice.

## Reasoning: why no edit
The only weights-only edit with some support in the literature is to make the middle layers blind to language identity: project the language component out of the input matrices of layers ~8-15 (W' = W·diag(1+g)(I−uuᵀ)diag(1+g)⁻¹), leaving the late layers alone so the answers stay Swedish. Published versions (INCLINE, LRP2, LSAR) estimate the direction from activations. From weights alone I would have to use an embedding-space direction, and the analysis shows two problems with that:
- (a) In layers 8-15 that direction is not special: they read it at random level. Projecting it out would be close to deleting 1 random dimension out of 1024, so roughly a no-op.
- (b) To the extent it carries information, that information is word frequency and tokenisation structure (fragment vs word start). Removing it is more likely to hurt both languages than help Swedish.

The real mid-layer language subspace is built by the layers themselves and cannot be seen without activations. Other options I considered and rejected:
- **Boosting copy or induction heads** to protect Swedish word pieces from the penalty: I cannot tell from weights which heads they are, it risks repetition loops, and it also affects English.
- **Amplifying or ablating Swedish-language neurons:** the literature shows this changes the output language and perplexity, not reasoning.
- **Global logit sharpening:** a decoding hack that is not Swedish-specific.

The evaluation noise is about ±3 pp (200 problems, sampled), so any edit I could justify would most likely be undetectable or slightly harmful.

## Exact edit
None. 0 tensors changed, 0 values changed.

## Confidence
- No edit: the Swedish and English accuracy changes are exactly 0 apart from sampling noise.
- My best candidate (mid-layer projection with an embedding-derived direction), had I applied it:
  - P(raises Swedish accuracy detectably, about +3 pp or more, while keeping answers Swedish and English intact) ≈ 0.07
  - P(roughly neutral) ≈ 0.65
  - P(hurts) ≈ 0.28

That is not good enough to justify an edit.

## What I would do with forward passes
1. **`generate` base on the 100 DEV problems.** Measure the noise floor and tag the failures: word corruption from the presence penalty, misread problems, or arithmetic slips. This tests the penalty hypothesis directly.
2. **`activations` on 128 parallel sv/en problem texts, all layers.** Get μ_sv and μ_en of the residual stream per layer and per-neuron MLP activation differences. This gives the real language subspace in the residual and shows where Swedish and English part ways.
3. **Asymmetric, bias-free edit.** In the middle layers where the difference is large but the output language is not yet decided, project out u = (μ_sv−μ_en) orthogonalised against μ_en, folded through `1+g`, from the input matrices. By construction English activations barely see it. Keep layers ≥16 untouched so the output stays Swedish. Then check with `activations(which="edited")` that English residuals barely move.
4. **`generate(which="both")` on DEV** to confirm the accuracy gain and that answers stay Swedish, then sweep the projection strength and layer range over a couple of passes.
5. If the presence-penalty failures dominate, report that the gap is partly a decoding artifact rather than force a weight fix.
