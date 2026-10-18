# Simple LLM repetition: external baselines and corpus patterns

This bounded diagnostic compares the final Simple LLM checkpoint with Pythia-160M at approximately 4.19B and 299.89B training tokens, and SmolLM-135M at 600B tokens. These are base models, not instruction-tuned models. No training updates or Hugging Face writes occur.

## Primary comparison: full FP32 and equal text length

To compare different tokenizers fairly, this table measures repeated normalized character sixteen-grams in the **first 128 normalized characters**, using the identical eligible prompt/seed pairs in every model. All ten greedy prompts qualify. The T=0.1 column uses all twenty pairs; T=0.8 uses nineteen of twenty; T=1.0 uses all ten. Shorter outputs are excluded jointly across every model, not separately.

| Model | Greedy | T=0.1 / top-k50 | T=0.8 / top-k50 | T=1.0 / top-p0.95 |
|---|---:|---:|---:|---:|
| Simple LLM 124M / 5B | 32.8% | 30.5% | 4.7% | 2.3% |
| Pythia 160M / 4.19B | 43.6% | 38.7% | 9.8% | 0.0% |
| Pythia 160M / 299.89B | 13.6% | 16.7% | 2.5% | 1.2% |
| SmolLM 135M / 600B | 12.6% | 11.2% | 1.4% | 1.7% |

Our early looping is not an outlier relative to a published model at a similar training-token budget on this small set. Longer-trained Pythia and SmolLM are substantially less repetitive in these equal-length greedy prefixes. Within Pythia, the architecture and tokenizer stay fixed, making its early/final comparison more informative about training duration than a comparison between unrelated models. This does not isolate data quality, optimization or every potential implementation issue in ours.

Longer outputs still expose loops. Over entire capped completions, normalized **word** four-gram repetition for greedy is 71.8% in ours, 66.7% in early Pythia, 55.9% in final Pythia and 36.9% in SmolLM. Word/character metrics and length-controlled prefixes measure different aspects; neither is an accuracy score. Raw native-token four-gram rates are retained but should not be compared directly between tokenizers.

**Precision is not the fix for our model:** full FP32 gives 77.20% native-token greedy repetition versus 77.12% in the earlier BF16 run, with nine of ten greedy completions token-identical. Sampled values change with precision and seed, but the severe greedy problem persists. This negative result prevents transferring the Pythia-specific precision finding to our model without evidence.

All 180 FP32 external completions and sixty matched own FP32 completions are included. The own raw artifact retains twenty additional preset completions for provenance. The [paired-prefix summary](paired-prefix-summary.json) lists every included prompt/seed; [full metrics](metrics-table.md) expose lengths, EOS and coverage. The primary comparison uses native tokenizer prefixes for external models and the trained BOS token for ours.

## Initial BF16 results — precision-confounded

Repeated normalized **word** four-grams over whole generated completions (ten greedy outputs; twenty at T=0.1 and T=0.8; ten at T=1.0):

| Model | Greedy | T=0.1 / top-k50 | T=0.8 / top-k50 | T=1.0 / top-p0.95 |
|---|---:|---:|---:|---:|
| Simple LLM 124M / 5B | 71.6% | 70.5% | 14.9% | 3.6% |
| Pythia 160M / 4.19B | 54.0% | 54.9% | 11.2% | 1.1% |
| Pythia 160M / 299.89B | 66.8% | 8.1% | 2.1% | 0.0% |
| SmolLM 135M / 600B | 39.5% | 36.7% | 7.9% | 2.2% |

These initial BF16-autocast measurements are retained as a negative result, not a clean model-quality comparison. An independent Transformers `generate` check reproduced both tested manual BF16 greedy outputs exactly, but full FP32 inference substantially changed them in this PyTorch/Transformers setup. This is not a claim that BF16 universally breaks these models. For example, Pythia with its tokenizer-default prefix answered the France prompt with Paris in FP32 and a repeated “a state” sequence in BF16. Prefixing EOS also changes continuations. It would therefore be misleading to infer from the initial table that final Pythia normally repeats this badly. [Verification outputs](pythia-verification.json) retain all eight combinations of two prompts, FP32/BF16 and with/without an extra EOS prefix.

The initial BF16 metrics and eligible subsets are retained separately in [initial metrics](metrics-table-bf16.md) and [initial paired prefixes](paired-prefix-summary-bf16.json). They should not be substituted for the corrected primary comparison.

Manual review of the initial BF16 outputs illustrates why repetition and competence need separate checks: SmolLM greedily starts the France answer with Paris but then repeats the answer. It starts addition with `return a + b`, gives a correct recursive factorial base case and describes the Sun/planets/moons. Our earlier continuations were weaker on these beginnings. Conversely, varied sampled text can still be wrong. This is qualitative review, not an independently scored factual/programming benchmark.

## Comparison protocol

The same ten prompts and 128-new-token cap are used throughout. Settings: greedy; temperature 0.1/top-k 50 with seeds 1000+prompt index and 2000+prompt index; temperature 0.8/top-k 50 with the same two seeds; temperature 1.0/top-p 0.95 with seed 1000+prompt index. The primary Simple LLM outputs come from a fresh FP32 run of the same pinned final checkpoint. Initial BF16 Simple LLM outputs come from the earlier pinned-checkpoint studies. The initial external models used PyTorch 2.6, H100, FP32 weights and BF16 autocast, without KV caching. The corrected primary external comparison uses full FP32 forward passes with TF32 disabled and tokenizer-default prefixes. Initial prefix choices were explicit: Simple LLM BOS, a prepended Pythia EOS token as a beginning-of-document marker, and SmolLM tokenizer defaults. The corrected external comparison removes the extra Pythia EOS prefix. The independent Pythia sensitivity check also tests tokenizer defaults without that marker. External models use Transformers eager attention; Simple LLM uses its original custom SDPA implementation. This is a shared generation protocol, not bit-identical implementations or an architecture-controlled training experiment.

Token four-gram rates use each model's tokenizer and therefore are not directly comparable across models. Word and character metrics use lowercased, whitespace-collapsed text. Word four-grams use Unicode word sequences; character sixteen-grams use overlapping characters. Full-output metrics remain sensitive to completion length. First-256-character and first-32-word metrics additionally control text length, and explicitly report coverage: short or EOS-ended outputs are ineligible rather than silently padded. [Paired prefix metrics](paired-prefix-summary.json) additionally use identical prompt/seed intersections across all models at exactly 128 or 256 normalized characters, so their eligible subsets match. No single repetition metric measures correctness, readability or useful code.

All outputs, seeds, generated IDs, model revisions and package versions are retained in [external-results.json](external-results.json). [All external completions](external-completions.md) preserve text using JSON escaping. [Summary](summary.json) and [metrics table](metrics-table.md) are generated by `build_report.py` from raw outputs. The ten prompts were used in prior investigations and are not an independent held-out generation benchmark. Recipe selection should add a frozen independent prompt set instead of optimizing only these examples. Ten greedy outputs or twenty sampled outputs per setting are exploratory evidence, not population estimates.

## Corpus evidence

The [read-only corpus audit](corpus-patterns.json) verifies the same manifest hash as the completed run and samples the first 64 eligible training documents per source in content-hash order: 768 documents total. It measures the first eight nonoverlapping complete 128-token windows per document, giving each eligible document equal weight. Documents shorter than 128 tokens contribute no window metric. This corrects the earlier comparison between whole-document repetition and short generated continuations.

| Source | Whole-document token4 | 128-token-window token4 | Duplicate normalized lines within document |
|---|---:|---:|---:|
| Educational web | 4.8% | 1.9% | 0.0% |
| Python | 25.0% | 7.9% | 10.1% |
| Mathematics | 18.6% | 7.1% | 3.5% |
| arXiv | 31.0% | 3.6% | 1.1% |

These short real-data windows are substantially less repetitive than the model's approximately 77% greedy generated-token rate. The whole-document rates exaggerated the apparent similarity. Windows come from documents rather than the same prompt continuation distribution, so even the controlled-length comparison is descriptive rather than causal.

Across the sample there are 229 normalized line hashes appearing in more than one document. The three most frequent shared lines were table-wrapper extraction markers (29 and 19 documents) and a competing-interests declaration (14 documents), confirmed by hash lookup in the frozen local database. Shared lines can also include legitimate imports, licenses and equations. No exact duplicate full text was found in the sample, which is expected because the database uses a content-hash primary key; this does not establish global near-deduplication or clean split separation.

[Manual inspection](manual-inspection.json) of six deliberately selected high-repetition document beginnings found browser-compatibility/navigation boilerplate in one mathematics page and a repeated question quoted in a forum reply in another. Python examples contained repeated unit-test scaffolding and license headers; academic examples contained author/affiliation and mathematical structure. These are concrete candidates for targeted extraction cleanup, not a prevalence estimate. Preserve provenance and licensing metadata when excluding boilerplate from training text.

## Training-recipe review

These are document-continuation models. Continuing beyond a short factual answer is expected; repeating it indefinitely is a separate defect. Lack of EOS within a 128-token cap alone does not establish missing EOS supervision.

The completed model saw 4,999,981,176 valid next-token targets from approximately 2.745B eligible unique tokens, with at most two passes per source. It used 14 layers, width 768, 12 heads, 32K tied embeddings, RMSNorm, RoPE and SwiGLU. AdamW peak LR was 3e-4, warmed over 20M tokens and decayed to 3e-5 by 5B. The global batch was 32 document-local windows, approximately 78K valid targets per update; 8K padding left approximately 30% of positions valid.

The retained earlier audits passed exact target shift, EOS, padding-label, document-boundary and cursor-resume checks on 280 real windows; independent attention/normalization/RoPE/causality references and a two-rank CPU/Gloo accumulation/AdamW reference also passed. They do not prove the entire CUDA/BF16 pipeline bug-free. Ignored right-padding explains wasted compute, not direct training on PAD targets, and causal masking prevents those trailing positions influencing valid earlier logits.

The mixture was 10% educational web, 30% Python, 20% mathematics, 15% scholarly material and 25% books. This differs from a prose-first educational recipe. Its low mixture-weighted held-out loss does not establish general prose competence, particularly because easily predicted structured sources contribute heavily. Do not compare loss numbers between models with different tokenizers. Global near-deduplication, some parent/split reconciliation and tokenizer/split-contamination checks remain pending; current held-out results are exploratory.

## Prioritized controlled experiments

1. **Extraction cleanup:** preserve model, tokenizer, source weights, valid-target budget and optimizer; compare baseline extraction with targeted navigation/forum-quote/boilerplate removal plus near-deduplication. Preserve license/provenance records. Evaluate source-stratified held-out loss, fixed prose/code continuations and repetition at fixed decoding settings. This tests a concrete data hypothesis instead of simply deleting all repeated mathematical/code structures.
2. **Mixture:** at the same architecture and valid-target budget, compare the current mixture with a more prose-heavy, curated educational mixture. Freeze disjoint parent-level evaluation sets before selection. This separates domain allocation from total token count.
3. **Optimization:** after choosing a fixed corpus, test a small LR sweep, for example 3e-4/6e-4/1e-3, with matched effective valid-target batches and the same schedule. Competitors' larger learning rates are a reason to test, not evidence our LR caused repetition. Keep context fixed for this comparison; test a 2K/4K-first, 8K-extension curriculum separately rather than changing both factors together.
4. **Longer training:** only after the bounded comparisons, extend a promising recipe to a staged 20–50B valid-token budget with additional curated unique data and an explicitly revised LR schedule. The existing cosine schedule has already decayed by 5B, so a continuation requires a deliberate schedule decision. Respect the two-pass/source policy; endlessly recycling the current corpus is a different experiment. No new training was launched by this investigation.

Special repetition-penalizing training objectives would be a later experiment after basic corpus and optimizer controls, because they can suppress legitimate repetitions without correcting knowledge or coherence.

## Reproduce

From this directory, with Modal credentials and read access to the project data volume:

```sh
modal run modal_baselines.py
modal run modal_simple_llm_fp32.py
modal run modal_corpus_patterns.py
modal run modal_verify_pythia.py
python build_report.py
python build_report.py --bf16
```

The own FP32 runner is derived from [PR #20](https://github.com/koogle/lm-exploration/pull/20), with a local output filename change; [source provenance](own-runner-provenance.json) records the source hash. A copy is included so this report does not require that PR to be merged. It verifies pinned model source, tokenizer and weight identity and resumes matching completed results. The separate `modal_baselines_bf16.py` reproduces the retained initial BF16 outputs; it writes a different filename.

The baseline job has three independently bounded H100 functions, each with a 30-minute timeout and no automatic retry. The corpus audit is CPU-only with a ten-minute timeout and read-only SQLite access. The local caller writes returned JSON here. In this managed environment, use the documented system trust integration when invoking Modal:

```sh
python -c "import truststore,runpy; truststore.inject_into_ssl(); runpy.run_module('modal',run_name='__main__')" run modal_baselines.py
```

Primary model documentation: [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m), [SmolLM training blog](https://huggingface.co/blog/smollm). The final Pythia training budget uses its published 299,892,736,000-token total; step 2000 is 2000 × 2,097,152 = 4,194,304,000 tokens. This within-Pythia checkpoint comparison measures behavior over training, while differences from our model also include architecture, data, tokenizer and optimization.
