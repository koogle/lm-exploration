# Expanded EU continuation validation

Completed: **7,620,187 matched targets**, 4,615 windows and 2,782 source-family
selections per checkpoint. All 12 original categories worsened, with category-wise
95% paired intervals above zero; ECB and EMA improved. Python loss rose from
1.559 to 3.994. See the [full results and intervals](report.md),
[plot](expanded-validation.png), and [aggregate JSON](comparison-20261019.json).

Compare the original 124M/5B checkpoint and the completed 878,610,362-token
ECB/EMA continuation on identical held-out windows. The small original diagnostic
used four windows per original source and 32 EU prefixes per institution. This
study samples up to 256 distinct families per category, one document per family,
and up to two randomly positioned, nonoverlapping 2K windows per document.

Family selection, document selection and offsets are deterministic with seed
20261019. All available families are selected when a category has fewer than 256.
Document and family coverage, tokens and nonprefix-window counts are reported.
No test split is used, no model is trained and no HF repository is modified.
Existing prepared raw databases are reused and checksum-verified.

Two H100 jobs score the checkpoints independently on the same checksummed
sample. Both use full FP32, TF32 disabled, evaluation mode and exact next-token
labels including EOS. Trailing padding is ignored. Token NLL sums accumulate in
FP64. Window identities, parent identities and target counts must match before
comparison. Partial scoring is durable and can resume with identical identities.

Primary statistics are token-weighted loss over this capped family sample, the
paired difference after-minus-before, and its 95% percentile bootstrap interval
from 2,000 document-family resamples. Multiple windows from a family are resampled
together. Intervals are category-wise, conditional on sampled families, and are
not simultaneous multiple-comparison intervals. The sample is not corpus-token
uniform; these losses are not whole-corpus perplexities. Teacher-forced 2K loss
does not establish generation correctness or 8K context quality. Existing source
split, near-duplicate and contamination audit limitations remain.

Private documents, sampled token IDs and per-window records remain on Modal.
Only aggregate statistics, counts, hashes and reproducible source are exported.

Run from `training/`:

```sh
python -c 'import truststore,runpy; truststore.inject_into_ssl(); runpy.run_module("modal",run_name="__main__")' deploy modal_eu_expanded_validation.py
```

Explicitly spawn the deployed `start(run_id, provenance)` through the authenticated
SDK. Deployment alone starts no work. The isolated app is
`simple-llm-eu-expanded-validation`; run prefix
`simple-llm-eu-expanded-validation-20261019`. The launch record stores the durable
controller call ID. Use its `inspect(run_id)` function to read aggregate status,
coverage and comparison. Nothing redeploys the training app.

CPU checks validate order-independent family selection, one-document-per-family
sampling, valid deterministic offsets and paired token-weighted aggregation against
a hand-calculated example, including clustered uncertainty and mismatch rejection.
