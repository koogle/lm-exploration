# Simple LLM training curves

Build reproducible scientific plots from completed Modal segment logs:

```sh
python analysis/loss-curves/build_report.py --input /path/to/downloaded/logs --output /path/to/figures
```

The input contains `<run-id>-sNN/{metrics.jsonl,validation.jsonl,report.json}`.
The builder verifies completed token counters, monotonic updates, and phase lineage;
it rejects replayed update records. CSV includes all logged updates, while figures
show token-weighted 128-update loss bins and subsampled individual updates.
Optimizer plots include pre-clipping gradient norm, learning rate and step throughput.
Phase mixtures differ; their losses are not interchangeable quality measurements.
Old validation curves retain their one-window diagnostic limitation. New overnight
curves use frozen representative held-outs and are labeled accordingly.

Historical completed phases end at 10,878,562,619 lifetime targets. The generated
figures/CSV/provenance are retained privately on the workspace and attempted for
publication under `evaluation/training-curves` in `jfrick/124M-10.879B`.
The upload was rejected by Hugging Face's private-storage quota. No successful
publication is claimed. Overnight finalization retains curves on Modal and retries
private publication, recording any rejection independently of training success.
