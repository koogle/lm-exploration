# Repeat repair SFT from the newest 124M base

Completed 20,000 updates / 41,579,095 supervised answer tokens on Modal using the 20.879B-token parent:
`simple-llm-code60-unique-10b-20261020-r01-s04/step-000153342.pt`.
The prior private-patch SFT initialized from the older 10.879B-token parent.
This run uses fresh AdamW state and the same fixed data and training settings.

- Run: `repair-sft-20b-private-patches-20261020-r01`
- Data: `repair-private-patches-20261020-v1`
- Dataset SHA-256: `6197c4178e737172184d830ed52a3da5a3251ba642a03e20d336d05143b93702`
- 79,441 training examples; 1,617 validation examples; 12 languages
- One H100, 20,000 updates, batch 8, context 2,048, LR 1e-5, seed 42
- Evaluate every 500 updates; save every 1,000; four-hour loop limit
- Training implementation unchanged; launcher now accepts a validated parent path
- Ten repair SFT tests and launcher lint passed

[Launch record](launch.json) contains the durable Modal call ID.
[Modal deployment](https://modal.com/apps/radiantai/main/deployed/simple-llm-repair-sft-scale).
The final checkpoint is `repair-sft-20b-private-patches-20261020-r01/step-000020000.pt`.
Validation answer-token loss improved from 0.712214 to 0.521979; all twelve
languages improved relative to this run’s baseline. Executable evaluation is complete: **0/18 correct repairs**,
10/18 reaching output comparison, and 12/18 emitting EOS.
[Full generations and diagnostics](generation-evaluation.json). The data hash is independently checked by the worker, and parent,
tokenizer, code and settings hashes are stored in the training artifacts.

This reuses the prepared 79K corpus, not the cancelled expanded-corpus preparation.
The newer base's pretraining included private code and patches, so SFT split
separation does not establish that the private validation examples were unseen
in pretraining. The unchanged synthetic executable diagnostic is also required
before claiming improved repair correctness; the newer base scored 0/18 before SFT.

The final checkpoint was evaluated with the same 18 original synthetic held-out
tasks across nine languages, seven cases per task, greedy BF16 decoding capped
at 128 tokens, and disposable network-blocked execution sandboxes. Grader
calibration passed in every language. The earlier 20.879B base scored 0/18; the
new SFT checkpoint also scores 0/18. This small diagnostic does not measure all
private repository repairs. The best observed validation loss was 0.519240 at
step 15,000; executable evaluation here uses the final step 20,000 checkpoint.
