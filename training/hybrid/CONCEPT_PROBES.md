# Native KDA concept probes

This study starts from `main` revision `6d2f9ab`, using its unchanged hybrid model,
fused convolution, delta recurrence, loss and Muon/AdamW implementation. It is an
exploratory small-model study, not a rerun or continuation of the architecture
campaign. Source is on `study/kda-concept-probes`.

The motivation is Nick Cammarata's [Neural Networks Are Already Clean](https://already-clean.goodfire.com/):
inspect native mechanisms, both activation signs, surrounding context and downstream
consequences before inventing a feature decomposition. The essay is a working
hypothesis, not evidence that KDA matrix entries are monosemantic. Its own discussion
leaves the appropriate level of attention-head grouping open.

## Training and bounds

One random seed (17), approximately 98M parameters, 12 layers, width 448, seven
64-dimensional KDA heads, the main 3:1 hybrid pattern, no MTP auxiliary head,
1,024-token windows. Existing GLM-tokenized hybrid-40b-v1 prepared data provides
70% ClimbMix, 20% code, 10% math; last shards are held out. Target 12,208 updates
of 8,192 positions (100,007,936 token positions). This is deliberately small and
likely undertrained; failed semantic tasks cannot establish absent representation.
Muon/AdamW uses main defaults, 200-update warmup and cosine decay to 10%.

One H100, six-hour absolute launch budget (minus two minutes), execution timeout
19,800 seconds, retries zero. An independent remote supervisor tracks and cancels
the worker, including queued time. Training reserves 90 minutes for probes.
The worker saves metrics every 256 updates and a resumable checkpoint every 1,024.
No automatic restart. Generated artifacts belong in the private HF dataset
`jfrick/lm-exploration-results`, under `training/results/<run>`.

## Measurements

1. Check the instrumented FP32 recurrence against main's reference; verify that
   self-patching is a no-op, prefix outputs are unchanged and EOS resets erase a patch.
2. Run an untrained control on the same model seed before optimization.
3. Build a signed native read-channel atlas from 24 held-out windows (256 tokens
   each), beyond the validation prefix. Fix 16 evenly spaced head/value coordinates
   per KDA layer before inspecting activations. Retain the strongest six positive
   and negative examples with preceding and following tokens. These are raw
   `q^T S` reads before normalization/output gating; they are not neuron labels.
4. Probe color (red/blue), location (Paris/Rome), and code boolean (True/False)
   counterfactuals in KDA layers 0, 4, 8. Four discovery entities select a head and
   value column by mean squared state difference; four distinct test entities
   evaluate it, with 0, 32, 128 identical distractor tokens. There is no probe-task
   fine-tuning and no claim of lexical generalization beyond these alternatives.
5. In both directions, transplant whole state, selected head, selected value
   column, another head, or a norm-matched random state difference. Also zero the
   selected head. Change only the recurrent state after the prefix output; preserve
   convolution histories and other layers. Report raw alternative log-odds and
   oriented causal effects, plus clean task competence. No unstable division by a
   near-zero clean gap. The untrained control uses two test entities, zero lag.

A detectable feature is not necessarily used; a patch effect is not sufficient to
name a concept. Only interpret selective, held-out effects against controls when
the model actually distinguishes the alternatives. Full-attention layers can also
carry the fact, so a negative single-layer intervention is not absence of memory.
Read coordinates are basis-dependent; evaluate head/column organization rather
than claiming each matrix cell is a permanent semantic slot.

## Execution

From `training/`, supply the verified tokenizer at
`artifacts/concepts/glm-tokenizer.json`, then run `launch_kda_concepts.py` with the
Modal environment (truststore is injected before SDK calls). The launcher records
source hashes, main revision, worker/supervisor IDs and the absolute deadline.
The worker runs qualification, untrained probes, training and trained probes in
that order and records failures in `completion.json`.
