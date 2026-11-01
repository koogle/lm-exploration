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

## Launch record

Run `kda-concepts-1793489611` is launched on
[Modal](https://modal.com/apps/radiantai/main/ap-0rR2mM5h8KzvdYxB8Lsfb2).
The recurrence qualification passed on GPU: zero difference against the FP32
reference, self-patch identity, unchanged prefix outputs and correct EOS reset.
The untrained controls completed and the first optimization step completed with
finite loss and gradients. The model has 98,295,828 parameters; initial held-out
CE is 12.0406. Training and final probe results are pending. This does not yet establish any
interpretable concept or useful model capability.

The preceding launch `kda-concepts-1793489410` failed in the supervisor's import
path before GPU training. Its app was stopped, the supervisor was given the
project image, and the failed record was archived in the same HF results dataset.

## Completed findings

The run completed 100,007,936 training positions and all probes in 2,874 seconds
(about 48 minutes). Held-out CE decreased from 12.0406 to 4.7181. This is one
undertrained 98.3M model, not evidence about a mature billion-parameter model.
Artifacts, including the 1,296 trained intervention rows, are archived in the
private dataset `jfrick/lm-exploration-results`, folder
`training/results/kda-concepts-1793489611`.

The clearest localized effect is the Paris/Rome contrast in block 8, head 3,
value column 13 (all zero-based). The head was chosen using four discovery names;
the effect was tested on four other names in both transplant directions. Head
transplants moved answer log-odds toward the donor in all 24 tested cases across
0/32/128 distractor tokens. Mean oriented effects were 0.0781, 0.0469 and 0.0410
nats. Column-only effects were 0.0605, 0.0273 and 0.0215; other-head and random
whole-state-direction controls were near zero or negative. These are small
predictive shifts, not answer flips: the model preferred Paris even in Rome
contexts. The result supports a causally used state component carrying this
specific contrast, not a general location neuron or monosemantic matrix cell.
The column intervention acts on all key rows of a value column, not one scalar.

The model correctly preferred red versus blue for every held-out test pair at
all three distractor lengths, but swapping one tested KDA layer's recurrent
state had little effect. This does not locate color memory in full attention:
other KDA layers, convolution histories and full attention remained intact.
The experiment demonstrates limited leverage of these particular single-layer
interventions, not absence of color information in KDA. Boolean preferences
remained biased toward True and did not demonstrate reliable copying.

Numerical precision limits interpretation. Answer logits were computed under
BF16 autocast then cast to FP32, leaving visibly quantized log-odds. Several
color/boolean effects are at this resolution. The trained production/reference
log-odds qualification discrepancy was at most 0.001953; self-patch errors were
zero. A follow-up should compute the final readout in FP32 and replicate across
seeds, more entities, templates and lexical alternatives before interpreting
small effects or claiming semantic selectivity.

The signed read-channel atlas contains 144 fixed coordinates over 24 held-out
windows. It supplies examples for hypotheses, not validated semantic labels:
several apparent clusters are adjacent tokens in the same passage. The study
therefore does not confirm the essay's broad cleanliness hypothesis. Its useful
output is a reproducible native-state causal assay and one modest localized
memory effect, with explicit competence and precision limitations.
