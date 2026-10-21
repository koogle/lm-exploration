# Exploration

[Latent workspace](latent-workspace/) contains the existing frozen-Qwen,
representation-conditioning and speculative-decoding experiments, including the
DFlash-style drafter, Modal runners, recorded results and unresolved questions.

The large-data curriculum and qualification study are complete. Start with the
[short learnings write-up](latent-workspace/docs/implementing-a-diffusion-style-drafter-from-scratch-and-learnings.md)
or [research PDF](latent-workspace/results/drafter-learnings.pdf).
The selected 79M drafter for frozen Qwen2.5-0.5B achieved 3.94 consecutive new
matches out of seven on the held-out test, and all 24 longer verified rollouts
matched the target. See [qualification findings](latent-workspace/results/qualification-findings.md)
for the protocol and limits. These results do not establish a general speedup;
further J-space and representation experiments remain separate, unfinished work.

Run its commands from `exploration/latent-workspace/`. Its research trains a
drafter around an existing frozen model; [training](../training/) is the separate
effort to learn how to train a language model from scratch.

Implementation lives here alongside the reports: [model and decoding code](latent-workspace/src/latent_workspace/),
[drafter training](latent-workspace/src/latent_workspace/dflash_train.py),
[Modal training runner](latent-workspace/modal_dflash.py),
[J-space study](latent-workspace/jspace_study.py),
[Modal J-space runner](latent-workspace/modal_jspace_study.py),
[configs](latent-workspace/configs/) and [tests](latent-workspace/tests/).

[Fixed-teacher rollouts](teacher-rollouts/) is a separate, bounded Qwen3.5-9B
Modal generation benchmark for future distillation. The harness is implemented;
GPU measurements await an explicit budget. It does not launch student training.
