# Post-training

Research that starts from an existing pretrained model and changes what it can do.

[Detection](detection/) post-trains a small vision-language model to detect requested
objects in one forward pass. It holds the evaluation harness, measured baselines for
Moondream 2 and Moondream 3.1, custom CUDA decode kernels and a first training run of
a slot head with regressed boxes and one-pass diffusion labels. Run its commands from
`post-training/detection/`.
