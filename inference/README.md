# Inference

- [Web](web/): preserved browser/WebGPU prototype and LM implementation notes.
- [Phone](phone/): native inference on iPhone 17 Pro and iPad Air M4, with
  handwritten Metal kernels and measured hardware limits.

The eventual integration target is a model trained in [training](../training/)
running on our Apple hardware through our own inference kernels. Start with
pretrained models and reference runtimes to establish correctness and useful
performance comparisons.
