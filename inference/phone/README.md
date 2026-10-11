# Phone and iPad inference

Explore what we can squeeze out of an iPhone 17 Pro and iPad Air M4 using our own
Metal inference kernels.

## Current material

- [Model selection and kernel roadmap](research/iphone-model-selection.md): start
  with SmolLM2-135M-Instruct in FP16, validate a complete forward pass, then optimize
  decode, dispatches and packed weights against reference outputs.
- [iPad interface studies](designs/ipad-design-studies.html): the selected visual
  direction is **Token Scope** (layout C). These historical studies use Qwen3.5-2B;
  that is distinct from the smaller first kernel implementation target above.
- [Historical GPT-2 artifact](../web/models/gpt2/README.md): retained with the web
  experiment, including its license, source revision and checksum.

No native inference engine or custom Metal kernels are implemented yet. The
designs are mockups; their timing estimates are not on-device measurements.
Open the HTML file locally to explore the layouts.

Use standard runtimes as correctness and speed references. The long-term goal is
to run our own [trained model](../../training/) through these custom kernels.
