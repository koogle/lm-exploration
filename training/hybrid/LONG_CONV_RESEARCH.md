# Longer learned convolutions around KDA: research precedents

The existing four-tap depthwise causal convolution is already learned. Its
coefficients vary by channel but are fixed across examples after training.
Lengthening its support, predicting its coefficients from the current input,
and replacing it with a compact recurrent filter are distinct design changes.
The present experiment changes only KDA's input preprocessing, retaining the
KDA recurrence. Most cited architectures made broader changes, so their results
are motivation rather than evidence of improvement in our stack.

## Historical map

| Period | Work | Connection to this experiment |
|---|---|---|
| 1989 | [Time-delay neural networks](https://doi.org/10.1109/29.21701) | Learned weights shared across temporal windows; temporal convolution predates attention by decades. |
| 2016 | [WaveNet](https://arxiv.org/abs/1609.03499), [ByteNet](https://arxiv.org/abs/1610.10099) | Causal dilated convolution stacks provide large receptive fields with small kernels. ByteNet applies them to language/translation. |
| 2016–17 | [Gated CNN language models](https://arxiv.org/abs/1612.08083), [QRNN](https://arxiv.org/abs/1611.01576) | Gated temporal feature construction, and convolution combined with recurrent pooling. |
| 2019 | [Lightweight and Dynamic Convolutions](https://arxiv.org/abs/1901.10430) | Kernels can be predicted from the current timestep, selecting which nearby positions matter. This differs from simply training a fixed kernel. |
| 2021–22 | [CKConv](https://arxiv.org/abs/2102.02611), [S4](https://arxiv.org/abs/2111.00396) | Parameterize long filters as continuous functions or structured state-space impulse responses. Kernel length need not dictate the number of free parameters. |
| 2022–23 | [MEGA](https://arxiv.org/abs/2209.10655) | Multi-dimensional damped exponential moving averages combined with gated attention: a close precedent for learned temporal preprocessing before retrieval. |
| 2022–23 | [H3](https://arxiv.org/abs/2212.14052) | Shift SSM on keys plus a diagonal SSM on key-value products, with multiplicative interaction. Explicitly separates temporal event detection from longer memory. |
| 2023 | [RWKV](https://arxiv.org/abs/2305.13048) | Learned current/previous-token mixing before receptance, key and value projections. Closely related to the strong one-token-lag coefficients observed in our filters, though mixing before and after projection is not generally equivalent. |
| 2023 | [LongConv](https://arxiv.org/abs/2302.06646) | Directly learns long convolution coefficients; studies regularization, smoothness and initialization rather than requiring an SSM parameterization. |
| 2023 | [Hyena](https://arxiv.org/abs/2302.10866), [FlashFFTConv](https://arxiv.org/abs/2311.05908) | Implicit long filters interleaved with input-dependent gating, plus GPU algorithms making long convolutions practical. Hyena also retains short depthwise convolutions in its projections. |
| 2023–24 | [Mamba](https://arxiv.org/abs/2312.00752) | Input-dependent state transitions provide selective temporal filtering. This selection breaks the ordinary fixed convolution formulation and requires a scan algorithm. |

The most direct reading order for our question is LongConv, MEGA, H3, Hyena,
then Lightweight/Dynamic Convolutions. RWKV is a particularly useful comparison
for the one-token temporal shift seen in the checkpoint diagnostics.

## Longer does not mean indiscriminate smoothing

LongConv's Squash operation is elementwise soft thresholding; its Smooth
operation averages neighboring kernel coefficients. These regularize the
*filter*. They do not require taking a positive uniform average of hidden
states. Signed filters can detect changes, delays and temporal patterns.

There is also a genuine qualification: Hyena's Appendix D.2 reports worse
solutions and slower convergence with excessively smooth filter initialization,
and Appendix D.3 discusses retaining high-frequency content. The papers study
different parameterizations and regimes. Neither establishes that more smoothing
is universally beneficial.

Our fitted q filters emphasize the current hidden state, while k/v filters
emphasize the preceding hidden state. Their timing structure and signs should
be preserved as an available solution when adding longer support. The diagnostic
ablations establish reliance on the local filter, not evidence that a long
uniform average would help.

## Concrete candidates

1. **Wider v convolution:** compare lengths 4, 8, 16 and 32 with the current
   RoPE-on-q/k architecture, keeping document masking and dtype fixed. Direct
   convolution is the first implementation to benchmark at these modest sizes.
   The small parameter increment does not imply a similarly small runtime cost.

2. **Short path plus gated long tail:** let `u_t = W_v x_t` and compute
   `v_t = SiLU(local_4(u)_t + g_t * long_tail(u)_t)`, where the tail begins at lag
   four and the gate starts small. Preserve the strong local path while letting
   the model learn when older inputs help. This is a proposal for our model,
   drawing on gated long-convolution designs; it has not been tested here.

3. **A small bank of recurrent timescales:** maintain traces
   `h[r,t] = a[r] * h[r,t-1] + b[r] * u[t]`, then learn a signed mixture of their
   outputs alongside the short path. Fixed coefficients give exponential
   convolution kernels; input-conditioned coefficients give a selective system.
   MEGA/S4 supply close precedents. A bank of R traces per value channel adds a
   latent dimension with O(R*d_v) state, rather than R copies of the entire KDA
   matrix. We previously tried a related four-mode rotating-memory mixer on all
   q/k/v paths in the short pilot; its extra runtime was not justified. A v-only
   residual bank preserving the local convolution is different, but unproven.

4. **Input-dependent mixtures of filters:** learn a small set of signed basis
   filters and choose their mixture from the current hidden state. This is a
   practical restricted dynamic convolution, avoiding a separate fully dense
   length-L kernel per channel per token. It trades flexibility against basis
   count, gate cost, and decoding buffers.

For much longer finite kernels, CKConv/Hyena-style implicit parameterizations
and FFT algorithms become relevant. An arbitrary length-L FIR generally needs
O(L*d) history for straightforward exact autoregressive decoding. A structured
SSM can use a fixed-size recurrent state. FFT-friendly training alone does not
solve the decoding-cache problem. Packed documents also require actual segmented
convolution or per-document processing; ordinary FFT convolution across a packed
stream would leak across boundaries. Our current 1,024-token training context
cannot establish the quality of dependencies beyond that length.

## What Goodfire changes about the investigation

[Neural Networks Are Already Clean](https://already-clean.goodfire.com/) describes
itself as an informal research essay and working hypothesis. Its main relevance
is methodological: study native neurons and circuits, use their upstream and
downstream connections, test counterexamples and interventions, and observe
how mechanisms develop. It does not establish that longer convolutions improve
language-model loss, that our small model's channels have single clean meanings,
or that superposition is universally absent.

For our model, that suggests inspecting value channels before and after their
convolution, collecting both positive and negative activation examples, plotting
lag responses, and asking what each additional span computes. Test hypotheses
with activation patching and targeted lag/channel interventions on held-out
examples. Compare activation explanations with actual downstream effects.
Inspect q/k channels before RoPE when looking for stable coordinate-level
patterns, since rotation changes their coordinate representation by position.
KDA state entries are dynamic memory coordinates, not automatically independently
interpretable semantic neurons. The model also has SwiGLU blocks, where looking
only at positive gate activations can miss signed interactions discussed in the
essay.

The practical synthesis is to test longer *structured temporal features* while
measuring what they add beyond the short path. Interpretability can guide those
tests; validation loss, task behavior, throughput and decoding state remain the
criteria for choosing the architecture.
