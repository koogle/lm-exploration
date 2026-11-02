# Pre-training the 720M hybrid for agentic post-training (plan, 2026-11-02)

Goal: a pre-training mix and token budget for the 720M hybrid
([HYBRID_720M.md](HYBRID_720M.md)) after which fine-tuning on the same ~500k
trajectories as the Qwen reference (`scale-500k` in `modal_scripts/modal_agent_sft.py`)
resolves held-out PR tasks, and then SWE-bench Verified and Terminal-Bench 2.1 tasks.
Nothing here is implemented or launched; every stage below needs approval before it
spends compute.

## What the measurements say

| Model | Pre-training | Agent SFT trajectories | Held-out PR tasks resolved (of 179) |
| --- | --- | ---: | ---: |
| 720M hybrid, mid-training export | 40B + 3B tokens | 356 (GLM-5.3) | 0 |
| Qwen3-0.6B-Base | ~36T tokens (Qwen's report) | 10k / 20k (mini-coder) | 0 / 0 |
| Qwen3-1.7B-Base | ~36T tokens | 10k (mini-coder) | 1 |
| Qwen3-1.7B-Base | ~36T tokens | 20k–49k (public mixes) | 0–3 |
| mini-coder-1.7b (Qwen3-1.7B) | ~36T tokens | 396k (mini-coder) | 11 |

- Agent success at 1.7B needs hundreds of thousands of trajectories, and with 10–20k
  even a 36T-token 0.6B model resolves nothing. It is not yet measured whether a
  0.6B-class model resolves tasks at 500k; that measurement decides whether 720M is
  worth more pre-training (gate 0).
- The 720M base is far behind Qwen3-0.6B-Base on code (HumanEval 4.3 against 29.9;
  MBPP 5.0 against 38.8) after 40B tokens, about 900 times fewer than Qwen's.
- Mid-training with web text at 21% of tokens lowered HellaSwag and ARC by 2–3
  points (`hybrid-720m-mid32k-01`), so web text stays at half of the mix or more
  until the final anneal.
- Open-SWE-Traces trajectories average 23k tokens and 23% exceed 40,960 tokens, so
  the base must use context to about 40K. The 32K mid-training worked: loss falls
  with position.

## Size

**Model:** 720M (KDA:full 3:1, MTP, GLM-4.7-Flash tokenizer), unchanged. A larger
model (the ~1.5B variant in [HYBRID_V2.md](HYBRID_V2.md#stages)) is chosen instead if
gate 0 shows that a 0.6B model does not resolve tasks after 500k trajectories either.

**Tokens:** from the 30B export (`tokens-030000807936` in `jfrick/hybrid-720m-40b`,
taken before the decay started at 34B, so training continues at the constant
learning rate without re-warming):

| Phase | Tokens | Context | Cost at measured throughput |
| --- | ---: | --- | ---: |
| Stable (constant LR) | +120B, to 150B | 4K | ~$3,400 (371.8K tokens/s, $28.5 per billion) |
| Anneal (LR to 0) | 25B: 15B at 32K, 10B at 64K | 32K → 64K | ~$500 at the 32K rate (513K tokens/s); 64K not yet measured |
| Agent SFT on `scale-500k` | ~7.5B, one epoch | 40,960 | ~$250 (hybrid SFT measured 317K tokens/s) |

175B tokens is about 240 tokens per parameter, four times the 40B run. The decay
phase of the 40B run lowered held-out bits per byte by 11–21%, so the anneal carries
the code and agentic data, where it acts most per token.

## Mix

### Stable phase (+120B at 4K)

| Source | Share | Notes |
| --- | ---: | --- |
| ClimbMix (current web source) | 50% | keeps commonsense; CC BY-NC, accepted |
| Code files | 30% | github-code-clean allowlist (current), `jfrick/curated-oss-code-20261019`, `jfrick/mit-claude-codex-code-20261019`; an educational-quality filter in the style of Stack-Edu (scored Stack v2 files; contents come from Software Heritage) if the download is practical |
| Commits and issues | 5% | CommitPack (`bigcode/commitpack`, MIT metadata, permissive repositories): commit message plus diff, which is the patch format the agent produces |
| Math | 10% | FineMath (ODC-By, current) |
| Shell and tool documentation | 5% | man pages and tool docs in permissive collections, to be identified |

### Anneal (25B, 32K then 64K)

| Source | Share | Notes |
| --- | ---: | --- |
| Web, highest-quality slice | 30% | ClimbMix's top clusters or FineWeb-Edu (ODC-By) |
| Repository-level code (files of one repository in path order) | 20% | prepared by `hybrid-long32k-v1` |
| Commits and PR diffs | 10% | CommitPack; histories of our PR-task training repositories |
| Agent trajectories not used in SFT, loss on every token | 10% | Open-SWE-Traces unresolved trajectories (~59% of its mini-swe-agent subset) and resolved ones beyond the two per task `scale-500k` keeps, `SWE-bench/SWE-smith-trajectories` (MIT); `scale-500k` uses every mini-coder trajectory that fits, so none are left for the anneal; rendered in the SFT format; held-out and SWE-bench Verified repositories dropped as in SFT |
| Synthetic code instruction and answer data | 10% | OpenCoder's annealing corpus (`OpenCoder-LLM/opc-annealing-corpus`, ODC-By) |
| Reasoning traces | 10% | OpenThoughts3 and smoltalk2 think splits (as in `hybrid-mid32k-v1`) |
| Math | 5% | FineMath |
| Replay of the stable mix | 5% | |

The trajectory share in the anneal is the new element. Loss on the observations
teaches what commands print in a repository, and these trajectories are never seen
again in SFT. Nemotron's licensed pre-training sets (`nvidia/Nemotron-Pretraining-Code-v1/v2`,
`Nemotron-CC-Math-v1`) need a manual access request and are left out until access and
terms are checked.

## Gates

0. **Is 0.6–0.7B enough?** (~$500, existing code.) Qwen3-0.6B-Base on `scale-500k` with
   `modal_agent_sft.py::train --base Qwen/Qwen3-0.6B-Base`, plus the existing 720M
   mid-training export fine-tuned on the same trajectories in GLM format. Both are
   evaluated on the 179 held-out tasks at 100 turns, sampled.
   - If Qwen3-0.6B resolves ≥ 5 and the 720M about 0: the gap is pre-training, so continue with this plan.
   - If Qwen3-0.6B resolves about 0 too: size is the limit, so switch to the ~1.5B variant.
1. **Anneal mix ablation** (~$450). Three 5B-token anneals from the 30B export: the
   anneal mix above, the same mix without trajectories, and the
   `hybrid-mid32k-v1` mix. The cheap proxy for agentic readiness is held-out loss
   on the assistant tokens of resolved trajectories from the held-out repositories,
   alongside HumanEval/MBPP and HellaSwag/ARC within one point of the 40B export.
2. **Stable phase and anneal** (~$3,900), in 24-hour segments with the existing
   workflow and supervisors. HumanEval/MBPP and the proxy loss are measured at every
   5B export.
   *Gate:* HumanEval and MBPP at least 20 before agent SFT, otherwise extend the
   stable phase rather than starting SFT.
3. **Agent SFT and evaluation** on `scale-500k` (GLM format, 40,960 context). The model
   is evaluated on the held-out PR tasks, Terminal-Bench 2.1 (Harbor, Terminus 2), and
   SWE-bench Verified in the bash-only format once it resolves held-out tasks.

## Implementation needed

- `scale-500k` rendered for the GLM tokenizer and chat format (the Qwen run renders
  Qwen3 ChatML); `prepare_mix` takes the tokenizer and renderer as parameters.
- Recipes for the stable phase and anneal under `hybrid/configs/`, with the CommitPack,
  trajectory and OpenCoder sources added to the hybrid data preparation.
- A 64K throughput and memory check on 8×H100 before the anneal.
- The proxy-loss evaluation: held-out trajectory assistant-token loss, added to the
  hybrid held-out evaluation.
