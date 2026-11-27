# Malama training log

Every launch behind Malama v0.0.1, from pretraining to the release, in the order it ran,
with its outcome. It includes the launches that failed or did not help. Results and
conclusions are in [MALAMA_V0_0_1.md](MALAMA_V0_0_1.md) and
[CODING_REASONER_1B_FINDINGS.md](CODING_REASONER_1B_FINDINGS.md).

## Run log

Every launch, by workflow run ID (dates UTC). Launches up to reasoning SFT v4 are from each
launch's workflow record and are discussed in
[CODING_REASONER_1B_FINDINGS.md](CODING_REASONER_1B_FINDINGS.md); their records are under
`training/results/coding-reasoner-1b/runs/<run ID>/` in `jfrick/lm-exploration-results`.
Records of later launches are under `pr-197/runs/` of the same dataset. LiveCodeBench scores are v6 pass@1 in the thinking protocol.

| Run ID | Workflow | Dates | Outcome |
|---|---|---|---|
| `kda-960m-foundation-20261110-01` | `kda-960m-foundation` | Nov 10–14 | Completed in seven launches: Malama Base, 100B tokens at 4K context ([card](https://huggingface.co/jfrick/malama-base)) |
| `kda-960m-fim-midtrain-20261114-01` | `kda-960m-fim-midtrain` | Nov 14–15 | Completed: about 15B tokens with fill-in-the-middle from Malama Base at 84.8B tokens |
| `kda-960m-lcb-format-sft-20261115-01` | `kda-960m-lcb-format-sft` | Nov 15 | Failed before training: the recipe was not prepared on the data volume |
| `kda-960m-lcb-format-sft-20261115-02` | `kda-960m-lcb-format-sft` | Nov 15 | Failed at startup: the trainer could not read the recipe's data files |
| `kda-960m-lcb-format-sft-20261115-03` | `kda-960m-lcb-format-sft` | Nov 15 | Completed (answer-only format SFT, two variants); 4.7 pass@1 on an 8-problem LiveCodeBench check |
| `kda-960m-reasoning-pilot-20261115-01` | `kda-960m-reasoning-pilot` | Nov 15 | Failed in the 32K context extension stage |
| `kda-960m-reasoning-pilot-20261115-02` | `kda-960m-reasoning-pilot` | Nov 15 | Completed: 32K context extension, then the reasoning pilot (3.8 pass@1) |
| `kda-960m-code-anneal-20261117-01` | `kda-960m-code-anneal` | Nov 17 | Failed in data preparation: one source had fewer training tokens than the recipe asked for |
| `kda-960m-code-anneal-20261117-02` | `kda-960m-code-anneal` | Nov 17–18 | Completed: 5B code tokens at 32K |
| `kda-960m-code-anneal-20b-20261117-01` | `kda-960m-code-anneal-20b` | Nov 18 | Completed: 20B code tokens at 4K |
| `kda-960m-reasoning-sft-v3-20261118-01` | `kda-960m-reasoning-sft-v3` | Nov 18 | Completed: reasoning SFT v3 (8.4 pass@1) |
| `kda-960m-merge-sft-v3-code20b-l05`, `-l10` | `modal_tools::merge` | Nov 19 | Weight merges; both broke the model |
| `kda-960m-contest-rl-20261119-01` | `kda-960m-contest-rl` | Nov 19–20 | Illegal memory access while sampling at step 64; its resumed launch hung (FlexAttention recompilation) |
| `kda-960m-contest-rl-20261120-01` | `kda-960m-contest-rl` | Nov 20 | From step 50; illegal memory access at step 62 of the run |
| `kda-960m-contest-rl-20261120-02` | `kda-960m-contest-rl` | Nov 20–21 | Completed: from step 100 to 400 (14.1 pass@1) |
| `kda-960m-contest-rl-hard-20261121-01` | `kda-960m-contest-rl-hard` | Nov 21 | Out of GPU memory at 16K tokens |
| `kda-960m-contest-rl-hard-20261121-02` | `kda-960m-contest-rl-hard` | Nov 21–22 | Stopped at step 146 of 200 (11.8 pass@1 at step 100, 12.4 at 146) |
| `hybrid-720m-loop-test-03g` | `hybrid-720m-loop-test` | Nov 21–22 | Completed: closed-gate looping conversion ([looped LM plan](LOOPED_LM.md)) |
| `kda-960m-reasoning-sft-v4-20261121-01` | `kda-960m-reasoning-sft-v4` | Nov 21–22 | Function timeout at step 900 |
| `kda-960m-reasoning-sft-v4-20261122-01` | `kda-960m-reasoning-sft-v4` | Nov 22 | Stopped after 5 minutes and relaunched with the fixed session handling |
| `kda-960m-reasoning-sft-v4-20261122-02` | `kda-960m-reasoning-sft-v4` | Nov 22 | Completed: 4,100 steps (14.0 pass@1, 26.6 pass@6) |
| `kda-960m-distill-smoke-20261122-01` | `kda-960m-distill-smoke` | Nov 22 | Completed: 3-step smoke run of on-policy distillation on 2 H100s |
| `kda-960m-distill-pilot-a/b/c-20261122-01` | `kda-960m-distill-pilot-a/b/c` | Nov 22 | First launches of the learning-rate pilots; relaunched as `-02` after scoring counted the teacher's end of turn as the student's EOS |
| `kda-960m-distill-pilot-a/b/c-20261122-02` | `kda-960m-distill-pilot-a/b/c` | Nov 22 | Completed: three learning rates. Only the lowest stayed stable (13.2 pass@1 against 14.1 for its parent) |
| `kda-960m-distill-pilot-d-20261122-01`, `-e-20261122-01` | `kda-960m-distill-pilot-d`, `-e` | Nov 22 | Completed: sharpened teacher, and the lowest rate for three passes. Looping returned as teacher KL fell |
| `kda-960m-contest-rl-v4-20261122-01` | `kda-960m-contest-rl-v4` | Nov 22 | Completed: 100 RL steps from SFT v4 (16.3 pass@1) |
| `kda-960m-contest-rl-v4-20261123-01` | `kda-960m-contest-rl-v4` | Nov 23 | Completed: from step 100 to 400 (18.1 pass@1, 25.0 pass@6) |
| Teacher comparison | `modal_livecodebench --pool train --openrouter` | Nov 23 | Five hosted teachers on 60 problems the Qwen3-30B-A3B teacher never solved: DeepSeek V4 Flash solved 28 for $0.43 at list price and was chosen (DeepSeek V4 Pro 32, DeepSeek V3.2 25, GLM-5.3 and GLM-5.3 Flash 21 each) |
| Teacher traces v2 | `modal_livecodebench --pool codecontests --openrouter` | Nov 23 | DeepSeek V4 Flash on DeepMind CodeContests problems, and the Qwen3-30B-A3B run rebuilt with up to four passing traces per problem: 31,568 traces against 2,234 in v1, in `jfrick/lcb-teacher-traces` |
| `kda-960m-reasoning-sft-v5-20261123-01` | `kda-960m-reasoning-sft-v5` | Nov 23–24 | Completed: 4,660 steps (12.8 pass@1, 23.3 pass@6) |
| `kda-960m-contest-rl-v5-20261124-01` | `kda-960m-contest-rl-v5` | Nov 24–25 | Stopped at about step 650: 15.6 pass@1 at step 100, 18.9 at 250, 17.0 at 500 |
| `kda-960m-contest-functions-rl-20261124-01` | `kda-960m-contest-functions-rl` | Nov 24–25 | Stopped at step 45: the first mixed pool gave MBPP problems without their assertions and held-out MBPP stayed at 0% |
| `kda-960m-contest-functions-rl-20261125-01` | `kda-960m-contest-functions-rl` | Nov 25 | Completed: 400 mixed RL steps from contest RL step 250 (19.1 pass@1, HumanEval 50.1, MBPP 41.6). Step 400 is Malama v0.0.1 |
| Perplexity suite | `modal_benchmarks --suite perplexity` | Nov 25 | Malama Base and eleven open base models on 1,000 arXiv abstracts of 2026-11-24 and WikiText-2: Malama Base 0.949 and 0.833 bits per byte |
| Release | | Nov 27 | Step 400 of the mixed RL run published as [`jfrick/malama-v0.0.1`](https://huggingface.co/jfrick/malama-v0.0.1); evaluation outputs, RL pools and run logs in `jfrick/lm-exploration-results` under `pr-197/` |
