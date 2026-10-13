# Completed large-data curriculum training

Both separate 79M drafters for frozen Qwen2.5-0.5B completed 100000 updates
(268.8M supervised NEW-token exposures per arm). Full reports and shared
initialization/source/validation checks: [archive](large-curriculum-final.json).
[Learning curves](large-curriculum-final.png). Test remains sealed.

| Response cap | Best validation step | Best NEW / 7 | Prose | Code | Translation | Best-checkpoint NLL | Final NEW / 7 | Final NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 51475 | 2.19454 | 2.55469 | 2.39453 | 1.63439 | 3.89692 | 2.10009 | 4.30723 |
| 256 | 70585 | 4.03251 | 5.33594 | 5.10547 | 1.65613 | 2.10504 | 4.00693 | 2.17090 |

Select by consecutive correct NEW tokens, not minimum NLL; these criteria select
different steps. Long responses outperform short responses substantially on this
common long-validation view. The long selected result is +0.42901 (~11.9%) above
the previous 49k/256 selected score 3.60350. Dataset size, update exposure and the
new curriculum differ; this is not an isolated causal data-size effect or a
statistical-significance claim. One seed and repeatedly reused validation remain
limitations. Long repetitive teacher trajectories do not establish human quality.

The short arm shows clear late overfitting: training diagnostic loss decreases
while validation loss rises and acceptance plateaus/declines. The long arm also
shows late loss overfitting, but acceptance mostly saturates near 4 and declines
only slightly after its peak. The diagnostic covers twelve training responses,
not the full corpus. Keep the selected long checkpoint, not the final weights,
for subsequent qualification. Further repetitions of this same corpus are not
currently justified by these curves.

The curriculum begins on 131072 prose/code examples, then adds translation with
all 196608 examples, retaining all earlier groups and the same optimizer/RNG and
100k global schedule. Transition steps are 17985 (short) and 11472 (long), so
domain-exposure timing differs between arms. Training weights initially random,
no external DFlash or historical trained draft weights loaded. Target frozen.

Final workers are `fresh-grid-196608-r64-08` and `fresh-grid-196608-r256-11`.
Their inherited `best.pt` files contain the selected checkpoints. Package SHA:
`47d71073e55a03e87c4bdb59df6aa684cedc532815d30c9f44167ff97da919ae`.
Each selected head has twelve 32-token-cap rollout checks (four/domain), all
matching the frozen target's greedy output in the documented FP32 verifier.
Actual rollout draft acceptance is lower than cached-trajectory validation;
these checks establish limited output parity, not broad speed or quality.

Still required: freeze/checksum selected checkpoint and exact evaluation config,
run longer verified rollouts and repetition/context-position slices, then evaluate
the sealed drafting test once if the rollout requirements pass. This is training
completion, not completion of the research/qualification goal.

Coordinator09 returned terminal status with no active workers. Authoritative
reservation ledger was synced to local `artifacts/budget.json`; ownership handle
archived as `artifacts/budget_remote_controller-09-complete.json`. The current
user-reset accounting epoch reserves $162.125/$500, not actual billed spending.
Historical pre-reset and failed reservations remain recorded. No new GPU jobs
were submitted during this completion review.
