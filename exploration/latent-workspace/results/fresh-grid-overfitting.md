# Interim overfitting review

These are adaptive validation observations, not a completed grid or sealed-test
result. Exact reports, hashes and UTC retrieval time are archived in
[snapshot](fresh-grid-50k-interim.json); [chart](fresh-grid-50k-interim.png).
Both separate 79M drafters started with identical random weights for frozen
Qwen2.5-0.5B and use common long-validation trajectories (192 responses).

| Training cap | NLL minimum / step | Latest NLL / step | NEW at NLL minimum → latest |
| --- | --- | --- | --- |
| 64 | 3.81283 / 14000 | 5.30673 / 55593 | 2.04180 → 2.04131 |
| 256 | 2.53554 / 13612 | 2.85405 / 37246 | 3.43127 → 3.60350 |

The fixed training diagnostic uses only twelve responses (four per domain), not
the full training corpus. Its NLL falls to 0.225 for the short arm and 0.962 for
the long arm. Together with increasing held-out NLL, this strongly supports
overfitting. It does not establish per-layer instability or full-corpus fit.

Short-arm acceptance plateaus near 2; best observed 2.10427 at 31618. Continued
exposure mostly improves training fit. Recommend early stopping this arm, preserving
its selected acceptance checkpoint and the recorded NLL minimum for comparison.

Long-arm acceptance still improves, despite worsening NLL: +0.17222 NEW (+5.0%)
over 23634 further updates after the loss minimum. Prose/code gain; translation
at the NLL minimum is 1.23913 versus latest 1.16206. NLL and greedy acceptance
measure different aspects: more confident incorrect predictions can worsen NLL
even as more argmax predictions become correct. That mechanism is plausible;
calibration/entropy diagnostics have not been run to confirm it.

Recommend only a bounded further acceptance check for the long arm (three more
evaluations), then stopping if gains do not persist across domains. Favor more
distinct training examples over automatically completing 100000 updates on 49k.
This is a recommendation, not an executed scheduler change: current captured
controller and schedules remain active. Any early stop must be recorded as an
adaptive deviation; larger-data curves should still be compared at matched updates.
The strongest acceptance checkpoint needs actual verified-rollout evaluation,
followed by a frozen configuration and sealed test before qualification. No test
responses or metrics have been accessed.
