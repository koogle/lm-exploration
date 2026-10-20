# Independent source campaign settlement

At 2026-10-20 05:58 UTC, private source snapshots contained 42,753 saved
documents: ECB 19,303 and EMA 23,450, with a cached exact-token lower bound
of 1,071,021,720. This is 2,495 more documents than the previous check-in.
EMA's latest checkpoint was 05:56:45 UTC. ECB's last three-round campaign
finished at 04:10:46 UTC with two configured candidates remaining.

Modal returned the shared tick ancestor, the completed ECB campaign and its
three successful workers, plus the still-running EMA sibling campaign in
`get_call_graph()`. The old inspector flattened the whole returned graph and
therefore retained ECB's reservation until EMA finished. A direct result read
confirmed that ECB's campaign itself was complete.

The inspector now locates exactly the requested function-call ID and checks
that subtree. Missing or ambiguous matches remain unknown; active descendants
still retain the reservation even when their supervisor failed. Sibling work
does not block settlement. Regression tests include active siblings and an
absent target. All 391 tests, Ruff lint/format and whitespace checks passed.

The existing goal app was deployed, retaining its minute schedule. Invoking its
existing tick (`fc-01M5BY9PHEHXRKHGBFH7T7TQWW`) settled the old ECB campaign
and dispatched `goal-ecb-1792475996-dff5d1fe2c`, call
`fc-01M5BYAB0MGYNY91CEJDN0WAM5`. At readback control revision
`7f55c7d9c60a194a339e3826db7867593ec5459a`, the old campaign was complete,
the new call had a live subtree, and EMA remained running. Source workers and
all budgets, cooldowns and histories are unchanged. Two configured candidates
are not a claim of website exhaustion; deferred types and other coverage gaps
remain. No independent source writer was launched.
