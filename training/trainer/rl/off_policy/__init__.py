"""Off-policy algorithms and the trajectory sources they train on.

Their answers do not all come from the weights being updated: answers the model sampled
a few updates ago (`"reuse"`), or answers another model wrote (`"file"`, for example the
Qwen teacher's). An algorithm that corrects with token-level importance ratios needs the
sampling weights' log-probabilities (`needs_behaviour`), which exist only for the model's
own answers; a teacher with another tokenizer has none, so its answers need an algorithm
without ratios.
"""
