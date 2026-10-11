import time

import pytest
import torch
from test_dflash import fixture

from latent_workspace.dflash import features
from latent_workspace.dflash_recache import add_features, sequence_digest


def test_feature_expansion_preserves_labels_old_features_and_causal_new_features():
    target, _draft = fixture()
    tokens = torch.randint(31, (1, 12))
    old = features(target.model(tokens, output_hidden_states=True), (1, 3))[0]
    data = {"train": [{"tokens": tokens[0], "memory": old.clone(), "group": "a"}], "validation": []}
    before = sequence_digest(data)
    identity = add_features(target, data, (1, 3), (1, 2, 3), 2, time.monotonic() + 10)
    assert identity == before == sequence_digest(data)
    expanded = data["train"][0]["memory"]
    torch.testing.assert_close(expanded[:, :16], old[:, :16], atol=0, rtol=0)
    torch.testing.assert_close(expanded[:, -16:], old[:, -16:], atol=0, rtol=0)
    expected = features(target.model(tokens, output_hidden_states=True), (2,))[0]
    torch.testing.assert_close(expanded[:, 16:32], expected)
    changed = tokens.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 31
    other = features(target.model(changed, output_hidden_states=True), (2,))[0]
    torch.testing.assert_close(expanded[:6, 16:32], other[:6])
    with pytest.raises(ValueError, match="Only training"):
        add_features(target, {**data, "test": []}, (1, 3), (1, 2, 3), 2, time.monotonic() + 10)
