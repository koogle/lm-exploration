# Original web history

`web-inference.bundle` is a complete Git bundle of the original `web-inference`
main and `claude/design-studies` histories, including their original commit IDs,
messages, authors and trees. It was verified with `git bundle verify`.

The normal consolidated history imports those commits after removing only
`models/gpt2/gpt2.Q4_0.gguf`. GitHub would not accept the original pointer in any
reachable commit while the destination LFS object was unavailable. See
[the commit map](web-inference-commit-map.txt) for original-to-imported identities.
The source repository and its LFS object remain available.

To inspect the exact original history from the consolidated repository root:

```sh
git bundle verify docs/history/web-inference.bundle
git init /tmp/web-inference-history
git -C /tmp/web-inference-history fetch "$PWD/docs/history/web-inference.bundle" \
  refs/remotes/web-inference/main:refs/heads/main \
  refs/remotes/web-inference/claude/design-studies:refs/heads/design-studies
GIT_LFS_SKIP_SMUDGE=1 git -C /tmp/web-inference-history switch main
```

Git bundles contain Git objects, not LFS binary payloads. To retrieve the original
weights from that recovered checkout, install Git LFS, add
`https://github.com/koogle/web-inference.git` as a remote, and fetch its LFS object.
Alternatively use the [pinned upstream model download](../../inference/web/models/gpt2/README.md).

The other three source histories are retained directly as original commits in
the consolidated repository and do not require bundles.
