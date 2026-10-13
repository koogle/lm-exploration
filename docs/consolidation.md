# Repository consolidation

Consolidation date: 2026-10-11. The destination is the existing private
`koogle/0-to-llm` repository, since renamed to `koogle/lm-exploration`.

## Research follow-up, 2026-10-13

The completed latent-workspace training, qualification, reports and one-page
write-up are already on consolidated `main` through `9d8b9bb`. The
`drafter-data-coverage` branch is fully contained in that history (no outstanding
commits), and the standalone source has no code or result changes since the initial
import; only repository-status README notices differ. No older source snapshot was
copied over the newer consolidated work.

The root README now links directly to the research, learnings PDF and qualification
results. Continue from `exploration/latent-workspace/` in this repository. The study
is complete; the separately documented J-space follow-up is not a completed result.
Verification at reconciliation: all 126 local tests, Ruff lint/format checks and
README link checks pass. No GPU training or evaluation was launched.

Earlier notes below describe the original migration and temporary period of
iteration. This reconciliation does not archive or delete the standalone repository.

## Layout and provenance

| Source | Imported tip | Destination |
| --- | --- | --- |
| `koogle/0-to-llm` main | `acb4a473469891497df3c70d831301f9801190ca` | Existing destination history |
| `koogle/web-inference` main | `ee69a4d9310773dd6b113f13a9fbf1644e38bebf` | Web reference/model artifacts and phone research |
| `koogle/web-inference` `claude/design-studies`, PR #1 | `3906ff5ae5cceb1ce7064ede8550f6820fa681fe` | `training/designs/` and `inference/phone/designs/` |
| `koogle/phone-inference` main | `1d44469e546fc63386a28d654d41a530551a3137` | `inference/phone/` scaffold |
| `koogle/latent-workspace` main | `7dd10a4aab850194b5ebebfee5424c0f512d8007` | `exploration/latent-workspace/` |

The original `0-to-llm`, research and phone commits are retained as ancestors of
the consolidated main branch. The web history is imported with only the historical
model pointer removed, because GitHub rejects a push whose history references an
unavailable LFS object. Its commit identities change; the mapping and **complete
original web Git history** are preserved in [history](history/README.md). The design PR's
head includes the imported web main tip. GitHub PR discussion remains at
[the original PR](https://github.com/koogle/web-inference/pull/1).

The browser prototype was restored from
`0e833f768bf72b0d6b18016395144e4c939eb833`, immediately before its removal in the
phone pivot. It is a Rust/WASM/WebGPU **fractal demo**, with LM inference only
planned. Its source and implementation notes are under `inference/web/`.

The GPT-2 model remains in the source repository's LFS storage. Destination upload
was blocked at `lfs.github.com` verification, and a separate destination download
check confirmed the object is unavailable there. The current tree therefore uses
the pinned upstream download, license and checksum rather than an unresolved LFS
pointer. Weight files are locally ignored. Original commits in the bundled archive
retain their pointer; use the source LFS remote when inspecting those historical
versions. No source model object or history was deleted. See [model instructions](../inference/web/models/gpt2/README.md).

The three HTML designs and phone model-selection report were relocated unchanged.
The training Run Board is a mockup, not a deployed Modal application. The new
[training README](../training/README.md) records the current user direction; further
notes from other conversations have not yet been supplied.

All 197 research-project files are retained: 192 unchanged, plus five documentation
files adjusted for the new location and scope. Executable code, lockfile, configs,
tests, reports, plots and raw results are unchanged. Existing research Modal
apps/volumes are not renamed. Untracked local checkpoints and the budget ledger
are outside Git; see [operations](../exploration/latent-workspace/docs/operations.md)
before starting any further GPU work.

Older projects such as `mlx-playground` and `llama_training` are outside this
consolidation of the four recent repositories.

## Verification

- Original destination, research and phone tips, and the imported web code tip,
  are ancestors of the consolidated branch. Original web tips are in the verified bundle.
- File-by-file comparison verified the unchanged research files and all four
  relocated design/research files.
- The 106,554,880-byte GPT-2 Git LFS object was retrieved and its size and SHA-256
  verified against the retained model manifest.
- The moved research project passed all 69 tests, Ruff lint and Ruff format checks
  using its frozen uv lockfile.
- JavaScript syntax passed for all three HTML mockups and the browser demo.
- A fresh clone from the published destination succeeded; its commit matched the
  local consolidation, the original-history bundle verified, and all 123 relative
  Markdown file links resolved.
- No GPU training or Modal deployment was performed. Rust/WASM compilation and
  interactive browser behavior were not revalidated in this environment; Rust
  tooling is unavailable. The restored browser source is unchanged.

## GitHub administration still required

The consolidated contents are published on destination `main`. Web and phone
source READMEs point to the destination, and web-inference PR #1 is closed with
links to its imported designs. Per the user's subsequent instruction,
`koogle/latent-workspace` remains active for ongoing iteration and must not be
archived. Its imported copy is a snapshot, not an automatically synchronized mirror.
Source code and the original model remain available.

The current environment supports Git fetch/push, but its network policy blocks
the GitHub administration API. The connected GitHub tools do not expose repository
rename/archive actions. Therefore publishing the consolidated contents does not
complete the desired rename or archive the source repositories.

After verifying the destination, finish these steps in GitHub Settings or from
an authenticated GitHub CLI with administration access:

```sh
gh repo rename lm-exploration --repo koogle/0-to-llm --yes
gh repo archive koogle/web-inference --yes
gh repo archive koogle/phone-inference --yes
```

Keep the destination private. Renaming preserves the destination repository's
identity and GitHub redirects the old URL. Archive only `web-inference` and `phone-inference`, preserving their history and
original discussions. Keep `latent-workspace` and the consolidated destination
active. No repository needs to be deleted.
