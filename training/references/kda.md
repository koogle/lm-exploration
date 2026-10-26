# Kimi Delta Attention, step by step

A worked example of the KDA recurrence in [`hybrid/kda.py`](../hybrid/kda.py), the
linear-attention layer of the hybrid model. One head is shrunk to width 4 with
one-hot keys, so every row of the state is one memory slot. Three tokens write
into it: token 1 and token 3 share a key with different values. α and β are set
directly through the raw gate inputs (`A_log = 0`, `dt_bias = 0`).
[`kda/figures.py`](kda/figures.py) computes the states with the loop body of
`kda_reference`, checks that `kda_reference` itself returns the same reads, and
takes the line numbers and code shown in the figure from `kda.py`:

```sh
cd training
uv run --with matplotlib python references/kda/figures.py references/kda
```

![KDA recurrence mapped to kda.py](kda/kda_recurrence.png)

## The recurrence

Each KDA head keeps one d × d matrix `state` (S): rows are key channels, columns
are value channels. `kda_reference` updates it once per token:

```python
for t in range(length):
    keep = (positions[:, t] > 0).to(dtype)[:, None, None, None]
    state = state * keep * decay[:, t, ..., None]                                  # forget
    k_t = k[:, t]
    error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k_t, state)                   # predict
    state = state + torch.einsum("bhk,bhv->bhkv", beta[:, t, :, None] * k_t, error)  # correct
    outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))                 # read
```

- **Forget:** every row of S is scaled by its own decay α in (0, 1), computed per
  token from `-exp(A_log) * softplus(f_proj(x) + dt_bias)`. `keep` is 0 at a
  document start and zeroes the state there.
- **Predict:** `kᵀS` is what the memory currently returns for this key; `error`
  is the gap to the token's value.
- **Correct:** the outer product `β k errorᵀ` moves the slot toward the value.
  β = sigmoid(`b_proj(x)`) sets how far: at β = 1 the slot holds exactly the new
  value.
- **Read:** the output is `qᵀS` after the update.

Together: `S_t = diag(α_t) S_(t−1) + β_t k_t (v_t − k_tᵀ diag(α_t) S_(t−1))ᵀ`,
`o_t = q_tᵀ S_t`. Without the predict step, the update adds `β k vᵀ` (plain
linear attention): in the example, slot 1 would hold a blend of the old and new
values (`[−.27, 0, .36, 1]`) instead of the new value (`[−1, 0, 0, 1]`).

The state has the same size for every sequence length. On CUDA, fla's
`chunk_kda` computes the same recurrence in parallel chunks, with gate, β and
L2 normalization inside the kernel and document starts from `cu_seqlens`.

## How the code is checked

`tests/test_hybrid_kda.py` (CPU) checks the decay and the reset:

- `test_reference_matches_matrix_form` compares `kda_reference` in float64 with
  `S_t = (I − β k kᵀ) diag(α) S_(t−1) + β k vᵀ` written with explicit `diag`,
  `outer` and the gate formula. It pins the gate formula, that α scales key
  channels, and that the decay comes before the delta update.
- `test_reference_resets_state_at_documents` requires a document to give
  bit-identical outputs packed after another document and alone.
  `test_hybrid_model_isolates_documents` repeats this through a mixed KDA/full stack.

Each of these edits to `kda.py` fails at least one of these tests: dropping `keep`,
decaying value channels, removing or moving the decay, dropping `dt_bias`, flipping
the gate sign, and resetting at `positions > 1`. The CUDA check
(`tests/check_hybrid_cuda.py`, `kda_against_reference`) compares fla's kernel with
`kda_reference` on packed documents, including a one-token document, for the output
and all input gradients.
