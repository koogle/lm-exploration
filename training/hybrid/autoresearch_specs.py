"""Pure-Python experiment registry and budgets, safe to import in Modal launchers."""

VARIANTS = (
    "baseline",
    "fused",
    "head_timescales",
    "slow_bank",
    "banks2",
    "key_wide",
    "heads_narrow",
    "kv8",
    "kv16",
    "dynamic_kv",
    "input_gate",
    "memory_gate",
    "layout_dense",
    "layout_sparse",
    "layout_front",
)
PROXY = {
    "vocab_size": 154880,
    "eos_id": 154820,
    "block_size": 4096,
    "n_layer": 12,
    "n_embd": 1024,
    "n_head": 8,
    "n_kv_head": 2,
    "n_kda_head": 8,
    "head_dim": 128,
    "mlp_hidden": 2816,
    "loss_chunk_size": 512,
}
TARGET = dict(PROXY, n_layer=24, n_embd=2048, n_head=16, n_kda_head=16, mlp_hidden=5504)


TOKENS_PER_STEP = 8192
SCREEN_STEPS = 500_000_000 // TOKENS_PER_STEP
FINAL_STEPS = 2_000_000_000 // TOKENS_PER_STEP
