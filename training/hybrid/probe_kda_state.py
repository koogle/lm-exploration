"""Small exact-recurrence probes of trained KDA activations; no model updates."""

import json
import time
from pathlib import Path

from .branch_study import prepared_volume
from .diagnose_kda import SOURCE_RUN
from .modal_study import app, image, runs


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=1800,
    retries=0,
    volumes={"/runs": runs, "/data": prepared_volume},
)
def probe_states(run, deadline):
    import model_hybrid.study_mixers as mixers
    import numpy as np
    import torch
    from model_hybrid.kda import kda_gate
    from model_hybrid.model import HybridConfig, HybridLM
    from torch.nn import functional as F

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    runs.reload()
    root = Path("/runs") / run
    examples = np.load(root / "baseline-17/examples.npz")
    x = torch.tensor(examples["x"][128:129, :256], device="cuda", dtype=torch.long)
    results = []
    original = mixers.kda_kernel
    for variant in ["baseline", "none", "rope_triton"]:
        for seed in [17, 29]:
            assert time.time() < deadline - 120
            cp = torch.load(
                Path("/runs") / SOURCE_RUN / f"{variant}-{seed}/checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            model = mixers.configure(HybridLM(HybridConfig(**cp["config"])), variant).cuda().eval()
            model.load_state_dict(cp["model"])
            del cp
            captured = []

            def capture(q, k, v, g, beta, A_log, dt_bias, documents, captured=captured):
                captured.append(
                    tuple(t.detach() for t in [q, k, v, g, beta, A_log, dt_bias, documents])
                )
                return original(q, k, v, g, beta, A_log, dt_bias, documents)

            mixers.kda_kernel = capture
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.hidden_states(x)
            finally:
                mixers.kda_kernel = original
            layers = []
            with torch.no_grad():
                for q, k, v, g, beta, A_log, dt_bias, documents in captured:
                    q = F.normalize(q.float(), dim=-1)
                    k = F.normalize(k.float(), dim=-1)
                    v = v.float()
                    alpha = kda_gate(g, A_log, dt_bias).exp()
                    beta = beta.float().sigmoid()
                    record = {
                        "beta_mean": beta.mean().item(),
                        "decay_efold_quantiles": torch.quantile(
                            -1 / alpha.log().clamp(max=-1e-8),
                            torch.tensor([0.1, 0.5, 0.9], device="cuda"),
                        ).tolist(),
                        "query_key_same_token_cosine": (q * k).sum(-1).mean().item(),
                        "lag_key_cosine": {},
                        "lag_query_key_cosine": {},
                    }
                    for lag in [1, 2, 3, 4, 16, 64]:
                        mask = (documents[:, lag:] == documents[:, :-lag])[:, :, None]
                        record["lag_key_cosine"][str(lag)] = (
                            (k[:, lag:] * k[:, :-lag])
                            .sum(-1)[mask.expand(-1, -1, k.shape[2])]
                            .mean()
                            .item()
                        )
                        record["lag_query_key_cosine"][str(lag)] = (
                            (q[:, lag:] * k[:, :-lag])
                            .sum(-1)[mask.expand(-1, -1, k.shape[2])]
                            .mean()
                            .item()
                        )
                    state = torch.zeros((1, k.shape[2], k.shape[3], v.shape[3]), device="cuda")
                    error_sq = value_sq = write_sq = state_sq = 0.0
                    for t in range(k.shape[1]):
                        if t and documents[0, t] != documents[0, t - 1]:
                            state.zero_()
                        state = state * alpha[:, t, ..., None]
                        pred = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
                        error = v[:, t] - pred
                        write = torch.einsum("bhk,bhv->bhkv", beta[:, t, :, None] * k[:, t], error)
                        state = state + write
                        error_sq += error.square().sum()
                        value_sq += v[:, t].square().sum()
                        write_sq += write.square().sum()
                        state_sq += state.square().sum()
                    record["prediction_error_to_value_energy"] = (error_sq / value_sq).item()
                    record["write_to_state_energy"] = (write_sq / state_sq).item()
                    layers.append(record)
            results.append({"variant": variant, "seed": seed, "layers": layers})
            (root / "state-probes.json").write_text(json.dumps(results))
            runs.commit()
            print("STATE PROBE", variant, seed, flush=True)
            del model, captured
            torch.cuda.empty_cache()
    return results
