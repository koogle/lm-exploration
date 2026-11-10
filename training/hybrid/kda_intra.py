# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Launcher adapted from fla-core 0.6.0; MIT license in kernels/LICENSE.fla.
"""H100 range-guarded KDA intra-backward backend."""

import torch
import triton
from fla.backends import BaseBackend, register
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.graph import get_static_buffer
from fla.utils import IS_GATHER_SUPPORTED

from .kernels.kda_intra import intra_backward


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_graph: bool = False,
):
    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    if use_graph:
        dq2 = get_static_buffer("intra_dq2", tuple(dq.shape), dq.dtype, dq.device)
        dk2 = get_static_buffer("intra_dk2", tuple(dk.shape), dk.dtype, dk.device)
        db2 = get_static_buffer("intra_db2", (NK, *beta.shape), torch.float, beta.device)
        dg2 = get_static_buffer("intra_dg2", tuple(dg.shape), torch.float, dg.device)
    else:
        dq2 = torch.empty_like(dq)
        dk2 = torch.empty_like(dk)
        db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
        dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK * NC * NT, B * HV)
    intra_backward[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        SAFE_GATE=safe_gate,
        GATE_SHIFT=(-lower_bound if lower_bound is not None else 2.0) * RCP_LN2,
        USE_GATHER=IS_GATHER_SUPPORTED,
        USE_GRAPH=use_graph,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=2,
        num_stages=1,
    )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = dg2

    return dq, dk, db, dg


class H100Intra(BaseBackend):
    backend_type = "project_h100_intra"
    priority = 0
    chunk_kda_bwd_intra = staticmethod(backward)

    def verify(self, function_name, *args, **kwargs):
        if function_name != "chunk_kda_bwd_intra":
            return False, None
        q = kwargs["q"] if "q" in kwargs else args[0]
        g = kwargs["g"] if "g" in kwargs else args[2]
        return (
            q.is_cuda
            and torch.cuda.get_device_capability(q.device) == (9, 0)
            and q.dtype in (torch.bfloat16, torch.float32)
            and q.shape[-1] == 128
            and q.shape[2] == g.shape[2]
            and kwargs.get("chunk_size", 64) == 64
            and not kwargs.get("safe_gate", False)
            and not kwargs.get("use_graph", False)
        ), None


def install():
    import fla.ops.kda.chunk_intra  # noqa: F401

    register("kda")(H100Intra)
