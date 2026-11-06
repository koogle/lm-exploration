# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Launcher adapted from fla-core 0.6.0; MIT license in kernels/LICENSE.fla.
"""H100 KDA backward specialization with shorter live ranges and eight warps."""

import torch
import triton
from fla.backends import BaseBackend, register
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.graph import get_static_buffer

from .kernels.kda_wy import wy_backward


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_graph: bool = False,
):
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # dq, dk are allocated at HV dimension; caller reduces to H if GVA
    if use_graph:
        dq = get_static_buffer("wy_dq", (B, T, HV, K), torch.float, q.device)
        dk = get_static_buffer("wy_dk", (B, T, HV, K), torch.float, q.device)
        dv2 = get_static_buffer("wy_dv2", tuple(v.shape), v.dtype, v.device)
        dg = get_static_buffer("wy_dg", tuple(g.shape), torch.float, g.device)
        db = get_static_buffer("wy_db", tuple(beta.shape), torch.float, beta.device)
        dA = get_static_buffer("wy_dA", tuple(A.shape), torch.float, A.device)
    else:
        dq = g.new_empty(B, T, HV, K, dtype=torch.float)
        dk = g.new_empty(B, T, HV, K, dtype=torch.float)
        dv2 = torch.empty_like(v)
        dg = torch.empty_like(g, dtype=torch.float)
        db = torch.empty_like(beta, dtype=torch.float)
        dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * HV)
    wy_backward[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
        USE_GRAPH=use_graph,
        IS_VARLEN=cu_seqlens is not None,
        BK=32,
        BV=128,
        num_warps=8,
        num_stages=1,
    )
    dv = dv2
    return dq, dk, dv, db, dg, dA


class H100WY(BaseBackend):
    backend_type = "project_h100_wy"
    priority = 0
    chunk_kda_bwd_wy_dqkg_fused = staticmethod(backward)

    def verify(self, function_name, *args, **kwargs):
        if function_name != "chunk_kda_bwd_wy_dqkg_fused":
            return False, None
        q = kwargs["q"] if "q" in kwargs else args[0]
        v = kwargs["v"] if "v" in kwargs else args[2]
        supported = (
            q.is_cuda
            and q.dtype in (torch.bfloat16, torch.float32)
            and torch.cuda.get_device_capability(q.device) == (9, 0)
            and q.shape[-1] == v.shape[-1] == 128
            and kwargs.get("chunk_size", 64) == 64
        )
        return supported, None


def install():
    # Load the owning registry before adding our operation-specific specialization.
    import fla.ops.kda.chunk_bwd  # noqa: F401

    register("kda")(H100WY)
