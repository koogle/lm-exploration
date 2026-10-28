"""Triton segmented complex scans for the rotating-memory input mixer.

One program handles a batch row and a channel pair, scanning all timesteps and
four modes. The backward scans adjoints in reverse and accumulates analytic
parameter derivatives. Document boundaries break both scans.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def compose(ar, ai, br, bi, cr, ci, dr, di):
    return cr * ar - ci * ai, cr * ai + ci * ar, cr * br - ci * bi + dr, cr * bi + ci * br + di


@tr.jit
def forward_kernel(
    U,
    POS,
    LOGTAU,
    THETA,
    MIX,
    M,
    Y,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    b = tl.program_id(0)
    p = tl.program_id(1)
    t = tl.arange(0, BLOCK)[:, None]
    s = tl.arange(0, 4)[None, :]
    rho = tl.exp(-tl.exp(-tl.load(LOGTAU + s)))
    theta = tl.load(THETA + s)
    ar = rho * tl.cos(theta)
    ai = rho * tl.sin(theta)
    keep = (tl.load(POS + b * T + t, t < T, other=0) > 0) & (t < T)
    ur = tl.load(U + (b * T + t) * D + 2 * p, t < T, other=0).to(tl.float32)
    ui = tl.load(U + (b * T + t) * D + 2 * p + 1, t < T, other=0).to(tl.float32)
    _, _, mr, mi = tl.associative_scan(
        (tl.where(keep, ar, 0.0), tl.where(keep, ai, 0.0), (1 - rho) * ur, (1 - rho) * ui),
        0,
        compose,
    )
    offset = ((b * T + t) * 4 + s) * D + 2 * p
    tl.store(M + offset, mr, t < T)
    tl.store(M + offset + 1, mi, t < T)
    wr = tl.load(MIX + s * D + 2 * p)
    wi = tl.load(MIX + s * D + 2 * p + 1)
    yr = tl.sum(mr * wr, 1)
    yi = tl.sum(mi * wi, 1)
    if RESIDUAL:
        yr += tl.sum(ur, 1)
        yi += tl.sum(ui, 1)
    tt = tl.arange(0, BLOCK)
    tl.store(Y + (b * T + tt) * D + 2 * p, yr, tt < T)
    tl.store(Y + (b * T + tt) * D + 2 * p + 1, yi, tt < T)


@tr.jit
def backward_kernel(
    U,
    POS,
    LOGTAU,
    THETA,
    MIX,
    M,
    DY,
    DU,
    GP,
    GW,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    b = tl.program_id(0)
    p = tl.program_id(1)
    t = tl.arange(0, BLOCK)[:, None]
    s = tl.arange(0, 4)[None, :]
    invtau = tl.exp(-tl.load(LOGTAU + s))
    rho = tl.exp(-invtau)
    theta = tl.load(THETA + s)
    c = tl.cos(theta)
    sn = tl.sin(theta)
    dr = tl.load(DY + (b * T + t) * D + 2 * p, t < T, other=0).to(tl.float32)
    di = tl.load(DY + (b * T + t) * D + 2 * p + 1, t < T, other=0).to(tl.float32)
    wr = tl.load(MIX + s * D + 2 * p)
    wi = tl.load(MIX + s * D + 2 * p + 1)
    nextkeep = tl.load(POS + b * T + t + 1, t + 1 < T, other=0) > 0
    _, _, rr, ri = tl.associative_scan(
        (tl.where(nextkeep, rho * c, 0.0), tl.where(nextkeep, -rho * sn, 0.0), dr * wr, di * wi),
        0,
        compose,
        reverse=True,
    )
    tt = tl.arange(0, BLOCK)
    # Autograd rounds the memory branch to the input dtype before adding the
    # residual branch. Preserve that BF16 accumulation boundary.
    tl.store(
        DU + (b * T + tt) * D + 2 * p,
        tl.sum((1 - rho) * rr, 1).to(U.dtype.element_ty).to(tl.float32) + tl.sum(dr, 1) * RESIDUAL,
        tt < T,
    )
    tl.store(
        DU + (b * T + tt) * D + 2 * p + 1,
        tl.sum((1 - rho) * ri, 1).to(U.dtype.element_ty).to(tl.float32) + tl.sum(di, 1) * RESIDUAL,
        tt < T,
    )
    offset = ((b * T + t) * 4 + s) * D + 2 * p
    mr = tl.load(M + offset, t < T, other=0)
    mi = tl.load(M + offset + 1, t < T, other=0)
    keep = (tl.load(POS + b * T + t, t < T, other=0) > 0) & (t > 0) & (t < T)
    pr = tl.load(M + offset - 4 * D, keep, other=0)
    pi = tl.load(M + offset - 4 * D + 1, keep, other=0)
    ur = tl.load(U + (b * T + t) * D + 2 * p, t < T, other=0).to(tl.float32)
    ui = tl.load(U + (b * T + t) * D + 2 * p + 1, t < T, other=0).to(tl.float32)
    gr = tl.sum(
        tl.where(t < T, rr * (c * pr - sn * pi - ur) + ri * (sn * pr + c * pi - ui), 0.0), 0
    ) * tl.sum(rho * invtau, 0)
    gt = tl.sum(tl.where(t < T, rho * (rr * (-sn * pr - c * pi) + ri * (c * pr - sn * pi)), 0.0), 0)
    ss = tl.arange(0, 4)
    base = ((b * (D // 2) + p) * 4 + ss) * 2
    tl.store(GP + base, gr)
    tl.store(GP + base + 1, gt)
    tl.store(GW + base, tl.sum(dr * mr, 0))
    tl.store(GW + base + 1, tl.sum(di * mi, 0))


class _Rotating(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, positions, log_tau, theta, mix, residual):
        u = u.contiguous()
        positions = positions.contiguous()
        b, t, d = u.shape
        if d % 2 or t > 1024:
            raise ValueError("Rotating kernel supports even widths and lengths <=1024")
        memory = torch.empty((b, t, 4, d), device=u.device, dtype=torch.float32)
        y = torch.empty_like(u)
        forward_kernel[(b, d // 2)](
            u,
            positions,
            log_tau,
            theta,
            mix,
            memory,
            y,
            t,
            d,
            tr.next_power_of_2(t),
            residual,
            num_warps=8,
        )
        ctx.save_for_backward(u, positions, log_tau, theta, mix, memory)
        ctx.residual = residual
        return y

    @staticmethod
    def backward(ctx, dy):
        u, positions, log_tau, theta, mix, memory = ctx.saved_tensors
        b, t, d = u.shape
        du = torch.empty_like(u)
        gp = torch.empty((b, d // 2, 4, 2), device=u.device, dtype=torch.float32)
        gw = torch.empty_like(gp)
        backward_kernel[(b, d // 2)](
            u,
            positions,
            log_tau,
            theta,
            mix,
            memory,
            dy.contiguous(),
            du,
            gp,
            gw,
            t,
            d,
            tr.next_power_of_2(t),
            ctx.residual,
            num_warps=8,
        )
        grads = gp.sum((0, 1))
        mixgrad = gw.sum(0).permute(1, 0, 2).reshape(4, d)
        return du, None, grads[:, 0], grads[:, 1], mixgrad, None


def rotating_memory(u, positions, log_tau, theta, mix, residual=True):
    return _Rotating.apply(u, positions, log_tau, theta, mix, residual)
