import torch
import triton
import triton.language as tl


def _get_fused_configs():
    """Symmetric configs: BLOCK_SIZE_M == BLOCK_SIZE_N to avoid mirror overlap."""
    return [
        triton.Config(
            {"BLOCK_SIZE_M": b, "BLOCK_SIZE_N": b, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8},
            num_stages=s,
            num_warps=w,
        )
        for b in [64, 128]
        for bk in [64, 128]
        for s, w in [(3, 4), (3, 8), (4, 4)]
    ]


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    return batch_idx, pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N


@triton.jit
def _load_block(
    X_ptr,
    Y_ptr,
    offs_row,
    offs_k,
    k_mask,
    Mx,
    x_stride_r,
    x_stride_c,
    y_stride_r,
    y_stride_c,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Load a (BLOCK_SIZE_ROW, BLOCK_SIZE_K) tile from the virtual stacked
    matrix Z = [X; Y].  Rows [0, Mx) come from X, rows [Mx, Mx+My) from Y.

    We load from both X and Y using clamped indices and blend with a mask.
    This is branchless and works regardless of whether the tile spans the
    X/Y boundary.
    """
    # Which rows belong to X vs Y
    is_x = offs_row < Mx  # (BLOCK_SIZE_ROW,)

    # Clamp indices so both pointer arrays are in-bounds (masked out later)
    x_rows = tl.where(is_x, offs_row, 0)
    y_rows = tl.where(is_x, 0, offs_row - Mx)

    x_ptrs = X_ptr + (x_rows[:, None] * x_stride_r + offs_k[None, :] * x_stride_c)
    y_ptrs = Y_ptr + (y_rows[:, None] * y_stride_r + offs_k[None, :] * y_stride_c)

    x_vals = tl.load(x_ptrs, mask=is_x[:, None] & k_mask[None, :], other=0.0)
    y_vals = tl.load(y_ptrs, mask=(~is_x)[:, None] & k_mask[None, :], other=0.0)

    return x_vals + y_vals


# ---------------------------------------------------------------------------
# Fused kernel: computes [X; Y] @ diag(w) @ [X; Y]^T as a single symmetric
# matmul, writing results to three separate output matrices XTX, XTY, YTY.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_get_fused_configs(),
    key=["M_total", "K", "x_stride_r", "x_stride_c", "y_stride_r", "y_stride_c"],
    reset_to_zero=["C_XTX_ptr", "C_XTY_ptr", "C_YTY_ptr"],
)
@triton.jit
def fused_weighted_outer_kernel(
    # Input pointers
    X_ptr,
    Y_ptr,
    W_ptr,
    # Output pointers
    C_XTX_ptr,
    C_XTY_ptr,
    C_YTY_ptr,
    # Dimensions
    Mx,
    My,
    K,
    M_total,
    # Input strides
    x_stride_b,
    x_stride_r,
    x_stride_c,
    y_stride_b,
    y_stride_r,
    y_stride_c,
    # Output strides
    xtx_stride_b,
    xtx_stride_r,
    xtx_stride_c,
    xty_stride_b,
    xty_stride_r,
    xty_stride_c,
    yty_stride_b,
    yty_stride_r,
    yty_stride_c,
    # Tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid,
        M_total,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        GROUP_SIZE_M,
    )

    # Symmetric: skip blocks above the diagonal
    if m_idx + BLOCK_SIZE_M <= n_idx:
        return

    # Advance to batch
    X_ptr += batch_idx * x_stride_b
    Y_ptr += batch_idx * y_stride_b
    C_XTX_ptr += batch_idx * xtx_stride_b
    C_XTY_ptr += batch_idx * xty_stride_b
    C_YTY_ptr += batch_idx * yty_stride_b

    # Row and column offsets in the virtual (Mx+My) × (Mx+My) output
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M_total
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M_total
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining
        k_offset = k * BLOCK_SIZE_K

        # Load row tile from Z = [X; Y]
        a = _load_block(
            X_ptr,
            Y_ptr,
            offs_m,
            offs_k + k_offset,
            k_mask,
            Mx,
            x_stride_r,
            x_stride_c,
            y_stride_r,
            y_stride_c,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K,
        )

        # Load col tile (transposed) from Z
        bt = _load_block(
            X_ptr,
            Y_ptr,
            offs_n,
            offs_k + k_offset,
            k_mask,
            Mx,
            x_stride_r,
            x_stride_c,
            y_stride_r,
            y_stride_c,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )

        # Apply weights
        w = tl.load(W_ptr + k_offset + offs_k, mask=k_mask, other=0.0)
        a = (a * w[None, :]).to(a.dtype)

        acc = tl.dot(a, tl.trans(bt), acc)

    out_dtype = C_XTX_ptr.dtype.element_ty
    c_out = acc.to(out_dtype)

    # Store primary block — dispatch to correct output matrix
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    _store_tile(
        c_out,
        C_XTX_ptr,
        C_XTY_ptr,
        C_YTY_ptr,
        offs_cm,
        offs_cn,
        Mx,
        My,
        xtx_stride_r,
        xtx_stride_c,
        xty_stride_r,
        xty_stride_c,
        yty_stride_r,
        yty_stride_c,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    # Mirror store (skip diagonal blocks)
    if m_idx != n_idx:
        c_out_t = tl.trans(c_out)
        _store_tile(
            c_out_t,
            C_XTX_ptr,
            C_XTY_ptr,
            C_YTY_ptr,
            offs_cn,
            offs_cm,
            Mx,
            My,
            xtx_stride_r,
            xtx_stride_c,
            xty_stride_r,
            xty_stride_c,
            yty_stride_r,
            yty_stride_c,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
        )


@triton.jit
def _store_tile(
    output,
    C_XTX_ptr,
    C_XTY_ptr,
    C_YTY_ptr,
    row_abs,
    col_abs,
    Mx,
    My,
    xtx_stride_r,
    xtx_stride_c,
    xty_stride_r,
    xty_stride_c,
    yty_stride_r,
    yty_stride_c,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """Store output[i, j] to the correct matrix based on which quadrant
    (row_abs[i], col_abs[j]) falls in within the virtual (Mx+My)×(Mx+My) grid.

    Quadrant mapping:
        row < Mx  and col < Mx  → XTX[row, col]
        row < Mx  and col >= Mx → XTY[row, col - Mx]
        row >= Mx and col < Mx  → XTY[col, row - Mx]   (YTX = XTY^T)
        row >= Mx and col >= Mx → YTY[row - Mx, col - Mx]
    """
    row_is_x = row_abs < Mx
    col_is_x = col_abs < Mx
    row_valid = row_abs < (Mx + My)
    col_valid = col_abs < (Mx + My)
    valid = row_valid[:, None] & col_valid[None, :]

    # XTX: row < Mx, col < Mx
    xtx_mask = row_is_x[:, None] & col_is_x[None, :] & valid
    xtx_ptrs = C_XTX_ptr + (row_abs[:, None] * xtx_stride_r + col_abs[None, :] * xtx_stride_c)
    tl.store(xtx_ptrs, output, mask=xtx_mask)

    # XTY: row < Mx, col >= Mx → XTY[row, col - Mx]
    xty_mask = row_is_x[:, None] & (~col_is_x)[None, :] & valid
    xty_cols = tl.where(col_is_x, 0, col_abs - Mx)  # clamp for safety
    xty_ptrs = C_XTY_ptr + (row_abs[:, None] * xty_stride_r + xty_cols[None, :] * xty_stride_c)
    tl.store(xty_ptrs, output, mask=xty_mask)

    # YTX: row >= Mx, col < Mx → XTY[col, row - Mx]  (transposed into XTY)
    ytx_mask = (~row_is_x)[:, None] & col_is_x[None, :] & valid
    ytx_rows = tl.where(row_is_x, 0, row_abs - Mx)  # clamp for safety
    ytx_ptrs = C_XTY_ptr + (col_abs[None, :] * xty_stride_r + ytx_rows[:, None] * xty_stride_c)
    tl.store(ytx_ptrs, output, mask=ytx_mask)

    # YTY: row >= Mx, col >= Mx
    yty_mask = (~row_is_x)[:, None] & (~col_is_x)[None, :] & valid
    yty_rows = tl.where(row_is_x, 0, row_abs - Mx)
    yty_cols = tl.where(col_is_x, 0, col_abs - Mx)
    yty_ptrs = C_YTY_ptr + (yty_rows[:, None] * yty_stride_r + yty_cols[None, :] * yty_stride_c)
    tl.store(yty_ptrs, output, mask=yty_mask)


# ---------------------------------------------------------------------------
# Python interface
# ---------------------------------------------------------------------------


def weighted_outer_products(
    X: torch.Tensor,
    Y: torch.Tensor,
    w: torch.Tensor,
    XTX_out: torch.Tensor,
    XTY_out: torch.Tensor,
    YTY_out: torch.Tensor,
    accumulate: bool = True,
) -> None:
    """
    Compute in a single fused kernel launch:

        XTX_out (+)= X  @ diag(w) @ X^T     (Mx × Mx, symmetric)
        XTY_out (+)= X  @ diag(w) @ Y^T     (Mx × My)
        YTY_out (+)= Y  @ diag(w) @ Y^T     (My × My, symmetric)

    Conceptually computes Z @ diag(w) @ Z^T where Z = [X; Y] stacked along
    rows, then dispatches quadrants to the three output matrices.

    Args:
        X:       (B, Mx, K) or (Mx, K)
        Y:       (B, My, K) or (My, K)
        w:       (K,)       weights
        XTX_out: (B, Mx, Mx) or (Mx, Mx)
        XTY_out: (B, Mx, My) or (Mx, My)
        YTY_out: (B, My, My) or (My, My)
        accumulate: if True, adds to existing output; if False, overwrites.
    """
    if not (X.shape[-1] == Y.shape[-1] == w.shape[0]):
        raise ValueError(f"Shape mismatch: X[-1]={X.shape[-1]}, Y[-1]={Y.shape[-1]}, w[0]={w.shape[0]} must be equal")
    Mx, K = X.shape[-2], X.shape[-1]
    My = Y.shape[-2]
    M_total = Mx + My
    batch = X.shape[0] if X.ndim == 3 else 1

    if accumulate:
        xtx_tmp = torch.zeros_like(XTX_out)
        xty_tmp = torch.zeros_like(XTY_out)
        yty_tmp = torch.zeros_like(YTY_out)
    else:
        xtx_tmp = XTX_out
        xty_tmp = XTY_out
        yty_tmp = YTY_out

    def grid(meta):
        return (batch * triton.cdiv(M_total, meta["BLOCK_SIZE_M"]) * triton.cdiv(M_total, meta["BLOCK_SIZE_N"]),)

    fused_weighted_outer_kernel[grid](
        X_ptr=X,
        Y_ptr=Y,
        W_ptr=w,
        C_XTX_ptr=xtx_tmp,
        C_XTY_ptr=xty_tmp,
        C_YTY_ptr=yty_tmp,
        Mx=Mx,
        My=My,
        K=K,
        M_total=M_total,
        x_stride_b=X.stride(0) if X.ndim == 3 else 0,
        x_stride_r=X.stride(-2),
        x_stride_c=X.stride(-1),
        y_stride_b=Y.stride(0) if Y.ndim == 3 else 0,
        y_stride_r=Y.stride(-2),
        y_stride_c=Y.stride(-1),
        xtx_stride_b=xtx_tmp.stride(0) if xtx_tmp.ndim == 3 else 0,
        xtx_stride_r=xtx_tmp.stride(-2),
        xtx_stride_c=xtx_tmp.stride(-1),
        xty_stride_b=xty_tmp.stride(0) if xty_tmp.ndim == 3 else 0,
        xty_stride_r=xty_tmp.stride(-2),
        xty_stride_c=xty_tmp.stride(-1),
        yty_stride_b=yty_tmp.stride(0) if yty_tmp.ndim == 3 else 0,
        yty_stride_r=yty_tmp.stride(-2),
        yty_stride_c=yty_tmp.stride(-1),
    )

    if accumulate:
        XTX_out.add_(xtx_tmp)
        XTY_out.add_(xty_tmp)
        YTY_out.add_(yty_tmp)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, Mx, My, K = 2, 256, 192, 512
    device = "cuda"
    dtype = torch.float32

    X = torch.randn(B, Mx, K, device=device, dtype=dtype)
    Y = torch.randn(B, My, K, device=device, dtype=dtype)
    w = torch.rand(K, device=device, dtype=dtype) + 0.1

    # Reference
    W = torch.diag(w)
    XTX_ref = X @ W @ X.transpose(-1, -2)
    XTY_ref = X @ W @ Y.transpose(-1, -2)
    YTY_ref = Y @ W @ Y.transpose(-1, -2)

    # Triton
    XTX_out = torch.zeros(B, Mx, Mx, device=device, dtype=dtype)
    XTY_out = torch.zeros(B, Mx, My, device=device, dtype=dtype)
    YTY_out = torch.zeros(B, My, My, device=device, dtype=dtype)

    weighted_outer_products(X, Y, w, XTX_out, XTY_out, YTY_out, accumulate=True)

    rtol, atol = 1e-3, 1e-1
    print(
        "XTX close:",
        torch.allclose(XTX_out, XTX_ref, rtol=rtol, atol=atol),
        f" max err: {(XTX_out - XTX_ref).abs().max().item():.6f}",
    )
    print(
        "XTY close:",
        torch.allclose(XTY_out, XTY_ref, rtol=rtol, atol=atol),
        f" max err: {(XTY_out - XTY_ref).abs().max().item():.6f}",
    )
    print(
        "YTY close:",
        torch.allclose(YTY_out, YTY_ref, rtol=rtol, atol=atol),
        f" max err: {(YTY_out - YTY_ref).abs().max().item():.6f}",
    )

    # Test accumulation
    XTX_out2 = XTX_out.clone()
    XTY_out2 = XTY_out.clone()
    YTY_out2 = YTY_out.clone()
    weighted_outer_products(X, Y, w, XTX_out2, XTY_out2, YTY_out2, accumulate=True)
    print(
        "XTX accum close:",
        torch.allclose(XTX_out2, 2 * XTX_ref, rtol=rtol, atol=atol),
        f" max err: {(XTX_out2 - 2*XTX_ref).abs().max().item():.6f}",
    )
    print(
        "XTY accum close:",
        torch.allclose(XTY_out2, 2 * XTY_ref, rtol=rtol, atol=atol),
        f" max err: {(XTY_out2 - 2*XTY_ref).abs().max().item():.6f}",
    )
    print(
        "YTY accum close:",
        torch.allclose(YTY_out2, 2 * YTY_ref, rtol=rtol, atol=atol),
        f" max err: {(YTY_out2 - 2*YTY_ref).abs().max().item():.6f}",
    )

    # Test overwrite
    XTX_out3 = torch.full((B, Mx, Mx), 999.0, device=device, dtype=dtype)
    XTY_out3 = torch.full((B, Mx, My), 999.0, device=device, dtype=dtype)
    YTY_out3 = torch.full((B, My, My), 999.0, device=device, dtype=dtype)
    weighted_outer_products(X, Y, w, XTX_out3, XTY_out3, YTY_out3, accumulate=False)
    print(
        "XTX overwrite:",
        torch.allclose(XTX_out3, XTX_ref, rtol=rtol, atol=atol),
        f" max err: {(XTX_out3 - XTX_ref).abs().max().item():.6f}",
    )
    print(
        "XTY overwrite:",
        torch.allclose(XTY_out3, XTY_ref, rtol=rtol, atol=atol),
        f" max err: {(XTY_out3 - XTY_ref).abs().max().item():.6f}",
    )
    print(
        "YTY overwrite:",
        torch.allclose(YTY_out3, YTY_ref, rtol=rtol, atol=atol),
        f" max err: {(YTY_out3 - YTY_ref).abs().max().item():.6f}",
    )
