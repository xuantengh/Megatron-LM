# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Gate and normalization pre-passes for the CuTe DSL GDP kernel.

These are the steps FLA performs in ``chunk_gated_delta_product_fwd`` before any
matmul, which the CuTe kernel does not fold in. Keeping them here (rather than
inside the CuTe kernel) matches the sibling ``cutedsl_mamba2_ssd`` package, where
``softplus`` and the per-chunk cumsum live in ``_fused_cumsum.py``.

Rather than reimplementing the scans, this module reuses FLA's already-tuned
Triton kernels — ``fla.ops.utils.chunk_local_cumsum`` and
``fla.modules.l2norm.l2norm_fwd`` — so the two backends share bit-identical
preprocessing and only the main recurrence differs.

Base of the exponential
-----------------------
FLA pre-scales the gate by ``RCP_LN2 = 1/ln 2`` inside the cumsum and then uses
``exp2`` in every consumer kernel (``chunk_deltaproduct_h.py``,
``chunk_deltaproduct_o.py``). ``2^(x/ln2) == e^x``, so this is exact, and
``exp2`` is a single SASS instruction. The CuTe kernel **must** follow the same
convention: the cumulative gates handed to it are already divided by ``ln 2``
and every decay inside the kernel is ``exp2``, never ``exp``.
"""

# pylint: disable=line-too-long

import torch
from fla.modules.l2norm import l2norm_fwd as _fla_l2norm_fwd
from fla.ops.utils import chunk_local_cumsum as _fla_chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2


def l2norm_fwd(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize over the last dimension.

    Applied to ``q`` and ``k`` when the caller passes
    ``use_qk_l2norm_in_kernel=True``, matching FLA, which normalizes inside its
    autograd function rather than in the chunk kernels.

    Args:
        x: Tensor whose last dimension is the head dimension.

    Returns:
        The normalized tensor, same shape and dtype as ``x``.
    """
    y = _fla_l2norm_fwd(x)
    # FLA's l2norm_fwd returns (y, rstd) in some versions and y alone in others.
    return y[0] if isinstance(y, tuple) else y


def build_g_interleaved(g: torch.Tensor, num_householder: int) -> torch.Tensor:
    """Expand the token-timeline gate onto the delta-product timeline.

    Writes ``g`` into sub-step 0 of every token and leaves the remaining
    ``M - 1`` sub-steps at zero, so that (in log space) only the first delta
    update of each token is decayed and the rest are pure delta-rule steps.

    Args:
        g: Log-space gate, ``[B, T, H]``.
        num_householder: ``M``.

    Returns:
        ``[B, T * M, H]``, float32, contiguous.
    """
    B, T, H = g.shape
    out = g.new_zeros(B, T, num_householder, H, dtype=torch.float32)
    out[:, :, 0] = g
    return out.reshape(B, T * num_householder, H).contiguous()


def fused_gdp_gate_preprocess(
    g: torch.Tensor | None,
    num_householder: int,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    meta: dict | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build the two chunk-local cumulative gates the kernel consumes.

    Produces the token-timeline gate (used by the readout stage) and the
    delta-product-timeline gate (used by the WY and state stages), each scaled by
    ``RCP_LN2`` so the kernel can use ``exp2``. Both scans are chunk-local:
    inclusive within a chunk, reset at every chunk and document boundary. Note
    ``chunk_size`` is applied in each timeline's own units, so the delta-product
    scan resets every ``chunk_size`` **sub-steps**, not every ``chunk_size``
    tokens.

    Args:
        g: Log-space gate, ``[B, T, H]``, or ``None`` for the ungated variant.
        num_householder: ``M``.
        chunk_size: Chunk length, in each timeline's own units.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units.
        meta: Optional launch metadata from ``gdp_cutedsl._chunk_meta``; when
            given, its cached ``chunk_indices`` tensors are reused instead of
            being rebuilt (each rebuild costs a device-to-host sync).

    Returns:
        ``(g_cumsum, g_interleaved_cumsum)`` with shapes ``[B, T, H]`` and
        ``[B, T * M, H]``, both float32. ``(None, None)`` when ``g is None``.
    """
    if g is None:
        return None, None

    cu_seqlens_dp = cu_seqlens * num_householder if cu_seqlens is not None else None
    chunk_indices = meta.get("chunk_indices") if meta is not None else None
    chunk_indices_dp = meta.get("chunk_indices_dp") if meta is not None else None

    g_cumsum = _fla_chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices,
    )
    g_interleaved_cumsum = _fla_chunk_local_cumsum(
        build_g_interleaved(g, num_householder),
        chunk_size=chunk_size,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices_dp,
    )
    return g_cumsum, g_interleaved_cumsum
