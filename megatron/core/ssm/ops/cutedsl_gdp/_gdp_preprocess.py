# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Triton pre-passes for the CuTe DSL GDP kernel.

These are the steps FLA performs in ``chunk_gated_delta_product_fwd`` before any
matmul, which the CuTe kernel does not fold in. Keeping them here (rather than
inside the CuTe kernel) matches the sibling SSD package, where ``softplus`` and
the per-chunk cumsum live in ``_fused_cumsum.py``.

Reference semantics, from ``fla/ops/gated_delta_product/chunk.py``::

    g_interleaved = zeros(B, T, M, H, dtype=float32)
    g_interleaved[:, :, 0] = g          # gate only on the first sub-step
    g_interleaved = rearrange(g_interleaved, 'b l n h -> b (l n) h')
    g            = chunk_local_cumsum(g,            64, cu_seqlens)
    g_interleaved = chunk_local_cumsum(g_interleaved, 64, cu_seqlens * M)

Both cumsums are *chunk-local*: inclusive within each chunk, reset at every chunk
and every document boundary. Cross-chunk propagation is the kernel's job, carried
through the recurrent state.
"""

import torch


def l2norm_fwd(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize over the last dimension.

    Applied to ``q`` and ``k`` when the caller passes
    ``use_qk_l2norm_in_kernel=True``, matching FLA's in-kernel normalization.

    Args:
        x: Tensor whose last dimension is the head dimension.

    Returns:
        The normalized tensor, same shape and dtype as ``x``.
    """
    raise NotImplementedError("l2norm_fwd: fused rsqrt normalization over the head dim")


def build_g_interleaved(g: torch.Tensor, num_householder: int) -> torch.Tensor:
    """Expand the token-timeline gate onto the delta-product timeline.

    Writes ``g`` into sub-step 0 of every token and leaves the remaining ``M - 1``
    sub-steps at zero, so that (in log space) only the first delta update of each
    token is decayed.

    Args:
        g: Log-space gate, ``[B, T, H]``.
        num_householder: ``M``.

    Returns:
        ``[B, T * M, H]`` float32 tensor.
    """
    raise NotImplementedError("build_g_interleaved: scatter g into sub-step 0 and flatten")


def chunk_local_cumsum(
    g: torch.Tensor, chunk_size: int, cu_seqlens: torch.Tensor | None = None
) -> torch.Tensor:
    """Inclusive cumulative sum within each chunk, reset at chunk and document boundaries.

    Args:
        g: Log-space gate, ``[B, T, H]``.
        chunk_size: Chunk length on ``g``'s own timeline.
        cu_seqlens: Cumulative sequence lengths on ``g``'s own timeline, or
            ``None`` for a fixed-length batch.

    Returns:
        Float32 tensor of the same shape as ``g``.
    """
    raise NotImplementedError("chunk_local_cumsum: per-chunk inclusive scan")


def fused_gdp_gate_preprocess(
    g: torch.Tensor | None,
    num_householder: int,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    meta: dict,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build both cumulative gates in a single Triton launch.

    Fuses :func:`build_g_interleaved` and the two :func:`chunk_local_cumsum`
    passes so the gate is read once. Returns ``(None, None)`` for the ungated
    variant, where the kernel is compiled with the gate path switched off.

    Args:
        g: Log-space gate, ``[B, T, H]``, or ``None``.
        num_householder: ``M``.
        chunk_size: Chunk length on the token timeline. The delta-product
            timeline is chunked at ``chunk_size * M``.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units.
        meta: Launch metadata from ``gdp_cutedsl._chunk_meta``.

    Returns:
        ``(g_cumsum, g_interleaved_cumsum)`` with shapes ``[B, T, H]`` and
        ``[B, T * M, H]``, both float32, chunk-major to match the kernel's TMA
        descriptors.
    """
    if g is None:
        return None, None
    raise NotImplementedError("fused_gdp_gate_preprocess: single-launch interleave + dual cumsum")
