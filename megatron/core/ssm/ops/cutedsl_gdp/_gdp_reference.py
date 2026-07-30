# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Torch reference for the Gated Delta Product recurrence, used by tests.

Deliberately written as a plain sequential loop over sub-steps rather than a
chunked formulation: it is the definition the CuTe kernel is checked against, so
it should be obviously correct rather than fast. Numerically it matches
``fla.ops.gated_delta_product.chunk_gated_delta_product`` to within bf16 chunked-
accumulation tolerance.

Recurrence, per head, with state ``S`` of shape ``[K, V]``::

    for t in range(T):
        S = exp(g[t]) * S                       # gate once per token
        for m in range(M):                      # M householder / delta updates
            i = t * M + m
            S = S + beta[i] * k[i][:, None] * (v[i] - S.T @ k[i])[None, :]
        o[t] = S.T @ q[t] * scale               # one readout per token
"""

import torch


def gdp_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    num_householder: int,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute the GDP recurrence in float64 with an explicit sub-step loop.

    Argument shapes and semantics are identical to
    :func:`~megatron.core.ssm.ops.cutedsl_gdp.gdp_cutedsl.chunk_gated_delta_product_cutedsl`.

    Args:
        q: Queries, ``[B, T, H, K]``.
        k: Keys, ``[B, T * M, H, K]``.
        v: Values, ``[B, T * M, H, V]``.
        g: Log-space gate, ``[B, T, H]``, or ``None``.
        beta: Step sizes, ``[B, T * M, H]``.
        num_householder: ``M``.
        scale: Query scale. Defaults to ``K ** -0.5``.
        initial_state: Seed state, ``[N, H, K, V]``.
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: L2-normalize ``q`` and ``k`` first.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units.

    Returns:
        ``(o, final_state)`` with ``o`` of shape ``[B, T, H, V]``.
    """
    raise NotImplementedError("gdp_reference: sequential float64 recurrence")


def gdp_chunked_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    num_householder: int,
    chunk_size: int,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked WY-transform reference mirroring the kernel's own decomposition.

    Useful as an intermediate target when debugging: it isolates errors in the
    chunked formulation (the ``(I - A)^-1`` solve, the state pass, the two
    cumulative-gate timelines) from errors in the CuTe lowering, because it
    shares the former with the kernel and none of the latter.

    Args:
        q: Queries, ``[B, T, H, K]``.
        k: Keys, ``[B, T * M, H, K]``.
        v: Values, ``[B, T * M, H, V]``.
        g: Log-space gate, ``[B, T, H]``, or ``None``.
        beta: Step sizes, ``[B, T * M, H]``.
        num_householder: ``M``.
        chunk_size: Chunk length on the token timeline.
        scale: Query scale. Defaults to ``K ** -0.5``.
        initial_state: Seed state, ``[N, H, K, V]``.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units.

    Returns:
        ``(o, final_state)`` with ``o`` of shape ``[B, T, H, V]``.
    """
    raise NotImplementedError("gdp_chunked_reference: per-chunk WY transform + state pass")
