# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Torch references for the Gated Delta Product recurrence, used by tests.

Two references, forming a debugging ladder:

* :func:`gdp_reference` — a sequential float64 loop. This is the definition; it
  makes no assumption about chunking and is the ground truth for correctness.
* :func:`gdp_chunked_reference` — the same math expressed with the chunked WY
  decomposition the fused kernel uses. It shares the *algorithm* with the kernel
  and none of the CuTe lowering, so a mismatch against :func:`gdp_reference`
  isolates a decomposition bug while a mismatch against the kernel isolates a
  lowering bug.

FLA ships its own references (``fla.ops.gated_delta_product.naive`` and
``.chunk_ref``) but neither honours ``cu_seqlens`` — ``naive`` accepts the
argument and ignores it — so they cannot validate the packed-THD path this
backend targets. :func:`fla_reference` wraps ``chunk_ref`` for the fixed-length
case as a cross-check.

Base of the exponential
-----------------------
These references use natural ``exp`` on the raw gate. FLA (and the fused kernel)
pre-scale the gate by ``RCP_LN2 = 1/ln 2`` in the cumsum and then use ``exp2``,
which is mathematically identical (``2^(x/ln2) == e^x``) and one SASS
instruction instead of several. Do not "fix" the reference to match.
"""

import math

import torch


def _seq_bounds(cu_seqlens: torch.Tensor | None, batch: int, seq_len: int) -> list[tuple[int, int, int]]:
    """Return ``(batch_index, begin, end)`` triples covering every sequence.

    Args:
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units, or ``None``.
        batch: ``B``.
        seq_len: ``T``.

    Returns:
        One triple per sequence. For the fixed-length case each batch row is one
        sequence; for the varlen case ``B`` must be 1 and the triples come from
        ``cu_seqlens``.
    """
    if cu_seqlens is None:
        return [(b, 0, seq_len) for b in range(batch)]
    assert batch == 1, "cu_seqlens requires the batch to be flattened to B == 1"
    bounds = cu_seqlens.tolist()
    return [(0, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


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

    Per head, with state ``S`` of shape ``[K, V]``::

        for t in range(T):
            S = exp(g[t]) * S                                  # gate once per token
            for m in range(M):                                 # M delta updates
                i = t * M + m
                S = S + beta[i] * outer(k[i], v[i] - S.T @ k[i])
            o[t] = (S.T @ q[t]) * scale                        # one readout per token

    Args:
        q: Queries, ``[B, T, H, K]``.
        k: Keys, ``[B, T * M, H, K]``.
        v: Values, ``[B, T * M, H, V]``.
        g: Log-space gate, ``[B, T, H]``, or ``None`` for the ungated variant.
        beta: Step sizes, ``[B, T * M, H]``, already passed through sigmoid.
        num_householder: ``M``.
        scale: Query scale. Defaults to ``K ** -0.5``.
        initial_state: Seed state, ``[N, H, K, V]``.
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: L2-normalize ``q`` and ``k`` before the recurrence.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units.

    Returns:
        ``(o, final_state)``; ``o`` has shape ``[B, T, H, V]`` and the input dtype,
        ``final_state`` has shape ``[N, H, K, V]`` or is ``None``.
    """
    out_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    M = num_householder
    if scale is None:
        scale = K**-0.5

    q64, k64, v64, beta64 = (t.double() for t in (q, k, v, beta))
    if use_qk_l2norm_in_kernel:
        q64 = q64 / q64.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k64 = k64 / k64.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    g64 = g.double() if g is not None else None

    o = torch.zeros(B, T, H, V, dtype=torch.float64, device=q.device)
    sequences = _seq_bounds(cu_seqlens, B, T)
    final = torch.zeros(len(sequences), H, K, V, dtype=torch.float64, device=q.device)

    for n, (b, begin, end) in enumerate(sequences):
        for h in range(H):
            state = torch.zeros(K, V, dtype=torch.float64, device=q.device)
            if initial_state is not None:
                state = initial_state[n, h].double().clone()
            for t in range(begin, end):
                if g64 is not None:
                    state = state * torch.exp(g64[b, t, h])
                for m in range(M):
                    i = t * M + m
                    k_i = k64[b, i, h]
                    v_i = v64[b, i, h]
                    # delta rule: move the state's prediction of v toward v itself
                    state = state + beta64[b, i, h] * torch.outer(k_i, v_i - state.transpose(0, 1) @ k_i)
                o[b, t, h] = (state.transpose(0, 1) @ q64[b, t, h]) * scale
            final[n, h] = state

    return o.to(out_dtype), (final.to(out_dtype) if output_final_state else None)


def gdp_chunked_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    num_householder: int,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked WY-transform reference mirroring the fused kernel's decomposition.

    Reproduces FLA's four-stage pipeline exactly, including the two chunk grids.
    ``chunk_size`` counts **sub-steps** for the WY stages and **tokens** for the
    state and readout stages, so one token-chunk spans ``M`` WY sub-chunks — the
    ``i_t % num_householder == 0`` condition in FLA's ``chunk_deltaproduct_h``.

    Per WY sub-chunk (``chunk_size`` sub-steps)::

        A  = tril_strict(beta_i * (k_i . k_j) * exp(gd_i - gd_j))
        Ai = (I + A)^-1
        w  = Ai @ (k * beta * exp(gd));  u = Ai @ (v * beta)

    Per WY sub-chunk, carrying ``S`` of shape ``[K, V]``::

        v_new = u - w @ S
        v_new *= exp(gd_last - gd);  S *= exp(gd_last);  S += k^T @ v_new

    Per token-chunk (``chunk_size`` tokens), with ``h`` the state at its start::

        o = (q @ h) * exp(gt)
        for m in range(M):
            o += (tril(q @ k_m^T) * exp(gt_i - gt_j)) @ v_new_m
        o *= scale

    Args:
        q: Queries, ``[B, T, H, K]``.
        k: Keys, ``[B, T * M, H, K]``.
        v: Values, ``[B, T * M, H, V]``.
        g: Log-space gate, ``[B, T, H]``, or ``None``.
        beta: Step sizes, ``[B, T * M, H]``.
        num_householder: ``M``.
        chunk_size: Chunk length; sub-steps for WY, tokens for state/readout.
        scale: Query scale. Defaults to ``K ** -0.5``.
        initial_state: Seed state, ``[N, H, K, V]``.
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: L2-normalize ``q`` and ``k`` first.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units.

    Returns:
        ``(o, final_state)``; shapes as in :func:`gdp_reference`.
    """
    out_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    M, BT = num_householder, chunk_size
    if scale is None:
        scale = K**-0.5

    q64, k64, v64, beta64 = (t.double() for t in (q, k, v, beta))
    if use_qk_l2norm_in_kernel:
        q64 = q64 / q64.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k64 = k64 / k64.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    o = torch.zeros(B, T, H, V, dtype=torch.float64, device=q.device)
    sequences = _seq_bounds(cu_seqlens, B, T)
    final = torch.zeros(len(sequences), H, K, V, dtype=torch.float64, device=q.device)
    eye = torch.eye(BT, dtype=torch.float64, device=q.device)

    for n, (b, begin, end) in enumerate(sequences):
        # Chunk-local cumulative gates, reset at every chunk and document boundary.
        gt = _chunk_local_cumsum_ref(g, b, begin, end, BT, torch.float64) if g is not None else None
        gd = (
            _chunk_local_cumsum_ref(_interleave_gate(g, M), b, begin * M, end * M, BT, torch.float64)
            if g is not None
            else None
        )

        for h in range(H):
            state = torch.zeros(K, V, dtype=torch.float64, device=q.device)
            if initial_state is not None:
                state = initial_state[n, h].double().clone()

            for tok_start in range(begin, end, BT):
                tok_end = min(tok_start + BT, end)
                n_tok = tok_end - tok_start
                h_chunk = state.clone()  # state at the token-chunk boundary
                v_new_sub = []

                # --- Stages A + B, M sub-chunks of BT sub-steps each ----------
                for m in range(M):
                    lo = tok_start * M + m * BT
                    hi = min(lo + BT, tok_end * M)
                    if lo >= hi:
                        break
                    rows = hi - lo
                    k_c = k64[b, lo:hi, h]  # [rows, K]
                    v_c = v64[b, lo:hi, h]  # [rows, V]
                    beta_c = beta64[b, lo:hi, h].unsqueeze(-1)  # [rows, 1]
                    gd_c = gd[lo - begin * M : hi - begin * M, h] if gd is not None else None

                    # Stage A: WY transform.
                    a = k_c @ k_c.transpose(0, 1)
                    if gd_c is not None:
                        a = a * torch.exp(gd_c.unsqueeze(1) - gd_c.unsqueeze(0))
                    a = torch.tril(a * beta_c, diagonal=-1)
                    a_inv = torch.linalg.inv(eye[:rows, :rows] + a)
                    kb = k_c * beta_c
                    if gd_c is not None:
                        kb = kb * torch.exp(gd_c).unsqueeze(-1)
                    w = a_inv @ kb
                    u = a_inv @ (v_c * beta_c)

                    # Stage B: state pass. The readout consumes the *ungated*
                    # v_new; the decay to the sub-chunk end is applied only to the
                    # copy that feeds the state. FLA does the same by storing
                    # v_new inside the SAVE_NEW_VALUE block *before* multiplying
                    # by exp2(gd_last - gd) (chunk_deltaproduct_h.py). Applying
                    # the gate to the stored copy double-counts the decay,
                    # because stage C already carries it in the token-timeline
                    # mask -- and it leaves the final state exactly right, so the
                    # error shows up only in the output.
                    v_new = u - w @ state
                    v_new_sub.append(v_new)
                    if gd_c is not None:
                        gd_last = gd_c[-1]
                        v_new = v_new * torch.exp(gd_last - gd_c).unsqueeze(-1)
                        state = state * torch.exp(gd_last)
                    state = state + k_c.transpose(0, 1) @ v_new

                # Stage B produced v_new one *sub-chunk* at a time; the readout
                # needs it indexed by *sub-step within each token*. Concatenating
                # restores the contiguous delta-product timeline for this
                # token-chunk, after which `[m::M]` is the stride-M gather the
                # readout wants (FLA reaches the same rows via the `i_dp*H*V`
                # offset and `num_householder*H*V` stride in chunk_deltaproduct_o).
                v_new_all = torch.cat(v_new_sub, dim=0)  # [n_tok * M, V]

                # --- Stage C: readout on the token timeline ------------------
                q_c = q64[b, tok_start:tok_end, h]  # [n_tok, K]
                gt_c = gt[tok_start - begin : tok_end - begin, h] if gt is not None else None
                out_c = q_c @ h_chunk  # [n_tok, V]
                if gt_c is not None:
                    out_c = out_c * torch.exp(gt_c).unsqueeze(-1)
                    mask = torch.tril(torch.exp(gt_c.unsqueeze(1) - gt_c.unsqueeze(0)))
                else:
                    mask = torch.tril(torch.ones(n_tok, n_tok, dtype=torch.float64, device=q.device))
                for m in range(M):
                    k_m = k64[b, tok_start * M + m : tok_end * M : M, h]  # [n_tok, K]
                    v_m = v_new_all[m::M]  # [n_tok, V]
                    out_c = out_c + (torch.tril(q_c @ k_m.transpose(0, 1)) * mask) @ v_m
                o[b, tok_start:tok_end, h] = out_c * scale

            final[n, h] = state

    return o.to(out_dtype), (final.to(out_dtype) if output_final_state else None)


def _interleave_gate(g: torch.Tensor, num_householder: int) -> torch.Tensor:
    """Scatter the token-timeline gate into sub-step 0 of each token.

    Args:
        g: Log-space gate, ``[B, T, H]``.
        num_householder: ``M``.

    Returns:
        ``[B, T * M, H]``; sub-steps ``1 .. M-1`` are zero, i.e. undecayed.
    """
    B, T, H = g.shape
    out = g.new_zeros(B, T, num_householder, H)
    out[:, :, 0] = g
    return out.reshape(B, T * num_householder, H)


def _chunk_local_cumsum_ref(
    g: torch.Tensor, b: int, begin: int, end: int, chunk_size: int, dtype: torch.dtype
) -> torch.Tensor:
    """Inclusive cumulative sum within each chunk, reset at chunk boundaries.

    Args:
        g: Gate tensor, ``[B, L, H]``, on whichever timeline the caller wants.
        b: Batch index to slice.
        begin: First index of this sequence on that timeline.
        end: One past the last index.
        chunk_size: Chunk length in that timeline's own units.
        dtype: Accumulation dtype.

    Returns:
        ``[end - begin, H]``, chunk-local inclusive prefix sums.
    """
    seq = g[b, begin:end].to(dtype)
    out = torch.empty_like(seq)
    for start in range(0, seq.shape[0], chunk_size):
        stop = min(start + chunk_size, seq.shape[0])
        out[start:stop] = torch.cumsum(seq[start:stop], dim=0)
    return out


def fla_reference(
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
    """Cross-check against FLA's own reference, which routes GDP through gated delta rule.

    ``fla.ops.gated_delta_product.chunk_ref`` expands the query onto the
    delta-product timeline (placing it in the last of each token's ``M`` slots)
    and calls ``chunk_gated_delta_rule``. Useful as an independent oracle, but it
    inherits FLA's chunked numerics, so prefer :func:`gdp_reference` for
    tolerance-tight comparisons.

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
        use_qk_l2norm_in_kernel: L2-normalize ``q`` and ``k`` inside FLA's kernel.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units.

    Returns:
        ``(o, final_state)``; shapes as in :func:`gdp_reference`.

    Raises:
        ImportError: If FLA is not installed.
    """
    from fla.ops.gated_delta_product.chunk_ref import chunk_gated_delta_product_ref

    return chunk_gated_delta_product_ref(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        num_householder=num_householder,
        scale=scale if scale is not None else math.sqrt(1.0 / q.shape[-1]),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
