# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Host front-end for the Blackwell CuTe DSL Gated Delta Product (GDP) kernel.

The public entry point :func:`chunk_gated_delta_product_cutedsl` is a drop-in
replacement for ``fla.ops.gated_delta_product.chunk_gated_delta_product`` as
called by :class:`~megatron.core.ssm.gated_delta_product.GatedDeltaProductMixer`,
so the mixer can dispatch to either backend without reshaping anything.

Timelines
---------
GDP runs on two interleaved timelines and nearly every shape bug comes from
confusing them:

* **token timeline** (length ``T``) — carries ``q``, ``g``, and the output ``o``.
* **delta-product timeline** (length ``T * M``, ``M = num_householder``) —
  carries ``k``, ``v``, ``beta``, and the interleaved decay ``g_interleaved``.

Each token applies ``M`` sequential rank-1 delta updates to the ``K x V`` matrix
state, with the gate applied once at the first of those ``M`` sub-steps, and then
performs a single query readout. Varlen boundaries follow the same split:
``cu_seqlens`` is in token units and ``cu_seqlens * M`` in sub-step units.

Preprocessing the kernel does NOT do (and that this front-end performs, matching
FLA semantics): the householder interleaving of ``g``, the per-chunk local
cumsum on both timelines, and the optional L2 normalization of ``q`` / ``k``.
See :mod:`._gdp_preprocess`.
"""

# pylint: disable=line-too-long

import logging
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from fla.ops.utils.index import prepare_chunk_indices

from ._gdp_kernel import GDPKernel
from ._gdp_preprocess import fused_gdp_gate_preprocess, l2norm_fwd

logger = logging.getLogger(__name__)

# Kernel chunk length. Applied in each timeline's own units, exactly as FLA does:
# the WY stages chunk the delta-product timeline every KERNEL_CHUNK_SIZE
# *sub-steps*, while the state and readout stages chunk the token timeline every
# KERNEL_CHUNK_SIZE *tokens*. One token-chunk therefore spans exactly M WY
# sub-chunks. Matches FLA's hardcoded 64 so the two backends share chunk
# boundaries and stay numerically comparable.
KERNEL_CHUNK_SIZE = 64

# Supported (K, V) head-dimension pairs. K is mamba_state_dim, V is
# mamba_head_dim -- these are asymmetric by default in GDP (128, 64); GDN-style
# configs use (128, 128). Extend as tile shapes are added.
SUPPORTED_HEAD_DIMS = ((128, 64), (128, 128))

_MAX_ACTIVE_CLUSTERS = None
_COMPILE_CACHE: dict = {}
_WORKSPACE_CACHE: dict = {}


def is_cutedsl_gdp_available() -> bool:
    """Return ``True`` if the CuTe DSL runtime is importable and the GPU is sm100+.

    Returns:
        Whether :func:`chunk_gated_delta_product_cutedsl` can be called at all.
        A ``True`` result does not imply a given batch is supported — use
        :func:`cutedsl_gdp_unsupported_reason` for the per-call guard.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _torch_to_cute_dtype(dtype: torch.dtype) -> Type[cutlass.Numeric]:
    """Map a torch io dtype onto the corresponding CuTe DSL numeric type."""
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise ValueError(f"Unsupported io dtype for CuTe DSL GDP kernel: {dtype}")


def _to_cute(torch_tensor: torch.Tensor, dynamic_modes: list[int]) -> cute.Tensor:
    """Convert a ``torch.Tensor`` to a ``cute.Tensor`` via dlpack, marking dynamic modes."""
    ct = from_dlpack(torch_tensor, assumed_align=16)
    stride_order = torch_tensor.dim_order()
    for mode in dynamic_modes:
        ct = ct.mark_compact_shape_dynamic(mode=mode, stride_order=stride_order)
    return ct


def _current_cute_stream() -> cuda.CUstream:
    """Return the current torch CUDA stream as a CUDA driver stream handle."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _chunk_meta(cu_seqlens: torch.Tensor, num_householder: int, chunk_size: int) -> dict:
    """Derive the host-side launch metadata for one varlen batch.

    Computes, for both the token and delta-product timelines, the per-sequence
    chunk counts and start offsets that the persistent tile scheduler walks, plus
    the divisibility flag the dispatch guard keys on.

    Args:
        cu_seqlens: Cumulative token counts, shape ``[N + 1]``, token units.
        num_householder: ``M``, the number of householder sub-steps per token.
        chunk_size: Kernel chunk length on the token timeline.

    Returns:
        A dict with ``N``, ``n_real_tokens``, ``total_chunks``, ``divisible``,
        ``seq_chunk_start`` and ``seq_n_chunks`` (both int32 CUDA tensors), and
        the cached ``chunk_indices`` / ``chunk_indices_dp`` tables.
    """
    device = cu_seqlens.device
    bounds = cu_seqlens.tolist()  # one D2H sync; amortized by the workspace cache
    lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    n_seq = len(lens)

    # The persistent scheduler walks whole chunks, so a partial tail would need
    # per-chunk predication that the v1 kernel does not implement. Divisibility on
    # the token timeline implies it on the delta-product timeline as well, since
    # that one is exactly M times longer.
    divisible = all(length % chunk_size == 0 for length in lens)

    chunks_per_seq = [(length + chunk_size - 1) // chunk_size for length in lens]
    starts, running = [], 0
    for count in chunks_per_seq:
        starts.append(running)
        running += count

    return {
        "N": n_seq,
        "n_real_tokens": bounds[-1],
        "total_chunks": running,
        "divisible": divisible,
        "seq_lens": lens,
        # Chunk-space offsets (first chunk index of each sequence).
        "seq_chunk_start": torch.tensor(starts, dtype=torch.int32, device=device),
        # Token-space offsets. This is what the kernel's mSeqStart wants: it uses
        # the value directly as `tok_begin`, so it must be cu_seqlens, not the
        # chunk offsets above. Conflating the two silently reads the wrong tokens.
        "seq_start_tokens": cu_seqlens.to(torch.int32),
        "seq_n_chunks": torch.tensor(chunks_per_seq, dtype=torch.int32, device=device),
        "chunk_indices": prepare_chunk_indices(cu_seqlens, chunk_size),
        "chunk_indices_dp": prepare_chunk_indices(cu_seqlens * num_householder, chunk_size),
    }


def cutedsl_gdp_unsupported_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    num_householder: int,
    *,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = KERNEL_CHUNK_SIZE,
) -> str | None:
    """Check whether this batch can run on the CuTe DSL GDP kernel.

    Mirrors ``cutedsl_unsupported_reason`` in the sibling SSD package: callers
    dispatch to FLA whenever this returns a non-``None`` reason, so every
    restriction must be stated here rather than asserted inside the kernel.

    Args:
        q: Queries, ``[B, T, H, K]``, on the token timeline.
        k: Keys, ``[B, T * M, H, K]``, on the delta-product timeline.
        v: Values, ``[B, T * M, H, V]``.
        g: Log-space forget gate, ``[B, T, H]``, or ``None`` for the ungated variant.
        beta: Delta-rule step sizes, ``[B, T * M, H]``.
        num_householder: ``M``.
        initial_state: Optional carried state, ``[N, H, K, V]``.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units.
        chunk_size: Kernel chunk length on the token timeline.

    Returns:
        ``None`` if the batch is supported, else a human-readable reason string.
    """
    if not is_cutedsl_gdp_available():
        return "CuTeDSL GDP: runtime unavailable or pre-Blackwell GPU"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"CuTeDSL GDP: unsupported io dtype {q.dtype}"
    if (q.shape[-1], v.shape[-1]) not in SUPPORTED_HEAD_DIMS:
        return f"CuTeDSL GDP: unsupported (K, V) head dims ({q.shape[-1]}, {v.shape[-1]})"
    if cu_seqlens is not None and q.shape[0] != 1:
        return "CuTeDSL GDP: varlen batches must be flattened to B == 1"
    if g is None:
        return "CuTeDSL GDP: the ungated (pure delta product) variant is not implemented"
    if beta.shape[1] != k.shape[1]:
        return "CuTeDSL GDP: beta must be on the delta-product timeline"
    if cu_seqlens is None:
        return "CuTeDSL GDP: fixed-length batches are not implemented; pass cu_seqlens"

    meta = _chunk_meta(cu_seqlens, num_householder, chunk_size)
    if not meta["divisible"]:
        return f"CuTeDSL GDP: sequence lengths must be multiples of the kernel chunk size ({chunk_size})"
    if initial_state is not None and initial_state.shape[0] != meta["N"]:
        return (
            f"CuTeDSL GDP: initial_state has {initial_state.shape[0]} rows but "
            f"cu_seqlens describes {meta['N']} sequences"
        )
    return None


def _get_workspace(key: tuple, *shape_args, stream: cuda.CUstream) -> dict:
    """Get-or-create the cached device workspace and compiled kernel for a shape key.

    The workspace holds the buffers whose shapes depend only on the key (the
    interleaved and cumulative gates, the packed final-state buffer, the tile
    scheduler's per-sequence index tensors) together with their ``cute.Tensor``
    descriptors, so that steady-state calls allocate nothing and rebuild no
    descriptors.

    Args:
        key: Shape/feature-flag tuple identifying this workspace.
        shape_args: Unpacked problem dimensions used to size the buffers.
        stream: Stream the compilation is issued on.

    Returns:
        A dict of buffers, ``cute.Tensor`` descriptors and the compiled kernel.
    """
    raise NotImplementedError("_get_workspace: allocate buffers, build descriptors, compile")


def _get_compiled(
    io_dtype: Type[cutlass.Numeric],
    chunk_size: int,
    head_dim_k: int,
    head_dim_v: int,
    num_householder: int,
    has_initial: bool,
    output_final_state: bool,
    num_tokens: int,
    num_seqs: int,
    num_heads: int,
    *tensor_descriptors,
    stream: cuda.CUstream,
):
    """Compile (and cache) :class:`._gdp_kernel.GDPKernel` for one shape/feature key.

    Args:
        io_dtype: CuTe numeric type of ``q``/``k``/``v``.
        chunk_size: Kernel chunk length on the token timeline.
        head_dim_k: ``K``, the q/k head dimension (the SSM state dimension).
        head_dim_v: ``V``, the v head dimension.
        num_householder: ``M``.
        has_initial: Whether the state is seeded from ``initial_state``.
        output_final_state: Whether the final state is written out.
        tensor_descriptors: Placeholder ``cute.Tensor`` args matching the kernel
            signature, used only to specialize the compile.
        stream: Stream the compilation is issued on.

    Returns:
        The compiled kernel, callable with the real tensors.
    """
    global _MAX_ACTIVE_CLUSTERS
    if _MAX_ACTIVE_CLUSTERS is None:
        _MAX_ACTIVE_CLUSTERS = cutlass.utils.HardwareInfo().get_max_active_clusters(1)

    # The descriptors are built with no dynamic modes, so token count and
    # sequence count are baked into the compiled kernel as static shapes. They
    # MUST therefore be part of the cache key -- otherwise the first shape
    # compiled gets silently reused for every later one, which shows up as
    # multi-sequence batches producing wrong numbers while single-sequence ones
    # pass. (Marking dynamic modes instead would let one compile serve all
    # shapes; worth doing once the kernel is stable.)
    key = (
        io_dtype,
        chunk_size,
        head_dim_k,
        head_dim_v,
        num_householder,
        has_initial,
        output_final_state,
        num_tokens,
        num_seqs,
        num_heads,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        kernel = GDPKernel(
            io_dtype=io_dtype,
            gate_dtype=cutlass.Float32,
            acc_dtype=cutlass.Float32,
            chunk_size=chunk_size,
            head_dim_k=head_dim_k,
            head_dim_v=head_dim_v,
            num_householder=num_householder,
            has_initial=has_initial,
            output_final_state=output_final_state,
        )
        compiled = cute.compile(
            kernel,
            *tensor_descriptors,
            _MAX_ACTIVE_CLUSTERS,
            stream,
            options="--generate-line-info",
        )
        _COMPILE_CACHE[key] = compiled
    return compiled


def chunk_gated_delta_product_cutedsl(
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
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_size: int = KERNEL_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the chunked Gated Delta Product recurrence on the CuTe DSL kernel.

    Signature-compatible with ``fla.ops.gated_delta_product.chunk_gated_delta_product``
    so it can be swapped in at the mixer's call site. Callers must first check
    :func:`cutedsl_gdp_unsupported_reason` and fall back to FLA on a non-``None``
    result; this function assumes a supported batch and does not re-validate.

    Args:
        q: Queries, ``[B, T, H, K]``.
        k: Keys, ``[B, T * M, H, K]``.
        v: Values, ``[B, T * M, H, V]``.
        g: Log-space forget gate, ``[B, T, H]``, or ``None`` for the ungated variant.
        beta: Delta-rule step sizes, ``[B, T * M, H]``, already passed through sigmoid.
        num_householder: ``M``, the number of householder sub-steps per token.
        scale: Query scale. Defaults to ``K ** -0.5``.
        initial_state: Optional carried state, ``[N, H, K, V]``. ``N`` must equal
            ``len(cu_seqlens) - 1`` when ``cu_seqlens`` is given.
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: L2-normalize ``q`` and ``k`` in the preprocessing
            pass instead of requiring the caller to do it.
        cu_seqlens: Cumulative token counts, ``[N + 1]``, token units (never
            pre-multiplied by ``M``). Requires ``B == 1``.
        cu_seqlens_cpu: Optional CPU mirror of ``cu_seqlens``, used to build the
            scheduler index tensors without a device-to-host sync.
        chunk_size: Kernel chunk length on the token timeline.

    Returns:
        A tuple ``(o, final_state)`` where ``o`` has shape ``[B, T, H, V]`` and
        ``final_state`` has shape ``[N, H, K, V]`` (or is ``None`` when
        ``output_final_state`` is ``False``).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    M = num_householder
    if scale is None:
        scale = K**-0.5

    assert k.shape == (B, T * M, H, K), f"k must be on the delta-product timeline, got {k.shape}"
    assert v.shape == (B, T * M, H, V), f"v must be on the delta-product timeline, got {v.shape}"
    assert beta.shape == (
        B,
        T * M,
        H,
    ), f"beta must be on the delta-product timeline, got {beta.shape}"
    if g is not None:
        assert g.shape == (B, T, H), f"g must be on the token timeline, got {g.shape}"

    io_dtype = q.dtype
    cute_io_dtype = _torch_to_cute_dtype(io_dtype)
    stream = _current_cute_stream()

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    meta = _chunk_meta(cu_seqlens, M, chunk_size)

    # Build g_interleaved on the delta-product timeline and the two chunk-local
    # cumsums, in one fused launch. See _gdp_preprocess for the exact semantics.
    g_cumsum, g_interleaved_cumsum = fused_gdp_gate_preprocess(
        g, num_householder=M, chunk_size=chunk_size, cu_seqlens=cu_seqlens, meta=meta
    )

    device = q.device
    n_seq = meta["N"]
    has_initial = initial_state is not None

    out = torch.empty(B, T, H, V, device=device, dtype=io_dtype)
    final_state = torch.zeros(n_seq, H, K, V, device=device, dtype=torch.float32)
    # The kernel always takes an initial-state tensor so the compile type-checks;
    # when has_initial is False it is never read.
    init_state = (
        initial_state.to(torch.float32)
        if has_initial
        else torch.zeros(n_seq, H, K, V, device=device, dtype=torch.float32)
    )

    # The kernel indexes (T, H, D) directly -- drop the packed batch dim, which
    # cu_seqlens already forces to 1.
    descriptors = [
        _to_cute(q[0].contiguous(), []),
        _to_cute(k[0].contiguous(), []),
        _to_cute(v[0].contiguous(), []),
        _to_cute(beta[0].to(torch.float32).contiguous(), []),
        _to_cute(g_cumsum[0].contiguous(), []),
        _to_cute(g_interleaved_cumsum[0].contiguous(), []),
        _to_cute(out[0], []),
        _to_cute(final_state, []),
        _to_cute(init_state, []),
        _to_cute(meta["seq_start_tokens"], []),
        _to_cute(meta["seq_n_chunks"], []),
    ]

    # `scale` is a runtime argument of the kernel's __call__, so it has to appear
    # in the compile-time argument list too -- cute.compile type-checks the full
    # signature. Omitting it fails with CALL_MISSING_ARG naming `stream`, because
    # the trailing arguments shift by one.
    scale_arg = cutlass.Float32(scale)
    compiled = _get_compiled(
        cute_io_dtype,
        chunk_size,
        K,
        V,
        M,
        has_initial,
        output_final_state,
        T,
        n_seq,
        H,
        *descriptors,
        scale_arg,
        stream=stream,
    )
    compiled(*descriptors, scale_arg, stream)
    return out, (final_state.to(io_dtype) if output_final_state else None)
