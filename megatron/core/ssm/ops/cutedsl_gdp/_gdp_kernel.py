# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# pylint: disable=line-too-long,missing-class-docstring,missing-function-docstring,too-many-statements
"""Fused Blackwell CuTe DSL kernel for the Gated Delta Product recurrence.

Fuses the six Triton kernels FLA launches (``chunk_scaled_dot_kkt_fwd``,
``solve_tril``, ``recompute_w_u_fwd``, ``chunk_gated_delta_product_fwd_h``,
``chunk_gated_delta_product_fwd_o``, plus the gate cumsums) into a single
persistent kernel. The win is HBM traffic, not MMA throughput: ``A``, ``Ai``,
``w``, ``u``, ``h`` and ``v_new`` are pure intermediates that FLA round-trips
through global memory (~1.1 GB at ``T=8192, H=32, K=V=128, M=3``) and that never
leave SMEM here, with the recurrent state ``S`` resident in registers/SMEM for a
whole sequence.

Math per token-chunk (``L`` tokens = ``L * M`` sub-steps = ``M`` WY sub-chunks of
``L`` sub-steps each). Stages 1-3 repeat once per sub-chunk; stage 4 runs once::

    1. A  = tril_strict(beta_i * (K K^T)_ij * 2^(gd_i - gd_j))      # delta-product
       Ai = (I + A)^-1                                             # blocked fwd substitution
    2. w  = Ai @ (K * beta * 2^gd);  u = Ai @ (V * beta)            # delta-product
    3. v_new = u - w @ S                                           # STORED UNGATED
       v_new_gated = v_new * 2^(gd_last - gd)
       S = S * 2^gd_last + K^T @ v_new_gated
       (S snapshotted as `h` at each token-chunk boundary)
    4. o = (Q @ h) * 2^gt                                          # token timeline
       o += sum_m [tril(Q @ K_m^T) * 2^(gt_i - gt_j)] @ v_new_m
       o *= scale

Two invariants that are easy to get wrong and produce plausible-looking numbers:

* **Stage 4 consumes the UNGATED ``v_new``.** FLA stores it before the
  ``2^(gd_last - gd)`` scaling; only the state update sees the gated copy.
  Applying the gate to the stored copy leaves ``final_state`` exactly right while
  the output is off by orders of magnitude.
* **Stage 4 indexes ``v_new`` by sub-step, not sub-chunk** — the stride-``M``
  gather ``v_new[m::M]``, matching FLA's ``i_dp*H*V`` offset with
  ``num_householder*H*V`` stride.

Base of the exponential: every decay is ``exp2``, never ``exp``. The cumulative
gates arrive pre-scaled by ``1/ln 2`` from :mod:`._gdp_preprocess`, matching
FLA >= 0.5.1. Mixing bases silently produces wrong numbers.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp

from ._gdp_tile_scheduler import GDPTileScheduler, GDPTileSchedulerParams


class GDPKernel:
    """One compiled specialization of the fused GDP forward kernel."""

    def __init__(
        self,
        io_dtype: type,
        gate_dtype: type,
        acc_dtype: type,
        chunk_size: int,
        head_dim_k: int,
        head_dim_v: int,
        num_householder: int,
        has_initial: bool = False,
        output_final_state: bool = False,
        num_warps: int = 4,
    ):
        """Fix the compile-time configuration.

        Args:
            io_dtype: Element type of ``q``/``k``/``v``/``o``. Half precision only.
            gate_dtype: Element type of the cumulative gates. Float32 only.
            acc_dtype: MMA accumulator type. Float32 only.
            chunk_size: ``L``, the chunk length on the token timeline.
            head_dim_k: ``K``, the q/k head dimension (the SSM state dimension).
            head_dim_v: ``V``, the v head dimension.
            num_householder: ``M``, the number of delta sub-steps per token.
            has_initial: Seed the recurrent state from ``initial_state``.
            output_final_state: Write the post-sequence state to ``final_state``.
            num_warps: Warps per CTA.
        """
        assert io_dtype in {cutlass.Float16, cutlass.BFloat16}, "io dtype must be fp16/bf16"
        assert acc_dtype in {Float32}, "acc dtype must be fp32"
        assert gate_dtype in {Float32}, "gate dtype must be fp32"

        self.io_dtype = io_dtype
        self.gate_dtype = gate_dtype
        self.acc_dtype = acc_dtype
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.num_householder = num_householder
        self.has_initial = has_initial
        self.output_final_state = output_final_state
        self.num_warps = num_warps
        self.num_threads = num_warps * 32

        # One token-chunk spans L * M sub-steps, i.e. exactly M WY sub-chunks of
        # L sub-steps. FLA relies on the same identity via
        # `if i_t % num_householder == 0` in chunk_deltaproduct_h.
        self.wy_subchunks_per_chunk = num_householder

        # MMA instruction and tiling. The WY stages work on ONE sub-chunk at a
        # time (L sub-steps, not L * M); only the readout sees a full token-chunk.
        self.mma_inst_mnk = (16, 8, 16)
        self.atom_layout_mnk = (num_warps, 1, 1)

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.shared_storage = None

    # ------------------------------------------------------------------
    # Host-side setup
    # ------------------------------------------------------------------
    def _setup_attributes(self):
        """Derive SMEM layouts and the shared-storage struct.

        Called at the top of :meth:`__call__` so every layout is a compile-time
        constant available to the TMA/MMA construction below.
        """
        L, DK, DV, M = self.chunk_size, self.head_dim_k, self.head_dim_v, self.num_householder

        # Operand tiles, one WY sub-chunk at a time.
        self.sK_layout = cute.make_ordered_layout((L, DK), order=(1, 0))
        self.sV_layout = cute.make_ordered_layout((L, DV), order=(1, 0))
        self.sQ_layout = cute.make_ordered_layout((L, DK), order=(1, 0))
        # A / Ai are L x L on the delta-product timeline.
        self.sA_layout = cute.make_ordered_layout((L, L), order=(1, 0))
        # w is L x DK, u/v_new are L x DV. v_new must persist for ALL M sub-chunks
        # because the readout gathers it with stride M, so it is sized L * M.
        self.sW_layout = cute.make_ordered_layout((L, DK), order=(1, 0))
        self.sVnew_layout = cute.make_ordered_layout((L * M, DV), order=(1, 0))
        # The recurrent state, fp32, resident across the whole sequence.
        self.sState_layout = cute.make_ordered_layout((DK, DV), order=(1, 0))
        # Cumulative gates: token timeline (L) and delta-product timeline (L * M).
        self.sGt_layout = cute.make_layout(L)
        self.sGd_layout = cute.make_layout(L * M)
        self.sBeta_layout = cute.make_layout(L * M)

        io, gate, acc = self.io_dtype, self.gate_dtype, self.acc_dtype

        @cute.struct
        class SharedStorage:
            sState: cute.struct.Align[cute.struct.MemRange[acc, cute.cosize(self.sState_layout)], 1024]
            # Snapshot of sState at the token-chunk boundary. Stage 4 reads the
            # state as it was BEFORE this chunk's M sub-chunk updates -- that is
            # what FLA materializes as `h` (stored at `i_t % M == 0`, i.e. at the
            # top of the sub-chunk loop, before the state advances).
            sH: cute.struct.Align[cute.struct.MemRange[acc, cute.cosize(self.sState_layout)], 1024]
            sVnew: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(self.sVnew_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(self.sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(self.sV_layout)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(self.sQ_layout)], 1024]
            sW: cute.struct.Align[cute.struct.MemRange[io, cute.cosize(self.sW_layout)], 1024]
            sA: cute.struct.Align[cute.struct.MemRange[acc, cute.cosize(self.sA_layout)], 1024]
            # fp32, NOT io_dtype. This is a monolithic L-row forward substitution
            # and the entries of (I+A)^-1 grow with row index; in bf16 the lower
            # rows overflow to inf. FLA sidesteps this entirely by inverting four
            # 16x16 blocks and merging (solve_tril), which is also faster -- worth
            # adopting once correctness is locked down.
            sAi: cute.struct.Align[cute.struct.MemRange[acc, cute.cosize(self.sA_layout)], 1024]
            sGt: cute.struct.Align[cute.struct.MemRange[gate, cute.cosize(self.sGt_layout)], 128]
            sGd: cute.struct.Align[cute.struct.MemRange[gate, cute.cosize(self.sGd_layout)], 128]
            sBeta: cute.struct.Align[cute.struct.MemRange[gate, cute.cosize(self.sBeta_layout)], 128]

        self.shared_storage = SharedStorage
        # `@cute.struct` exposes its own byte size; cute.size_in_bytes() is for
        # (dtype, layout) pairs, not structs.
        self.smem_bytes = SharedStorage.size_in_bytes()
        assert self.smem_bytes <= self.smem_capacity, (
            f"GDP kernel needs {self.smem_bytes} B of SMEM but sm_100 provides "
            f"{self.smem_capacity} B; reduce chunk_size or head dims"
        )

    def _make_tiled_mma(self):
        """Build the tiled MMA used by every stage.

        A single warp-level ``mma.sync`` tiling is reused across the four GEMM
        shapes; only the operand tensors differ. Returns the TiledMma.
        """
        op = warp.MmaF16BF16Op(self.io_dtype, self.acc_dtype, self.mma_inst_mnk)
        return cute.make_tiled_mma(op, cute.make_layout(self.atom_layout_mnk))

    # ------------------------------------------------------------------
    # JIT entry
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mBeta: cute.Tensor,
        mGt: cute.Tensor,
        mGd: cute.Tensor,
        mO: cute.Tensor,
        mFinalState: cute.Tensor,
        mInitialState: cute.Tensor,
        mSeqStart: cute.Tensor,
        mSeqChunks: cute.Tensor,
        scale: Float32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the persistent kernel.

        Args:
            mQ: Queries, token timeline, ``(T, H, DK)``.
            mK: Keys, delta-product timeline, ``(T * M, H, DK)``.
            mV: Values, delta-product timeline, ``(T * M, H, DV)``.
            mBeta: Step sizes, delta-product timeline, ``(T * M, H)``.
            mGt: Chunk-local cumulative gate, token timeline, ``(T, H)``.
            mGd: Chunk-local cumulative gate, delta-product timeline, ``(T * M, H)``.
            mO: Output, token timeline, ``(T, H, DV)``.
            mFinalState: Post-sequence state, ``(N, H, DK, DV)``.
            mInitialState: Seed state, same shape. Ignored unless ``has_initial``,
                but a valid tensor is still required so the compile type-checks.
            mSeqStart: Per-sequence first token index, ``(N + 1,)`` int32.
            mSeqChunks: Per-sequence token-chunk count, ``(N,)`` int32.
            scale: Query scale factor.
            max_active_clusters: Resident-CTA budget from ``HardwareInfo``.
            stream: Launch stream.
        """
        self._setup_attributes()
        tiled_mma = self._make_tiled_mma()

        num_heads = cute.size(mQ, mode=[1])
        num_seqs = cute.size(mSeqChunks, mode=[0])
        tile_sched_params = GDPTileSchedulerParams(
            Int32(num_seqs), Int32(num_heads), Int32(self.num_householder)
        )
        grid = tile_sched_params.get_grid_shape(max_active_clusters)

        self.kernel(
            mQ,
            mK,
            mV,
            mBeta,
            mGt,
            mGd,
            mO,
            mFinalState,
            mInitialState,
            mSeqStart,
            mSeqChunks,
            scale,
            tile_sched_params,
            tiled_mma,
            self.sK_layout,
            self.sV_layout,
            self.sQ_layout,
            self.sA_layout,
            self.sW_layout,
            self.sVnew_layout,
            self.sState_layout,
            self.sGt_layout,
            self.sGd_layout,
            self.sBeta_layout,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Device body
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mBeta: cute.Tensor,
        mGt: cute.Tensor,
        mGd: cute.Tensor,
        mO: cute.Tensor,
        mFinalState: cute.Tensor,
        mInitialState: cute.Tensor,
        mSeqStart: cute.Tensor,
        mSeqChunks: cute.Tensor,
        scale: Float32,
        tile_sched_params: GDPTileSchedulerParams,
        tiled_mma: cute.TiledMma,
        sK_layout: cute.Layout,
        sV_layout: cute.Layout,
        sQ_layout: cute.Layout,
        sA_layout: cute.Layout,
        sW_layout: cute.Layout,
        sVnew_layout: cute.Layout,
        sState_layout: cute.Layout,
        sGt_layout: cute.Layout,
        sGd_layout: cute.Layout,
        sBeta_layout: cute.Layout,
    ):
        """One persistent CTA per ``(sequence, head)``, walking that sequence's chunks."""
        tidx, _, _ = cute.arch.thread_idx()
        L = const_expr(self.chunk_size)
        DK = const_expr(self.head_dim_k)
        DV = const_expr(self.head_dim_v)
        M = const_expr(self.num_householder)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sState = storage.sState.get_tensor(sState_layout)
        sH = storage.sH.get_tensor(sState_layout)
        sVnew = storage.sVnew.get_tensor(sVnew_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sW = storage.sW.get_tensor(sW_layout)
        sA = storage.sA.get_tensor(sA_layout)
        sAi = storage.sAi.get_tensor(sA_layout)
        sGt = storage.sGt.get_tensor(sGt_layout)
        sGd = storage.sGd.get_tensor(sGd_layout)
        sBeta = storage.sBeta.get_tensor(sBeta_layout)

        tile_sched = GDPTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.get_current_work()

        while work_tile.is_valid_tile:
            seq_idx, head_idx, _ = work_tile.tile_idx
            tok_begin = mSeqStart[seq_idx]
            n_chunks = mSeqChunks[seq_idx]

            self._init_state(sState, mInitialState, seq_idx, head_idx, tidx, DK, DV)
            cute.arch.barrier()

            for chunk in cutlass.range(n_chunks, unroll=1):
                tok0 = tok_begin + chunk * L
                self._load_chunk_gates(sGt, sGd, sBeta, mGt, mGd, mBeta, tok0, head_idx, tidx, L, M)
                # Snapshot `h` before any sub-chunk advances the state.
                for i in cutlass.range(tidx, DK * DV, self.num_threads, unroll=1):
                    sH[i // DV, i % DV] = sState[i // DV, i % DV]
                cute.arch.barrier()

                # --- Stages 1-3, once per WY sub-chunk ----------------------
                for m in cutlass.range_constexpr(M):
                    sub0 = tok0 * M + m * L
                    self._load_kv(sK, sV, mK, mV, sub0, head_idx, tidx, L, DK, DV)
                    cute.arch.barrier()

                    self._stage_a_wy(sA, sAi, sK, sGd, sBeta, tiled_mma, tidx, m, L, DK)
                    cute.arch.barrier()

                    self._stage_b_state(
                        sState, sVnew, sW, sAi, sK, sV, sGd, sBeta, tiled_mma, tidx, m, L, DK, DV
                    )
                    cute.arch.barrier()

                # --- Stage 4, once per token-chunk --------------------------
                self._stage_c_readout(
                    mO, mQ, mK, sQ, sVnew, sH, sGt, tiled_mma, scale,
                    tok0, tok_begin, head_idx, tidx, L, DK, DV, M,
                )
                cute.arch.barrier()

            if const_expr(self.output_final_state):
                self._store_state(mFinalState, sState, seq_idx, head_idx, tidx, DK, DV)
                cute.arch.barrier()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    # ------------------------------------------------------------------
    # Stage helpers. Split out so each can be unit-tested against the torch
    # reference in isolation during bring-up.
    # ------------------------------------------------------------------
    @cute.jit
    def _init_state(self, sState, mInitialState, seq_idx, head_idx, tidx, DK, DV):
        """Zero the recurrent state, or seed it from ``initial_state``."""
        for i in cutlass.range(tidx, DK * DV, self.num_threads, unroll=1):
            r, c = i // DV, i % DV
            if const_expr(self.has_initial):
                sState[r, c] = mInitialState[seq_idx, head_idx, r, c].to(Float32)
            else:
                sState[r, c] = Float32(0.0)

    @cute.jit
    def _store_state(self, mFinalState, sState, seq_idx, head_idx, tidx, DK, DV):
        """Write the post-sequence state out."""
        for i in cutlass.range(tidx, DK * DV, self.num_threads, unroll=1):
            r, c = i // DV, i % DV
            mFinalState[seq_idx, head_idx, r, c] = sState[r, c].to(mFinalState.element_type)

    @cute.jit
    def _load_chunk_gates(self, sGt, sGd, sBeta, mGt, mGd, mBeta, tok0, head_idx, tidx, L, M):
        """Stage the token- and delta-product-timeline gates for one token-chunk."""
        for i in cutlass.range(tidx, L, self.num_threads, unroll=1):
            sGt[i] = mGt[tok0 + i, head_idx]
        for i in cutlass.range(tidx, L * M, self.num_threads, unroll=1):
            sGd[i] = mGd[tok0 * M + i, head_idx]
            sBeta[i] = mBeta[tok0 * M + i, head_idx].to(Float32)

    @cute.jit
    def _load_kv(self, sK, sV, mK, mV, sub0, head_idx, tidx, L, DK, DV):
        """Stage one WY sub-chunk's keys and values."""
        for i in cutlass.range(tidx, L * DK, self.num_threads, unroll=1):
            r, c = i // DK, i % DK
            sK[r, c] = mK[sub0 + r, head_idx, c]
        for i in cutlass.range(tidx, L * DV, self.num_threads, unroll=1):
            r, c = i // DV, i % DV
            sV[r, c] = mV[sub0 + r, head_idx, c]

    @cute.jit
    def _stage_a_wy(self, sA, sAi, sK, sGd, sBeta, tiled_mma, tidx, m, L, DK):
        """``A = tril_strict(beta * (K K^T) * 2^(gd_i - gd_j))`` then ``Ai = (I + A)^-1``.

        The inverse is a unit-lower-triangular forward substitution, which is
        inherently serial along rows; it is the one stage that is not a GEMM and
        the reason a dedicated warp group exists in the warp-specialized variant.
        """
        base = m * L
        # A = beta_i * (k_i . k_j) * 2^(gd_i - gd_j), strictly lower.
        for i in cutlass.range(tidx, L * L, self.num_threads, unroll=1):
            r, c = i // L, i % L
            acc = Float32(0.0)
            if r > c:
                for d in cutlass.range_constexpr(DK):
                    acc += sK[r, d].to(Float32) * sK[c, d].to(Float32)
                acc *= cute.arch.exp2(sGd[base + r] - sGd[base + c]) * sBeta[base + r]
            sA[r, c] = acc
        cute.arch.barrier()

        # Ai = (I + A)^-1 by forward substitution, one row at a time.
        # Row r of the inverse: Ai[r, :] = -A[r, :r] @ Ai[:r, :] with Ai[r, r] = 1.
        if tidx < L:
            diag0 = Float32(0.0)
            if tidx == 0:
                diag0 = Float32(1.0)
            sAi[0, tidx] = diag0.to(sAi.element_type)
        cute.arch.barrier()
        for r in cutlass.range(1, L, 1, unroll=1):
            if tidx < L:
                acc = Float32(0.0)
                for j in cutlass.range_constexpr(L):
                    if j < r:
                        acc -= sA[r, j] * sAi[j, tidx].to(Float32)
                if tidx == r:
                    acc += Float32(1.0)
                sAi[r, tidx] = acc.to(sAi.element_type)
            cute.arch.barrier()

    @cute.jit
    def _stage_b_state(
        self, sState, sVnew, sW, sAi, sK, sV, sGd, sBeta, tiled_mma, tidx, m, L, DK, DV
    ):
        """``w``/``u``, then ``v_new = u - w @ S`` and the gated state update."""
        base = m * L
        # w = Ai @ (k * beta * 2^gd)
        for i in cutlass.range(tidx, L * DK, self.num_threads, unroll=1):
            r, c = i // DK, i % DK
            acc = Float32(0.0)
            for j in cutlass.range_constexpr(L):
                if j <= r:
                    kb = sK[j, c].to(Float32) * sBeta[base + j] * cute.arch.exp2(sGd[base + j])
                    acc += sAi[r, j].to(Float32) * kb
            sW[r, c] = acc.to(sW.element_type)
        cute.arch.barrier()

        # u = Ai @ (v * beta), then v_new = u - w @ S. Stored UNGATED for stage 4.
        for i in cutlass.range(tidx, L * DV, self.num_threads, unroll=1):
            r, c = i // DV, i % DV
            u = Float32(0.0)
            for j in cutlass.range_constexpr(L):
                if j <= r:
                    u += sAi[r, j].to(Float32) * (sV[j, c].to(Float32) * sBeta[base + j])
            ws = Float32(0.0)
            for d in cutlass.range_constexpr(DK):
                ws += sW[r, d].to(Float32) * sState[d, c]
            sVnew[base + r, c] = (u - ws).to(sVnew.element_type)
        cute.arch.barrier()

        # S = S * 2^gd_last + K^T @ (v_new * 2^(gd_last - gd))
        gd_last = sGd[base + L - 1]
        for i in cutlass.range(tidx, DK * DV, self.num_threads, unroll=1):
            r, c = i // DV, i % DV
            acc = sState[r, c] * cute.arch.exp2(gd_last)
            for j in cutlass.range_constexpr(L):
                vg = sVnew[base + j, c].to(Float32) * cute.arch.exp2(gd_last - sGd[base + j])
                acc += sK[j, r].to(Float32) * vg
            sState[r, c] = acc

    @cute.jit
    def _stage_c_readout(
        self, mO, mQ, mK, sQ, sVnew, sH, sGt, tiled_mma, scale,
        tok0, tok_begin, head_idx, tidx, L, DK, DV, M,
    ):
        """``o = (Q @ h) * 2^gt + sum_m [tril(Q @ K_m^T) * decay] @ v_new_m``.

        ``sH`` is the snapshot of the state taken at the *start* of this
        token-chunk, not the live ``sState`` (which the M sub-chunks have already
        advanced). This mirrors FLA storing ``h`` at ``i_t % M == 0``.
        """
        for i in cutlass.range(tidx, L * DK, self.num_threads, unroll=1):
            r, c = i // DK, i % DK
            sQ[r, c] = mQ[tok0 + r, head_idx, c]
        cute.arch.barrier()

        for i in cutlass.range(tidx, L * DV, self.num_threads, unroll=1):
            r, c = i // DV, i % DV
            # Inter-chunk: q @ h, decayed to this token.
            acc = Float32(0.0)
            for d in cutlass.range_constexpr(DK):
                acc += sQ[r, d].to(Float32) * sH[d, c]
            acc *= cute.arch.exp2(sGt[r])
            # Intra-chunk: M separate causal products, each against the m-th
            # sub-step of every token (stride M, NOT the m-th sub-chunk).
            for m in cutlass.range_constexpr(M):
                for j in cutlass.range_constexpr(L):
                    if j <= r:
                        qk = Float32(0.0)
                        for d in cutlass.range_constexpr(DK):
                            qk += sQ[r, d].to(Float32) * mK[(tok0 + j) * M + m, head_idx, d].to(
                                Float32
                            )
                        acc += qk * cute.arch.exp2(sGt[r] - sGt[j]) * sVnew[j * M + m, c].to(
                            Float32
                        )
            mO[tok0 + r, head_idx, c] = (acc * scale).to(mO.element_type)
