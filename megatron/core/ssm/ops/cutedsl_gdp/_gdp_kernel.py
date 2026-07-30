# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# pylint: disable=line-too-long,missing-class-docstring,missing-function-docstring,bad-builtin
"""Warp-specialized Blackwell CuTe DSL kernel for the Gated Delta Product recurrence.

Structure follows the CUTLASS ``CuTeDSL/cute/blackwell/kernel`` examples and the
sibling ``_mamba2_ssd_kernel_varlen.SSDKernel``: a plain Python class whose
``__init__`` fixes the compile-time configuration, ``_setup_attributes`` derives
the SMEM/TMEM layouts, ``@cute.jit __call__`` builds the TMA atoms and launches,
and ``@cute.kernel kernel`` is the device body.

Math implemented per chunk (chunk length ``L`` tokens, i.e. ``L * M`` sub-steps):

1. ``A = tril(diag(beta) @ K @ K^T * decay)`` on the delta-product timeline,
   then ``T = (I - A)^-1`` by blocked forward substitution (the WY transform).
2. ``w, u = T @ (beta * K), T @ (beta * V)`` — the corrected delta-rule operands.
3. State pass: ``S <- decay * S + K^T @ u - w^T @ (K @ S)``, applied ``M``
   sub-steps at a time, carrying ``S`` across chunks inside the CTA.
4. Readout: ``o = (Q @ S) * decay + tril(Q @ K^T * decay) @ u``, on the token
   timeline, using the non-interleaved cumulative gate.

Only stage 1 and 2 live on the delta-product timeline; the readout is one
query per token, which is why the tile shapes below differ between the intra
(``L x L``) and readout (``L x V``) MMAs.
"""

from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils

from ._gdp_tile_scheduler import GDPTileScheduler, GDPTileSchedulerParams


class GDPKernel:
    def __init__(
        self,
        io_dtype: Type[cutlass.Numeric],
        gate_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        chunk_size: int,
        head_dim_k: int,
        head_dim_v: int,
        num_householder: int,
        has_initial: bool = False,
        output_final_state: bool = False,
    ):
        """Fix the compile-time configuration of one kernel specialization.

        Args:
            io_dtype: Element type of ``q``/``k``/``v``/``o``. Half precision only.
            gate_dtype: Element type of the cumulative gates. Float32 only.
            acc_dtype: MMA accumulator type. Float32 only.
            chunk_size: ``L``, the chunk length on the token timeline.
            head_dim_k: ``K``, the q/k head dimension (the SSM state dimension).
            head_dim_v: ``V``, the v head dimension.
            num_householder: ``M``, the number of delta sub-steps per token.
            has_initial: Seed the recurrent state from ``initial_state`` instead of zero.
            output_final_state: Write the post-chunk state to ``final_state``.
        """
        self.io_dtype: Type[cutlass.Numeric] = io_dtype
        self.gate_dtype: Type[cutlass.Numeric] = gate_dtype
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype

        assert io_dtype in {cutlass.Float16, cutlass.BFloat16}, "Do not support other I/O types."
        assert acc_dtype in {cutlass.Float32}, "Do not support other ACC types."
        assert gate_dtype in {cutlass.Float32}, "Do not support other gate types."

        self.chunk_size: int = chunk_size
        self.head_dim_k: int = head_dim_k
        self.head_dim_v: int = head_dim_v
        self.num_householder: int = num_householder
        self.has_initial: bool = has_initial
        self.output_final_state: bool = output_final_state

        L, DK, DV, M = chunk_size, head_dim_k, head_dim_v, num_householder
        self.tile_shape = (L, DK, DV)
        # Sub-step tile: one chunk spans L * M rows on the delta-product timeline.
        self.tile_shape_dp = (L * M, DK, DV)

        # MMA tile shapes, one per stage of the chunk math above.
        # kkt / wy run on the delta-product timeline; qk / state / readout mix.
        self.tile_shape_mnk_kkt = (L * M, L * M, DK)  # stage 1: K @ K^T
        self.tile_shape_mnk_wy = (L * M, DV, L * M)  # stage 2: T @ (beta * V)
        self.tile_shape_mnk_state = (DK, DV, L * M)  # stage 3: K^T @ u
        self.tile_shape_mnk_readout = (L, DV, DK)  # stage 4: Q @ S

        # Hardcoded launch configuration, matching the SSD kernel's defaults.
        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.epi_tile = (128, 32)
        self.cta_group = cute.nvgpu.tcgen05.CtaGroup.ONE
        self.occupancy = 1

        # Warp specialization. One MMA warp per timeline, dedicated TMA warps for
        # the two operand groups, and softmax-free producer warps for the WY
        # transform (which is a serial triangular solve, not an MMA).
        self.mma_state_warp_id = 0
        self.mma_intra_warp_id = 1
        self.tma_kv_warp_id = 2
        self.tma_q_gate_warp_id = 3
        self.wy_warp_id = [4, 5, 6, 7]
        self.state_warp_id = [8, 9, 10, 11]
        self.epilog_warp_id = [12, 13, 14, 15]
        self.threads_per_cta = 32 * len(
            (
                self.mma_state_warp_id,
                self.mma_intra_warp_id,
                self.tma_kv_warp_id,
                self.tma_q_gate_warp_id,
                *self.wy_warp_id,
                *self.state_warp_id,
                *self.epilog_warp_id,
            )
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        # Named barriers.
        self.wy_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=len(self.wy_warp_id) * 32
        )
        self.state_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=len(self.state_warp_id) * 32
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=len(self.epilog_warp_id) * 32
        )
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4, num_threads=self.threads_per_cta
        )

        # Per-warp register budgets, tuned in _setup_attributes.
        self.num_regs_uniform_warps = 24
        self.num_regs_wy_warps = 192
        self.num_regs_state_warps = 168
        self.num_regs_epilogue_warps = 112

        # Filled in by _setup_attributes.
        self.shared_storage = None
        self.tmem_kkt_acc_offset = 0
        self.tmem_wy_acc_offset = 0
        self.tmem_state_offset = 0
        self.tmem_readout_acc_offset = 0
        self.num_tmem_cols_total = 0

    def _setup_attributes(self):
        """Derive SMEM layouts, pipeline stage counts and TMEM offsets.

        Called at the top of :meth:`__call__`, before any TMA atom is built, so
        that every layout is available as a compile-time constant.
        """
        raise NotImplementedError("_setup_attributes: build smem layouts, stages and tmem offsets")

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        beta: cute.Tensor,
        g_cumsum: cute.Tensor,
        g_interleaved_cumsum: cute.Tensor,
        o: cute.Tensor,
        final_state: cute.Tensor,
        initial_state: cute.Tensor,
        seq_chunk_start: cute.Tensor,
        seq_n_chunks: cute.Tensor,
        scale: cutlass.Float32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Build the TMA atoms and launch the persistent kernel.

        Args:
            q: Queries, token timeline, ``(DK, L, num_chunks, H, B)`` chunk-major.
            k: Keys, delta-product timeline, ``(DK, L * M, num_chunks, H, B)``.
            v: Values, delta-product timeline, ``(DV, L * M, num_chunks, H, B)``.
            beta: Step sizes, delta-product timeline, ``(L * M, num_chunks, H, B)``.
            g_cumsum: Chunk-local cumulative gate, token timeline.
            g_interleaved_cumsum: Chunk-local cumulative gate, delta-product timeline.
            o: Output, token timeline, same layout as ``q`` with ``DV`` in place of ``DK``.
            final_state: Post-chunk state, ``(DK, DV, H, N)``.
            initial_state: Seed state, same layout as ``final_state``. Ignored
                unless ``self.has_initial``; a placeholder descriptor is still
                required so the compile has a valid tensor argument.
            seq_chunk_start: Per-sequence first chunk index, ``(N,)`` int32.
            seq_n_chunks: Per-sequence chunk count, ``(N,)`` int32.
            scale: Query scale factor.
            max_active_clusters: Persistent-CTA budget from ``HardwareInfo``.
            stream: Launch stream.
        """
        self._setup_attributes()
        raise NotImplementedError(
            "__call__: make tiled MMAs, TMA atoms, tile scheduler params, then kernel(...).launch(...)"
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_beta: cute.CopyAtom,
        tma_tensor_beta: cute.Tensor,
        tma_atom_g: cute.CopyAtom,
        tma_tensor_g: cute.Tensor,
        tma_atom_g_dp: cute.CopyAtom,
        tma_tensor_g_dp: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        tma_atom_init: Optional[cute.CopyAtom],
        tma_tensor_init: cute.Tensor,
        final_state: cute.Tensor,
        seq_chunk_start: cute.Tensor,
        seq_n_chunks: cute.Tensor,
        scale: cutlass.Float32,
        tile_sched_params: GDPTileSchedulerParams,
        tiled_mma_kkt: cute.TiledMma,
        tiled_mma_wy: cute.TiledMma,
        tiled_mma_state: cute.TiledMma,
        tiled_mma_readout: cute.TiledMma,
    ):
        """Device body: one persistent CTA per ``(sequence, head)`` work item.

        Each CTA walks only its own sequence's chunks in order, keeping the
        ``DK x DV`` recurrent state resident in TMEM across the whole sequence so
        no inter-CTA state passing is needed. Warp roles are as assigned in
        ``__init__``; the WY warps run the serial triangular solve while the MMA
        warps overlap the next chunk's ``K @ K^T``.
        """
        raise NotImplementedError("kernel: warp-specialized chunk loop")

    @staticmethod
    def _compute_stages(smem_capacity: int) -> int:
        """Return the number of pipeline stages that fit in the SMEM budget."""
        raise NotImplementedError("_compute_stages")

    @staticmethod
    def _compute_grid(
        num_work_items: int, max_active_clusters: int
    ) -> tuple[GDPTileSchedulerParams, tuple[int, int, int]]:
        """Return the tile scheduler params and the persistent grid shape."""
        raise NotImplementedError("_compute_grid")

    def make_tiled_mmas(self):
        """Build the four tiled MMAs (kkt, wy, state, readout)."""
        raise NotImplementedError("make_tiled_mmas")

    @staticmethod
    def make_tile_scheduler(
        params: GDPTileSchedulerParams, block_idx, grid_dim
    ) -> GDPTileScheduler:
        """Instantiate the persistent ``(sequence, head)`` scheduler for this CTA."""
        return GDPTileScheduler.create(params, block_idx, grid_dim)
