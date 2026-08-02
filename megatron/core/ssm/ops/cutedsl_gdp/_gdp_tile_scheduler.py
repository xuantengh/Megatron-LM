# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# pylint: disable=missing-function-docstring
"""Persistent tile scheduler for the CuTe DSL Gated Delta Product kernel.

One work item is a ``(sequence, head)`` pair. The CTA that owns a work item walks
that sequence's token-chunks in order and keeps the ``K x V`` recurrent state
resident, so the scheduler never hands a partially-advanced state between CTAs.
That is the whole reason the decomposition is per-``(sequence, head)`` and not
per-chunk: the chunk axis is sequentially dependent and cannot be parallelized
without a second state-passing pass.

Mirrors ``_mamba2_ssd_tile_scheduler.Mamba2SSDTileScheduler``; the work-item
decomposition is simpler here because GDP has no group/head-ratio broadcast (the
mixer expands GQA before the call, so every head is independent).
"""

from typing import Tuple

from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Int32,
    Integer,
    dsl_user_op,
    extract_mlir_values,
    min,
    new_from_mlir_values,
)
from cutlass.utils import WorkTileInfo


class GDPTileSchedulerParams:
    """Parameters describing the ``(sequence, head)`` work-item space.

    Args:
        num_sequences: ``N``, the number of real (non-empty) sequences.
        num_heads: ``H``, the head count after GQA expansion.
        num_householder: ``M``, carried so the kernel can convert a token-chunk
            index into its delta-product sub-step range without a second tensor.
    """

    def __init__(
        self, num_sequences: Int32, num_heads: Int32, num_householder: Int32, *, loc=None, ip=None
    ):
        self.num_sequences = num_sequences
        self.num_heads = num_heads
        self.num_householder = num_householder
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.num_sequences, self.num_heads, self.num_householder]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.num_sequences, self.num_heads, self.num_householder], self._values_pos
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return GDPTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    @dsl_user_op
    def get_grid_shape(
        self, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Integer, Integer, Integer]:
        """Return the persistent grid shape, capped at the resident-CTA budget.

        Args:
            max_active_clusters: Resident-CTA budget from ``HardwareInfo``.

        Returns:
            A ``(x, 1, 1)`` grid. When there are fewer work items than resident
            CTAs the grid shrinks, so short packed batches do not launch idle
            CTAs that immediately exit.
        """
        return (
            min(self.num_sequences * self.num_heads, max_active_clusters, loc=loc, ip=ip),
            Int32(1),
            Int32(1),
        )


class GDPTileScheduler:
    """Strided persistent scheduler over ``(sequence, head)`` work items."""

    def __init__(
        self,
        params: GDPTileSchedulerParams,
        num_persistent_ctas: Int32,
        current_work_linear_idx: Int32,
        num_tiles_executed: Int32,
    ):
        self.params = params
        self.num_persistent_ctas = num_persistent_ctas
        self._current_work_linear_idx = current_work_linear_idx
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.params)
        values.extend(extract_mlir_values(self.num_persistent_ctas))
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        return values

    def __new_from_mlir_values__(self, values) -> "GDPTileScheduler":
        n_params = len(extract_mlir_values(self.params))
        params = new_from_mlir_values(self.params, values[:n_params])
        rest = values[n_params:]
        return GDPTileScheduler(
            params,
            new_from_mlir_values(self.num_persistent_ctas, [rest[0]]),
            new_from_mlir_values(self._current_work_linear_idx, [rest[1]]),
            new_from_mlir_values(self._num_tiles_executed, [rest[2]]),
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: GDPTileSchedulerParams, block_idx, grid_dim, *, loc=None, ip=None
    ) -> "GDPTileScheduler":
        """Create a scheduler instance for the calling CTA.

        Args:
            params: The work-item space description.
            block_idx: ``cute.arch.block_idx()`` triple.
            grid_dim: ``cute.arch.grid_dim()`` triple.

        Returns:
            A scheduler seeded so that CTA ``i`` starts on work item ``i`` and
            then strides by the persistent CTA count.
        """
        bidx, _, _ = block_idx
        gdim_x, _, _ = grid_dim
        return GDPTileScheduler(params, Int32(gdim_x), Int32(bidx), Int32(0))

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        """Return the current ``(sequence, head, 0)`` work item and its validity.

        Returns:
            A ``WorkTileInfo`` whose tile index is ``(seq, head, 0)``. ``is_valid``
            is ``False`` once this CTA has strided past the last work item, which
            is how the persistent loop terminates.
        """
        num_items = self.params.num_sequences * self.params.num_heads
        is_valid = self._current_work_linear_idx < num_items
        # Head-major: consecutive linear indices hit consecutive heads of the same
        # sequence, so co-scheduled CTAs read overlapping q/k/v cache lines.
        seq_idx = self._current_work_linear_idx // self.params.num_heads
        head_idx = self._current_work_linear_idx % self.params.num_heads
        return WorkTileInfo((Int32(seq_idx), Int32(head_idx), Int32(0)), is_valid)

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None) -> None:
        """Stride the linear work index forward by the persistent CTA count.

        Args:
            advance_count: Number of work items to skip, normally 1.
        """
        self._current_work_linear_idx += Int32(advance_count) * self.num_persistent_ctas
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self) -> Int32:
        """Number of work items this CTA has completed."""
        return self._num_tiles_executed


def compute_grid(
    num_sequences: int, num_heads: int, num_householder: int, max_active_clusters: int
) -> tuple[GDPTileSchedulerParams, tuple[int, int, int]]:
    """Build the scheduler params and persistent grid shape on the host.

    Args:
        num_sequences: ``N``.
        num_heads: ``H``.
        num_householder: ``M``.
        max_active_clusters: Resident-CTA budget from ``HardwareInfo``.

    Returns:
        ``(params, grid)``. The grid is capped at the work-item count so a short
        packed batch does not launch CTAs that exit immediately.
    """
    params = GDPTileSchedulerParams(
        Int32(num_sequences), Int32(num_heads), Int32(num_householder)
    )
    # NOTE: `min` at module scope is cutlass's DSL min (imported above for
    # get_grid_shape), which builds IR rather than comparing Python ints. This is
    # host code, so compare with a plain conditional instead.
    work_items = num_sequences * num_heads
    grid_x = work_items if work_items < max_active_clusters else max_active_clusters
    return params, (grid_x, 1, 1)
