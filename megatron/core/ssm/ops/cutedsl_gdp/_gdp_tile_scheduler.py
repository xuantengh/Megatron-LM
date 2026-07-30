# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# pylint: disable=missing-function-docstring
"""Persistent tile scheduler for the CuTe DSL Gated Delta Product kernel.

One work item is a ``(sequence, head)`` pair. The CTA that owns a work item walks
that sequence's chunks in order and keeps the recurrent state resident, so the
scheduler never needs to hand a partially-advanced state between CTAs. This
mirrors ``_mamba2_ssd_tile_scheduler.Mamba2SSDTileScheduler``; only the work-item
decomposition differs, because GDP has no group/head-ratio broadcast (the mixer
already expands GQA before the call).
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
    """Compile-time-shaped parameters describing the work-item space.

    Args:
        num_sequences: ``N``, the number of real (non-empty) sequences.
        num_heads: ``H``, the head count after GQA expansion.
        num_householder: ``M``, carried so the scheduler can convert a token-timeline
            chunk index into its delta-product sub-step range.
    """

    def __init__(
        self, num_sequences: int, num_heads: int, num_householder: int, *, loc=None, ip=None
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
        """Return the persistent grid shape, capped at the resident-CTA budget."""
        return (min(self.num_sequences * self.num_heads, max_active_clusters), 1, 1)


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
        values = extract_mlir_values(self.num_persistent_ctas)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        return values

    def __new_from_mlir_values__(self, values) -> "GDPTileScheduler":
        raise NotImplementedError("__new_from_mlir_values__: rebuild from flattened values")

    @staticmethod
    @dsl_user_op
    def create(
        params: GDPTileSchedulerParams, block_idx, grid_dim, *, loc=None, ip=None
    ) -> "GDPTileScheduler":
        """Create a scheduler instance for the calling CTA."""
        raise NotImplementedError("create: seed the linear work index from block_idx")

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        """Return the current ``(sequence, head)`` work item and its validity flag."""
        raise NotImplementedError("get_current_work: decode linear idx -> (seq, head)")

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None) -> None:
        """Stride the linear work index forward by the persistent CTA count."""
        raise NotImplementedError("advance_to_next_work")

    @property
    def num_tiles_executed(self) -> Int32:
        """Number of work items this CTA has completed."""
        return self._num_tiles_executed
