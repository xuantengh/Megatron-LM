# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional Triton kernels for DSA-GQA dynamic inference."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from triton.runtime.errors import OutOfResources as _TritonOutOfResources

    HAVE_TRITON = True
    _TRITON_RESOURCE_ERRORS = (_TritonOutOfResources,)
except ImportError:
    HAVE_TRITON = False
    _TRITON_RESOURCE_ERRORS = ()


if HAVE_TRITON:

    _SPARSE_ATTENTION_CONFIGS = [
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=4),
    ]

    @triton.jit
    def _thd_grouped_indexer_scores_kernel(
        q_ptr,
        k_ptr,
        query_lengths_ptr,
        kv_lengths_ptr,
        request_kv_offsets_ptr,
        query_offsets_ptr,
        kv_offsets_ptr,
        scores_ptr,
        max_query_length,
        max_kv_length,
        q_stride_t: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_t: tl.constexpr,
        k_stride_d: tl.constexpr,
        scores_stride_b: tl.constexpr,
        scores_stride_q: tl.constexpr,
        scores_stride_k: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute one request's score tile directly from packed THD Q and K."""
        request_idx = tl.program_id(0)
        query_positions = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        key_positions = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        query_length = tl.load(query_lengths_ptr + request_idx)
        kv_length = tl.load(kv_lengths_ptr + request_idx)
        request_kv_offset = tl.load(request_kv_offsets_ptr + request_idx)
        query_start = tl.load(query_offsets_ptr + request_idx)
        kv_start = tl.load(kv_offsets_ptr + request_idx)

        valid_queries = query_positions < query_length
        valid_keys = key_positions < kv_length
        valid_dims = dim_offsets < HEAD_DIM
        query = tl.load(
            q_ptr
            + (query_start + query_positions[:, None]) * q_stride_t
            + dim_offsets[None, :] * q_stride_d,
            mask=valid_queries[:, None] & valid_dims[None, :],
            other=0.0,
        )
        key = tl.load(
            k_ptr
            + (kv_start + key_positions[:, None]) * k_stride_t
            + dim_offsets[None, :] * k_stride_d,
            mask=valid_keys[:, None] & valid_dims[None, :],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key), input_precision="ieee", out_dtype=tl.float32)
        scores *= SCORE_SCALE

        causal = key_positions[None, :] <= (request_kv_offset + query_positions[:, None])
        valid_scores = valid_queries[:, None] & valid_keys[None, :] & causal
        scores = tl.where(valid_scores, scores, -float("inf"))
        # Match the PyTorch path's harmless padded-query row. Downstream top-k and
        # sparse softmax need one finite slot even though the row is discarded later.
        padded_query_sentinel = (~valid_queries[:, None]) & (key_positions[None, :] == 0)
        scores = tl.where(padded_query_sentinel, 0.0, scores)

        output_bounds = (query_positions[:, None] < max_query_length) & (
            key_positions[None, :] < max_kv_length
        )
        tl.store(
            scores_ptr
            + request_idx * scores_stride_b
            + query_positions[:, None] * scores_stride_q
            + key_positions[None, :] * scores_stride_k,
            scores,
            mask=output_bounds,
        )

    @triton.autotune(
        configs=_SPARSE_ATTENTION_CONFIGS, key=["topk", "HEAD_DIM", "VALUE_DIM", "VALUE_DTYPE"]
    )
    @triton.jit
    def _thd_sparse_attention_forward_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        topk_indices_ptr,
        query_request_indices_ptr,
        query_positions_ptr,
        kv_lengths_ptr,
        request_kv_offsets_ptr,
        kv_offsets_ptr,
        output_ptr,
        softmax_scale: tl.constexpr,
        topk: tl.constexpr,
        repeat_factor: tl.constexpr,
        q_stride_t: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_t: tl.constexpr,
        k_stride_g: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_t: tl.constexpr,
        v_stride_g: tl.constexpr,
        v_stride_d: tl.constexpr,
        ti_stride_b: tl.constexpr,
        ti_stride_q: tl.constexpr,
        ti_stride_k: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_K: tl.constexpr,
        VALUE_DTYPE: tl.constexpr,
        USE_DOT_MMA: tl.constexpr,
    ):
        """Attend one packed query/head directly to its request-local top-k K/V."""
        query_token_idx = tl.program_id(0)
        query_head_idx = tl.program_id(1)
        group_idx = query_head_idx // repeat_factor

        request_idx = tl.load(query_request_indices_ptr + query_token_idx)
        query_position = tl.load(query_positions_ptr + query_token_idx)

        kv_length = tl.load(kv_lengths_ptr + request_idx)
        request_kv_offset = tl.load(request_kv_offsets_ptr + request_idx)
        kv_start = tl.load(kv_offsets_ptr + request_idx)

        dim_offsets = tl.arange(0, BLOCK_D)
        value_dim_offsets = tl.arange(0, BLOCK_DV)
        dim_block = tl.broadcast_to(tl.expand_dims(dim_offsets, 0), (BLOCK_K, BLOCK_D))
        dim_mask = dim_block < HEAD_DIM
        query = tl.load(
            query_ptr
            + query_token_idx * q_stride_t
            + query_head_idx * q_stride_h
            + dim_offsets * q_stride_d,
            mask=dim_offsets < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        query_block = tl.broadcast_to(tl.expand_dims(query, 0), (BLOCK_K, BLOCK_D))

        running_max = tl.full((), -float("inf"), dtype=tl.float32)
        running_sum = tl.full((), 0.0, dtype=tl.float32)
        for support_start in range(0, topk, BLOCK_K):
            support_offsets = support_start + tl.arange(0, BLOCK_K)
            support_mask = support_offsets < topk
            selected_local = tl.load(
                topk_indices_ptr
                + request_idx * ti_stride_b
                + query_position * ti_stride_q
                + support_offsets * ti_stride_k,
                mask=support_mask,
                other=-1,
            )
            valid = (
                support_mask
                & (selected_local >= 0)
                & (selected_local < kv_length)
                & (selected_local <= request_kv_offset + query_position)
            )
            selected_global = kv_start + tl.where(valid, selected_local, 0)
            key = tl.load(
                key_ptr
                + selected_global[:, None] * k_stride_t
                + group_idx * k_stride_g
                + dim_block * k_stride_d,
                mask=valid[:, None] & dim_mask,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * query_block, axis=1) * softmax_scale
            scores = tl.where(valid, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            block_probs = tl.where(valid, tl.exp(scores - new_max), 0.0)
            running_sum = running_sum * old_scale + tl.sum(block_probs, axis=0)
            running_max = new_max

        output_accumulator = tl.zeros((BLOCK_DV,), dtype=tl.float32)
        for support_start in range(0, topk, BLOCK_K):
            support_offsets = support_start + tl.arange(0, BLOCK_K)
            support_mask = support_offsets < topk
            selected_local = tl.load(
                topk_indices_ptr
                + request_idx * ti_stride_b
                + query_position * ti_stride_q
                + support_offsets * ti_stride_k,
                mask=support_mask,
                other=-1,
            )
            valid = (
                support_mask
                & (selected_local >= 0)
                & (selected_local < kv_length)
                & (selected_local <= request_kv_offset + query_position)
            )
            selected_global = kv_start + tl.where(valid, selected_local, 0)
            key = tl.load(
                key_ptr
                + selected_global[:, None] * k_stride_t
                + group_idx * k_stride_g
                + dim_block * k_stride_d,
                mask=valid[:, None] & dim_mask,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * query_block, axis=1) * softmax_scale
            scores = tl.where(valid, scores, -float("inf"))
            probabilities = tl.where(valid, tl.exp(scores - running_max) / running_sum, 0.0)
            value = tl.load(
                value_ptr
                + selected_global[:, None] * v_stride_t
                + group_idx * v_stride_g
                + value_dim_offsets[None, :] * v_stride_d,
                mask=valid[:, None] & (value_dim_offsets[None, :] < VALUE_DIM),
                other=0.0,
            )
            if USE_DOT_MMA:
                if VALUE_DTYPE == 1:
                    probabilities_for_value = probabilities.to(tl.float16)
                elif VALUE_DTYPE == 2:
                    probabilities_for_value = probabilities.to(tl.bfloat16)
                else:
                    probabilities_for_value = probabilities
                dot_rows = tl.arange(0, 16)
                probabilities_for_dot = tl.where(
                    dot_rows[:, None] == 0, probabilities_for_value[None, :], 0.0
                )
                if VALUE_DTYPE == 1:
                    probabilities_for_dot = probabilities_for_dot.to(tl.float16)
                elif VALUE_DTYPE == 2:
                    probabilities_for_dot = probabilities_for_dot.to(tl.bfloat16)
                value_accumulator = tl.dot(probabilities_for_dot, value, out_dtype=tl.float32)
                output_accumulator += tl.sum(value_accumulator, axis=0)
            else:
                output_accumulator += tl.sum(probabilities[:, None] * value.to(tl.float32), axis=0)

        tl.store(
            output_ptr
            + query_token_idx * out_stride_t
            + query_head_idx * out_stride_h
            + value_dim_offsets * out_stride_d,
            output_accumulator,
            mask=value_dim_offsets < VALUE_DIM,
        )


def _supported_metadata(tensor: torch.Tensor, expected_numel: int) -> bool:
    return (
        tensor.is_cuda
        and tensor.dtype in (torch.int32, torch.int64)
        and tensor.dim() == 1
        and tensor.numel() == expected_numel
        and tensor.is_contiguous()
    )


def triton_thd_grouped_indexer_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    query_lengths: torch.Tensor,
    kv_lengths: torch.Tensor,
    request_kv_offsets: torch.Tensor,
    query_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_query_length: int,
    max_kv_length: int,
    softmax_scale: float,
) -> Optional[torch.Tensor]:
    """Compute causal simplified-indexer scores for a variable-length THD batch.

    Q and K remain packed as ``[total_q, head_dim]`` and ``[total_kv, head_dim]``.
    The output is padded only at the score boundary to ``[batch, max_q, max_kv]`` so
    the caller can run one batched ``torch.topk``. Unsupported inputs return ``None``
    and let the model use its PyTorch fallback.
    """
    if not HAVE_TRITON or torch.is_grad_enabled():
        return None
    if (
        not q.is_cuda
        or not k.is_cuda
        or q.device != k.device
        or q.dtype != k.dtype
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dim() != 2
        or k.dim() != 2
        or q.size(1) != k.size(1)
        or q.stride(1) != 1
        or k.stride(1) != 1
    ):
        return None

    batch_size = query_lengths.numel()
    metadata = (
        (query_lengths, batch_size),
        (kv_lengths, batch_size),
        (request_kv_offsets, batch_size),
        (query_offsets, batch_size + 1),
        (kv_offsets, batch_size + 1),
    )
    if any(
        not _supported_metadata(tensor, expected_numel) or tensor.device != q.device
        for tensor, expected_numel in metadata
    ):
        return None
    if batch_size == 0 or max_query_length == 0 or max_kv_length == 0:
        return torch.empty(
            (batch_size, max_query_length, max_kv_length), dtype=torch.float32, device=q.device
        )

    scores = torch.empty(
        (batch_size, max_query_length, max_kv_length), dtype=torch.float32, device=q.device
    )
    block_m = 16
    block_n = 32
    block_d = max(16, triton.next_power_of_2(q.size(1)))
    grid = (batch_size, triton.cdiv(max_query_length, block_m), triton.cdiv(max_kv_length, block_n))
    try:
        _thd_grouped_indexer_scores_kernel[grid](
            q,
            k,
            query_lengths,
            kv_lengths,
            request_kv_offsets,
            query_offsets,
            kv_offsets,
            scores,
            max_query_length,
            max_kv_length,
            *q.stride(),
            *k.stride(),
            *scores.stride(),
            SCORE_SCALE=float(softmax_scale),
            HEAD_DIM=q.size(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return scores


def _value_dtype_tag(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0


def _use_dot_mma(device: torch.device) -> bool:
    """Avoid the Triton 3.7 TMEM matvec lowering issue on Blackwell."""
    major, _ = torch.cuda.get_device_capability(device)
    return major < 10


def triton_thd_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    query_request_indices: torch.Tensor,
    query_positions: torch.Tensor,
    kv_lengths: torch.Tensor,
    request_kv_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    softmax_scale: float,
) -> Optional[torch.Tensor]:
    """Run simplified DSA-GQA attention over packed variable-length requests.

    ``topk_indices`` contains request-local KV indices. The Triton kernel converts
    them to packed global offsets and streams selected K/V directly, avoiding the
    padded and gathered K/V intermediates used by the PyTorch fallback.
    """
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if not HAVE_TRITON or torch.is_grad_enabled():
        return None
    if (
        not query.is_cuda
        or not key.is_cuda
        or not value.is_cuda
        or query.device != key.device
        or query.device != value.device
        or query.dtype != key.dtype
        or query.dtype not in supported_dtypes
        or value.dtype not in supported_dtypes
        or query.dim() != 3
        or key.dim() != 3
        or value.dim() != 3
        or key.shape[:2] != value.shape[:2]
        or query.size(2) != key.size(2)
        or query.stride(2) != 1
        or key.stride(2) != 1
        or value.stride(2) != 1
        or topk_indices.dim() != 3
        or topk_indices.dtype not in (torch.int32, torch.int64)
        or not topk_indices.is_cuda
        or topk_indices.device != query.device
        or topk_indices.stride(2) != 1
    ):
        return None

    total_query_tokens, num_query_heads, head_dim = query.shape
    num_query_groups = key.size(1)
    value_dim = value.size(2)
    topk = topk_indices.size(2)
    batch_size = kv_lengths.numel()
    if (
        num_query_groups == 0
        or num_query_heads % num_query_groups != 0
        or head_dim > 256
        or value_dim > 256
        or topk == 0
        or topk > 2048
        or topk_indices.size(0) != batch_size
    ):
        return None

    metadata = (
        (query_request_indices, total_query_tokens),
        (query_positions, total_query_tokens),
        (kv_lengths, batch_size),
        (request_kv_offsets, batch_size),
        (kv_offsets, batch_size + 1),
    )
    if any(
        not _supported_metadata(tensor, expected_numel) or tensor.device != query.device
        for tensor, expected_numel in metadata
    ):
        return None
    if total_query_tokens == 0:
        return value.new_empty((0, num_query_heads, value_dim))

    output = value.new_empty((total_query_tokens, num_query_heads, value_dim))
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_dv = max(32, triton.next_power_of_2(value_dim))
    grid = (total_query_tokens, num_query_heads)
    try:
        _thd_sparse_attention_forward_kernel[grid](
            query,
            key,
            value,
            topk_indices,
            query_request_indices,
            query_positions,
            kv_lengths,
            request_kv_offsets,
            kv_offsets,
            output,
            float(softmax_scale),
            topk,
            num_query_heads // num_query_groups,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *topk_indices.stride(),
            *output.stride(),
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
            VALUE_DTYPE=_value_dtype_tag(value.dtype),
            USE_DOT_MMA=_use_dot_mma(value.device),
        )
    except _TRITON_RESOURCE_ERRORS:
        return None
    return output


__all__ = ["HAVE_TRITON", "triton_thd_grouped_indexer_scores", "triton_thd_sparse_attention"]
