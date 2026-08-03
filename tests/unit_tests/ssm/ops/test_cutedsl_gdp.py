# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the CuTe DSL Gated Delta Product backend.

The suite is layered so a failure localizes itself:

1. ``TestReferences`` — the chunked WY reference against the sequential ground
   truth. Pure torch, no FLA, no CuTe. A failure here means the decomposition is
   wrong and everything downstream is meaningless.
2. ``TestFlaParity`` — FLA's Triton kernels against the same references. Pins the
   semantics the fused kernel must reproduce, and catches FLA version drift
   (0.5.1 moved every decay to base-2 ``exp2``).
3. ``TestCuteDslGdp`` — the fused CuTe kernel against FLA and the references.
"""

import pytest
import torch

from megatron.core.ssm.ops.cutedsl_gdp import (
    HAVE_CUTEDSL_GDP,
    chunk_gated_delta_product_cutedsl,
    cutedsl_gdp_unsupported_reason,
    is_cutedsl_gdp_available,
)
from megatron.core.ssm.ops.cutedsl_gdp._gdp_reference import gdp_chunked_reference, gdp_reference

try:
    from fla.ops.gated_delta_product import chunk_gated_delta_product

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


NUM_HOUSEHOLDER = 3
CHUNK = 64


def _make_inputs(seq_lens, num_heads=2, head_dim_k=32, head_dim_v=16, dtype=torch.bfloat16, seed=0):
    """Build a packed-THD GDP input set.

    Args:
        seq_lens: Per-sequence token counts.
        num_heads: ``H``.
        head_dim_k: ``K`` (the state dimension).
        head_dim_v: ``V`` (the value head dimension).
        dtype: io dtype for q/k/v/beta.
        seed: RNG seed.

    Returns:
        A dict of tensors plus ``cu_seqlens``, shaped for the FLA/CuTe contract.
    """
    torch.manual_seed(seed)
    device = "cuda"
    total = sum(seq_lens)
    m = NUM_HOUSEHOLDER
    q = torch.randn(1, total, num_heads, head_dim_k, device=device, dtype=dtype)
    k = torch.randn(1, total * m, num_heads, head_dim_k, device=device, dtype=dtype)
    v = torch.randn(1, total * m, num_heads, head_dim_v, device=device, dtype=dtype)
    beta = torch.rand(1, total * m, num_heads, device=device, dtype=dtype).sigmoid()
    # Gate must be negative (a decay) and not so large that 2^g underflows.
    g = -torch.rand(1, total, num_heads, device=device, dtype=torch.float32) * 0.5
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int64, device=device
    )
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "cu_seqlens": cu_seqlens}


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the max absolute error normalized by the reference's magnitude.

    Deliberately *not* an elementwise relative error. GDP outputs span several
    orders of magnitude within one tensor, so dividing elementwise makes a bf16
    rounding difference on a near-zero element (abs diff ~3e-4 on a ~1e-3 value)
    report as a 35% error while the tensor as a whole agrees to 3 digits.
    Normalizing by ``max|expected|`` is the metric that actually tracks whether
    two implementations of this kernel agree.
    """
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-12)).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestReferences:
    """The chunked WY decomposition must reproduce the sequential recurrence."""

    @pytest.mark.parametrize("seq_lens", [(64,), (128,), (64, 128), (192, 64)])
    def test_chunked_matches_sequential(self, seq_lens):
        x = _make_inputs(seq_lens)
        expected, expected_state = gdp_reference(
            **x,
            num_householder=NUM_HOUSEHOLDER,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        actual, actual_state = gdp_chunked_reference(
            **x,
            num_householder=NUM_HOUSEHOLDER,
            chunk_size=CHUNK,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert _rel_err(actual, expected) < 1e-4, f"output mismatch: {_rel_err(actual, expected)}"
        assert _rel_err(actual_state, expected_state) < 1e-4

    def test_initial_state_is_honoured(self):
        x = _make_inputs((64,))
        h0 = torch.randn(1, 2, 32, 16, device="cuda", dtype=torch.float32) * 0.1
        with_state, _ = gdp_chunked_reference(
            **x, num_householder=NUM_HOUSEHOLDER, initial_state=h0
        )
        without_state, _ = gdp_chunked_reference(**x, num_householder=NUM_HOUSEHOLDER)
        assert not torch.allclose(with_state.double(), without_state.double())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed")
class TestFlaParity:
    """FLA's Triton pipeline defines the semantics the fused kernel must match."""

    @pytest.mark.parametrize("seq_lens", [(64, 128), (192,)])
    def test_fla_matches_reference(self, seq_lens):
        x = _make_inputs(seq_lens)
        expected, _ = gdp_reference(
            **x, num_householder=NUM_HOUSEHOLDER, use_qk_l2norm_in_kernel=True
        )
        actual, _ = chunk_gated_delta_product(
            x["q"],
            x["k"],
            x["v"],
            g=x["g"],
            beta=x["beta"],
            num_householder=NUM_HOUSEHOLDER,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=x["cu_seqlens"],
        )
        # bf16 chunked accumulation. Measured ~2e-3 on GB200 with the
        # magnitude-normalized metric; 1e-2 leaves headroom without hiding a
        # genuine semantic divergence (a scale/convention bug shows up as O(1)).
        assert _rel_err(actual, expected) < 1e-2, f"FLA vs reference: {_rel_err(actual, expected)}"

    def test_gate_is_base_two(self):
        """FLA >= 0.5.1 scales the gate by 1/ln2 and uses exp2; guard against drift."""
        from fla.ops.utils.constant import RCP_LN2

        assert abs(RCP_LN2 - 1.4426950216) < 1e-9
        import fla.ops.gated_delta_product.chunk_deltaproduct_h as h_mod

        assert hasattr(h_mod, "exp2"), "FLA switched away from exp2; the CuTe kernel must follow"


@pytest.mark.skipif(not HAVE_CUTEDSL_GDP, reason="CuTe DSL is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestCuteDslGdp:
    """The fused CuTe DSL kernel."""

    def test_dispatch_guard_rejects_ragged(self):
        """Lengths that are not a multiple of the chunk size must fall back to FLA."""
        x = _make_inputs((100,))
        reason = cutedsl_gdp_unsupported_reason(
            x["q"], x["k"], x["v"], x["g"], x["beta"], NUM_HOUSEHOLDER, cu_seqlens=x["cu_seqlens"]
        )
        assert reason is not None and "multiples of the kernel chunk size" in reason

    def test_dispatch_guard_accepts_aligned(self):
        x = _make_inputs((64, 128), head_dim_k=128, head_dim_v=64)
        reason = cutedsl_gdp_unsupported_reason(
            x["q"], x["k"], x["v"], x["g"], x["beta"], NUM_HOUSEHOLDER, cu_seqlens=x["cu_seqlens"]
        )
        if is_cutedsl_gdp_available():
            assert reason is None, reason

    @pytest.mark.parametrize("seq_lens", [(64,), (64, 128)])
    def test_matches_reference(self, seq_lens):
        x = _make_inputs(seq_lens, head_dim_k=128, head_dim_v=64)
        if cutedsl_gdp_unsupported_reason(
            x["q"], x["k"], x["v"], x["g"], x["beta"], NUM_HOUSEHOLDER, cu_seqlens=x["cu_seqlens"]
        ):
            pytest.skip("batch not supported by the CuTe DSL backend")
        expected, _ = gdp_reference(
            **x, num_householder=NUM_HOUSEHOLDER, use_qk_l2norm_in_kernel=True
        )
        actual, _ = chunk_gated_delta_product_cutedsl(
            x["q"],
            x["k"],
            x["v"],
            x["g"],
            x["beta"],
            NUM_HOUSEHOLDER,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=x["cu_seqlens"],
        )
        assert _rel_err(actual, expected) < 2e-2
