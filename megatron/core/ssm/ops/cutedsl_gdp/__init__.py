# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CuTe DSL Blackwell kernel for the Gated Delta Product (GDP) chunked recurrence.

This package mirrors the layout of the CUTLASS ``CuTeDSL/cute/blackwell/kernel``
examples and of the sibling :mod:`megatron.core.ssm.ops.cutedsl_mamba2_ssd`
package:

==========================  ====================================================
``gdp_cutedsl.py``          Host front-end: dispatch guard, workspace/compile
                            caches, dlpack -> ``cute.Tensor`` conversion, and the
                            public entry point that mirrors FLA's
                            ``chunk_gated_delta_product`` signature.
``_gdp_kernel.py``          ``GDPKernel``: the warp-specialized device kernel
                            (``@cute.jit __call__`` + ``@cute.kernel kernel``).
``_gdp_tile_scheduler.py``  Persistent per-``(sequence, head)`` tile scheduler.
``_gdp_preprocess.py``      Triton pre-passes the CuTe kernel does not perform
                            (householder interleaving of ``g`` and the per-chunk
                            local cumsums).
``_gdp_reference.py``       Torch reference implementation used by tests.
==========================  ====================================================

The CuTe DSL runtime (``cutlass``, ``cuda.bindings``) is an optional dependency,
so importing this package never hard-fails: on a system without it,
:data:`HAVE_CUTEDSL_GDP` is ``False`` and the public symbols are ``None``.
Callers must gate on :func:`is_cutedsl_gdp_available` before use.
"""

try:
    from .gdp_cutedsl import (
        chunk_gated_delta_product_cutedsl,
        cutedsl_gdp_unsupported_reason,
        is_cutedsl_gdp_available,
    )

    HAVE_CUTEDSL_GDP = True
except ImportError:
    chunk_gated_delta_product_cutedsl = None
    cutedsl_gdp_unsupported_reason = None
    HAVE_CUTEDSL_GDP = False

    def is_cutedsl_gdp_available() -> bool:
        """Return ``False`` because the CuTe DSL runtime is not importable."""
        return False


__all__ = [
    "HAVE_CUTEDSL_GDP",
    "chunk_gated_delta_product_cutedsl",
    "cutedsl_gdp_unsupported_reason",
    "is_cutedsl_gdp_available",
]
