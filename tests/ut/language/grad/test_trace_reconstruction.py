# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for trace reconstruction functionality."""

import pytest

import pypto.language as pl


class TestTraceReconstruction:
    """Test trace reconstruction functionality."""

    def test_reconstruct_simple_function(self):
        """Test reconstructing a simple function with one operation."""

        @pl.function
        def simple_add(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
            result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
            return result

        trace_result = pl.trace(simple_add)
        reconstructed_func = pl.reconstruct_from_trace(trace_result)

        assert reconstructed_func is not None
        assert reconstructed_func.name == "simple_add"
        assert len(reconstructed_func.params) == 1

    def test_reconstruct_multiple_operations(self):
        """Test reconstructing a function with multiple operations."""

        @pl.function
        def multi_op(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp1: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, temp1)
            temp3: pl.Tensor[[64, 128], pl.FP32] = pl.div(temp2, temp2)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(temp3)
            return result

        trace_result = pl.trace(multi_op)
        reconstructed_func = pl.reconstruct_from_trace(trace_result)

        assert reconstructed_func is not None
        assert reconstructed_func.name == "multi_op"

    def test_reconstruct_with_unary_ops(self):
        """Test reconstructing function with unary operations."""

        @pl.function
        def unary_ops(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp1: pl.Tensor[[64, 128], pl.FP32] = pl.exp(x)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(temp1)
            return result

        trace_result = pl.trace(unary_ops)
        reconstructed_func = pl.reconstruct_from_trace(trace_result)

        assert reconstructed_func is not None
        assert reconstructed_func.name == "unary_ops"

    def test_reconstruct_preserves_function_type(self):
        """Test that reconstruction preserves function type."""

        @pl.function
        def typed_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
            result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
            return result

        trace_result = pl.trace(typed_func)
        reconstructed_func = pl.reconstruct_from_trace(trace_result)

        assert reconstructed_func.func_type == typed_func.func_type

    def test_reconstruct_preserves_params(self):
        """Test that reconstruction preserves function parameters."""

        @pl.function
        def multi_param(
            x: pl.Tensor[[64, 128], pl.FP16],
            y: pl.Tensor[[64, 128], pl.FP16],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, y)
            return result

        trace_result = pl.trace(multi_param)
        reconstructed_func = pl.reconstruct_from_trace(trace_result)

        assert len(reconstructed_func.params) == 2
        assert reconstructed_func.params[0].name_hint == "x"
        assert reconstructed_func.params[1].name_hint == "y"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
