# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for trace functionality."""

import json

import pytest

import pypto.language as pl


class TestTraceBasicFunctionality:
    """Test basic trace functionality."""

    def test_trace_simple_function(self):
        """Test tracing a simple function with one operation."""

        @pl.function
        def simple_add(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
            result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
            return result

        trace_result = pl.trace(simple_add)

        assert trace_result.function_name == "simple_add"
        assert trace_result.count() == 1
        assert len(trace_result.operations) == 1

        op = trace_result.operations[0]
        assert op.op_name == "tensor.add"
        assert op.op_type == "tensor"
        assert len(op.args) == 2
        assert op.index == 0

    def test_trace_multiple_operations(self):
        """Test tracing a function with multiple operations."""

        @pl.function
        def multi_op(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp, temp)
            return result

        trace_result = pl.trace(multi_op)

        assert trace_result.count() == 2
        assert trace_result.operations[0].op_name == "add"
        assert trace_result.operations[1].op_name == "mul"
        assert trace_result.operations[0].index == 0
        assert trace_result.operations[1].index == 1

    def test_trace_function_info(self):
        """Test that function metadata is captured correctly."""

        @pl.function
        def traced_func(x: pl.Tensor[[32, 64], pl.FP16], y: pl.Tensor[[32, 64], pl.FP16]) -> pl.Tensor[[32, 64], pl.FP32]:
            result: pl.Tensor[[32, 64], pl.FP32] = pl.add(x, y)
            return result

        trace_result = pl.trace(traced_func)

        assert trace_result.function_name == "traced_func"
        assert len(trace_result.param_info) == 2
        assert "x" in trace_result.param_info[0]
        assert "y" in trace_result.param_info[1]
        assert len(trace_result.return_types) >= 1


class TestTraceFiltering:
    """Test trace filtering functionality."""

    def test_filter_by_op(self):
        """Test filtering operations by name."""

        @pl.function
        def filter_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp1: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, temp1)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(temp2, temp2)
            return result

        trace_result = pl.trace(filter_test)

        add_ops = trace_result.filter_by_op("add")
        mul_ops = trace_result.filter_by_op("mul")

        assert len(add_ops) == 2
        assert len(mul_ops) == 1
        assert all(op.op_name == "add" for op in add_ops)
        assert all(op.op_name == "mul" for op in mul_ops)

    def test_filter_by_type(self):
        """Test filtering operations by type."""

        @pl.function
        def type_filter_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(type_filter_test)

        tensor_ops = trace_result.filter_by_type("tensor")
        tile_ops = trace_result.filter_by_type("tile")

        assert len(tensor_ops) >= 1
        assert len(tile_ops) == 0

    def test_get_operation(self):
        """Test getting operations by index."""

        @pl.function
        def index_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp, temp)
            return result

        trace_result = pl.trace(index_test)

        op0 = trace_result.get_operation(0)
        op1 = trace_result.get_operation(1)
        op_invalid = trace_result.get_operation(10)

        assert op0 is not None
        assert op1 is not None
        assert op_invalid is None
        assert op0.op_name == "add"
        assert op1.op_name == "mul"


class TestTraceOutputFormats:
    """Test trace output formatting."""

    def test_text_format(self):
        """Test text format output."""

        @pl.function
        def text_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(text_test)
        text_output = str(trace_result)

        assert "Trace for function 'text_test'" in text_output
        assert "add" in text_output
        assert "Operations (1 total)" in text_output
        assert "Type: tensor" in text_output

    def test_json_format(self):
        """Test JSON format output."""

        @pl.function
        def json_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(json_test)
        json_output = trace_result.to_json()

        data = json.loads(json_output)
        assert data["function_name"] == "json_test"
        assert "operations" in data
        assert len(data["operations"]) == 1
        assert data["operations"][0]["op_name"] == "add"

    def test_detailed_format(self):
        """Test detailed format output."""

        @pl.function
        def detailed_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(detailed_test)

        from pypto.language.grad.trace import _format_detailed

        detailed_output = _format_detailed(trace_result)

        assert "DETAILED TRACE FOR FUNCTION" in detailed_output
        assert "detailed_test" in detailed_output
        assert "PARAMETERS:" in detailed_output
        assert "OPERATIONS:" in detailed_output
        assert "TRACE SUMMARY" in detailed_output

    def test_print_trace_console(self, capsys):
        """Test printing trace to console."""

        @pl.function
        def print_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(print_test)
        pl.print_trace(trace_result, format="text", output="console")

        captured = capsys.readouterr()
        assert "Trace for function 'print_test'" in captured.out
        assert "add" in captured.out


class TestTraceErrorHandling:
    """Test trace error handling."""

    def test_trace_non_function(self):
        """Test that tracing non-function objects raises error."""

        with pytest.raises(TypeError):
            pl.trace("not a function")

        with pytest.raises(TypeError):
            pl.trace(123)

    def test_print_trace_invalid_format(self):
        """Test that invalid format raises error."""

        @pl.function
        def format_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(format_test)

        with pytest.raises(ValueError):
            pl.print_trace(trace_result, format="invalid")

    def test_print_trace_file_without_filename(self):
        """Test that file output without filename raises error."""

        @pl.function
        def file_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(file_test)

        with pytest.raises(ValueError):
            pl.print_trace(trace_result, output="file")


class TestTraceComplexFunctions:
    """Test tracing more complex functions."""

    def test_trace_with_constants(self):
        """Test tracing function with constant arguments."""

        @pl.function
        def const_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(const_test)

        assert trace_result.count() >= 1

    def test_trace_math_operations(self):
        """Test tracing various math operations."""

        @pl.function
        def math_test(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp1: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, temp1)
            temp3: pl.Tensor[[64, 128], pl.FP32] = pl.div(temp2, temp2)
            temp4: pl.Tensor[[64, 128], pl.FP32] = pl.exp(temp3)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(temp4)
            return result

        trace_result = pl.trace(math_test)

        assert trace_result.count() == 5
        op_names = [op.op_name for op in trace_result.operations]
        assert "add" in op_names
        assert "mul" in op_names
        assert "div" in op_names
        assert "exp" in op_names
        assert "sqrt" in op_names


class TestTraceEdgeCases:
    """Test trace edge cases."""

    def test_trace_empty_function(self):
        """Test tracing function with minimal operations."""

        @pl.function
        def empty_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
            return x

        trace_result = pl.trace(empty_test)

        # Function might have no operations or just return
        assert trace_result.function_name == "empty_test"

    def test_trace_single_param_function(self):
        """Test tracing function with single parameter."""

        @pl.function
        def single_param(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(single_param)

        assert len(trace_result.param_info) == 1
        assert "x" in trace_result.param_info[0]

    def test_trace_multiple_params_function(self):
        """Test tracing function with multiple parameters."""

        @pl.function
        def multi_param(
            x: pl.Tensor[[64, 128], pl.FP16],
            y: pl.Tensor[[64, 128], pl.FP16],
            z: pl.Tensor[[64, 128], pl.FP16],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, y)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(temp, z)
            return result

        trace_result = pl.trace(multi_param)

        assert len(trace_result.param_info) == 3
        assert trace_result.count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])