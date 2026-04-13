# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for trace functionality with pl.range."""

import pytest

import pypto.language as pl


class TestTraceWithRange:
    """Test trace functionality with pl.range loops."""

    def test_trace_simple_range(self):
        """Test tracing a function with simple pl.range loop."""

        @pl.function
        def simple_range_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(5, init_values=(init,)):
                new_acc: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        trace_result = pl.trace(simple_range_func)

        print("=== test_trace_simple_range ===")
        print(str(trace_result))

        assert trace_result.function_name == "simple_range_func"
        assert trace_result.count() >= 1

    def test_trace_range_with_start_stop(self):
        """Test tracing a function with pl.range(start, stop)."""

        @pl.function
        def range_start_stop_func(
            x: pl.Tensor[[64, 128], pl.FP16]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(0, 10, init_values=(init,)):
                new_acc: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        trace_result = pl.trace(range_start_stop_func)

        print("=== test_trace_range_with_start_stop ===")
        print(str(trace_result))

        assert trace_result.function_name == "range_start_stop_func"
        assert trace_result.count() >= 1

    def test_trace_range_with_step(self):
        """Test tracing a function with pl.range(start, stop, step)."""

        @pl.function
        def range_step_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(0, 20, 2, init_values=(init,)):
                new_acc: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        trace_result = pl.trace(range_step_func)

        print("=== test_trace_range_with_step ===")
        print(str(trace_result))

        assert trace_result.function_name == "range_step_func"
        assert trace_result.count() >= 1

    def test_trace_nested_range(self):
        """Test tracing a function with nested pl.range loops."""

        @pl.function
        def nested_range_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            final_result: pl.Tensor[[64, 128], pl.FP32] = init
            for i, (outer,) in pl.range(3, init_values=(init,)):
                for j, (inner,) in pl.range(2, init_values=(outer,)):
                    new_inner: pl.Tensor[[64, 128], pl.FP32] = pl.add(inner, x)
                    final_result = pl.yield_(new_inner)
            return final_result

        trace_result = pl.trace(nested_range_func)

        print("=== test_trace_nested_range ===")
        print(str(trace_result))

        assert trace_result.function_name == "nested_range_func"
        assert trace_result.count() >= 1

    def test_trace_range_multiple_iter_args(self):
        """Test tracing a function with pl.range and multiple iter_args."""

        @pl.function
        def multiple_iter_args_func(
            x: pl.Tensor[[64, 128], pl.FP16]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            init1: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            init2: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc1, acc2) in pl.range(5, init_values=(init1, init2)):
                new_acc1: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc1, x)
                new_acc2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(acc2, x)
                result1, result2 = pl.yield_(new_acc1, new_acc2)
            return result1

        trace_result = pl.trace(multiple_iter_args_func)

        print("=== test_trace_range_multiple_iter_args ===")
        print(str(trace_result))

        assert trace_result.function_name == "multiple_iter_args_func"
        assert trace_result.count() >= 1

    def test_trace_range_operations_in_body(self):
        """Test tracing operations inside pl.range loop body."""

        @pl.function
        def range_ops_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(5, init_values=(init,)):
                temp1: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, temp1)
                temp3: pl.Tensor[[64, 128], pl.FP32] = pl.div(temp2, 2.0)
                result = pl.yield_(temp3)
            return result

        trace_result = pl.trace(range_ops_func)

        print("=== test_trace_range_operations_in_body ===")
        print(str(trace_result))

        assert trace_result.function_name == "range_ops_func"
        assert trace_result.count() >= 3

        op_names = [op.op_name for op in trace_result.operations]
        assert "tensor.add" in op_names
        assert "tensor.mul" in op_names
        assert "tensor.divs" in op_names

    def test_trace_range_filter_operations(self):
        """Test filtering operations in function with pl.range."""

        @pl.function
        def filter_range_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(5, init_values=(init,)):
                temp1: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, temp1)
                result = pl.yield_(temp2)
            return result

        trace_result = pl.trace(filter_range_func)

        print("=== test_trace_range_filter_operations ===")
        print(str(trace_result))

        add_ops = trace_result.filter_by_op("tensor.add")
        mul_ops = trace_result.filter_by_op("tensor.mul")

        assert len(add_ops) >= 1
        assert len(mul_ops) >= 1
        assert all(op.op_name == "tensor.add" for op in add_ops)
        assert all(op.op_name == "tensor.mul" for op in mul_ops)

    def test_trace_range_json_format(self):
        """Test JSON format output for function with pl.range."""

        @pl.function
        def json_range_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(3, init_values=(init,)):
                new_acc: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        trace_result = pl.trace(json_range_func)

        print("=== test_trace_range_json_format ===")
        print(str(trace_result))

        json_output = trace_result.to_json()

        import json

        data = json.loads(json_output)
        assert data["function_name"] == "json_range_func"
        assert "operations" in data
        assert len(data["operations"]) >= 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        pytest.main([__file__, "-v"])
    else:
        test_instance = TestTraceWithRange()
        print("Running test_trace_simple_range...")
        test_instance.test_trace_simple_range()
        print()

        print("Running test_trace_range_with_start_stop...")
        test_instance.test_trace_range_with_start_stop()
        print()

        print("Running test_trace_range_with_step...")
        test_instance.test_trace_range_with_step()
        print()

        print("Running test_trace_nested_range...")
        test_instance.test_trace_nested_range()
        print()

        print("Running test_trace_range_multiple_iter_args...")
        test_instance.test_trace_range_multiple_iter_args()
        print()

        print("Running test_trace_range_operations_in_body...")
        test_instance.test_trace_range_operations_in_body()
        print()

        print("Running test_trace_range_filter_operations...")
        test_instance.test_trace_range_filter_operations()
        print()

        print("Running test_trace_range_json_format...")
        test_instance.test_trace_range_json_format()
        print()

        print("All tests passed!")
