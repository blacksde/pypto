# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may obtain this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test grad_by_trace_v2 with pl.range loops."""

import pytest

import pypto.language as pl
from pypto.language.grad.auto_register import auto_register_gradients


class TestGradByTraceV2WithRange:
    """Test grad_by_trace_v2 functionality with pl.range loops."""

    @classmethod
    def setup_class(cls):
        """Setup test class - register gradients."""
        auto_register_gradients()

    def test_grad_simple_range_sum(self):
        """Test gradient for simple sum using pl.range.
        
        Forward: result = x + x + x + x + x (5 iterations)
        Expected gradient: dresult/dx = 5 * doutput
        """

        @pl.function
        def range_sum(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(5, init_values=(init,)):
                new_acc: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                result = pl.yield_(new_acc)
            return result

        grad_func = pl.grad_by_trace_v2(range_sum)

        assert grad_func is not None
        assert grad_func.name == "grad_range_sum"
        
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(range_sum.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)

    def test_grad_range_with_mul(self):
        """Test gradient for pl.range with multiplication.
        
        Forward: result = x * 2 * 2 * 2 * 2 * 2 (5 iterations, multiply by 2 each time)
        Expected gradient: dresult/dx = 32 * doutput (2^5)
        """

        @pl.function
        def range_mul(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(5, init_values=(init,)):
                new_acc: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
                scaled: pl.Tensor[[64, 128], pl.FP32] = pl.mul(new_acc, 2.0)
                result = pl.yield_(scaled)
            return result

        grad_func = pl.grad_by_trace_v2(range_mul)

        assert grad_func is not None
        assert grad_func.name == "grad_range_mul"
        
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(range_mul.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)

    def test_grad_range_multiple_inputs(self):
        """Test gradient for pl.range with multiple inputs.
        
        Forward: result = (x + y) * 5 (5 iterations)
        Expected gradient: dresult/dx = 5 * doutput, dresult/dy = 5 * doutput
        """

        @pl.function
        def range_multi_input(
            x: pl.Tensor[[64, 128], pl.FP32],
            y: pl.Tensor[[64, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            for i, (acc,) in pl.range(5, init_values=(init,)):
                temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, y)
                new_acc: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, temp)
                result = pl.yield_(new_acc)
            return result

        grad_func = pl.grad_by_trace_v2(range_multi_input)

        assert grad_func is not None
        assert grad_func.name == "grad_range_multi_input"
        
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(range_multi_input.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)

    def test_grad_nested_range(self):
        """Test gradient for nested pl.range loops.
        
        Forward: outer loop 3 times, inner loop 2 times
        Total iterations: 3 * 2 = 6
        Expected gradient: dresult/dx = 6 * doutput
        """

        @pl.function
        def nested_range_sum(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            final_result: pl.Tensor[[64, 128], pl.FP32] = init
            for i, (outer,) in pl.range(3, init_values=(init,)):
                for j, (inner,) in pl.range(2, init_values=(outer,)):
                    new_inner: pl.Tensor[[64, 128], pl.FP32] = pl.add(inner, x)
                    final_result = pl.yield_(new_inner)
            return final_result

        grad_func = pl.grad_by_trace_v2(nested_range_sum)

        assert grad_func is not None
        assert grad_func.name == "grad_nested_range_sum"
        
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(nested_range_sum.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)


def main():
    """Run all tests or specific test cases."""
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        print(f"Running test: {test_name}")
        print("=" * 80)
        
        # Map test names to test methods
        test_obj = TestGradByTraceV2WithRange()
        test_obj.setup_class()
        
        if test_name == 'simple_range_sum':
            test_obj.test_grad_simple_range_sum()
        elif test_name == 'range_with_mul':
            test_obj.test_grad_range_with_mul()
        elif test_name == 'range_multiple_inputs':
            test_obj.test_grad_range_multiple_inputs()
        elif test_name == 'nested_range':
            test_obj.test_grad_nested_range()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: simple_range_sum, range_with_mul, range_multiple_inputs, nested_range")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print(f"✓ Test '{test_name}' PASSED")
        print("=" * 80)
    else:
        # Run all tests
        print("=" * 80)
        print("Running all tests for grad_by_trace_v2 with pl.range")
        print("=" * 80)
        print()
        
        test_obj = TestGradByTraceV2WithRange()
        test_obj.setup_class()
        
        # Test 1: simple_range_sum
        print("\n" + "-" * 80)
        print("Test 1: grad_simple_range_sum")
        print("-" * 80)
        test_obj.test_grad_simple_range_sum()
        print("✓ PASSED")
        
        # Test 2: range_with_mul
        print("\n" + "-" * 80)
        print("Test 2: grad_range_with_mul")
        print("-" * 80)
        test_obj.test_grad_range_with_mul()
        print("✓ PASSED")
        
        # Test 3: range_multiple_inputs
        print("\n" + "-" * 80)
        print("Test 3: grad_range_multiple_inputs")
        print("-" * 80)
        test_obj.test_grad_range_multiple_inputs()
        print("✓ PASSED")
        
        # Test 4: nested_range
        print("\n" + "-" * 80)
        print("Test 4: grad_nested_range")
        print("-" * 80)
        test_obj.test_grad_nested_range()
        print("✓ PASSED")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)


if __name__ == "__main__":
    main()