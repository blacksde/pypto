# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for grad_by_trace_v2 functionality."""

import pytest

import pypto.language as pl
from pypto.language.grad.auto_register import auto_register_gradients


class TestGradByTraceV2Basic:
    """Test basic grad_by_trace_v2 functionality."""

    @classmethod
    def setup_class(cls):
        """Setup test class - register gradients."""
        auto_register_gradients()

    def test_grad_simple_add(self):
        """Test gradient for simple add operation."""

        @pl.function
        def simple_add(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        grad_func = pl.grad_by_trace_v2(simple_add)

        assert grad_func is not None
        assert grad_func.name == "grad_simple_add"
        assert len(grad_func.params) >= 3  # x, result, dresult
        assert len(grad_func.return_types) == 1
        
        # Print as_python output
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(simple_add.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)
    
    def test_grad_simple_mul(self):
        """Test gradient for simple mul operation."""

        @pl.function
        def simple_mul(x: pl.Tensor[[64, 128], pl.FP32], y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.mul(x, y)
            return result

        grad_func = pl.grad_by_trace_v2(simple_mul)

        assert grad_func is not None
        assert grad_func.name == "grad_simple_mul"
        assert len(grad_func.params) >= 4  # x, y, result, dresult
        assert len(grad_func.return_types) == 2  # dx, dy
        
        # Print as_python output
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(simple_mul.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)

    def test_grad_chain_operations(self):
        """Test gradient for chain of operations."""
        
        @pl.function
        def chain_ops(x: pl.Tensor[[64, 128], pl.FP32], y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, y)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(temp, temp)
            return result

        grad_func = pl.grad_by_trace_v2(chain_ops)

        assert grad_func is not None
        assert grad_func.name == "grad_chain_ops"
        assert len(grad_func.params) >= 4
        assert len(grad_func.return_types) == 2
        
        # Print as_python output
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(chain_ops.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)

    def test_grad_multiple_inputs(self):
        """Test gradient for function with multiple inputs."""

        @pl.function
        def multi_inputs(
            x: pl.Tensor[[64, 128], pl.FP32],
            y: pl.Tensor[[64, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, y)
            return result

        grad_func = pl.grad_by_trace_v2(multi_inputs)

        assert grad_func is not None
        assert grad_func.name == "grad_multi_inputs"
        assert len(grad_func.params) >= 4  # x, y, result, dresult
        assert len(grad_func.return_types) == 2  # dx, dy
        
        # Print as_python output
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(multi_inputs.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)
    
    def test_grad_complex_add_mul(self):
        """Test gradient for complex mixed add and mul operations."""

        @pl.function
        def complex_add_mul(
            x: pl.Tensor[[64, 128], pl.FP32],
            y: pl.Tensor[[64, 128], pl.FP32],
            z: pl.Tensor[[64, 128], pl.FP32]
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            # t1 = x + y
            t1: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, y)
            # t2 = (x + y) * z
            t2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(t1, z)
            # t3 = (x + y) * z + (x + y), t1 被使用两次
            t3: pl.Tensor[[64, 128], pl.FP32] = pl.add(t2, t1)
            # result = ((x + y) * z + (x + y)) * x, x 被使用两次
            result: pl.Tensor[[64, 128], pl.FP32] = pl.mul(t3, x)
            return result

        grad_func = pl.grad_by_trace_v2(complex_add_mul)

        assert grad_func is not None
        assert grad_func.name == "grad_complex_add_mul"
        assert len(grad_func.params) >= 5  # x, y, z, result, dresult
        assert len(grad_func.return_types) == 3  # dx, dy, dz
        
        # Print as_python output
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(complex_add_mul.as_python())
        print()
        print("=" * 80)
        print("Gradient function as_python():")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)


class TestGradByTraceV2Errors:
    """Test error cases for grad_by_trace_v2."""

    @classmethod
    def setup_class(cls):
        """Setup test class - register gradients."""
        auto_register_gradients()

    def test_grad_non_function(self):
        """Test grad_by_trace_v2 with non-function input."""
        
        def regular_func(x):
            return x + x
        
        with pytest.raises(TypeError, match="Expected Function object"):
            pl.grad_by_trace_v2(regular_func)

    def test_grad_unregistered_op(self):
        """Test grad_by_trace_v2 with unregistered operation."""

        @pl.function
        def unregistered_op(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(x)
            return result

        auto_register_gradients()
        
        with pytest.raises(ValueError, match="Gradient functions not registered"):
            pl.grad_by_trace_v2(unregistered_op)


class TestGradByTraceV2Comparison:
    """Compare grad_by_trace_v2 with grad_by_trace."""

    @classmethod
    def setup_class(cls):
        """Setup test class - register gradients."""
        auto_register_gradients()

    def test_compare_simple_add(self):
        """Compare grad_by_trace_v2 and grad_by_trace for simple add."""

        @pl.function
        def simple_add(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        grad_func_v1 = pl.grad_by_trace(simple_add)
        grad_func_v2 = pl.grad_by_trace_v2(simple_add)

        assert grad_func_v1 is not None
        assert grad_func_v2 is not None
        assert grad_func_v1.name == grad_func_v2.name
        assert len(grad_func_v1.params) == len(grad_func_v2.params)
        
        # Print as_python output
        print("\n" + "=" * 80)
        print("Forward function as_python():")
        print("=" * 80)
        print(simple_add.as_python())
        print()
        print("=" * 80)
        print("Gradient function (grad_by_trace) as_python():")
        print("=" * 80)
        print(grad_func_v1.as_python())
        print()
        print("=" * 80)
        print("Gradient function (grad_by_trace_v2) as_python():")
        print("=" * 80)
        print(grad_func_v2.as_python())
        print("=" * 80)

    def test_compare_chain_ops(self):
        """Compare grad_by_trace_v2 and grad_by_trace for chain operations."""

        @pl.function
        def chain_ops(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            y: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            z: pl.Tensor[[64, 128], pl.FP32] = pl.mul(y, y)
            return z

        grad_func_v1 = pl.grad_by_trace(chain_ops)
        grad_func_v2 = pl.grad_by_trace_v2(chain_ops)

        assert grad_func_v1 is not None
        assert grad_func_v2 is not None
        assert grad_func_v1.name == grad_func_v2.name


def main():
    """Run all tests or specific test cases."""
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        print(f"Running test: {test_name}")
        print("=" * 80)
        
        # Map test names to test classes
        test_map = {
            'simple_add': TestGradByTraceV2Basic(),
            'simple_mul': TestGradByTraceV2Basic(),
            'chain_ops': TestGradByTraceV2Basic(),
            'multiple_inputs': TestGradByTraceV2Basic(),
            'complex_add_mul': TestGradByTraceV2Basic(),
            'compare_simple_add': TestGradByTraceV2Comparison(),
        }
        
        if test_name in test_map:
            test_obj = test_map[test_name]
            test_obj.setup_class()
            
            # Get the test method
            if test_name == 'simple_add':
                test_obj.test_grad_simple_add()
            elif test_name == 'simple_mul':
                test_obj.test_grad_simple_mul()
            elif test_name == 'chain_ops':
                test_obj.test_grad_chain_operations()
            elif test_name == 'multiple_inputs':
                test_obj.test_grad_multiple_inputs()
            elif test_name == 'complex_add_mul':
                test_obj.test_grad_complex_add_mul()
            elif test_name == 'compare_simple_add':
                test_obj.test_compare_simple_add()
            
            print("\n" + "=" * 80)
            print(f"✓ Test '{test_name}' PASSED")
            print("=" * 80)
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests:", ", ".join(test_map.keys()))
            sys.exit(1)
    else:
        # Run all passing tests
        print("=" * 80)
        print("Running all passing tests for grad_by_trace_v2")
        print("=" * 80)
        print()
        
        # Test 1: simple_add
        print("\n" + "-" * 80)
        print("Test 1: grad_simple_add")
        print("-" * 80)
        test_basic = TestGradByTraceV2Basic()
        test_basic.setup_class()
        test_basic.test_grad_simple_add()
        print("✓ PASSED")
        
        # Test 2: simple_mul
        print("\n" + "-" * 80)
        print("Test 2: grad_simple_mul")
        print("-" * 80)
        test_basic.test_grad_simple_mul()
        print("✓ PASSED")
        
        # Test 3: chain_ops
        print("\n" + "-" * 80)
        print("Test 3: grad_chain_operations")
        print("-" * 80)
        test_basic.test_grad_chain_operations()
        print("✓ PASSED")
        
        # Test 4: multiple_inputs
        print("\n" + "-" * 80)
        print("Test 4: grad_multiple_inputs")
        print("-" * 80)
        test_basic.test_grad_multiple_inputs()
        print("✓ PASSED")
        
        # Test 5: complex_add_mul
        print("\n" + "-" * 80)
        print("Test 5: grad_complex_add_mul")
        print("-" * 80)
        test_basic.test_grad_complex_add_mul()
        print("✓ PASSED")
        
        # Test 6: compare_simple_add
        print("\n" + "-" * 80)
        print("Test 6: compare_simple_add")
        print("-" * 80)
        test_compare = TestGradByTraceV2Comparison()
        test_compare.setup_class()
        test_compare.test_compare_simple_add()
        print("✓ PASSED")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)


if __name__ == "__main__":
    main()