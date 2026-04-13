# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for gradient registrations in auto_register.py.

This file tests each gradient function registered in auto_register.py
to ensure they are properly registered and can generate gradient functions.
"""

import pytest

import pypto.language as pl
from pypto.language.grad.auto_register import auto_register_gradients
from pypto.ir.grad_registry import GradRegistry


class TestTensorOpGradients:
    """Test gradient registrations for tensor operations."""

    @classmethod
    def setup_class(cls):
        """Setup test class - register gradients."""
        auto_register_gradients()
        cls.registry = GradRegistry.get_instance()

    def test_tensor_add_gradient_registered(self):
        """Test that tensor.add gradient is registered."""
        assert self.registry.has_grad("tensor.add")
        grad_func = self.registry.get_grad("tensor.add")
        assert grad_func is not None

    def test_tensor_sub_gradient_registered(self):
        """Test that tensor.sub gradient is registered."""
        assert self.registry.has_grad("tensor.sub")
        grad_func = self.registry.get_grad("tensor.sub")
        assert grad_func is not None

    def test_tensor_mul_gradient_registered(self):
        """Test that tensor.mul gradient is registered."""
        assert self.registry.has_grad("tensor.mul")
        grad_func = self.registry.get_grad("tensor.mul")
        assert grad_func is not None

    def test_tensor_div_gradient_registered(self):
        """Test that tensor.div gradient is registered."""
        assert self.registry.has_grad("tensor.div")
        grad_func = self.registry.get_grad("tensor.div")
        assert grad_func is not None

    def test_tensor_neg_gradient_registered(self):
        """Test that tensor.neg gradient is registered."""
        assert self.registry.has_grad("tensor.neg")
        grad_func = self.registry.get_grad("tensor.neg")
        assert grad_func is not None

    def test_tensor_matmul_gradient_registered(self):
        """Test that tensor.matmul gradient is registered."""
        assert self.registry.has_grad("tensor.matmul")
        grad_func = self.registry.get_grad("tensor.matmul")
        assert grad_func is not None

    def test_tensor_tensor_transpose_gradient_registered(self):
        """Test that tensor.transpose gradient is registered."""
        assert self.registry.has_grad("tensor.transpose")
        grad_func = self.registry.get_grad("tensor.transpose")
        assert grad_func is not None

    def test_tensor_mean_gradient_registered(self):
        """Test that tensor.mean gradient is registered."""
        assert self.registry.has_grad("tensor.mean")
        grad_func = self.registry.get_grad("tensor.mean")
        assert grad_func is not None

    def test_tensor_relu_gradient_registered(self):
        """Test that tensor.relu gradient is registered."""
        assert self.registry.has_grad("tensor.relu")
        grad_func = self.registry.get_grad("tensor.relu")
        assert grad_func is not None

    def test_tensor_sigmoid_gradient_registered(self):
        """Test that tensor.sigmoid gradient is registered."""
        assert self.registry.has_grad("tensor.sigmoid")
        grad_func = self.registry.get_grad("tensor.sigmoid")
        assert grad_func is not None

    def test_tensor_tanh_gradient_registered(self):
        """Test that tensor.tanh gradient is registered."""
        assert self.registry.has_grad("tensor.tanh")
        grad_func = self.registry.get_grad("tensor.tanh")
        assert grad_func is not None

    def test_tensor_sqrt_gradient_registered(self):
        """Test that tensor.sqrt gradient is registered."""
        assert self.registry.has_grad("tensor.sqrt")
        grad_func = self.registry.get_grad("tensor.sqrt")
        assert grad_func is not None

    def test_tensor_exp_gradient_registered(self):
        """Test that tensor.exp gradient is registered."""
        assert self.registry.has_grad("tensor.exp")
        grad_func = self.registry.get_grad("tensor.exp")
        assert grad_func is not None

    def test_tensor_maximum_gradient_registered(self):
        """Test that tensor.maximum gradient is registered."""
        assert self.registry.has_grad("tensor.maximum")
        grad_func = self.registry.get_grad("tensor.maximum")
        assert grad_func is not None

    def test_tensor_muls_gradient_registered(self):
        """Test that tensor.muls gradient is registered."""
        assert self.registry.has_grad("tensor.muls")
        grad_func = self.registry.get_grad("tensor.muls")
        assert grad_func is not None

    def test_tensor_adds_gradient_registered(self):
        """Test that tensor.adds gradient is registered."""
        assert self.registry.has_grad("tensor.adds")
        grad_func = self.registry.get_grad("tensor.adds")
        assert grad_func is not None

    def test_tensor_subs_gradient_registered(self):
        """Test that tensor.subs gradient is registered."""
        assert self.registry.has_grad("tensor.subs")
        grad_func = self.registry.get_grad("tensor.subs")
        assert grad_func is not None

    def test_tensor_divs_gradient_registered(self):
        """Test that tensor.divs gradient is registered."""
        assert self.registry.has_grad("tensor.divs")
        grad_func = self.registry.get_grad("tensor.divs")
        assert grad_func is not None


class TestTileOpGradients:
    """Test gradient registrations for tile operations."""

    @classmethod
    def setup_class(cls):
        """Setup test class - register gradients."""
        auto_register_gradients()
        cls.registry = GradRegistry.get_instance()

    def test_tile_add_gradient_registered(self):
        """Test that tile.add gradient is registered."""
        assert self.registry.has_grad("tile.add")
        grad_func = self.registry.get_grad("tile.add")
        assert grad_func is not None

    def test_tile_sub_gradient_registered(self):
        """Test that tile.sub gradient is registered."""
        assert self.registry.has_grad("tile.sub")
        grad_func = self.registry.get_grad("tile.sub")
        assert grad_func is not None

    def test_tile_mul_gradient_registered(self):
        """Test that tile.mul gradient is registered."""
        assert self.registry.has_grad("tile.mul")
        grad_func = self.registry.get_grad("tile.mul")
        assert grad_func is not None

    def test_tile_div_gradient_registered(self):
        """Test that tile.div gradient is registered."""
        assert self.registry.has_grad("tile.div")
        grad_func = self.registry.get_grad("tile.div")
        assert grad_func is not None

    def test_tile_neg_gradient_registered(self):
        """Test that tile.neg gradient is registered."""
        assert self.registry.has_grad("tile.neg")
        grad_func = self.registry.get_grad("tile.neg")
        assert grad_func is not None

    def test_tile_addc_gradient_registered(self):
        """Test that tile.addc gradient is registered."""
        assert self.registry.has_grad("tile.addc")
        grad_func = self.registry.get_grad("tile.addc")
        assert grad_func is not None

    def test_tile_subc_gradient_registered(self):
        """Test that tile.subc gradient is registered."""
        assert self.registry.has_grad("tile.subc")
        grad_func = self.registry.get_grad("tile.subc")
        assert grad_func is not None

    def test_tile_mulc_gradient_registered(self):
        """Test that tile.mulc gradient is registered."""
        assert self.registry.has_grad("tile.mulc")
        grad_func = self.registry.get_grad("tile.mulc")
        assert grad_func is not None

    def test_tile_divc_gradient_registered(self):
        """Test that tile.divc gradient is registered."""
        assert self.registry.has_grad("tile.divc")
        grad_func = self.registry.get_grad("tile.divc")
        assert grad_func is not None

    def test_tile_addsc_gradient_registered(self):
        """Test that tile.addsc gradient is registered."""
        assert self.registry.has_grad("tile.addsc")
        grad_func = self.registry.get_grad("tile.addsc")
        assert grad_func is not None

    def test_tile_subsc_gradient_registered(self):
        """Test that tile.subsc gradient is registered."""
        assert self.registry.has_grad("tile.subsc")
        grad_func = self.registry.get_grad("tile.subsc")
        assert grad_func is not None


class TestSystemOpGradients:
    """Test gradient registrations for system operations."""

    @classmethod
    def setup_class(cls):
        """Setup test class - register gradients."""
        auto_register_gradients()
        cls.registry = GradRegistry.get_instance()

    def test_system_tpush_to_aiv_gradient_registered(self):
        """Test that system.tpush_to_aiv gradient is registered."""
        assert self.registry.has_grad("system.tpush_to_aiv")
        grad_func = self.registry.get_grad("system.tpush_to_aiv")
        assert grad_func is not None

    def test_system_tpop_from_aic_gradient_registered(self):
        """Test that system.tpop_from_aic gradient is registered."""
        assert self.registry.has_grad("system.tpop_from_aic")
        grad_func = self.registry.get_grad("system.tpop_from_aic")
        assert grad_func is not None


class TestGradientFunctionGeneration:
    """Test gradient function generation for various operations."""

    @classmethod
    def setup_class(cls):
        """Setup test class - register gradients."""
        auto_register_gradients()

    def _print_functions(self, forward_func, grad_func):
        """Print forward and gradient functions as Python code."""
        print("\n" + "=" * 80)
        print(f"Forward function: {forward_func.name}")
        print("=" * 80)
        print(forward_func.as_python())
        print()
        print("=" * 80)
        print(f"Gradient function: {grad_func.name}")
        print("=" * 80)
        print(grad_func.as_python())
        print("=" * 80)

    def test_grad_for_add_operation(self):
        """Test gradient function generation for add operation."""
        @pl.function
        def add_func(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        grad_func = pl.grad_by_trace_v2(add_func)
        assert grad_func is not None
        assert grad_func.name == "grad_add_func"

        self._print_functions(add_func, grad_func)

    def test_grad_for_mul_operation(self):
        """Test gradient function generation for mul operation."""
        @pl.function
        def mul_func(x: pl.Tensor[[64, 128], pl.FP32], y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.mul(x, y)
            return result

        grad_func = pl.grad_by_trace_v2(mul_func)
        assert grad_func is not None
        assert grad_func.name == "grad_mul_func"
        assert len(grad_func.return_types) == 2

        self._print_functions(mul_func, grad_func)

    def test_grad_for_sub_operation(self):
        """Test gradient function generation for sub operation."""
        @pl.function
        def sub_func(x: pl.Tensor[[64, 128], pl.FP32], y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.sub(x, y)
            return result

        grad_func = pl.grad_by_trace_v2(sub_func)
        assert grad_func is not None
        assert grad_func.name == "grad_sub_func"

        self._print_functions(sub_func, grad_func)

    def test_grad_for_div_operation(self):
        """Test gradient function generation for div operation."""
        @pl.function
        def div_func(x: pl.Tensor[[64, 128], pl.FP32], y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.div(x, y)
            return result

        grad_func = pl.grad_by_trace_v2(div_func)
        assert grad_func is not None
        assert grad_func.name == "grad_div_func"

        self._print_functions(div_func, grad_func)

    def test_grad_for_neg_operation(self):
        """Test gradient function generation for neg operation."""
        @pl.function
        def neg_func(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.neg(x)
            return result

        grad_func = pl.grad_by_trace_v2(neg_func)
        assert grad_func is not None
        assert grad_func.name == "grad_neg_func"

        self._print_functions(neg_func, grad_func)

    def test_grad_for_sqrt_operation(self):
        """Test gradient function generation for sqrt operation."""
        @pl.function
        def sqrt_func(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(x)
            return result

        grad_func = pl.grad_by_trace_v2(sqrt_func)
        assert grad_func is not None
        assert grad_func.name == "grad_sqrt_func"

        self._print_functions(sqrt_func, grad_func)

    def test_grad_for_exp_operation(self):
        """Test gradient function generation for exp operation."""
        @pl.function
        def exp_func(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.exp(x)
            return result

        grad_func = pl.grad_by_trace_v2(exp_func)
        assert grad_func is not None
        assert grad_func.name == "grad_exp_func"

        self._print_functions(exp_func, grad_func)

    def test_grad_for_maximum_operation(self):
        """Test gradient function generation for maximum operation."""
        @pl.function
        def max_func(x: pl.Tensor[[64, 128], pl.FP32], y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.maximum(x, y)
            return result

        grad_func = pl.grad_by_trace_v2(max_func)
        assert grad_func is not None
        assert grad_func.name == "grad_max_func"
        assert len(grad_func.return_types) == 2

        self._print_functions(max_func, grad_func)

    def test_grad_for_chain_operations(self):
        """Test gradient function generation for chain of operations."""
        @pl.function
        def chain_func(x: pl.Tensor[[64, 128], pl.FP32], y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, y)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp, temp)
            return result

        grad_func = pl.grad_by_trace_v2(chain_func)
        assert grad_func is not None
        assert grad_func.name == "grad_chain_func"
        assert len(grad_func.return_types) == 2

        self._print_functions(chain_func, grad_func)

    def test_grad_for_complex_operations(self):
        """Test gradient function generation for complex operations."""
        @pl.function
        def complex_func(x: pl.Tensor[[64, 128], pl.FP32], y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            t1: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, y)
            t2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(t1, t1)
            result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(t2)
            return result

        grad_func = pl.grad_by_trace_v2(complex_func)
        assert grad_func is not None
        assert grad_func.name == "grad_complex_func"
        assert len(grad_func.return_types) == 2

        self._print_functions(complex_func, grad_func)


def main():
    """Main function to run all tests and print results."""
    print("Running gradient registration tests...")
    print("=" * 80)

    # Register gradients
    auto_register_gradients()

    # Create test instance
    test_instance = TestGradientFunctionGeneration()

    # Run all test methods
    test_methods = [
        test_instance.test_grad_for_add_operation,
        test_instance.test_grad_for_mul_operation,
        test_instance.test_grad_for_sub_operation,
        test_instance.test_grad_for_div_operation,
        test_instance.test_grad_for_neg_operation,
        test_instance.test_grad_for_sqrt_operation,
        test_instance.test_grad_for_exp_operation,
        test_instance.test_grad_for_maximum_operation,
        test_instance.test_grad_for_chain_operations,
        test_instance.test_grad_for_complex_operations,
    ]

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            test_method()
            passed += 1
        except Exception as e:
            print(f"\nERROR in {test_method.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 80)


if __name__ == "__main__":
    main()
