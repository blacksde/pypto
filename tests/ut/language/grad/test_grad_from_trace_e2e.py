# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
End-to-end test suite for pl.grad_from_trace().

This test suite covers:
1. Simple function gradient computation
2. Function call scenarios
3. Accumulation scenarios
"""

import sys
import pypto.language as pl


class TestSimpleFunctionGradient:
    """Test simple function gradient computation."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def test_simple_add_gradient(self):
        """Test gradient of simple addition function."""
        print("Test: Simple Addition Function Gradient")
        print("-" * 50)

        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32],
                   y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result = pl.add(x, y)
            return result

        try:
            # Generate gradient function
            grad_func = pl.grad_from_trace(forward)

            print(f"Forward function: {forward.name}")
            print(f"Gradient function: {grad_func.name}")
            print(f"Gradient function parameters: {len(grad_func.params)}")
            print(f"Gradient function return types: {len(grad_func.return_types)}")

            # Print forward function code
            print("\nForward function code:")
            print("=" * 50)
            forward_code = forward.as_python()
            print(forward_code)
            print("=" * 50)

            # Print gradient function code
            print("\nGradient function code:")
            print("=" * 50)
            grad_code = grad_func.as_python()
            print(grad_code)
            print("=" * 50)

            if grad_func is not None and grad_func.name == f"grad_{forward.name}":
                print("✓ PASS: Simple addition gradient function generated")
                self.passed += 1
                return True
            else:
                print("✗ FAIL: Gradient function generation failed")
                self.failed += 1
                return False

        except Exception as e:
            print(f"✗ FAIL: {e}")
            self.failed += 1
            return False

        print()

    def test_simple_mul_gradient(self):
        """Test gradient of simple multiplication function."""
        print("Test: Simple Multiplication Function Gradient")
        print("-" * 50)

        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32],
                   y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result = pl.mul(x, y)
            return result

        try:
            # Generate gradient function using grad_by_trace (includes grad_output and forward_output as inputs)
            grad_func = pl.grad_by_trace(forward)

            print(f"Forward function: {forward.name}")
            print(f"Gradient function: {grad_func.name}")

            # Print forward function code
            print("\nForward function code:")
            print("=" * 50)
            forward_code = forward.as_python()
            print(forward_code)
            print("=" * 50)

            # Print gradient function code
            print("\nGradient function code:")
            print("=" * 50)
            grad_code = grad_func.as_python()
            print(grad_code)
            print("=" * 50)

            # Verify that gradient function has correct parameters
            param_names = [p.name_hint for p in grad_func.params]
            expected_params = ['x', 'y', 'result', 'dresult']
            
            if param_names == expected_params:
                print(f"✓ PASS: Gradient function has correct parameters: {param_names}")
            else:
                print(f"✗ FAIL: Expected parameters {expected_params}, got {param_names}")
                self.failed += 1
                return False

            # Verify that gradient function body has dx and dy definitions
            grad_code_str = grad_code.lower()
            if 'dx' in grad_code_str and 'dy' in grad_code_str:
                print("✓ PASS: Gradient function body contains dx and dy definitions")
            else:
                print("✗ FAIL: Gradient function body missing dx or dy definitions")
                self.failed += 1
                return False

            if grad_func is not None and grad_func.name == f"grad_{forward.name}":
                print("✓ PASS: Simple multiplication gradient function generated")
                self.passed += 1
                return True
            else:
                print("✗ FAIL: Gradient function generation failed")
                self.failed += 1
                return False

        except Exception as e:
            print(f"✗ FAIL: {e}")
            self.failed += 1
            return False

        print()


class TestFunctionCallGradient:
    """Test function call scenarios."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def test_nested_operations_gradient(self):
        """Test gradient of function with nested operations."""
        print("Test: Nested Operations Function Gradient")
        print("-" * 50)

        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32],
                   y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp1 = pl.add(x, y)
            temp2 = pl.mul(temp1, x)
            result = pl.add(temp2, y)
            return result

        try:
            # Generate gradient function
            grad_func = pl.grad_from_trace(forward)

            print(f"Forward function: {forward.name}")
            print(f"Gradient function: {grad_func.name}")
            print(f"Gradient function parameters: {len(grad_func.params)}")

            # Print forward function code
            print("\nForward function code:")
            print("=" * 50)
            forward_code = forward.as_python()
            print(forward_code)
            print("=" * 50)

            # Print gradient function code
            print("\nGradient function code:")
            print("=" * 50)
            grad_code = grad_func.as_python()
            print(grad_code)
            print("=" * 50)

            if grad_func is not None and grad_func.name == f"grad_{forward.name}":
                print("✓ PASS: Nested operations gradient function generated")
                self.passed += 1
                return True
            else:
                print("✗ FAIL: Gradient function generation failed")
                self.failed += 1
                return False

        except Exception as e:
            print(f"✗ FAIL: {e}")
            self.failed += 1
            return False

        print()

    def test_chained_operations_gradient(self):
        """Test gradient of function with chained operations."""
        print("Test: Chained Operations Function Gradient")
        print("-" * 50)

        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32],
                   y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp1 = pl.add(x, y)
            temp2 = pl.mul(temp1, temp1)
            result = pl.div(temp2, x)
            return result

        try:
            # Generateane gradient function
            grad_func = pl.grad_from_trace(forward)

            print(f"Forward function: {forward.name}")
            print(f"Gradient function: {grad_func.name}")

            # Print forward function code
            print("\nForward function code:")
            print("=" * 50)
            forward_code = forward.as_python()
            print(forward_code)
            print("=" * 50)

            # Print gradient function
            print("\nGradient function code:")
            print("=" * 50)
            grad_code = grad_func.as_python()
            print(grad_code)
            print("=" * 50)

            if grad_func is not None and grad_func.name == f"grad_{forward.name}":
                print("✓ PASS: Chained operations gradient function generated")
                self.passed += 1
                return True
            else:
                print("✗ FAIL: Gradient function generation failed")
                self.failed += 1
                return False

        except Exception as e:
            print(f"✗ FAIL: {e}")
            self.failed += 1
            return False

        print()


class TestAccumulationGradient:
    """Test accumulation scenarios."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def test_multi_use_accumulation_gradient(self):
        """Test gradient of function with multi-use variables (requires accumulation)."""
        print("Test: Multi-Use Accumulation Function Gradient")
        print("-" * 50)

        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32],
                   y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            # x and y are used multiple times
            temp1 = pl.add(x, y)
            temp2 = pl.mul(x, y)
            result = pl.add(temp1, temp2)
            return result

        try:
            # Generate gradient function
            grad_func = pl.grad_from_trace(forward)

            print(f"Forward function: {forward.name}")
            print(f"Gradient function: {grad_func.name}")
            print(f"Gradient function parameters: {len(grad_func.params)}")

            # Print forward function code
            print("\nForward function code:")
            print("=" * 50)
            forward_code = forward.as_python()
            print(forward_code)
            print("=" * 50)

            # Print gradient function code
            print("\nGradient function code:")
            print("=" * 50)
            grad_code = grad_func.as_python()
            print(grad_code)
            print("=" * 50)

            if grad_func is not None and grad_func.name == f"grad_{forward.name}":
                print("✓ PASS: Multi-use accumulation gradient function generated")
                self.passed += 1
                return True
            else:
                print("✗ FAIL: Gradient function generation failed")
                self.failed += 1
                return False

        except Exception as e:
            print(f"✗ FAIL: {e}")
            self.failed += 1
            return False

        print()

    def test_complex_accumulation_gradient(self):
        """Test gradient of function with complex accumulation pattern."""
        print("Test: Complex Accumulation Function Gradient")
        print("-" * 50)

        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32],
                   y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            # Multiple uses of x
            temp1 = pl.add(x, y)
            temp2 = pl.mul(x, x)
            temp3 = pl.add(temp1, temp2)
            result = pl.mul(temp3, x)
            return result

        try:
            # Generate gradient function
            grad_func = pl.grad_from_trace(forward)

            print(f"Forward function: {forward.name}")
            print(f"Gradient function: {grad_func.name}")

            # Print forward function code
            print("\nForward function code:")
            print("=" * 50)
            forward_code = forward.as_python()
            print(forward_code)
            print("=" * 50)

            # Print gradient function code
            print("\nGradient function code:")
            print("=" * 50)
            grad_code = grad_func.as_python()
            print(grad_code)
            print("=" * 50)

            if grad_func is not None and grad_func.name == f"grad_{forward.name}":
                print("✓ PASS: Complex accumulation gradient function generated")
                self.passed += 1
                return True
            else:
                print("✗ FAIL: Gradient function generation failed")
                self.failed += 1
                return False

        except Exception as e:
            print(f"✗ FAIL: {e}")
            self.failed += 1
            return False

        print()


def run_all_tests():
    """Run all test suites."""
    print("=" * 80)
    print("PyPTO pl.grad_from_trace() End-to-End Test Suite")
    print("=" * 80)
    print()

    total_passed = 0
    total_failed = 0

    # Test suite 1: Simple function gradients
    # print("SUITE 1: Simple Function Gradients")
    # print("=" * 80)
    # test_simple = TestSimpleFunctionGradient()
    # test_simple.test_simple_add_gradient()
    # test_simple.test_simple_mul_gradient()
    # total_passed += test_simple.passed
    # total_failed += test_simple.failed

    # Test suite 2: Function call gradients
    print("\nSUITE 2: Function Call Gradients")
    print("=" * 80)
    test_call = TestFunctionCallGradient()
    test_call.test_nested_operations_gradient()
    # test_call.test_chained_operations_gradient()
    total_passed += test_call.passed
    total_failed += test_call.failed

    # Test suite 3: Accumulation gradients
    # print("\nSUITE 3: Accumulation Gradients")
    # print("=" * 80)
    # test_accum = TestAccumulationGradient()
    # test_accum.test_multi_use_accumulation_gradient()
    # test_accum.test_complex_accumulation_gradient()
    # total_passed += test_accum.passed
    # total_failed += test_accum.failed

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests passed: {total_passed}")
    print(f"Total tests failed: {total_failed}")
    print(f"Total tests: {total_passed + total_failed}")

    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total_failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
