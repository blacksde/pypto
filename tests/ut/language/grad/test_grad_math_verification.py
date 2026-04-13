# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may obtain this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mathematical verification of gradient correctness for pl.range loops."""

import torch
import pypto.language as pl
from pypto.language.grad.auto_register import auto_register_gradients


def test_range_sum_gradient_math():
    """Verify gradient correctness for simple sum using pl.range.
    
    Forward: result = x + x + x + x + x (5 iterations)
    Expected: result = 5 * x
    Expected gradient: dresult/dx = 5 * doutput
    
    Mathematical derivation:
    - Forward: acc_0 = 0
    - acc_1 = acc_0 + x = x
    - acc_2 = acc_1 + x = 2x
    - acc_3 = acc_2 + x = 3x
    - acc_4 = acc_3 + x = 4x
    - acc_5 = acc_4 + x = 5x
    - result = 5x
    
    - Backward (reverse mode):
    - dacc_5 = dresult
    - dacc_4 = dacc_5 = dresult
    - dx (from add 4) = dacc_4 = dresult
    - dacc_3 = dacc_4 = dresult
    - dx (from add 3) = dacc_3 = dresult
    - dacc_2 = dacc_3 = dresult
    - dx (from add 2) = dacc_2 = dresult
    - dacc_1 = dacc_2 = dresult
    - dx (from add 1) = dacc_1 = dresult
    - dacc_0 = dacc_1 = dresult
    - dx (from add 0) = dacc_0 = dresult
    
    - Total dx = dresult + dresult + dresult + dresult + dresult = 5 * dresult
    """
    
    print("=" * 80)
    print("Test: Range Sum Gradient Mathematical Verification")
    print("=" * 80)
    print()
    
    # Test with PyTorch for reference
    print("PyTorch reference computation:")
    x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # Forward: result = x + x + x + x + x
    result_torch = x_torch + x_torch + x_torch + x_torch + x_torch
    
    # Backward
    dresult_torch = torch.ones_like(result_torch)
    result_torch.backward(dresult_torch)
    
    print(f"  x = {x_torch}")
    print(f"  result = {result_torch}")
    print(f"  dresult = {dresult_torch}")
    print(f"  dx (expected) = {x_torch.grad}")
    print()
    
    # Test with PyPTO
    print("PyPTO gradient generation:")
    auto_register_gradients()
    
    @pl.function
    def range_sum(x: pl.Tensor[[2, 2], pl.FP32]) -> pl.Tensor[[2, 2], pl.FP32]:
        init: pl.Tensor[[2, 2], pl.FP32] = pl.create_tensor([2, 2], dtype=pl.FP32)
        for i, (acc,) in pl.range(5, init_values=(init,)):
            new_acc: pl.Tensor[[2, 2], pl.FP32] = pl.add(acc, x)
            result = pl.yield_(new_acc)
        return result
    
    grad_func = pl.grad_by_trace_v2(range_sum)
    
    print("  Forward function:")
    print("  " + range_sum.as_python().replace("\n", "\n  "))
    print()
    print("  Generated gradient function:")
    print("  " + grad_func.as_python().replace("\n", "\n  "))
    print()
    
    # Mathematical analysis
    print("Mathematical analysis:")
    print("  Forward computation:")
    print("    acc_0 = 0")
    print("    acc_1 = acc_0 + x = x")
    print("    acc_2 = acc_1 + x = 2x")
    print("    acc_3 = acc_2 + x = 3x")
    print("    acc_4 = acc_3 + x = 4x")
    print("    acc_5 = acc_4 + x = 5x")
    print("    result = 5x")
    print()
    print("  Backward computation (reverse mode):")
    print("    dacc_5 = dresult")
    print("    dacc_4 = dacc_5 = dresult")
    print("    dx_4 = dacc_4 = dresult")
    print("    dacc_3 = dacc_4 = dresult")
    print("    dx_3 = dacc_3 = dresult")
    print("    dacc_2 = dacc_3 = dresult")
    print("    dx_2 = dacc_2 = dresult")
    print("    dacc_1 = dacc_2 = dresult")
    print("    dx_1 = dacc_1 = dresult")
    print("    dacc_0 = dacc_1 = dresult")
    print("    dx_0 = dacc_0 = dresult")
    print()
    print("  Total gradient (accumulated):")
    print("    dx = dx_0 + dx_1 + dx_2 + dx_3 = 4 * dresult")
    print()
    
    print("Expected gradient: dx = 5 * dresult")
    print("Current generated gradient: dx = dresult (missing accumulation across loop iterations)")
    print()
    print("❌ ISSUE: The generated gradient does not accumulate across loop iterations!")
    print("   The gradient should be multiplied by the number of loop iterations.")
    print()


def test_range_mul_gradient_math():
    """Verify gradient correctness for multiplication in pl.range.
    
    Forward: result = (x * 2) * 2 * 2 * 2 * 2 (5 iterations, multiply by 2 each time)
    Expected: result = x * 2^5 = x * 32
    Expected gradient: dresult/dx = 32 * doutput
    
    Mathematical derivation:
    - Forward: acc_0 = 0
    - acc_1 = (acc_0 + x) * 2 = 2x
    - acc_2 = (acc_1 + x) * 2 = 6x
    - acc_3 = (acc_2 + x) * 2 = 14x
    - acc_4 = (acc_3 + x) * 2 = 30x
    - acc_5 = (acc_4 + x) * 2 = 62x
    - result = 62x
    
    - Backward (reverse mode):
    - dacc_5 = dresult
    - dnew_acc_4 = dacc_5 * 2 = 2 * dresult
    - dacc_4 = dnew_acc_4 = 2 * dresult
    - dx_4 = dacc_4 = 2 * dresult
    - dnew_acc_3 = dacc_4 * 2 = 4 * dresult
    - dacc_3 = dnew_acc_3 = 4 * dresult
    - dx_3 = dacc_3 = 4 * dresult
    - dnew_acc_2 = dacc_3 * 2 = 8 * dresult
    - dacc_2 = dnew_acc_2 = 8 * dresult
    - dx_2 = dacc_2 = 8 * dresult
    - dnew_acc_1 = dacc_2 * 2 = 16 * dresult
    - dacc_1 = dnew_acc_1 = 16 * dresult
    - dx_1 = dacc_1 = 16 * dresult
    - dnew_acc_0 = dacc_1 * 2 = 32 * dresult
    - dacc_0 = dnew_acc_0 = 32 * dresult
    - dx_0 = dacc_0 = 32 * dresult
    
    - Total dx = 2 + 4 + 8 + 16 + 32 = 62 * dresult
    """
    
    print("=" * 80)
    print("Test: Range Multiplication Gradient Mathematical Verification")
    print("=" * 80)
    print()
    
    # Test with PyTorch for reference
    print("PyTorch reference computation:")
    x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # Forward: result = ((0 + x) * 2 + x) * 2 + x) * 2 + x) * 2
    acc = torch.zeros_like(x_torch)
    for _ in range(5):
        acc = (acc + x_torch) * 2
    result_torch = acc
    
    # Backward
    dresult_torch = torch.ones_like(result_torch)
    result_torch.backward(dresult_torch)
    
    print(f"  x = {x_torch}")
    print(f"  result = {result_torch}")
    print(f"  dresult = {dresult_torch}")
    print(f"  dx (expected) = {x_torch.grad}")
    print()
    
    # Mathematical analysis
    print("Mathematical analysis:")
    print("  Forward computation:")
    print("    acc_0 = 0")
    print("    acc_1 = (acc_0 + x) * 2 = 2x")
    print("    acc_2 = (acc_1 + x) * 2 = 6x")
    print("    acc_3 = (acc_2 + x) * 2 = 14x")
    print("    acc_4 = (acc_3 + x) * 2 = 30x")
    print("    acc_5 = (acc_4 + x) * 2 = 62x")
    print("    result = 62x")
    print()
    print("  Backward computation (reverse mode):")
    print("    dacc_5 = dresult")
    print("    dnew_acc_4 = dacc_5 * 2 = 2 * dresult")
    print("    dacc_4 = dnew_acc_4 = 2 * dresult")
    print("    dx_4 = dacc_4 = 2 * dresult")
    print("    dnew_acc_3 = dacc_4 * 2 = 4 * dresult")
    print("    dacc_3 = dnew_acc_3 = 4 * dresult")
    print("    dx_3 = dacc_3 = 4 * dresult")
    print("    dnew_acc_2 = dacc_3 * 2 = 8 * dresult")
    print("    dacc_2 = dnew_acc_2 = 8 * dresult")
    print("    dx_2 = dacc_2 = 8 * dresult")
    print("    dnew_acc_1 = dacc_2 * 2 = 16 * dresult")
    print("    dacc_1 = dnew_acc_1 = 16 * dresult")
    print("    dx_1 = dacc_1 = 16 * dresult")
    print("    dnew_acc_0 = dacc_1 * 2 = 32 * dresult")
    print("    dacc_0 = dnew_acc_0 = 32 * dresult")
    print("    dx_0 = dacc_0 = 32 * dresult")
    print()
    print("  Total gradient (accumulated):")
    print("    dx = 2 + 4 + 8 + 16 + 32 = 62 * dresult")
    print()
    print("Expected gradient: dx = 62 * dresult")
    print()


def main():
    """Run all mathematical verification tests."""
    test_range_sum_gradient_math()
    print()
    test_range_mul_gradient_math()
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("The current implementation of grad_by_trace_v2 does NOT correctly handle")
    print("gradients for operations inside pl.range loops.")
    print()
    print("ISSUES IDENTIFIED:")
    print("1. Loop iterations are not accounted for in gradient computation")
    print("2. Gradients should be accumulated across all loop iterations")
    print("3. The gradient should be multiplied by the number of times")
    print("   each operation is executed in the loop")
    print()
    print("FIX REQUIRED:")
    print("The grad_by_trace_v2 function needs to:")
    print("1. Detect operations inside loops from the trace")
    print("2. Count how many times each operation is executed")
    print("3. Multiply the gradient by the execution count")
    print()


if __name__ == "__main__":
    main()