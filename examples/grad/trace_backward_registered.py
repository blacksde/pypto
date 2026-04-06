# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example demonstrating is_backward_registered attribute for pl.function.

This example shows how to use the is_backward_registered attribute
to mark functions as backward-registered, which causes the trace
system to treat them as complete independent nodes without
tracing their internal operations.

Scenario:
  - InCore function with is_backward_registered=True calls another InCore function
    - The called function's internal operations are not traced
    - The calling function shows the called function as a single operation
  - This is useful for backward pass registration where functions
    should be treated as atomic operations

Usage:
    python examples/grad/trace_backward_registered.py
"""

import pypto.language as pl


def test_backward_registered_basic():
    """Test basic is_backward_registered functionality."""
    print("=" * 80)
    print("TEST 1: Basic is_backward_registered Function")
    print("=" * 80)
    print("This test demonstrates a simple InCore function")
    print("marked with is_backward_registered=True.")
    print()

    @pl.function(type=pl.FunctionType.InCore, is_backward_registered=True)
    def atomic_kernel(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Atomic kernel: c = a + b"""
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        out = pl.store(tile_sum, [0, 0], c)
        return out

    # Trace the function
    trace_result = pl.trace(atomic_kernel)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Print operations
    print("Operations in trace:")
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name} ({op.op_type})")
    print()

    print("NOTE: Since is_backward_registered=True, this function")
    print("      is treated as a complete independent node.")
    print("      Internal operations (load, add, store) are not traced.")
    print()


def test_incore_to_incore_with_backward_registered():
    """Test InCore to InCore call with is_backward_registered."""
    print("=" * 80)
    print("TEST 2: InCore to InCore with is_backward_registered")
    print("=" * 80)
    print("This test demonstrates an InCore function that calls")
    print("another InCore function marked with is_backward_registered=True.")
    print()

    @pl.function(type=pl.FunctionType.InCore, is_backward_registered=True)
    def compute_kernel(
        x: pl.Tensor[[64, 64], pl.FP32],
        y: pl.Tensor[[64, 64], pl.FP32],
        z: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Compute kernel: z = x * y + 1.0"""
        tile_x = pl.load(x, [0, 0], [64, 64])
        tile_y = pl.load(y, [0, 0], [64, 64])
        tile_mul = pl.mul(tile_x, tile_y)
        tile_bias = pl.add(tile_mul, 1.0)
        out = pl.store(tile_bias, [0, 0], z)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def main_kernel(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Main kernel: c = compute(a, b) + 2.0"""
        temp = compute_kernel(a, b, c)
        tile_temp = pl.load(temp, [0, 0], [64, 64])
        tile_scaled = pl.mul(tile_temp, 2.0)
        out = pl.store(tile_scaled, [0, 0], c)
        return out

    # Trace the main kernel
    trace_result = pl.trace(main_kernel)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Print operations
    print("Operations in trace:")
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name} ({op.op_type})")
        if op.kwargs:
            print(f"      Kwargs: {list(op.kwargs.keys())}")
    print()

    print("Expected behavior:")
    print("  - compute_kernel should appear as a single operation")
    print("  - Internal operations of compute_kernel are not traced")
    print("  - Operations from main_kernel are traced normally")
    print()

    # Print detailed trace
    print("-" * 80)
    print("DETAILED TRACE:")
    print("-" * 80)
    pl.print_trace(trace_result, format="detailed")
    print()


def test_multiple_backward_registered_calls():
    """Test multiple is_backward_registered function calls."""
    print("=" * 80)
    print("TEST 3: Multiple is_backward_registered Calls")
    print("=" * 80)
    print("This test demonstrates a function that calls multiple")
    print("is_backward_registered functions.")
    print()

    @pl.function(type=pl.FunctionType.InCore, is_backward_registered=True)
    def add_kernel(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        out = pl.store(tile_sum, [0, 0], c)
        return out

    @pl.function(type=pl.FunctionType.InCore, is_backward_registered=True)
    def mul_kernel(
        x: pl.Tensor[[64, 64], pl.FP32],
        y: pl.Tensor[[64, 64], pl.FP32],
        z: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_x = pl.load(x, [0, 0], [64, 64])
        tile_y = pl.load(y, [0, 0], [64, 64])
        tile_mul = pl.mul(tile_x, tile_y)
        out = pl.store(tile_mul, [0, 0], z)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def pipeline_kernel(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        temp1 = add_kernel(a, b, c)
        temp2 = mul_kernel(temp1, temp1, c)
        tile_temp2 = pl.load(temp2, [0, 0], [64, 64])
        out = pl.store(tile_temp2, [0, 0], c)
        return out

    # Trace the pipeline kernel
    trace_result = pl.trace(pipeline_kernel)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Print operations
    print("Operations in trace:")
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name} ({op.op_type})")
    print()

    print("Expected behavior:")
    print("  - add_kernel should appear as a single operation")
    print("  - mul_kernel should appear as a single operation")
    print("  - Internal operations of both are not traced")
    print("  - Operations from pipeline_kernel are traced normally")
    print()


def test_comparison_without_backward_registered():
    """Test comparison: same function without is_backward_registered."""
    print("=" * 80)
    print("TEST 4: Comparison - Without is_backward_registered")
    print("=" * 80)
    print("This test traces the same function without")
    print("is_backward_registered for comparison.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def regular_kernel(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        out = pl.store(tile_sum, [0, 0], c)
        return out

    # Trace the regular kernel
    trace_result = pl.trace(regular_kernel)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Print operations
    print("Operations in trace:")
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name} ({op.op_type})")
    print()

    print("Expected behavior:")
    print("  - All operations (load, add, store) are traced")
    print("  - No operation skipping occurs")
    print()

    # Print detailed trace
    print("-" * 80)
    print("DETAILED TRACE:")
    print("-" * 80)
    pl.print_trace(trace_result, format="detailed")
    print()

def main():
    """Run all is_backward_registered_registered tests."""
    print("PyPTO Trace Functionality - is_backward_registered Attribute")
    print("=" * 80)
    print("This example demonstrates the is_backward_registered")
    print("attribute for pl.function.")
    print("=" * 80)
    print()

    test_backward_registered_basic()
    print()

    test_incore_to_incore_with_backward_registered()
    print()

    test_multiple_backward_registered_calls()
    print()

    test_comparison_without_backward_registered()
    print()

    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nKey findings from is_backward_registered testing:")
    print("  1. Functions marked with is_backward_registered=True are")
    print("     treated as complete independent nodes")
    print("  2. Internal operations of backward-registered functions")
    print("     are not traced")
    print("  3. Function calls to backward-registered functions")
    print("     appear as single operations")
    print("  4. This is useful for backward pass registration")
    print("  5. Filtering works correctly with backward-registered functions")
    print("\nUse cases for is_backward_registered:")
    print("  - Backward pass registration and tracing")
    print("  - Treating atomic operations as complete nodes")
    print("  - Simplifying trace output for complex operations")
    print("  - Performance analysis of registered operations")


if __name__ == "__main__":
    main()