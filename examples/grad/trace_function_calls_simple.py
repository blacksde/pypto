# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example demonstrating trace functionality in function call scenarios.

This example shows how to use pl.trace() to analyze operations
when one pl.function calls another pl.function. It demonstrates
the current behavior: tracing a function only captures operations in
that function, not in called functions.

To trace all operations in the entire computation, you need to
trace each function separately.

Usage:
    python examples/grad/trace_function_calls_simple.py
"""

import pypto.language as pl


def test_single_function_trace():
    """Test tracing a single function."""
    print("=" * 80)
    print("TEST 1: Single Function Tracing")
    print("=" * 80)
    print("This demonstrates tracing a simple function.")
    print()

    @pl.function
    def simple_add(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        result: pl.Tensor[[64, 64], pl.FP32] = pl.add(x, x)
        return result

    # Trace the function
    trace_result = pl.trace(simple_add)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Print operations
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name} ({op.op_type})")
    print()


def test_orchestration_trace():
    """Test tracing an orchestration function."""
    print("=" * 80)
    print("TEST 2: Orchestration Function Tracing")
    print("=" * 80)
    print("This demonstrates tracing an orchestration function")
    print("that calls InCore functions.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def compute_kernel(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        out = pl.store(tile_sum, [0, 0], c)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        c = compute_kernel(a, b, c)
        return c

    # Trace the orchestration function
    trace_result = pl.trace(orchestrator)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Print operations
    print("Operations in orchestration function:")
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name} ({op.op_type})")
    print()

    print("NOTE: This trace only shows operations in the")
    print("      orchestration function, not in the called InCore function.")
    print("      To see operations in compute_kernel, trace it separately.")
    print()


def test_incore_trace():
    """Test tracing an InCore function."""
    print("=" * 80)
    print("TEST 3: InCore Function Tracing")
    print("=" * 80)
    print("This demonstrates tracing an InCore function.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def incore_func(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        tile_scaled = pl.mul(tile_sum, 2.0)
        out = pl.store(tile_scaled, [0, 0], c)
        return out

    # Trace the InCore function
    trace_result = pl.trace(incore_func)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Print operations
    print("Operations in InCore function:")
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name} ({op.op_type})")
    print()


def test_multiple_function_traces():
    """Test tracing multiple functions separately."""
    print("=" * 80)
    print("TEST 4: Multiple Function Traces")
    print("=" * 80)
    print("This demonstrates tracing each function separately")
    print("to collect all operations in the computation graph.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_func(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        out = pl.store(tile_sum, [0, 0], c)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        c = kernel_func(a, b, c)
        return c

    # Trace both functions separately
    print("Tracing kernel function:")
    kernel_trace = pl.trace(kernel_func)
    print(f"  Operations: {kernel_trace.count()}")
    for i, op in enumerate(kernel_trace.operations):
        print(f"  [{i}] {op.op_name}")
    print()

    print("Tracing orchestration function:")
    orch_trace = pl.trace(orchestrator)
    print(f"  Operations: {orch_trace.count()}")
    for i, op in enumerate(orch_trace.operations):
        print(f"  [{i}] {op.op_name}")
    print()

    # Combine traces
    total_ops = kernel_trace.count() + orch_trace.count()
    print(f"\nTotal operations across both functions: {total_ops}")
    print()


def test_filtering_operations():
    """Test filtering operations in traced functions."""
    print("=" * 80)
    print("TEST 5: Operation Filtering")
    print("=" * 80)
    print("This demonstrates filtering operations by type and name.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def mixed_ops(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        tile_scaled = pl.mul(tile_sum, 2.0)
        tile_relu = pl.relu(tile_scaled)
        out = pl.store(tile_relu, [0, 0], c)
        return out

    # Trace the function
    trace_result = pl.trace(mixed_ops)

    print(f"Total operations: {trace_result.count()}")
    print()

    # Filter by operation type
    print("Filtering by operation type:")
    load_ops = trace_result.filter_by_type("tensor")
    print(f"  Tensor operations: {len(load_ops)}")

    # Filter by operation name
    print("\nFiltering by operation name:")
    add_ops = trace_result.filter_by_op("tile.add")
    mul_ops = trace_result.filter_by_op("tile.mul")
    relu_ops = trace_result.filter_by_op("tile.relu")

    print(f"  Add operations: {len(add_ops)}")
    print(f"  Multiply operations: {len(mul_ops)}")
    print(f"  ReLU operations: {len(relu_ops)}")
    print()


def main():
    """Run all tests."""
    print("PyPTO Trace Functionality - Function Call Scenarios")
    print("=" * 80)
    print("This example demonstrates pl.trace() behavior")
    print("when pl.function calls other pl.function.")
    print("=" * 80)
    print()

    test_single_function_trace()
    print()

    test_orchestration_trace()
    print()

    test_incore_trace()
    print()

    test_multiple_function_traces()
    print()

    test_filtering_operations()
    print()

    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nKey findings:")
    print("  1. Tracing a function captures operations in that function")
    print("  2. Tracing orchestration function shows function calls,")
    print("     not operations inside called functions")
    print("  3. To trace all operations, trace each function separately")
    print("  4. Filtering works by operation type and name")
    print("  5. Multiple output formats are supported")
    print("\nUse cases for trace functionality:")
    print("  - Debugging multi-kernel programs")
    print("  - Analyzing operation distribution")
    print("  - Verifying operation fusion")
    print("  - Performance optimization")
    print("  - Memory usage analysis")


if __name__ == "__main__":
    main()