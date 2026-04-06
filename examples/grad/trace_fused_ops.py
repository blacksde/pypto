# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example demonstrating trace functionality for fused operations.

This example shows how to use pl.trace() to analyze and debug
fused operations in PyPTO programs. Each fused operation is traced
and detailed information about operations is printed.

Programs traced:
  - FusedAddScale   — c = (a + b) * 2.0           (vector only)
  - FusedAddRelu    — c = relu(a + b)              (vector only)
  - FusedMatmulBias — c = matmul(a, b) + bias      (cube + vector)

Usage:
    python examples/grad/trace_fused_ops.py
"""

import pypto.language as pl


def trace_fused_add_scale():
    """Trace FusedAddScale operation: c = (a + b) * 2.0"""
    print("=" * 80)
    print("TRACING: FusedAddScale - c = (a + b) * 2.0")
    print("=" * 80)
    print("This operation demonstrates:")
    print("  - Loading two input tiles")
    print("  - Adding them together")
    print("  - Scaling the result by 2.0")
    print("  - Storing the output")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def fused_add_scale(
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Fused: load a, b -> add -> scale by 2.0 -> store c."""
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_sum = pl.add(tile_a, tile_b)
        tile_c = pl.mul(tile_sum, 2.0)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    # Trace the function
    trace_result = pl.trace(fused_add_scale)

    # Print summary
    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Type: {trace_result.function_type}")
    print(f"  Parameters: {len(trace_result.param_info)}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Print operation details
    print("Operations found:")
    for op in trace_result.operations:
        print(f"  [{op.index}] {op.op_name}")
        print(f"      Type: {op.op_type}")
        print(f"      Arguments: {len(op.args)}")
        if op.kwargs:
            print(f"      Keyword args: {list(op.kwargs.keys())}")
    print()

    # Print detailed trace
    print("-" * 80)
    print("DETAILED TRACE:")
    print("-" * 80)
    pl.print_trace(trace_result, format="detailed")
    print()


def trace_fused_add_relu():
    """Trace FusedAddRelu operation: c = relu(a + b)"""
    print("=" * 80)
    print("TRACING: FusedAddRelu - c = relu(a + b)")
    print("=" * 80)
    print("This operation demonstrates:")
    print("  - Loading two input tiles")
    print("  - Adding them together")
    print("  - Applying ReLU activation")
    print("  - Storing the output")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def fused_add_relu(
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Fused: load a, b -> add -> relu -> store c."""
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_sum = pl.add(tile_a, tile_b)
        tile_c = pl.relu(tile_sum)
        out_c = pl.store(tile_c, [0, 0], c)
        return out_c

    # Trace the function
    trace_result = pl.trace(fused_add_relu)

    # Print summary
    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Count operations by type
    op_types = {}
    for op in trace_result.operations:
        op_types[op.op_type] = op_types.get(op.op_type, 0) + 1

    print("Operation breakdown by type:")
    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type}: {count}")
    print()

    # Print detailed trace
    print("-" * 80)
    print("DETAILED TRACE:")
    print("-" * 80)
    pl.print_trace(trace_result, format="detailed")
    print()


def trace_fused_matmul_bias():
    """Trace FusedMatmulBias operation: c = matmul(a, b) + bias"""
    print("=" * 80)
    print("TRACING: FusedMatmulBias - c = matmul(a, b) + bias")
    print("=" * 80)
    print("This operation demonstrates:")
    print("  - Matrix multiplication (cube unit)")
    print("  - Bias addition (vector unit)")
    print("  - Multi-kernel orchestration")
    print("  - Memory space usage (L1, L0A, L0B)")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def matmul_kernel(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Cube InCore: compute a @ b and store to output."""
        tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.store(tile_c_l0c, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def add_bias_kernel(
        x: pl.Tensor[[64, 64], pl.FP32],
        bias: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Vector InCore: add bias to x and store to output."""
        tile_x = pl.load(x, [0, 0], [64, 64])
        tile_bias = pl.load(bias, [0, 0], [64, 64])
        tile_c = pl.add(tile_x, tile_bias)
        out = pl.store(tile_c, [0, 0], output)
        return out

    # Trace both kernels
    print("Tracing matmul_kernel:")
    trace_matmul = pl.trace(matmul_kernel)
    print(f"  Operations: {trace_matmul.count()}")
    print(f"  Memory moves: {len(trace_matmul.filter_by_op('tile.move'))}")
    print(f"  Matmul ops: {len(trace_matmul.filter_by_op('tile.matmul'))}")
    print()

    print("Tracing add_bias_kernel:")
    trace_add_bias = pl.trace(add_bias_kernel)
    print(f"  Operations: {trace_add_bias.count()}")
    print(f"  Add ops: {len(trace_add_bias.filter_by_op('tile.add'))}")
    print()

    # Print detailed trace for matmul kernel
    print("-" * 80)
    print("DETAILED TRACE FOR MATMUL KERNEL:")
    print("-" * 80)
    pl.print_trace(trace_matmul, format="detailed")
    print()


def demonstrate_filtering():
    """Demonstrate trace filtering capabilities."""
    print("=" * 80)
    print("DEMONSTRATING TRACE FILTERING")
    print("=" * 80)
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def mixed_ops(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Mixed operations for filtering demo."""
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        tile_scaled = pl.mul(tile_sum, 2.0)
        tile_activated = pl.relu(tile_scaled)
        out = pl.store(tile_activated, [0, 0], c)
        return out

    # Trace the function
    trace_result = pl.trace(mixed_ops)

    print(f"Total operations: {trace_result.count()}")
    print()

    # Filter by operation type
    print("Filtering by operation type:")
    load_ops = trace_result.filter_by_type("tile")
    print(f"  Tile operations: {len(load_ops)}")

    # Filter by specific operation
    print("\nFiltering by specific operation:")
    add_ops = trace_result.filter_by_op("tile.add")
    mul_ops = trace_result.filter_by_op("tile.mul")
    relu_ops = trace_result.filter_by_op("tile.relu")

    print(f"  Add operations: {len(add_ops)}")
    print(f"  Multiply operations: {len(mul_ops)}")
    print(f"  ReLU operations: {len(relu_ops)}")
    print()

    # Get operations by index
    print("Accessing operations by index:")
    first_op = trace_result.get_operation(0)
    last_op = trace_result.get_operation(trace_result.count() - 1)
    invalid_op = trace_result.get_operation(999)

    print(f"  First operation: {first_op.op_name if first_op else 'None'}")
    print(f"  Last operation: {last_op.op_name if last_op else 'None'}")
    print(f"  Invalid index: {invalid_op}")
    print()


def demonstrate_json_export():
    """Demonstrate JSON export functionality."""
    print("=" * 80)
    print("DEMONSTRATING JSON EXPORT")
    print("=" * 80)
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def simple_ops(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Simple operations for JSON demo."""
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        out = pl.store(tile_sum, [0, 0], c)
        return out

    # Trace and export as JSON
    trace_result = pl.trace(simple_ops)
    json_output = trace_result.to_json()

    print("JSON export:")
    print(json_output)
    print()

    # Parse JSON to demonstrate structure
    import json
    data = json.loads(json_output)
    print("JSON structure validation:")
    print(f"  ✓ Function name: {data['function_name']}")
    print(f"  ✓ Operation count: {len(data['operations'])}")
    print(f"  ✓ Has parameters: {len(data['param_info']) > 0}")
    print(f"  ✓ Has return types: {len(data['return_types']) > 0}")
    print()


def main():
    """Main function to run all trace demonstrations."""
    print("PyPTO Trace Functionality Example")
    print("=" * 80)
    print("This example demonstrates the pl.trace() functionality for")
    print("analyzing and debugging fused operations in PyPTO programs.")
    print("=" * 80)
    print()

    # Run all demonstrations
    trace_fused_add_scale()
    print()

    trace_fused_add_relu()
    print()

    trace_fused_matmul_bias()
    print()

    demonstrate_filtering()
    print()

    demonstrate_json_export()
    print()

    print("=" * 80)
    print("TRACE DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nKey capabilities demonstrated:")
    print("  1. Tracing individual functions")
    print("  2. Collecting operation information")
    print("  3. Filtering operations by type and name")
    print("  4. Accessing operations by index")
    print("  5. Exporting trace data as JSON")
    print("  6. Multiple output formats (text, detailed, JSON)")
    print("\nUse cases for trace functionality:")
    print("  - Debugging compilation issues")
    print("  - Analyzing operation patterns")
    print("  - Verifying operation fusion")
    print("  - Performance optimization")
    print("  - Memory usage analysis")


if __name__ == "__main__":
    main()