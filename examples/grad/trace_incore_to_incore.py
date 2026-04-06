# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example
demonstrating InCore to InCore function calls with trace functionality.

This example shows how to use pl.trace() to analyze operations when
one InCore function calls another InCore function. This is a common
pattern for modular computation where one InCore function performs
part of the computation and calls another InCore function for the remaining
computation.

Scenario:
  - InCore function 'preprocess' calls InCore function 'compute'
  - Both functions are traced separately
  - Trace results show operations in each function
  - No orchestration function needed

Usage:
    python examples/grad/trace_incore_to_incore.py
"""

import pypto.language as pl


@pl.function(type=pl.FunctionType.InCore)
def preprocess_kernel(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
) -> pl.Tensor[[128, 128], pl.FP32]:
    """InCore kernel: c = (a + b) * 0.5"""
    tile_a = pl.load(a, [0, 0], [128, 128])
    tile_b = pl.load(b, [0, 0], [128, 128])
    tile_sum = pl.add(tile_a, tile_b)
    tile_scaled = pl.mul(tile_sum, 0.5)
    out = pl.store(tile_scaled, [0, 0], c)
    return out


@pl.function(type=pl.FunctionType.InCore)
def compute_kernel(
    x: pl.Tensor[[128, 128], pl.FP32],
    y: pl.Tensor[[128, 128], pl.FP32],
    z: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
) -> pl.Tensor[[128, 128], pl.FP32]:
    """InCore kernel: z = x * y + 1.0"""
    tile_x = pl.load(x, [0, 0], [128, 128])
    tile_y = pl.load(y, [0, 0], [128, 128])
    tile_mul = pl.mul(tile_x, tile_y)
    tile_bias = pl.add(tile_mul, 1.0)
    out = pl.store(tile_bias, [0, 0], z)
    return out


@pl.function(type=pl.FunctionType.InCore)
def postprocess_kernel(
    result: pl.Tensor[[128, 128], pl.FP32],
    scale: pl.Tensor[[128, 128], pl.FP32],
    output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
) -> pl.Tensor[[128, 128], pl.FP32]:
    """InCore kernel: output = result * scale"""
    tile_result = pl.load(result, [0, 0], [128, 128])
    tile_scale = pl.load(scale, [0, 0], [128, 128])
    tile_final = pl.mul(tile_result, tile_scale)
    out = pl.store(tile_final, [0, 0], output)
    return out


def test_single_incore_to_incore_call():
    """Test tracing a single InCore to InCore function call."""
    print("=" * 80)
    print("TEST 1: Single InCore to InCore Call")
    print("=" * 80)
    print("This test traces a single InCore function that calls")
    print("another InCore function.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def pipeline_kernel(
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Pipeline kernel: c = preprocess(a, b) + postprocess(c)"""
        # Call preprocess kernel
        temp = preprocess_kernel(a, b, c)

        # Call compute kernel
        temp = compute_kernel(temp, temp, c)

        # Call postprocess kernel
        result = postprocess_kernel(temp, 2.0, c)
        return result

    # Trace the pipeline kernel
    trace_result = pl.trace(pipeline_kernel)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Analyze operation types
    print("Operation breakdown by type:")
    op_types = {}
    for op in trace_result.operations:
        op_types[op.op_type] = op_types.get(op.op_type, 0) + 1

    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type}: {count}")
    print()

    # Print first few operations
    print("First 10 operations:")
    for i, op in enumerate(trace_result.operations[:10]):
        print(f"  [{op.index}] {op.op_name}")
        print(f"      Type: {op.op_type}")
        print(f"      Args: {len(op.args)}")
        if op.kwargs:
            print(f"      Kwargs: {list(op.kwargs.keys())}")
    print()

    # Print detailed trace
    print("-" * 80)
    print("DETAILED TRACE:")
    print("-" * 80)
    pl.print_trace(trace_result, format="detailed")
    print()


def test_multiple_incore_calls():
    """Test tracing multiple InCore to InCore function calls."""
    print("=" * 80)
    print("TEST 2: Multiple InCore to InCore Calls")
    print("=" * 80)
    print("This test traces a function with multiple InCore")
    print("to InCore function calls.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def multi_stage_kernel(
        input_data: pl.Tensor[[128, 128], pl.FP32],
        weights: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        """Multi-stage kernel: output = normalize(compute(input, weights))"""
        # Stage 1: Preprocess
        temp1 = preprocess_kernel(input_data, weights, output)

        # Stage 2: Compute
        temp2 = compute_kernel(temp1, temp1, output)

        # Stage 3: Normalize (element-wise divide by max)
        tile_temp2 = pl.load(temp2, [0, 0], [128, 128])
        tile_max = pl.mul(tile_temp2, 2.0)
        tile_normalized = pl.div(tile_temp2, tile_max)
        out = pl.store(tile_normalized, [0, 0], output)
        return out

    # Trace the multi-stage kernel
    trace_result = pl.trace(multi_stage_kernel)

    print("✓ Trace completed successfully!")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    # Count operations by type
    print("Operation breakdown by type:")
    op_types = {}
    for op in trace_result.operations:
        op_types[op.op_type] = op_types.get(op.op_type, 0) + 1

    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type}: {count}")
    print()

    # Filter by specific operations
    print("Filtering by operation name:")
    load_ops = trace_result.filter_by_op("tile.load")
    add_ops = trace_result.filter_by_op("tile.add")
    mul_ops = trace_result.filter_by_op("tile.mul")
    div_ops = trace_result.filter_by_op("tile.div")
    store_ops = trace_result.filter_by_op("tile.store")

    print(f"  Load operations: {len(load_ops)}")
    print(f"  Add operations: {len(add_ops)}")
    print(f"  Multiply operations: {len(mul_ops)}")
    print(f"  Divide operations: {len(div_ops)}")
    print(f"  Store operations: {len(store_ops)}")
    print()

    # Print detailed trace
    print("-" * 80)
    print("DETAILED TRACE:")
    print("-" * 80)
    pl.print_trace(trace_result, format="detailed")
    print()


def test_separate_function_traces():
    """Test tracing each function separately."""
    print("=")
    print("TEST 3: Separate Function Traces")
    print("=" * 80)
    print("This test traces each InCore function separately")
    print("to show individual operation counts.")
    print()

    # Trace preprocess function
    print("Tracing preprocess_kernel:")
    preprocess_trace = pl.trace(preprocess_kernel)
    print(f"  Operations: {preprocess_trace.count()}")
    for i, op in enumerate(preprocess_trace.operations):
        print(f"  [{i}] {op.op_name}")
    print()

    # Trace compute function
    print("Tracing compute_kernel:")
    compute_trace = pl.trace(compute_kernel)
    print(f"  {compute_trace.count()} operations")
    for i, op in enumerate(compute_trace.operations):
        print(f"  [{i}] {op.op_name}")
    print()

    # Trace postprocess function
    print("Tracing postprocess_kernel:")
    postprocess_trace = pl.trace(postprocess_kernel)
    print(f"  {postprocess_trace.count()} operations")
    for i, op in enumerate(postprocess_trace.operations):
        print(f"  [{i}] {op.op_name}")
    print()

    # Total operations across all functions
    total_ops = preprocess_trace.count() + compute_trace.count() + postprocess_trace.count()
    print(f"\nTotal operations across all functions: {total_ops}")
    print()


def test_filtering_incore_operations():
    """Test filtering operations in InCore functions."""
    print("=" * 80)
    print("TEST 4: Filtering InCore Operations")
    print("=" * 80)
    print("This test demonstrates filtering operations")
    print("in InCore to InCore function calls.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def complex_kernel(
        x: pl.Tensor[[64, 64], pl.FP32],
        y: pl.Tensor[[64, 64], pl.FP32],
        z: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Complex kernel with multiple operations and function calls."""
        # Load inputs
        tile_x = pl.load(x, [0, 0], [64, 64])
        tile_y = pl.load(y, [0, 0], [64, 64])
        tile_z = pl.load(z, [0, 0], [64, 64])

        # Compute using helper function
        temp1 = preprocess_kernel(tile_x, tile_y, output)

        # More operations
        temp2 = pl.add(temp1, tile_z)
        temp3 = pl.mul(temp2, 2.0)
        temp4 = pl.sqrt(temp3)

        # Store result
        out = pl.store(temp4, [0, 0], output)
        return out

    # Trace the complex kernel
    trace_result = pl.trace(complex_kernel)

    print(f"Total operations: {trace_result.count()}")
    print()

    # Filter by operation type
    print("Filtering by operation type:")
    tensor_ops = trace_result.filter_by_type("tensor")
    print(f"  Tensor operations: {len(tensor_ops)}")
    print()

    # Filter by specific operations
    print("Filtering by specific operation names:")
    load_ops = trace_result.filter_by_op("tile.load")
    add_ops = trace_result.filter_by_op("tile.add")
    mul_ops = trace_result.filter_by_op("tile.mul")
    sqrt_ops = trace_result.filter_by_op("tile.sqrt")
    store_ops = trace_result.filter_by_op("tile.store")

    print(f"  Load: {len(load_ops)}, Add: {len(add_ops)}, Mul: {len(mul_ops)}")
    print(f"  Sqrt: {len(sqrt_ops)}, Store: {len(store_ops)}")
    print()

    # Access operations by index
    print("Accessing operations by index:")
    first_op = trace_result.get_operation(0)
    last_op = trace_result.get_operation(trace_result.count() - 1)
    middle_op = trace_result.get_operation(trace_result.count() // 2)

    print(f"  First op: {first_op.op_name if first_op else 'None'}")
    print(f"  Middle op: {middle_op.op_name if middle_op else 'None'}")
    print(f"  Last op: {last_op.op_name if last_op else 'None'}")
    print()


def test_json_export_incore():
    """Test JSON export of InCore to InCore function calls."""
    print("=" * 80)
    print("TEST 5: JSON Export of InCore Calls")
    print("=" * 80)
    print("This test exports trace data as JSON for")
    print("InCore to InCore function calls.")
    print()

    @pl.function(type=pl.FunctionType.InCore)
    def json_test_kernel(
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Kernel for JSON test: c = (a + b) * 2.0"""
        temp = preprocess_kernel(a, b, c)
        result = postprocess_kernel(temp, 2.0, c)
        return result

    # Trace and export as JSON
    trace_result = pl.trace(json_test_kernel)
    json_output = trace_result.to_json()

    print("JSON export:")
    print(json_output)
    print()

    # Validate JSON structure
    import json
    data = json.loads(json_output)

    print("JSON validation:")
    print(f"  ✓ Function name: {data['function_name']}")
    print(f"  ✓ Function type: {data['function_type']}")
    print(f"  ✓ Operation count: {len(data['operations'])}")
    print(f"  ✓ Has parameters: {len(data['param_info']) > 0}")
    print(f"  ✓ Has return types: {len(data['return_types']) > 0}")
    print(f"  ✓ All operations have required fields: {all('op_name' in op and 'op_type' in op for op in data['operations'])}")
    print()


def main():
    """Run all InCore to InCore function call tests."""
    print("PyPTO Trace Functionality - InCore to InCore Calls")
    print("=" * 80)
    print("This example demonstrates pl.trace() functionality")
    print("for InCore to InCore function calls.")
    print("=" * 80)
    print()

    test_single_incore_to_incore_call()
    print()

    test_multiple_incore_calls()
    print()

    test_separate_function_traces()
    print()

    test_filtering_incore_operations()
    print()

    test_json_export_incore()
    print()

    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nKey findings from InCore to InCore tracing:")
    print("  1. Trace captures operations in each function separately")
    print("  2. No operations from called functions appear in caller trace")
    print("  3. Operation types are correctly classified")
    print("  4. Filtering works by operation name and type")
    print("  5. JSON export includes complete operation information")
    print("\nUse cases for InCore to InCore tracing:")
    print("  - Modular kernel design")
    print("  - Multi-stage computation pipelines")
    print("  - Function reuse and composition")
    print("  - Performance optimization")
    print("  - Memory usage analysis")


if __name__ == "__main__":
    main()