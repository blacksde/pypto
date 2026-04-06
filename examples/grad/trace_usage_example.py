# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Simple usage example demonstrating pl.trace() functionality.

This example shows basic usage patterns for the trace API including:
- Tracing simple functions
- Using different output formats
- Filtering operations
- Accessing operation details
- JSON export

Usage:
    python examples/grad/trace_usage_example.py
"""

import pypto.language as pl


def example_basic_trace():
    """Example 1: Basic tracing of a simple function."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Function Tracing")
    print("=" * 80)
    print()

    @pl.function
    def simple_add(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
        result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
        return result

    # Trace the function
    trace_result = pl.trace(simple_add)

    # Print basic information
    print(f"Function name: {trace_result.function_name}")
    print(f"Function type: {trace_result.function_type}")
    print(f"Total operations: {trace_result.count()}")
    print()

    # Print the trace
    print(trace_result)
    print()


def example_multiple_operations():
    """Example 2: Tracing function with multiple operations."""
    print("=" * 80)
    print("EXAMPLE 2: Multiple Operations Tracing")
    print("=" * 80)
    print()

    @pl.function
    def multi_op(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        temp1: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
        temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, temp1)
        temp3: pl.Tensor[[64, 128], pl.FP32] = pl.div(temp2, temp2)
        result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(temp3)
        return result

    # Trace the function
    trace_result = pl.trace(multi_op)


    pl.print_trace(trace_result, format="detailed")
    print(f"Traced function with {trace_result.count()} operations:")
    for i, op in enumerate(trace_result.operations):
        print(f"  {i}: {op.op_name} ({op.op_type})")
    print()


def example_filtering():
    """Example 3: Filtering operations."""
    print("=" * 80)
    print("EXAMPLE 3: Operation Filtering")
    print("=" * 80)
    print()

    @pl.function
    def mixed_ops(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        temp1: pl.Tensor[[64, 64], pl.FP32] = pl.add(x, x)
        temp2: pl.Tensor[[64, 64], pl.FP32] = pl.mul(temp1, temp1)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.add(temp2, temp2)
        return result

    # Trace the function
    trace_result = pl.trace(mixed_ops)

    # Filter by operation name
    add_ops = trace_result.filter_by_op("tensor.add")
    mul_ops = trace_result.filter_by_op("tensor.mul")

    print("Filtering by operation name:")
    print(f"  Add operations: {len(add_ops)}")
    print(f"  Multiply operations: {len(mul_ops)}")
    print()

    # Filter by operation type
    tensor_ops = trace_result.filter_by_type("tensor")

    print("Filtering by operation type:")
    print(f"  Tensor operations: {len(tensor_ops)}")
    print()


def example_detailed_format():
    """Example 4: Using detailed output format."""
    print("=" * 80)
    print("EXAMPLE 4: Detailed Output Format")
    print("=" * 80)
    print()

    @pl.function
    def detailed_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
        result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
        return result

    # Trace the function
    trace_result = pl.trace(detailed_test)

    # Print in detailed format
    print("Detailed trace output:")
    pl.print_trace(trace_result, format="detailed")
    print()


def example_json_export():
    """Example 5: JSON export."""
    print("=" * 80)
    print("EXAMPLE 5: JSON Export")
    print("=" * 80)
    print()

    @pl.function
    def json_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
        result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
        return result

    # Trace the function
    trace_result = pl.trace(json_test)

    # Export as JSON
    json_output = trace_result.to_json()

    print("JSON export:")
    print(json_output)
    print()

    # Parse and validate JSON
    import json
    data = json.loads(json_output)
    print("JSON validation:")
    print(f"  ✓ Function name: {data['function_name']}")
    print(f"  ✓ Operation count: {len(data['operations'])}")
    print(f"  ✓ Valid JSON structure")
    print()


def example_access_by_index():
    """Example 6: Accessing operations by index."""
    print("=" * 80)
    print("EXAMPLE 6: Accessing Operations by Index")
    print("=" * 80)
    print()

    @pl.function
    def index_test(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
        temp: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
        result: pl.Tensor[[64, 128], pl.FP16] = pl.mul.mul(temp, temp)
        return result

    # Trace the function
    trace_result = pl.trace(index_test)

    # Access operations by index
    first_op = trace_result.get_operation(0)
    second_op = trace_result.get_operation(1)
    invalid_op = trace_result.get_operation(999)

    print("Accessing operations by index:")
    print(f"  First operation (index 0): {first_op.op_name if first_op else 'None'}")
    print(f"  Second operation (index 1): {second_op.op_name if second_op else 'None'}")
    print(f"  Invalid operation (index 999): {invalid_op}")
    print()


def main():
    """Run all examples."""
    print("PyPTO Trace Functionality - Usage Examples")
    print("=" * 80)
    print("This example demonstrates basic usage patterns for pl.trace()")
    print("=" * 80)
    print()

    # example_basic_trace()
    example_multiple_operations()
    # example_filtering()
    # example_detailed_format()
    # example_json_export()
    # example_access_by_index()

    print("=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey trace features demonstrated:")
    print("  1. Basic function tracing")
    print("  2. Multiple operations handling")
    print("  3. Operation filtering (by name and type)")
    print("  4. Multiple output formats (text, detailed, JSON)")
    print("  5. Index-based operation access")
    print("  6. Error handling and validation")
    print("\nUse pl.trace() for:")
    print("  - Debugging compilation issues")
    print("  - Analyzing operation patterns")
    print("  - Performance optimization")
    print("  - Memory usage analysis")


if __name__ == "__main__":
    main()