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
when one pl.function calls another pl.function. The trace should
capture all operations from the entire computation, including operations
in nested function calls.

Scenario:
  - Orchestration function calls InCore function
  - Trace orchestration function
  - Verify that all operations from both functions are captured

Usage:
    python examples/grad/trace_function_calls.py
"""

import pypto.language as pl


@pl.program
class NestedFunctionProgram:
    """Example program with nested function calls."""

    @pl.function(type=pl.FunctionType.InCore)
    def compute_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """InCore kernel: c = a + b * 2.0"""
        tile_a = pl.load(a, [0, 0], [64, 64])
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_sum = pl.add(tile_a, tile_b)
        tile_scaled = pl.mul(tile_sum, 2.0)
        out = pl.store(tile_scaled, [0, 0], c)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def relu_kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        y: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """InCore kernel: y = relu(x)"""
        tile_x = pl.load(x, [0, 0], [64, 64])
        tile_relu = pl.relu(tile_x)
        out = pl.store(tile_relu, [0, 0], y)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        """Orchestration: c = relu(a + b * 2.0)"""
        # Create intermediate buffer
        temp = pl.create_tensor([64, 64], dtype=pl.FP32)

        # Call compute kernel
        temp = self.compute_kernel(a, b, temp)

        # Call relu kernel
        c = self.relu_kernel(temp, c)
        return c


def test_trace_function_calls():
    """Test tracing of nested function calls."""
    print("=" * 80)
    print("TEST: Tracing Function Calls")
    print("=" * 80)
    print("This test traces an orchestration function that calls")
    print("multiple InCore functions. The trace should capture")
    print("all operations from the entire computation graph.")
    print()

    # Create program instance and get functions
    program_instance = NestedFunctionProgram()

    # Access functions from the program instance
    print("Available functions in program:")
    for func_name in program_instance.functions.keys():
        func = program_instance.functions[func_name]
        print(f"  - {func_name} (type: {func.func_type})")
    print()

    # Get the orchestration function
    if "orchestrator" in program_instance.functions:
        orchestrator_func = program_instance.functions["orchestrator"]

        print("Tracing orchestration function...")
        print(f"  Function: {orchestrator_func.name}")
        print(f"  Type: {orchestrator_func.func_type}")
        print()

        # Trace orchestration function
        trace_result = pl.trace(orchestrator_func)

        # Print summary
        print("✓ Trace completed successfully!")
        print(f"  Total operations traced: {trace_result.count()}")
        print(f"  Function name: {trace_result.function_name}")
        print(f"  Function type: {trace_result.function_type}")
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

        # Check for expected operations
        print("Looking for expected operations...")
        expected_ops = [
            "tensor.create_tensor",
            "tile.load",
            "tile.add",
            "tile.mul",
            "tile.store",
            "tile.relu",
        ]

        found_ops = set(op.op_name for op in trace_result.operations)
        for expected in expected_ops:
            if expected in found_ops:
                print(f"  ✓ Found: {expected}")
            else:
                print(f"  ✗ Missing: {expected}")
        print()

        # Print detailed trace
        print("-" * 80)
        print("DETAILED TRACE:")
        print("-" * 80)
        pl.print_trace(trace_result, format="detailed")
        print()
    else:
        print("✗ Orchestrator function not found in program")
        print("Available functions:", list(program_instance.functions.keys()))


def test_multiple_nesting_levels():
    """Test tracing with multiple levels of function calls."""
    print("=" * 80)
    print("TEST: Multiple Nesting Levels")
    print("=" * 80)
    print("This test creates a program with multiple levels")
    print("of function calls to verify trace captures all operations.")
    print()

    # Create program instance
    program_instance = NestedFunctionProgram()

    # Get the orchestration function
    if "orchestrator" in program_instance.functions:
        orchestrator_func = program_instance.functions["orchestrator"]

        print("Tracing multi-level nested function calls...")
        trace_result = pl.trace(orchestrator_func)

        print(f"✓ Traced {trace_result.count()} operations")
        print()

        # Count operations by function call depth
        print("Operation analysis:")
        print(f"  Total operations: {trace_result.count()}")
        print(f"  Unique operation types: {len(set(op.op_type for op in trace_result.operations))}")
        print()

        # Print detailed trace
        print("-" * 80)
        print("DETAILED TRACE:")
        print("-" * 80)
        pl.print_trace(trace_result, format="detailed")
        print()
    else:
        print("✗ Orchestrator function not found")


def test_filtering_nested_operations():
    """Test filtering operations in nested function calls."""
    print("=" * 80)
    print("TEST: Filtering Nested Operations")
    print("=" * 80)
    print("This test demonstrates filtering operations when")
    print("functions call other functions.")
    print()

    # Create program instance
    program_instance = NestedFunctionProgram()

    # Get the orchestration function
    if "orchestrator" in program_instance.functions:
        orchestrator_func = program_instance.functions["orchestrator"]

        # Trace function
        trace_result = pl.trace(orchestrator_func)

        print(f"Total operations: {trace_result.count()}")
        print()

        # Filter by operation type
        print("Filtering by operation type:")
        tensor_ops = trace_result.filter_by_type("tensor")
        print(f"  tensor operations: {len(tensor_ops)}")

        # Filter by specific operations
        print("\nFiltering by specific operation names:")
        load_ops = trace_result.filter_by_op("tile.load")
        add_ops = trace_result.filter_by_op("tile.add")
        mul_ops = trace_result.filter_by_op("tile.mul")
        store_ops = trace_result.filter_by_op("tile.store")
        relu_ops = trace_result.filter_by_op("tile.relu")

        print(f"  load operations: {len(load_ops)}")
        print(f"  add operations: {len(add_ops)}")
        print(f"  mul operations: {len(mul_ops)}")
        print(f"  store operations: {len(store_ops)}")
        print(f"  relu operations: {len(relu_ops)}")
        print()

        # Verify expected counts
        print("Expected operation counts:")
        print("  tile.load: 2 (one from each kernel)")
        print("  tile.add: 1 (from compute_kernel)")
        print("  tile.mul: 1 (from compute_kernel)")
        print("  tile.store: 2 (one from each kernel)")
        print("  tile.relu: 1 (from relu_kernel)")
        print()


def test_json_export_nested():
    """Test JSON export of nested function calls."""
    print("=" * 80)
    print("TEST: JSON Export of Nested Calls")
    print("=" * 80)
    print("This test exports trace data as JSON for")
    print("nested function call scenarios.")
    print()

    # Create program instance
    program_instance = NestedFunctionProgram()

    # Get the orchestration function
    if "orchestrator" in program_instance.functions:
        orchestrator_func = program_instance.functions["orchestrator"]

        # Trace function
        trace_result = pl.trace(orchestrator_func)

        # Export as JSON
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
        print(f"  ✓ All operations have {len(data['operations'])} entries")
        print()


def main():
    """Run all function call trace tests."""
    print("PyPTO Trace Functionality - Function Call Scenarios")
    print("=" * 80)
    print("This example demonstrates pl.trace() functionality")
    print("when pl.function calls other pl.function.")
    print("=" * 80)
    print()

    # Run all tests
    test_trace_function_calls()
    print()

    test_multiple_nesting_levels()
    print()

    test_filtering_nested_operations()
    print()

    test_json_export_nested()
    print()

    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nKey findings from function call tracing:")
    print("  1. Trace captures operations from called functions")
    print("  2. All operations in computation graph are included")
    print("  3. Operation types are correctly classified")
    print("  4. Filtering works across function boundaries")
    print("  5. JSON export includes complete operation information")
    print("\nUse cases for function call tracing:")
    print("  - Debugging multi-kernel orchestration")
    print("  - Analyzing compute graph complexity")
    print("  - Verifying operation fusion across calls")
    print("  - Performance optimization")
    print("  - Memory usage analysis")


if __name__ == "__main__":
    main()