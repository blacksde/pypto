# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example demonstrating trace reconstruction functionality.

This example shows how to:
1. Trace a function using pl.trace()
2. Reconstruct the function from trace results
3. Compare original and reconstructed functions
4. Use the reconstructed function for further compilation

Usage:
    python examples/grad/trace_reconstruction_example.py
"""

import pypto.language as pl
from pypto.pypto_core import ir
from pypto.ir import Program


def example_basic_reconstruction():
    """Example 1: Basic function reconstruction."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Function Reconstruction")
    print("=" * 80)
    print()

    @pl.function
    def original_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
        result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
        return result

    print("Original function:")
    print(f"  Name: {original_func.name}")
    print(f"  Type: {original_func.func_type}")
    print(f"  Parameters: {len(original_func.params)}")
    print()

    trace_result = pl.trace(original_func)
    print("Trace result:")
    print(f"  Function name: {trace_result.function_name}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    reconstructed_func = pl.reconstruct_from_trace(trace_result)
    print("Reconstructed function:")
    print(f"  Name: {reconstructed_func.name}")
    print(f"  Type: {reconstructed_func.func_type}")
    print(f"  Parameters: {len(reconstructed_func.params)}")
    print()

    print("Comparing as_python() output:")
    print("-" * 80)
    print("Original function as_python():")
    print(original_func.as_python())
    print()
    print("Reconstructed function as_python():")
    print(reconstructed_func.as_python())
    print("-" * 80)
    print()

    print("✓ Successfully reconstructed function from trace!")
    print()


def example_complex_reconstruction():
    """Example 2: Complex function with multiple operations."""
    print("=" * 80)
    print("EXAMPLE 2: Complex Function Reconstruction")
    print("=" * 80)
    print()

    @pl.function
    def complex_func(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        temp1: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
        temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, temp1)
        temp3: pl.Tensor[[64, 128], pl.FP32] = pl.exp(temp2)
        result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(temp3)
        return result

    print("Original function with multiple operations:")
    print(f"  Name: {complex_func.name}")
    print()

    trace_result = pl.trace(complex_func)
    print("Trace analysis:")
    print(f"  Total operations: {trace_result.count()}")
    for i, op in enumerate(trace_result.operations):
        print(f"    [{i}] {op.op_name} ({op.op_type})")
    print()

    reconstructed_func = pl.reconstruct_from_trace(trace_result)
    print("Reconstructed function:")
    print(f"  Name: {reconstructed_func.name}")
    print(f"  Parameters: {len(reconstructed_func.params)}")
    print()

    print("Comparing as_python() output:")
    print("-" * 80)
    print("Original function as_python():")
    print(complex_func.as_python())
    print()
    print("Reconstructed function as_python():")
    print(reconstructed_func.as_python())
    print("-" * 80)
    print()

    print("✓ Successfully reconstructed complex function!")
    print()


def example_program_reconstruction():
    """Example 3: Reconstructing functions in a program."""
    print("=" * 80)
    print("EXAMPLE 3: Program Reconstruction")
    print("=" * 80)
    print()

    @pl.function
    def func1(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
        result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, x)
        return result

    @pl.function
    def func2(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
        temp: pl.Tensor[[64, 128], pl.FP16] = pl.mul(x, x)
        result: pl.Tensor[[64, 128], pl.FP16] = pl.add(temp, x)
        return result

    print("Creating program with multiple functions...")
    prog = Program([func1, func2], "example_program", ir.Span.unknown())
    print(f"  Program name: {prog.name}")
    print(f"  Number of functions: {len(prog.functions)}")
    print()

    print("Reconstructing each function from trace...")
    original_funcs = [func1, func2]
    reconstructed_funcs = []
    for func in original_funcs:
        trace_result = pl.trace(func)
        reconstructed = pl.reconstruct_from_trace(trace_result)
        reconstructed_funcs.append(reconstructed)
        print(f"  ✓ Reconstructed: {reconstructed.name}")
    print()

    print("Creating new program with reconstructed functions...")
    new_prog = Program(reconstructed_funcs, "reconstructed_program", ir.Span.unknown())
    print(f"  New program name: {new_prog.name}")
    print(f"  Number of functions: {len(new_prog.functions)}")
    print()

    print("Comparing as_python() output:")
    print("-" * 80)
    print("Original program as_python():")
    print(prog.as_python())
    print()
    print("Reconstructed program as_python():")
    print(new_prog.as_python())
    print("-" * 80)
    print()

    print("✓ Successfully reconstructed entire program!")
    print()

    print("✓ Successfully reconstructed entire program!")
    print()


def example_trace_analysis():
    """Example 4: Analyzing trace before reconstruction."""
    print("=" * 80)
    print("EXAMPLE 4: Trace Analysis Before Reconstruction")
    print("=" * 80)
    print()

    @pl.function
    def analysis_func(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        temp1: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
        temp2: pl.Tensor[[64, 128], pl.FP32] = pl.mul(temp1, temp1)
        result: pl.Tensor[[64, 128], pl.FP32] = pl.sqrt(temp2)
        return result

    trace_result = pl.trace(analysis_func)

    print("Detailed trace analysis:")
    print(f"  Function: {trace_result.function_name}")
    print(f"  Function type: {trace_result.function_type}")
    print(f"  Total operations: {trace_result.count()}")
    print()

    print("  Operation breakdown:")
    op_counts = {}
    for op in trace_result.operations:
        op_counts[op.op_type] = op_counts.get(op.op_type, 0) + 1

    for op_type, count in sorted(op_counts.items()):
        print(f"    {op_type}: {count}")
    print()

    print("  Operation details:")
    for op in trace_result.operations:
        print(f"    [{op.index}] {op.op_name}")
        print(f"        Type: {op.op_type}")
        print(f"        Args: {len(op.args)}")
        print(f"        Return type: {op.return_type}")
    print()

    print("Reconstructing function...")
    reconstructed_func = pl.reconstruct_from_trace(trace_result)
    print(f"  ✓ Reconstructed: {reconstructed_func.name}")
    print()

    print("Comparing as_python() output:")
    print("-" * 80)
    print("Original function as_python():")
    print(analysis_func.as_python())
    print()
    print("Reconstructed function as_python():")
    print(reconstructed_func.as_python())
    print("-" * 80)
    print()


def main():
    """Run all examples."""
    print("PyPTO Trace Reconstruction - Usage Examples")
    print("=" * 80)
    print("This example demonstrates trace reconstruction functionality")
    print("=" * 80)
    print()

    example_basic_reconstruction()
    example_complex_reconstruction()
    example_program_reconstruction()
    example_trace_analysis()

    print("=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey reconstruction features demonstrated:")
    print("  1. Basic function reconstruction from trace")
    print("  2. Complex function with multiple operations")
    print("  3. Program-level reconstruction")
    print("  4. Trace analysis before reconstruction")
    print("\nUse pl.reconstruct_from_trace() for:")
    print("  - Function optimization and transformation")
    print("  - Debugging and analysis")
    print("  - Code generation from trace results")
    print("  - Function cloning and modification")


if __name__ == "__main__":
    main()
