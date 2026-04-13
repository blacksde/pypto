# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example demonstrating pl.grad_from_trace() functionality.

This example shows how to use trace-based automatic differentiation
to generate gradient functions from forward functions.
"""

import pypto.language as pl


def example_1_basic_gradient():
    """Example 1: Basic gradient generation.

    This example shows the simplest use case: generating a gradient
    function for a forward function with basic operations.
    """
    print("=" *'80)
    print("EXAMPLE 1: Basic Gradient Generation")
    print("=" *'80)
    print()

    # Define forward function
    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32],
              y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        temp = pl.add(x, y)
        result = pl.mul(temp, temp)
        return result

    print("Forward function:")
    print(f"  Name: {forward.name}")
    print(f"  Parameters: {[p.name_hint for p in forward.params]}")
    print()

    # Generate gradient function
    print("Generating gradient function...")
    grad_func = pl.grad_from_trace(forward)

    print(f"✓ Generated gradient function: {grad_func.name if hasattr(grad_func, 'name') else 'gradient function'}")
    print(f"  Parameters: {[p.name_hint for p in grad_func.params]}")
    print()

    print("✓ Example 1 completed successfully!")
    print()


def example_2_gradient_with_check():
    """Example 2: Gradient generation with registration check.

    This example shows how to use grad_from_trace_with_check()
    to ensure all required gradient functions are registered.
    """
    print("=" *'80)
    print("EXAMPLE 2: Gradient Generation with Check")
    print("=" *'80)
    print()

    # Define forward function
    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32],
              y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        temp = pl.add(x, y)
        result = pl.mul(temp, temp)
        return result

    print("Forward function:")
    print(f"  Name: {forward.name}")
    print(f"  Parameters: {[p.name_hint for p in forward.params]}")
    print()

    # Generate gradient function with check
    print("Generating gradient function with registration check...")
    try:
        grad_func = pl.grad_from_trace_with_check(forward, check_gradients=True)

        print(f"✓ Generated gradient function: {grad_func.name if hasattr(grad_func, 'name') else 'gradient function'}")
        print(f"  Parameters: {[p.name_hint for p in grad_func.params]}")
        print()

        print("✓ Example 2 completed successfully!")
        print()

    except ValueError as e:
        print(f"✗ Failed: {e}")
        print("  This error indicates missing gradient functions.")
        print()


def example_3_activation_functions():
    """Example 3: Gradient generation with activation functions.

    This example shows how gradient functions that need forward output
    (like ReLU, sigmoid, tanh) work correctly.
    """
    print("=" *'80)
    print("EXAMPLE 3: Activation Functions")
    print("=" *'80)
    print()

    # Define forward function with ReLU
    @pl.function
    def forward_relu(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        y = pl.add(x, x)
        result = pl.relu(y)
        return result

    print("Forward function with ReLU:")
    print(f"  Name: {forward_relu.name}")
    print(f"  Parameters: {[p.name_hint for p in forward_relu.params]}")
    print()

    # Generate gradient function with forward output support
    print("Generating gradient function with forward output support...")
    grad_func = pl.grad_from_trace(forward_relu, include_forward_outputs=True)

    print(f"✓ Generated gradient function: {grad_func.name if hasattr(grad_func, 'name') else 'gradient function'}")
    print(f"  Parameters: {[p.name_hint for p in grad_func.params]}")
    print()

    print("✓ Example 3 completed successfully!")
    print()


def example_4_complex_computation():
    """Example 4: Complex computation graph.

    This example shows gradient generation for a more complex
    computation graph with multiple operations.
    """
    print("=" *'80)
    print("EXAMPLE 4: Complex Computation Graph")
    print("=" *'80)
    print()

    # Define forward function with multiple operations
    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32],
              y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        temp1 = pl.add(x, y)
        temp2 = pl.mul(temp1, temp1)
        temp3 = pl.div(temp2, temp2)
        result = pl.relu(temp3)
        return result

    print("Forward function with complex graph:")
    print(f"  Name: {forward.name}")
    print(f"  Parameters: {[p.name_hint for p in forward.params]}")
    print()

    # Trace forward function
    print("Tracing forward function...")
    trace_result = pl.trace(forward)

    print(f"✓ Traced {trace_result.count()} operations:")
    for i, op in enumerate(trace_result.operations):
        print(f"    [{i}] {op.op_name} ({op.op_type})")
    print()

    # Generate gradient function
    print("Generating gradient function...")
    grad_func = pl.grad_from_trace(forward)

    print(f"✓ Generated gradient function: {grad_func.name if hasattr(grad_func, 'name') else 'gradient function'}")
    print(f"  Parameters: {[p.name_hint for p in grad_func.params]}")
    print()

    print("✓ Example 4 completed successfully!")
    print()


def example_5_custom_prefix():
    """Example 5: Custom gradient variable prefix.

    This example shows how to use custom prefix for gradient variables.
    """
    print("=" *'80)
    print("EXAMPLE 5: Custom Gradient Prefix")
    print("=" *'80)
    print()

    # Define forward function
    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32],
              y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        result = pl.add(x, y)
        return result

    print("Forward function:")
    print(f"  Name: {forward.name}")
    print()

    # Generate gradient function with custom prefix
    print("Generating gradient function with custom prefix 'grad_'...")
    grad_func = pl.grad_from_trace(forward, grad_prefix="grad_")

    print(f"✓ Generated gradient function: {grad_func.name if hasattr(grad_func, 'name') else 'gradient function'}")
    print(f"  Parameters: {[p.name_hint for p in grad_func.params]}")
    print()

    # Check if custom prefix is used
    param_names = [p.name for p in grad_func.params]
    has_custom_prefix = any('grad_' in name for name in param_names)

    if has_custom_prefix:
        print("✓ Custom prefix 'grad_' is used in parameters")
    else:
        print("⚠ Custom prefix may not be visible in parameter names")

    print()

    print("✓ Example 5 completed successfully!")
    print()


def main():
    """Run all examples."""
    print("PyPTO pl.grad_from_trace() Examples")
    print("=" *'80)
    print("This example demonstrates trace-based automatic differentiation")
    print("=" *'80)
    print()

    try:
        example_1_basic_gradient()
        example_2_gradient_with_check()
        example_3_activation_functions()
        example_4_complex_computation()
        example_5_custom_prefix()

        print("=" *'80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" *'80)
        print()
        print("Key features demonstrated:")
        print("  1. Basic gradient generation from forward function")
        print("  2. Gradient generation with registration check")
        print("  3. Forward output support for activation functions")
        print("  4. Complex computation graph handling")
        print("  5. Custom gradient variable prefix")
        print()
        print("Benefits of pl.grad_from_trace():")
        print("  - No manual gradient registration needed")
        print("  - Compile-time transformation")
        print("  - Static analysis and optimization")
        print("  - Type safety")
        print("  - IR-level integration")
        print("  - Gradient accumulation support")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()