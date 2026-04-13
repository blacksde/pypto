# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test example for trace-based automatic differentiation.

This example demonstrates the grad_from_trace() functionality
including gradient accumulation for multi-use variables.
"""

import sys
import os

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
python_path = os.path.join(project_root, 'python')

sys.path.insert(0, python_path)
sys.path.insert(0, os.path.join(python_path, 'pypto'))

try:
    import pypto.language as pl
except ImportError as e:
    print(f"Failed to import pypto: {e}")
    print("Please make sure pypto is properly installed and built.")
    sys.exit(1)


def example_basic_gradient():
    """Example 1: Basic gradient generation."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Gradient Generation")
    print("=" * 80)
    print()

    # Register gradient functions
    @pl.register_grad("tensor.add")
    def grad_add(inputs, grad_output, forward_output=None):
        """Gradient for addition."""
        return [grad_output, grad_output]

    @pl.register_grad("tensor.mul")
    def grad_mul(inputs, grad_output, forward_output=None):
        """Gradient for multiplication."""
        a, b = inputs
        return [pl.mul(grad_output, b), pl.mul(a, grad_output)]

    # Define forward function
    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32],
              y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        temp = pl.add(x, y)
        result = pl.mul(temp, temp)
        return result

    print("Forward function defined:")
    print(f"  Name: {forward.name}")
    print(f"  Parameters: {[p.name_hint for p in forward.params]}")
    print()

    # Trace forward function
    trace_result = pl.trace(forward)
    print(f"Trace result: {trace_result.count()} operations")
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name}")
    print()

    # Generate gradient function
    try:
        grad_func = pl.grad_from_trace(forward)
        print("✓ Gradient function generated successfully!")
        print(f"  Name: {grad_func.name}")
        print(f"  Parameters: {[p.name_hint for p in grad_func.params]}")
        print()

    except Exception as e:
        print(f"✗ Failed to generate gradient function: {e}")
        print()


def example_gradient_accumulation():
    """Example 2: Gradient accumulation for multi-use variables."""
    print("=" * 80)
    print("EXAMPLE 2: Gradient Accumulation")
    print("=" * 80)
    print()

    # Register gradient functions
    @pl.register_grad("tensor.add")
    def grad_add(inputs, grad_output, forward_output=None):
        """Gradient for addition."""
        return [grad_output, grad_output]

    # Define forward function with multi-use variables
    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32],
              y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        temp1 = pl.add(x, y)    # x, y used once
        temp2 = pl.add(x, y)    # x, y used again (needs accumulation)
        result = pl.add(temp1, temp2)
        return result

    print("Forward function with multi-use variables:")
    print(f"  Name: {forward.name}")
    print()

    # Trace forward function
    trace_result = pl.trace(forward)
    print(f"Trace result: {trace_result.count()} operations")
    for i, op in enumerate(trace_result.operations):
        print(f"  [{i}] {op.op_name} - inputs: {len(op.args)}")
    print()

    # Analyze variable usage
    from pypto.language.grad.variable_usage_analyzer import VariableUsageAnalyzer

    analyzer = VariableUsageAnalyzer()
    usage_count = analyzer.analyze_trace(trace_result)

    print("Variable usage analysis:")
    for var_name, count in sorted(usage_count.items()):
        needs_acc = " (needs accumulation)" if count > 1 else ""
        print(f"  {var_name}: used {count} time(s){needs_acc}")
    print()

    # Generate gradient function
    try:
        grad_func = pl.grad_from_trace(forward)
        print("✓ Gradient function generated with accumulation support!")
        print(f"  Name: {grad_func.name}")
        print(f"  Parameters: {[p.name_hint for p in grad_func.params]}")
        print()

    except Exception as e:
        print(f"✗ Failed to generate gradient function: {e}")
        print()


def example_gradient_with_forward_output():
    """Example 3: Gradient function using forward output."""
    print("=" * 80)
    print("EXAMPLE 3: Gradient with Forward Output")
    print("=" * 80)
    print()

    # Register gradient functions
    @pl.register_grad("tensor.relu")
    def grad_relu(inputs, grad_output, forward_output=None):
        """Gradient for ReLU - needs forward output."""
        if forward_output is None:
            raise ValueError("ReLU gradient requires forward output")

        # grad = grad_output if x > 0 else 0
        # Simplified: return grad_output (in real implementation, would use forward_output)
        return [grad_output]

    @pl.register_grad("tensor.add")
    def grad_add(inputs, grad_output, forward_output=None):
        """Gradient for addition."""
        return [grad_output, grad_output]

    # Define forward function with ReLU
    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        y = pl.add(x, x)
        result = pl.relu(y)
        return result

    print("Forward function with ReLU:")
    print(f"  Name: {forward.name}")
    print()

    # Generate gradient function with forward outputs
    try:
        grad_func = pl.grad_from_trace(forward, include_forward_outputs=True)
        print("✓ Gradient function generated with forward outputs!")
        print(f"  Name: {grad_func.name}")
        print(f"  Parameters: {[p.name_hint for p in grad_func.params]}")
        print()

    except Exception as e:
        print(f"✗ Failed to generate gradient function: {e}")
        print()


def example_gradient_check():
    """Example 4: Gradient generation with registration check."""
    print("=" * 80)
    print("EXAMPLE 4: Gradient Generation with Check")
    print("=" * 80)
    print()

    # Register gradient functions
    @pl.register_grad("tensor.add")
    def grad_add(inputs, grad_output, forward_output=None):
        """Gradient for addition."""
        return [grad_output, grad_output]

    # Define forward function
    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32],
              y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        result = pl.add(x, y)
        return result

    print("Testing gradient generation with check...")

    # Test 1: All gradients registered
    try:
        grad_func = pl.grad_from_trace_with_check(forward, check_gradients=True)
        print("✓ All gradient functions registered, generation succeeded!")
        print()

    except ValueError as e:
        print(f"✗ Missing gradient functions: {e}")
        print()

    # Test 2: Missing gradient function
    @pl.function
    def forward_mul(x: pl.Tensor[[64, 128], pl.FP32],
                    y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        result = pl.mul(x, y)
        return result

    try:
        grad_func = pl.grad_from_trace_with_check(forward_mul, check_gradients=True)
        print("✗ Should have failed due to missing gradient!")
        print()

    except ValueError as e:
        print(f"✓ Correctly detected missing gradient: {e}")
        print()


def main():
    """Run all examples."""
    print("PyPTO Trace-Based Automatic Differentiation Examples")
    print("=" * 80)
    print("This example demonstrates grad_from_trace() functionality")
    print("=" * 80)
    print()

    try:
        example_basic_gradient()
        example_gradient_accumulation()
        example_gradient_with_forward_output()
        example_gradient_check()

        print("=" * 80)
        print("EXAMPLES COMPLETE")
        print("=" * 80)
        print("\nKey features demonstrated:")
        print("  1. Basic gradient generation from trace")
        print("  2. Gradient accumulation for multi-use variables")
        print("  3. Forward output support for activation functions")
        print("  4. Gradient registration checking")
        print("\nBenefits of trace-based AD:")
        print("  - Compile-time transformation")
        print("  - Static analysis and optimization")
        print("  - Type safety")
        print("  - IR-level integration")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()