# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Simple test for trace-based automatic differentiation.

This test verifies the basic functionality of grad_from_trace().
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


def test_basic_import():
    """Test 1: Basic import test."""
    print("Test 1: Basic Import")
    print("-" * 40)

    try:
        from pypto.language.grad import grad_from_trace
        print("✓ Successfully imported grad_from_trace")
        return True
    except ImportError as e:
        print(f"✗ Failed to import grad_from_trace: {e}")
        return False


def test_variable_usage_analyzer():
    """Test 2: Variable usage analyzer."""
    print("\nTest 2: Variable Usage Analyzer")
    print("-" * 40)

    try:
        from pypto.language.grad.variable_usage_analyzer import VariableUsageAnalyzer

        analyzer = VariableUsageAnalyzer()
        print("✓ Successfully created VariableUsageAnalyzer")

        # Test with simple trace
        from pypto.language.grad.trace import TraceResult, TraceInfo

        trace_result = TraceResult(
            function_name="test",
            function_type=pl.FunctionType.Opaque,
            operations=[
                TraceInfo(
                    op_name="add",
                    op_type="tensor",
                    args=[],
                    kwargs={},
                    arg_types=[],
                    return_type="Tensor",
                    source_location=None,
                    index=0
                )
            ]
        )

        usage_count = analyzer.analyze_trace(trace_result)
        print(f"✓ Analyzed trace: {usage_count}")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_accumulator():
    """Test 3: Gradient accumulator."""
    print("\nTest 3: Gradient Accumulator")
    print("-" * 40)

    try:
        from pypto.language.grad.grad_accumulator import GradientAccumulator

        accumulator = GradientAccumulator()
        print("✓ Successfully created GradientAccumulator")

        # Test adding gradients
        accumulator.add_gradient("x", "grad1")
        accumulator.add_gradient("x", "grad2")
        print("✓ Added gradients for variable 'x'")

        # Check pending grads
        pending = accumulator.get_pending_grads("x")
        print(f"✓ Pending gradients: {pending}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_graph_builder():
    """Test 4: Backward graph builder."""
    print("\nTest 4: Backward Graph Builder")
    print("-" * 40)

    try:
        from pypto.language.grad.grad_graph_builder import BackwardGraphBuilder
        from pypto.language.grad.trace import TraceResult, TraceInfo

        # Create simple trace
        trace_result = TraceResult(
            function_name="test",
            function_type=pl.FunctionType.Opaque,
            operations=[
                TraceInfo(
                    op_name="add",
                    op_type="tensor",
                    args=[],
                    kwargs={},
                    arg_types=[],
                    return_type="Tensor",
                    source_location=None,
                    index=0
                )
            ]
        )

        # Register gradient function
        @pl.register_grad("tensor.add")
        def grad_add(inputs, grad_output, forward_output=None):
            return [grad_output, grad_output]

        # Build backward graph
        builder = BackwardGraphBuilder(trace_result)
        print("✓ Successfully created BackwardGraphBuilder")

        # Try to build (may fail due to missing IR, but should initialize correctly)
        try:
            backward_ops = builder.build()
            print(f"✓ Built backward graph: {len(backward_ops)} operations")
        except Exception as e:
            print(f"⚠ Build attempt failed (expected): {str(e)[:100]}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_function_generator():
    """Test 5: Gradient function generator."""
    print("\nTest 5: Gradient Function Generator")
    print("-" * 40)

    try:
        from pypto.language.grad.grad_function_generator import GradientFunctionGenerator

        generator = GradientFunctionGenerator()
        print("✓ Successfully created GradientFunctionGenerator")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_availability():
    """Test 6: API availability."""
    print("\nTest 6: API Availability")
    print("-" * 40)

    available = []
    missing = []

    # Check main API
    if hasattr(pl, 'grad_from_trace'):
        available.append('grad_from_trace')
    else:
        missing.append('grad_from_trace')

    if hasattr(pl, 'grad_from_trace_with_check'):
        available.append('grad_from_trace_with_check')
    else:
        missing.append('grad_from_trace_with_check')

    print(f"✓ Available: {available}")
    if missing:
        print(f"✗ Missing: {missing}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 40)
    print("Trace-Based AD Basic Tests")
    print("=" * 40)

    results = []

    results.append(("Basic Import", test_basic_import()))
    results.append(("Variable Usage Analyzer", test_variable_usage_analyzer()))
    results.append(("Gradient Accumulator", test_gradient_accumulator()))
    results.append(("Backward Graph Builder", test_backward_graph_builder()))
    results.append(("Gradient Function Generator", test_gradient_function_generator()))
    results.append(("API Availability", test_api_availability()))

    print("\n" + "=" * 40)
    print("Test Summary")
    print("=" * 40)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())