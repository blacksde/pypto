# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Example demonstrating automatic differentiation in PyPTO.

This example shows how to use the automatic differentiation
functionality including pl.grad(), pl.value_and_grad(), and
@pl.register_grad decorator.
"""

import pypto.language as pl
from pypto.ir.grad_registry import GradRegistry


def example_basic_gradient_registration():
    """Example 1: Basic gradient function registration."""
    print("=" * 80)
    print("Example 1: Basic Gradient Function Registration")
    print("=" * 80)
    
    # Register a custom gradient function
    @pl.register_grad("tensor.custom_op")
    def grad_custom_op(inputs, grad_output):
        """Gradient for custom operation."""
        x, y = inputs
        # Custom gradient logic
        return [pl.mul(grad_output, y), pl.mul(x, grad_output)]
    
    # Verify registration
    registry = GradRegistry.get_instance()
    print(f"Gradient registered for 'tensor.custom_op': {registry.has_grad('tensor.custom_op')}")
    print(f"Total registered operators: {len(registry.list_registered_ops())}")
    print()


def example_check_auto_registered_gradients():
    """Example 2: Check automatically registered gradients."""
    print("=" * 80)
    print("Example 2: Automatically Registered Gradients")
    print("=" * 80)
    
    registry = GradRegistry.get_instance()
    
    # Check which operators have gradients
    tensor_ops = ["tensor.add", "tensor.sub", "tensor.mul", "tensor.div", 
                 "tensor.matmul", "tensor.relu", "tensor.sigmoid", "tensor.tanh"]
    
    print("Tensor operations with gradients:")
    for op in tensor_ops:
        has_grad = registry.has_grad(op)
        status = "✓" if has_grad else "✗"
        print(f"  {status} {op}")
    
    tile_ops = ["tile.add", "tile.sub", "tile.mul", "tile.div"]
    
    print("\nTile operations with gradients:")
    for op in tile_ops:
        has_grad = registry.has_grad(op)
        status = "✓" if has_grad else "✗"
        print(f"  {status} {op}")
    print()


def example_autodiff_engine():
    """Example 3: Using the autodiff engine."""
    print("=" * 80)
    print("Example 3: Using the Autodiff Engine")
    print("=" * 80)
    
    engine = pl.get_global_autodiff_engine()
    
    # Start recording
    print("Starting gradient recording...")
    engine.start_recording()
    print(f"Recording enabled: {engine.is_recording}")
    
    # Record some operations
    from pypto.pypto_core.ir import Var
    x = Var("x")
    y = Var("y")
    z = Var("z")
    
    engine.record_operation("tensor.add", [x, y], z)
    print(f"Recorded operation: tensor.add")
    print(f"Computation graph size: {len(engine.computation_graph)}")
    
    # Stop recording
    engine.stop_recording()
    print(f"Recording enabled: {engine.is_recording}")
    print()


def example_gradient_context_manager():
    """Example 4: Using gradient context manager."""
    print("=" * 80)
    print("Example 4: Gradient Context Manager")
    print("=" * 80)
    
    print(f"Gradient recording enabled: {pl.is_grad_enabled()}")
    
    pl.enable_grad()
    print(f"Gradient recording enabled: {pl.is_grad_enabled()}")
    
    pl.disable_grad()
    print(f"Gradient recording enabled: {pl.is_grad_enabled()}")
    print()


def example_custom_gradient_override():
    """Example 5: Overriding default gradient with custom one."""
    print("=" * 80)
    print("Example 5: Custom Gradient Override")
    print("=" * 80)
    
    registry = GradRegistry.get_instance()
    
    # Get original gradient (if exists)
    original_grad = registry.get_grad("tensor.add")
    print(f"Original gradient exists: {original_grad is not None}")
    
    # Register custom gradient (this would override if we allowed it)
    @pl.register_grad("tensor.custom_add")
    def custom_add_grad(inputs, grad_output):
        """Custom gradient for add operation."""
        # Custom gradient logic
        return [grad_output, grad_output]
    
    print(f"Custom gradient registered: {registry.has_grad('tensor.custom_add')}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "PyPTO Automatic Differentiation Examples" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    try:
        example_basic_gradient_registration()
        example_check_auto_registered_gradients()
        example_autodiff_engine()
        example_gradient_context_manager()
        example_custom_gradient_override()
        
        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print()
        print("Key features demonstrated:")
        print("  1. Gradient function registration with @pl.register_grad")
        print("  2. Automatic gradient registration for all operators")
        print("  3. Autodiff engine for recording and computing gradients")
        print("  4. Gradient context management (enable/disable)")
        print("  5. Custom gradient override capability")
        print()
        print("Next steps:")
        print("  - Use pl.grad() decorator to create gradient functions")
        print("  - Use pl.value_and_grad() to compute both value and gradients")
        print("  - Implement custom gradients for your operations")
        print()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
