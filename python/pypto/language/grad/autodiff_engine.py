# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Automatic differentiation engine for PyPTO.

This module provides a core automatic differentiation engine that
implements reverse-mode automatic differentiation (backpropagation)
using eager execution.
"""

from typing import Any, Callable, Dict, List, Optional

try:
    from pypto.ir.grad_registry import GradRegistry
except (ImportError, ModuleNotFoundError):
    # For testing, import directly
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ir'))
    import grad_registry as grad_registry_module
    GradRegistry = grad_registry_module.GradRegistry

try:
    from pypto.pypto_core.ir import Expr
except (Exception, ImportError, ModuleNotFoundError):
    # For testing without C++ module, use Any as fallback
    Expr = Any


class AutodiffEngine:
    """Eager execution automatic differentiation engine.
    
    This engine implements reverse-mode automatic differentiation
    by recording operations during forward pass and computing
    gradients during backward pass.
    """
    
    def __init__(self):
        """Initialize the autodiff engine."""
        self.computation_graph: List[Dict[str, Any]] = []
        self.gradient_map: Dict[Expr, Expr] = {}
        self.is_recording: bool = False
    
    def start_recording(self) -> None:
        """Start recording operations for gradient computation.
        
        This method initializes the computation graph and gradient map
        for a new forward pass.
        """
        self.is_recording = True
        self.computation_graph = []
        self.gradient_map = {}
    
    def stop_recording(self) -> None:
        """Stop recording operations.
        
        This method should be called after the forward pass is complete.
        """
        self.is_recording = False
    
    def record_operation(self, op_name: str, inputs: List[Expr], output: Expr) -> None:
        """Record an operation in the computation graph.
        
        Args:
            op_name: Name of the operation (e.g., "tensor.add")
            inputs: List of input expressions
            output: Output expression
        """
        if self.is_recording:
            self.computation_graph.append({
                'op_name': op_name,
                'inputs': inputs,
                'output': output,
            })
    
    def compute_gradients(
        self,
        targets: List[Expr],
        grad_outputs: List[Expr]
    ) -> Dict[Expr, Expr]:
        """Compute gradients using reverse-mode automatic differentiation.
        
        This method implements the backward pass by traversing the
        computation graph in reverse order and applying the chain rule.
        
        Args:
            targets: List of output expressions to compute gradients for
            grad_outputs: List of initial gradient values (usually ones_like)
        
        Returns:
            Dictionary mapping input expressions to their gradients
        
        Raises:
            ValueError: If no gradient function is registered for an operator
        """
        # Initialize gradient map with target gradients
        for target, grad_output in zip(targets, grad_outputs):
            self.gradient_map[target] = grad_output
        
        # Backward pass: traverse computation graph in reverse
        for op in reversed(self.computation_graph):
            op_name = op['op_name']
            inputs = op['inputs']
            output = op['output']
            
            # Get gradient function
            grad_func = GradRegistry.get_instance().get_grad(op_name)
            if grad_func is None:
                raise ValueError(f"No gradient function registered for {op_name}")
            
            # Get gradient of output
            grad_output = self.gradient_map.get(output)
            if grad_output is None:
                continue
            
            # Call gradient function
            input_grads = grad_func(inputs, grad_output)
            
            # Accumulate gradients for inputs
            for input_expr, input_grad in zip(inputs, input_grads):
                if input_expr in self.gradient_map:
                    # Gradient already exists, accumulate
                    from pypto.ir.op.unified_ops import add
                    self.gradient_map[input_expr] = add(
                        self.gradient_map[input_expr],
                        input_grad
                    )
                else:
                    # First gradient for this input
                    self.gradient_map[input_expr] = input_grad
        
        return self.gradient_map
    
    def get_gradient(self, expr: Expr) -> Optional[Expr]:
        """Get the gradient for a specific expression.
        
        Args:
            expr: Expression to get gradient for
        
        Returns:
            Gradient expression if computed, None otherwise
        """
        return self.gradient_map.get(expr)
    
    def clear(self) -> None:
        """Clear the computation graph and gradient map."""
        self.computation_graph = []
        self.gradient_map = {}


# Global autodiff engine instance
_global_autodiff_engine = AutodiffEngine()


def get_global_autodiff_engine() -> AutodiffEngine:
    """Get the global autodiff engine instance.
    
    Returns:
        The global AutodiffEngine instance
    """
    return _global_autodiff_engine
