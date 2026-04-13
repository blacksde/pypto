# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Automatic differentiation API for PyPTO.

This module provides user-facing decorators and functions for automatic
differentiation, including pl.grad(), pl.value_and_grad(), and
@pl.register_grad decorator.
"""

from typing import Any, Callable, List, Tuple, TypeVar
import sys
import os

# Import GradRegistry with fallback for testing
try:
    from pypto.ir.grad_registry import GradRegistry
except (ImportError, ModuleNotFoundError):
    # For testing, import directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ir'))
    import grad_registry as grad_registry_module
    GradRegistry = grad_registry_module.GradRegistry

# Import autodiff_engine with fallback for testing
try:
    from pypto.language.grad.autodiff_engine import get_global_autodiff_engine
except (ImportError, ModuleNotFoundError):
    # For testing, import directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import autodiff_engine as autodiff_engine_module
    get_global_autodiff_engine = autodiff_engine_module.get_global_autodiff_engine

try:
    from pypto.pypto_core.ir import Expr
except (Exception, ImportError, ModuleNotFoundError):
    # For testing without C++ module, use Any as fallback
    Expr = Any

F = TypeVar('F', bound=Callable[..., Expr])


def register_grad(op_name: str) -> Callable[[Callable], Callable]:
    """Decorator to register a gradient function for an operator.
    
    This decorator provides a clean Pythonic API for registering
    gradient computation functions. The decorated function should
    take the operator's inputs and the gradient of the output,
    and return gradients for each input.
    
    Args:
        op_name: Name of the operator (e.g., "tensor.add")
    
    Returns:
        Decorator function
    
    Example:
        @pl.register_grad("tensor.add")
        def grad_add(lhs, rhs, grad_output):
            return grad_output, grad_output
        
        @pl.register_grad("tensor.mul")
        def grad_mul(lhs, rhs, grad_output):
            return mul(grad_output, rhs), mul(lhs, grad_output)
    """
    def decorator(grad_func: Callable) -> Callable:
        GradRegistry.get_instance().register_grad(op_name, grad_func)
        return grad_func
    return decorator


def grad(func: F) -> Callable[..., List[Expr]]:
    """Decorator to create a gradient function.
    
    This decorator transforms a function into a gradient function that
    computes gradients of the function's outputs with respect to its inputs.
    
    Args:
        func: Function to differentiate
    
    Returns:
        Gradient function that takes the same inputs as the original function
        and returns gradients for each input
    
    Example:
    @pl.grad
    def square(x):
        return pl.mul(x, x)
    
    grad_func = pl.grad(square)
    gradients = grad_func(x_value)
    
    # Or use as decorator:
    @pl.grad
    def square_grad(x):
        return pl.mul(x, x)
    
    gradients = square_grad(x_value)
    """
    def wrapper(*args: Any, **kwargs: Any) -> List[Expr]:
        engine = get_global_autodiff_engine()
        
        # Start recording
        engine.start_recording()
        
        # Forward pass
        result = func(*args, **kwargs)
        
        # Stop recording
        engine.stop_recording()
        
        # Create gradient function
        def grad_wrapper(*grad_args: Any, **grad_kwargs: Any) -> List[Expr]:
            # Compute gradients
            from pypto.ir.op.unified_ops import ones_like
            grad_outputs = [ones_like(result)]
            gradients_map = engine.compute_gradients([result], grad_outputs)
            
            # Return gradients for inputs
            return [gradients_map.get(arg, None) for arg in args]
        
        return grad_wrapper(*args, **kwargs)
    
    return wrapper


def value_and_grad(func: F) -> Callable[..., Tuple[Expr, List[Expr]]]:
    """Decorator to create a function that returns both value and gradients.
    
    This decorator transforms a function into a function that computes
    both the function's output and its gradients with respect to inputs.
    This is useful for optimization where both the loss value and
    gradients are needed.
    
    Args:
        func: Function to differentiate
    
    Returns:
        Function that returns a tuple of (value, gradients)
    
    Example:
        @pl.value_and_grad
        def loss_fn(x, y):
            pred = pl.matmul(x, weights)
            return pl.mean(pl.sub(pred, y))
        
        loss, grads = loss_fn(x_value, y_value)
        print(f"Loss: {loss}")
        print(f"Gradients: {grads}")
    """
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Expr, List[Expr]]:
        engine = get_global_autodiff_engine()
        
        # Start recording
        engine.start_recording()
        
        # Forward pass
        result = func(*args, **kwargs)
        
        # Stop recording
        engine.stop_recording()
        
        # Compute gradients
        from pypto.ir.op.unified_ops import ones_like
        grad_outputs = [ones_like(result)]
        gradients_map = engine.compute_gradients([result], grad_outputs)
        
        # Return both value and gradients
        gradients = [gradients_map.get(arg, None) for arg in args]
        return result, gradients
    
    return wrapper


def enable_grad() -> None:
    """Enable gradient recording.
    
    This method starts recording operations in the global
    autodiff engine for gradient computation.
    """
    engine = get_global_autodiff_engine()
    engine.start_recording()


def disable_grad() -> None:
    """Disable gradient recording.
    
    This method stops recording operations in the global
    autodiff engine.
    """
    engine = get_global_autodiff_engine()
    engine.stop_recording()


def is_grad_enabled() -> bool:
    """Check if gradient recording is enabled.
    
    Returns:
        True if gradient recording is enabled, False otherwise
    """
    engine = get_global_autodiff_engine()
    return engine.is_recording
