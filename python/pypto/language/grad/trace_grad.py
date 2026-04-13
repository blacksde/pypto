# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Trace-based automatic differentiation for PyPTO.

This module provides functionality to generate gradient functions
from pl.function decorated functions using trace-based analysis.

The main API is grad_from_trace(), which takes a forward function
and generates a corresponding gradient function.

Example:
    import pypto.language as pl

    @pl.function
    def forward(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        y = pl.add(x, x)
        z = pl.mul(y, y)
        return z

    # Generate gradient function
    grad_func = pl.grad_from_trace(forward)

    # Register gradient functions for operations
    @pl.register_grad("tensor.add")
    def grad_add(inputs, grad_output, forward_output=None):
        return [grad_output, grad_output]

    @pl.register_grad("tensor.mul")
    def grad_mul(inputs, grad_output, forward_output=None):
        a, b = inputs
        return [pl.mul(grad_output, b), pl.mul(a, grad_output)]
"""

from typing import Optional

from pypto.pypto_core.ir import Function
from pypto.language.grad.trace import trace, TraceResult
from pypto.language.grad.grad_function_generator import GradientFunctionGenerator


def grad_from_trace(
    func: Function,
    include_forward_outputs: bool = True,
    grad_prefix: str = "d"
) -> Function:
    """Generate gradient function from pl.function using trace.

    This function analyzes the forward function's IR, traces all operations,
    and generates a corresponding gradient function that computes gradients
    with respect to all inputs.

    The gradient function will have the following signature:
        grad_forward(
            # Forward inputs
            x: Tensor[...],
            y: Tensor[...],
            # Forward outputs (if include_forward_outputs=True)
            result: Tensor[...],
            # Gradient of output
            dresult: Tensor[...]
        ) -> Tuple[
            Tensor[...],  # dx
            Tensor[...]   # dy
        ]

    Args:
        func: pl.function decorated function to differentiate
        include_forward_outputs: Whether to pass forward outputs to gradient
                              functions. Some operations (e.g., ReLU, sigmoid)
                              need forward output to compute gradient.
                              Default is True.
        grad_prefix: Prefix for gradient variable names.
                     Default is "d" (e.g., "dx", "dy").

    Returns:
        New pl.function that computes gradients for all inputs

    Raises:
        TypeError: If func is not a Function object
        ValueError: If gradient function cannot be generated

    Example:
        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32],
                   y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp = pl.add(x, y)
            result = pl.mul.mul(temp, temp)
            return result

        # Generate gradient function
        grad_func = pl.grad_from_trace(forward)

        # Use gradient function
        # (Note: This is a compile-time transformation)
        # The actual gradient computation happens at runtime
    """
    # Validate input
    if not isinstance(func, Function):
        raise TypeError(
            f"Expected Function object, got {type(func).__name__}. "
            f"Make sure to use @pl.function decorator."
        )

    # Step 1: Trace the forward function
    try:
        trace_result = trace(func)
    except Exception as e:
        raise ValueError(
            f"Failed to trace function '{func.name}': {e}"
        ) from e

    # Step 2: Generate gradient function
    try:
        generator = GradientFunctionGenerator(
            include_forward_outputs=include_forward_outputs,
            grad_prefix=grad_prefix
        )
        grad_func = generator.generate(func, trace_result)
    except Exception as e:
        raise ValueError(
            f"Failed to generate gradient function for '{func.name}': {e}"
        ) from e

    return grad_func


def grad_from_trace_with_check(
    func: Function,
    include_forward_outputs: bool = True,
    grad_prefix: str = "d",
    check_gradients: bool = True
) -> Function:
    """Generate gradient function with optional gradient registration check.

    This is a variant of grad_from_trace() that optionally checks
    if all required gradient functions are registered before generating
    the gradient function.

    Args:
        func: pl.function decorated function to differentiate
        include_forward_outputs: Whether to pass forward outputs to gradient functions
        grad_prefix: Prefix for gradient variable names
        check_gradients: If True, check if all gradient functions are
                         registered before generating. Default is True.

    Returns:
        New pl.function that computes gradients for all inputs

    Raises:
        TypeError: If func is not a Function object
        ValueError: If gradient function cannot be generated or
                   if check_gradients=True and some gradient
                   functions are not registered

    Example:
        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            y = pl.add(x, x)
            return y

        # Check if gradient functions are registered
        grad_func = pl.grad_from_trace_with_check(forward, check_gradients=True)
    """
    # Validate input
    if not isinstance(func, Function):
        raise TypeError(
            f"Expected Function object, got {type(func).__name__}"
        )

    # Trace the forward function
    trace_result = trace(func)

    # Check if all gradient functions are registered
    if check_gradients:
        from pypto.language.grad.grad_registry import GradRegistry

        registry = GradRegistry.get_instance()
        missing_grads = []

        for op in trace_result.operations:
            if not registry.has_grad(op.op_name):
                missing_grads.append(op.op_name)

        if missing_grads:
            raise ValueError(
                f"Gradient functions not registered for operations: {missing_grads}. "
                f"Please register gradient functions using @pl.register_grad() "
                f"for each missing operation."
            )

    # Generate gradient function
    generator = GradientFunctionGenerator(
        include_forward_outputs=include_forward_outputs,
        grad_prefix=grad_prefix
    )
    grad_func = generator.generate(func, trace_result)

    return grad_func


def grad_by_trace(
    func: Function,
    grad_prefix: str = "d"
) -> Function:
    """Generate gradient function with grad_output and forward_output as inputs.

    This function generates a gradient function that takes:
    1. Forward inputs
    2. Forward outputs
    3. grad_output (gradient of the output)
    
    And returns gradients for all forward inputs.

    The gradient function will have the following signature:
        grad_forward(
            # Forward inputs
            x: Tensor[...],
            y: Tensor[...],
            # Forward outputs
            result: Tensor[...],
            # Gradient of output
            dresult: Tensor[...]
        ) -> Tuple[
            Tensor[...],  # dx
            Tensor[...]   # dy
        ]

    Args:
        func: pl.function decorated function to differentiate
        grad_prefix: Prefix for gradient variable names.
                     Default is "d" (e.g., "dx", "dy").

    Returns:
        New pl.function that computes gradients for all inputs

    Raises:
        TypeError: If func is not a Function object
        ValueError: If gradient function cannot be generated or
                   if gradient functions are not registered

    Example:
        @pl.function
        def forward(x: pl.Tensor[[64, 128], pl.FP32],
                   y: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
            temp = pl.add(x, y)
            result = pl.mul(temp, temp)
            return result

        # Generate gradient function with grad_output and forward_output as inputs
        grad_func = pl.grad_by_trace(forward)

        # The gradient function will have signature:
        # grad_func(x: Tensor, y: Tensor, result: Tensor, dresult: Tensor) -> (Tensor, Tensor)
    """
    # Validate input
    if not isinstance(func, Function):
        raise TypeError(
            f"Expected Function object, got {type(func).__name__}. "
            f"Make sure to use @pl.function decorator."
        )

    # Step 1: Trace the forward function
    try:
        trace_result = trace(func)
    except Exception as e:
        raise ValueError(
            f"Failed to trace function '{func.name}': {e}"
        ) from e

    # Step 2: Check if all gradient functions are registered in auto_register
    from pypto.ir.grad_registry import GradRegistry
    registry = GradRegistry.get_instance()
    missing_grads = []

    for op in trace_result.operations:
        if not registry.has_grad(op.op_name):
            missing_grads.append(op.op_name)

    if missing_grads:
        raise ValueError(
            f"Gradient functions not registered for operations: {missing_grads}. "
            f"Please register gradient functions in auto_register.py "
            f"or using @pl.register_grad() for each missing operation."
        )

    # Step 3: Generate gradient function with grad_output and forward_output as inputs
    try:
        generator = GradientFunctionGenerator(
            include_forward_outputs=True,
            grad_prefix=grad_prefix
        )
        grad_func = generator.generate(func, trace_result)
    except Exception as e:
        raise ValueError(
            f"Failed to generate gradient function for '{func.name}': {e}"
        ) from e

    return grad_func


def _calculate_loop_execution_counts(trace_result):
    """Calculate execution counts for operations inside loops.
    
    This function analyzes the trace to determine how many times each operation
    is executed based on loop nesting structure.
    
    Args:
        trace_result: TraceResult from pl.trace()
    
    Returns:
        Dict mapping operation index to execution count
    """
    from pypto.pypto_core.ir import ConstInt
    
    execution_counts = {}
    loop_stack = []  # Stack of (loop_start_idx, loop_count)
    
    for i, op in enumerate(trace_result.operations):
        if op.op_name == 'range':
            # Extract loop count from args
            # args are [start, stop, step]
            loop_count = 1
            if len(op.args) >= 2:
                try:
                    start = op.args[0].value if isinstance(op.args[0], ConstInt) else 0
                    stop = op.args[1].value if isinstance(op.args[1], ConstInt) else 1
                    step = op.args[2].value if len(op.args) >= 3 and isinstance(op.args[2], ConstInt) else 1
                    if step != 0:
                        loop_count = max(0, (stop - start) // step)
                except (AttributeError, IndexError, ZeroDivisionError):
                    loop_count = 1
            
            # Push loop onto stack
            loop_stack.append((i, loop_count))
        elif op.op_name in ('tensor.yield_',):
            # This is a end of a loop iteration
            pass
        else:
            # Calculate execution count for this operation
            # It's the product of all enclosing loop counts
            exec_count = 1
            for loop_start_idx, loop_count in loop_stack:
                exec_count *= loop_count
            
            execution_counts[i] = exec_count
    
    return execution_counts


def grad_by_trace_v2(
    func: Function,
    grad_prefix: str = "d"
) -> Function:
    """Generate gradient function using direct reverse traversal of trace.
    
    This function generates a gradient function by:
    1. Using pl.trace to get trace_result
    2. Reversing trace_result.operations
    3. Getting backward operations from auto_register.py
    4. Reconstructing gradient function similar to reconstruct_from_trace
    
    Args:
        func: pl.function decorated function to differentiate
        grad_prefix: Prefix for gradient variable names
    
    Returns:
        New pl.function that computes gradients for all inputs
    """
    from pypto.ir.grad_registry import GradRegistry
    
    if not isinstance(func, Function):
        raise TypeError(
            f"Expected Function object, got {type(func).__name__}. "
            f"Make sure to use @pl.function decorator."
        )

    trace_result = trace(func)
    
    # Calculate execution counts for operations inside loops
    execution_counts = _calculate_loop_execution_counts(trace_result)
    
    registry = GradRegistry.get_instance()
    missing_grads = []
    
    # Operations that don't need gradients (control flow, tensor creation, etc.)
    skip_ops = {'range', 'tensor.create', 'tensor.yield_'}
    
    for op in trace_result.operations:
        if op.op_name in skip_ops:
            continue
        if not registry.has_grad(op.op_name):
            missing_grads.append(op.op_name)
    
    if missing_grads:
        raise ValueError(
            f"Gradient functions not registered for operations: {missing_grads}. "
            f"Please register gradient functions in auto_register.py."
        )

    backward_operations = []
    
    # Operations that don't need gradients (control flow, tensor creation, etc.)
    skip_ops = {'range', 'tensor.create', 'tensor.yield_'}
    
    for forward_op in reversed(trace_result.operations):
        if forward_op.op_name in skip_ops:
            continue
            
        grad_func_registered = registry.get_grad(forward_op.op_name)
        
        if grad_func_registered is None:
            raise ValueError(
                f"No gradient function registered for '{forward_op.op_name}'"
            )
        
        # Get execution count for this operation
        exec_count = execution_counts.get(forward_op.index, 1)
        
        backward_op_info = {
            'forward_op': forward_op,
            'grad_func': grad_func_registered,
            'execution_count': exec_count,
        }
        backward_operations.append(backward_op_info)
    
    generator = GradientFunctionGenerator(
        include_forward_outputs=True,
        grad_prefix=grad_prefix
    )
    grad_func = generator.generate(func, trace_result, backward_operations)
    
    return grad_func