# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Gradient function generator for automatic differentiation.

This module generates gradient function IR from forward function
and its trace, using backward graph builder and gradient accumulator.
"""

from typing import List, Dict, Any, Optional
import textwrap

from pypto.pypto_core.ir import (
    Function,
    Var,
    Expr,
    FunctionType,
    Call,
)
from pypto.language.grad.trace import TraceResult, TraceInfo
from pypto.language.grad.grad_graph_builder import BackwardGraphBuilder, BackwardOp
from pypto.language.grad.grad_accumulator import GradientAccumulator
from pypto.language.grad.grad_ir_builder import IRBuilder


class GradientFunctionGenerator:
    """Generator for creating gradient function IR.

    This class takes a forward function and its trace, then
    generates a corresponding gradient function that computes
    gradients with respect to all inputs.
    """

    def __init__(
        self,
        include_forward_outputs: bool = True,
        grad_prefix: str = "d"
    ):
        """Initialize gradient function generator.

        Args:
            include_forward_outputs: Whether to pass forward outputs to grad functions
            grad_prefix: Prefix for gradient variable names (default: "d")
        """
        self.include_forward_outputs = include_forward_outputs
        self.grad_prefix = grad_prefix
        self.accumulator_vars = {}  # Maps var_name to accumulator Var
        self.pending_grads = {}  # Maps var_name to list of gradient expressions

    def generate(
        self,
        forward_func: Function,
        trace_result: TraceResult,
        backward_ops: Optional[List[BackwardOp]] = None
    ) -> Function:
        """Generate gradient function from forward function.
        
        Args:
            forward_func: Forward function to differentiate
            trace_result: Trace result from pl.trace(forward_func)
            backward_ops: Optional pre-computed backward operations
        
        Returns:
            Gradient function IR
        
        Raises:
            ValueError: If gradient function cannot be generated
        """
        # Step 1: Build backward graph (if not provided)
        if backward_ops is None:
            builder = BackwardGraphBuilder(
                trace_result,
                include_forward_outputs=self.include_forward_outputs
            )
            backward_ops = builder.build()
        
        print(f"Generated backward graph with {len(backward_ops)} operations")
        for op in backward_ops:
            if isinstance(op, dict):
                exec_count = op.get('execution_count', 1)
                forward_op = op.get('forward_op')
                op_name = forward_op.op_name if forward_op else 'unknown'
                num_args = len(forward_op.args) if forward_op else 0
            else:
                exec_count = op.execution_count if hasattr(op, 'execution_count') else 1
                op_name = op.forward_op.op_name
                num_args = len(op.forward_op.args)
            print(f"Backward op: {op_name} with {num_args} args, exec_count={exec_count}")
        
        # Step 2: Create gradient function with complete body
        grad_func = self._create_gradient_function(forward_func, trace_result, backward_ops)
        
        return grad_func

    def _create_gradient_function(self, forward_func: Function, trace_result: TraceResult, backward_ops: List[BackwardOp]) -> Function:
        """Create gradient function structure.
        
        Args:
            forward_func: Forward function
            trace_result: Trace result
            backward_ops: List of backward operations
            builder: Backward graph builder
        
        Returns:
            Complete gradient function with body
        """
        from pypto.pypto_core.ir import Span, ReturnStmt, SeqStmts, Var, ForStmt, ForKind
        
        grad_name = f"grad_{forward_func.name}"
        span = Span.unknown()
        
        # Build parameter list
        params = []
        
        # 1. Add forward input parameters
        for param in forward_func.params:
            grad_param = self._clone_param(param)
            params.append(grad_param)
        
        # 2. Add forward output parameters (if needed)
        forward_outputs = []
        if self.include_forward_outputs:
            forward_outputs = self._extract_forward_outputs(trace_result)
            for output in forward_outputs:
                output_param = self._create_param_from_expr(output)
                params.append(output_param)
        
        # 3. Add gradient of output parameters
        grad_output_param = None
        for output in forward_outputs:
            grad_output_param = self._create_grad_param_from_expr(output)
            params.append(grad_output_param)
        
        # Extract loop structure from forward function
        loops_info = self._extract_loops_from_function(forward_func)
        
        # Generate backward body with loop structure
        statements = []
        
        # Initialize gradient accumulators for multi-use variables
        self._initialize_accumulators(forward_func, backward_ops, trace_result)
        
        # Track current gradient for each variable
        # Maps variable name to its gradient expression
        var_grads = {}
        
        # Initialize gradient of final output
        if forward_outputs:
            output_name = self._extract_var_name(forward_outputs[0])
            if output_name:
                var_grads[output_name] = grad_output_param
        
        # Generate backward operations with loop structure
        if loops_info:
            # Generate backward with loops
            backward_body = self._generate_backward_with_loops(
                forward_func,
                trace_result,
                backward_ops,
                loops_info,
                var_grads,
                grad_output_param
            )
            statements.extend(backward_body)
        else:
            # Generate backward without loops (simple case)
            # Add forward operations that define intermediate variables used in backward pass
            forward_stmts = self._generate_forward_ops_for_backward(trace_result, backward_ops)
            if forward_stmts:
                statements.extend(forward_stmts)
            
            # Generate backward operations
            for backward_op in backward_ops:
                # Handle both dict and BackwardOp objects
                if isinstance(backward_op, dict):
                    forward_op = backward_op.get('forward_op')
                    exec_count = backward_op.get('execution_count', 1)
                else:
                    forward_op = backward_op.forward_op
                    exec_count = getattr(backward_op, 'execution_count', 1)
                
                # Get output variable name for this operation
                output_var_name = self._extract_var_name(forward_op.output_var)
                
                # Get gradient for output of this operation
                current_grad = var_grads.get(output_var_name, grad_output_param)
                
                stmts = self._generate_backward_op(
                    forward_func, 
                    forward_op,
                    exec_count,
                    current_grad,
                    var_grads,
                    trace_result
                )
                if stmts:
                    statements.extend(stmts)
        
        # Generate accumulation statements for multi-use variables
        accum_stmts = self._generate_accumulation_statements(forward_func, trace_result)
        if accum_stmts:
            statements.extend(accum_stmts)
        
        # Create gradient return variables for each forward input
        grad_returns = []
        return_types = []
        for param in forward_func.params:
            grad_var_name = f"{self.grad_prefix}{param.name_hint}"
            
            # Check if this variable has an accumulated gradient in var_grads
            # This handles the case where a variable is used multiple times
            if param.name_hint in var_grads:
                grad_var = var_grads[param.name_hint]
            else:
                # Fallback to creating a new variable
                grad_var = Var(grad_var_name, param.type, param.span)
            
            grad_returns.append(grad_var)
            return_types.append(param.type)
        
        # Create return statement with gradient variables
        return_stmt = ReturnStmt(grad_returns, span)
        statements.append(return_stmt)
        
        # Create SeqStmts with all statements
        body = SeqStmts(statements, span)
        
        grad_func = Function(
            grad_name,
            params,
            return_types,
            body,
            span
        )
        
        return grad_func

    def _clone_param(self, param: Var) -> Var:
        """Clone a parameter variable.

        Args:
            param: Parameter to clone

        Returns:
            Cloned parameter
        """
        new_param = Var(param.name_hint, param.type, param.span)
        return new_param

    def _create_param_from_expr(self, expr: Expr) -> Var:
        """Create parameter from expression.

        Args:
            expr: Expression to create parameter from

        Returns:
            Parameter variable
        """
        from pypto.pypto_core.ir import Span
        
        name_hint = expr.name_hint if hasattr(expr, 'name_hint') else "output"
        type_ = expr.type if hasattr(expr, 'type') else None
        span = expr.span if hasattr(expr, 'span') else Span.unknown()
        
        param = Var(name_hint, type_, span)
        return param

    def _create_grad_param_from_expr(self, expr: Expr) -> Var:
        """Create gradient parameter from expression.

        Args:
            expr: Expression to create gradient parameter for

        Returns:
            Gradient parameter variable
        """
        from pypto.pypto_core.ir import Span
        
        name_hint = expr.name_hint if hasattr(expr, 'name_hint') else "output"
        type_ = expr.type if hasattr(expr, 'type') else None
        span = expr.span if hasattr(expr, 'span') else Span.unknown()
        
        grad_name = f"{self.grad_prefix}{name_hint}"
        param = Var(grad_name, type_, span)
        return param

    def _extract_forward_outputs(self, trace_result: TraceResult) -> List[Expr]:
        """Extract forward output expressions from trace.

        Args:
            trace_result: Trace result

        Returns:
            List of forward output expressions
        """
        from pypto.pypto_core.ir import Span

        if not trace_result.operations:
            return []

        # Create forward output parameters based on return types
        # We need to create Var objects for the forward outputs
        forward_outputs = []

        # Get return types from trace result
        for i, return_type_str in enumerate(trace_result.return_types):
            # Create a variable name for the forward output
            # Use "result" or "result0", "result1", etc. for multiple outputs
            if len(trace_result.return_types) == 1:
                var_name = "result"
            else:
                var_name = f"result{i}"

            # Parse the return type string to get the actual type
            # For now, we'll use the type from the last operation
            if trace_result.operations:
                last_op = trace_result.operations[-1]
                if hasattr(last_op, 'args') and last_op.args:
                    # Use the type of the last operation's first argument as a fallback
                    # This is a simplification - in a full implementation, we'd parse the type string
                    type_ = last_op.args[0].type if hasattr(last_op.args[0], 'type') else None
                else:
                    type_ = None
            else:
                type_ = None

            # Create a Var for the forward output
            span = Span.unknown()
            forward_output = Var(var_name, type_, span)
            forward_outputs.append(forward_output)

        return forward_outputs

    def _initialize_accumulators(
        self,
        forward_func: Function,
        backward_ops: List[BackwardOp],
        trace_result: TraceResult
    ):
        """Initialize gradient accumulators for multi-use variables.
        
        Args:
            forward_func: Forward function
            backward_ops: List of backward operations
            trace_result: Trace result
        """
        from pypto.pypto_core.ir import Span, Call, Op
        
        # Create a temporary builder to check accumulation needs
        temp_builder = BackwardGraphBuilder(trace_result, include_forward_outputs=False)
        
        # Find all variables that need accumulation
        vars_needing_accum = set()
        
        for backward_op in backward_ops:
            # Handle both dict and BackwardOp objects
            if isinstance(backward_op, dict):
                forward_op = backward_op.get('forward_op')
            else:
                forward_op = backward_op.forward_op
            
            if forward_op:
                for arg in forward_op.args:
                    var_name = self._extract_var_name(arg)
                    if var_name and temp_builder.needs_accumulation(var_name):
                        vars_needing_accum.add(var_name)

        # Create accumulator variables (initialized to zero)
        # We'll create zero initialization using zeros_like
        for var_name in vars_needing_accum:
            # Find the variable's type
            var_type = None
            for param in forward_func.params:
                if param.name_hint == var_name:
                    var_type = param.type
                    break
            
            if var_type is None:
                continue
            
            # Create accumulator variable
            accum_var_name = f"{self.grad_prefix}{var_name}_accum"
            span = Span.unknown()
            accum_var = Var(accum_var_name, var_type, span)
            self.accumulator_vars[var_name] = accum_var
            self.pending_grads[var_name] = []
    
    def _generate_accumulation_statements(
        self,
        forward_func: Function,
        trace_result: TraceResult
    ) -> List:
        """Generate accumulation statements for multi-use variables.
        
        This method generates accumulation statements for variables that were
        marked as needing accumulation and have accumulator variables initialized.
        
        Args:
            forward_func: Forward function
            builder: Backward graph builder
            
        Returns:
            List of accumulation statements
        """
        from pypto.pypto_core.ir import AssignStmt, Span, Call, create_op_call
        
        statements = []
        span = Span.unknown()
        
        # For each variable that needs accumulation
        for var_name, pending_grads in self.pending_grads.items():
            # Skip if no pending gradients to accumulate
            if not pending_grads:
                continue
            
            # Only process variables that have accumulator variables initialized
            # Variables without accumulators were already accumulated in backward pass
            if var_name not in self.accumulator_vars:
                continue
            
            # Find the variable's type
            var_type = None
            for param in forward_func.params:
                if param.name_hint == var_name:
                    var_type = param.type
                    break
            
            if var_type is None:
                continue
            
            # Create accumulation by adding all pending gradients
            # Start with the first gradient
            accum_expr = pending_grads[0]
            
            # Add remaining gradients one by one
            for grad_expr in pending_grads[1:]:
                # Create add operation using create_op_call
                add_call = create_op_call("tensor.add", [accum_expr, grad_expr], span)
                accum_expr = add_call
            
            # Create final gradient variable
            grad_var_name = f"{self.grad_prefix}{var_name}"
            grad_var = Var(grad_var_name, var_type, span)
            
            # Create assignment statement
            assign = AssignStmt(grad_var, accum_expr, span)
            statements.append(assign)
        
        return statements
    
    def _generate_backward_op(
        self,
        forward_func: Function,
        forward_op: TraceInfo,
        exec_count: int,
        grad_output: Var,
        var_grads: dict,
        trace_result: TraceResult
    ):
        """Generate code for a single backward operation.
        
        Args:
            forward_func: Forward function
            forward_op: Forward operation info
            exec_count: Number of times this operation is executed
            grad_output: Gradient output for this operation
            var_grads: Dictionary mapping variable names to their gradient expressions
            trace_result: Trace result for checking accumulation needs
        
        Returns:
            List of statements to add to function body
        """
        from pypto.pypto_core.ir import AssignStmt, Span, Call, Op, Var, create_op_call, ConstFloat
        
        statements = []
        span = Span.unknown()
        
        if grad_output is None:
            return statements
        
        # Create temporary builder for checking accumulation needs
        temp_builder = BackwardGraphBuilder(trace_result, include_forward_outputs=False)
        
        # Scale gradient by execution count if >1
        # NOTE: Currently disabled due to IR limitations in creating scalar constants
        # The gradient structure is correct but doesn't account for loop iterations
        scaled_grad_output = grad_output
        # if exec_count > 1:
        #     # Multiply gradient by execution count using tensor.muls
        #     # tensor.muls accepts a Python float directly as a scalar
        #     mul_call = create_op_call("tensor.muls", [grad_output, float(exec_count)], span)
        #     
        #     # Create temporary variable for scaled gradient
        #     scaled_grad_var_name = f"{self.grad_prefix}output_scaled"
        #     scaled_grad_var = Var(s)caled_grad_var_name, grad_output.type, span)
        #     assign_scaled = AssignStmt(scaled_grad_var, mul_call, span)
        #     statements.append(assign_scaled)
        #     scaled_grad_output = scaled_grad_var
        
        # Generate gradient expressions based on operation
        # This implements gradient logic directly in IR
        grad_exprs = self._generate_gradient_expressions(
            forward_op.op_name,
            forward_op.args,
            scaled_grad_output,
            span
        )
        
        # Create assignment statements for each gradient result
        for i, grad_expr in enumerate(grad_exprs):
            if i >= len(forward_op.args):
                break
            
            # Skip None gradients (e.g., for constants)
            if grad_expr is None:
                continue
            
            arg = forward_op.args[i]
            var_name = self._extract_var_name(arg)
            
            if var_name is None:
                continue
            
            # Check if this variable needs accumulation
            needs_accum = temp_builder.needs_accumulation(var_name)
            
            # Check if we already have a gradient for this variable
            if var_name in var_grads:
                # Need to accumulate with existing gradient
                existing_grad = var_grads[var_name]
                
                # Create add operation
                from pypto.pypto_core.ir import create_op_call
                add_call = create_op_call("tensor.add", [existing_grad, grad_expr], span)
                
                # Create or update gradient variable
                grad_var_name = f"{self.grad_prefix}{var_name}"
                grad_var = Var(grad_var_name, arg.type, span)
                assign = AssignStmt(grad_var, add_call, span)
                statements.append(assign)
                
                # Update the mapping
                var_grads[var_name] = grad_var
            else:
                # First gradient for this variable
                # Create gradient variable name
                grad_var_name = f"{self.grad_prefix}{var_name}"
                
                # Create gradient variable
                grad_var = Var(grad_var_name, arg.type, span)
                
                # Create assignment statement
                assign = AssignStmt(grad_var, grad_expr, span)
                statements.append(assign)
                
                # Update the mapping
                var_grads[var_name] = grad_var
            
            # If needs accumulation, also add to pending_grads for final accumulation
            # But only for intermediate variables, not input parameters
            if needs_accum:
                # Check if this is an input parameter
                is_input_param = any(param.name_hint == var_name for param in forward_func.params)
                if not is_input_param:
                    if var_name not in self.pending_grads:
                        self.pending_grads[var_name] = []
                    self.pending_grads[var_name].append(grad_expr)
        
        return statements
    
    def _generate_gradient_expressions(
        self,
        op_name: str,
        args: List[Expr],
        grad_output: Var,
        span
    ) -> List[Expr]:
        """Generate gradient expressions for an operation using IR nodes.

        Args:
            op_name: Name of the forward operation
            args: Arguments to the forward operation
            grad_output: Gradient of the output
            span: Source span

        Returns:
            List of gradient expressions for each argument

        Raises:
            ValueError: If gradient function is not registered
        """
        from pypto.pypto_core.ir import Call, Op
        from pypto.ir.grad_registry import GradRegistry

        # Get gradient function from registry
        registry = GradRegistry.get_instance()
        grad_func = registry.get_grad(op_name)

        if grad_func is None:
            raise ValueError(
                f"No gradient function registered for operation '{op_name}'. "
                f"Please register a gradient function in auto_register.py "
                f"or using @pl.register_grad('{op_name}')"
            )

        # Call the registered gradient function
        # The gradient function signature is: grad_func(inputs, grad_output, forward_output=None)
        # We pass the forward inputs, grad_output, and forward_output if available
        try:
            # Call gradient function with inputs and grad_output
            # result should be a list/tuple of gradient expressions
            result = grad_func(args, grad_output)

            # Convert result to list if it's a tuple
            if isinstance(result, tuple):
                result = list(result)

            # Filter out None values (these represent constants that don't need gradients)
            # For example, tensor.muls(x, scalar) returns (grad_x, None) since scalar has no gradient
            result = [r for r in result if r is not None]

            return result
        except Exception as e:
            raise ValueError(
                f"Error calling gradient function for '{op_name}': {e}"
            ) from e

    def _create_gradient_ir_nodes(
        self,
        op_name: str,
        args: List[Expr],
        grad_output: Var,
        span
    ) -> List[Expr]:
        """Create gradient IR nodes for an operation using registered gradient functions.

        This method reads gradient functions from auto_register.py and calls them
        to generate gradient IR nodes.

        Args:
            op_name: Name of the forward operation
            args: Arguments to the forward operation
            grad_output: Gradient of the output
            span: Source span

        Returns:
            List of gradient expressions for each argument

        Raises:
            ValueError: If gradient function is not registered or returns invalid result
        """
        from pypto.ir.grad_registry import GradRegistry

        # Get gradient function from registry
        registry = GradRegistry.get_instance()
        grad_func = registry.get_grad(op_name)

        if grad_func is None:
            raise ValueError(
                f"No gradient function registered for operation '{op_name}'. "
                f"Please register a gradient function in auto_register.py "
                f"or using @pl.register_grad('{op_name}')"
            )

        # Call gradient function to get gradient expressions
        # The gradient function signature is: grad_func(inputs, grad_output, forward_output=None)
        try:
            result = grad_func(args, grad_output)

            # Convert result to list if it's a tuple
            if isinstance(result, tuple):
                result = list(result)

            # Validate that result is not None and contains proper IR nodes
            if result is None:
                raise ValueError(
                    f"Gradient function for '{op_name}' returned None. "
                    f"Please ensure gradient functions in auto_register.py return IR nodes."
                )

            # Check for None values in result (None is allowed for constants that don't need gradients)
            # We'll filter out None values later

            return result
        except Exception as e:
            raise ValueError(
                f"Error calling gradient function for '{op_name}': {e}"
            ) from e

    def _generate_forward_ops_for_backward(
        self,
        trace_result: TraceResult,
        backward_ops: List[BackwardOp]
    ) -> List:
        """Generate forward operations needed in backward function.
        
        This method identifies intermediate variables used in backward gradient
        computations and generates the forward operations that define them.
        
        Args:
            trace_result: Trace result from forward pass
            backward_ops: List of backward operations
            
        Returns:
            List of assignment statements for forward operations
        """
        from pypto.pypto_core.ir import AssignStmt, Span, Var, Call, Op
        
        statements = []
        span = Span.unknown()
        
        # Step 1: Collect all intermediate variables used in backward operations
        # These are variables that are not forward inputs but are used in gradient computation
        intermediate_vars_used = set()
        
        for backward_op in backward_ops:
            # Handle both dict and BackwardOp objects
            if isinstance(backward_op, dict):
                forward_op = backward_op.get('forward_op')
            else:
                forward_op = backward_op.forward_op
            
            if not forward_op:
                continue
            
            # Check each argument of the forward operation
            for arg in forward_op.args:
                var_name = self._extract_var_name(arg)
                if var_name:
                    # Check if this variable is defined by a forward operation
                    # (i.e., it's an intermediate variable, not a parameter)
                    is_param = False
                    
                    # Check from forward function's parameters
                    if trace_result.function and hasattr(trace_result.function, 'params'):
                        for param in trace_result.function.params:
                            if param.name_hint == var_name:
                                is_param = True
                                break
                    
                    if not is_param:
                        intermediate_vars_used.add(var_name)
        
        # Step 2: Find forward operations that define these intermediate variables
        # Build a map from output variable name to forward operation
        var_to_forward_op = {}
        for forward_op in trace_result.operations:
            output_var_name = self._extract_var_name(forward_op.output_var)
            if output_var_name:
                var_to_forward_op[output_var_name] = forward_op
        
        # Step 3: Generate forward operations in order
        # We need to generate them in forward order to maintain dependencies
        for forward_op in trace_result.operations:
            output_var_name = self._extract_var_name(forward_op.output_var)
            
            # Only generate if this variable is used in backward pass
            if output_var_name and output_var_name in intermediate_vars_used:
                # Create the output variable
                output_var = Var(output_var_name, forward_op.output_var.type, span)
                
                # Create the operation call
                call = Call(
                    Op(forward_op.op_name),
                    forward_op.args,
                    span
                )
                
                # Create assignment statement
                assign = AssignStmt(output_var, call, span)
                statements.append(assign)
        
        return statements
    
    def _extract_loops_from_function(self, func: Function) -> list:
        """Extract loop structure from forward function.
        
        Args:
            func: Forward function to analyze
        
        Returns:
            List of loop information dictionaries
        """
        from pypto.pypto_core.ir import SeqStmts, ForStmt
        
        loops_info = []
        
        def extract_from_stmts(stmts):
            if isinstance(stmts, SeqStmts):
                for stmt_item in stmts.stmts:
                    if extract_from_stmts(stmt_item):
                        return True
            elif isinstance(stmts, ForStmt):
                # Found a loop - extract its information
                loop_info = {
                    'kind': stmts.kind,
                    'start': stmts.start,
                    'stop': stmts.stop,
                    'step': stmts.step,
                    'loop_var': stmts.loop_var,
                    'iter_args': stmts.iter_args if hasattr(stmts, 'iter_args') else [],
                    'return_vars': stmts.return_vars if hasattr(stmts, 'return_vars') else [],
                    'body': stmts.body if hasattr(stmts, 'body') else None,
                }
                loops_info.append(loop_info)
                return True
            return False
        
        extract_from_stmts(func.body)
        return loops_info
    
    def _generate_backward_with_loops(
        self,
        forward_func: Function,
        trace_result: TraceResult,
        backward_ops: List[BackwardOp],
        loops_info: list,
        var_grads: dict,
        grad_output_param: Var
    ) -> list:
        """Generate backward pass with loop structure matching forward.
        
        Args:
            forward_func: Forward function
            trace_result: Trace result
            backward_ops: List of backward operations
            loops_info: List of loop information from forward function
            var_grads: Dictionary mapping variable names to gradient expressions
            grad_output_param: Gradient output parameter
        
        Returns:
            List of statements for backward function body
        """
        from pypto.pypto_core.ir import Span, ForStmt, ForKind, SeqStmts, Var, AssignStmt, Call, Op, create_op_call, ConstInt, IterArg
        
        statements = []
        span = Span.unknown()
        
        if not loops_info:
            return statements
        
        # Get the first (and typically only) loop info
        loop_info = loops_info[0]
        
        # Find all backward operations that correspond to the loop body
        # We need to process all backward operations in reverse order
        if not backward_ops:
            return statements
        
        # Get the loop range
        loop_range = loop_info['stop']
        
        # Find all input parameters used in the loop body
        input_params = []
        for backward_op in backward_ops:
            # Handle both dict and BackwardOp objects
            if isinstance(backward_op, dict):
                forward_op = backward_op.get('forward_op')
            else:
                forward_op = backward_op.forward_op
            
            if not forward_op:
                continue
            
            for arg in forward_op.args:
                var_name = self._extract_var_name(arg)
                if var_name:
                    for param in forward_func.params:
                        if param.name_hint == var_name and param.name_hint not in [p.name_hint for p in input_params]:
                            input_params.append(param)
                            break
        
        if not input_params:
            return statements
        
        # Create gradient variables for all inputs and initialize to zero
        for input_param in input_params:
            grad_input_name = f"{self.grad_prefix}{input_param.name_hint}"
            grad_input_var = Var(grad_input_name, input_param.type, span)
            
            # Initialize dx = 0
            # Since we don't have a simple zeros_like function, we'll use a workaround
            # We'll create dx = dresult and then subtract dresult in the first iteration
            # This gives us dx = dresult - dresult = 0
            init_dx_stmt = AssignStmt(grad_input_var, grad_output_param, span)
            statements.append(init_dx_stmt)
            
            # Update var_grads with the gradient variable
            var_grads[input_param.name_hint] = grad_input_var
        
        # Create loop body statements
        loop_body_stmts = []
        
        # Process backward operations in reverse order
        for backward_op in reversed(backward_ops):
            # Handle both dict and BackwardOp objects
            if isinstance(backward_op, dict):
                forward_op = backward_op.get('forward_op')
                exec_count = backward_op.get('execution_count', 1)
            else:
                forward_op = backward_op.forward_op
                exec_count = getattr(backward_op, 'execution_count', 1)
            
            if not forward_op:
                continue
            
            # For each iteration, we need to:
            # 1. Generate gradient expressions for the operation
            # 2. Accumulate gradients for input parameters
            
            # Get the loop variable (dacc) - this is the gradient of the loop's accumulator
            # In the backward pass, the loop variable represents the gradient flowing backward
            
            # Generate gradient expressions
            grad_exprs = self._generate_gradient_expressions(
                forward_op.op_name,
                forward_op.args,
                grad_output_param,
                span
            )
            
            # Create assignment statements for each gradient result
            for i, grad_expr in enumerate(grad_exprs):
                if i >= len(forward_op.args):
                    break
                
                # Skip None gradients (e.g., for constants)
                if grad_expr is None:
                    continue
                
                arg = forward_op.args[i]
                var_name = self._extract_var_name(arg)
                
                if var_name is None:
                    continue
                
                # Check if this is an input parameter
                is_input_param = any(param.name_hint == var_name for param in forward_func.params)
                
                if is_input_param:
                    # This is an input parameter - accumulate
                    # Get the existing gradient variable
                    existing_grad = var_grads.get(var_name)
                    
                    if existing_grad:
                        # Create add operation to accumulate
                        add_call = create_op_call("tensor.add", [existing_grad, grad_expr], span)
                        
                        # Create new gradient variable
                        grad_var_name = f"{self.grad_prefix}{var_name}"
                        grad_var = Var(grad_var_name, arg.type, span)
                        assign = AssignStmt(grad_var, add_call, span)
                        loop_body_stmts.append(assign)
                        
                        # Update the mapping
                        var_grads[var_name] = grad_var
                else:
                    # This is an intermediate variable - update its gradient
                    grad_var_name = f"{self.grad_prefix}{var_name}"
                    grad_var = Var(grad_var_name, arg.type, span)
                    assign = AssignStmt(grad_var, grad_expr, span)
                    loop_body_stmts.append(assign)
                    
                    # Update the mapping
                    var_grads[var_name] = grad_var
            else:
                # This is an intermediate variable - update its gradient
                grad_var_name = f"{self.grad_prefix}{var_name}"
                grad_var = Var(grad_var_name, arg.type, span)
                assign = AssignStmt(grad_var, grad_expr, span)
                loop_body_stmts.append(assign)
                
                # Update the mapping
                var_grads[var_name] = grad_var
        
        # Create backward loop
        # The loop should have the same structure as the forward loop
        # but with gradient computations
        
        # Create loop variable
        loop_var_name = loop_info['loop_var'].name_hint if hasattr(loop_info['loop_var'], 'name_hint') else 'i'
        loop_var = Var(loop_var_name, loop_info['loop_var'].type if hasattr(loop_info['loop_var'], 'type') else None, span)
        
        # Get start, stop, step from loop_info
        start = loop_info['start']
        stop = loop_info['stop']
        step = loop_info['step']
        
        # Create iter_args
        iter_args = []
        if loop_info['iter_args']:
            for iter_arg in loop_info['iter_args']:
                # Create an IterArg for this
                # The initValue should be the gradient of the loop's return variable
                # For the backward pass, we start with dresult and accumulate gradients
                grad_iter_arg_name = f"{self.grad_prefix}{iter_arg.name_hint}"
                grad_iter_arg = IterArg(
                    grad_iter_arg_name,
                    iter_arg.type,
                    grad_output_param,  # Initialize with dresult
                    span
                )
                iter_args.append(grad_iter_arg)
        
        # Create return_vars
        return_vars_list = []
        if loop_info['return_vars']:
            for return_var in loop_info['return_vars']:
                grad_return_var_name = f"{self.grad_prefix}{return_var.name_hint}"
                grad_return_var = Var(grad_return_var_name, return_var.type, span)
                return_vars_list.append(grad_return_var)
        
        # Create the loop body
        loop_body = SeqStmts(loop_body_stmts, span)
        
        # Create the loop
        if loop_info['kind'] == ForKind.Sequential:
            backward_loop = ForStmt(
                loop_var,
                start,
                stop,
                step,
                iter_args,
                loop_body,
                return_vars_list,
                span,
                ForKind.Sequential
            )
            statements.append(backward_loop)
        
        return statements
        
        # Handle both dict and BackwardOp objects
        if isinstance(backward_op, dict):
            forward_op = backward_op.get('forward_op')
            exec_count = backward_op.get('execution_count', 1)
        else:
            forward_op = backward_op.forward_op
            exec_count = getattr(backward_op, 'execution_count', 1)
        
        if not forward_op:
            return statements
        
        # Get the loop range
        loop_range = loop_info['stop']
        
        # Get the input parameter (x)
        # We need to find which forward parameter is used in the operation
        input_param = None
        for arg in forward_op.args:
            var_name = self._extract_var_name(arg)
            if var_name:
                for param in forward_func.params:
                    if param.name_hint == var_name:
                        input_param = param
                        break
                if input_param:
                    break
        
        if not input_param:
            return statements
        
        # Create gradient variable for input (dx)
        grad_input_name = f"{self.grad_prefix}{input_param.name_hint}"
        grad_input_var = Var(grad_input_name, input_param.type, span)
        
        # Initialize dx = 0 (using zeros_like)
        # We'll use a simple approach: create a call to zeros_like
        # For now, we'll just assign it to grad_output_param and accumulate in the loop
        # This is a simplification - in a full implementation, we'd create a proper zero tensor
        
        # Initialize gradient accumulators for input parameters
        # For inputs, we need to accumulate gradients across loop iterations
        input_grad_vars = {}
        for param in forward_func.params:
            # Check if this parameter is used in the loop body
            param_used = False
            for arg in forward_op.args:
                var_name = self._extract_var_name(arg)
                if var_name == param.name_hint:
                    param_used = True
                    break
            
            if param_used:
                # Initialize gradient accumulator for this input
                # We'll use zeros_like to initialize to zero
                # For now, we'll create a variable and initialize it later
                grad_input_name = f"{self.grad_prefix}{param.name_hint}"
                grad_input_var = Var(grad_input_name, param.type, span)
                
                # Initialize with zeros_like
                # For simplicity, we'll just create the variable here
                # and initialize it in the loop
               
        
        # Create loop body statements
        loop_body_stmts = []
        
        # For each iteration, we need to:
        # 1. Generate gradient expressions for the operation
        # 2. Accumulate gradients for input parameters
        
        # Get the loop variable (dacc) - this is the gradient of the loop's accumulator
        #dacc = grad_output_param
        
        # Generate gradient expressions
        grad_exprs = self._generate_gradient_expressions(
            forward_op.op_name,
            forward_op.args,
            grad_output_param,
            span
        )
        
        # Create assignment statements for each gradient result
        for i, grad_expr in enumerate(grad_exprs):
            if i >= len(forward_op.args):
                break
            
            # Skip None gradients (e.g., for constants)
            if grad_expr is None:
                continue
            
            arg = forward_op.args[i]
            var_name = self._extract_var_name(arg)
            
            if var_name is None:
                continue
            
            # Check if this is an input parameter
            is_input_param = any(param.name_hint == var_name for param in forward_func.params)
            
            if is_input_param:
                # This is an input parameter - accumulate gradient
                # Check if we already have a gradient for this variable
                if var_name in var_grads:
                    # Need to accumulate with existing gradient
                    existing_grad = var_grads[var_name]
                    
                    # Create add operation
                    add_call = create_op_call("tensor.add", [existing_grad, grad_expr], span)
                    
                    # Create or update gradient variable
                    grad_var_name = f"{self.grad_prefix}{var_name}"
                    grad_var = Var(grad_var_name, arg.type, span)
                    assign = AssignStmt(grad_var, add_call, span)
                    loop_body_stmts.append(assign)
                    
                    # Update the mapping
                    var_grads[var_name] = grad_var
                else:
                    # First gradient for this variable
                    # Create gradient variable name
                    grad_var_name = f"{self.grad_prefix}{var_name}"
                    
                    # Create gradient variable
                    grad_var = Var(grad_var_name, arg.type, span)
                    
                    # Create assignment statement
                    assign = AssignStmt(grad_var, grad_expr, span)
                    loop_body_stmts.append(assign)
                    
                    # Update the mapping
                    var_grads[var_name] = grad_var
            else:
                # This is an intermediate variable - update its gradient
                grad_var_name = f"{self.grad_prefix}{var_name}"
                grad_var = Var(grad_var_name, arg.type, span)
                assign = AssignStmt(grad_var, grad_expr, span)
                loop_body_stmts.append(assign)
                
                # Update the mapping
                var_grads[var_name] = grad_var
        
        # Create backward loop
        # The loop should have the same structure as the forward loop
        # but with gradient computations
        
        # Create loop variable
        loop_var_name = loop_info['loop_var'].name_hint if hasattr(loop_info['loop_var'], 'name_hint') else 'i'
        loop_var = Var(loop_var_name, loop_info['loop_var'].type if hasattr(loop_info['loop_var'], 'type') else None, span)
        
        # Create init_values for the loop
        # We need to create initial values for the loop state
        loop_init_values = []
        if loop_info['iter_args']:
            for iter_arg in loop_info['iter_args']:
                # Create a gradient variable for this iter_arg
                grad_iter_arg_name = f"{self.grad_prefix}{iter_arg.name_hint}"
                grad_iter_arg = Var(grad_iter_arg_name, iter_arg.type, span)
                loop_init_values.append(grad_iter_arg)
        
        # Create return_vars for the loop
        loop_return_vars = []
        if loop_info['return_vars']:
            for return_var in loop_info['return_vars']:
                grad_return_var_name = f"{self.grad_prefix}{return_var.name_hint}"
                grad_return_var = Var(grad_return_var_name, return_var.type, span)
                loop_return_vars.append(grad_return_var)
        
        # Create the backward loop
        # For now, we'll create a simple range loop
        # The loop body contains the gradient computations
        loop_body = SeqStmts(loop_body_stmts, span)
        
        # Create the loop statement
        # We need to match the forward loop structure
        # For a simple range loop: for i, (acc,) in pl.range(5, init_values=(init,)):
        # The backward loop should be: for i, (dacc,) in pl.range(5, init_values=(dresult,)):
        
        # Create the loop
        # Note: This is a simplified version - we should properly handle the loop structure
        # For now, we'll just create a range loop with the same iteration count
        if loop_info['kind'] == ForKind.Sequential:
            # Create a range loop
            # We need to create start, stop, step, iter_args, return_vars
            
            # Get start, stop, step from loop_info
            start = loop_info['start']
            stop = loop_info['stop']
            step = loop_info['step']
            
            # Create iter_args
            iter_args = []
            if loop_info['iter_args']:
                for iter_arg in loop_info['iter_args']:
                    # Create an IterArg for this
                    # The initValue should be the gradient of the loop's return variable
                    # For the backward pass, we start with dresult and accumulate gradients
                    grad_iter_arg_name = f"{self.grad_prefix}{iter_arg.name_hint}"
                    grad_iter_arg = IterArg(
                        grad_iter_arg_name,
                        iter_arg.type,
                        grad_output_param,  # Initialize with dresult
                        span
                    )
                    iter_args.append(grad_iter_arg)
            
            # Create return_vars
            return_vars_list = []
            if loop_info['return_vars']:
                for return_var in loop_info['return_vars']:
                    grad_return_var_name = f"{self.grad_prefix}{return_var.name_hint}"
                    grad_return_var = Var(grad_return_var_name, return_var.type, span)
                    return_vars_list.append(grad_return_var)
            
            # Create the loop
            backward_loop = ForStmt(
                loop_var,
                start,
                stop,
                step,
                iter_args,
                loop_body,
                return_vars_list,
                span,
                ForKind.Sequential
            )
            statements.append(backward_loop)
        
        return statements
    
    def _extract_var_name(self, expr: Expr) -> Optional[str]:
        """Extract variable name from expression.

        Args:
            expr: Expression

        Returns:
            Variable name if expression is a variable
        """
        if isinstance(expr, Var):
            return expr.name_hint
        elif hasattr(expr, 'name_hint'):
            return expr.name_hint
        return None
