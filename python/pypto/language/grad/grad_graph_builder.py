# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Backward graph builder for automatic differentiation.

This module builds the backward computation graph by traversing
the forward trace in reverse order and applying gradient functions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pypto.pypto_core.ir import Function, Expr, Var
from pypto.language.grad.trace import TraceResult, TraceInfo
from pypto.ir.grad_registry import GradRegistry
from pypto.language.grad.variable_usage_analyzer import VariableUsageAnalyzer
from pypto.language.grad.grad_accumulator import GradientAccumulator


@dataclass
class BackwardOp:
    """Information about a backward operation.

    Attributes:
        forward_op: Original forward operation
        grad_func: Gradient function to call
        grad_inputs: List of gradient input expressions
        forward_output: Forward output expression (optional)
        output_grads: List of gradient output expressions
    """
    forward_op: TraceInfo
    grad_func: Any
    grad_inputs: List[Expr]
    forward_output: Optional[Expr]
    output_grads: List[Expr]


class BackwardGraphBuilder:
    """Builder for creating backward computation graph.

    This class traverses the forward trace in reverse order and
    constructs the backward computation graph using registered
    gradient functions.
    """

    def __init__(
        self,
        trace_result: TraceResult,
        include_forward_outputs: bool = True
    ):
        """Initialize backward graph builder.

        Args:
            trace_result: TraceResult from pl.trace()
            include_forward_outputs: Whether to pass forward outputs to grad functions
        """
        self.trace = trace_result
        self.include_forward_outputs = include_forward_outputs
        self.backward_ops: List[BackwardOp] = []
        self.usage_analyzer = VariableUsageAnalyzer()
        self.accumulator = GradientAccumulator()

    def build(self) -> List[BackwardOp]:
        """Build backward computation graph.

        Returns:
            List of backward operations in execution order

        Raises:
            ValueError: If gradient function is not registered for an operation
        """
        # Step 1: Analyze variable usage in forward pass
        self.usage_analyzer.analyze_trace(self.trace)

        # Operations that don't need gradients (control flow, tensor creation, etc.)
        skip_ops = {'range', 'tensor.create', 'tensor.yield_'}

        # Step 2: Traverse forward trace in reverse order
        for forward_op in reversed(self.trace.operations):
            if forward_op.op_name in skip_ops:
                continue
            self._process_backward_op(forward_op)

        return self.backward_ops

    def _process_backward_op(self, forward_op: TraceInfo):
        """Process a single forward operation and create backward op.

        Args:
            forward_op: Forward operation to process

        Raises:
            ValueError: If gradient function is not registered
        """
        # Get gradient function for this operation from auto_register
        print(f"Processing forward op: {forward_op.op_name} with args {forward_op.args}")
        grad_func = GradRegistry.get_instance().get_grad(forward_op.op_name)

        if grad_func is None:
            raise ValueError(
                f"No gradient function registered for operation '{forward_op.op_name}'. "
                f"Please register a gradient function in auto_register.py "
                f"or using @pl.register_grad('{forward_op.op_name}')"
            )

        # Get forward output if needed
        forward_output = None
        if self.include_forward_outputs:
            forward_output = self._extract_forward_output(forward_op)

        # Create backward operation
        backward_op = BackwardOp(
            forward_op=forward_op,
            grad_func=grad_func,
            grad_inputs=[],
            forward_output=forward_output,
            output_grads=[]
        )

        # Add to list
        self.backward_ops.append(backward_op)

    def _extract_forward_output(self, forward_op: TraceInfo) -> Optional[Expr]:
        """Extract forward output expression from operation.

        Args:
            forward_op: Forward operation

        Returns:
            Forward output expression if available
        """
        # Try to get output expression from trace info
        if hasattr(forward_op, 'return_expr') and forward_op.return_expr:
            return forward_op.return_expr

        # If not available, try to infer from last argument
        if forward_op.args:
            return forward_op.args[-1]

        return None

    def get_variable_usage(self, var_name: str) -> int:
        """Get usage count for a variable.

        Args:
            var: Name of variable

        Returns:
            Number of times variable is used
        """
        return self.usage_analyzer.get_usage_count(var_name)

    def needs_accumulation(self, var_name: str) -> bool:
        """Check if variable's gradient needs accumulation.

        Args:
            var_name: Name of variable

        Returns:
            True if variable is used more than once
        """
        return self.usage_analyzer.needs_accumulation(var_name)

    def get_accumulator(self) -> GradientAccumulator:
        """Get gradient accumulator.

        Returns:
            GradientAccumulator instance
        """
        return self.accumulator