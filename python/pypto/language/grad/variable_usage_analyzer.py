# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Variable usage analyzer for gradient computation.

This module analyzes trace results to determine how many times each
variable is used in the forward pass, which is essential for
determining whether gradient accumulation is needed.
"""

from typing import Dict, Set
from pypto.pypto_core.ir import Var, Call


class VariableUsageAnalyzer:
    """Analyzer for tracking variable usage in forward pass.

    This class analyzes a trace result to count how many times each
    variable is used as an input to operations. This information
    is used to determine whether gradient accumulation is needed.
    """

    def __init__(self):
        """Initialize the variable usage analyzer."""
        self.usage_count: Dict[str, int] = {}
        self.defined_vars: Set[str] = set()

    def analyze_trace(self, trace_result) -> Dict[str, int]:
        """Analyze trace to count variable usage.

        Args:
            trace_result: TraceResult object from pl.trace()

        Returns:
            Dictionary mapping variable names to their usage counts
        """
        self.usage_count = {}
        self.defined_vars = set()

        # Track function parameters as defined variables
        if hasattr(trace_result, 'function') and trace_result.function:
            for param in trace_result.function.params:
                if hasattr(param, 'name_hint'):
                    self.defined_vars.add(param.name_hint)

        for op in trace_result.operations:
            # Track which variables are defined (outputs)
            self._track_defined_variables(op)

            # Count usage of input variables
            self._count_variable_usage(op)

        return self.usage_count

    def _track_defined_variables(self, op):
        """Track variables that are defined by this operation.

        Args:
            op: TraceInfo object representing an operation
        """
        # The output of this operation defines a new variable
        if hasattr(op, 'output_var') and op.output_var:
            var_name = self._extract_var_name(op.output_var)
            if var_name:
                self.defined_vars.add(var_name)

    def _count_variable_usage(self, op):
        """Count how many times each variable is used as input.

        Args:
            op: TraceInfo object representing an operation
        """
        for arg in op.args:
            var_name = self._extract_var_name(arg)
            if var_name and var_name in self.defined_vars:
                self.usage_count[var_name] = self.usage_count.get(var_name, 0) + 1

    def _extract_var_name(self, expr) -> str | None:
        """Extract variable name from expression.

        Args:
            expr: Expression object

        Returns:
            Variable name if expression is a variable, None otherwise
        """
        if isinstance(expr, Var):
            return expr.name_hint
        elif isinstance(expr, Call):
            # For call expressions, we might want to track them too
            if hasattr(expr, 'op') and hasattr(expr.op, 'name'):
                return expr.op.name
        return None

    def needs_accumulation(self, var_name: str) -> bool:
        """Check if a variable's gradient needs accumulation.

        Args:
            var_name: Name of the variable

        Returns:
            True if variable is used more than once, False otherwise
        """
        return self.usage_count.get(var_name, 0) > 1

    def get_usage_count(self, var_name: str) -> int:
        """Get the usage count for a variable.

        Args:
            var_name: Name of the variable

        Returns:
            Number of times the variable is used
        """
        return self.usage_count.get(var_name, 0)
