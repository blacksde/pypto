# Copyright (c) Py PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Gradient accumulator for automatic differentiation.

This module manages gradient accumulation logic for variables that
are used multiple times in the forward pass.
"""

from typing import Dict, Optional


class GradientAccumulator:
    """Manages gradient accumulation for multiple-use variables.

    This class handles the logic of accumulating gradients from multiple
    backward paths that point to the same variable.
    """

    def __init__(self):
        """Initialize gradient accumulator."""
        self.accumulators = {}
        self.pending_grads = {}

    def get_or_create_accumulator(
        self,
        var_name: str,
        grad_type,
        builder
    ):
        """Get or create a gradient accumulator variable.

        Args:
            var_name: Name of the variable to accumulate gradients for
            grad_type: Type of the gradient variable
            builder: IR builder for creating variables

        Returns:
            Accumulator variable (initialized to zero)
        """
        if var_name not in self.accumulators:
            # Create accumulator variable, initialized to zero
            acc_var = builder.create_var(
                f"d{var_name}_acc",
                grad_type
            )
            self.accumulators[var_name] = acc_var
            self.pending_grads[var_name] = []
        return self.accumulators[var_name]

    def add_gradient(self, var_name: str, grad_expr):
        """Add a gradient expression to the pending list.

        Args:
            var_name: Name of the variable
            grad_expr: Gradient expression to add
        """
        if var_name not in self.pending_grads:
            self.pending_grads[var_name] = []
        self.pending_grads[var_name].append(grad_expr)

    def has_pending_grads(self, var_name: str) -> bool:
        """Check if there are pending gradients for a variable.

        Args:
            var_name: Name of the variable

        Returns:
            True if there are pending gradients, False otherwise
        """
        return var_name in self.pending_grads and len(self.pending_grads[var_name]) > 0

    def get_pending_grads(self, var_name: str) -> list:
        """Get pending gradients for a variable.

        Args:
            var_name: Name of the variable

        Returns:
            List of pending gradient expressions
        """
        return self.pending_grads.get(var_name, [])

    def get_all_accumulators(self):
        """Get all accumulator variables.

        Returns:
            Dictionary mapping variable names to accumulator variables
        """
        return self.accumulators

    def clear(self):
        """Clear all accumulators and pending gradients."""
        self.accumulators = {}
        self.pending_grads = {}