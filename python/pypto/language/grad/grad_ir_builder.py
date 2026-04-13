# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
IR builder utilities for gradient function generation.

This module provides helper functions for constructing IR nodes
when generating gradient functions.
"""

from typing import Any
from pypto.pypto_core.ir import Var, Call, ConstInt, ConstFloat, ConstBool


class IRBuilder:
    """Helper class for building IR nodes."""

    @staticmethod
    def create_var(name: str, var_type) -> Var:
        """Create a variable node.

        Args:
            name: Variable name
            var_type: Variable type

        Returns:
            Var node
        """
        var = Var()
        var.name_hint = name
        var.type = var_type
        return var

    @staticmethod
    def create_call(op_name: str, args: list, kwargs: dict = None) -> Call:
        """Create a call node.

        Args:
            op_name: Name of operation to call
            args: List of argument expressions
            kwargs: Dictionary of keyword arguments

        Returns:
            Call node
        """
        call = Call()
        call.op = IRBuilder._create_op_ref(op_name)
        call.args = args
        call.kwargs = kwargs if kwargs else {}
        return call

    @staticmethod
    def _create_op_ref(op_name: str):
        """Create operation reference.

        Args:
            op_name: Name of operation

        Returns:
            Operation reference object
        """
        op_ref = type('OpRef', (), {})()
        op_ref.name = op_name
        return op_ref

    @staticmethod
    def create_const_int(value: int) -> ConstInt:
        """Create a constant integer node.

        Args:
            value: Integer value

        Returns:
            ConstInt node
        """
        const = ConstInt()
        const.value = value
        return const

    @staticmethod
    def create_const_float(value: float) -> ConstFloat:
        """Create a constant float node.

        Args:
            value: Float value

        Returns:
            ConstFloat node
        """
        const = ConstFloat()
        const.value = value
        return const

    @staticmethod
    def create_const_bool(value: bool) -> ConstBool:
        """Create a constant boolean node.

        Args:
            value: Boolean value

        Returns:
            ConstBool node
        """
        const = ConstBool()
        const.value = value
        return const