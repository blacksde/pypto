# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TOES-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Gradient registry interface for PyPTO.

This module provides a Python interface to C++ gradient registry,
allowing registration and lookup of gradient computation functions.
"""

from typing import Callable, Optional, Any

try:
    from pypto.pypto_core.ir import Expr
except (ImportError, ModuleNotFoundError):
    # For testing without C++ module, use Any as fallback
    Expr = Any


class GradRegistry:
    """Python interface to C++ gradient registry.
    
    This class provides a singleton pattern for accessing the global
    gradient registry, which stores gradient computation functions
    for automatic differentiation.
    """
    
    _instance: Optional['GradRegistry'] = None
    
    @classmethod
    def get_instance(cls) -> 'GradRegistry':
        """Get the singleton instance of the gradient registry.
        
        Returns:
            The global GradRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the gradient registry.
        
        Note: Use get_instance() to access the singleton instance.
        """
        self._grad_registry: dict[str, Callable] = {}
    
    def register_grad(self, op_name: str, grad_func: Callable) -> None:
        """Register a gradient computation function for an operator.
        
        Args:
            op_name: Name of the operator (e.g., "tensor.add")
            grad_func: Gradient computation function that takes inputs and grad_output
                      and returns gradients for each input
        
        Raises:
            ValueError: If a gradient function is already registered for this operator
        """
        if op_name in self._grad_registry:
            raise ValueError(f"Gradient function for operator '{op_name}' is already registered")
        self._grad_registry[op_name] = grad_func
    
    def get_grad(self, op_name: str) -> Optional[Callable]:
        """Get the gradient function for an operator.
        
        Args:
            op_name: Name of the operator
        
        Returns:
            Gradient function if registered, None otherwise
        """
        return self._grad_registry.get(op_name)
    
    def has_grad(self, op_name: str) -> bool:
        """Check if an operator has a registered gradient function.
        
        Args:
            op_name: Name of the operator
        
        Returns:
            True if gradient function is registered, False otherwise
        """
        return op_name in self._grad_registry
    
    def list_registered_ops(self) -> list[str]:
        """Get a list of all operators with registered gradient functions.
        
        Returns:
            List of operator names
        """
        return list(self._grad_registry.keys())
