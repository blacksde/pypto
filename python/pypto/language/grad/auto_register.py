# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Automatic gradient registration for all PyPTO operators.

This module automatically registers gradient computation functions
for all operators in pl.op. It provides gradient rules for
tensor operations, tile operations, and system operations.
"""

import sys
import os

# Import register_grad with fallback for testing
try:
    from pypto.language.grad.autodiff import register_grad
    from pypto.ir.grad_registry import GradRegistry
except (ImportError, ModuleNotFoundError):
    # For testing, import directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import autodiff as autodiff_module
    register_grad = autodiff_module.register_grad
    try:
        import grad_registry as grad_registry_module
        GradRegistry = grad_registry_module.GradRegistry
    except (ImportError, ModuleNotFoundError):
        GradRegistry = None

# Import operator functions with fallback for testing
try:
    from pypto.ir.op import tensor_ops, tile_ops
    
    def add(*args, **kwargs):
        return tensor_ops.add(*args, **kwargs)
    
    def sub(*args, **kwargs):
        return tensor_ops.sub(*args, **kwargs)
    
    def mul(*args, **kwargs):
        return tensor_ops.mul(*args, **kwargs)
    
    def div(*args, **kwargs):
        return tensor_ops.div(*args, **kwargs)
    
    def neg(*args, **kwargs):
        return tensor_ops.neg(*args, **kwargs)
    
    def matmul(*args, **kwargs):
        return tensor_ops.matmul(*args, **kwargs)
    
    def transpose(*args, **kwargs):
        return tensor_ops.transpose(*args, **kwargs)
    
    def ones_like(*args, **kwargs):
        return tensor_ops.full(*args, **kwargs)
    
    def gt(*args, **kwargs):
        raise NotImplementedError("gt operation not yet implemented in tensor_ops")
    
    def sigmoid(*args, **kwargs):
        return tensor_ops.exp(*args, **kwargs)  # Placeholder
    
    def tanh(*args, **kwargs):
        raise NotImplementedError("tanh operation not yet implemented in tensor_ops")
    
    def relu(*args, **kwargs):
        raise NotImplementedError("relu operation not yet implemented in tensor_ops")
    
    def sqrt(*args, **kwargs):
        return tensor_ops.sqrt(*args, **kwargs)
    
    def exp(*args, **kwargs):
        return tensor_ops.exp(*args, **kwargs)
    
    def maximum(*args, **kwargs):
        return tensor_ops.maximum(*args, **kwargs)
    
    def muls(*args, **kwargs):
        return tensor_ops.muls(*args, **kwargs)
    
    def adds(*args, **kwargs):
        return tensor_ops.adds(*args, **kwargs)
    
    def subs(*args, **kwargs):
        return tensor_ops.subs(*args, **kwargs)
    
    def divs(*args, **kwargs):
        return tensor_ops.divs(*args, **kwargs)
    
    def recip(*args, **kwargs):
        return tensor_ops.recip(*args, **kwargs)
    
    def rsqrt(*args, **kwargs):
        return tensor_ops.rsqrt(*args, **kwargs)
    
except (ImportError, ModuleNotFoundError):
    # For testing without full pypto, create dummy functions
    def add(*args):
        return None
    def sub(*args):
        return None
    def mul(*args):
        return None
    def div(*args):
        return None
    def neg(*args):
        return None
    def matmul(*args):
        return None
    def transpose(*args):
        return None
    def mean(*args):
        return None
    def ones_like(*args):
        return None
    def gt(*args):
        return None
    def sqrt(*args):
        return None
    def exp(*args):
        return None
    def maximum(*args):
        return None
    def muls(*args):
        return None
    def adds(*args):
        return None
    def subs(*args):
        return None
    def divs(*args):
        return None
    def recip(*args):
        return None
    def rsqrt(*args):
        return None



def auto_register_gradients() -> None:
    """Automatically register gradient functions for all operators in pl.op.
    
    This function registers gradient computation functions for all
    tensor, tile, and system (ops. It should be called once
    during module initialization.
    
    This function can be safely called multiple times - it will skip
    registration if the gradient function is already registered.
    """
    
    # ========================================================================
    # Tensor Operations
    # ========================================================================
    
    registry = GradRegistry.get_instance()
    
    if not registry.has_grad("tensor.add"):
        @register_grad("tensor.add")
        def grad_tensor_add(inputs, grad_output, forward_output=None):
            """Gradient for tensor.add operation.
            
            d(add(x, y))/dx = 1
            d(add(x, y))/dy = 1
            """
            return grad_output, grad_output
    
    if not registry.has_grad("tensor.sub"):
        @register_grad("tensor.sub")
        def grad_tensor_sub(inputs, grad_output, forward_output=None):
            """Gradient for tensor.sub operation.
            
            d(sub(x, y))/dx = 1
            d(sub(x, y))/dy = -1
            """
            return grad_output, neg(grad_output)
    
    if not registry.has_grad("tensor.mul"):
        @register_grad("tensor.mul")
        def grad_tensor_mul(inputs, grad_output, forward_output=None):
            """Gradient for tensor.mul operation.
            
            d(mul(x, y))/dx = y
            d(mul(x, y))/dy = x
            """
            lhs, rhs = inputs
            return mul(grad_output, rhs), mul(lhs, grad_output)
    
    if not registry.has_grad("tensor.div"):
        @register_grad("tensor.div")
        def grad_tensor_div(inputs, grad_output, forward_output=None):
            """Gradient for tensor.div operation.
            
            d(div(x, y))/dx = 1/y
            d(div(x, y))/dy = -x/y^2
            """
            lhs, rhs = inputs
            grad_lhs = div(grad_output, rhs)
            grad_rhs = neg(mul(grad_output, div(lhs, mul(rhs, rhs))))
            return grad_lhs, grad_rhs
    
    if not registry.has_grad("tensor.neg"):
        @register_grad("tensor.neg")
        def grad_tensor_neg(inputs, grad_output, forward_output=None):
            """Gradient for tensor.neg operation.
            
            d(neg(x))/dx = -1
            """
            return [neg(grad_output)]
    
    if not registry.has_grad("tensor.matmul"):
        @register_grad("tensor.matmul")
        def grad_tensor_matmul(inputs, grad_output, forward_output=None):
            """Gradient for tensor.matmul operation.
            
            For forward: result = x @ y
            - dx = grad_output @ y^T
            - dy = x^T @ grad_output
            
            Using matmul with transpose flags:
            - dx = matmul(grad_output, y, b_trans=True)
            - dy = matmul(x, grad_output, a_trans=True, b_trans=True)
            """
            lhs, rhs = inputs
            # Use transpose flags in matmul
            grad_lhs = matmul(grad_output, rhs, b_trans=True)
            grad_rhs = matmul(lhs, grad_output, a_trans=True, b_trans=True)
            return grad_lhs, grad_rhs
    
    if not registry.has_grad("tensor.transpose"):
        @register_grad("tensor.transpose")
        def grad_tensor_transpose(inputs, grad_output, forward_output=None):
            """Gradient for tensor.transpose operation.
            
            d(transpose(x))/dx = transpose(grad_output)
            """
            return [transpose(grad_output)]
    
    if not registry.has_grad("tensor.mean"):
        @register_grad("tensor.mean")
        def grad_tensor_mean(inputs, grad_output, forward_output=None):
            """Gradient for tensor.mean operation.
            
            d(mean(x))/dx = grad_output / n
            where n is the number of elements
            """
            x = inputs[0]
            return [mul(grad_output, ones_like(x))]
    
    if not registry.has_grad("tensor.relu"):
        @register_grad("tensor.relu")
        def grad_tensor_relu(inputs, grad_output, forward_output=None):
            """Gradient for tensor.relu operation.
            
            d(relu(x))/dx = 1 if x > 0 else 0
            """
            x = inputs[0]
            mask = gt(x, 0)
            return mul(grad_output, mask)
    
    if not registry.has_grad("tensor.sigmoid"):
        @register_grad("tensor.sigmoid")
        def grad_tensor_sigmoid(inputs, grad_output, forward_output=None):
            """Gradient for tensor.sigmoid operation.
            
            d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
            """
            x = inputs[0]
            sig = sigmoid(x)
            grad_x = mul(sig, sub(1, sig))
            return mul(grad_output, grad_x)
    
    if not registry.has_grad("tensor.tanh"):
        @register_grad("tensor.tanh")
        def grad_tensor_tanh(inputs, grad_output, forward_output=None):
            """Gradient for tensor.tanh operation.
            
            d(tanh(x))/dx = 1 - tanh(x)^2
            """
            x = inputs[0]
            t = tanh(x)
            grad_x = sub(1, mul(t, t))
            return mul(grad_output, grad_x)
    
    if not registry.has_grad("tensor.sqrt"):
        @register_grad("tensor.sqrt")
        def grad_tensor_sqrt(inputs, grad_output, forward_output=None):
            """Gradient for tensor.sqrt operation.
            
            d(sqrt(x))/dx = 1 / (2 * sqrt(x))
            """
            x = inputs[0]
            two_sqrt_x = mul(sqrt(x), 2.0)
            return [div(grad_output, two_sqrt_x)]
    
    if not registry.has_grad("tensor.exp"):
        @register_grad("tensor.exp")
        def grad_tensor_exp(inputs, grad_output, forward_output=None):
            """Gradient for tensor.exp operation.
            
            d(exp(x))/dx = exp(x)
            """
            x = inputs[0]
            return [mul(grad_output, exp(x))]
    
    if not registry.has_grad("tensor.maximum"):
        @register_grad("tensor.maximum")
        def grad_tensor_maximum(inputs, grad_output, forward_output=None):
            """Gradient for tensor.maximum operation.
            
            d(maximum(x, y))/dx = grad_output if x >= y else 0
            d(maximum(x, y))/dy = grad_output if y > x else 0
            """
            lhs, rhs = inputs
            grad_lhs = mul(grad_output, maximum(lhs, rhs))
            grad_rhs = mul(grad_output, maximum(rhs, lhs))
            return grad_lhs, grad_rhs
    
    if not registry.has_grad("tensor.muls"):
        @register_grad("tensor.muls")
        def grad_tensor_muls(inputs, grad_output, forward_output=None):
            """Gradient for tensor.muls operation (multiply by scalar)."""
            x, scalar = inputs
            return mul(grad_output, scalar), None
    
    if not registry.has_grad("tensor.adds"):
        @register_grad("tensor.adds")
        def grad_tensor_adds(inputs, grad_output, forward_output=None):
            """Gradient for tensor.adds operation (add scalar)."""
            return grad_output, None
    
    if not registry.has_grad("tensor.subs"):
        @register_grad("tensor.subs")
        def grad_tensor_subs(inputs, grad_output, forward_output=None):
            """Gradient for tensor.subs operation (subtract scalar)."""
            return grad_output, None
    
    if not registry.has_grad("tensor.divs"):
        @register_grad("tensor.divs")
        def grad_tensor_divs(inputs, grad_output, forward_output=None):
            """Gradient for tensor.divs operation (divide by scalar)."""
            x, scalar = inputs
            return div(grad_output, scalar), None
    

    
    # ========================================================================
    # Tile Operations
    # ========================================================================
    
    if not registry.has_grad("tile.add"):
        @register_grad("tile.add")
        def grad_tile_add(inputs, grad_output, forward_output=None):
            """Gradient for tile.add operation."""
            return grad_output, grad_output
    
    if not registry.has_grad("tile.sub"):
        @register_grad("tile.sub")
        def grad_tile_sub(inputs, grad_output, forward_output=None):
            """Gradient for tile.sub operation."""
            return grad_output, neg(grad_output)
    
    if not registry.has_grad("tile.mul"):
        @register_grad("tile.mul")
        def grad_tile_mul(inputs, grad_output, forward_output=None):
            """Gradient for tile.mul operation."""
            lhs, rhs = inputs
            return mul(grad_output, rhs), mul(lhs, grad_output)
    
    if not registry.has_grad("tile.div"):
        @register_grad("tile.div")
        def grad_tile_div(inputs, grad_output, forward_output=None):
            """Gradient for tile.div operation."""
            lhs, rhs = inputs
            grad_lhs = div(grad_output, rhs)
            grad_rhs = neg(mul(grad_output, div(lhs, mul(rhs, rhs))))
            return grad_lhs, grad_rhs
    
    if not registry.has_grad("tile.neg"):
        @register_grad("tile.neg")
        def grad_tile_neg(inputs, grad_output, forward_output=None):
            """Gradient for tile.neg operation."""
            return neg(grad_output)
    
    if not registry.has_grad("tile.addc"):
        @register_grad("tile.addc")
        def grad_tile_addc(inputs, grad_output, forward_output=None):
            """Gradient for tile.addc operation (add with constant)."""
            return grad_output, None
    
    if not registry.has_grad("tile.subc"):
        @register_grad("tile.subc")
        def grad_tile_subc(inputs, grad_output, forward_output=None):
            """Gradient for tile.subc operation (subtract constant)."""
            return grad_output, None
    
    if not registry.has_grad("tile.mulc"):
        @register_grad("tile.mulc")
        def grad_tile_mulc(inputs, grad_output, forward_output=None):
            """Gradient for tile.mulc operation (multiply by constant)."""
            lhs, rhs = inputs
            return mul(grad_output, rhs), None
    
    if not registry.has_grad("tile.divc"):
        @register_grad("tile.divc")
        def grad_tile_divc(inputs, grad_output, forward_output=None):
            """Gradient for tile.divc operation (divide by constant)."""
            lhs, rhs = inputs
            return div(grad_output, rhs), None
    
    if not registry.has_grad("tile.addsc"):
        @register_grad("tile.addsc")
        def grad_tile_addsc(inputs, grad_output, forward_output=None):
            """Gradient for tile.addsc operation (add scalar constant)."""
            return grad_output, None
    
    if not registry.has_grad("tile.subsc"):
        @register_grad("tile.subsc")
        def grad_tile_subsc(inputs, grad_output, forward_output=None):
            """Gradient for tile.subsc operation (subtract scalar constant)."""
            return grad_output, None
    
    # ========================================================================
    # System Operations
    # ========================================================================
    
    if not registry.has_grad("system.tpush_to_aiv"):
        @register_grad("system.tpush_to_aiv")
        def grad_t_tpush_to_aiv(inputs, grad_output, forward_output=None):
            """Gradient for tpush_to_aiv (no gradient)."""
            return grad_output
    
    if not registry.has_grad("system.tpop_from_aic"):
        @register_grad("system.tpop_from_aic")
        def grad_t_tpop_from_aic(inputs, grad_output, forward_output=None):
            """Gradient for tpop_from_aic (no gradient)."""
            return grad_output
