"""
内置算子梯度规则

注册所有 tile/tensor/scalar 算子的梯度规则
"""

from typing import List, Dict, Any
from .gradient_registry import register_grad, GradientRegistry


# =============================================================================
# Tile 算子梯度规则
# =============================================================================


@register_grad("tile.add", "tile")
def tile_add_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.add(a, b) 梯度

    数学: y = a + b
    梯度: da = dy, db = dy
    """
    return [d_output, d_output]


@register_grad("tile.sub", "tile")
def tile_sub_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.sub(a, b) 梯度

    数学: y = a - b
    梯度: da = dy, db = -dy
    """
    from pypto import ir

    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()
    dtype = d_output.type.dtype if hasattr(d_output, "type") else ir.DataType.FP32
    neg_d = ir.Neg(d_output, dtype, span)
    return [d_output, neg_d]


@register_grad("tile.mul", "tile")
def tile_mul_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.mul(a, b) 梯度

    数学: y = a * b
    梯度: da = dy * b, db = dy * a

    需要保存前向的 a 和 b
    """
    from pypto import ir

    a, b = saved_inputs
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    # da = d_output * b
    da = ir.Call(ir.Op("tile.mul"), [d_output, b], {}, a.type if hasattr(a, "type") else None, span)
    # db = d_output * a
    db = ir.Call(ir.Op("tile.mul"), [d_output, a], {}, b.type if hasattr(b, "type") else None, span)

    return [da, db]


@register_grad("tile.div", "tile")
def tile_div_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.div(a, b) 梯度

    数学: y = a / b
    梯度: da = dy / b, db = -dy * a / b²
    """
    from pypto import ir

    a, b = saved_inputs
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()
    dtype = d_output.type.dtype if hasattr(d_output, "type") else ir.DataType.FP32

    # da = d_output / b
    da = ir.Call(ir.Op("tile.div"), [d_output, b], {}, a.type if hasattr(a, "type") else None, span)

    # b²
    b_squared = ir.Call(ir.Op("tile.mul"), [b, b], {}, b.type if hasattr(b, "type") else None, span)

    # -d_output * a
    a_times_d = ir.Call(ir.Op("tile.mul"), [a, d_output], {}, a.type if hasattr(a, "type") else None, span)
    neg_a_times_d = ir.Neg(a_times_d, dtype, span)

    # db = -d_output * a / b²
    db = ir.Call(
        ir.Op("tile.div"), [neg_a_times_d, b_squared], {}, b.type if hasattr(b, "type") else None, span
    )

    return [da, db]


@register_grad("tile.matmul", "tile")
def tile_matmul_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.matmul(A, B) 梯度

    数学: C = A @ B
    梯度: dA = dC @ B^T, dB = A^T @ dC
    """
    from pypto import ir

    A, B = saved_inputs
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    # B^T
    B_T = ir.Call(ir.Op("tile.transpose"), [B], {}, B.type if hasattr(B, "type") else None, span)

    # dA = dC @ B^T
    dA = ir.Call(ir.Op("tile.matmul"), [d_output, B_T], {}, A.type if hasattr(A, "type") else None, span)

    # A^T
    A_T = ir.Call(ir.Op("tile.transpose"), [A], {}, A.type if hasattr(A, "type") else None, span)

    # dB = A^T @ dC
    dB = ir.Call(ir.Op("tile.matmul"), [A_T, d_output], {}, B.type if hasattr(B, "type") else None, span)

    return [dA, dB]


@register_grad("tile.transpose", "tile")
def tile_transpose_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.transpose(X) 梯度

    数学: Y = X^T
    梯度: dX = dY^T
    """
    from pypto import ir

    X = saved_inputs[0]
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    dX = ir.Call(ir.Op("tile.transpose"), [d_output], {}, X.type if hasattr(X, "type") else None, span)

    return [dX]


@register_grad("tile.exp", "tile")
def tile_exp_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.exp(x) 梯度

    数学: y = exp(x)
    梯度: dx = dy * exp(x) = dy * y
    """
    from pypto import ir

    x = saved_inputs[0]
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    # 使用前向结果（已保存）
    forward_result = kwargs.get("forward_result")
    if forward_result:
        exp_x = forward_result
    else:
        exp_x = ir.Call(ir.Op("tile.exp"), [x], {}, x.type if hasattr(x, "type") else None, span)

    dx = ir.Call(ir.Op("tile.mul"), [d_output, exp_x], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


@register_grad("tile.log", "tile")
def tile_log_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.log(x) 梯度

    数学: y = log(x)
    梯度: dx = dy / x
    """
    from pypto import ir

    x = saved_inputs[0]
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    dx = ir.Call(ir.Op("tile.div"), [d_output, x], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


@register_grad("tile.sqrt", "tile")
def tile_sqrt_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.sqrt(x) 梯度

    数学: y = sqrt(x)
    梯度: dx = dy / (2 * sqrt(x)) = dy / (2 * y)
    """
    from pypto import ir

    x = saved_inputs[0]
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()
    dtype = x.type.dtype if hasattr(x, "type") else ir.DataType.FP32

    forward_result = kwargs.get("forward_result")
    if forward_result:
        sqrt_x = forward_result
    else:
        sqrt_x = ir.Call(ir.Op("tile.sqrt"), [x], {}, x.type if hasattr(x, "type") else None, span)

    two = ir.ConstFloat(2.0, dtype, span)
    two_sqrt_x = ir.Call(ir.Op("tile.mul"), [two, sqrt_x], {}, x.type if hasattr(x, "type") else None, span)

    dx = ir.Call(ir.Op("tile.div"), [d_output, two_sqrt_x], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


@register_grad("tile.relu", "tile")
def tile_relu_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.relu(x) 梯度

    数学: y = max(0, x)
    梯度: dx = dy * (x > 0 ? 1 : 0)
    """
    from pypto import ir

    x = saved_inputs[0]
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()
    dtype = x.type.dtype if hasattr(x, "type") else ir.DataType.FP32

    zero = ir.ConstFloat(0.0, dtype, span)
    mask = ir.Call(ir.Op("tile.gt"), [x, zero], {}, x.type if hasattr(x, "type") else None, span)
    mask_float = ir.Call(
        ir.Op("tile.cast"), [mask], {"dtype": dtype}, x.type if hasattr(x, "type") else None, span
    )

    dx = ir.Call(ir.Op("tile.mul"), [d_output, mask_float], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


@register_grad("tile.sin", "tile")
def tile_sin_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.sin(x) 梯度

    数学: y = sin(x)
    梯度: dx = dy * cos(x)
    """
    from pypto import ir

    x = saved_inputs[0]
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    cos_x = ir.Call(ir.Op("tile.cos"), [x], {}, x.type if hasattr(x, "type") else None, span)
    dx = ir.Call(ir.Op("tile.mul"), [d_output, cos_x], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


@register_grad("tile.cos", "tile")
def tile_cos_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tile.cos(x) 梯度

    数学: y = cos(x)
    梯度: dx = -dy * sin(x)
    """
    from pypto import ir

    x = saved_inputs[0]
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()
    dtype = x.type.dtype if hasattr(x, "type") else ir.DataType.FP32

    sin_x = ir.Call(ir.Op("tile.sin"), [x], {}, x.type if hasattr(x, "type") else None, span)
    d_times_sin = ir.Call(
        ir.Op("tile.mul"), [d_output, sin_x], {}, x.type if hasattr(x, "type") else None, span
    )
    dx = ir.Neg(d_times_sin, dtype, span)

    return [dx]


# =============================================================================
# Tensor 算子梯度规则（与 Tile 类似）
# =============================================================================


@register_grad("tensor.add", "tensor")
def tensor_add_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    return [d_output, d_output]


@register_grad("tensor.mul", "tensor")
def tensor_mul_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    from pypto import ir

    x, y = saved_inputs
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    dx = ir.Call(ir.Op("tensor.mul"), [d_output, y], {}, x.type if hasattr(x, "type") else None, span)
    dy = ir.Call(ir.Op("tensor.mul"), [d_output, x], {}, y.type if hasattr(y, "type") else None, span)

    return [dx, dy]


@register_grad("tensor.sub", "tensor")
def tensor_sub_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    from pypto import ir

    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()
    dtype = d_output.type.dtype if hasattr(d_output, "type") else ir.DataType.FP32
    neg_d = ir.Neg(d_output, dtype, span)
    return [d_output, neg_d]


@register_grad("tensor.muls", "tensor")
def tensor_muls_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tensor.muls(x, scalar) 梯度

    数学: y = x * scalar
    梯度: dx = dy * scalar
    """
    from pypto import ir

    x, scalar = saved_inputs
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    dx = ir.Call(ir.Op("tensor.mul"), [d_output, scalar], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


@register_grad("tensor.adds", "tensor")
def tensor_adds_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tensor.adds(x, scalar) 梯度

    数学: y = x + scalar
    梯度: dx = dy
    """
    return [d_output]


@register_grad("tensor.subs", "tensor")
def tensor_subs_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tensor.subs(x, scalar) 梯度

    数学: y = x - scalar
    梯度: dx = dy
    """
    return [d_output]


@register_grad("tensor.divs", "tensor")
def tensor_divs_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """
    tensor.divs(x, scalar) 梯度

    数学: y = x / scalar
    梯度: dx = dy / scalar
    """
    from pypto import ir

    x, scalar = saved_inputs
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    dx = ir.Call(ir.Op("tensor.div"), [d_output, scalar], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


@register_grad("tile.muls", "tile")
def tile_muls_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """tile.muls(x, scalar) 梯度"""
    from pypto import ir

    x, scalar = saved_inputs
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    dx = ir.Call(ir.Op("tile.mul"), [d_output, scalar], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


@register_grad("tile.adds", "tile")
def tile_adds_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """tile.adds(x, scalar) 梯度"""
    return [d_output]


@register_grad("tile.subs", "tile")
def tile_subs_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """tile.subs(x, scalar) 梯度"""
    return [d_output]


@register_grad("tile.divs", "tile")
def tile_divs_grad(saved_inputs: List, d_output, kwargs: Dict[str, Any]) -> List:
    """tile.divs(x, scalar) 梯度"""
    from pypto import ir

    x, scalar = saved_inputs
    span = d_output.span if hasattr(d_output, "span") else ir.Span.unknown()

    dx = ir.Call(ir.Op("tile.div"), [d_output, scalar], {}, x.type if hasattr(x, "type") else None, span)

    return [dx]


# =============================================================================
# 注册函数
# =============================================================================


def register_all_builtin_rules():
    """注册所有内置梯度规则"""
    tile_ops = GradientRegistry.list_by_category("tile")
    tensor_ops = GradientRegistry.list_by_category("tensor")

    print(f"[Autodiff] Registered {len(tile_ops)} tile gradient rules")
    print(f"[Autodiff] Registered {len(tensor_ops)} tensor gradient rules")
