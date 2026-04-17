# PyPTO 自动微分梯度规则注册设计

## 1. 概述

PyPTO 自动微分框架的核心机制是**梯度规则注册**，它定义了每个算子在前向计算时如何生成反向梯度。本文档详细说明：

1. **内置算子梯度规则**：如何注册、存储、查找
2. **算法中的梯度获取**：反向模式变换器如何使用梯度规则
3. **自定义函数梯度**：用户如何扩展框架注册自己的梯度规则

---

## 2. 梯度规则注册架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    梯度规则注册架构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐         ┌──────────────────────────┐          │
│  │ 内置算子梯度规则  │         │ GradientRegistry (全局)   │          │
│  │ grad_rules.py   │ ──────> │                          │          │
│  │                 │         │  registry: Dict[str,     │          │
│  │ @register_grad  │         │    Callable]             │          │
│  │ 'tile.add'      │         │                          │          │
│  │ 'tile.mul'      │         │  ┌────────────────────┐  │          │
│  │ 'tile.matmul'   │         │  │ 'tile.add' → func  │  │          │
│  │ ...             │         │  │ 'tile.mul' → func  │  │          │
│  └─────────────────┘         │  │ 'tile.matmul'→ func │  │          │
│                              │  │ ...                │  │          │
│  ┌─────────────────┐         │  └────────────────────┘  │          │
│  │ 用户自定义梯度    │         │                          │          │
│  │ custom_grad.py  │ ──────> │  Methods:                │          │
│  │                 │         │  - register()            │          │
│  │ CustomGrad()    │         │  - get()                 │          │
│  │ @register_grad  │         │  - has()                 │          │
│  │ 'my.custom_op'  │         │  - list_all()            │          │
│  └─────────────────┘         └──────────────────────────┘          │
│                                     │                              │
│                                     │ get(op_name)                 │
│                                     ▼                              │
│  ┌───────────────────────────────────────────────────────┐         │
│  │          ReverseModeMutator (反向模式变换器)            │         │
│  │                                                        │         │
│  │  _reverse_call(call):                                  │         │
│  │    op_name = call.op.name                              │         │
│  │    grad_rule = GradientRegistry.get(op_name)          │         │
│  │    if grad_rule:                                       │         │
│  │      grad_exprs = grad_rule(saved_inputs, d_out, kw)  │         │
│  │    else:                                               │         │
│  │      raise NotImplementedError                         │         │
│  └───────────────────────────────────────────────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
python/pypto/autodiff/
│
├── __init__.py                    # 高层 API
│   └── 导入 GradientRegistry
│
├── gradient_registry.py           # 注册表核心（无依赖）
│   └── class GradientRegistry
│
├── grad_rules.py                  # 内置规则（依赖 registry）
│   └── 调用 GradientRegistry.register()
│
├── reverse_mode.py                # 反向变换器（依赖 registry）
│   └── 调用 GradientRegistry.get()
│
├── custom_grad.py                 # 自定义梯度（依赖 registry）
│   └── 调用 GradientRegistry.register()
│
└── builtin_grad_register.py       # 模块初始化时注册内置规则
    └── import grad_rules
```

---

## 3. GradientRegistry 设计

### 3.1 核心类定义

```python
# python/pypto/autodiff/gradient_registry.py

"""
梯度规则全局注册表

每个算子的梯度规则是一个 Callable，签名如下：
def grad_rule(saved_inputs: List[ir.Expr], 
              d_output: ir.Expr, 
              kwargs: Dict[str, Any]) -> List[ir.Expr]:
    """
    计算算子输入的梯度
    
    Args:
        saved_inputs: 前向计算时保存的输入值（用于反向计算）
        d_output: 输出的梯度（从下游传播而来）
        kwargs: 前向计算的 kwargs（可能包含 forward_result 等）
    
    Returns:
        List[ir.Expr]: 每个输入对应的梯度表达式
    """
    return [d_input1, d_input2, ...]

注册表存储：
    op_name -> grad_rule
"""

from typing import Dict, Callable, Optional, List, Any
from pypto import ir


class GradientRegistry:
    """
    全局梯度规则注册表（单例模式）
    
    使用类级别的字典存储，确保全局唯一：
    - 所有内置算子自动注册
    - 用户自定义算子可扩展注册
    - 反向变换器统一查询
    """
    
    # 类级别存储（全局唯一）
    _registry: Dict[str, Callable] = {}
    
    # 已注册算子分类（用于文档和调试）
    _categories: Dict[str, str] = {}  # op_name -> category
    
    @classmethod
    def register(
        cls,
        op_name: str,
        grad_func: Callable,
        category: str = "user"
    ) -> None:
        """
        注册算子梯度规则
        
        Args:
            op_name: 算子名称，如 'tile.matmul', 'tensor.add'
            grad_func: 梯度计算函数
            category: 分类标签，如 'tile', 'tensor', 'scalar', 'user'
        
        Raises:
            ValueError: 如果已存在同名算子（除非允许覆盖）
        
        Example:
            >>> GradientRegistry.register('tile.matmul', matmul_grad, 'tile')
        """
        if op_name in cls._registry:
            # 允许覆盖（用于自定义算子覆盖默认行为）
            import warnings
            warnings.warn(f"Overwriting existing gradient rule for '{op_name}'")
        
        cls._registry[op_name] = grad_func
        cls._categories[op_name] = category
    
    @classmethod
    def get(cls, op_name: str) -> Optional[Callable]:
        """
        获取算子梯度规则
        
        Args:
            op_name: 算子名称
        
        Returns:
            梯度函数，如果未注册则返回 None
        
        Example:
            >>> grad_rule = GradientRegistry.get('tile.add')
            >>> if grad_rule:
            >>>     grads = grad_rule([a, b], d_output, {})
        """
        return cls._registry.get(op_name)
    
    @classmethod
    def has(cls, op_name: str) -> bool:
        """
        检查算子是否已注册
        
        Args:
            op_name: 算子名称
        
        Returns:
            是否已注册
        """
        return op_name in cls._registry
    
    @classmethod
    def list_all(cls) -> List[str]:
        """
        列出所有已注册的算子
        
        Returns:
            算子名称列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """
        列出指定分类的算子
        
        Args:
            category: 分类名称
        
        Returns:
            该分类下的算子列表
        """
        return [op for op, cat in cls._categories.items() if cat == category]
    
    @classmethod
    def unregister(cls, op_name: str) -> bool:
        """
        移除算子注册（用于测试或覆盖）
        
        Args:
            op_name: 算子名称
        
        Returns:
            是否成功移除
        """
        if op_name in cls._registry:
            del cls._registry[op_name]
            del cls._categories[op_name]
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """
        清空所有注册（用于测试）
        """
        cls._registry.clear()
        cls._categories.clear()
    
    @classmethod
    def get_info(cls, op_name: str) -> Optional[Dict[str, Any]]:
        """
        获取算子注册信息
        
        Args:
            op_name: 算子名称
        
        Returns:
            包含 category, grad_func 信息
        """
        if op_name not in cls._registry:
            return None
        return {
            'op_name': op_name,
            'category': cls._categories[op_name],
            'grad_func': cls._registry[op_name],
        }
```

### 3.2 注册表初始化时机

```python
# python/pypto/autodiff/builtin_grad_register.py

"""
模块加载时自动注册所有内置算子梯度规则

时机：当 autodiff 模块被导入时，自动执行此文件
"""

# 导入即注册
from .grad_rules import (
    register_tile_ops,
    register_tensor_ops,
    register_scalar_ops,
)

# 注册所有内置算子
register_tile_ops()      # tile.add, tile.mul, tile.matmul, ...
register_tensor_ops()    # tensor.add, tensor.mul, ...
register_scalar_ops()    # add, mul, div, exp, log, ...

print(f"[Autodiff] Registered {len(GradientRegistry.list_all())} gradient rules")
```

```python
# python/pypto/autodiff/__init__.py

"""
autodiff 模块入口

导入顺序：
1. 先导入 gradient_registry（定义注册表）
2. 再导入 builtin_grad_register（注册内置规则）
3. 最后导出 API
"""

from .gradient_registry import GradientRegistry

# 注册内置规则
from .builtin_grad_register import *  # noqa: F401, F403

# 导出 API
from . import grad_rules  # 用户可查看已注册规则
from .reverse_mode import ReverseModeMutator
from .custom_grad import CustomGrad

__all__ = [
    'GradientRegistry',
    'grad_rules',
    'ReverseModeMutator',
    'CustomGrad',
    # ... 高层 API
]
```

---

## 4. 内置算子梯度规则注册

### 4.1 注册装饰器

```python
# python/pypto/autodiff/grad_rules.py

"""
内置算子梯度规则

使用 @register_grad 装饰器简化注册流程
"""

from typing import List, Dict, Any
from pypto import ir
from .gradient_registry import GradientRegistry


def register_grad(op_name: str, category: str = 'builtin'):
    """
    梯度规则注册装饰器
    
    Args:
        op_name: 算子名称
        category: 分类
    
    Returns:
        装饰器函数
    
    Example:
        >>> @register_grad('tile.add', 'tile')
        >>> def add_grad(saved_inputs, d_output, kwargs):
        >>>     return [d_output, d_output]
    """
    def decorator(grad_func):
        GradientRegistry.register(op_name, grad_func, category)
        return grad_func
    return decorator
```

### 4.2 Tile 算子梯度规则

```python
# python/pypto/autodiff/grad_rules.py (续)

# =============================================================================
# Tile 算子梯度规则
# =============================================================================

@register_grad('tile.add', 'tile')
def tile_add_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.add(x, y) 的梯度
    
    数学: z = x + y
    梯度: dx = dz, dy = dz
    
    Args:
        saved_inputs: [x, y] 前向输入
        d_output: dz 输出梯度
    
    Returns:
        [dx, dy] = [dz, dz]
    """
    return [d_output, d_output]


@register_grad('tile.sub', 'tile')
def tile_sub_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.sub(x, y) 的梯度
    
    数学: z = x - y
    梯度: dx = dz, dy = -dz
    """
    span = d_output.span
    dtype = d_output.type.dtype if hasattr(d_output.type, 'dtype') else ir.DataType.FP32
    
    neg_d = ir.Neg(d_output, dtype, span)
    return [d_output, neg_d]


@register_grad('tile.mul', 'tile')
def tile_mul_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.mul(x, y) 的梯度
    
    数学: z = x * y
    梯度: dx = dz * y, dy = dz * x
    
    注意: 需要保存前向的 x 和 y 值
    """
    x, y = saved_inputs
    span = d_output.span
    dtype = d_output.type.dtype if hasattr(d_output.type, 'dtype') else ir.DataType.FP32
    
    # dx = d_output * y
    dx = ir.Call(ir.Op('tile.mul'), [d_output, y], {}, x.type, span)
    
    # dy = d_output * x
    dy = ir.Call(ir.Op('tile.mul'), [d_output, x], {}, y.type, span)
    
    return [dx, dy]


@register_grad('tile.div', 'tile')
def tile_div_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.div(x, y) 的梯度
    
    数学: z = x / y
    梯度: dx = dz / y, dy = -dz * x / y²
    """
    x, y = saved_inputs
    span = d_output.span
    dtype = d_output.type.dtype if hasattr(d_output.type, 'dtype') else ir.DataType.FP32
    
    # dx = d_output / y
    dx = ir.Call(ir.Op('tile.div'), [d_output, y], {}, x.type, span)
    
    # dy = -d_output * x / y²
    y_squared = ir.Call(ir.Op('tile.mul'), [y, y], {}, y.type, span)
    x_times_d = ir.Call(ir.Op('tile.mul'), [x, d_output], {}, x.type, span)
    neg_x_times_d = ir.Neg(x_times_d, dtype, span)
    dy = ir.Call(ir.Op('tile.div'), [neg_x_times_d, y_squared], {}, y.type, span)
    
    return [dx, dy]


@register_grad('tile.matmul', 'tile')
def tile_matmul_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.matmul(A, B) 的梯度
    
    数学: C = A @ B
    梯度: dA = dC @ B^T, dB = A^T @ dC
    
    注意: 需要保存前向的 A 和 B
    """
    A, B = saved_inputs
    span = d_output.span
    
    # B_T = transpose(B)
    B_T = ir.Call(ir.Op('tile.transpose'), [B], {}, B.type, span)
    
    # dA = dC @ B_T
    dA = ir.Call(ir.Op('tile.matmul'), [d_output, B_T], {}, A.type, span)
    
    # A_T = transpose(A)
    A_T = ir.Call(ir.Op('tile.transpose'), [A], {}, A.type, span)
    
    # dB = A_T @ dC
    dB = ir.Call(ir.Op('tile.matmul'), [A_T, d_output], {}, B.type, span)
    
    return [dA, dB]


@register_grad('tile.transpose', 'tile')
def tile_transpose_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.transpose(X) 的梯度
    
    数学: Y = X^T
    梯度: dX = dY^T (transpose 的梯度是 transpose)
    """
    span = d_output.span
    
    # dX = transpose(d_output)
    dX = ir.Call(ir.Op('tile.transpose'), [d_output], {}, saved_inputs[0].type, span)
    
    return [dX]


@register_grad('tile.exp', 'tile')
def tile_exp_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.exp(x) 的梯度
    
    数学: y = exp(x)
    梯度: dx = dy * exp(x) = dy * y
    
    优化: 可以使用前向结果 y（已保存在 kwargs['forward_result']）
    """
    x = saved_inputs[0]
    span = d_output.span
    
    # 使用前向保存的结果（避免重复计算）
    forward_result = kwargs.get('forward_result')
    if forward_result:
        exp_x = forward_result
    else:
        exp_x = ir.Call(ir.Op('tile.exp'), [x], {}, x.type, span)
    
    # dx = d_output * exp_x
    dx = ir.Call(ir.Op('tile.mul'), [d_output, exp_x], {}, x.type, span)
    
    return [dx]


@register_grad('tile.log', 'tile')
def tile_log_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.log(x) 的梯度
    
    数学: y = log(x)
    梯度: dx = dy / x
    """
    x = saved_inputs[0]
    span = d_output.span
    
    # dx = d_output / x
    dx = ir.Call(ir.Op('tile.div'), [d_output, x], {}, x.type, span)
    
    return [dx]


@register_grad('tile.sqrt', 'tile')
def tile_sqrt_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.sqrt(x) 的梯度
    
    数学: y = sqrt(x)
    梯度: dx = dy / (2 * sqrt(x)) = dy / (2 * y)
    """
    x = saved_inputs[0]
    span = d_output.span
    dtype = x.type.dtype if hasattr(x.type, 'dtype') else ir.DataType.FP32
    
    # 使用前向结果
    forward_result = kwargs.get('forward_result')
    if forward_result:
        sqrt_x = forward_result
    else:
        sqrt_x = ir.Call(ir.Op('tile.sqrt'), [x], {}, x.type, span)
    
    # two = ConstFloat(2.0)
    two = ir.ConstFloat(2.0, dtype, span)
    
    # 2 * sqrt_x
    two_sqrt_x = ir.Call(ir.Op('tile.mul'), [two, sqrt_x], {}, x.type, span)
    
    # dx = d_output / (2 * sqrt_x)
    dx = ir.Call(ir.Op('tile.div'), [d_output, two_sqrt_x], {}, x.type, span)
    
    return [dx]


@register_grad('tile.relu', 'tile')
def tile_relu_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.relu(x) 的梯度
    
    数学: y = max(0, x)
    梯度: dx = dy * (x > 0 ? 1 : 0)
    
    需要保存前向的 x 值来确定 mask
    """
    x = saved_inputs[0]
    span = d_output.span
    dtype = x.type.dtype if hasattr(x.type, 'dtype') else ir.DataType.FP32
    
    # mask = (x > 0)
    zero = ir.ConstFloat(0.0, dtype, span)
    mask = ir.Call(ir.Op('tile.gt'), [x, zero], {}, x.type, span)
    
    # cast mask to float
    mask_float = ir.Call(ir.Op('tile.cast'), [mask], {'dtype': dtype}, x.type, span)
    
    # dx = d_output * mask
    dx = ir.Call(ir.Op('tile.mul'), [d_output, mask_float], {}, x.type, span)
    
    return [dx]


@register_grad('tile.sin', 'tile')
def tile_sin_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.sin(x) 的梯度
    
    数学: y = sin(x)
    梯度: dx = dy * cos(x)
    """
    x = saved_inputs[0]
    span = d_output.span
    
    # cos(x)
    cos_x = ir.Call(ir.Op('tile.cos'), [x], {}, x.type, span)
    
    # dx = d_output * cos(x)
    dx = ir.Call(ir.Op('tile.mul'), [d_output, cos_x], {}, x.type, span)
    
    return [dx]


@register_grad('tile.cos', 'tile')
def tile_cos_grad(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    tile.cos(x) 的梯度
    
    数学: y = cos(x)
    梯度: dx = -dy * sin(x)
    """
    x = saved_inputs[0]
    span = d_output.span
    dtype = x.type.dtype if hasattr(x.type, 'dtype') else ir.DataType.FP32
    
    # sin(x)
    sin_x = ir.Call(ir.Op('tile.sin'), [x], {}, x.type, span)
    
    # dy * sin(x)
    d_times_sin = ir.Call(ir.Op('tile.mul'), [d_output, sin_x], {}, x.type, span)
    
    # -dy * sin(x)
    dx = ir.Neg(d_times_sin, dtype, span)
    
    return [dx]


# =============================================================================
# Tensor 算子梯度规则（与 Tile 类似）
# =============================================================================

@register_grad('tensor.add', 'tensor')
def tensor_add_grad(saved_inputs, d_output, kwargs):
    return [d_output, d_output]


@register_grad('tensor.mul', 'tensor')
def tensor_mul_grad(saved_inputs, d_output, kwargs):
    x, y = saved_inputs
    span = d_output.span
    dx = ir.Call(ir.Op('tensor.mul'), [d_output, y], {}, x.type, span)
    dy = ir.Call(ir.Op('tensor.mul'), [d_output, x], {}, y.type, span)
    return [dx, dy]


# ... 其他 tensor 算子


# =============================================================================
# Scalar 算子梯度规则
# =============================================================================

@register_grad('scalar.add', 'scalar')
def scalar_add_grad(saved_inputs, d_output, kwargs):
    return [d_output, d_output]


@register_grad('scalar.mul', 'scalar')
def scalar_mul_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    span = d_output.span
    dtype = d_output.type.dtype
    da = ir.Mul(d_output, b, dtype, span)
    db = ir.Mul(d_output, a, dtype, span)
    return [da, db]


@register_grad('scalar.exp', 'scalar')
def scalar_exp_grad(saved_inputs, d_output, kwargs):
    a = saved_inputs[0]
    span = d_output.span
    dtype = a.type.dtype
    
    # 使用前向结果
    forward_result = kwargs.get('forward_result')
    if forward_result:
        exp_a = forward_result
    else:
        exp_a = ir.Call(ir.Op('scalar.exp'), [a], {}, a.type, span)
    
    da = ir.Mul(d_output, exp_a, dtype, span)
    return [da]


# =============================================================================
# 注册函数
# =============================================================================

def register_tile_ops():
    """注册所有 tile 算子梯度规则"""
    # 已通过装饰器自动注册
    tile_ops = GradientRegistry.list_by_category('tile')
    print(f"  Registered {len(tile_ops)} tile gradient rules")


def register_tensor_ops():
    """注册所有 tensor 算子梯度规则"""
    tensor_ops = GradientRegistry.list_by_category('tensor')
    print(f"  Registered {len(tensor_ops)} tensor gradient rules")


def register_scalar_ops():
    """注册所有 scalar 算子梯度规则"""
    scalar_ops = GradientRegistry.list_by_category('scalar')
    print(f"  Registered {len(scalar_ops)} scalar gradient rules")
```

---

## 5. 算法中获取梯度规则

### 5.1 ReverseModeMutator 获取梯度

```python
# python/pypto/autodiff/reverse_mode.py

"""
反向模式自动微分变换器

核心：从 GradientRegistry 获取梯度规则，构建反向 IR
"""

from typing import Dict, List, Optional
from pypto import ir
from pypto.ir.transforms.base import IRMutator
from .gradient_registry import GradientRegistry
from .gradient_accumulator import GradientAccumulator


class ReverseModeMutator(IRMutator):
    """
    反向模式自动微分 IR 变换器
    
    核心流程：
    1. 反向遍历前向 IR 的语句
    2. 对每个算子调用，从注册表获取梯度规则
    3. 使用梯度规则生成反向梯度表达式
    4. 累加梯度到梯度累加器
    """
    
    def __init__(
        self,
        grad_params: List[ir.Var],
        grad_output: Optional[str] = None
    ):
        self.grad_params = grad_params           # 需要梯度的参数
        self.grad_output = grad_output           # 指定输出的梯度
        
        # 梯度变量映射
        self.grad_var_map: Dict[ir.Var, ir.Var] = {}
        
        # 梯度累加器（处理多次使用）
        self.accumulator = GradientAccumulator()
        
        # 前向保存的值
        self.saved_values: Dict[ir.Var, ir.Expr] = {}
        
        # Tape（前向记录）
        self.tape: List[TapeRecord] = []
    
    def differentiate(self, forward_func: ir.Function) -> ir.Function:
        """
        构建反向梯度函数
        
        Args:
            forward_func: 前向函数 IR
        
        Returns:
            反向梯度函数 IR
        """
        # 1. 创建梯度变量
        self._create_grad_vars(forward_func)
        
        # 2. 反向遍历函数体
        reverse_stmts = []
        body = forward_func.body
        
        if isinstance(body, ir.SeqStmts):
            # 反向遍历语句序列
            for stmt in reversed(body.stmts):
                reverse_stmt = self._reverse_stmt(stmt)
                if reverse_stmt:
                    reverse_stmts.append(reverse_stmt)
        
        # 3. 构建反向函数体
        reverse_body = self._build_reverse_body(reverse_stmts)
        
        # 4. 创建反向函数
        backward_func = ir.Function(
            f"grad_{forward_func.name}",
            self._create_backward_params(forward_func),
            self._create_backward_return_types(),
            reverse_body,
            forward_func.span
        )
        
        return backward_func
    
    def _reverse_call(self, call: ir.Call) -> List[ir.Stmt]:
        """
        处理算子调用的反向
        
        核心：从 GradientRegistry 获取梯度规则
        
        Args:
            call: 前向算子调用
        
        Returns:
            反向梯度语句列表
        """
        op_name = call.op.name
        
        # ============================================
        # 核心：从注册表获取梯度规则
        # ============================================
        grad_rule = GradientRegistry.get(op_name)
        
        if grad_rule is None:
            # 未注册的算子
            raise NotImplementedError(
                f"No gradient rule registered for '{op_name}'. "
                f"You can register using @register_grad('{op_name}') or "
                f"GradientRegistry.register('{op_name}', grad_func)"
            )
        
        # 获取输出梯度
        d_output = self._get_output_grad(call)
        
        # 获取保存的输入值
        saved_inputs = self._get_saved_inputs(call)
        
        # 构建 kwargs（包含前向结果等）
        kwargs = {
            'forward_result': self.saved_values.get(call),
            'call_kwargs': call.kwargs,
        }
        
        # ============================================
        # 核心：调用梯度规则
        # ============================================
        grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
        
        # 将梯度表达式累加到输入变量的梯度
        result_stmts = []
        for i, (input_expr, grad_expr) in enumerate(zip(call.args, grad_exprs)):
            if isinstance(input_expr, ir.Var):
                # 梯度累加
                self.accumulator.add_gradient(input_expr, grad_expr)
                
                # 生成赋值语句
                if input_expr in self.grad_var_map:
                    grad_var = self.grad_var_map[input_expr]
                    total_grad = self.accumulator.get_total_gradient(input_expr)
                    assign_stmt = ir.AssignStmt(grad_var, total_grad, call.span)
                    result_stmts.append(assign_stmt)
        
        return result_stmts
    
    def _get_saved_inputs(self, call: ir.Call) -> List[ir.Expr]:
        """获取前向保存的输入值"""
        saved = []
        for arg in call.args:
            if isinstance(arg, ir.Var):
                # 从保存值中获取
                if arg in self.saved_values:
                    saved.append(self.saved_values[arg])
                else:
                    saved.append(arg)  # 使用原值
            else:
                saved.append(arg)
        return saved
    
    def _get_output_grad(self, call: ir.Call) -> ir.Expr:
        """获取输出的梯度"""
        # 从梯度累加器或梯度种子获取
        if isinstance(call, ir.Var):
            if call in self.grad_var_map:
                return self.accumulator.get_total_gradient(call)
        
        # 默认梯度种子
        return ir.ConstFloat(1.0, ir.DataType.FP32, call.span)
```

### 5.2 GradientAccumulator 实现

```python
# python/pypto/autodiff/gradient_accumulator.py

"""
梯度累加器

处理变量多次使用时的梯度累加
"""

from typing import Dict, List
from pypto import ir


class GradientAccumulator:
    """
    梯度累加器
    
    当一个变量在多个表达式中使用时，其梯度需要累加所有贡献。
    
    Example:
        y = a * b + a * c
        # a 的梯度: da = dy * b + dy * c (两个贡献累加)
    """
    
    def __init__(self):
        # var -> [grad_expr1, grad_expr2, ...]
        self._gradients: Dict[ir.Var, List[ir.Expr]] = {}
    
    def add_gradient(self, var: ir.Var, grad_expr: ir.Expr) -> None:
        """
        添加梯度贡献
        
        Args:
            var: 前向变量
            grad_expr: 梯度表达式
        """
        if var not in self._gradients:
            self._gradients[var] = []
        self._gradients[var].append(grad_expr)
    
    def get_total_gradient(self, var: ir.Var) -> ir.Expr:
        """
        获取累积后的总梯度
        
        Args:
            var: 前向变量
        
        Returns:
            累加后的梯度表达式（如果只有一个贡献，直接返回；
                              多个贡献，生成 Add 表达式）
        """
        grads = self._gradients.get(var, [])
        
        if not grads:
            # 无梯度，返回零
            return self._create_zero(var.type, var.span)
        
        if len(grads) == 1:
            # 单一贡献，直接返回
            return grads[0]
        
        # 多个贡献，累加
        span = var.span
        dtype = var.type.dtype if hasattr(var.type, 'dtype') else ir.DataType.FP32
        
        result = grads[0]
        for grad in grads[1:]:
            # 构建 Add 表达式
            if isinstance(var.type, ir.ScalarType):
                result = ir.Add(result, grad, dtype, span)
            else:
                # Tile/Tensor: 使用 tile.add / tensor.add
                result = ir.Call(ir.Op('tile.add'), [result, grad], {}, var.type, span)
        
        return result
    
    def get_all_gradients(self) -> Dict[ir.Var, ir.Expr]:
        """获取所有变量的总梯度"""
        return {
            var: self.get_total_gradient(var)
            for var in self._gradients
        }
    
    def clear(self) -> None:
        """清空累加器"""
        self._gradients.clear()
    
    def has_gradient(self, var: ir.Var) -> bool:
        """检查是否有梯度"""
        return var in self._gradients and len(self._gradients[var]) > 0
    
    def _create_zero(self, type: ir.Type, span: ir.Span) -> ir.Expr:
        """创建零值"""
        if isinstance(type, ir.ScalarType):
            dtype = type.dtype
            if dtype.IsFloat():
                return ir.ConstFloat(0.0, dtype, span)
            return ir.ConstInt(0, dtype, span)
        
        # Tile/Tensor: 调用 zeros
        if isinstance(type, ir.TileType):
            return ir.Call(
                ir.Op('tile.zeros'),
                [],
                {'shape': type.shape, 'dtype': type.dtype},
                type,
                span
            )
        
        if isinstance(type, ir.TensorType):
            return ir.Call(
                ir.Op('tensor.zeros'),
                [],
                {'shape': type.shape, 'dtype': type.dtype},
                type,
                span
            )
        
        return ir.ConstFloat(0.0, ir.DataType.FP32, span)
```

---

## 6. 自定义函数梯度注册

### 6.1 使用 @register_grad 装饰器

```python
# python/pypto/autodiff/custom_examples.py

"""
用户自定义梯度规则示例

方式 1: 使用 @register_grad 装饰器注册自定义算子梯度
"""

import pypto as pl
from pypto.autodiff import register_grad, GradientRegistry
from pypto import ir


# =============================================================================
# 示例 1: 注册自定义算子梯度
# =============================================================================

@register_grad('my.custom_softmax', 'user')
def custom_softmax_grad(saved_inputs, d_output, kwargs):
    """
    自定义 softmax 梯度
    
    数学: y = softmax(x) = exp(x) / sum(exp(x))
    梯度: dx = y * (d_output - sum(d_output * y))
    
    Args:
        saved_inputs: [x]
        d_output: dy
        kwargs: {'forward_result': y}
    
    Returns:
        [dx]
    """
    x = saved_inputs[0]
    span = d_output.span
    dtype = x.type.dtype if hasattr(x.type, 'dtype') else ir.DataType.FP32
    
    # 使用前向保存的 softmax 结果
    y = kwargs.get('forward_result')
    if y is None:
        # 重新计算 softmax
        y = ir.Call(ir.Op('my.custom_softmax'), [x], {}, x.type, span)
    
    # d_output * y
    d_times_y = ir.Call(ir.Op('tile.mul'), [d_output, y], {}, x.type, span)
    
    # sum(d_output * y) - 需要 reduce 操作
    sum_d_times_y = ir.Call(ir.Op('tile.reduce_sum'), [d_times_y], {}, x.type, span)
    
    # d_output - sum(d_output * y)
    d_minus_sum = ir.Call(ir.Op('tile.sub'), [d_output, sum_d_times_y], {}, x.type, span)
    
    # dx = y * (d_output - sum(d_output * y))
    dx = ir.Call(ir.Op('tile.mul'), [y, d_minus_sum], {}, x.type, span)
    
    return [dx]


# =============================================================================
# 示例 2: 注册复合算子梯度
# =============================================================================

@register_grad('my.layer_norm', 'user')
def layer_norm_grad(saved_inputs, d_output, kwargs):
    """
    Layer Normalization 梯度
    
    数学: y = layer_norm(x) = (x - mean) / sqrt(var + eps)
    
    梯度较复杂，需要分别对 x、mean、var 求导
    """
    x, weight, bias = saved_inputs  # layer_norm 通常有 weight 和 bias
    span = d_output.span
    
    # 获取前向保存的值
    forward_result = kwargs.get('forward_result')
    mean = kwargs.get('saved_mean')
    var = kwargs.get('saved_var')
    eps = kwargs.get('eps', 1e-5)
    
    # ... 复杂的梯度计算
    
    # 简化示例：假设只对 x 计算梯度
    # dx = d_output * (1 / sqrt(var + eps)) * (1 - x * ...)
    
    return [dx, dw, db]  # 返回三个输入的梯度


# =============================================================================
# 示例 3: 注册不可导算子（零梯度）
# =============================================================================

@register_grad('tile.argmax', 'user')
def argmax_grad(saved_inputs, d_output, kwargs):
    """
    argmax 不可导（离散操作），梯度为零
    
    Args:
        saved_inputs: [x]
        d_output: 梯度种子
    
    Returns:
        [zero] - 返回零梯度
    """
    x = saved_inputs[0]
    span = d_output.span
    
    # 创建零值
    zero = ir.Call(
        ir.Op('tile.zeros'),
        [],
        {'shape': x.type.shape, 'dtype': x.type.dtype},
        x.type,
        span
    )
    
    return [zero]
```

### 6.2 使用 GradientRegistry.register 方法

```python
# python/pypto/autodiff/custom_examples.py (续)

"""
方式 2: 直接调用 GradientRegistry.register()
"""

# =============================================================================
# 示例 4: 动态注册梯度规则
# =============================================================================

def register_dynamic_grad_rule(op_name: str, grad_func: Callable):
    """
    动态注册梯度规则（运行时）
    
    Args:
        op_name: 算子名称
        grad_func: 梯度函数
    """
    GradientRegistry.register(op_name, grad_func, category='dynamic')
    print(f"Registered gradient rule for '{op_name}'")


# 注册一个简单的自定义操作
def my_op_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    span = d_output.span
    da = ir.Call(ir.Op('tile.mul'), [d_output, ir.ConstFloat(2.0, ir.DataType.FP32, span)], {}, a.type, span)
    db = d_output
    return [da, db]

register_dynamic_grad_rule('my.double_add', my_op_grad)


# =============================================================================
# 示例 5: 覆盖内置梯度规则
# =============================================================================

def custom_matmul_grad(saved_inputs, d_output, kwargs):
    """
    自定义 matmul 梯度（优化版本）
    
    可以覆盖默认的 tile.matmul 梯度规则
    """
    A, B = saved_inputs
    span = d_output.span
    
    # 使用优化的反向 matmul 算子
    dA = ir.Call(ir.Op('tile.matmul_backward_A'), [d_output, B], {}, A.type, span)
    dB = ir.Call(ir.Op('tile.matmul_backward_B'), [A, d_output], {}, B.type, span)
    
    return [dA, dB]

# 覆盖默认规则（会发出警告）
GradientRegistry.register('tile.matmul', custom_matmul_grad, 'user_override')
```

### 6.3 使用 CustomGrad 创建自定义梯度算子

```python
# python/pypto/autodiff/custom_grad.py

"""
自定义梯度算子创建

方式 3: 使用 CustomGrad() 创建新算子并自动注册梯度
"""

from typing import Callable, Optional
from pypto import ir
from .gradient_registry import GradientRegistry


def CustomGrad(
    forward_func: Callable,
    backward_func: Callable,
    name: Optional[str] = None,
    category: str = 'custom'
) -> ir.Op:
    """
    创建自定义梯度算子
    
    同时注册前向算子和反向梯度规则
    
    Args:
        forward_func: 前向计算函数（Python DSL）
        backward_func: 反向梯度函数（Python DSL）
        name: 算子名称（默认使用 forward_func 名称）
        category: 分类
    
    Returns:
        自定义算子 Op
    
    Example:
        >>> def sigmoid_forward(x):
        >>>     return pl.div(1, pl.add(1, pl.exp(pl.neg(x))))
        >>> 
        >>> def sigmoid_backward(x, d_output):
        >>>     s = sigmoid_forward(x)
        >>>     return pl.mul(d_output, pl.mul(s, pl.sub(1, s)))
        >>> 
        >>> sigmoid_op = CustomGrad(sigmoid_forward, sigmoid_backward, 'my.sigmoid')
        >>> 
        >>> # 在函数中使用
        >>> @pl.function
        >>> def f(x):
        >>>     return sigmoid_op(x)
    """
    if name is None:
        name = f"custom.{forward_func.__name__}"
    
    # 创建算子 Op
    op = ir.Op(name, {})
    
    # ============================================
    # 核心：自动注册梯度规则
    # ============================================
    def grad_rule(saved_inputs, d_output, kwargs):
        """
        包装 backward_func 为标准梯度规则格式
        """
        # 调用用户的反向函数
        grad_exprs = backward_func(*saved_inputs, d_output, **kwargs)
        
        # 确保返回列表
        if not isinstance(grad_exprs, list):
            grad_exprs = [grad_exprs]
        
        return grad_exprs
    
    # 注册到全局注册表
    GradientRegistry.register(name, grad_rule, category)
    
    return op


# =============================================================================
# 使用示例
# =============================================================================

def gelu_forward(x):
    """
    GELU 激活函数前向
    
    数学: gelu(x) = x * Φ(x) ≈ x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    import pypto as pl
    sqrt2 = pl.sqrt(2.0)
    erf_arg = pl.div(x, sqrt2)
    erf_x = pl.erf(erf_arg)  # 需要 erf 算子
    gelu = pl.mul(x, pl.mul(0.5, pl.add(1, erf_x)))
    return gelu


def gelu_backward(x, d_output):
    """
    GELU 激活函数反向
    
    数学: dgelu/dx ≈ Φ(x) + x * φ(x/sqrt(2)) / sqrt(2)
    """
    import pypto as pl
    sqrt2 = pl.sqrt(2.0)
    sqrt2pi = pl.sqrt(2 * pl.pi)
    
    # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    erf_arg = pl.div(x, sqrt2)
    erf_x = pl.erf(erf_arg)
    phi_x = pl.mul(0.5, pl.add(1, erf_x))
    
    # φ(x/sqrt(2)) = exp(-x²/2) / sqrt(2π)
    neg_x_sq_half = pl.neg(pl.div(pl.mul(x, x), 2))
    gaussian = pl.div(pl.exp(neg_x_sq_half), sqrt2pi)
    
    # dx = d_output * (Φ(x) + x * gaussian / sqrt(2))
    dx_term = pl.mul(x, pl.div(gaussian, sqrt2))
    dx = pl.mul(d_output, pl.add(phi_x, dx_term))
    
    return dx


# 创建 GELU 自定义算子
gelu_op = CustomGrad(gelu_forward, gelu_backward, 'my.gelu')

# 现在可以在 pl.function 中使用
# @pl.function
# def f(x):
#     return gelu_op(x)

# 自动微分时会自动调用 gelu_backward
```

---

## 7. 查询已注册梯度规则

### 7.1 查询 API

```python
# python/pypto/autodiff/registry_query.py

"""
梯度规则查询工具
"""

from .gradient_registry import GradientRegistry


def list_all_grad_rules():
    """列出所有已注册的梯度规则"""
    rules = GradientRegistry.list_all()
    print(f"Total registered gradient rules: {len(rules)}")
    
    for op_name in sorted(rules):
        info = GradientRegistry.get_info(op_name)
        print(f"  {op_name}: category={info['category']}")


def list_grad_rules_by_category(category: str):
    """列出指定分类的梯度规则"""
    rules = GradientRegistry.list_by_category(category)
    print(f"Gradient rules in category '{category}': {len(rules)}")
    
    for op_name in sorted(rules):
        print(f"  {op_name}")


def check_grad_rule_exists(op_name: str):
    """检查梯度规则是否存在"""
    if GradientRegistry.has(op_name):
        print(f"✓ Gradient rule for '{op_name}' is registered")
        info = GradientRegistry.get_info(op_name)
        print(f"  Category: {info['category']}")
        print(f"  Function: {info['grad_func'].__name__}")
    else:
        print(f"✗ Gradient rule for '{op_name}' is NOT registered")
        print(f"  You can register using:")
        print(f"    @register_grad('{op_name}')")
        print(f"    def custom_grad(saved_inputs, d_output, kwargs):")
        print(f"        return [...]")


def get_grad_rule_details(op_name: str):
    """获取梯度规则详细信息"""
    info = GradientRegistry.get_info(op_name)
    if info:
        grad_func = info['grad_func']
        
        print(f"Gradient rule for '{op_name}':")
        print(f"  Category: {info['category']}")
        print(f"  Function name: {grad_func.__name__}")
        print(f"  Signature: (saved_inputs, d_output, kwargs) -> List[Expr]")
        
        # 尝试获取源码
        import inspect
        try:
            source = inspect.getsource(grad_func)
            print(f"  Source:\n{source}")
        except:
            print(f"  Source: (unable to retrieve)")
    else:
        print(f"No gradient rule found for '{op_name}'")


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == '__main__':
    # 列出所有规则
    list_all_grad_rules()
    
    # 列出 tile 分类
    list_grad_rules_by_category('tile')
    
    # 检查特定规则
    check_grad_rule_exists('tile.matmul')
    check_grad_rule_exists('my.custom_op')
    
    # 获取详细信息
    get_grad_rule_details('tile.add')
```

### 7.2 CLI 工具

```python
# python/pypto/autodiff/cli.py

"""
命令行工具：查询和管理梯度规则
"""

import argparse
from .gradient_registry import GradientRegistry


def main():
    parser = argparse.ArgumentParser(description='PyPTO Autodiff Registry CLI')
    parser.add_argument('command', choices=['list', 'check', 'info', 'categories'])
    parser.add_argument('--op', type=str, help='Operator name')
    parser.add_argument('--category', type=str, help='Category filter')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        if args.category:
            rules = GradientRegistry.list_by_category(args.category)
        else:
            rules = GradientRegistry.list_all()
        print("Registered gradient rules:")
        for op in sorted(rules):
            print(f"  {op}")
    
    elif args.command == 'check':
        if not args.op:
            print("Error: --op required for check command")
            return
        if GradientRegistry.has(args.op):
            print(f"✓ '{args.op}' is registered")
        else:
            print(f"✗ '{args.op}' is NOT registered")
    
    elif args.command == 'info':
        if not args.op:
            print("Error: --op required for info command")
            return
        info = GradientRegistry.get_info(args.op)
        if info:
            print(f"Operator: {args.op}")
            print(f"Category: {info['category']}")
            print(f"Function: {info['grad_func'].__name__}")
        else:
            print(f"No info for '{args.op}'")
    
    elif args.command == 'categories':
        categories = set(GradientRegistry._categories.values())
        print("Available categories:")
        for cat in sorted(categories):
            count = len(GradientRegistry.list_by_category(cat))
            print(f"  {cat}: {count} rules")


if __name__ == '__main__':
    main()
```

---

## 8. 梯度规则注册完整流程

### 8.1 内置算子注册流程

```
┌─────────────────────────────────────────────────────────────┐
│              内置算子梯度规则注册流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 模块导入                                                 │
│     import pypto.autodiff                                   │
│         │                                                   │
│         ▼                                                   │
│  2. 加载 gradient_registry.py                               │
│     - 定义 GradientRegistry 类                              │
│     - 创建空的 _registry 字典                                │
│         │                                                   │
│         ▼                                                   │
│  3. 加载 builtin_grad_register.py                           │
│     - 调用 register_tile_ops()                              │
│     - 调用 register_tensor_ops()                            │
│     - 调用 register_scalar_ops()                            │
│         │                                                   │
│         ▼                                                   │
│  4. 加载 grad_rules.py                                       │
│     - 执行 @register_grad 装饰器                             │
│     - 自动注册到 GradientRegistry                            │
│         │                                                   │
│         ▼                                                   │
│  5. 注册完成                                                 │
│     GradientRegistry._registry:                            │
│       'tile.add' → tile_add_grad                            │
│       'tile.mul' → tile_mul_grad                            │
│       'tile.matmul' → tile_matmul_grad                      │
│       ...                                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 算法获取梯度流程

```
┌─────────────────────────────────────────────────────────────┐
│            反向变换器获取梯度规则流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 反向遍历前向 IR                                          │
│     for stmt in reversed(forward_func.body):               │
│         │                                                   │
│         ▼                                                   │
│  2. 遇到 Call 节点                                          │
│     call = ir.Call(op, args, kwargs, type, span)           │
│         │                                                   │
│         ▼                                                   │
│  3. 提取算子名称                                             │
│     op_name = call.op.name  # e.g., 'tile.matmul'          │
│         │                                                   │
│         ▼                                                   │
│  4. 查询注册表                                               │
│     grad_rule = GradientRegistry.get(op_name)              │
│         │                                                   │
│         ├──> 如果找到:                                       │
│         │    grad_exprs = grad_rule(saved, d_out, kw)      │
│         │                                                   │
│         ├──> 如果未找到:                                     │
│         │    raise NotImplementedError                      │
│         │                                                   │
│         ▼                                                   │
│  5. 处理梯度表达式                                           │
│     for input, grad in zip(call.args, grad_exprs):         │
│         accumulator.add_gradient(input, grad)              │
│         │                                                   │
│         ▼                                                   │
│  6. 生成反向 IR                                              │
│     assign_stmt = AssignStmt(grad_var, total_grad)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 自定义梯度注册流程

```
┌─────────────────────────────────────────────────────────────┐
│           用户自定义梯度规则注册流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  方式 1: @register_grad 装饰器                               │
│  ─────────────────────────────                              │
│  @register_grad('my.custom_op', 'user')                    │
│  def my_grad(saved_inputs, d_output, kwargs):               │
│      return [grad_expr]                                     │
│      │                                                      │
│      ▼  (装饰器执行时自动注册)                                │
│  GradientRegistry.register('my.custom_op', my_grad)        │
│                                                             │
│                                                             │
│  方式 2: GradientRegistry.register()                        │
│  ─────────────────────────────────────                      │
│  def my_grad(saved_inputs, d_output, kwargs):               │
│      return [grad_expr]                                     │
│                                                             │
│  GradientRegistry.register('my.custom_op', my_grad)        │
│                                                             │
│                                                             │
│  方式 3: CustomGrad()                                        │
│  ─────────────────────                                       │
│  def forward(x):                                            │
│      return ...                                             │
│                                                             │
│  def backward(x, d_out):                                    │
│      return grad                                            │
│                                                             │
│  op = CustomGrad(forward, backward, 'my.op')               │
│      │                                                      │
│      ▼  (CustomGrad 内部自动注册)                            │
│  GradientRegistry.register('my.op', wrapped_backward)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 已注册算子完整列表

### 9.1 Tile 算子

| 算子 | 梯度规则 | 数学 |
|------|----------|------|
| `tile.add` | `tile_add_grad` | `ā=ȳ, b̄=ȳ` |
| `tile.sub` | `tile_sub_grad` | `ā=ȳ, b̄=-ȳ` |
| `tile.mul` | `tile_mul_grad` | `ā=ȳ·b, b̄=ȳ·a` |
| `tile.div` | `tile_div_grad` | `ā=ȳ/b, b̄=-ȳ·a/b²` |
| `tile.matmul` | `tile_matmul_grad` | `Ā=ȳ@B^T, B̄=A^T@ȳ` |
| `tile.transpose` | `tile_transpose_grad` | `̄=transpose(ȳ)` |
| `tile.exp` | `tile_exp_grad` | `ā=ȳ·e^a` |
| `tile.log` | `tile_log_grad` | `ā=ȳ/a` |
| `tile.sqrt` | `tile_sqrt_grad` | `ā=ȳ/(2√a)` |
| `tile.relu` | `tile_relu_grad` | `ā=ȳ·sign(a>0)` |
| `tile.sin` | `tile_sin_grad` | `ā=ȳ·cos(a)` |
| `tile.cos` | `tile_cos_grad` | `ā=-ȳ·sin(a)` |

### 9.2 Tensor 算子

| 算子 | 梯度规则 | 与 Tile 关系 |
|------|----------|--------------|
| `tensor.add` | `tensor_add_grad` | 同 tile.add |
| `tensor.mul` | `tensor_mul_grad` | 同 tile.mul |
| `tensor.matmul` | `tensor_matmul_grad` | 同 tile.matmul |
| ... | ... | ... |

### 9.3 Scalar 算子

| 算子 | 梯度规则 | 数学 |
|------|----------|------|
| `scalar.add` | `scalar_add_grad` | `ā=ȳ` |
| `scalar.mul` | `scalar_mul_grad` | `ā=ȳ·b` |
| `scalar.exp` | `scalar_exp_grad` | `ā=ȳ·e^a` |

---

## 10. 总结

### 10.1 三种注册方式对比

| 方式 | 适用场景 | 示例 |
|------|----------|------|
| `@register_grad` | 注册新算子梯度 | `@register_grad('my.op')` |
| `GradientRegistry.register()` | 动态注册、覆盖默认 | `Registry.register('tile.matmul', custom)` |
| `CustomGrad()` | 创建新的自定义算子 | `op = CustomGrad(fwd, bwd)` |

### 10.2 梯度规则获取流程

```
call.op.name → GradientRegistry.get(op_name) → grad_rule(saved, d_out, kw) → grad_exprs
```

### 10.3 框架扩展性

- **内置算子**: 自动注册，无需用户干预
- **自定义算子**: 三种注册方式，灵活扩展
- **覆盖默认**: 允许覆盖内置规则（发出警告）
- **分类管理**: 按类别组织，便于查询

### 10.4 文件位置

| 文件 | 功能 |
|------|------|
| `gradient_registry.py` | 注册表核心 |
| `grad_rules.py` | 内置梯度规则 |
| `builtin_grad_register.py` | 初始化注册 |
| `reverse_mode.py` | 使用梯度规则 |
| `custom_grad.py` | 自定义算子支持 |
| `cli.py` | 命令行查询工具 |