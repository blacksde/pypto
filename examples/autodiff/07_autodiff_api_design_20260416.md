# PyPTO 自动微分 API 设计

## 1. API 概述

PyPTO 自动微分提供简洁的高层接口，支持从 `pl.function` 自动生成反向梯度函数。

### 1.1 核心功能

| API | 功能 | 返回 |
|------|------|------|
| `pl.grad(func, params)` | 计算梯度函数 | 反向函数 IR |
| `pl.value_and_grad(func, params)` | 前向+反向函数 | 函数对 |
| `pl.jacobian(func, params)` | 计算 Jacobian | Jacobian 函数 |
| `pl.vjp(func)` | Vector-Jacobian Product | VJP 函数 |

### 1.2 使用示例

```python
import pypto as pl

# 定义前向函数
@pl.function
def forward(x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
    return result

# 生成梯度函数
grad_func = pl.grad(forward, params=['x', 'y'])

# 生成前向+反向函数
forward_func, backward_func = pl.value_and_grad(forward, params=['x', 'y'])
```

---

## 2. API 详细设计

### 2.1 pl.grad

生成梯度函数（反向模式）。

#### 函数签名

```python
def grad(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None,
    grad_output: Optional[str] = None,
    name: Optional[str] = None
) -> ir.Function
```

#### 参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `func` | `ir.Function` 或 `Callable` | 必需 | 前向函数（IR 或 Python DSL） |
| `params` | `List[str]` | 所有输入参数 | 需要计算梯度的参数名称列表 |
| `grad_output` | `str` | 第一个输出 | 指定梯度计算的输出变量 |
| `name` | `str` | `grad_{func_name}` | 梯度函数名称 |

#### 返回

返回反向梯度函数 `ir.Function`，签名如下：

```python
# 前向函数签名
forward(a: T1, b: T2, c: T3) -> (y: T_out, z: T_out2)

# 梯度函数签名
grad_forward(a: T1, b: T2, c: T3, 
             d_y: T_out,      # 输出的梯度种子
             d_z: T_out2      # 可选，如果 grad_output='all'
             ) -> (d_a: T1, d_b: T2)  # params 中指定的参数梯度
```

#### 使用示例

```python
import pypto as pl

# 示例 1: 基本使用
@pl.function
def linear(x: pl.Tensor[[64], pl.FP32], 
           w: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.mul(x, w)

grad_linear = pl.grad(linear)
# grad_linear(x, w, d_output) -> (d_x, d_w)

# 示例 2: 指定参数
grad_linear_x_only = pl.grad(linear, params=['x'])
# grad_linear_x_only(x, w, d_output) -> d_x

# 示例 3: 从 IR 直接生成
forward_ir = linear  # 已有的 IR Function
backward_ir = pl.grad(forward_ir, params=['x', 'w'])
```

---

### 2.2 pl.value_and_grad

生成前向函数和梯度函数的组合，支持同时计算值和梯度。

#### 函数签名

```python
def value_and_grad(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> Tuple[ir.Function, ir.Function]
```

#### 参数

同 `pl.grad`。

#### 返回

返回函数对 `(forward_func, backward_func)`：

- `forward_func`: 前向计算函数（保存中间值）
- `backward_func`: 反向梯度函数（使用保存的值）

#### 使用示例

```python
@pl.function
def matmul(A: pl.Tile[[16, 16], pl.FP16], 
           B: pl.Tile[[16, 16], pl.FP16]) -> pl.Tile[[16, 16], pl.FP16]:
    return pl.matmul(A, B)

forward_func, backward_func = pl.value_and_grad(matmul)

# forward_func(A, B) -> C, saved_values
# backward_func(saved_values, d_C) -> (d_A, d_B)
```

---

### 2.3 pl.jacobian

计算 Jacobian 矩阵（多输出对所有参数的梯度）。

#### 函数签名

```python
def jacobian(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> ir.Function
```

#### 返回

返回 Jacobian 函数，计算所有输出对所有参数的梯度矩阵。

#### 使用示例

```python
@pl.function
def multi_output(x: pl.Tensor[[3], pl.FP32]) -> Tuple[pl.Tensor[[3], pl.FP32], pl.Tensor[[3], pl.FP32]]:
    y1 = pl.mul(x, 2.0)
    y2 = pl.add(x, 1.0)
    return y1, y2

jacobian_func = pl.jacobian(multi_output)
# jacobian_func(x) -> J (Jacobian matrix)
# J[i, j] = dy_i / dx_j
```

---

### 2.4 pl.vjp

生成 Vector-Jacobian Product 函数，用于高效梯度计算。

#### 函数签名

```python
def vjp(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None
) -> Tuple[ir.Function, Callable]
```

#### 返回

返回 `(forward_func, vjp_func)`：

- `forward_func`: 前向函数，返回 `(output, vjp_closure)`
- `vjp_closure`: 可调用的 VJP 函数，`(v, ...) -> grads`

#### 使用示例

```python
@pl.function
def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.exp(x)

forward, vjp_closure = pl.vjp(f)

# 使用方式
# output, vjp_fn = forward(x)
# grads = vjp_fn(d_output)  # d_output 是梯度向量
```

---

## 3. 高级 API

### 3.1 pl.CustomGrad

支持自定义梯度规则。

#### 函数签名

```python
def CustomGrad(
    forward_func: Callable,
    backward_func: Callable,
    name: Optional[str] = None
) -> ir.Op
```

#### 使用示例

```python
# 自定义 ReLU 梯度
def relu_forward(x):
    return pl.max(x, 0.0)

def relu_backward(x, d_output):
    mask = pl.gt(x, 0.0)
    return pl.mul(d_output, mask)

relu_op = pl.CustomGrad(relu_forward, relu_backward)

# 在函数中使用
@pl.function
def my_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return relu_op(x)
```

---

### 3.2 pl.stop_gradient

停止梯度传播（用于不需要梯度的部分）。

#### 函数签名

```python
def stop_gradient(expr: ir.Expr) -> ir.Expr
```

#### 使用示例

```python
@pl.function
def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    # x 的这部分不参与梯度计算
    x_fixed = pl.stop_gradient(x)
    return pl.mul(x, x_fixed)
# 梯度: dy/dx = x_fixed (只有乘法的第一个 x 参与梯度)
```

---

### 3.3 pl.checkpoint

内存优化：选择性保存前向值，减少内存占用。

#### 函数签名

```python
def checkpoint(
    func: Union[ir.Function, Callable],
    checkpoint_policy: str = 'auto'
) -> ir.Function
```

#### 参数

| policy | 描述 |
|--------|------|
| `'auto'` | 自动选择保存点 |
| `'full'` | 保存所有中间值 |
| `'none'` | 不保存，反向时重新计算 |
| `' selective'` | 用户指定保存点 |

#### 使用示例

```python
@pl.function
def large_network(x: pl.Tensor[[1024], pl.FP32]) -> pl.Tensor[[1024], pl.FP32]:
    # 多层计算
    h1 = pl.layer1(x)
    h2 = pl.layer2(h1)
    h3 = pl.layer3(h2)
    return h3

# 使用 checkpoint 减少内存
checkpointed_func = pl.checkpoint(large_network, policy='selective')
grad_func = pl.grad(checkpointed_func)
```

---

## 4. 算子级梯度注册

### 4.1 注册梯度规则

```python
# 注册算子的梯度规则
@pl.register_grad('tile.matmul')
def matmul_grad(saved_values, d_output):
    A, B = saved_values
    # d_A = d_C @ B.T
    B_T = pl.transpose(B)
    d_A = pl.matmul(d_output, B_T)
    
    # d_B = A.T @ d_C
    A_T = pl.transpose(A)
    d_B = pl.matmul(A_T, d_output)
    
    return d_A, d_B
```

### 4.2 内置算子梯度表

| 算子 | 梯度规则 |
|------|----------|
| `tile.add` | `ā = ȳ, b̄ = ȳ` |
| `tile.sub` | `ā = ȳ, b̄ = -ȳ` |
| `tile.mul` | `ā = ȳ·b, b̄ = ȳ·a` |
| `tile.div` | `ā = ȳ/b, b̄ = -ȳ·a/b²` |
| `tile.exp` | `ā = ȳ·exp(a)` |
| `tile.log` | `ā = ȳ/a` |
| `tile.sqrt` | `ā = ȳ/(2·sqrt(a))` |
| `tile.relu` | `ā = ȳ·(a>0)` |
| `tile.matmul` | `Ā = ȳ@B^T, B̄ = A^T@ȳ` |
| `tile.transpose` | `̄ = transpose(ȳ)` |
| `tile.sin` | `ā = ȳ·cos(a)` |
| `tile.cos` | `ā = -ȳ·sin(a)` |

---

## 5. 完整实现设计

### 5.1 模块结构

```
python/pypto/autodiff/
├── __init__.py           # 高层 API (grad, value_and_grad, ...)
├── reverse_mode.py       # 反向模式核心实现
├── liveness.py           # 活性分析
├── grad_rules.py         # 算子梯度规则注册表
├── gradient_registry.py  # 梯度规则注册机制
├── checkpoint.py         # 内存优化 checkpoint
├── vjp.py                # Vector-Jacobian Product
├── custom_grad.py        # 自定义梯度支持
└── utils.py              # 辅助工具
```

### 5.2 高层 API 实现

```python
# python/pypto/autodiff/__init__.py

from typing import Union, List, Optional, Tuple, Callable
from pypto import ir
from pypto.ir import Function
from .reverse_mode import ReverseModeMutator
from .liveness import LivenessAnalyzer
from .grad_rules import get_grad_rule

__all__ = [
    'grad',
    'value_and_grad',
    'jacobian',
    'vjp',
    'CustomGrad',
    'stop_gradient',
    'checkpoint',
    'register_grad',
]


def grad(
    func: Union[Function, Callable],
    params: Optional[List[str]] = None,
    grad_output: Optional[str] = None,
    name: Optional[str] = None
) -> Function:
    """
    Generate gradient function for a forward function.
    
    Args:
        func: Forward function (IR or Python DSL)
        params: Parameter names to compute gradients for (default: all inputs)
        grad_output: Output variable to compute gradient from (default: first output)
        name: Name for gradient function (default: grad_{func_name})
    
    Returns:
        Gradient function IR
    
    Example:
        @pl.function
        def forward(x, y):
            return pl.add(x, y)
        
        backward = pl.grad(forward, params=['x', 'y'])
        # backward(x, y, d_output) -> (d_x, d_y)
    """
    # 获取 IR Function
    if callable(func) and not isinstance(func, Function):
        # 从 Python DSL 获取 IR
        func_ir = _get_ir_from_callable(func)
    else:
        func_ir = func
    
    # 确定参数
    if params is None:
        params = [p.name_hint for p in func_ir.params]
    
    # 找到参数 IR Var
    param_vars = []
    for p_name in params:
        for param in func_ir.params:
            if param.name_hint == p_name:
                param_vars.append(param)
                break
    
    # 生成反向函数
    mutator = ReverseModeMutator(param_vars, grad_output)
    backward_ir = mutator.differentiate(func_ir)
    
    # 设置名称
    if name:
        backward_ir.name = name
    
    return backward_ir


def value_and_grad(
    func: Union[Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> Tuple[Function, Function]:
    """
    Generate forward and gradient functions with tape recording.
    
    Args:
        func: Forward function
        params: Parameter names to compute gradients for
        name: Base name for generated functions
    
    Returns:
        (forward_func, backward_func) tuple
    
    Example:
        forward, backward = pl.value_and_grad(matmul)
        # forward(A, B) -> C, tape
        # backward(tape, d_C) -> (d_A, d_B)
    """
    func_ir = _ensure_ir_function(func)
    
    if params is None:
        params = [p.name_hint for p in func_ir.params]
    
    param_vars = _find_param_vars(func_ir, params)
    
    # 生成前向函数（带 tape）
    forward_mutator = ForwardWithTapeMutator(param_vars)
    forward_ir = forward_mutator.transform(func_ir)
    
    # 生成反向函数
    backward_mutator = ReverseModeMutator(param_vars)
    backward_ir = backward_mutator.differentiate(func_ir)
    
    if name:
        forward_ir.name = name
        backward_ir.name = f"grad_{name}"
    
    return forward_ir, backward_ir


def jacobian(
    func: Union[Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> Function:
    """
    Generate Jacobian computation function.
    
    Computes Jacobian matrix J where J[i,j] = d(output_i) / d(param_j).
    
    Args:
        func: Forward function
        params: Parameter names for Jacobian computation
        name: Function name
    
    Returns:
        Jacobian function IR
    
    Example:
        @pl.function
        def f(x):
            return pl.mul(x, 2.0), pl.add(x, 1.0)
        
        J = pl.jacobian(f)
        # J(x) -> [[2.0], [1.0]]  (Jacobian matrix)
    """
    func_ir = _ensure_ir_function(func)
    
    if params is None:
        params = [p.name_hint for p in func_ir.params]
    
    # 对每个输出计算梯度
    jacobian_builder = JacobianBuilder(func_ir, params)
    jacobian_ir = jacobian_builder.build()
    
    return jacobian_ir


def vjp(
    func: Union[Function, Callable],
    params: Optional[List[str]] = None
) -> Tuple[Function, Callable]:
    """
    Generate Vector-Jacobian Product functions.
    
    VJP is the core primitive for reverse-mode AD.
    
    Args:
        func: Forward function
        params: Parameter names
    
    Returns:
        (forward, vjp_closure) tuple
    
    Example:
        forward, vjp_fn = pl.vjp(f)
        output, vjp_closure = forward(x)
        grads = vjp_closure(v)  # v @ J
    """
    func_ir = _ensure_ir_function(func)
    
    if params is None:
        params = [p.name_hint for p in func_ir.params]
    
    param_vars = _find_param_vars(func_ir, params)
    
    # 构建 VJP
    vjp_builder = VJPBuilder(func_ir, param_vars)
    forward_ir, vjp_closure = vjp_builder.build()
    
    return forward_ir, vjp_closure


def stop_gradient(expr: ir.Expr) -> ir.Expr:
    """
    Stop gradient propagation through an expression.
    
    The expression value is used in forward computation,
    but treated as constant in gradient computation.
    
    Args:
        expr: Expression to stop gradient
    
    Returns:
        Expression with gradient stopped
    
    Example:
        @pl.function
        def f(x):
            x_const = pl.stop_gradient(x)
            return pl.mul(x, x_const)  # dy/dx = x_const only
    """
    return ir.Call(ir.Op("autodiff.stop_gradient"), [expr], {}, expr.type, expr.span)


def checkpoint(
    func: Union[Function, Callable],
    checkpoint_policy: str = 'auto'
) -> Function:
    """
    Apply checkpointing for memory optimization.
    
    Args:
        func: Forward function
        checkpoint_policy: Checkpoint strategy ('auto', 'full', 'none', 'selective')
    
    Returns:
        Checkpointed function
    
    Example:
        @pl.function
        def large_net(x):
            h1 = layer1(x)
            h2 = layer2(h1)
            return h2
        
        checkpointed = pl.checkpoint(large_net)
        grad_func = pl.grad(checkpointed)  # Less memory usage
    """
    func_ir = _ensure_ir_function(func)
    
    checkpointer = Checkpointer(func_ir, policy=checkpoint_policy)
    return checkpointer.transform()


def register_grad(op_name: str):
    """
    Decorator to register gradient rule for an operator.
    
    Args:
        op_name: Operator name (e.g., 'tile.matmul')
    
    Example:
        @pl.register_grad('tile.custom_op')
        def custom_grad(saved_inputs, d_output):
            # Compute gradients
            return d_input1, d_input2
    """
    def decorator(grad_func):
        GradientRegistry.register(op_name, grad_func)
        return grad_func
    return decorator


# 辅助函数

def _ensure_ir_function(func: Union[Function, Callable]) -> Function:
    """确保输入是 IR Function"""
    if isinstance(func, Function):
        return func
    elif callable(func):
        return _get_ir_from_callable(func)
    else:
        raise TypeError(f"Expected Function or Callable, got {type(func)}")


def _get_ir_from_callable(func: Callable) -> Function:
    """从 Python DSL 函数获取 IR"""
    # 检查是否有 __pypto_ir__ 属性
    if hasattr(func, '__pypto_ir__'):
        return func.__pypto_ir__
    
    # 否则使用 @pl.function 装饰器获取
    # 这需要解析源码，比较复杂
    raise ValueError("Function must be decorated with @pl.function")


def _find_param_vars(func_ir: Function, param_names: List[str]) -> List[ir.Var]:
    """找到参数对应的 IR Var"""
    param_vars = []
    for name in param_names:
        for param in func_ir.params:
            if param.name_hint == name:
                param_vars.append(param)
                break
        else:
            raise ValueError(f"Parameter '{name}' not found in function")
    return param_vars
```

---

## 6. 梯度规则注册实现

```python
# python/pypto/autodiff/gradient_registry.py

from typing import Dict, Callable, List, Any
from pypto import ir


class GradientRegistry:
    """
    全局梯度规则注册表
    
    存储所有算子的反向梯度计算规则
    """
    
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, op_name: str, grad_func: Callable):
        """注册算子梯度规则"""
        cls._registry[op_name] = grad_func
    
    @classmethod
    def get(cls, op_name: str) -> Optional[Callable]:
        """获取算子梯度规则"""
        return cls._registry.get(op_name)
    
    @classmethod
    def has(cls, op_name: str) -> bool:
        """检查是否已注册"""
        return op_name in cls._registry
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册的算子"""
        return list(cls._registry.keys())


# 梯度规则签名
# def grad_rule(saved_inputs: List[ir.Expr], d_output: ir.Expr, kwargs: Dict) -> List[ir.Expr]:
#     """计算梯度"""
#     return [d_input1, d_input2, ...]


# 内置梯度规则注册

@register_grad('tile.add')
def add_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    return [d_output, d_output]  # ā = ȳ, b̄ = ȳ


@register_grad('tile.sub')
def sub_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    return [d_output, ir.Neg(d_output, d_output.type.dtype, d_output.span)]


@register_grad('tile.mul')
def mul_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    a_grad = ir.Mul(d_output, b, d_output.type.dtype, d_output.span)
    b_grad = ir.Mul(d_output, a, d_output.type.dtype, d_output.span)
    return [a_grad, b_grad]


@register_grad('tile.div')
def div_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    span = d_output.span
    dtype = d_output.type.dtype
    
    a_grad = ir.FloatDiv(d_output, b, dtype, span)
    b_squared = ir.Mul(b, b, dtype, span)
    neg_a_times_d = ir.Neg(ir.Mul(a, d_output, dtype, span), dtype, span)
    b_grad = ir.FloatDiv(neg_a_times_d, b_squared, dtype, span)
    
    return [a_grad, b_grad]


@register_grad('tile.exp')
def exp_grad(saved_inputs, d_output, kwargs):
    a = saved_inputs[0]
    span = d_output.span
    
    # ā = ȳ * exp(a)
    # 使用前向保存的结果（或重新计算）
    exp_a = kwargs.get('forward_result')
    if exp_a is None:
        exp_a = ir.Call(ir.Op('tile.exp'), [a], {}, a.type, span)
    
    a_grad = ir.Mul(d_output, exp_a, d_output.type.dtype, span)
    return [a_grad]


@register_grad('tile.log')
def log_grad(saved_inputs, d_output, kwargs):
    a = saved_inputs[0]
    span = d_output.span
    
    # ā = ȳ / a
    a_grad = ir.FloatDiv(d_output, a, d_output.type.dtype, span)
    return [a_grad]


@register_grad('tile.sqrt')
def sqrt_grad(saved_inputs, d_output, kwargs):
    a = saved_inputs[0]
    span = d_output.span
    
    # ā = ȳ / (2 * sqrt(a))
    sqrt_a = kwargs.get('forward_result')
    if sqrt_a is None:
        sqrt_a = ir.Call(ir.Op('tile.sqrt'), [a], {}, a.type, span)
    
    two = ir.ConstFloat(2.0, d_output.type.dtype, span)
    two_sqrt = ir.Mul(two, sqrt_a, d_output.type.dtype, span)
    a_grad = ir.FloatDiv(d_output, two_sqrt, d_output.type.dtype, span)
    
    return [a_grad]


@register_grad('tile.relu')
def relu_grad(saved_inputs, d_output, kwargs):
    a = saved_inputs[0]
    span = d_output.span
    
    # ā = ȳ * (a > 0)
    zero = ir.ConstFloat(0.0, a.type.dtype, span)
    mask = ir.Call(ir.Op('tile.gt'), [a, zero], {}, ir.TileType(...), span)
    mask_casted = ir.Call(ir.Op('tile.cast'), [mask], {'dtype': d_output.type.dtype}, 
                          d_output.type, span)
    a_grad = ir.Mul(d_output, mask_casted, d_output.type.dtype, span)
    
    return [a_grad]


@register_grad('tile.matmul')
def matmul_grad(saved_inputs, d_output, kwargs):
    A, B = saved_inputs
    span = d_output.span
    
    # d_A = d_C @ B.T
    B_T = ir.Call(ir.Op('tile.transpose'), [B], {}, B.type, span)
    A_grad = ir.Call(ir.Op('tile.matmul'), [d_output, B_T], {}, A.type, span)
    
    # d_B = A.T @ d_C
    A_T = ir.Call(ir.Op('tile.transpose'), [A], {}, A.type, span)
    B_grad = ir.Call(ir.Op('tile.matmul'), [A_T, d_output], {}, B.type, span)
    
    return [A_grad, B_grad]


@register_grad('tile.transpose')
def transpose_grad(saved_inputs, d_output, kwargs):
    # transpose 的梯度是再次 transpose
    return [ir.Call(ir.Op('tile.transpose'), [d_output], {}, saved_inputs[0].type, d_output.span)]


@register_grad('tile.sin')
def sin_grad(saved_inputs, d_output, kwargs):
    a = saved_inputs[0]
    span = d_output.span
    
    # ā = ȳ * cos(a)
    cos_a = ir.Call(ir.Op('tile.cos'), [a], {}, a.type, span)
    a_grad = ir.Mul(d_output, cos_a, d_output.type.dtype, span)
    
    return [a_grad]


@register_grad('tile.cos')
def cos_grad(saved_inputs, d_output, kwargs):
    a = saved_inputs[0]
    span = d_output.span
    
    # ā = -ȳ * sin(a)
    sin_a = ir.Call(ir.Op('tile.sin'), [a], {}, a.type, span)
    neg_sin = ir.Neg(ir.Mul(d_output, sin_a, d_output.type.dtype, span), 
                     d_output.type.dtype, span)
    
    return [neg_sin]
```

---

## 7. CustomGrad 实现

```python
# python/pypto/autodiff/custom_grad.py

from typing import Callable, Optional
from pypto import ir


class CustomGradOp(ir.Op):
    """自定义梯度算子"""
    
    def __init__(
        self,
        name: str,
        forward_func: Callable,
        backward_func: Callable,
        span: ir.Span = ir.Span.unknown()
    ):
        super().__init__(name, {})
        self.forward_func = forward_func
        self.backward_func = backward_func
        self.span = span


def CustomGrad(
    forward_func: Callable,
    backward_func: Callable,
    name: Optional[str] = None
) -> CustomGradOp:
    """
    创建自定义梯度算子
    
    Args:
        forward_func: 前向计算函数
        backward_func: 反向梯度函数
        name: 算子名称
    
    Returns:
        自定义梯度算子
    
    Example:
        def my_relu_forward(x):
            return pl.max(x, 0.0)
        
        def my_relu_backward(x, d_output):
            mask = pl.gt(x, 0.0)
            return pl.mul(d_output, mask)
        
        relu_op = pl.CustomGrad(my_relu_forward, my_relu_backward)
        
        @pl.function
        def f(x):
            return relu_op(x)
    """
    if name is None:
        name = f"custom_grad_{forward_func.__name__}"
    
    op = CustomGradOp(name, forward_func, backward_func)
    
    # 注册到梯度规则表
    from .gradient_registry import GradientRegistry
    
    def custom_grad_rule(saved_inputs, d_output, kwargs):
        return backward_func(*saved_inputs, d_output)
    
    GradientRegistry.register(name, custom_grad_rule)
    
    return op
```

---

## 8. 完整使用示例

### 8.1 基本梯度计算

```python
import pypto as pl

@pl.function
def simple_add(x: pl.Tensor[[64], pl.FP32], 
               y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.add(x, y)

# 生成梯度函数
grad_func = pl.grad(simple_add)

# 查看梯度函数签名
print(grad_func)
# def grad_simple_add(x, y, d_output) -> (d_x, d_y)

# 验证梯度正确性
# 前向: result = x + y
# 反向: d_x = d_output, d_y = d_output
```

### 8.2 矩阵乘法梯度

```python
@pl.function
def matmul_func(A: pl.Tile[[16, 16], pl.FP16],
                B: pl.Tile[[16, 16], pl.FP16]) -> pl.Tile[[16, 16], pl.FP16]:
    return pl.matmul(A, B)

grad_matmul = pl.grad(matmul_func)

# 梯度函数签名
# def grad_matmul(A, B, d_C) -> (d_A, d_B)
# 其中:
#   d_A = d_C @ transpose(B)
#   d_B = transpose(A) @ d_C
```

### 8.3 带循环的梯度

```python
@pl.function
def accumulate(x: pl.Tensor[[64], pl.FP32],
               n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
    acc: pl.Tensor[[64], pl.FP32] = pl.zeros([64], dtype=pl.FP32)
    for i, (acc_iter,) in pl.range(n, init_values=(acc,)):
        acc_new: pl.Tensor[[64], pl.FP32] = pl.add(acc_iter, x)
        acc_out = pl.yield_(acc_new)
    return acc_out

grad_accumulate = pl.grad(accumulate, params=['x'])

# 梯度: d_x = n * d_output
```

### 8.4 嵌套梯度计算

```python
@pl.function
def nested(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    z: pl.Tensor[[64], pl.FP32] = pl.add(y, x)  # x 使用两次
    return z

grad_nested = pl.grad(nested)

# 梯度: d_x = 2.0 + 1.0 = 3.0 (累加)
```

### 8.5 自定义梯度

```python
def sigmoid_forward(x):
    return pl.div(1.0, pl.add(1.0, pl.exp(pl.neg(x))))

def sigmoid_backward(x, d_output):
    # σ(x) = 1/(1+e^{-x})
    # dσ/dx = σ(x)(1-σ(x))
    sigma = sigmoid_forward(x)
    one_minus_sigma = pl.sub(1.0, sigma)
    grad = pl.mul(sigma, one_minus_sigma)
    return pl.mul(d_output, grad)

sigmoid_op = pl.CustomGrad(sigmoid_forward, sigmoid_backward)

@pl.function
def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return sigmoid_op(x)

grad_f = pl.grad(f)
```

---

## 9. DSL 集成

### 9.1 语言模块导出

```python
# python/pypto/language/__init__.py 添加

from pypto.autodiff import (
    grad,
    value_and_grad,
    jacobian,
    vjp,
    CustomGrad,
    stop_gradient,
    checkpoint,
    register_grad,
)

__all__ = [
    # ... 现有导出
    'grad',
    'value_and_grad',
    'jacobian',
    'vjp',
    'CustomGrad',
    'stop_gradient',
    'checkpoint',
    'register_grad',
]
```

### 9.2 函数装饰器扩展

```python
# 扩展 @pl.function 支持自动生成梯度

@pl.function(with_grad=True, grad_params=['x', 'w'])
def linear(x: pl.Tensor[[64], pl.FP32],
           w: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.mul(x, w)

# linear.forward() -> 前向函数
# linear.backward() -> 反向函数
# linear.grad() -> 梯度函数
```

---

## 10. API 稳定性保证

| API | 稳定性 | 说明 |
|------|--------|------|
| `pl.grad` | 稳定 | 核心接口，长期支持 |
| `pl.value_and_grad` | 稳定 | 核心接口 |
| `pl.jacobian` | 实验性 | 多输出场景，可能调整 |
| `pl.vjp` | 实验性 | 高级用户使用 |
| `pl.CustomGrad` | 稳定 | 自定义梯度支持 |
| `pl.stop_gradient` | 稳定 | 常用功能 |
| `pl.checkpoint` | 实验性 | 内存优化策略 |
| `@register_grad` | 稳定 | 算子扩展机制 |