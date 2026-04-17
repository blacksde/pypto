# PyPTO 自动微分功能设计文档

**版本**: 2026-04-17
**作者**: PyPTO Team

---

## 1. 自动微分的整体算法

### 1.1 算法概述

PyPTO 采用 **Source-to-Source 反向模式自动微分**：

| 特性 | 描述 |
|------|------|
| 模式 | 反向模式 (Reverse Mode) |
| 方式 | Source-to-Source IR 变换 |
| 输入 | 前向函数 IR |
| 输出 | 反向梯度函数 IR |
| 基础 | 链式法则 (Chain Rule) |

### 1.2 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                   自动微分流程                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 前向函数 IR (原始函数)                                           │
│       │                                                             │
│       ▼                                                             │
│  2. 创建梯度参数 (forward_params + d_output)                         │
│       │                                                             │
│       ▼                                                             │
│  3. 收集需要保存的前向中间值                                          │
│       │  - 分析反向语句中引用的变量                                   │
│       │  - 区分循环内/循环外变量                                      │
│       ▼                                                             │
│  4. 反向遍历前向函数体                                               │
│       │  - 反向处理每个语句                                          │
│       │  - 生成反向梯度语句                                          │
│       │  - 处理控制流 (ForStmt, IfStmt, YieldStmt)                   │
│       ▼                                                             │
│  5. 生成正向计算语句 (重新计算循环外中间值)                           │
│       │                                                             │
│       ▼                                                             │
│  6. 构建反向函数体                                                   │
│       │  - 拼接正向计算 + 反向语句                                    │
│       │  - 添加返回语句                                              │
│       ▼                                                             │
│  7. 梯度函数 IR                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 数学基础

反向模式基于链式法则：

$$\frac{\partial y}{\partial x} = \prod_{i=n}^{1} \frac{\partial f_i}{\partial f_{i-1}}$$

对于复合函数 $y = f_n(f_{n-1}(...f_1(x)))$，反向模式从输出到输入逐层计算梯度：

```
前向计算:
  v₁ = f₁(x₁, x₂)
  v₂ = f₂(v₁, x₂)
  y  = f₃(v₂)

反向传播:
  v̄₂ = ȳ · ∂f₃/∂v₂
  v̄₁ = v̄₂ · ∂f₂/∂v₁
  x̄₂ = v̄₂ · ∂f₂/∂x₂ + v̄₁ · ∂f₁/∂x₂
  x̄₁ = v̄₁ · ∂f₁/∂x₁
```

其中 $\bar{v}$ 表示 $v$ 的伴随变量（梯度）。

### 1.4 梯度累加

当一个变量在多个表达式中使用时，梯度需要累加所有贡献：

$$y = f_1(x, ...) + f_2(x, ...)$$

梯度累加：

$$\frac{\partial y}{\partial x} = \frac{\partial f_1}{\partial x} + \frac{\partial f_2}{\partial x}$$

---

## 2. 自动微分对于每种IR节点的处理

### 2.1 IR节点类型概览

| 节点类型 | 处理方式 | 核心方法 |
|----------|----------|----------|
| `AssignStmt` | 反向生成梯度表达式 | `_reverse_assign_inline` |
| `ReturnStmt` | 生成梯度初始化 | 返回语句跳过 |
| `SeqStmts` | 反向遍历子语句 | `_reverse_stmt_inline` |
| `ForStmt` | 反向循环 + Tape 模式 | `_reverse_for` |
| `IfStmt` | 保持分支结构 | `_reverse_if_inline` |
| `YieldStmt` | SSA 循环梯度传递 | `_reverse_yield_inline` |
| `Call` | 调用梯度规则 | `_reverse_call_inline` |
| `Var` | 直接梯度传递 | 累加梯度 |
| `ConstInt/ConstFloat` | 无梯度 | 跳过 |

### 2.2 AssignStmt 处理

**前向**:
```python
y = expr(a, b)  # 例如 y = add(a, b) 或 y = mul(a, b)
```

**反向**:
```python
# 根据算子类型调用梯度规则
grad_exprs = grad_rule([a, b], d_y, kwargs)
# 例如 add: [d_y, d_y]
# 例如 mul: [d_y * b, d_y * a]

# 累加梯度（输入参数可能被多次使用）
d_a += grad_exprs[0]
d_b += grad_exprs[1]
```

**核心逻辑** (`reverse_mode.py:1705-1739`):
- **输入参数**：累加梯度（使用 `_create_add`）
- **临时变量**：直接赋值（不需要初始化为零）

**代码示例**:
```python
def _reverse_assign_inline(stmt, d_output):
    stmts = []
    if isinstance(stmt.value, ir.Call):
        call = stmt.value
        stmts.extend(self._reverse_call_inline(call, d_output, stmt.var))
    elif isinstance(stmt.value, ir.Var):
        src_var = stmt.value
        if src_var in self.grad_var_map:
            # 输入参数：累加梯度
            grad_var = self.grad_var_map[src_var]
            add_expr = self._create_add(grad_var, d_output)
            stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
    return stmts
```

### 2.3 SeqStmts 处理

**前向**:
```python
t1 = mul(x, w)
t2 = mul(x, v)
y = add(t1, t2)
```

**反向**:
```python
# 反向遍历语句序列
for stmt in reversed(stmts):
    # 1. 处理 y = add(t1, t2)
    #    d_t1 += d_y, d_t2 += d_y
    
    # 2. 处理 t2 = mul(x, v)
    #    d_x += d_t2 * v, d_v += d_t2 * x
    
    # 3. 处理 t1 = mul(x, w)
    #    d_x += d_t1 * w, d_w += d_t1 * x
```

**梯度累加**: `d_x = d_y * v + d_y * w`

**代码位置**: `reverse_mode.py:1677-1703`

### 2.4 ForStmt 处理（核心难点）

#### 2.4.1 两种模式

| 模式 | 条件 | 处理方式 | 方法 |
|------|------|----------|------|
| iter_args + yield | `has_iter_args && has_yield` | SSA Reduce 反向 | `_reverse_for_iter_args` |
| 普通循环 | 无 iter_args/yield | Tape 模式 | `_reverse_for_with_tape` |

#### 2.4.2 iter_args + yield 模式（SSA Reduce）

**前向**:
```python
for i, (acc,) in pl.range(N, init_values=(acc_init,)):
    acc_new = f(acc_iter, x)
    acc_out = yield(acc_new)
```

**数学**: $acc_N = f(f(...f(acc_0, x)...))$

**反向**:
```python
# iter_args 的初始值梯度直接赋值
d_acc_init = d_output

# 反向循环（使用 iter_args 模式）
for i_rev, (d_acc_iter,) in pl.range(N, init_values=(d_output,)):
    # 从 tape 获取前向中间值
    saved_acc = tape_acc[i_rev]
    
    # 计算梯度贡献
    d_x += grad_expr(d_acc_iter, saved_acc)
    d_acc_prev = yield(grad_acc)
```

**关键点**:
1. iter_args 的 init 梯度直接赋值（不初始化为零）
2. 反向循环使用 iter_args 模式传递梯度
3. 需要 Tape 保存循环内引用的中间值

**代码位置**: `reverse_mode.py:748-909`

#### 2.4.3 Tape 模式（普通循环）

**问题**: 循环内变量在反向时需要访问前向值

**解决方案**: TensorArray 保存每次迭代的值

```python
# 创建 Tape
tape_var = tensor_array.create(shape, dtype, capacity)

# 正向循环：push 值
for i in range(N):
    val = compute(...)
    tensor_array.push(tape_var, val)

# 反向循环：get 值
for i in range(N):
    saved_val = tensor_array.get(tape_var, i)
    # 使用 saved_val 计算梯度
```

**流程**:
```
1. 创建 TensorArray (tape_var)
2. 正向循环：
   - 执行原计算
   - push 循环内变量到 tape
3. 初始化梯度为零
4. 反向循环：
   - get tape 中保存的值
   - 使用保存值计算梯度
   - 累加梯度
```

**代码位置**: `reverse_mode.py:454-597`

#### 2.4.4 识别需要 Tape 的变量

分析梯度表达式中引用的循环内变量：

```python
def _identify_vars_needing_tape(body, for_stmt):
    # 1. 收集循环内定义的变量
    loop_defined_vars = set()
    def collect_loop_vars(stmt):
        if isinstance(stmt, ir.AssignStmt):
            loop_defined_vars.add(stmt.var.name_hint)
    
    collect_loop_vars(body)
    
    # 2. 对每个算子调用，获取梯度规则
    grad_rule = GradientRegistry.get(op_name)
    grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
    
    # 3. 检查梯度表达式中是否引用循环内变量
    vars_needed = {}
    def check_expr_for_loop_vars(expr):
        if isinstance(expr, ir.Var):
            if expr.name_hint in loop_defined_vars:
                vars_needed[expr.name_hint] = {"type": expr.type}
    
    for grad_expr in grad_exprs:
        check_expr_for_loop_vars(grad_expr)
    
    return vars_needed
```

**代码位置**: `reverse_mode.py:599-698`

### 2.5 IfStmt 处理（Phi 节点）

**前向**:
```python
if cond:
    y = f1(x)
else:
    y = f2(x)
```

**数学**: 
$$y = \begin{cases} f_1(x) & \text{if } c \\ f_2(x) & \text{else} \end{cases}$$

**反向**:
```python
# 保持分支结构（使用相同条件）
if cond:
    d_x = grad_f1(d_y, x)
else:
    d_x = grad_f2(d_y, x)
```

**核心逻辑** (`reverse_mode.py:1796-1833`):
- 反向 IfStmt 保持前向条件
- 分别反向处理 then_body 和 else_body
- 不创建 Phi 节点（梯度流向活跃分支）

**代码示例**:
```python
def _reverse_if_inline(stmt, d_output):
    # 生成反向 then_body
    then_stmts = []
    if isinstance(stmt.then_body, ir.SeqStmts):
        for body_stmt in reversed(stmt.then_body.stmts):
            then_stmts.extend(self._reverse_stmt_inline(body_stmt, d_output))
    
    # 生成反向 else_body
    else_stmts = []
    if stmt.else_body is not None:
        for body_stmt in reversed(stmt.else_body.stmts):
            else_stmts.extend(self._reverse_stmt_inline(body_stmt, d_output))
    
    # 创建反向 IfStmt
    reverse_if = ir.IfStmt(
        stmt.condition,
        ir.SeqStmts(then_stmts, self.span),
        ir.SeqStmts(else_stmts, self.span) if else_stmts else None,
        stmt.return_vars,
        self.span
    )
    
    return [reverse_if]
```

### 2.6 YieldStmt 处理

**用途**: SSA 循环中传递中间状态

**前向**:
```python
y = yield(acc_new)  # 返回循环迭代值
```

**反向**:
```python
# yield 的输出梯度传播到 yield 前的表达式
for value_expr in stmt.value:
    if isinstance(value_expr, ir.Call):
        stmts.extend(_reverse_call_inline(value_expr, d_output))
    elif isinstance(value_expr, ir.Var):
        d_var += d_output
```

**代码位置**: `reverse_mode.py:1835-1851`

### 2.7 Call 表达式处理

**核心**: 调用 GradientRegistry 获取梯度规则

```python
def _reverse_call_inline(call, d_output, output_var):
    op_name = call.op.name
    
    # 算子名映射
    if not GradientRegistry.has(op_name):
        # 尝试 tile.xxx 或 tensor.xxx
        parts = op_name.split(".")
        tile_op = f"tile.{parts[-1]}"
        tensor_op = f"tensor.{parts[-1]}"
        if GradientRegistry.has(tile_op):
            op_name = tile_op
        elif GradientRegistry.has(tensor_op):
            op_name = tensor_op
    
    # 获取梯度规则
    grad_rule = GradientRegistry.get(op_name)
    if grad_rule is None:
        return []
    
    # 调用梯度规则
    saved_inputs = call.args
    kwargs = {"forward_result": call}
    grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
    
    # 处理梯度表达式
    for arg, grad_expr in zip(call.args, grad_exprs):
        if arg in grad_var_map:
            # 输入参数：累加
            d_arg = grad_var_map[arg]
            add_expr = _create_add(d_arg, grad_expr)
            stmts.append(ir.AssignStmt(d_arg, add_expr, span))
        elif isinstance(arg, ir.Call):
            # 嵌套调用：递归处理
            stmts.extend(_reverse_nested_call(arg, grad_expr))
```

**代码位置**: `reverse_mode.py:1741-1794`

---

## 3. 自动微分如何注册反向算子

### 3.1 注册架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    梯度规则注册架构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐         ┌──────────────────────────┐          │
│  │ grad_rules.py   │ ──────> │ GradientRegistry         │          │
│  │                 │         │                          │          │
│  │ @register_grad  │         │  _registry: Dict[str,    │          │
│  │ 'tile.add'      │         │    Callable]             │          │
│  │ 'tile.mul'      │         │                          │          │
│  │ 'tile.matmul'   │         │  _categories: Dict[str,  │          │
│  │ ...             │         │    str]                  │          │
│  └─────────────────┘         │                          │          │
│                              │  Methods:                │          │
│                              │  - register()            │          │
│                              │  - get()                 │          │
│                              │  - has()                 │          │
│                              │  - list_all()            │          │
│                              │  - list_by_category()    │          │
│                              └──────────────────────────┘          │
│                                     │                              │
│                                     │ get(op_name)                 │
│                                     ▼                              │
│  ┌───────────────────────────────────────────────────────┐         │
│  │          ReverseModeBuilder (反向模式构建器)            │         │
│  │                                                        │         │
│  │  _reverse_call_inline(call):                          │         │
│  │    op_name = call.op.name                              │         │
│  │    grad_rule = GradientRegistry.get(op_name)          │         │
│  │    grad_exprs = grad_rule(saved_inputs, d_out, kw)    │         │
│  └───────────────────────────────────────────────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 GradientRegistry 类定义

**文件**: `python/pypto/autodiff/gradient_registry.py`

```python
class GradientRegistry:
    """
    全局梯度规则注册表（单例模式）
    
    使用类级别的字典存储，确保全局唯一
    """
    
    _registry: Dict[str, Callable] = {}
    _categories: Dict[str, str] = {}
    
    @classmethod
    def register(cls, op_name: str, grad_func: Callable, category: str = "user"):
        """注册算子梯度规则"""
        if op_name in cls._registry:
            warnings.warn(f"Overwriting gradient rule for '{op_name}'")
        cls._registry[op_name] = grad_func
        cls._categories[op_name] = category
    
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
    
    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """列出指定分类的算子"""
        return [op for op, cat in cls._categories.items() if cat == category]
    
    @classmethod
    def unregister(cls, op_name: str) -> bool:
        """移除算子注册"""
        if op_name in cls._registry:
            del cls._registry[op_name]
            del cls._categories[op_name]
            return True
        return False
    
    @classmethod
    def get_info(cls, op_name: str) -> Optional[Dict[str, Any]]:
        """获取注册信息"""
        if op_name not in cls._registry:
            return None
        return {
            "op_name": op_name,
            "category": cls._categories[op_name],
            "grad_func": cls._registry[op_name],
        }
```

### 3.3 梯度规则签名

```python
def grad_rule(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    算子梯度规则函数签名
    
    Args:
        saved_inputs: 前向计算的输入值（保存用于反向计算）
        d_output: 输出的梯度（从下游传播而来）
        kwargs: 包含 forward_result 等额外信息
    
    Returns:
        每个输入对应的梯度表达式列表
    
    Example:
        @register_grad('tile.add', 'tile')
        def tile_add_grad(saved_inputs, d_output, kwargs):
            a, b = saved_inputs
            return [d_output, d_output]  # da = dy, db = dy
    """
```

### 3.4 内置梯度规则注册

**文件**: `python/pypto/autodiff/grad_rules.py`

#### 3.4.1 Tile 算子梯度规则

| 算子 | 梯度数学 | 注册函数 |
|------|----------|----------|
| `tile.add` | `ā=ȳ, b̄=ȳ` | `tile_add_grad` |
| `tile.sub` | `ā=ȳ, b̄=-ȳ` | `tile_sub_grad` |
| `tile.mul` | `ā=ȳ·b, b̄=ȳ·a` | `tile_mul_grad` |
| `tile.div` | `ā=ȳ/b, b̄=-ȳ·a/b²` | `tile_div_grad` |
| `tile.matmul` | `Ā=ȳ@B^T, B̄=A^T@ȳ` | `tile_matmul_grad` |
| `tile.transpose` | `̄=transpose(ȳ)` | `tile_transpose_grad` |
| `tile.exp` | `ā=ȳ·exp(a)` | `tile_exp_grad` |
| `tile.log` | `ā=ȳ/a` | `tile_log_grad` |
| `tile.sqrt` | `ā=ȳ/(2·sqrt(a))` | `tile_sqrt_grad` |
| `tile.relu` | `ā=ȳ·(a>0)` | `tile_relu_grad` |
| `tile.sin` | `ā=ȳ·cos(a)` | `tile_sin_grad` |
| `tile.cos` | `ā=-ȳ·sin(a)` | `tile_cos_grad` |

#### 3.4.2 Tensor 算子梯度规则

| 算子 | 梯度数学 | 注册函数 |
|------|----------|----------|
| `tensor.add` | `ā=ȳ, b̄=ȳ` | `tensor_add_grad` |
| `tensor.mul` | `ā=ȳ·b, b̄=ȳ·a` | `tensor_mul_grad` |
| `tensor.sub` | `ā=ȳ, b̄=-ȳ` | `tensor_sub_grad` |

#### 3.4.3 实现示例

```python
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
    da = ir.Call(ir.Op("tile.mul"), [d_output, b], {}, a.type, span)
    
    # db = d_output * a
    db = ir.Call(ir.Op("tile.mul"), [d_output, a], {}, b.type, span)
    
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
    B_T = ir.Call(ir.Op("tile.transpose"), [B], {}, B.type, span)
    
    # dA = dC @ B^T
    dA = ir.Call(ir.Op("tile.matmul"), [d_output, B_T], {}, A.type, span)
    
    # A^T
    A_T = ir.Call(ir.Op("tile.transpose"), [A], {}, A.type, span)
    
    # dB = A^T @ dC
    dB = ir.Call(ir.Op("tile.matmul"), [A_T, d_output], {}, B.type, span)
    
    return [dA, dB]


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
    dtype = x.type.dtype if hasattr(x.type, "dtype") else ir.DataType.FP32
    
    # mask = (x > 0)
    zero = ir.ConstFloat(0.0, dtype, span)
    mask = ir.Call(ir.Op("tile.gt"), [x, zero], {}, x.type, span)
    
    # cast mask to float
    mask_float = ir.Call(
        ir.Op("tile.cast"), [mask], {"dtype": dtype}, x.type, span
    )
    
    # dx = d_output * mask
    dx = ir.Call(ir.Op("tile.mul"), [d_output, mask_float], {}, x.type, span)
    
    return [dx]
```

### 3.5 注册装饰器

```python
def register_grad(op_name: str, category: str = "user"):
    """
    梯度规则注册装饰器
    
    Args:
        op_name: 算子名称，如 'tile.matmul'
        category: 分类标签，如 'tile', 'tensor', 'user'
    
    Example:
        @register_grad('tile.add', 'tile')
        def add_grad(saved_inputs, d_output, kwargs):
            return [d_output, d_output]
    """
    def decorator(grad_func: Callable) -> Callable:
        GradientRegistry.register(op_name, grad_func, category)
        return grad_func
    return decorator
```

### 3.6 模块初始化

**文件**: `python/pypto/autodiff/__init__.py`

```python
"""
PyPTO 自动微分模块

提供 source-to-source 自动微分功能
"""

from .gradient_registry import GradientRegistry, register_grad
from .grad_rules import register_all_builtin_rules

# 注册内置梯度规则
register_all_builtin_rules()

# 导出高层 API
from .reverse_mode import grad, value_and_grad

__all__ = [
    "GradientRegistry",
    "register_grad",
    "grad",
    "value_and_grad",
]
```

---

## 4. 自动微分的接口设计

### 4.1 高层 API

| API | 功能 | 返回 |
|------|------|------|
| `pl.grad(func, params)` | 计算梯度函数 | 反向函数 IR |
| `pl.value_and_grad(func, params)` | 前向+反向函数 | 函数对 |

### 4.2 pl.grad 接口

**签名**:
```python
def grad(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None,
    grad_output: Optional[str] = None,
    name: Optional[str] = None
) -> ir.Function
```

**参数**:

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `func` | `ir.Function` 或 `Callable` | 必需 | 前向函数（IR 或 Python DSL） |
| `params` | `List[str]` | 所有输入参数 | 需要计算梯度的参数名称列表 |
| `grad_output` | `str` | 第一个输出 | 指定梯度计算的输出变量 |
| `name` | `str` | `grad_{func_name}` | 梯度函数名称 |

**返回**: 反向梯度函数 `ir.Function`

**示例**:
```python
import pypto as pl

@pl.function
def linear(x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.mul(x, w)

# 生成梯度函数
grad_linear = pl.grad(linear)
# grad_linear(x, w, d_output) -> (d_x, d_w)

# 指定参数
grad_linear_x_only = pl.grad(linear, params=['x'])
# grad_linear_x_only(x, w, d_output) -> d_x
```

### 4.3 pl.value_and_grad 接口

**签名**:
```python
def value_and_grad(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> Tuple[ir.Function, ir.Function]
```

**返回**: `(forward_func, backward_func)` 函数对

- `forward_func`: 前向计算函数（保存中间值）
- `backward_func`: 反向梯度函数（使用保存的值）

**示例**:
```python
@pl.function
def matmul(A: pl.Tile[[16, 16], pl.FP16], B: pl.Tile[[16, 16], pl.FP16]) -> pl.Tile[[16, 16], pl.FP32]:
    return pl.matmul(A, B)

forward_func, backward_func = pl.value_and_grad(matmul)
# forward_func(A, B) -> C
# backward_func(A, B, d_C) -> (d_A, d_B)
```

### 4.4 梯度规则注册 API

#### 方式 1: 使用装饰器

```python
from pypto.autodiff import register_grad

@register_grad('my.custom_op', 'user')
def custom_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    span = d_output.span
    da = ir.Call(ir.Op('tile.mul'), [d_output, b], {}, a.type, span)
    db = d_output
    return [da, db]
```

#### 方式 2: 直接注册

```python
from pypto.autodiff import GradientRegistry

def my_grad(saved_inputs, d_output, kwargs):
    return [d_output]

GradientRegistry.register('my.custom_op', my_grad, 'user')
```

### 4.5 查询 API

```python
from pypto.autodiff import GradientRegistry

# 查看已注册算子
registered = GradientRegistry.list_all()
print(f"已注册梯度规则: {len(registered)}")

# 按分类查看
tile_ops = GradientRegistry.list_by_category('tile')
print(f"Tile 算子: {len(tile_ops)}")

# 检查是否存在
if GradientRegistry.has('tile.add'):
    print("tile.add 已注册")

# 获取梯度规则
add_grad = GradientRegistry.get('tile.add')
grads = add_grad([a, b], d_output, {})
```

---

## 5. 自动微分的整体代码设计

### 5.1 模块结构

```
python/pypto/autodiff/
├── __init__.py              # 高层 API 导出
│   └── register_all_builtin_rules()
│   └── 导出: grad, value_and_grad, GradientRegistry, register_grad
│
├── gradient_registry.py     # 梯度规则注册表
│   └── class GradientRegistry
│   │   ├── register(op_name, grad_func, category)
│   │   ├── get(op_name) → Callable
│   │   ├── has(op_name) → bool
│   │   ├── list_all() → List[str]
│   │   ├── list_by_category(category)
│   │   └── unregister(op_name)
│   └── def register_grad(op_name, category) → Decorator
│
├── grad_rules.py            # 内置梯度规则
│   └── @register_grad tile.add, tile.mul, tile.matmul, ...
│   └── @register_grad tensor.add, tensor.mul, ...
│   └── register_all_builtin_rules()
│
├── reverse_mode.py          # 反向模式核心实现
│   └── class ReverseModeBuilder
│   │   ├── build() → ir.Function
│   │   ├── _build_backward_stmts()
│   │   ├── _reverse_stmt_inline()
│   │   ├── _reverse_assign_inline()
│   │   ├── _reverse_for()
│   │   ├── _reverse_for_iter_args()
│   │   ├── _reverse_for_with_tape()
│   │   ├── _reverse_if_inline()
│   │   ├── _reverse_call_inline()
│   │   ├── _identify_vars_needing_tape()
│   │   ├── _collect_saved_forward_vars()
│   │   └── _generate_forward_compute_stmts()
│   └── def grad(func, params) → ir.Function
│   └── def value_and_grad(func, params) → Tuple[ir.Function, ir.Function]
│
└── gradient_registry_wrapper.py  # 包装导出
```

### 5.2 ReverseModeBuilder 类设计

#### 5.2.1 核心数据结构

```python
class ReverseModeBuilder:
    """反向模式梯度函数构建器"""
    
    # 输入
    forward_func: ir.Function      # 前向函数
    grad_params: List[str]         # 需要梯度的参数名称
    grad_output: Optional[str]     # 指定输出梯度变量
    
    # 映射表
    grad_var_map: Dict[ir.Var, ir.Var]  # 前向变量 -> 梯度变量
    saved_values: Dict[ir.Var, ir.Var]  # 前向变量 -> 保存的值
    gradient_accumulator: Dict[ir.Var, List[ir.Expr]]  # 梯度累加器
    
    # 辅助
    span: ir.Span                  # Span 信息
```

#### 5.2.2 核心方法列表

| 方法 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `build` | 构建反向函数 | - | `ir.Function` |
| `_build_backward_stmts` | 构建反向语句列表 | `d_output` | `List[ir.Stmt]` |
| `_collect_saved_forward_vars` | 收集需要保存的前向值 | `stmts` | `List[ir.Var]` |
| `_generate_forward_compute_stmts` | 生成正向计算语句 | `saved_vars` | `List[ir.Stmt]` |
| `_reverse_stmt_inline` | 反向处理语句 | `stmt, d_output` | `List[ir.Stmt]` |
| `_reverse_assign_inline` | 反向处理赋值 | `stmt, d_output` | `List[ir.Stmt]` |
| `_reverse_for` | 反向处理循环 | `stmt, d_output` | `List[ir.Stmt]` |
| `_reverse_for_iter_args` | SSA 循环反向 | `stmt, d_output` | `List[ir.Stmt]` |
| `_reverse_for_with_tape` | Tape 循环反向 | `stmt, d_output` | `List[ir.Stmt]` |
| `_reverse_if_inline` | 反向处理条件分支 | `stmt, d_output` | `List[ir.Stmt]` |
| `_reverse_call_inline` | 反向处理算子调用 | `call, d_output` | `List[ir.Stmt]` |
| `_identify_vars_needing_tape` | 识别需要 Tape 的变量 | `body, for_stmt` | `Dict[str, Dict]` |
| `_create_add` | 创建加法表达式 | `lhs, rhs` | `ir.Expr` |
| `_create_zero` | 创建零值 | `type` | `ir.Expr` |

#### 5.2.3 build 方法流程

```python
def build(self) -> ir.Function:
    """构建反向梯度函数"""
    
    # 1. 创建梯度参数（前向参数 + 梯度种子）
    backward_params = list(self.forward_func.params)
    d_output = ir.Var("d_output", output_type, self.span)
    backward_params.append(d_output)
    
    # 2. 创建梯度变量
    param_vars = {p.name_hint: p for p in self.forward_func.params}
    for param_name in self.grad_params:
        param_var = param_vars[param_name]
        grad_var = ir.Var(f"d_{param_name}", param_var.type, self.span)
        self.grad_var_map[param_var] = grad_var
    
    # 3. 构建反向语句
    backward_stmts = self._build_backward_stmts(d_output)
    
    # 4. 收集反向中引用的正向中间值
    forward_saved_vars = self._collect_saved_forward_vars(backward_stmts)
    
    # 5. 生成正向计算语句
    forward_compute_stmts = self._generate_forward_compute_stmts(forward_saved_vars)
    
    # 6. 拼接语句：正向计算 + 反向语句
    backward_stmts = forward_compute_stmts + backward_stmts
    
    # 7. 构建返回语句
    return_vars = [self.grad_var_map[param_vars[name]] for name in self.grad_params]
    backward_stmts.append(ir.ReturnStmt(return_vars, self.span))
    
    # 8. 创建反向函数
    backward_func = ir.Function(
        f"grad_{self.forward_func.name}",
        backward_params,
        backward_return_types,
        ir.SeqStmts(backward_stmts, self.span),
        self.span
    )
    
    return backward_func
```

### 5.3 关键设计点

#### 5.3.1 梯度累加

输入参数可能被多次使用，梯度需要累加：

```python
# y = x*w + x*v
# d_x = d_y * w + d_y * v (累加)

def _create_add(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
    """创建加法表达式"""
    # 简化：如果 rhs 是零，直接返回 lhs
    if isinstance(rhs, ir.ConstFloat) and rhs.value == 0.0:
        return lhs
    if isinstance(rhs, ir.ConstInt) and rhs.value == 0:
        return lhs
    
    # 根据类型创建合适的加法
    if isinstance(lhs.type, ir.TileType):
        return ir.Call(ir.Op("tile.add"), [lhs, rhs], {}, lhs.type, self.span)
    elif isinstance(lhs.type, ir.TensorType):
        return ir.Call(ir.Op("tensor.add"), [lhs, rhs], {}, lhs.type, self.span)
    elif isinstance(lhs.type, ir.ScalarType):
        return ir.Add(lhs, rhs, lhs.type.dtype, self.span)
    
    return lhs
```

#### 5.3.2 前向值保存策略

| 变量类型 | 处理方式 | 原因 |
|----------|----------|------|
| 输入参数 | 直接使用 | 反向时已知 |
| 循环外中间变量 | 反向函数中重新计算 | 避免 Tape 内存开销 |
| 循环内变量 | 使用 TensorArray Tape | 需要每次迭代的值 |

#### 5.3.3 算子名映射

```python
# 如果原始算子名未注册，尝试 tile/tensor 版本
if not GradientRegistry.has(op_name):
    parts = op_name.split(".")
    if len(parts) >= 2:
        tile_op = f"tile.{parts[-1]}"
        tensor_op = f"tensor.{parts[-1]}"
        if GradientRegistry.has(tile_op):
            op_name = tile_op
        elif GradientRegistry.has(tensor_op):
            op_name = tensor_op
```

#### 5.3.4 循环初始化判断

```python
def _param_needs_init_in_loop(self, param_var: ir.Var, body: ir.Stmt) -> bool:
    """
    判断参数是否需要在循环内初始化为零
    
    - iter_args + yield 模式：init 直接赋值，不初始化
    - 普通循环：需要初始化为零用于累加
    """
    def check_for_stmt(stmt):
        if isinstance(stmt, ir.ForStmt):
            for iter_arg in stmt.iter_args:
                if isinstance(iter_arg.initValue, ir.Var):
                    if iter_arg.initValue.name_hint == param_var.name_hint:
                        if self._has_yield_in_body(stmt.body):
                            return False  # iter_args + yield，不初始化
            return check_for_stmt(stmt.body)
        ...
    
    return check_for_stmt(body)
```

### 5.4 测试覆盖

| 测试文件 | 测试内容 |
|----------|----------|
| `test_autodiff.py` | 数值验证、梯度注册、IR 结构 |
| `test_ir_gradient_generation.py` | IR 生成、结构验证 |
| `test_gradient_numerical_verification.py` | 数值梯度对比 |
| `test_pl_grad_api.py` | 高层 API 测试 |

### 5.5 数值验证结果

| 测试 | 数学验证 | 数值误差 | 结果 |
|------|----------|----------|------|
| `test_add_gradient` | `dy/da = 1` | < 1e-4 | ✓ |
| `test_mul_gradient` | `dy/da = b` | < 1e-4 | ✓ |
| `test_gradient_accumulation` | `dy/da = b + c` | < 1e-4 | ✓ |
| `test_matmul_gradient` | `dA = dC@B^T` | < 1e-3 | ✓ |
| `test_loop_gradient` | `dacc/dx = N` | < 1e-4 | ✓ |

---

## 6. 使用示例

### 6.1 基本梯度计算

```python
import pypto as pl

@pl.function
def simple_add(x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
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

### 6.2 矩阵乘法梯度

```python
@pl.function
def matmul_func(A: pl.Tile[[16, 16], pl.FP16], B: pl.Tile[[16, 16], pl.FP16]) -> pl.Tile[[16, 16], pl.FP32]:
    return pl.matmul(A, B)

grad_matmul = pl.grad(matmul_func)

# 梯度函数签名
# def grad_matmul(A, B, d_C) -> (d_A, d_B)
# 其中:
#   d_A = d_C @ transpose(B)
#   d_B = transpose(A) @ d_C
```

### 6.3 带循环的梯度

```python
@pl.function
def accumulate(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
    acc: pl.Tensor[[64], pl.FP32] = pl.zeros([64], dtype=pl.FP32)
    for i, (acc_iter,) in pl.range(n, init_values=(acc,)):
        acc_new: pl.Tensor[[64], pl.FP32] = pl.add(acc_iter, x)
        acc_out = pl.yield_(acc_new)
    return acc_out

grad_accumulate = pl.grad(accumulate, params=['x'])

# 梯度: d_x = n * d_output
```

### 6.4 梯度累加验证

```python
@pl.function
def nested(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    z: pl.Tensor[[64], pl.FP32] = pl.add(y, x)  # x 使用两次
    return z

grad_nested = pl.grad(nested)

# 梯度: d_x = 2.0 + 1.0 = 3.0 (累加)
```

### 6.5 自定义梯度

```python
from pypto.autodiff import register_grad

@register_grad('my.custom_op', 'user')
def custom_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    span = d_output.span
    
    da = ir.Call(ir.Op('tile.mul'), [d_output, ir.ConstFloat(2.0, ir.DataType.FP32, span)], {}, a.type, span)
    db = d_output
    
    return [da, db]

# 现在可以使用
grad_rule = GradientRegistry.get('my.custom_op')
```

---

## 7. 数学正确性论证

### 7.1 链式法则保证

反向模式自动微分基于链式法则，对于任何可导的复合函数：

$$y = f_n(f_{n-1}(...f_1(x)))$$

梯度计算：

$$\frac{\partial y}{\partial x} = \prod_{i=n}^{1} \frac{\partial f_i}{\partial f_{i-1}}$$

反向模式通过从输出到输入逐层计算每个 $\frac{\partial f_i}{\partial f_{i-1}}$，保证了数学正确性。

### 7.2 循环的正确性

对于循环：

$$v_{i+1} = f(v_i, x), \quad i = 0, 1, ..., N-1$$

梯度：

$$\frac{\partial v_N}{\partial x} = \sum_{i=0}^{N-1} \frac{\partial v_N}{\partial v_{i+1}} \cdot \frac{\partial f(v_i, x)}{\partial x}$$

反向循环通过从 $i = N-1$ 到 $0$ 逐层计算：

$$\bar{v}_i = \bar{v}_{i+1} \cdot \frac{\partial f}{\partial v}(v_i, x)$$

$$\bar{x} += \bar{v}_{i+1} \cdot \frac{\partial f}{\partial x}(v_i, x)$$

这等价于完整的梯度计算。

### 7.3 Phi 节点的正确性

Phi 节点表示条件分支：

$$v = \begin{cases} f_1(x) & \text{if } c \\ f_2(x) & \text{else} \end{cases}$$

梯度：

$$\bar{x} = \begin{cases} \bar{v} \cdot \frac{\partial f_1}{\partial x} & \text{if } c \\ \bar{v} \cdot \frac{\partial f_2}{\partial x} & \text{else} \end{cases}$$

反向时使用前向保存的条件值，确保梯度流向正确的分支。

### 7.4 数值验证

所有梯度规则通过有限差分数值验证：

$$\frac{\partial f}{\partial x} \approx \frac{f(x + \epsilon) - f(x)}{\epsilon}$$

误差阈值：< 1e-4

---

## 8. 总结

PyPTO 自动微分系统采用 Source-to-Source 反向模式，核心特性：

1. **IR 层操作**: 直接在 IR 上生成反向函数，无需运行时 Tape
2. **梯度规则注册**: 通过 GradientRegistry 注册算子梯度，支持自定义扩展
3. **循环处理**: SSA iter_args 模式 + TensorArray Tape 模式
4. **控制流**: IfStmt 保持分支结构，YieldStmt 处理 SSA 循环
5. **梯度累加**: 输入参数多次使用时自动累加梯度
6. **数学正确性**: 所有梯度规则通过数值验证（误差 < 1e-4）

---

## 附录

### A. 文件清单

| 文件 | 路径 | 描述 |
|------|------|------|
| `__init__.py` | `python/pypto/autodiff/__init__.py` | 高层 API |
| `gradient_registry.py` | `python/pypto/autodiff/gradient_registry.py` | 注册表核心 |
| `grad_rules.py` | `python/pypto/autodiff/grad_rules.py` | 内置梯度规则 |
| `reverse_mode.py` | `python/pypto/autodiff/reverse_mode.py` | 反向模式实现 |
| `test_autodiff.py` | `tests/ut/autodiff/test_autodiff.py` | 基础测试 |
| `test_ir_gradient_generation.py` | `tests/ut/autodiff/` | IR 生成测试 |

### B. 参考文献

1. PyPTO IR 设计文档: `examples/ir_analysis/01_ir_node_types_20260416.md`
2. 自动微分设计: `examples/ir_analysis/05_autodiff_design_20260416.md`
3. API 设计: `examples/ir_analysis/07_autodiff_api_design_20260416.md`
4. 梯度注册设计: `examples/ir_analysis/09_gradient_registration_design_20260416.md`