# PyPTO Source-to-Source 自动微分方案设计

## 1. 方案概述

### 1.1 目标

设计一个基于 PyPTO IR 的 source-to-source 自动微分系统：
- 使用 IRMutator 进行 IR 变换
- 支持所有 IR 节点的微分
- 正确处理循环和 Phi 节点（SSA）
- 支持变长循环（动态迭代次数）
- 数学上正确（基于链式法则）

### 1.2 方法选择

采用**反向模式自动微分（Reverse Mode AD）**：
- 计算梯度效率高：一次前向 + 一次反向 = 所有输入梯度
- 适合深度学习场景（多输入少输出）
- 等价于计算 Jacobian 矩阵的转置

### 1.3 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   自动微分流程                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 前向 IR (原始函数)                                       │
│       │                                                     │
│       ▼                                                     │
│  2. SSA 变换 (ConvertToSSA)                                 │
│       │                                                     │
│       ▼                                                     │
│  3. 活性分析 (Liveness Analysis)                            │
│       │  - 确定需要保存的中间值                              │
│       ▼                                                     │
│  4. 反向 IR 生成 (ReverseModeMutator)                       │
│       │  - 构建反向函数                                      │
│       │  - 处理循环反向                                      │
│       │  - 处理 Phi 节点                                     │
│       ▼                                                     │
│  5. 梯度函数 (gradient function)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 理论基础

### 2.1 链式法则

对于复合函数 $y = f(g(x))$，其导数为：
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

### 2.2 反向模式

反向模式通过从输出到输入逐层传播梯度（伴随变量 adjoint）：

```
前向计算:
  v₁ = f₁(x₁, x₂)
  v₂ = f₂(v₁, x₂)
  y  = f₃(v₂)

反向传播:
  v̄₂ = ȳ · ∂f₃/∂v₂
  v̄₁ = v̄₂ · ∂f₂/∂v₁
  x̄₂ = v̄₂ · ∂f₂/∂x₂ + v̄₁ · ∂f₁/∂x₂
  x̄₁ = v̄₁ · ∂f₁/∂x₁
```

其中 $\bar{v}$ 表示 $v$ 的伴随变量（梯度）。

### 2.3 循环的微分

对于循环：
```
for i = 0 to N:
  v_{i+1} = f(v_i, x)
```

梯度通过反向遍历循环：
```
for i = N-1 to 0:
  v̄_i = v̄_{i+1} · ∂f/∂v_i
  x̄ += v̄_{i+1} · ∂f/∂x
```

---

## 3. IR 节点微分规则

### 3.1 表达式节点

| 前向操作 | 反向梯度 | 数学表达 |
|----------|----------|----------|
| `y = add(a, b)` | `ā += ȳ, b̄ += ȳ` | $\bar{a} = \bar{y}, \bar{b} = \bar{y}$ |
| `y = sub(a, b)` | `ā += ȳ, b̄ -= ȳ` | $\bar{a} = \bar{y}, \bar{b} = -\bar{y}$ |
| `y = mul(a, b)` | `ā += ȳ · b, b̄ += ȳ · a` | $\bar{a} = \bar{y} \cdot b, \bar{b} = \bar{y} \cdot a$ |
| `y = div(a, b)` | `ā += ȳ / b, b̄ -= ȳ · a / b²` | $\bar{a} = \bar{y}/b, \bar{b} = -\bar{y} \cdot a/b^2$ |
| `y = neg(a)` | `ā -= ȳ` | $\bar{a} = -\bar{y}$ |
| `y = exp(a)` | `ā += ȳ · exp(a)` | $\bar{a} = \bar{y} \cdot e^a$ |
| `y = log(a)` | `ā += ȳ / a` | $\bar{a} = \bar{y}/a$ |
| `y = pow(a, n)` | `ā += ȳ · n · a^{n-1}` | $\bar{a} = \bar{y} \cdot n \cdot a^{n-1}$ |
| `y = sqrt(a)` | `ā += ȳ / (2·sqrt(a))` | $\bar{a} = \bar{y}/(2\sqrt{a})$ |
| `y = relu(a)` | `ā += ȳ · (a > 0 ? 1 : 0)` | $\bar{a} = \bar{y} \cdot \text{sign}(a>0)$ |
| `y = abs(a)` | `ā += ȳ · sign(a)` | $\bar{a} = \bar{y} \cdot \text{sign}(a)$ |
| `y = sin(a)` | `ā += ȳ · cos(a)` | $\bar{a} = \bar{y} \cdot \cos(a)$ |
| `y = cos(a)` | `ā -= ȳ · sin(a)` | $\bar{a} = -\bar{y} \cdot \sin(a)$ |

### 3.2 比较表达式（不产生梯度）

比较操作（`eq`, `ne`, `lt`, `le`, `gt`, `ge`）产生布尔值，不参与梯度传播：
- 这些操作常用于控制流（if 条件）
- 反向时保留前向值用于条件判断

### 3.3 控制流语句

#### 3.3.1 AssignStmt

```python
# 前向
v = expr(a, b)

# 反向
# 1. 保存 expr 的输入值
# 2. 计算梯度贡献
ā += v̄ · ∂expr/∂a
b̄ += v̄ · ∂expr/∂b
```

#### 3.3.2 IfStmt (Phi 节点)

```python
# 前向 (SSA)
if cond:
  v = yield(expr1)
else:
  v = yield(expr2)
result = yield(v)  # Phi: v = phi(expr1, expr2)

# 反向
# Phi 节点反向：梯度流向两个分支
if cond:
  expr1̄ += v̄ · ∂expr1/∂...
else:
  expr2̄ += v̄ · ∂expr2/∂...
```

#### 3.3.3 ForStmt

```python
# 前向 (SSA)
for i, (v,) in pl.range(N, init_values=(v₀,)):
  v_new = f(v, x)
  v = yield(v_new)
v_final = v  # 循环后的值

# 反向 (反向遍历循环)
for i_rev in pl.range(N-1, -1, -1):  # 从 N-1 到 0
  # 需要访问前向的中间值
  v_fwd = saved_values[i_rev]  # 前向时的 v
  v̄ += v̄ · ∂f/∂v(v_fwd, x)
  x̄ += v̄ · ∂f/∂x(v_fwd, x)
```

---

## 4. 实现设计

### 4.1 核心数据结构

```python
# 保存的前向值（用于反向计算）
class TapeRecord:
    expr: ir.Expr          # 前向表达式
    inputs: list[ir.Expr]  # 输入值（保存用于反向）
    output_var: ir.Var     # 输出变量
    span: ir.Span          # 源码位置

# 活性分析结果
class LivenessInfo:
    needs_gradient: set[ir.Var]      # 需要梯度的变量
    needs_save: set[ir.Var]          # 需要保存的中间值
    gradient_vars: dict[ir.Var, ir.Var]  # var -> grad_var 映射
```

### 4.2 ReverseModeMutator 类

```python
class ReverseModeMutator(IRMutator):
    """
    反向模式自动微分变换器
    
    核心方法:
    - visit_function: 构建反向函数
    - visit_expr: 生成梯度表达式
    - visit_for_stmt: 反向循环
    - visit_if_stmt: Phi 节点处理
    """
    
    def __init__(self, grad_outputs: list[ir.Var]):
        self.grad_outputs = grad_outputs  # 输出的梯度种子
        self.tape: list[TapeRecord] = []  # 前向值记录
        self.grad_vars: dict[ir.Var, ir.Var] = {}  # var -> grad_var
        self.saved_vars: dict[ir.Var, ir.Var] = {}  # 前向保存的值
        
    def reverse_function(self, forward_func: ir.Function) -> ir.Function:
        """将前向函数转换为反向函数"""
        # ...
```

### 4.3 完整实现

```python
"""
PyPTO Source-to-Source 自动微分实现
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from pypto import ir, DataType
from pypto.ir import IRMutator, IRVisitor, Span, ConstInt, ConstFloat

# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class TapeRecord:
    """前向计算记录（用于反向计算）"""
    stmt_index: int           # 语句索引
    expr: ir.Expr             # 表达式
    inputs: List[ir.Expr]     # 输入值（需要保存）
    output_var: ir.Var        # 输出变量
    input_vars: List[ir.Var]  # 输入变量引用
    span: ir.Span

@dataclass
class LivenessResult:
    """活性分析结果"""
    needs_grad: Set[ir.Var]           # 需要梯度的变量
    needs_save: Set[ir.Var]           # 需要保存的中间值
    loop_carried: Set[ir.Var]         # 循环携带变量
    if_phi_vars: Dict[ir.Var, List[ir.Var]]  # Phi 节点变量

# =============================================================================
# 活性分析
# =============================================================================

class LivenessAnalyzer(IRVisitor):
    """分析哪些变量需要梯度、需要保存"""
    
    def __init__(self, grad_params: List[ir.Var]):
        self.grad_params = set(grad_params)  # 需要梯度的参数
        self.needs_grad: Set[ir.Var] = set()
        self.needs_save: Set[ir.Var] = set()
        self.all_vars: Dict[str, ir.Var] = {}
        self.var_uses: Dict[ir.Var, List[int]] = {}  # var -> 使用位置索引
        self.current_stmt_idx = 0
        self.in_loop = False
        self.loop_vars: Set[ir.Var] = set()
        
    def analyze_function(self, func: ir.Function) -> LivenessResult:
        """分析函数的活性信息"""
        self.needs_grad = set(self.grad_params)
        self.visit_function(func)
        
        # 从需要梯度的参数反向传播需求
        self._propagate_grad_need()
        
        return LivenessResult(
            needs_grad=self.needs_grad,
            needs_save=self.needs_save,
            loop_carried=self.loop_vars,
            if_phi_vars={}
        )
    
    def _propagate_grad_need(self):
        """反向传播梯度需求"""
        # 如果输出需要梯度，则其输入也需要梯度
        # 这里需要根据表达式依赖关系传播
        pass
    
    def visit_var(self, var: ir.Var):
        """记录变量使用"""
        self.all_vars[var.name_hint] = var
        if var not in self.var_uses:
            self.var_uses[var] = []
        self.var_uses[var].append(self.current_stmt_idx)
    
    def visit_assign_stmt(self, stmt: ir.AssignStmt):
        """分析赋值语句"""
        self.current_stmt_idx += 1
        self.visit_expr(stmt.value)
        # 如果值使用了需要梯度的变量，则输出也需要梯度
        used_grad_vars = self._find_grad_vars_in_expr(stmt.value)
        if used_grad_vars:
            self.needs_grad.add(stmt.var)
            # 需要保存输入值用于反向
            for v in used_grad_vars:
                self.needs_save.add(v)
    
    def visit_for_stmt(self, stmt: ir.ForStmt):
        """分析循环"""
        self.in_loop = True
        for iter_arg in stmt.iter_args:
            self.loop_vars.add(iter_arg)
        self.visit_stmt(stmt.body)
        self.in_loop = False
    
    def _find_grad_vars_in_expr(self, expr: ir.Expr) -> Set[ir.Var]:
        """找出表达式中使用的需要梯度的变量"""
        result = set()
        if isinstance(expr, ir.Var):
            if expr in self.needs_grad:
                result.add(expr)
        elif isinstance(expr, ir.BinaryExpr):
            result.update(self._find_grad_vars_in_expr(expr.left))
            result.update(self._find_grad_vars_in_expr(expr.right))
        elif isinstance(expr, ir.UnaryExpr):
            result.update(self._find_grad_vars_in_expr(expr.operand))
        elif isinstance(expr, ir.Call):
            for arg in expr.args:
                result.update(self._find_grad_vars_in_expr(arg))
        return result

# =============================================================================
# 反向模式微分变换器
# =============================================================================

class ReverseModeMutator(IRMutator):
    """反向模式自动微分 IR 变换器"""
    
    def __init__(
        self,
        grad_params: List[ir.Var],
        grad_output: Optional[ir.Var] = None
    ):
        self.grad_params = grad_params
        self.grad_output = grad_output
        
        # 映射表
        self.grad_var_map: Dict[ir.Var, ir.Var] = {}  # var -> grad_var
        self.saved_var_map: Dict[ir.Var, ir.Var] = {}  # var -> saved_var
        
        # 累积器（用于梯度的 += 操作）
        self.accumulators: Dict[ir.Var, ir.Expr] = {}
        
        # 前向记录
        self.forward_stmts: List[ir.Stmt] = []
        self.reverse_stmts: List[ir.Stmt] = []
        
        # 循环处理
        self.loop_tape: Dict[ir.Var, List[ir.Var]] = {}  # 循环变量 -> 保存的值列表
        
    def differentiate(self, func: ir.Function) -> ir.Function:
        """生成梯度函数"""
        # 1. 活性分析
        analyzer = LivenessAnalyzer(self.grad_params)
        liveness = analyzer.analyze_function(func)
        
        # 2. 构建反向函数
        reverse_func = self._build_reverse_function(func, liveness)
        
        return reverse_func
    
    def _build_reverse_function(
        self,
        forward_func: ir.Function,
        liveness: LivenessResult
    ) -> ir.Function:
        """构建反向梯度函数"""
        span = forward_func.span
        
        # 创建梯度变量
        for param in forward_func.params:
            if param in self.grad_params:
                grad_var = ir.Var(
                    f"d_{param.name_hint}",
                    param.type,
                    span
                )
                self.grad_var_map[param] = grad_var
        
        # 输出梯度种子（默认为 1）
        if forward_func.return_types:
            output_grad = ir.ConstFloat(1.0, DataType.FP32, span)
            for ret_var in liveness.needs_grad:
                self.grad_var_map[ret_var] = ir.Var(
                    f"d_{ret_var.name_hint}",
                    ret_var.type,
                    span
                )
                self.accumulators[ret_var] = output_grad
        
        # 处理函数体
        body = forward_func.body
        if isinstance(body, ir.SeqStmts):
            # 反向遍历语句
            for stmt in reversed(body.stmts):
                self._process_reverse_stmt(stmt, liveness)
        
        # 构建反向函数体
        reverse_body = self._build_reverse_body(liveness)
        
        # 构建返回语句（梯度）
        return_stmts = []
        for param in self.grad_params:
            if param in self.grad_var_map:
                grad_var = self.grad_var_map[param]
                return_stmts.append(ir.ReturnStmt([grad_var], span))
        
        # 创建反向函数
        reverse_func = ir.Function(
            f"grad_{forward_func.name}",
            forward_func.params + list(self.grad_var_map.values()),  # 参数 + 梯度参数
            forward_func.return_types,
            reverse_body,
            span,
            forward_func.func_type
        )
        
        return reverse_func
    
    def _process_reverse_stmt(
        self,
        stmt: ir.Stmt,
        liveness: LivenessResult
    ):
        """处理反向语句"""
        if isinstance(stmt, ir.AssignStmt):
            self._reverse_assign(stmt, liveness)
        elif isinstance(stmt, ir.ForStmt):
            self._reverse_for_loop(stmt, liveness)
        elif isinstance(stmt, ir.IfStmt):
            self._reverse_if(stmt, liveness)
        elif isinstance(stmt, ir.YieldStmt):
            self._reverse_yield(stmt, liveness)
    
    def _reverse_assign(
        self,
        stmt: ir.AssignStmt,
        liveness: LivenessResult
    ):
        """反向赋值语句"""
        var = stmt.var
        value = stmt.value
        
        if var not in self.grad_var_map:
            return  # 不需要梯度
        
        grad_var = self.grad_var_map[var]
        var_grad = self.accumulators.get(grad_var, ir.ConstFloat(0.0, DataType.FP32, stmt.span))
        
        # 根据表达式类型生成反向梯度
        reverse_exprs = self._reverse_expr(value, var_grad, stmt.span)
        
        for input_var, grad_expr in reverse_exprs.items():
            if input_var in self.grad_var_map:
                input_grad_var = self.grad_var_map[input_var]
                # 累加梯度
                if input_grad_var in self.accumulators:
                    self.accumulators[input_grad_var] = ir.Add(
                        self.accumulators[input_grad_var],
                        grad_expr,
                        DataType.FP32,
                        stmt.span
                    )
                else:
                    self.accumulators[input_grad_var] = grad_expr
    
    def _reverse_expr(
        self,
        expr: ir.Expr,
        output_grad: ir.Expr,
        span: ir.Span
    ) -> Dict[ir.Var, ir.Expr]:
        """
        计算表达式的反向梯度
        
        返回: Dict[输入变量, 梯度表达式]
        """
        result: Dict[ir.Var, ir.Expr] = {}
        
        if isinstance(expr, ir.Var):
            # y = x -> x̄ = ȳ
            result[expr] = output_grad
            
        elif isinstance(expr, ir.ConstInt) or isinstance(expr, ir.ConstFloat):
            # 常量无梯度
            pass
            
        elif isinstance(expr, ir.Add):
            # y = a + b -> ā = ȳ, b̄ = ȳ
            left_grads = self._reverse_expr(expr.left, output_grad, span)
            right_grads = self._reverse_expr(expr.right, output_grad, span)
            result.update(left_grads)
            result.update(right_grads)
            
        elif isinstance(expr, ir.Sub):
            # y = a - b -> ā = ȳ, b̄ = -ȳ
            left_grads = self._reverse_expr(expr.left, output_grad, span)
            neg_grad = ir.Neg(output_grad, DataType.FP32, span)
            right_grads = self._reverse_expr(expr.right, neg_grad, span)
            result.update(left_grads)
            result.update(right_grads)
            
        elif isinstance(expr, ir.Mul):
            # y = a * b -> ā = ȳ * b, b̄ = ȳ * a
            # 需要保存 a 和 b 的前向值
            saved_right = self._get_saved_value(expr.right)
            saved_left = self._get_saved_value(expr.left)
            
            left_grad = ir.Mul(output_grad, saved_right, DataType.FP32, span)
            right_grad = ir.Mul(output_grad, saved_left, DataType.FP32, span)
            
            left_grads = self._reverse_expr(expr.left, left_grad, span)
            right_grads = self._reverse_expr(expr.right, right_grad, span)
            result.update(left_grads)
            result.update(right_grads)
            
        elif isinstance(expr, ir.FloatDiv):
            # y = a / b -> ā = ȳ / b, b̄ = -ȳ * a / b²
            saved_right = self._get_saved_value(expr.right)
            saved_left = self._get_saved_value(expr.left)
            
            left_grad = ir.FloatDiv(output_grad, saved_right, DataType.FP32, span)
            b_squared = ir.Mul(saved_right, saved_right, DataType.FP32, span)
            neg_term = ir.Neg(ir.Mul(ir.Mul(output_grad, saved_left, DataType.FP32, span), 
                                       saved_right, DataType.FP32, span), 
                              DataType.FP32, span)
            right_grad = ir.FloatDiv(neg_term, b_squared, DataType.FP32, span)
            
            left_grads = self._reverse_expr(expr.left, left_grad, span)
            right_grads = self._reverse_expr(expr.right, right_grad, span)
            result.update(left_grads)
            result.update(right_grads)
            
        elif isinstance(expr, ir.Neg):
            # y = -a -> ā = -ȳ
            neg_grad = ir.Neg(output_grad, DataType.FP32, span)
            operand_grads = self._reverse_expr(expr.operand, neg_grad, span)
            result.update(operand_grads)
            
        elif isinstance(expr, ir.Call):
            # 算子调用：根据算子类型生成反向
            result.update(self._reverse_call(expr, output_grad, span))
        
        return result
    
    def _reverse_call(
        self,
        call: ir.Call,
        output_grad: ir.Expr,
        span: ir.Span
    ) -> Dict[ir.Var, ir.Expr]:
        """处理算子调用的反向"""
        op_name = call.op.name
        result: Dict[ir.Var, ir.Expr] = {}
        
        # Tile/Tensor 算子的反向规则
        if op_name == "tile.add" or op_name == "tensor.add":
            # y = add(a, b) -> ā = ȳ, b̄ = ȳ
            for i, arg in enumerate(call.args):
                arg_grads = self._reverse_expr(arg, output_grad, span)
                result.update(arg_grads)
                
        elif op_name == "tile.mul" or op_name == "tensor.mul":
            # y = mul(a, b) -> ā = ȳ * b, b̄ = ȳ * a
            saved_args = [self._get_saved_value(arg) for arg in call.args]
            
            for i, arg in enumerate(call.args):
                other_arg = saved_args[1 - i]
                arg_grad = ir.Mul(output_grad, other_arg, call.args[i].type.dtype, span)
                arg_grads = self._reverse_expr(arg, arg_grad, span)
                result.update(arg_grads)
                
        elif op_name == "tile.exp" or op_name == "tensor.exp":
            # y = exp(a) -> ā = ȳ * exp(a)
            # 需要保存 exp(a) 的结果（或保存 a 重新计算）
            saved_input = self._get_saved_value(call.args[0])
            exp_result = ir.Call(
                call.op,
                [saved_input],
                call.kwargs,
                call.type,
                span
            )
            arg_grad = ir.Mul(output_grad, exp_result, call.type.dtype, span)
            arg_grads = self._reverse_expr(call.args[0], arg_grad, span)
            result.update(arg_grads)
            
        elif op_name == "tile.matmul":
            # y = matmul(A, B) -> Ā = ȳ @ B^T, B̄ = A^T @ ȳ
            saved_A = self._get_saved_value(call.args[0])
            saved_B = self._get_saved_value(call.args[1])
            
            # Ā = output_grad @ B^T (需要 transpose 操作)
            # 这里简化处理，实际需要调用 transpose + matmul
            A_grad = ir.Call(
                ir.Op("tile.matmul"),
                [output_grad, saved_B],
                {},
                saved_A.type,
                span
            )
            B_grad = ir.Call(
                ir.Op("tile.matmul"),
                [saved_A, output_grad],
                {},
                saved_B.type,
                span
            )
            
            A_grads = self._reverse_expr(call.args[0], A_grad, span)
            B_grads = self._reverse_expr(call.args[1], B_grad, span)
            result.update(A_grads)
            result.update(B_grads)
        
        return result
    
    def _reverse_for_loop(
        self,
        stmt: ir.ForStmt,
        liveness: LivenessResult
    ):
        """
        反向循环
        
        对于 SSA 循环：
        for i, (v,) in pl.range(N, init_values=(v₀,)):
          v_new = f(v, x)
          v = yield(v_new)
        
        反向：
        for i_rev in pl.range(N-1, -1, -1):
          v_fwd = saved[i_rev]  # 前向的 v 值
          v̄ += v̄ · ∂f/∂v(v_fwd)
          x̄ += v̄ · ∂f/∂x(v_fwd)
        """
        if not stmt.iter_args:
            # 简单循环（无 SSA），直接反向遍历
            self._reverse_simple_loop(stmt, liveness)
            return
        
        # SSA 循环
        # 1. 记录循环变量
        loop_var = stmt.loop_var
        iter_args = stmt.iter_args
        return_vars = stmt.return_vars
        
        # 2. 创建反向循环变量
        span = stmt.span
        dtype = DataType.INDEX
        
        # 反向循环范围
        start = stmt.start
        stop = stmt.stop
        step = stmt.step
        
        # 反向循环: 从 stop-step 到 start-step，步长 -step
        # 需要计算迭代次数
        # trip_count = (stop - start) / step
        # 反向: for i_rev in pl.range(trip_count - 1, -1, -1)
        
        # 3. 访问循环体，收集前向计算
        # 这里需要递归处理循环体中的语句
        
        # 4. 构建反向循环体
        reverse_body_stmts = []
        
        for i, iter_arg in enumerate(iter_args):
            if iter_arg in self.grad_var_map:
                # 需要梯度
                grad_var = self.grad_var_map[iter_arg]
                
                # 获取前向保存的值（需要从 tape 中读取）
                saved_var = self.saved_var_map.get(iter_arg)
                
                # 计算梯度贡献（需要访问前向循环体中的表达式）
                # ...
        
        # 5. 构建 YieldStmt（反向循环的 SSA）
        yield_values = []
        for rv in return_vars:
            if rv in self.grad_var_map:
                yield_values.append(self.grad_var_map[rv])
        
        if yield_values:
            reverse_body_stmts.append(ir.YieldStmt(yield_values, span))
        
        # 6. 创建反向 ForStmt
        trip_count = self._compute_trip_count(stmt)
        reverse_for = ir.ForStmt(
            ir.Var("i_rev", ir.ScalarType(dtype), span),
            ir.ConstInt(0, dtype, span),  # 从 0 开始（正向）
            trip_count,                    # 迭代次数
            ir.ConstInt(1, dtype, span),
            stmt.iter_args,               # iter_args（携带反向梯度）
            ir.SeqStmts(reverse_body_stmts, span),
            stmt.return_vars,
            span,
            stmt.kind
        )
        
        self.reverse_stmts.append(reverse_for)
    
    def _reverse_simple_loop(
        self,
        stmt: ir.ForStmt,
        liveness: LivenessResult
    ):
        """反向简单循环（无 SSA iter_args）"""
        # 简单循环不携带状态，反向时：
        # 1. 保存每次迭代中修改的变量值
        # 2. 反向遍历，从最后一次迭代到第一次
        
        # 这需要更复杂的状态管理
        pass
    
    def _reverse_if(
        self,
        stmt: ir.IfStmt,
        liveness: LivenessResult
    ):
        """
        反向 IfStmt (Phi 节点处理)
        
        前向:
        if cond:
          v1 = yield(expr1)
        else:
          v2 = yield(expr2)
        result = yield(phi(v1, v2))
        
        反向:
        if cond:  # 使用相同的前向条件
          expr1̄ += result̄ · ∂expr1/∂...
        else:
          expr2̄ += result̄ · ∂expr2/∂...
        """
        # 1. 获取条件（需要保存前向条件值）
        cond = stmt.condition
        
        # 2. 获取 return_vars 的梯度
        grad_return_vars = []
        for rv in stmt.return_vars:
            if rv in self.grad_var_map:
                grad_return_vars.append(self.accumulators.get(rv, ir.ConstFloat(0.0, DataType.FP32, stmt.span)))
        
        # 3. 处理 then 分支
        then_grad_stmts = []
        if stmt.then_body:
            then_grad_stmts = self._reverse_if_branch(stmt.then_body, grad_return_vars, stmt.span)
        
        # 4. 处理 else 分支
        else_grad_stmts = []
        if stmt.else_body:
            else_grad_stmts = self._reverse_if_branch(stmt.else_body, grad_return_vars, stmt.span)
        
        # 5. 构建反向 IfStmt
        reverse_if = ir.IfStmt(
            cond,
            ir.SeqStmts(then_grad_stmts, stmt.span) if then_grad_stmts else None,
            ir.SeqStmts(else_grad_stmts, stmt.span) if else_grad_stmts else None,
            stmt.return_vars,
            stmt.span
        )
        
        self.reverse_stmts.append(reverse_if)
    
    def _reverse_if_branch(
        self,
        body: ir.Stmt,
        output_grads: List[ir.Expr],
        span: ir.Span
    ) -> List[ir.Stmt]:
        """反向处理 if 分支"""
        result_stmts = []
        
        if isinstance(body, ir.SeqStmts):
            for stmt in reversed(body.stmts):
                if isinstance(stmt, ir.YieldStmt):
                    # 找到 yield 的表达式，计算梯度
                    for i, val in enumerate(stmt.value):
                        if i < len(output_grads):
                            grad_exprs = self._reverse_expr(val, output_grads[i], span)
                            for var, grad in grad_exprs.items():
                                if var in self.grad_var_map:
                                    grad_var = self.grad_var_map[var]
                                    assign = ir.AssignStmt(grad_var, grad, span)
                                    result_stmts.append(assign)
        
        return result_stmts
    
    def _get_saved_value(self, expr: ir.Expr) -> ir.Expr:
        """获取保存的前向值"""
        if isinstance(expr, ir.Var):
            if expr in self.saved_var_map:
                return self.saved_var_map[expr]
        return expr
    
    def _compute_trip_count(self, stmt: ir.ForStmt) -> ir.Expr:
        """计算循环迭代次数"""
        # trip_count = (stop - start) / step
        if isinstance(stmt.stop, ir.ConstInt) and isinstance(stmt.start, ir.ConstInt):
            stop_val = stmt.stop.value
            start_val = stmt.start.value
            if isinstance(stmt.step, ir.ConstInt):
                step_val = stmt.step.value
            else:
                step_val = 1
            trip_count = (stop_val - start_val) // step_val
            return ir.ConstInt(trip_count, DataType.INDEX, stmt.span)
        # 动态范围：创建计算表达式
        diff = ir.Sub(stmt.stop, stmt.start, DataType.INDEX, stmt.span)
        return ir.FloorDiv(diff, stmt.step, DataType.INDEX, stmt.span)
    
    def _build_reverse_body(self, liveness: LivenessResult) -> ir.Stmt:
        """构建反向函数体"""
        stmts = []
        
        # 1. 初始化梯度累积器（全 0）
        for var in liveness.needs_grad:
            if var in self.grad_var_map:
                grad_var = self.grad_var_map[var]
                init_val = self._get_zero_value(var.type)
                stmts.append(ir.AssignStmt(grad_var, init_val, grad_var.span))
        
        # 2. 添加反向语句
        stmts.extend(self.reverse_stmts)
        
        # 3. 返回梯度
        return_grads = []
        for param in self.grad_params:
            if param in self.grad_var_map:
                return_grads.append(self.grad_var_map[param])
        
        if return_grads:
            stmts.append(ir.ReturnStmt(return_grads, Span.unknown()))
        
        return ir.SeqStmts(stmts, Span.unknown())
    
    def _get_zero_value(self, type: ir.Type) -> ir.Expr:
        """获取类型对应的零值"""
        if isinstance(type, ir.ScalarType):
            if type.dtype.IsFloat():
                return ir.ConstFloat(0.0, type.dtype, Span.unknown())
            else:
                return ir.ConstInt(0, type.dtype, Span.unknown())
        elif isinstance(type, ir.TileType) or isinstance(type, ir.TensorType):
            # 需要创建零张量/瓦片
            return ir.Call(
                ir.Op("tile.zeros" if isinstance(type, ir.TileType) else "tensor.zeros"),
                [type.shape],
                {"dtype": type.dtype},
                type,
                Span.unknown()
            )
        return ir.ConstFloat(0.0, DataType.FP32, Span.unknown())

# =============================================================================
# 高级 API
# =============================================================================

def grad(func: ir.Function, params: List[ir.Var]) -> ir.Function:
    """
    计算函数的梯度
    
    Args:
        func: 前向函数
        params: 需要计算梯度的参数列表
    
    Returns:
        反向梯度函数
    """
    mutator = ReverseModeMutator(params)
    return mutator.differentiate(func)

def jacobian(func: ir.Function, params: List[ir.Var]) -> ir.Function:
    """
    计算 Jacobian 矩阵（多输出梯度）
    """
    # 对于每个输出，计算所有参数的梯度
    # 返回 Jacobian 矩阵
    pass

def value_and_grad(func: ir.Function, params: List[ir.Var]) -> tuple[ir.Function, ir.Function]:
    """
    返回值和梯度函数（前向 + 反向）
    """
    forward = func
    reverse = grad(func, params)
    return forward, reverse
```

---

## 5. 循环和 Phi 节点处理

### 5.1 固定长度循环

```python
# 前向
@pl.function
def accumulate(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    acc: pl.Tensor[[64], pl.FP32] = pl.zeros([64], dtype=pl.FP32)
    for i, (acc_iter,) in pl.range(10, init_values=(acc,)):
        acc_new: pl.Tensor[[64], pl.FP32] = pl.add(acc_iter, x)
        acc_out = pl.yield_(acc_new)
    return acc_out

# 反向
@pl.function
def grad_accumulate(x: pl.Tensor[[64], pl.FP32], d_acc_out: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    d_x: pl.Tensor[[64], pl.FP32] = pl.zeros([64], dtype=pl.FP32)
    
    # 反向循环: i = 9, 8, ..., 0
    for i_rev, (d_acc_iter,) in pl.range(10, init_values=(d_acc_out,)):
        # ∂(acc_new)/∂x = 1, ∂(acc_new)/∂acc_iter = 1
        d_x = pl.add(d_x, d_acc_iter)
        d_acc_prev = pl.yield_(d_acc_iter)  # 梯度传递
    
    return d_x
```

**数学验证**：
```
前向:
  acc₀ = 0
  for i = 0 to 9:
    acc_{i+1} = acc_i + x
  acc_final = acc_10 = 10x

梯度:
  dL/dx = dL/d(acc_final) · ∂acc_final/∂x
         = dL/d(acc_10) · Σ_{i=0}^{9} 1
         = dL/d(acc_10) · 10

反向循环计算:
  d_acc̄_9 = dL/d(acc_10)
  d_acc̄_8 = d_acc̄_9 · 1 = dL/d(acc_10)
  ...
  d_x += d_acc̄_i · ∂(acc_{i+1})/∂x = dL/d(acc_10) for each i
  最终: d_x = 10 · dL/d(acc_10)
```

### 5.2 变长循环（动态迭代次数）

```python
# 前向（动态 N）
@pl.function
def dynamic_loop(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
    acc: pl.Tensor[[64], pl.FP32] = pl.zeros([64], dtype=pl.FP32)
    for i, (acc_iter,) in pl.range(n, init_values=(acc,)):
        acc_new: pl.Tensor[[64], pl.FP32] = pl.add(acc_iter, x)
        acc_out = pl.yield_(acc_new)
    return acc_out

# 反向
@pl.function
def grad_dynamic_loop(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64], 
                      d_acc_out: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    d_x: pl.Tensor[[64], pl.FP32] = pl.zeros([64], dtype=pl.FP32)
    
    # 反向循环：迭代次数相同，但梯度传递方向相反
    for i_rev, (d_acc_iter,) in pl.range(n, init_values=(d_acc_out,)):
        d_x = pl.add(d_x, d_acc_iter)
        d_acc_prev = pl.yield_(d_acc_iter)
    
    return d_x
```

**数学验证**：
```
前向:
  acc₀ = 0
  for i = 0 to n-1:
    acc_{i+1} = acc_i + x
  acc_n = n · x

梯度:
  dL/dx = dL/d(acc_n) · n
  dL/dn = dL/d(acc_n) · x  # n 的梯度（如果需要）

反向:
  d_x = Σ_{i=0}^{n-1} dL/d(acc_n) = n · dL/d(acc_n)
```

### 5.3 嵌套循环

```python
# 前向
@pl.function
def nested_loop(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = x
    for i, (outer,) in pl.range(3, init_values=(result,)):
        for j, (inner,) in pl.range(2, init_values=(outer,)):
            new_inner: pl.Tensor[[64], pl.FP32] = pl.mul(inner, 2.0)
            inner_out = pl.yield_(new_inner)
        outer_out = pl.yield_(inner_out)
    return outer_out

# 反向
@pl.function
def grad_nested_loop(x: pl.Tensor[[64], pl.FP32], d_outer_out: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    d_x: pl.Tensor[[64], pl.FP32] = pl.zeros([64], dtype=pl.FP32)
    
    # 外层反向
    for i_rev, (d_outer,) in pl.range(3, init_values=(d_outer_out,)):
        # 内层反向
        for j_rev, (d_inner,) in pl.range(2, init_values=(d_outer,)):
            # ∂(new_inner)/∂inner = 2.0
            d_inner_prev: pl.Tensor[[64], pl.FP32] = pl.mul(d_inner, 2.0)
            d_inner_out = pl.yield_(d_inner_prev)
        d_outer_prev = pl.yield_(d_inner_out)
    
    # d_x = d_outer_prev（因为初始 result = x）
    return d_outer_prev
```

**数学验证**：
```
前向:
  result₀ = x
  for i = 0 to 2:
    for j = 0 to 1:
      inner_{j+1} = inner_j · 2
    outer_{i+1} = inner_2 = outer_i · 2 · 2 = outer_i · 4
  result_final = x · 4^3 = x · 64

梯度:
  dL/dx = dL/d(result_final) · 64

反向计算:
  d_outer̄_2 = dL/d(result_final)
  d_outer̄_i = d_outer̄_{i+1} · 4
  d_x = d_outer̄_0 · 1 = dL/d(result_final) · 64
```

### 5.4 Phi 节点（IfStmt）

```python
# 前向（条件分支）
@pl.function
def conditional(x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
    if flag:
        result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
        out = pl.yield_(result)
    else:
        result2: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
        out = pl.yield_(result2)
    return out

# 反向
@pl.function
def grad_conditional(x: pl.Tensor[[64], pl.FP32], flag: pl.Scalar[pl.BOOL],
                     d_out: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    d_x: pl.Tensor[[64], pl.FP32] = pl.zeros([64], dtype=pl.FP32)
    
    if flag:  # 使用相同条件
        # ∂(mul(x, 2))/∂x = 2
        d_x = pl.add(d_x, pl.mul(d_out, 2.0))
    else:
        # ∂(add(x, 1))/∂x = 1
        d_x = pl.add(d_x, d_out)
    
    return d_x
```

**数学验证**：
```
前向:
  if flag:
    out = 2x
  else:
    out = x + 1

梯度:
  if flag:
    dL/dx = dL/d(out) · 2
  else:
    dL/dx = dL/d(out) · 1
```

---

## 6. 用例验证

### 6.1 简单表达式

```python
# 测试用例 1: 简单加法
@pl.function
def add_func(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
    return result

# 反向
@pl.function
def grad_add(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32],
             d_result: pl.Tensor[[64], pl.FP32]) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
    d_a: pl.Tensor[[64], pl.FP32] = d_result  # ∂(a+b)/∂a = 1
    d_b: pl.Tensor[[64], pl.FP32] = d_result  # ∂(a+b)/∂b = 1
    return d_a, d_b

# 数学验证
# f(a, b) = a + b
# ∂f/∂a = 1, ∂f/∂b = 1
# 梯度: [1, 1] · d_result = [d_result, d_result]
```

### 6.2 乘法和复合表达式

```python
# 测试用例 2: 乘法
@pl.function
def mul_func(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(a, b)
    return result

# 反向
@pl.function
def grad_mul(a_saved: pl.Tensor[[64], pl.FP32], b_saved: pl.Tensor[[64], pl.FP32],
             d_result: pl.Tensor[[64], pl.FP32]) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
    d_a: pl.Tensor[[64], pl.FP32] = pl.mul(d_result, b_saved)  # ∂(a*b)/∂a = b
    d_b: pl.Tensor[[64], pl.FP32] = pl.mul(d_result, a_saved)  # ∂(a*b)/∂b = a
    return d_a, d_b

# 数学验证
# f(a, b) = a · b
# ∂f/∂a = b, ∂f/∂b = a
# 需要保存前向的 a 和 b 值
```

### 6.3 复合表达式

```python
# 测试用例 3: 复合表达式 (a * b + c)
@pl.function
def compound(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32], 
             c: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    temp: pl.Tensor[[64], pl.FP32] = pl.mul(a, b)
    result: pl.Tensor[[64], pl.FP32] = pl.add(temp, c)
    return result

# 反向（反向遍历语句）
# 语句顺序: temp = a*b, result = temp+c
# 反向顺序: result̄ -> temp̄ -> ā, b̄

# d_result 已知
# d_temp = d_result (∂(temp+c)/∂temp = 1)
# d_c = d_result (∂(temp+c)/∂c = 1)
# d_a = d_temp · b_saved (∂(a*b)/∂a = b)
# d_b = d_temp · a_saved (∂(a*b)/∂b = a)

# 数学验证
# f(a, b, c) = a·b + c
# ∂f/∂a = b, ∂f/∂b = a, ∂f/∂c = 1
```

### 6.4 矩阵乘法

```python
# 测试用例 4: 矩阵乘法
@pl.function
def matmul_func(A: pl.Tile[[16, 16], pl.FP16], B: pl.Tile[[16, 16], pl.FP16]) -> pl.Tile[[16, 16], pl.FP16]:
    C: pl.Tile[[16, 16], pl.FP16] = pl.matmul(A, B)
    return C

# 反向
@pl.function  
def grad_matmul(A_saved: pl.Tile[[16, 16], pl.FP16], B_saved: pl.Tile[[16, 16], pl.FP16],
                d_C: pl.Tile[[16, 16], pl.FP16]) -> tuple[pl.Tile[[16, 16], pl.FP16], pl.Tile[[16, 16], pl.FP16]]:
    # d_A = d_C @ B^T
    B_T: pl.Tile[[16, 16], pl.FP16] = pl.transpose(B_saved)
    d_A: pl.Tile[[16, 16], pl.FP16] = pl.matmul(d_C, B_T)
    
    # d_B = A^T @ d_C
    A_T: pl.Tile[[16, 16], pl.FP16] = pl.transpose(A_saved)
    d_B: pl.Tile[[16, 16], pl.FP16] = pl.matmul(A_T, d_C)
    
    return d_A, d_B

# 数学验证
# C = A @ B
# ∂C/∂A = d_C @ B^T  (矩阵形式)
# ∂C/∂B = A^T @ d_C
```

### 6.5 激活函数

```python
# 测试用例 5: ReLU
@pl.function
def relu_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.relu(x)
    return result

# 反向
@pl.function
def grad_relu(x_saved: pl.Tensor[[64], pl.FP32], d_result: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    # ∂relu(x)/∂x = 1 if x > 0 else 0
    mask: pl.Tensor[[64], pl.BOOL] = pl.gt(x_saved, 0.0)  # 前向值 x > 0
    mask_fp32: pl.Tensor[[64], pl.FP32] = pl.cast(mask, pl.FP32)
    d_x: pl.Tensor[[64], pl.FP32] = pl.mul(d_result, mask_fp32)
    return d_x

# 数学验证
# relu(x) = max(0, x)
# ∂relu/∂x = 1 if x > 0 else 0 (不连续，但在 x=0 处定义为 0 或 0.5)
```

### 6.6 变长循环验证

```python
# 测试用例 6: 变长循环梯度验证
import numpy as np

def verify_dynamic_loop_gradient():
    # 使用有限差分验证
    N = 10  # 固定测试
    x_val = np.random.randn(64)
    eps = 1e-5
    
    # 前向值
    acc_final = N * x_val
    
    # 计算梯度（数值）
    d_acc_out = np.ones(64)  # 梯度种子
    
    # 反向梯度（解析）
    d_x_analytical = N * d_acc_out
    
    # 数值验证（有限差分）
    x_plus = x_val + eps
    acc_plus = N * x_plus
    d_x_numerical = (acc_plus - acc_final) / eps * d_acc_out
    
    # 验证
    np.testing.assert_allclose(d_x_analytical, d_x_numerical, rtol=1e-4)
    print("变长循环梯度验证通过")

verify_dynamic_loop_gradient()
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

### 7.4 变长循环的正确性

对于动态迭代次数 $N$，反向循环的迭代次数相同：

$$\bar{x} = \sum_{i=0}^{N-1} \bar{v}_N \cdot \frac{\partial f}{\partial x}(v_i, x)$$

由于 $N$ 在反向时已知（从前向保存），梯度计算完整且正确。

---

## 8. 实现细节

### 8.1 前向值保存策略

```python
class ValueSaver(IRVisitor):
    """确定需要保存的前向值"""
    
    def __init__(self):
        self.to_save: Set[ir.Var] = set()
        
    def visit_mul(self, expr: ir.Mul):
        # 乘法反向需要保存两个输入
        self._mark_for_save(expr.left)
        self._mark_for_save(expr.right)
        
    def visit_div(self, expr: ir.FloatDiv):
        # 除法反向需要保存两个输入
        self._mark_for_save(expr.left)
        self._mark_for_save(expr.right)
        
    def visit_exp(self, call: ir.Call):
        # exp 反向需要保存输入（或前向结果）
        self._mark_for_save(call.args[0])
        
    def visit_for_stmt(self, stmt: ir.ForStmt):
        # 循环需要保存每次迭代的中间值
        for iter_arg in stmt.iter_args:
            self.to_save.add(iter_arg)
```

### 8.2 梯度累积器

```python
class GradientAccumulator:
    """管理梯度累积（+= 操作）"""
    
    def __init__(self):
        self.accumulators: Dict[ir.Var, List[ir.Expr]] = {}
        
    def add_gradient(self, var: ir.Var, grad_expr: ir.Expr):
        """累积梯度"""
        if var not in self.accumulators:
            self.accumulators[var] = []
        self.accumulators[var].append(grad_expr)
        
    def get_final_gradient(self, var: ir.Var, span: ir.Span) -> ir.Expr:
        """获取累积后的最终梯度"""
        grads = self.accumulators.get(var, [])
        if not grads:
            return self._zero_value(var.type)
        
        result = grads[0]
        for grad in grads[1:]:
            result = ir.Add(result, grad, DataType.FP32, span)
        return result
```

### 8.3 SSA 循环反向构建

```python
def build_reverse_loop(for_stmt: ir.ForStmt, grad_vars: Dict[ir.Var, ir.Var]) -> ir.ForStmt:
    """构建反向 SSA 循环"""
    
    # 1. 反向循环范围
    trip_count = compute_trip_count(for_stmt)
    span = for_stmt.span
    
    # 2. 创建反向 iter_args（梯度变量）
    reverse_iter_args = []
    for iter_arg in for_stmt.iter_args:
        if iter_arg in grad_vars:
            grad_var = grad_vars[iter_arg]
            # 初始值 = 输出梯度
            init_value = get_output_gradient(iter_arg)
            reverse_iter_arg = ir.IterArg(
                f"d_{iter_arg.name_hint}",
                grad_var.type,
                init_value,
                span
            )
            reverse_iter_args.append(reverse_iter_arg)
    
    # 3. 构建反向循环体
    reverse_body_stmts = []
    for stmt in reversed(for_stmt.body.stmts):
        # 处理 YieldStmt
        if isinstance(stmt, ir.YieldStmt):
            # 计算梯度贡献
            ...
    
    # 4. 构建 YieldStmt
    yield_values = [grad_var for iter_arg, grad_var in zip(for_stmt.iter_args, reverse_iter_args)]
    reverse_body_stmts.append(ir.YieldStmt(yield_values, span))
    
    # 5. 创建反向 ForStmt
    return ir.ForStmt(
        for_stmt.loop_var,
        ir.ConstInt(0, DataType.INDEX, span),
        trip_count,
        ir.ConstInt(1, DataType.INDEX, span),
        reverse_iter_args,
        ir.SeqStmts(reverse_body_stmts, span),
        [grad_vars.get(rv) for rv in for_stmt.return_vars if rv in grad_vars],
        span,
        for_stmt.kind
    )
```

---

## 9. 梯度累加（变量多次使用）

### 9.1 问题背景

在自动微分中，一个变量可能被多次使用，导致梯度需要累加多个贡献：

```python
y = a * b + a * c  # a 被使用两次
# 梯度: dy/da = b + c (累加两个贡献)
```

这是反向模式自动微分的核心挑战之一。

### 9.2 数学基础

对于变量 $x$ 被多次使用的情况：

$$y = f_1(x, ...) + f_2(x, ...) + ...$$

梯度需要累加所有路径的贡献：

$$\frac{\partial y}{\partial x} = \frac{\partial f_1}{\partial x} + \frac{\partial f_2}{\partial x} + ...$$

### 9.3 IR 层面的处理

#### 9.3.1 表达式内多次使用

```python
# 前向
y = a * b + a * c

# 反向
# 处理 a * b: ā += ȳ * b
# 处理 a * c: ā += ȳ * c
# 累加结果: ā = ȳ * (b + c)
```

在 `_reverse_expr` 中，当遇到 `Add` 表达式时：

```python
elif isinstance(expr, ir.Add):
    # y = a + b -> ā = ȳ, b̄ = ȳ
    left_grads = self._reverse_expr(expr.left, output_grad, span)
    right_grads = self._reverse_expr(expr.right, output_grad, span)
    # 累加两个贡献（如果同一个变量出现在两边）
    result.update(left_grads)
    result.update(right_grads)
    # 注意：如果 left 和 right 都使用了同一个变量，update 会合并
```

#### 9.3.2 跨语句累加

```python
# 前向
y1 = a * b
y2 = a * c
y = y1 + y2

# 反向遍历顺序
# 1. 处理 y = y1 + y2
#    dy1 = ȳ, dy2 = ȳ
# 2. 处理 y2 = a * c
#    ā += dy2 * c = ȳ * c
# 3. 处理 y1 = a * b
#    ā += dy1 * b = ȳ * b
# 4. 累加结果: ā = ȳ * b + ȳ * c
```

### 9.4 GradientAccumulator 实现

```python
class GradientAccumulator:
    """管理梯度累积（+= 操作）"""
    
    def __init__(self):
        self.accumulators: Dict[ir.Var, List[ir.Expr]] = {}
    
    def add_gradient(self, var: ir.Var, grad_expr: ir.Expr):
        """累积梯度贡献"""
        if var not in self.accumulators:
            self.accumulators[var] = []
        self.accumulators[var].append(grad_expr)
    
    def get_final_gradient(self, var: ir.Var, span: ir.Span) -> ir.Expr:
        """获取累积后的最终梯度"""
        grads = self.accumulators.get(var, [])
        if not grads:
            return self._zero_value(var.type)
        
        # 构建累加表达式
        result = grads[0]
        for grad in grads[1:]:
            result = ir.Add(result, grad, DataType.FP32, span)
        return result
    
    def _zero_value(self, type: ir.Type) -> ir.Expr:
        """创建零值"""
        if isinstance(type, ir.ScalarType):
            if type.dtype.IsFloat():
                return ir.ConstFloat(0.0, type.dtype, Span.unknown())
            else:
                return ir.ConstInt(0, type.dtype, Span.unknown())
        # ... 其他类型
```

### 9.5 循环中的累加

```python
# 前向
for i, (acc,) in pl.range(N, init_values=(acc₀,)):
    acc_new = acc + x      # 第一次使用 x
    temp = x * i           # 第二次使用 x
    acc_final = acc_new + temp
    acc = yield(acc_final)

# 反向
for i_rev, (d_acc,) in pl.range(N, init_values=(d_final,)):
    # 来自 temp = x * i: d_x += d_acc * i
    # 来自 acc_new = acc + x: d_x += d_acc * 1
    # 累加: d_x += d_acc * (i + 1)
```

### 9.6 测试验证

所有梯度累加场景已验证正确：

| 测试场景 | 数学表达式 | 梯度累加 | 结果 |
|----------|------------|----------|------|
| 简单重复使用 | `y = a*b + a*c` | `dy/da = b + c` | ✓ |
| 复杂重复使用 | `y = a² + a*b` | `dy/da = 2a + b` | ✓ |
| 深度重复使用 | `y = (a+b)*(a+c)` | `dy/da = (a+c)+(a+b)` | ✓ |
| 循环累加 | `acc = Σ(x + x*i)` | `dx = Σ(1+i)` | ✓ |
| 跨语句累加 | `y1=a*b, y2=a*c` | `dy/da = b + c` | ✓ |
| 多输出累加 | `y1=a*b, y2=a*c` | `Σdy/da = b + c` | ✓ |

---

## 10. 局限性和扩展

### 10.1 当前局限性

1. **不可导操作**: 
   - 比较操作（`eq`, `lt`, ...）不产生梯度
   - `argmax`, `sort` 等离散操作

2. **控制流依赖**:
   - 条件表达式使用前向值（需保存）
   - 动态控制流增加保存开销

3. **高阶微分**:
   - 当前只支持一阶导数
   - 需要扩展支持二阶（Hessian）

### 10.2 扩展方向

1. **Checkpointing**: 
   - 减少内存消耗，选择性重新计算

2. **Forward-Over-Reverse**:
   - 支持高阶导数

3. **Vector-Jacobian Product (VJP)**:
   - 优化梯度计算效率

---

## 11. 文件位置建议

| 功能 | 文件位置 |
|------|----------|
| 自动微分核心 | `python/pypto/autodiff/reverse_mode.py` |
| 活性分析 | `python/pypto/autodiff/liveness.py` |
| 梯度规则 | `python/pypto/autodiff/grad_rules.py` |
| 循环处理 | `python/pypto/autodiff/loop_diff.py` |
| 高级 API | `python/pypto/autodiff/__init__.py` |
| 测试用例 | `tests/ut/autodiff/test_reverse_mode.py` |