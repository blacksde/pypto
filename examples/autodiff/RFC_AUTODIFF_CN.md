# RFC: PyPTO Source-to-Source 自动微分功能

- **RFC 编号**: RFC-001
- **作者**: PyPTO 团队
- **日期**: 2026-04-17
- **状态**: 草稿 (Draft)
- **类别**: 核心功能
- **目标版本**: PyPTO v0.1.0

---

## 概述

本 RFC 提议为 PyPTO 增加 **Source-to-Source 自动微分 (AD)** 功能。系统将直接从前向计算 IR 生成梯度函数，为深度学习和科学计算应用提供高效的梯度计算能力。

核心特性：
- **反向模式 AD**: 适用于多输入少输出的函数，计算效率高
- **IR 层变换**: 直接生成梯度 IR，无需运行时 Tape
- **梯度规则注册表**: 可扩展的自定义算子梯度系统
- **控制流支持**: 正确处理循环和条件分支

---

## 动机与背景

### 背景说明

PyPTO 是面向 GPU/加速器优化的 Tile 编程框架。在深度学习和科学计算中，梯度计算至关重要：

1. **深度学习训练**: 反向传播需要高效的梯度计算
2. **优化问题**: 梯度下降算法依赖导数计算
3. **科学计算**: 自动微分减少手动求导负担

现有框架 (PyTorch, TensorFlow) 使用运行时 Tape-based AD：
- 引入运行时开销
- 执行期间需存储中间值
- 无法提前优化梯度计算

### 为什么选择 Source-to-Source AD?

PyPTO 的 IR 层方法提供：

| 方法 | 运行时 Tape | Source-to-Source |
|------|-------------|-------------------|
| 执行方式 | 动态执行 | 静态生成 |
| 优化程度 | 受限 | 完整编译器优化 |
| 内存使用 | 存储所有中间值 | 选择性存储 |
| 控制流处理 | 运行时追踪 | IR 层处理 |

**优势**:
- **编译器优化**: 梯度 IR 可像其他代码一样优化
- **内存效率**: 选择性前向值存储（仅当需要时）
- **循环处理**: SSA 循环梯度无需运行时追踪
- **性能**: 无梯度构建的运行时开销

### 目标

1. 支持 tile/tensor 算子的梯度计算
2. 处理循环 (SSA iter_args + Tape 模式)
3. 处理条件分支 (Phi 节点处理)
4. 可扩展的梯度规则注册系统
5. 数学正确性验证

---

## 详细设计

### 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│              自动微分流水线                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  前向函数 IR                                                 │
│       │                                                     │
│       ├─> 1. 创建梯度参数                                    │
│       │      (forward_params + d_output)                    │
│       │                                                     │
│       ├─> 2. 收集需要保存的前向值                             │
│       │      - 分析反向语句中引用的变量                       │
│       │      - 区分循环内/循环外变量                          │
│       │                                                     │
│       ├─> 3. 反向遍历前向函数体                               │
│       │      - 处理每个语句                                  │
│       │      - 生成反向梯度语句                               │
│       │      - 处理控制流 (ForStmt, IfStmt)                  │
│       │                                                     │
│       ├─> 4. 生成正向计算语句                                 │
│       │      - 重新计算循环外中间值                           │
│       │                                                     │
│       ├─> 5. 构建反向函数体                                   │
│       │      - 拼接正向计算 + 反向语句                        │
│       │      - 添加返回语句                                  │
│       │                                                     │
│       └─> 梯度函数 IR                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 数学基础

**链式法则**:

$$\frac{\partial y}{\partial x} = \prod_{i=n}^{1} \frac{\partial f_i}{\partial f_{i-1}}$$

**反向模式**: 对于复合函数 $y = f_n(f_{n-1}(...f_1(x)))$，从输出到输入计算梯度：

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

**梯度累加**: 当变量被多次使用时：

$$\frac{\partial y}{\partial x} = \sum_{paths} \frac{\partial y}{\partial x}_{path}$$

### 3. IR 节点处理

#### 3.1 语句类型

| 节点 | 前向 | 反向 |
|------|------|------|
| `AssignStmt` | `y = op(a, b)` | `d_a += grad_op(dy, a, b)` |
| `SeqStmts` | 顺序执行 | 反向遍历 |
| `ForStmt` | 循环执行 | 反向循环 + Tape |
| `IfStmt` | 分支执行 | 分支结构保留 |
| `YieldStmt` | SSA 循环状态 | 梯度传播 |
| `ReturnStmt` | 返回值 | 梯度初始化 |

#### 3.2 循环处理（核心难点）

**两种模式**:

| 模式 | 条件 | 处理方法 |
|------|------|----------|
| iter_args + yield | SSA reduce 模式 | `_reverse_for_iter_args` |
| 普通循环 | 无 iter_args/yield | `_reverse_for_with_tape` |

**SSA Reduce 模式**:
```
前向:
  for i, (acc,) in range(N, init=(acc_init)):
    acc_new = f(acc_iter, x)
    acc = yield(acc_new)

反向:
  for i_rev, (d_acc,) in range(N, init=(d_output)):
    # 从 tape 获取前向保存的值
    saved = tape[i_rev]
    # 计算梯度贡献
    d_x += grad(d_acc, saved)
    d_acc_prev = yield(grad_acc)
```

**Tape 模式**:
- 创建 TensorArray 保存循环值
- 前向：每次迭代 push 值
- 反向：get 值用于梯度计算

#### 3.3 条件分支处理

```
前向:
  if cond:
    y = f1(x)
  else:
    y = f2(x)

反向:
  if cond:  # 使用相同条件
    d_x = grad_f1(d_y)
  else:
    d_x = grad_f2(d_y)
```

**关键点**: 反向保留分支结构，使用前向条件。

### 4. 梯度规则注册

#### 4.1 注册架构

```python
class GradientRegistry:
    _registry: Dict[str, Callable] = {}
    _categories: Dict[str, str] = {}
    
    @classmethod
    def register(op_name, grad_func, category):
        _registry[op_name] = grad_func
    
    @classmethod
    def get(op_name) -> Callable:
        return _registry.get(op_name)
```

#### 4.2 梯度规则签名

```python
def grad_rule(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    计算算子输入的梯度
    
    参数:
        saved_inputs: 前向输入值（保存用于反向）
        d_output: 输出梯度（来自下游）
        kwargs: 额外信息（如 forward_result）
    
    返回:
        每个输入对应的梯度表达式列表
    """
```

#### 4.3 内置算子

| 算子 | 梯度数学 |
|------|----------|
| `tile.add` | `ā=ȳ, b̄=ȳ` |
| `tile.mul` | `ā=ȳ·b, b̄=ȳ·a` |
| `tile.matmul` | `Ā=ȳ@B^T, B̄=A^T@ȳ` |
| `tile.exp` | `ā=ȳ·exp(a)` |
| `tile.relu` | `ā=ȳ·sign(a>0)` |
| `tile.sin` | `ā=ȳ·cos(a)` |

**示例**:
```python
@register_grad('tile.mul', 'tile')
def tile_mul_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    da = ir.Call(ir.Op('tile.mul'), [d_output, b], ...)
    db = ir.Call(ir.Op('tile.mul'), [d_output, a], ...)
    return [da, db]
```

### 5. 接口设计

#### 5.1 高层 API

```python
# 生成梯度函数
grad_func = pl.grad(forward_func, params=['x', 'w'])
# grad_func(x, w, d_output) -> (d_x, d_w)

# 生成前向+反向函数对
forward, backward = pl.value_and_grad(forward_func)
# forward(x, w) -> y
# backward(x, w, d_y) -> (d_x, d_w)
```

#### 5.2 自定义梯度注册

```python
# 装饰器方式
@register_grad('my.custom_op', 'user')
def custom_grad(saved_inputs, d_output, kwargs):
    return [grad_expr1, grad_expr2]

# 直接注册
GradientRegistry.register('my.op', grad_func, 'user')
```

#### 5.3 查询 API

```python
GradientRegistry.list_all()           # 列出所有已注册算子
GradientRegistry.list_by_category('tile')  # 按分类列出
GradientRegistry.has('tile.add')      # 检查是否已注册
GradientRegistry.get('tile.add')      # 获取梯度规则
```

---

## 实现路线图 (RoadMap)

### 第一阶段：核心基础设施 (第 1-2 周)

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 梯度注册表 | 实现 `GradientRegistry` 类 | 高 |
| 核心梯度规则 | 注册 15 个内置算子 | 高 |
| 基础 IR 生成 | 前向 IR 到反向 IR | 高 |
| 测试框架 | 数值验证测试 | 高 |

**交付物**:
- `gradient_registry.py` (注册表 + 装饰器)
- `grad_rules.py` (15 个内置规则)
- `reverse_mode.py` (核心构建器)
- 数值验证测试套件

### 第二阶段：控制流支持 (第 3-4 周)

| 任务 | 描述 | 优先级 |
|------|------|--------|
| ForStmt 处理 | SSA iter_args + Tape 模式 | 高 |
| IfStmt 处理 | 分支结构保留 | 中 |
| YieldStmt 处理 | SSA 循环状态 | 中 |
| 循环 Tape 分析 | 识别需要 Tape 的变量 | 高 |

**交付物**:
- 循环梯度生成
- Tape-based 循环处理
- 分支梯度处理
- 循环测试用例

### 第三阶段：高级特性 (第 5-6 周)

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 梯度累加 | 多次使用处理 | 中 |
| 前向值管理 | 选择性重计算 | 中 |
| 嵌套调用处理 | 递归梯度计算 | 中 |
| 算子名映射 | tile/tensor 变体 | 低 |

**交付物**:
- 多使用变量的梯度累加
- 高效的前向值策略
- 嵌套表达式梯度

### 第四阶段：文档与集成 (第 7-8 周)

| 任务 | 描述 | 优先级 |
|------|------|--------|
| API 文档 | 用户指南 + 示例 | 高 |
| 设计文档 | RFC + 技术文档 | 高 |
| 集成测试 | 端到端测试 | 高 |
| 性能基准 | 与 PyTorch 对比 | 中 |

**交付物**:
- 用户文档
- 开发者文档
- 集成测试
- 性能基准

### 第五阶段：社区贡献 (第 9-10 周)

| 任务 | 描述 | 优先级 |
|------|------|--------|
| RFC 审查流程 | 社区审查工作流 | 高 |
| 示例程序 | 教程示例 | 中 |
| 贡献指南 | 如何添加梯度规则 | 中 |
| Issue 模板 | 问题报告模板 | 低 |

**交付物**:
- 社区审查流程
- 示例程序
- 贡献指南
- Issue 模板

---

## 局限性分析

### 当前限制

1. **不可导操作**: argmax、sort、比较操作不产生梯度
2. **动态迭代次数**: 需要前向值存储 (Tape 开销)
3. **高阶导数**: 当前仅支持一阶梯度
4. **控制流依赖**: 分支条件需保存前向值

### 内存考虑

- Tape 模式在 TensorArray 中存储循环中间值
- 内存随迭代次数增长
- 替代方案：重计算策略（用计算换内存）

---

## 替代方案对比

### 1. 运行时 Tape-based AD (PyTorch 风格)

**优点**:
- 动态控制流支持
- 成熟的生态系统

**缺点**:
- 运行时开销
- 无法提前优化梯度代码
- 内存密集（存储所有中间值）

**决策**: 选择 Source-to-Source 以获得编译器优化优势

### 2. 前向模式 AD

**优点**:
- 实现更简单
- 少输入多输出时高效

**缺点**:
- 不适合深度学习（多参数）
- 对目标用例效率较低

**决策**: 选择反向模式以获得 DL 效率

### 3. 仅手动梯度规则

**优点**:
- 最大控制权
- 无自动系统复杂性

**缺点**:
- 手动求导易出错
- 扩展性有限
- 开发体验差

**决策**: 选择自动 + 可扩展注册表

---

## 未解决问题

1. **Checkpoint 策略**: 如何平衡重计算与存储？
   - 提议：用户可配置 checkpoint 策略
   - 需要：性能基准验证

2. **高阶导数**: 是否支持二阶导数 (Hessian)？
   - 当前：仅一阶
   - 未来：递归应用 AD

3. **算子覆盖**: tile/tensor 算子的完整覆盖？
   - 当前：15 个核心算子
   - 需要：社区贡献扩展算子

4. **性能对比**: 与 PyTorch/JAX 的基准对比？
   - 需要：系统性基准测试
   - 指标：计算时间、内存使用

---

## 参考文献

1. Baydin, A. G., et al. "Automatic differentiation in machine learning: a survey." JMLR 2018
2. PyTorch Autograd: https://pytorch.org/docs/stable/autograd.html
3. JAX Automatic Differentiation: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
4. PyPTO IR 设计文档: `examples/ir_analysis/01_ir_node_types_20260416.md`
5. PyPTO 自动微分设计文档: `examples/autodiff/autodiff_design_doc.md`

---

## 实现状态

| 组件 | 状态 | 文件 |
|------|------|------|
| 梯度注册表 | ✅ 完成 | `gradient_registry.py` |
| 核心梯度规则 | ✅ 完成 | `grad_rules.py` (15 算子) |
| 反向模式构建器 | ✅ 完成 | `reverse_mode.py` |
| 循环处理 | ✅ 完成 | SSA + Tape 模式 |
| 分支处理 | ✅ 完成 | IfStmt 结构保留 |
| 数值测试 | ✅ 完成 | 误差 < 1e-4 |
| API 函数 | ✅ 完成 | `grad`, `value_and_grad` |

---

## 结论

Source-to-source 自动微分为 PyPTO 的 Tile 编程模型提供显著优势：

- **编译器优化**: 梯度 IR 可像其他代码一样优化
- **内存效率**: 选择性存储，而非运行时 Tape
- **性能**: 无梯度构建的运行时开销
- **扩展性**: 社区可贡献梯度规则

本 RFC 提议一个全面的实现路线图，分阶段交付，使 PyPTO 能够支持深度学习和科学计算应用的梯度计算。社区贡献将通过标准化的 RFC 流程、梯度规则注册系统和完善的文档指南来促进。