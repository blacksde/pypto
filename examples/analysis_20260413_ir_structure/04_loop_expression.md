# PyPTO 循环表达分析

## 1. 循环的基础结构

### 1.1 ForStmt IR 节点定义

**文件**: `include/pypto/ir/stmt.h` (lines 488-560)

```cpp
class ForStmt : public Stmt {
 public:
  VarPtr loop_var_;                    // 循环变量 (如 for i in range 中的 i)
  ExprPtr start_;                      // 起始值表达式
  ExprPtr stop_;                       // 终止值表达式
  ExprPtr step_;                       // 步长表达式
  std::vector<IterArgPtr> iter_args_;  // 循环携带变量 (作用域限于循环体内)
  StmtPtr body_;                       // 循环体语句 (若有 iter_args 则必须包含 yield)
  std::vector<VarPtr> return_vars_;    // 捕获最终迭代值的变量
  ForKind kind_;                       // 循环类型 (Sequential, Parallel, Unroll)
  std::optional<ExprPtr> chunk_size_;  // 分块大小 (nullopt = 不分块)
  ChunkPolicy chunk_policy_;           // 分块分布策略
  LoopOrigin loop_origin_;             // 循环来源分类
};
```

### 1.2 相关枚举类型

**ForKind** (循环类型):

```cpp
enum class ForKind : uint8_t {
  Sequential = 0,  // 标准顺序循环 (默认)
  Parallel = 1,    // 并行循环
  Unroll = 2       // 编译时展开循环
};
```

| DSL 函数 | ForKind | 说明 |
|----------|---------|------|
| `pl.range()` | `Sequential` | 默认顺序循环 |
| `pl.parallel()` | `Parallel` | 可并行化循环 |
| `pl.unroll()` | `Unroll` | 编译时展开 (不支持 init_values) |

**LoopOrigin** (循环来源):

```cpp
enum class LoopOrigin : uint8_t {
  Original = 0,       // 常规循环 (默认)
  ChunkOuter = 1,     // 分块拆分后的外层循环
  ChunkInner = 2,     // 分块拆分后的内层循环
  ChunkRemainder = 3  // 分块拆分后的余数循环
};
```

**ChunkPolicy** (分块策略):

```cpp
enum class ChunkPolicy : uint8_t {
  LeadingFull = 0  // 先处理完整块，末尾处理较小的余数块
};
```

---

## 2. IR 上循环轴的表达

### 2.1 循环变量 (loop_var_)

循环变量是一个 `VarPtr`，表示迭代变量（如 `i` in `for i in range(...)`）：
- 循环轴通过 `loop_var` 隐式表达，该变量取值范围为 `[start, stop)`，步长为 `step`
- 循环变量的类型通常为 `ScalarType(DataType::INDEX)` 或 `ScalarType(DataType::INT64)`

### 2.2 范围指定 (start_, stop_, step_)

三个字段均为 `ExprPtr`，可以是：
- `ConstInt` (静态常量)
- 动态 `Scalar` 变量
- 任意表达式

```cpp
// 静态范围
start_ = ConstInt(0)
stop_  = ConstInt(10)
step_  = ConstInt(1)

// 动态范围
start_ = ConstInt(0)
stop_  = Var("n")      // 动态变量
step_  = ConstInt(1)

// 表达式范围
stop_ = Add(ConstInt(2), Var("n"))  // 表达式作为边界
```

### 2.3 DSL 语法映射

| Python DSL | IR 字段 |
|------------|---------|
| `pl.range(10)` | `start=0, stop=10, step=1` |
| `pl.range(0, n)` | `start=0, stop=n, step=1` |
| `pl.range(0, n, 2)` | `start=0, stop=n, step=2` |
| `pl.range(n * 2)` | `stop=Mul(n, 2)` |

---

## 3. IterArg：循环携带变量

### 3.1 IterArg 定义

**文件**: `include/pypto/ir/expr.h` (lines 261-321)

```cpp
class IterArg : public Var {
 public:
  ExprPtr initValue_;  // 首次迭代的初始值表达式

  IterArg(std::string name_hint, TypePtr type, ExprPtr initValue, Span span)
      : Var(std::move(name_hint), std::move(type), std::move(span)), 
        initValue_(std::move(initValue)) {}
};
```

### 3.2 作用域规则

| 变量类型 | 作用域 | 可访问区域 |
|----------|--------|------------|
| `loop_var_` | 循环体 | 循环体内 |
| `iter_args_` (IterArg) | 循环体 | 仅循环体内 |
| `return_vars_` (Var) | 循环外 | 循环结束后 |

**关键点**:
- `IterArg` 变量作用域仅限于循环体，循环外无法直接访问
- 必须通过 `return_vars` 将最终值暴露到循环外

### 3.3 使用模式

```
┌─────────────────────────────────────────────────────────────┐
│                    ForStmt 结构                              │
│                                                              │
│  iter_args (IterArg)                                         │
│  ├── initValue_ → 首次迭代初始值                              │
│  └── 作用域: 仅循环体内                                       │
│                                                              │
│  body                                                        │
│  ├── 使用 iter_args 计算新值                                  │
│  └── YieldStmt 返回新值给下一迭代                             │
│                                                              │
│  return_vars (Var)                                           │
│  ├── 捕获最终迭代值                                           │
│  └── 作用域: 循环外可访问                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. pl.range 和 pl.yield 的配合关系

### 4.1 pl.range DSL API

**文件**: `python/pypto/language/dsl_api.py`

```python
def range(
    *args: RangeArg,
    init_values: tuple[Any, ...] | None = None,
    chunk: int | None = None,
    chunk_policy: str = "leading_full",
) -> RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]:
    """创建循环迭代器。

    支持两种模式:
        简单模式:    for i in pl.range(10):
        迭代参数模式: for i, (var1, var2) in pl.range(16, init_values=(init1, init2)):
        分块模式:    for i in pl.range(0, 10, chunk=5):
    """
```

**参数说明**:
- `*args`: `stop` 或 `(start, stop)` 或 `(start, stop, step)`
- `init_values`: 循环携带变量的初始值（创建 `IterArg`）
- `chunk`: 分块大小（在 `pl.auto_incore()` 内使用）

**返回值**:
- 无 `init_values`: 仅返回循环变量 `Scalar`
- 有 `init_values`: 返回元组 `(loop_var, (iter_arg_values...))`

### 4.2 pl.yield_ DSL API

**文件**: `python/pypto/language/dsl_api.py`

```python
def yield_(*values: Any) -> Any | tuple[Any, ...]:
    """从作用域 (for, if) 返回值。

    此函数用于从嵌套作用域显式返回值，创建 SSA phi 节点。

    Args:
        *values: 要返回的值

    Returns:
        返回的值。单个值返回该值，多个值返回元组。
    """
    if len(values) == 1:
        return values[0]
    return tuple(values)
```

### 4.3 YieldStmt IR 节点

**文件**: `include/pypto/ir/stmt.h` (lines 385-424)

```cpp
class YieldStmt : public Stmt {
 public:
  std::vector<ExprPtr> value_;  // 要返回的表达式列表

  YieldStmt(std::vector<ExprPtr> value, Span span);
  explicit YieldStmt(Span span);  // 空 yield
};
```

### 4.4 yield 与循环的关系

```
迭代 N 的数据流:

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  IterArg.initValue_ ────────────────────────────────────────┐│
│       (首次迭代)                                             ││
│                                                              ││
│  ┌─────────────────────────────────────────────────────────┐││
│  │                    迭代 N                                │││
│  │                                                          │││
│  │  iter_args[N] ──→ 计算 ──→ YieldStmt.value[N]           │││
│  │                                                          │││
│  │  YieldStmt.value[N] ──→ iter_args[N+1]                   │││
│  │       (传递给下一迭代)                                    │││
│  └─────────────────────────────────────────────────────────┘││
│                                                              ││
│  最终迭代结束后:                                              ││
│  YieldStmt.value ──→ return_vars                             ││
│       (输出到循环外)                                          ││
│                                                              ││
└─────────────────────────────────────────────────────────────┘
```

**关键约束**:
- 有 `iter_args` 的循环**必须**在循环体中包含 `YieldStmt`
- `YieldStmt` 的值数量必须等于 `iter_args` 的数量
- 不匹配会触发 `YIELD_COUNT_MISMATCH` 验证错误

### 4.5 完整示例

**Python DSL**:

```python
@pl.function
def sum_loop(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
    for i, (sum_val,) in pl.range(0, n, 1, init_values=(0,)):
        new_sum = sum_val + i
        result = pl.yield_(new_sum)
    return result
```

**IR 结构**:

```
Function("sum_loop")
  params: [n: ScalarType(INT64)]
  return_types: [ScalarType(INT64)]
  body: ForStmt
    loop_var: Var("i", ScalarType(INDEX))
    start: ConstInt(0)
    stop: Var("n")
    step: ConstInt(1)
    iter_args: [IterArg("sum_val", ScalarType(INT64), initValue=ConstInt(0))]
    body: SeqStmts([
      AssignStmt(var=Var("new_sum"), value=Call(Add, [sum_val, i])),
      YieldStmt(value=[Var("new_sum")])
    ])
    return_vars: [Var("result", ScalarType(INT64))]
    kind: Sequential
```

---

## 5. 多种循环模式

### 5.1 简单循环 (无 iter_args)

```python
for i in pl.range(10):
    # 循环体
```

**IR**:
- `iter_args_`: 空
- `return_vars_`: 空
- `body_`: 不需要 `YieldStmt`

### 5.2 单个 iter_arg

```python
for i, (sum_val,) in pl.range(10, init_values=(0,)):
    new_sum = pl.add(sum_val, i)
    result = pl.yield_(new_sum)
```

**IR**:
- `iter_args_`: `[IterArg("sum_val", init=0)]`
- `return_vars_`: `[Var("result")]`
- `body_`: 包含 `YieldStmt([new_sum])`

### 5.3 多个 iter_args

```python
for i, (val1, val2) in pl.range(5, init_values=(init1, init2)):
    new1 = pl.add(val1, i)
    new2 = pl.mul(val2, 2)
    out1, out2 = pl.yield_(new1, new2)
```

**IR**:
- `iter_args_`: `[IterArg("val1", init=init1), IterArg("val2", init=init2)]`
- `return_vars_`: `[Var("out1"), Var("out2")]`
- `body_`: 包含 `YieldStmt([new1, new2])`

### 5.4 嵌套循环

```python
for i, (outer,) in pl.range(3, init_values=(init,)):
    for j, (inner,) in pl.range(2, init_values=(outer,)):
        new_inner = pl.add(inner, 1)
        inner_out = pl.yield_(new_inner)
    outer_out = pl.yield_(inner_out)
```

**数据流**:
- 外层 `iter_args` 的值传入内层循环的 `init_values`
- 内层循环的 `return_vars` 传入外层的 `yield`

### 5.5 If 语句中的 yield (phi 节点)

```python
if i == 0:
    new_val = pl.mul(acc, 2.0)
    val = pl.yield_(new_val)  # 分支 1
else:
    val = pl.yield_(acc)      # 分支 2
# val 是合并两个分支的 phi 节点
```

---

## 6. 如何遍历一个循环节点

### 6.1 IRVisitor 遍历方法

**文件**: `src/ir/transforms/visitor.cpp` (lines 180-199)

```cpp
void IRVisitor::VisitStmt_(const ForStmtPtr& op) {
  // 1. 访问循环变量 (及其类型形状)
  VisitExpr(op->loop_var_);
  
  // 2. 访问循环边界
  VisitExpr(op->start_);
  VisitExpr(op->stop_);
  VisitExpr(op->step_);
  
  // 3. 访问 iter_args (每个 IterArg 的 initValue_ 也会被访问)
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    VisitExpr(op->iter_args_[i]);
  }
  
  // 4. 访问循环体
  VisitStmt(op->body_);
  
  // 5. 访问 return_vars
  for (size_t i = 0; i < op->return_vars_.size(); ++i) {
    VisitExpr(op->return_vars_[i]);
  }
}
```

### 6.2 IRMutator 变换方法

**文件**: `src/ir/transforms/mutator.cpp` (lines 413-501)

```cpp
StmtPtr IRMutator::VisitStmt_(const ForStmtPtr& op) {
  // 1. 变换 loop_var 和边界
  auto new_loop_var = As<Var>(VisitExpr(op->loop_var_));
  auto new_start = VisitExpr(op->start_);
  auto new_stop = VisitExpr(op->stop_);
  auto new_step = VisitExpr(op->step_);

  // 2. 变换 iter_args，并在访问 body 前注册重映射
  std::vector<IterArgPtr> new_iter_args;
  for (const auto& ia : op->iter_args_) {
    auto new_ia = As<IterArg>(VisitExpr(ia));
    new_iter_args.push_back(new_ia);
    // 注册 old→new 映射，用于 body 内引用
    if (new_ia.get() != ia.get()) {
      var_remap_[ia.get()] = new_ia;
    }
  }

  // 3. 访问循环体 (重映射生效)
  auto new_body = VisitStmt(op->body_);

  // 4. 在访问 return_vars 前清理重映射
  for (const auto& old_ia : op->iter_args_) {
    var_remap_.erase(old_ia.get());
  }

  // 5. 访问 return_vars (清理后，它们是独立的 Var 对象)
  std::vector<VarPtr> new_return_vars;
  for (const auto& rv : op->return_vars_) {
    new_return_vars.push_back(As<Var>(VisitExpr(rv)));
  }

  // 6. 写时复制: 仅当有变化时重建
  if (changed) {
    return std::make_shared<ForStmt>(...);
  }
  return op;  // 未改变则返回原指针
}
```

### 6.3 遍历顺序总结

| 阶段 | 访问内容 | 说明 |
|------|----------|------|
| 1 | `loop_var_` | 循环变量定义 |
| 2 | `start_, stop_, step_` | 循环边界表达式 |
| 3 | `iter_args_` | 循环携带变量及其初始值 |
| 4 | `body_` | 循环体语句 |
| 5 | `return_vars_` | 循环输出变量 |

### 6.4 Python 遍历示例

```python
from pypto.ir import IRVisitor

class LoopVisitor(IRVisitor):
    def visit_for_stmt(self, stmt):
        # 访问循环变量
        loop_var = stmt.loop_var
        print(f"Loop variable: {loop_var.name_hint}")
        
        # 访问边界
        start = stmt.start
        stop = stmt.stop
        step = stmt.step
        
        # 访问 iter_args
        for ia in stmt.iter_args:
            print(f"IterArg: {ia.name_hint}, init={ia.initValue}")
        
        # 访问循环体
        self.visit_stmt(stmt.body)
        
        # 访问 return_vars
        for rv in stmt.return_vars:
            print(f"Return var: {rv.name_hint}")

visitor = LoopVisitor()
visitor.visit_function(func)
```

---

## 7. 常见遍历模式

### 7.1 模式 A: 基类委托

```cpp
StmtPtr VisitStmt_(const ForStmtPtr& op) override {
  if (op->kind_ != ForKind::Unroll) {
    return IRMutator::VisitStmt_(op);  // 委托基类处理
  }
  return UnrollForStmt(op);  // 自定义处理展开循环
}
```

### 7.2 模式 B: 变量替换保存/恢复

```cpp
// 处理循环前保存之前的替换
auto prev_loop_sub = SaveSubstitution(loop_var_key);
std::vector<SavedSubstitution> prev_ia_subs;
for (const auto& ia : op->iter_args_) {
  prev_ia_subs.push_back(SaveSubstitution(ia.get()));
}

// 处理循环体时应用替换
substitution_map_[loop_var_key] = substitution_expr;
auto inner_body = VisitStmt(op->body_);

// 处理完后恢复替换
RestoreSubstitution(prev_loop_sub);
```

### 7.3 模式 C: 绑定循环变量范围用于分析

```cpp
StmtPtr VisitStmt_(const ForStmtPtr& op) override {
  // 在绑定前简化边界
  auto new_start = analyzer_->Simplify(op->start_);
  auto new_stop = analyzer_->Simplify(op->stop_);

  // 当边界为常量时绑定循环变量范围
  auto start_ci = As<ConstInt>(new_start);
  auto stop_ci = As<ConstInt>(new_stop);
  if (start_ci && stop_ci) {
    analyzer_->Bind(op->loop_var_, start_ci->value_, stop_ci->value_);
  }

  // 绑定生效时访问循环体
  auto new_body = VisitStmt(op->body_);

  // 循环后解除绑定
  if (bound) analyzer_->Unbind(op->loop_var_);

  return changed ? std::make_shared<ForStmt>(...) : op;
}
```

### 7.4 模式 D: 收集循环定义用于符号表

```cpp
void VisitStmt_(const ForStmtPtr& op) override {
  // 注册 loop_var 为定义
  var_types[op->loop_var_.get()] = op->loop_var_->GetType();
  
  // 注册 iter_args 为定义 (作用域限于循环体)
  for (const auto& ia : op->iter_args_) {
    var_types[ia.get()] = ia->GetType();
  }
  
  // 注册 return_vars 为定义 (循环外可访问)
  for (const auto& rv : op->return_vars_) {
    var_types[rv.get()] = rv->GetType();
  }
  
  IRVisitor::VisitStmt_(op);  // 递归访问循环体
}
```

---

## 8. 组件访问总结

| 组件 | 访问路径 | 类型 | 说明 |
|------|----------|------|------|
| 循环变量 | `op->loop_var_` | `VarPtr` | 迭代变量 (如 `i`) |
| 起始值 | `op->start_` | `ExprPtr` | 通常为 `ConstInt` |
| 终止值 | `op->stop_` | `ExprPtr` | 可为常量或动态变量 |
| 步长 | `op->step_` | `ExprPtr` | 通常为 `ConstInt(1)` |
| 循环携带变量 | `op->iter_args_` | `vector<IterArgPtr>` | SSA 状态 |
| 初始值 | `ia->initValue_` | `ExprPtr` | 首次迭代的值 |
| 循环体 | `op->body_` | `StmtPtr` | 通常为 `SeqStmts` |
| 输出变量 | `op->return_vars_` | `vector<VarPtr>` | 循环外可访问的最终值 |
| 循环类型 | `op->kind_` | `ForKind` | Sequential/Parallel/Unroll |
| 分块大小 | `op->chunk_size_` | `optional<ExprPtr>` | 分块循环 |
| 循环来源 | `op->loop_origin_` | `LoopOrigin` | Original/ChunkOuter 等 |

---

## 9. 相关工具文件

| 文件 | 用途 |
|------|------|
| `include/pypto/ir/transforms/utils/var_collectors.h` | 收集循环中变量定义/使用 |
| `include/pypto/ir/transforms/utils/loop_state_repair.h` | 用新 iter_args/return_vars 重建循环 |
| `include/pypto/ir/transforms/utils/transform_utils.h` | 替换变量、查找 yield、展平语句 |
| `include/pypto/ir/arith/ir_mutator_with_analyzer.h` | 循环内范围感知简化 |

---

## 10. 最佳实践

1. **IterArg 重映射顺序**: 变换 iter_args 时，在访问 body 前注册 old→new 映射，在访问 return_vars 前清理映射。

2. **写时复制**: 未改变时返回原指针，保持对象标识用于高效比较。

3. **作用域管理**: 循环遍历时应用临时替换需使用保存/恢复模式。

4. **分析器绑定**: 使用 `IRMutatorWithAnalyzer` 时，绑定前简化边界，绑定后访问循环体，结束后解除绑定。

5. **Yield 语句处理**: 有 iter_args 的循环体必须以 `YieldStmt` 结束。使用 `transform_utils::GetLastYieldStmt(body)` 查找。

6. **定义顺序**: 收集定义时使用约定顺序: `loop_var → return_vars → iter_args → body`。

---

## 11. pl.create_tensor 作为 init_values：数学语义分析

### 11.1 数学符号约定

为便于理解 PyPTO 的 IR 表达，引入以下数学符号：

| PyPTO 符号 | 数学符号 | 说明 |
|------------|----------|------|
| `pl.Tensor[[M, N], T]` | $\mathbb{R}^{M \times N}$ | M×N 维实数张量 |
| `pl.Scalar[T]` | $s \in \mathbb{R}$ | 标量 |
| `pl.create_tensor([M, N])` | $\mathbf{0}_{M \times N}$ | 零初始化张量 |
| `pl.full([M, N], v)` | $\mathbf{v}_{M \times N}$ | 常量填充张量 |
| `pl.add(A, B)` | $A + B$ | 张量逐元素加法 |
| `pl.mul(A, B)` | $A \odot B$ | 张量逐元素乘法 |
| `pl.adds(A, s)` | $A + s$ | 张量加标量 |
| `IterArg("acc", init)` | $acc^{(0)} = init$ | 循环携带变量的初始状态 |
| `YieldStmt([new])` | $acc^{(k+1)} = new$ | 迭代间状态传递 |
| `return_vars` | $acc^{(final)}$ | 循环结束后的最终状态 |

### 11.2 循环的数学语义

**PyPTO 循环的数学定义**:

对于带有 `iter_args` 的循环：

$$
\begin{aligned}
acc^{(0)} &= init \quad &\text{(IterArg.initValue)} \\
acc^{(k)} &= f(acc^{(k-1)}, i_k, \text{其他参数}) \quad &\text{(循环体计算)} \\
result &= acc^{(N)} \quad &\text{(return_vars, N 为迭代次数)}
\end{aligned}
$$

其中 $f$ 是循环体定义的计算函数。

### 11.3 基本示例：Tensor 累加

**Python DSL**:

```python
@pl.function
def tensor_accum_loop(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
    init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
    for i, (acc,) in pl.range(4, init_values=(init,)):
        temp: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
        result = pl.yield_(temp)
    return result
```

**数学表达**:

$$
\begin{aligned}
init &= \mathbf{0}_{64 \times 128} \quad &\text{(零初始化张量)} \\
acc^{(0)} &= init = \mathbf{0}_{64 \times 128} \\
acc^{(k)} &= acc^{(k-1)} + x \quad &\text{(第 k 次迭代)} \\
result &= acc^{(4)} = \sum_{k=0}^{3} x = 4x
\end{aligned}
$$

**等价数学公式**:

$$result = \mathbf{0}_{64 \times 128} + x + x + x + x = 4x$$

**IR 与数学的对应**:

```
┌──────────────────────────────────────────────────────────────────┐
│  IR 组件                    │  数学对应                          │
├──────────────────────────────────────────────────────────────────┤
│  init = pl.create_tensor()  │  init = 𝟎                          │
│  IterArg.initValue = init   │  acc^(0) = init                    │
│  temp = pl.add(acc, x)      │  temp^(k) = acc^(k) + x            │
│  YieldStmt([temp])          │  acc^(k+1) = temp^(k)              │
│  return_var = result        │  result = acc^(N)                  │
└──────────────────────────────────────────────────────────────────┘
```

### 11.4 多 iter_args 示例：并行累加与乘法

**Python DSL**:

```python
@pl.function
def multi_tensor_iter(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    init1: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
    init2: pl.Tensor[[64], pl.FP32] = pl.full([64], 1.0)  # 初始化为 1.0

    for i, (acc1, acc2) in pl.range(5, init_values=(init1, init2)):
        new1: pl.Tensor[[64], pl.FP32] = pl.add(acc1, x)
        new2: pl.Tensor[[64], pl.FP32] = pl.mul(acc2, 2.0)
        out1, out2 = pl.yield_(new1, new2)

    return out1
```

**数学表达**:

$$
\begin{aligned}
acc_1^{(0)} &= \mathbf{0}_{64} \\
acc_2^{(0)} &= \mathbf{1}_{64} \\
acc_1^{(k)} &= acc_1^{(k-1)} + x \\
acc_2^{(k)} &= acc_2^{(k-1)} \odot \mathbf{2}_{64} = 2 \cdot acc_2^{(k-1)} \\
out_1 &= acc_1^{(5)} = \sum_{k=0}^{4} x = 5x \\
out_2 &= acc_2^{(5)} = 2^5 \cdot \mathbf{1}_{64} = \mathbf{32}_{64}
\end{aligned}
$$

**可视化迭代过程**:

```
迭代 k │  acc_1^(k)     │  acc_2^(k)
───────┼────────────────┼────────────────
   0   │  𝟎             │  𝟏
   1   │  𝟎 + x = x     │  𝟏 · 2 = 2
   2   │  x + x = 2x    │  2 · 2 = 4
   3   │  2x + x = 3x   │  4 · 2 = 8
   4   │  3x + x = 4x   │  8 · 2 = 16
   5   │  4x + x = 5x   │  16 · 2 = 32
───────┼────────────────┼────────────────
最终   │  out_1 = 5x    │  out_2 = 32
```

### 11.5 嵌套循环的数学语义

**Python DSL**:

```python
@pl.function
def nested_tensor_loop() -> pl.Tensor[[64, 128], pl.FP32]:
    outer_init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)

    for i, (outer_acc,) in pl.range(3, init_values=(outer_init,)):
        for j, (inner_acc,) in pl.range(2, init_values=(outer_acc,)):
            temp = pl.add(inner_acc, 1.0)
            inner_out = pl.yield_(temp)
        outer_out = pl.yield_(inner_out)

    return outer_out
```

**数学表达**:

$$
\begin{aligned}
outer^{(0)} &= \mathbf{0}_{64 \times 128} \\
&\text{对于外层迭代 } i = 0, 1, 2: \\
&\quad inner^{(0)}_i = outer^{(i)} \\
&\quad \text{对于内层迭代 } j = 0, 1: \\
&\quad\quad inner^{(j+1)}_i = inner^{(j)}_i + \mathbf{1} \\
&\quad inner_out_i = inner^{(2)}_i = outer^{(i)} + \mathbf{2} \\
&\quad outer^{(i+1)} = inner_out_i \\
outer_out &= outer^{(3)} = \mathbf{0} + \mathbf{2} + \mathbf{2} + \mathbf{2} = \mathbf{6}
\end{aligned}
$$

**展开计算**:

$$
\begin{aligned}
outer^{(0)} &= \mathbf{0} \\
outer^{(1)} &= outer^{(0)} + \mathbf{2} = \mathbf{2} \\
outer^{(2)} &= outer^{(1)} + \mathbf{2} = \mathbf{4} \\
outer^{(3)} &= outer^{(2)} + \mathbf{2} = \mathbf{6}
\end{aligned}
$$

### 11.6 状态转移的数学图示

**单 iter_arg 循环的状态转移图**:

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                          │
        │   init = pl.create_tensor()                              │
        │         ↓                                                │
        │   acc^(0) = init                         ┌─────────────┐ │
        │         ↓                                │             │ │
        │   ┌─────────────────────────────────────┤  循环体 f   │ │
        │   │                                     │             │ │
        │   │  acc^(k) ────────→ f(acc^(k), i) ───┤  pl.add    │ │
        │   │                        │            │             │ │
        │   │                        ↓            └─────────────┘ │
        │   │                    temp^(k)                         │
        │   │                        │                            │
        │   │                        ↓ (yield)                    │
        │   │                    acc^(k+1)                        │
        │   │                        │                            │
        │   └───←────────────────────┘ (迭代继续)                 │
        │                                                          │
        │                        ↓ (循环结束)                      │
        │                                                          │
        │   result = acc^(N)                                       │
        │                                                          │
        └─────────────────────────────────────────────────────────┘
```

**数学递推公式**:

$$acc^{(k+1)} = f(acc^{(k)}, i_k)$$

其中：
- $k \in \{0, 1, \ldots, N-1\}$ 是迭代索引
- $acc^{(k)}$ 是第 $k$ 次迭代开始时的状态
- $f$ 是循环体定义的变换函数

### 11.7 pl.create_tensor 的数学意义

`pl.create_tensor([M, N], dtype)` 的数学语义：

$$init = \mathbf{0}_{M \times N} = \begin{bmatrix} 0 & 0 & \cdots & 0 \\ 0 & 0 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 0 \end{bmatrix}_{M \times N}$$

这是一个**零张量**，作为累加操作的初始值。

**对比其他初始化方式**:

| 初始化方式 | 数学表达 | 用途 |
|------------|----------|------|
| `pl.create_tensor([M, N])` | $\mathbf{0}_{M \times N}$ | 累加初始值 |
| `pl.full([M, N], v)` | $\mathbf{v}_{M \times N}$ | 乘法初始值（如 $\mathbf{1}$） |
| `pl.full([M, N], -∞)` | $\mathbf{-\infty}$ | max 操作初始值 |
| `pl.full([M, N], +∞)` | $\mathbf{+\infty}$ | min 操作初始值 |
| 参数 `x: pl.Tensor[[M, N]]` | $x$ | 直接使用输入 |

### 11.8 典型应用场景的数学表达

#### 矩阵累加 (Matrix Accumulation)

```python
for i, (acc,) in pl.range(N, init_values=(pl.create_tensor([M, K]),)):
    acc = pl.matmul_acc(acc, A, B)
    result = pl.yield_(acc)
```

$$
\begin{aligned}
acc^{(0)} &= \mathbf{0}_{M \times K} \\
acc^{(k)} &= acc^{(k-1)} + A \cdot B \\
result &= acc^{(N)} = N \cdot (A \cdot B)
\end{aligned}
$$

#### 滑动窗口累加 (Sliding Window)

```python
for i, (window,) in pl.range(T, init_values=(pl.create_tensor([W]),)):
    new_val = pl.read(x, i)
    window_shifted = pl.slice(window, [W-1], [1])  # 移除最旧元素
    window_new = pl.concat(window_shifted, new_val)  # 添加新元素
    window = pl.yield_(window_new)
```

$$
\begin{aligned}
window^{(0)} &= \mathbf{0}_W \\
window^{(k)} &= [window^{(k-1)}_{1:}, x_k] \quad \text{(滑动窗口)}
\end{aligned}
$$

#### 乘法累加 (Multiply-Accumulate)

```python
init_sum = pl.create_tensor([M])    # 零初始化用于加法
init_prod = pl.full([M], 1.0)       # 1 初始化用于乘法

for i, (sum_acc, prod_acc) in pl.range(N, init_values=(init_sum, init_prod)):
    new_sum = pl.add(sum_acc, pl.mul(A[i], B[i]))
    new_prod = pl.mul(prod_acc, C[i])
    sum_out, prod_out = pl.yield_(new_sum, new_prod)
```

$$
\begin{aligned}
sum^{(0)} &= \mathbf{0}_M, \quad prod^{(0)} &= \mathbf{1}_M \\
sum^{(k)} &= sum^{(k-1)} + A_k \odot B_k \\
prod^{(k)} &= prod^{(k-1)} \odot C_k \\
sum_out &= \sum_{k=0}^{N-1} A_k \odot B_k \quad \text{(点积)} \\
prod_out &= \prod_{k=0}^{N-1} C_k \quad \text{(元素乘)}
\end{aligned}
$$

### 11.9 IR 组件与数学概念的对应表

```
┌──────────────────────────────────────────────────────────────────────┐
│                       PyPTO IR 与数学语义对应                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  IR 组件                        数学概念                              │
│  ─────────────────────────────  ─────────────────────────────────    │
│                                                                       │
│  ForStmt                        递推序列 acc^(0), acc^(1), ..., acc^(N)│
│  ├─ loop_var                    迭代索引 i ∈ {0, 1, ..., N-1}         │
│  ├─ start, stop, step           循环范围 [0, N)                       │
│  ├─ iter_args                   状态变量初始值 acc^(0)                │
│  │   └─ IterArg.initValue       acc^(0) = init                       │
│  ├─ body                        状态转移函数 f                        │
│  │   └─ YieldStmt               acc^(k+1) = f(acc^(k), i_k)          │
│  ├─ return_vars                 最终状态 acc^(N)                      │
│  └─ kind                        Sequential/Parallel/Unroll           │
│                                                                       │
│  pl.create_tensor([M, N])       𝟎_{M×N} (零张量)                     │
│  pl.full([M, N], v)             𝐯_{M×N} (常量张量)                   │
│  pl.add(A, B)                   A + B (逐元素加)                      │
│  pl.mul(A, B)                   A ⊙ B (逐元素乘)                      │
│  pl.adds(A, s)                  A + s (广播加)                        │
│  pl.muls(A, s)                  A · s (广播乘)                        │
│                                                                       │
│  YieldStmt                      状态转移: acc^(k) → acc^(k+1)        │
│  return_vars                    最终输出: result = acc^(N)           │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.10 从 DSL 到数学的完整转换示例

**DSL 代码**:

```python
@pl.function
def compute(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    sum_init = pl.create_tensor([64])
    max_init = pl.full([64], -1e9)
    
    for i, (sum_acc, max_acc) in pl.range(10, init_values=(sum_init, max_init)):
        val = pl.read(x, i)
        new_sum = pl.add(sum_acc, val)
        new_max = pl.maximum(max_acc, val)
        sum_out, max_out = pl.yield_(new_sum, new_max)
    
    return sum_out, max_out
```

**数学表达**:

$$
\begin{aligned}
&\text{输入: } x \in \mathbb{R}^{10 \times 64} \\
&\text{初始化:} \\
&\quad sum^{(0)} = \mathbf{0}_{64} \\
&\quad max^{(0)} = \mathbf{-\infty}_{64} \\
&\text{迭代 (} k = 0, 1, \ldots, 9 \text{):} \\
&\quad val^{(k)} = x[k, :] \quad \text{(读取第 k 行)} \\
&\quad sum^{(k+1)} = sum^{(k)} + val^{(k)} \\
&\quad max^{(k+1)} = \max(max^{(k)}, val^{(k)}) \\
&\text{输出:} \\
&\quad sum_out = \sum_{k=0}^{9} x[k, :] \quad \text{(列求和)} \\
&\quad max_out = \max_{k=0}^{9} x[k, :] \quad \text{(列最大值)}
\end{aligned}
$$

**直观理解**:

这个函数计算一个 $10 \times 64$ 矩阵的：
- 每列的**总和** (sum_out)
- 每列的**最大值** (max_out)

### 11.11 循环不变量分析

从数学角度，循环携带变量满足特定的**不变量**：

**加法累加循环**:
$$acc^{(k)} = \sum_{j=0}^{k-1} f(j)$$

**乘法累乘循环**:
$$acc^{(k)} = \prod_{j=0}^{k-1} f(j)$$

**混合操作循环**:
$$acc^{(k)} = \mathcal{F}(acc^{(k-1)}, i_k)$$

其中 $\mathcal{F}$ 是循环体定义的变换。

### 11.12 实现细节：类型推断与数学一致性

**文件**: `python/pypto/ir/builder.py`

类型推断确保数学一致性：

```python
def iter_arg(self, name: str, init_value, ...) -> ir.IterArg:
    init_expr = _normalize_expr(init_value, span)
    inferred_type = init_expr.type  # 数学: type(init) = type(acc^(0))
    
    # 循环不变量: type(acc^(k)) = type(acc^(0)) for all k
    iter_arg = ir.IterArg(name, inferred_type, init_expr, span)
    return iter_arg
```

**数学一致性约束**:

$$\text{type}(acc^{(k)}) = \text{type}(acc^{(0)}) = \text{type}(init) \quad \forall k$$

$$\text{shape}(acc^{(k)}) = \text{shape}(init) \quad \forall k$$

### 11.13 总结：PyPTO 循环的数学本质

| PyPTO 特性 | 数学本质 |
|------------|----------|
| `IterArg` + `initValue` | 递推序列的初始状态 $acc^{(0)}$ |
| 循环体 `body` | 状态转移函数 $f$ |
| `YieldStmt` | 状态转移 $acc^{(k+1)} = f(acc^{(k)})$ |
| `return_vars` | 最终状态 $acc^{(N)}$ |
| `pl.create_tensor` | 零张量 $\mathbf{0}$（加法初始值） |
| `pl.full(v)` | 常量张量 $\mathbf{v}$（乘法初始值） |

**核心洞察**:

PyPTO 的 `iter_args` + `yield` 机制本质上是 **SSA 形式的递推序列**，通过 `YieldStmt` 显式表达状态在迭代间的传递，使编译器能够：
1. 分析循环不变量
2. 进行循环优化（展开、向量化）
3. 推断中间结果的类型和形状