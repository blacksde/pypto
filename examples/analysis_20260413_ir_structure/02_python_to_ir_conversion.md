# PyPTO Python 到 IR 转换分析

## 1. 转换流水线概览

PyPTO 通过多阶段流水线将 Python DSL 代码转换为中间表示（IR），涉及 **AST 解析**、**类型解析**、**表达式求值** 和 **IR 构建**。系统支持基于装饰器的解析（`@pl.function`, `@pl.program`）和基于文本的解析（`pl.parse()`, `pl.loads()`）。

```
Python DSL 代码
     │
     ▼
[装饰器: @pl.function / @pl.program]
     │  捕获源码 + 闭包变量
     ▼
[ast.parse() → Python AST]
     │
     ▼
[ASTParser.parse_function()]
     │  ├── SpanTracker (源码位置追踪)
     │  ├── ScopeManager (变量作用域管理)
     │  ├── TypeResolver (类型注解解析)
     │  └── ExprEvaluator (闭包变量求值)
     ▼
[IRBuilder (Python 包装)]
     │  上下文管理器: function, for_loop, if_stmt
     ▼
[IRBuilder (C++ 核心)]
     │  Begin/End 模式，栈式上下文
     ▼
ir.Function / ir.Program
```

---

## 2. 入口点（装饰器）

**文件**: `python/pypto/language/parser/decorator.py`

转换通过装饰器启动，装饰器会：
- 捕获调用者作用域（闭包变量）
- 使用 `inspect.getsourcelines()` 提取源码
- 使用 Python 的 `ast.parse()` 解析 AST
- 创建 `ASTParser` 实例并调用 `parse_function()`

### 2.1 关键装饰器

| 装饰器 | 功能 | 输出 |
|--------|------|------|
| `@pl.function` | 解析单个 DSL 函数 | `ir.Function` |
| `@pl.program` | 解析包含多个 `@pl.function` 方法的类 | `ir.Program` |
| `@pl.inline` | 捕获 AST 用于调用点延迟内联 | 内联 AST |

### 2.2 装饰器处理流程

```python
# decorator.py:669-698
tree = _parse_ast_tree(source_code, "function")
func_def = _find_ast_node(tree, ast.FunctionDef, f.__name__, "function")

parser = ASTParser(
    source_file,
    source_lines,
    line_offset,
    col_offset,
    strict_ssa=strict_ssa,
    closure_vars=closure_vars,
)

ir_func = parser.parse_function(func_def, func_type=type, ...)
```

---

## 3. AST 解析器

**文件**: `python/pypto/language/parser/ast_parser.py`

`ASTParser` 类是核心组件，负责：
- 遍历 Python AST 节点
- 将 DSL 构造转换为 IR 构建器调用
- 管理作用域和变量跟踪
- 处理控制流（循环、if 语句、作用域）

### 3.1 关键组件

| 组件 | 功能 |
|------|------|
| `SpanTracker` | 追踪源码位置用于错误报告 |
| `ScopeManager` | 管理变量作用域和 SSA 强制 |
| `ExprEvaluator` | 对闭包变量求值表达式 |
| `TypeResolver` | 将类型注解解析为 IR 类型 |
| `IRBuilder` | 构造 IR 节点 |

### 3.2 主要解析方法

```python
# ast_parser.py:306-382
def parse_function(self, func_def: ast.FunctionDef) -> ir.Function:
    # 进入函数作用域
    self.scope_manager.enter_scope("function")
    
    with self.builder.function(func_name, ...) as f:
        # 解析带类型注解的参数
        for arg in func_def.args.args:
            param_type, param_direction = self.type_resolver.resolve_param_type(arg.annotation)
            param_var = f.param(param_name, param_type, ...)
            self.scope_manager.define_var(param_name, param_var)
        
        # 解析返回类型
        if func_def.returns:
            return_type = self.type_resolver.resolve_type(func_def.returns)
            f.return_type(return_type)
        
        # 解析函数体语句
        for stmt in func_def.body:
            self.parse_statement(stmt)
    
    return f.get_result()
```

### 3.3 语句解析映射

| Python AST 节点 | 解析方法 | IR 输出 |
|-----------------|----------|---------|
| `ast.AnnAssign` | `parse_annotated_assignment()` | 类型化变量定义 |
| `ast.Assign` | `parse_assignment()` | 赋值和 yield 处理 |
| `ast.For` | `parse_for_loop()` | `ForStmt` (pl.range, pl.parallel, pl.unroll) |
| `ast.While` | `parse_while_loop()` | `WhileStmt` |
| `ast.If` | `parse_if_statement()` | `IfStmt` (带 phi 节点) |
| `ast.With` | `parse_with_statement()` | `ScopeStmt` (pl.incore 等) |
| `ast.Return` | `parse_return()` | `ReturnStmt` |

---

## 4. 类型解析

**文件**: `python/pypto/language/parser/type_resolver.py`

`TypeResolver` 将 Python 类型注解转换为 IR 类型。

### 4.1 支持的类型注解

| Python 注解 | IR 类型 |
|-------------|---------|
| `pl.Tensor[[shape], dtype]` | `ir.TensorType` |
| `pl.Tile[[shape], dtype]` | `ir.TileType` |
| `pl.Scalar[dtype]` | `ir.ScalarType` |
| `tuple[T1, T2, ...]` | `list[ir.Type]` (多返回值) |

### 4.2 数据类型映射

```python
# type_resolver.py:103-124
_DTYPE_MAP: dict[str, DataType] = {
    "FP16": DataType.FP16,
    "FP32": DataType.FP32,
    "BF16": DataType.BF16,
    "INT32": DataType.INT32,
    "INT64": DataType.INT64,
    "INDEX": DataType.INDEX,
    ...
}
```

### 4.3 参数方向支持

| 注解 | 方向 |
|------|------|
| `pl.InOut[Tensor]` | `ir.ParamDirection.InOut` |
| `pl.Out[Tensor]` | `ir.ParamDirection.Out` |
| `pl.In[Tensor]` | `ir.ParamDirection.In` (默认) |

---

## 5. 表达式求值

**文件**: `python/pypto/language/parser/expr_evaluator.py`

`ExprEvaluator` 从闭包变量解析 Python 表达式：
- 使用带限制内置白名单的 `eval()` 保证安全
- 将 Python 值转换为 IR 表达式（`python_value_to_ir()`）
- 处理动态维度变量（`DynVar`）

### 5.1 Python 值到 IR 表达式转换

```python
# expr_evaluator.py:119-146
def python_value_to_ir(self, value: Any, span: ir.Span) -> ir.Expr:
    if isinstance(value, bool):
        return ir.ConstBool(value, span)
    if isinstance(value, int):
        return ir.ConstInt(value, DataType.INDEX, span)
    if isinstance(value, float):
        return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
    if isinstance(value, ir.Expr):
        return value
    if isinstance(value, DynVar):
        return self.get_or_create_dynvar(value, span)
    ...
```

---

## 6. IR 构建器

### 6.1 Python 层构建器

**文件**: `python/pypto/ir/builder.py`

Python `IRBuilder` 封装 C++ `IRBuilder`，提供：
- 上下文管理器（`with` 语句）实现简洁 API
- 使用 `inspect` 模块自动捕获 span
- 类型规范化和推断

```python
# builder.py:57-102
@contextmanager
def function(self, name: str, span: ir.Span | None = None, ...) -> Iterator["FunctionBuilder"]:
    begin_span = span if span is not None else self._capture_call_span()
    self._builder.begin_function(name, begin_span, type, level, role, attrs)
    builder_obj = FunctionBuilder(self)
    try:
        yield builder_obj
    finally:
        end_span = self._capture_call_span() if span is None else span
        combined_span = self._combine_spans(self._begin_spans[ctx_id], end_span)
        result = self._builder.end_function(combined_span)
        builder_obj._result = result
```

### 6.2 C++ 层构建器

**文件**: 
- `include/pypto/ir/builder.h`
- `src/ir/builder.cpp`

C++ `IRBuilder` 提供：
- 嵌套构造的栈式上下文管理
- 函数、循环、if 语句、作用域的 Begin/End 模式
- 嵌套正确性和 SSA 约束验证

```cpp
// builder.h:64-128
class IRBuilder {
public:
  void BeginFunction(const std::string& name, const Span& span, ...);
  VarPtr FuncArg(const std::string& name, const TypePtr& type, ...);
  void ReturnType(const TypePtr& type);
  FunctionPtr EndFunction(const Span& end_span);
  
  void BeginForLoop(const VarPtr& loop_var, ...);
  void AddIterArg(const IterArgPtr& iter_arg);
  void AddReturnVar(const VarPtr& var);
  StmtPtr EndForLoop(const Span& end_span);
  ...
};
```

---

## 7. 追踪机制（用于梯度）

**文件**: `python/pypto/language/grad/trace.py`

追踪系统使用 `IRVisitor`：
- 遍历 IR 收集操作信息
- 追踪执行顺序用于梯度计算
- 支持 `pl.trace()` API 用于调试

```python
# trace.py:272-334
class TraceVisitor(IRVisitor):
    def visit_function(self, func: Function) -> None:
        self.function_name = func.name
        # 收集参数和返回类型
        for param in func.params:
            self.param_info.append(f"{param.name_hint}: {param.type}")
        # 访问函数体收集操作
        if func.body:
            self.visit_stmt(func.body)
    
    def visit_call(self, op: Call) -> None:
        trace_info = self._extract_op_info(op)
        self.operations.append(trace_info)
```

---

## 8. 关键文件总结

| 分类 | 文件 | 功能 |
|------|------|------|
| **装饰器** | `python/pypto/language/parser/decorator.py` | 入口点 (@pl.function, @pl.program) |
| **AST 解析** | `python/pypto/language/parser/ast_parser.py` | 核心 AST → IR 转换 |
| **类型解析** | `python/pypto/language/parser/type_resolver.py` | 类型注解 → IR 类型 |
| **表达式求值** | `python/pypto/language/parser/expr_evaluator.py` | 闭包变量求值 |
| **作用域管理** | `python/pypto/language/parser/scope_manager.py` | 变量作用域和 SSA 跟踪 |
| **Span 追踪** | `python/pypto/language/parser/span_tracker.py` | 源码位置跟踪 |
| **IR Builder (Python)** | `python/pypto/ir/builder.py` | Python 层 IR 构建封装 |
| **IR Builder (C++)** | `include/pypto/ir/builder.h`, `src/ir/builder.cpp` | 核心 IR 构建 |
| **绑定** | `python/bindings/modules/ir_builder.cpp` | nanobind Python 绑定 |
| **追踪** | `python/pypto/language/grad/trace.py` | 梯度相关的 IR 遍历 |
| **文本解析** | `python/pypto/language/parser/text_parser.py` | 基于字符串/文件的解析 |

---

## 9. 关键转换模式

### 9.1 类型注解 → IR 类型

```python
pl.Tensor[[64, 128], pl.FP16]
# ↓
ir.TensorType([64, 128], DataType.FP16)
```

### 9.2 赋值 → Let/Assign

```python
x: Tensor = pl.add(a, b)
# ↓
ib.let("x", ir.Call(...))
```

### 9.3 For 循环 → ForStmt

```python
for i, (sum,) in pl.range(10, init_values=(0,)):
    ...
# ↓
ForStmt with iter_args and return_vars
```

### 9.4 Yield → Phi 节点

```python
result = pl.yield_(expr)
# ↓
YieldStmt  # 用于 SSA phi 节点
```

### 9.5 If 语句 → IfStmt

```python
if cond:
    ...
else:
    ...
# ↓
IfStmt with return_vars for phi nodes
```

### 9.6 作用域 → ScopeStmt

```python
with pl.incore():
    ...
# ↓
ScopeStmt(ir.ScopeKind.InCore)
```

---

## 10. 完整转换流程图

```
Python DSL 代码
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ 装饰器: @pl.function / @pl.program                        │
│   - 捕获源码 (inspect.getsourcelines)                    │
│   - 捕获闭包变量                                          │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ AST 解析: ast.parse()                                     │
│   - Python AST 生成                                       │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ ASTParser.parse_function()                                │
│   ├── SpanTracker: 源码位置追踪                           │
│   ├── ScopeManager: 变量作用域管理                        │
│   ├── TypeResolver: 类型注解 → IR 类型                    │
│   └── ExprEvaluator: 闭包变量求值                         │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ IRBuilder (Python 封装)                                   │
│   - 上下文管理器: function, for_loop, if_stmt             │
│   - 自动 span 捕获                                        │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ IRBuilder (C++ 核心)                                      │
│   - 栈式上下文管理                                        │
│   - Begin/End 模式                                        │
│   - SSA 约束验证                                          │
└─────────────────────────────────────────────────────────┘
     │
     ▼
ir.Function / ir.Program
```