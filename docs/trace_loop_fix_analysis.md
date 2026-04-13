# PyPTO Trace 问题分析与修复

## 问题描述

在使用 `pl.trace()` 函数追踪包含 `pl.range()` 循环的函数时，trace 结果中没有包含循环相关的操作信息。

### 问题表现

```python
import pypto.language as pl

@pl.function
def simple_range_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl. FP32]:
    init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
    for i, (acc,) in pl.range(5, init_values=(init,)):
        new_acc: pl.Tensor[[64, 128], pl.FP32] = pl.add(acc, x)
        result = pl.yield_(new_acc)
    return result

trace_result = pl.trace(simple_range_func)
print(f'Operations: {[op.op_name for op in trace_result.operations]}')
# 输出: Operations: ['tensor.create', 'tensor.add']
# 缺少 'range' 操作！
```

## 根本原因分析

### 1. IR 结构差异

在 PyPTO 的 IR 中，`pl.range()` 循环被表示为 `ForStmt`（控制流语句），而不是 `Call`（操作调用）：

```python
# ForStmt 结构
class ForStmt(Stmt):
    loop_var: Var              # 循环变量
    start: Expr                # 起始值表达式
    stop: Expr                 # 结束值表达式
    step: Expr                 # 步长表达式
    iter_args: list[IterArg]   # 循环携带的变量
    body: Stmt                 # 循环体
    return_vars: list[Var]     # 最终迭代值
    kind: ForKind              # 循环类型 (Sequential/Parallel/Unroll)
    # ... 其他属性
```

### 2. TraceVisitor 的限制

原始的 `TraceVisitor` 类只收集 `Call` 节点，忽略了控制流语句：

```python
class TraceVisitor(IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.operations: List[TraceInfo] = []
        # ... 其他初始化
    
    def visit_call(self, op: Call) -> None:
        # 只处理 Call 节点
        trace_info = self._extract_op_info(op)
        self.operations.append(trace_info)
    
    # 没有重写 visit_for_stmt 方法！
    # 没有重写 visit_while_stmt 方法！
```

### 3. 继承行为

`TraceVisitor` 继承了 `IRVisitor`，但 `IRVisitor.visit_for_stmt()` 的默认实现只是递归访问子节点，不创建 `TraceInfo` 记录：

```python
# IRVisitor 的默认行为
def visit_for_stmt(self, op: ForStmt) -> None:
    # 默认实现：只访问子节点，不记录循环信息
    self.visit_expr(op.loop_var)
    self.visit_expr(op.start)
    self.visit_expr(op.stop)
    self.visit_expr(op.step)
    for iter_arg in op.iter_args:
        self.visit_expr(iter_arg)
    self.visit_stmt(op.body)
    for return_var in op.return_vars:
        self.visit_expr(return_var)
```

## 修复方案

### 核心修复

在 `TraceVisitor` 类中添加 `visit_for_stmt()` 和 `visit_while_stmt()` 方法，将循环语句转换为特殊的控制流操作记录：

```python
def visit_for_stmt(self, stmt) -> None:
    """Visit a for loop statement and record loop information."""
    # 提取循环类型字符串
    kind_str = str(stmt.kind).split('.')[-1] if hasattr(stmt, 'kind') else 'Sequential'
    
    # 根据类型确定操作名称
    op_name_map = {
        'Sequential': 'range',
        'Parallel': 'parallel',
        'Unroll': 'unroll',
    }
    op_name = op_name_map.get(kind_str, f'for.{kind_str.lower()}')
    
    # 提取循环边界作为参数
    args = [stmt.start, stmt.stop, stmt.step]
    
    # 提取参数类型
    arg_types = [self._get_type_info(arg) for arg in args]
    
    # 提取源位置信息
    source_location = None
    if hasattr(stmt, 'span') and stmt.span:
        span = stmt.span
        if hasattr(span, 'file_path') and hasattr(span, 'line') and hasattr(span, 'column'):
            source_location = (span.file_path, span.line, span.column)
    
    # 创建循环信息作为特殊操作
    loop_info = TraceInfo(
        op_name=op_name,
        op_type='control_flow',  # 新的操作类型
        args=args,
        kwargs={
            'iter_args': len(stmt.iter_args) if hasattr(stmt, 'iter_args') else 0,
            'return_vars': len(stmt.return_vars) if hasattr(stmt, 'return_vars') else 0,
            'kind': kind_str,
        },
        arg_types=arg_types,
        return_type=f'ForLoop[{stmt.loop_var}]',
        source_location=source_location,
        index=self.current_index,
        output_var=None,
    )
    self.operations.append(loop_info)
    self.current_index += 1
    
    # 访问循环体以收集循环内的操作
    if hasattr(stmt, 'body') and stmt.body:
        self.visit_stmt(stmt.body)

def visit_while_stmt(self, stmt) -> None:
    """Visit a while loop statement and record loop information."""
    # 类似的实现，处理 while 循环
    # ...
```

### 修复效果

修复后的 trace 结果：

```python
trace_result = pl.trace(simple_range_func)
print(f'Operations: {[op.op_name for op in trace_result.operations]}')
# 输出: Operations: ['tensor.create', 'range', 'tensor.add']
# 现在包含了 'range' 操作！
```

详细的 trace 输出：

```
Trace for function 'simple_range_func'
Type: FunctionType.Opaque
Parameters:
    x: pl.Tensor[[64, 128], pl.FP16]
Return:
    pl.Tensor[[64, 128], pl.FP32]

Operations (3 total):
[0] tensor.create
    Type: tensor
    Args:
        - 0: MakeTuple: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]]
    Kwargs:
        dtype: fp32
        layout: TensorLayout.ND
    Return: Call(tensor.create): pl.Tensor[[64, 128], pl.FP32]

[1] range
    Type: control_flow
    Args:
        - 0: ConstInt(0)
        - 1: ConstInt(5)
        - 2: ConstInt(1)
    Kwargs:
        iter_args: 1
        return_vars: 1
        kind: Sequential
    Return: ForLoop[i]

[2] tensor.add
    Type: tensor
    Args:
        - 0: acc: pl.Tensor[[64, 128], pl.FP32]
        - 1: x: pl.Tensor[[64, 128], pl.FP16]
    Return: Call(tensor.add): pl.Tensor[[64, 128], pl.FP32]
```

## 支持的循环类型

修复后的 trace 功能支持所有 PyPTO 循环类型：

### 1. `pl.range()` - 顺序循环
```python
for i, (acc,) in pl.range(10, init_values=(init,)):
    # ...
# Trace 操作: 'range' (type: 'control_flow')
```

### 2. `pl.parallel()` - 并行循环
```python
for i, (acc,) in pl.parallel(10, init_values=(init,)):
    # ...
# Trace 操作: 'parallel' (type: 'control_flow')
```

### 3. `pl.unroll()` - 展开循环
```python
for i in pl.unroll(10):
    # ...
# Trace 操作: 'unroll' (type: 'control_flow')
```

### 4. `pl.while_()` - While 循环
```python
for (x,) in pl.while_(init_values=(initial,)):
    pl.cond(x < 10)
    # ...
# Trace 操作: 'while' (type: 'control_flow')
```

## 测试验证

所有相关的测试用例都通过了：

```bash
pytest tests/ut/language/grad/test_trace_with_range.py -v
```

测试覆盖：
- ✅ 简单的 `pl.range()` 循环
- ✅ 带参数的 `pl.range(start, stop)` 循环
- ✅ 带步长的 `pl.range(start, stop, step)` 循环
- ✅ 嵌套的 `pl.range()` 循环
- ✅ 多个迭代参数的 `pl.range()` 循环
- ✅ 循环体内的操作追踪
- ✅ 循环操作的过滤
- ✅ JSON 格式输出

## 设计考虑

### 1. 操作类型分类

引入了新的操作类型 `'control_flow'`，与现有的类型区分：
- `'tensor'` - 张量操作
- `'tile'` - Tile 操作
- `'system'` - 系统操作
- `'control_flow'` - 控制流操作（新增）

### 2. 信息完整性

循环操作记录了以下信息：
- 循环边界（start, stop, step）
- 循环类型（Sequential/Parallel/Unroll）
- 迭代参数数量
- 返回变量数量
- 循环变量信息
- 源代码位置

### 3. 递归访问

修复后的实现仍然正确地递归访问循环体，确保循环内的操作也被收集：

```python
# 访问循环体以收集循环内的操作
if hasattr(stmt, 'body') and stmt.body:
    self.visit_stmt(stmt.body)
```

## 总结

### 问题根源
`TraceVisitor` 只收集 `Call` 节点，忽略了 `ForStmt` 和 `WhileStmt` 控制流语句。

### 解决方案
重写 `visit_for_stmt()` 和 `visit_while_stmt()` 方法，将循环语句转换为 `'control_flow'` 类型的特殊操作记录。

### 修复效果
- ✅ Trace 结果现在包含所有循环操作
- ✅ 支持所有 PyPTO 循环类型（range/parallel/unroll/while）
- ✅ 保持了循环体内操作的完整追踪
- ✅ 提供了循环的详细元信息
- ✅ 所有现有测试用例通过

这个修复确保了 `pl.trace()` 功能能够完整地追踪 PyPTO 函数中的所有操作，包括控制流结构，为调试、分析和优化提供了更全面的视图。
