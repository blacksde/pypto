# PyPTO pl.range() 完整示例：从 Python 代码到 IR 生成

这个示例展示了 PyPTO 中 `pl.range()` 循环从 Python DSL 代码到 IR 生成的完整过程。

## 运行方法

```bash
# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# 运行示例
python examples/range_example_simple.py
```

## 示例分析

### 示例1：简单的 pl.range() 循环

**Python 代码：**
```python
@pl.function
def simple_loop() -> pl.Tensor[[1], pl.INT32]:
    init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

    for i, (sum_val,) in pl.range(10, init_values=(init,)):
        new_sum: pl.Tensor[[1], pl.INT32] = pl.add(sum_val, i)
        result = pl.yield_(new_sum)

    return result
```

**生成的 IR：**
```python
@pl.function
def simple_loop() -> pl.Tensor[[1], pl.INT32]:
    init: pl.Tensor[[1], pl.INT32] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
    for i, (sum_val,) in pl.range(10, init_values=(init,)):
        new_sum: pl.Tensor[[1], pl.INDEX] = pl.tensor.adds(sum_val, i)
        result: pl.Tensor[[1], pl.INT32] = pl.yield_(new_sum)
    return result
```

**IR 结构分析：**
- 函数名: `simple_loop`
- 参数数量: 0
- ForStmt 分析:
  - 循环变量: `i`
  - 循环类型: `ForKind.Sequential`
  - 起始值: `0`
  - 结束值: `10`
  - 步长: `1`
  - 迭代参数数量: `1`
  - 返回变量数量: `1`

### 示例2：带参数的 pl.range(start, stop, step)

**Python 代码：**
```python
@pl.function
def range_params() -> pl.Tensor[[1], pl.INT32]:
    init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

    for i, (acc,) in pl.range(0, 10, 2, init_values=(init,)):
        new_acc: pl.Tensor[[1], pl.INT32] = pl.add(acc, i)
        result = pl.yield_(new_acc)

    return result
```

**生成的 IR：**
```python
@pl.function
def range_params() -> pl.Tensor[[1], pl.INT32]:
    init: pl.Tensor[[1], pl.INT32] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
    for i, (acc,) in pl.range(0, 10, 2, init_values=(init,)):
        new_acc: pl.Tensor[[1], pl.INDEX] = pl.tensor.adds(acc, i)
        result: pl.Tensor[[1], pl.INT32] = pl.yield_(new_acc)
    return result
```

**关键特点：**
- 使用 `pl.range(0, 10, 2)` 指定了起始值、结束值和步长
- IR 中明确显示了 `start=0`, `stop=10`, `step=2`

### 示例3：嵌套的 pl.range() 循环

**Python 代码：**
```python
@pl.function
def nested_loops() -> pl.Tensor[[1], pl.INT32]:
    init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

    for i, (outer,) in pl.range(3, init_values=(init,)):
        for j, (inner,) in pl.range(2, init_values=(outer,)):
            new_inner: pl.Tensor[[1], pl.INT32] = pl.add(inner, 1)
            inner_out = pl.yield_(new_inner)

        outer_out = pl.yield_(inner_out)

    return outer_out
```

**生成的 IR：**
```python
@pl.function
def nested_loops() -> pl.Tensor[[1], pl.INT32]:
    init: pl.Tensor[[1], pl.INT32] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
    for i, (outer,) in pl.range(3, init_values=(init,)):
        for j, (inner,) in pl.range(2, init_values=(outer,)):
            new_inner: pl.Tensor[[1], pl.INDEX] = pl.tensor.adds(inner, 1)
            inner_out: pl.Tensor[[1], pl.INT32] = pl.yield_(new_inner)
        outer_out: pl.Tensor[[1], pl.INT32] = pl.yield_(inner_out)
    return outer_out
```

**关键特点：**
- 外层循环的迭代参数 `outer` 作为内层循环的初始值
- 每层循环都有自己的 `pl.yield_()` 来传递状态

### 示例4：多个迭代参数的 pl.range()

**Python 代码：**
```python
@pl.function
def multi_iter() -> pl.Tensor[[1], pl.INT32]:
    init1: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)
    init2: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

    for i, (val1, val2) in pl.range(5, init_values=(init1, init2)):
        new1: pl.Tensor[[1], pl.INT32] = pl.add(val1, i)
        new2: pl.Tensor[[1], pl.INT32] = pl.mul(val2, 2)
        out1, out2 = pl.yield_(new1, new2)

    return out1
```

**生成的 IR：**
```python
@pl.function
def multi_iter() -> pl.Tensor[[1], pl.INT32]:
    init1: pl.Tensor[[1], pl.INT32] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
    init2: pl.Tensor[[1], pl.INT32] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
    for i, (val1, val2) in pl.range(5, init_values=(init1, init2)):
        new1: pl.Tensor[[1], pl.INDEX] = pl.tensor.adds(val1, i)
        new2: pl.Tensor[[1], pl.INDEX] = pl.tensor.muls(val2, 2)
        out1, out2 = pl.yield_(new1, new2)
    return out1
```

**关键特点：**
- 使用 `init_values=(init1, init2)` 传递多个初始值
- 使用 `pl.yield_(new1, new2)` 返回多个更新值
- IR 中自动处理了多个迭代参数的传递

### 示例5：没有迭代参数的简单循环

**Python 代码：**
```python
@pl.function
def simple_no_iter() -> pl.Tensor[[1], pl.INDEX]:
    result: pl.Tensor[[1], pl.INDEX] = pl.create_tensor([1], dtype=pl.INDEX)

    for i in pl.range(10):
        temp: pl.Tensor[[1], pl.INDEX] = pl.add(result, i)
        result = temp

    return result
```

**生成的 IR：**
```python
@pl.function
def simple_no_iter() -> pl.Tensor[[1], pl.INDEX]:
    result: pl.Tensor[[1], pl.INDEX] = pl.tensor.create([1], dtype=pl.INDEX, layout=pl.TensorLayout.ND)
    for i in pl.range(10):
        temp: pl.Tensor[[1], pl.INDEX] = pl.tensor.adds(result, i)
        result: pl.Tensor[[1], pl.INDEX] = temp
    return result
```

**关键特点：**
- 使用 `for i in pl.range(10)` 而不是 `for i, (val,) in pl.range(10, init_values=(...))`
- 没有使用 `pl.yield_()`，直接在循环体内修改变量
- IR 中没有 `iter_args` 和 `return_vars`

## 从 Python 代码到 IR 的生成过程

### 1. Python DSL 层

**位置：** `python/pypto/language/dsl_api.py:148-223`

**RangeIterator 类：**
```python
class RangeIterator(Generic[T]):
    def __init__(
        self,
        stop: RangeArg,
        start: RangeArg = 0,
        step: RangeArg = 1,
        init_values: tuple[Any, ...] | None = None,
        chunk: int | None = None,
        chunk_policy: str = "leading_full",
    ):
        # 存储循环参数
        self.start = start
        self.stop = stop
        self.step = step
        self.init_values = init_values or ()
        self.chunk = chunk
        self.chunk_policy = chunk_policy
```

### 2. 解析器层

**位置：** `python/pypto/language/parser/ast_parser.py:843-986`

**关键步骤：**
1. **验证循环迭代器：** `_validate_for_loop_iterator()` - 确保使用 `pl.range()`
2. **解析循环参数：** `_parse_range_call()` - 提取 start/stop/step/init_values/chunk
3. **映射循环类型：** `"range"` → `ForKind.Sequential`
4. **创建 ForStmt：** 通过 `IRBuilder.for_loop()` 创建 IR 节点

**ForKind 映射：**
```python
_ITERATOR_TO_KIND = {
    "range": ir.ForKind.Sequential,
    "parallel": ir.ForKind.Parallel,
    "unroll": ir.ForKind.Unroll,
}
```

### 3. IR 构建层

**位置：** `python/pypto/ir/builder.py:105-173, 843-1025`

**ForStmt 结构：**
```python
class ForStmt(Stmt):
    loop_var: Var              # 循环变量
    start: Expr                # 起始值表达式
    stop: Expr                 # 结束值表达式
    step: Expr                 # 步长表达式
    iter_args: list[IterArg]   # 循环携带的变量
    body: Stmt                 # 循环体
    return_vars: list[Var]     # 最终迭代值
    kind: ForKind              # 循环类型 (Sequential/Parallel/Unroll)
    chunk_size: Expr | None    # 分块大小
    chunk_policy: ChunkPolicy   # 分块策略
    loop_origin: LoopOrigin    # 循环来源
```

### 4. IR 打印层

**位置：** `python/pypto/ir/printer.py`

**ForStmt 打印：**
- 将 `ForKind.Sequential` 转换为 `pl.range()`
- 将 `ForKind.Parallel` 转换为 `pl.parallel()`
- 将 `ForKind.Unroll` 转换为 `pl.unroll()`

## 关键设计特点

### 1. 循环类型映射

- `pl.range()` → `ForKind.Sequential` (顺序循环)
- `pl.parallel()` → `ForKind.Parallel` (并行循环)
- `pl.unroll()` → `ForKind.Unroll` (展开循环)

### 2. 类型推断

- 循环变量的数据类型从范围边界推断
- 如果边界是 `INT64` 类型，循环变量也是 `INT64`
- 否则默认为 `INDEX` 类型

### 3. 循环携带状态

- 使用 `init_values` 传递初始值
- 使用 `pl.yield_()` 更新状态
- `iter_args` 和 `return_vars` 实现状态传递

### 4. 分块支持

- `chunk` 参数指定分块大小
- `chunk_policy` 控制分块分布策略
- 支持自动循环分块优化

## 总结

这个示例展示了 PyPTO 中 `pl.range()` 的完整工作流程：

1. **Python DSL:** 用户友好的循环语法
2. **解析器:** 将 Python AST 转换为 IR
3. **IR 构建:** 创建结构化的 ForStmt 节点
4. **IR 打印:** 可读的 Python 风格输出

整个过程确保了类型安全、语义正确和可优化的 IR 表示。
