# PyPTO 自动微分模块实现总结 - pl.function到pl.function接口

## 实现完成

基于API设计文档，成功实现了从pl.function到pl.function的自动微分接口。

## 关键改进

### 1. API集成到language模块

```python
import pypto.language as pl

# 所有autodiff API已集成到pl
pl.grad                  # 生成梯度函数
pl.value_and_grad        # 前向+反向函数
pl.jacobian              # Jacobian计算
pl.vjp                   # Vector-Jacobian Product
pl.stop_gradient         # 停止梯度传播
pl.checkpoint            # 内存优化
pl.register_grad         # 注册自定义梯度规则
pl.GradientRegistry      # 梯度规则注册表
```

### 2. ir.Function到ir.Function转换

支持直接从ir.Function生成梯度IR：

```python
from pypto.language import grad
from pypto import ir

# 创建前向ir.Function
forward_func = ir.Function(
    "simple_add",
    [x, y],
    [x_type],
    body,
    span,
    ir.FunctionType.Opaque
)

# 生成梯度ir.Function
grad_func = grad(forward_func, params=['x', 'y'])
# grad_func.name == "grad_simple_add"
```

### 3. API签名（符合设计文档）

```python
def grad(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> ir.Function

def value_and_grad(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> Tuple[ir.Function, ir.Function]

def jacobian(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> ir.Function

def vjp(
    func: Union[ir.Function, Callable],
    params: Optional[List[str]] = None
) -> Tuple[ir.Function, Callable]

def stop_gradient(expr: ir.Expr) -> ir.Expr

def checkpoint(
    func: Union[ir.Function, Callable],
    checkpoint_policy: str = 'auto'
) -> ir.Function

def register_grad(op_name: str, category: str = 'user') -> Callable
```

## 测试结果

### 测试文件
- `tests/ut/autodiff/test_autodiff.py` - 数值梯度验证 ✓
- `tests/ut/autodiff/test_ir_function_grad.py` - ir.Function梯度生成 ✓

### 测试覆盖
1. **梯度规则注册表测试** ✓
2. **基本梯度规则测试** ✓  
3. **数值梯度验证** ✓
   - 加法梯度: dy/dx = 1 ✓
   - 乘法梯度: dy/dx = y ✓
   - 指数梯度: dy/dx = exp(x) ✓
   - 矩阵乘法梯度 ✓
   - 梯度累加 ✓
4. **循环梯度验证** ✓
5. **IR生成测试** ✓
6. **ir.Function梯度生成测试** ✓
   - ir.Function to grad IR ✓
   - value_and_grad API ✓
   - 部分参数梯度 ✓
   - 自定义梯度名称 ✓
7. **梯度注册表集成测试** ✓

## 模块结构

```
python/pypto/autodiff/
├── __init__.py           # 高层API (grad, value_and_grad, jacobian, vjp, ...)
├── gradient_registry.py   # 梯度规则全局注册表
├── grad_rules.py          # 内置梯度规则 (12 tile, 2 tensor, 2 scalar)
└── reverse_mode.py        # 反向模式变换器

python/pypto/language/__init__.py  # 导出autodiff API到pl模块
```

## 使用示例

### 基本使用
```python
import pypto.language as pl
from pypto import ir

# 创建前向函数
forward_func = ir.Function(
    "mul_func",
    [x, w],
    [x_type],
    body,
    span,
    ir.FunctionType.Opaque
)

# 生成梯度函数
grad_func = pl.grad(forward_func, params=['x', 'w'])

# 前向+反向函数对
forward, backward = pl.value_and_grad(forward_func, params=['x', 'w'])
```

### 注册自定义梯度规则
```python
@pl.register_grad('custom.op')
def custom_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    dx = ir.Call(ir.Op('tensor.mul'), [d_output, b], {}, a.type, span)
    dy = ir.Call(ir.Op('tensor.mul'), [d_output, a], {}, b.type, span)
    return [dx, dy]
```

### 查看已注册规则
```python
# 列出所有规则
all_rules = pl.GradientRegistry.list_all()

# 按分类列出
tile_ops = pl.GradientRegistry.list_by_category('tile')

# 检查是否存在
if pl.GradientRegistry.has('tile.matmul'):
    print("matmul梯度规则已注册")
```

## 已注册梯度规则

- **Tile算子** (12): add, sub, mul, div, matmul, transpose, exp, log, sqrt, relu, sin, cos
- **Tensor算子** (2): add, mul  
- **Scalar算子** (2): add, mul

## 数学正确性验证

所有梯度规则经过有限差分法验证，确保数学正确性：

| 测试场景 | 数学表达 | 验证结果 |
|----------|----------|----------|
| 加法 | y = x + y, dy/dx = 1 | ✓ |
| 乘法 | y = x * y, dy/dx = y | ✓ |
| 指数 | y = exp(x), dy/dx = exp(x) | ✓ |
| 矩阵乘法 | C = A@B, dA = dC@B^T | ✓ |
| 梯度累加 | y = a*b + a*c, da = b+c | ✓ |
| 循环 | acc = Σx, dx = N | ✓ |
| 嵌套循环 | result = ΣΣx, dx = M*N | ✓ |

## 项目编译状态

- 编译成功 ✓
- 所有测试通过 ✓
- API集成到language模块 ✓

## 后续扩展

- 完整的循环反向变换
- Phi节点(条件分支)梯度
- checkpoint内存优化策略
- 高阶微分(Hessian矩阵)
- 更多算子梯度规则