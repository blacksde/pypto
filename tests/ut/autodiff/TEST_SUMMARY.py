"""
PyPTO 自动微分接口开发和测试总结

创建时间: 2026-04-16
环境: torch_v27

测试结果: 所有测试通过
"""

# =============================================================================
# 1. 自动微分模块结构
# =============================================================================

"""
python/pypto/autodiff/
├── __init__.py              # 高层 API，导出 GradientRegistry
├── gradient_registry.py     # 梯度规则注册表
└── grad_rules.py            # 内置梯度规则（15个）

tests/ut/autodiff/
├── test_autodiff.py         # 基础测试
└── test_ir_generation.py    # IR 生成测试
"""

# =============================================================================
# 2. 梯度规则注册结果
# =============================================================================

"""
已注册梯度规则数量: 15

Tile 算子 (12):
  tile.add, tile.sub, tile.mul, tile.div
  tile.matmul, tile.transpose
  tile.exp, tile.log, tile.sqrt
  tile.relu, tile.sin, tile.cos

Tensor 算子 (3):
  tensor.add, tensor.mul, tensor.sub
"""

# =============================================================================
# 3. 前向 IR 生成示例
# =============================================================================

"""
前向 IR (add):
@pl.function
def add(x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    y: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x, w)
    return y

前向 IR (mul):
@pl.function
def mul(x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    y: pl.Tensor[[64], pl.FP32] = pl.tensor.mul(x, w)
    return y

前向 IR (复合表达式):
@pl.function
def composite(x, w, v):
    t1: pl.Tensor[[64], pl.FP32] = pl.tensor.mul(x, w)
    t2: pl.Tensor[[64], pl.FP32] = pl.tensor.mul(x, v)
    y: pl.Tensor[[64], pl.FP32] = pl.tensor.add(t1, t2)
    return y
"""

# =============================================================================
# 4. 反向梯度数学验证
# =============================================================================

"""
add 反向梯度:
  dx = d_y (∂y/∂x = 1)
  dw = d_y (∂y/∂w = 1)

mul 反向梯度:
  dx = tile.mul(d_y, w) (∂y/∂x = w)
  dw = tile.mul(d_y, x) (∂y/∂w = x)

matmul 反向梯度:
  dA = tile.matmul(d_C, tile.transpose(B)) (∂C/∂A = B^T)
  dB = tile.matmul(tile.transpose(A), d_C) (∂C/∂B = A^T)

梯度累加验证:
  y = x*w + x*v
  dy/dx = w + v = 3.0 + 4.0 = 7.0
  数值梯度: 7.0000 ✓
"""

# =============================================================================
# 5. 使用示例
# =============================================================================

"""
# 导入自动微分模块
from pypto.autodiff import GradientRegistry, register_grad

# 查看已注册算子
registered = GradientRegistry.list_all()
print(f"已注册梯度规则: {len(registered)}")

# 获取梯度规则
add_grad = GradientRegistry.get('tile.add')
grads = add_grad(saved_inputs, d_output, kwargs)

# 注册自定义梯度
@register_grad('my.custom_op', 'user')
def my_grad(saved_inputs, d_output, kwargs):
    return [d_output]

# 检查算子是否已注册
if GradientRegistry.has('tile.matmul'):
    print("tile.matmul 已注册梯度规则")
"""

# =============================================================================
# 6. 测试结果汇总
# =============================================================================

"""
======================================================================
PyPTO 自动微分测试
======================================================================

[1] 梯度数值验证
✓ add 梯度数值验证通过
✓ mul 梯度数值验证通过
✓ 梯度累加验证通过: dy/da = 3.0 + 4.0 = 7.0
✓ matmul 梯度数值验证通过
✓ 循环梯度验证通过: dacc/dx = 10

[2] 梯度规则注册
✓ GradientRegistry 类导入成功
✓ tile.add 已注册
✓ tile.mul 已注册
✓ tile.matmul 已注册
✓ 成功获取 tile.add 梯度规则: tile_add_grad
✓ 自定义梯度注册测试通过

[3] IR 基础结构
✓ pypto 导入成功
✓ 常量表达式创建成功
✓ 变量创建成功
✓ Call 表达式创建成功

======================================================================
所有测试通过！
======================================================================
"""
