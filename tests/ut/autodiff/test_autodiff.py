"""
PyPTO 自动微分测试

测试梯度规则注册和数学正确性验证
"""

import pytest
import numpy as np
import sys
import os

# =============================================================================
# 数值验证测试（不依赖 IR）
# =============================================================================


class TestGradientNumericalVerification:
    """使用数值方法验证梯度数学正确性"""

    def test_add_gradient(self):
        """
        验证 y = a + b 的梯度

        数学: dy/da = 1, dy/db = 1
        """

        def forward(a, b):
            return a + b

        eps = 1e-5
        a, b = 3.0, 5.0
        y = forward(a, b)

        grad_a = (forward(a + eps, b) - y) / eps
        grad_b = (forward(a, b + eps) - y) / eps

        np.testing.assert_almost_equal(grad_a, 1.0, decimal=4)
        np.testing.assert_almost_equal(grad_b, 1.0, decimal=4)
        print("✓ add 梯度数值验证通过")

    def test_mul_gradient(self):
        """
        验证 y = a * b 的梯度

        数学: dy/da = b, dy/db = a
        """

        def forward(a, b):
            return a * b

        eps = 1e-5
        a, b = 3.0, 5.0
        y = forward(a, b)

        grad_a = (forward(a + eps, b) - y) / eps
        grad_b = (forward(a, b + eps) - y) / eps

        np.testing.assert_almost_equal(grad_a, b, decimal=4)
        np.testing.assert_almost_equal(grad_b, a, decimal=4)
        print("✓ mul 梯度数值验证通过")

    def test_gradient_accumulation(self):
        """
        验证 y = a * b + a * c 的梯度累加

        数学: dy/da = b + c (两个贡献累加)
        """

        def forward(a, b, c):
            return a * b + a * c

        eps = 1e-5
        a, b, c = 2.0, 3.0, 4.0
        y = forward(a, b, c)

        grad_a = (forward(a + eps, b, c) - y) / eps

        # 解析梯度: b + c (累加)
        expected_grad_a = b + c

        np.testing.assert_almost_equal(grad_a, expected_grad_a, decimal=4)
        print(f"✓ 梯度累加验证通过: dy/da = {b} + {c} = {expected_grad_a}")

    def test_matmul_gradient(self):
        """
        验证矩阵乘法 C = A @ B 的梯度

        数学: dA = dC @ B^T, dB = A^T @ dC
        """
        A = np.random.randn(4, 3)
        B = np.random.randn(3, 5)
        dC = np.ones((4, 5))

        C = A @ B

        # 解析梯度
        dA_analytical = dC @ B.T
        dB_analytical = A.T @ dC

        # 数值梯度
        eps = 1e-5
        dA_numerical = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_plus = A.copy()
                A_plus[i, j] += eps
                C_plus = A_plus @ B
                dA_numerical[i, j] = np.sum((C_plus - C) * dC) / eps

        np.testing.assert_almost_equal(dA_analytical, dA_numerical, decimal=3)
        print("✓ matmul 梯度数值验证通过")

    def test_loop_gradient(self):
        """
        验证循环 acc = Σ x 的梯度

        数学: dacc/dx = N
        """

        def forward(x, N):
            acc = 0.0
            for i in range(N):
                acc += x
            return acc

        eps = 1e-5
        x = 5.0
        N = 10

        grad_x = (forward(x + eps, N) - forward(x, N)) / eps

        np.testing.assert_almost_equal(grad_x, N, decimal=4)
        print(f"✓ 循环梯度验证通过: dacc/dx = {N}")


# =============================================================================
# 梯度规则注册测试
# =============================================================================


class TestGradientRegistry:
    """测试梯度规则注册"""

    def test_registry_exists(self):
        """测试 GradientRegistry 类存在"""
        from pypto.autodiff import GradientRegistry

        assert GradientRegistry is not None
        print("✓ GradientRegistry 类导入成功")

    def test_builtin_ops_registered(self):
        """测试内置算子已注册"""
        from pypto.autodiff import GradientRegistry

        registered = GradientRegistry.list_all()

        print(f"\n已注册梯度规则数量: {len(registered)}")

        # 验证基本算子已注册
        basic_ops = ["tile.add", "tile.mul", "tile.matmul"]
        for op in basic_ops:
            assert GradientRegistry.has(op), f"{op} 应已注册"
            print(f"  ✓ {op} 已注册")

    def test_get_gradient_rule(self):
        """测试获取梯度规则"""
        from pypto.autodiff import GradientRegistry

        add_grad = GradientRegistry.get("tile.add")

        assert add_grad is not None
        assert callable(add_grad)
        print(f"✓ 成功获取 tile.add 梯度规则: {add_grad.__name__}")

    def test_register_custom_gradient(self):
        """测试注册自定义梯度"""
        from pypto.autodiff import GradientRegistry

        def my_grad(saved, d_out, kw):
            return [d_out]

        GradientRegistry.register("test.custom", my_grad, "test")

        assert GradientRegistry.has("test.custom")
        retrieved = GradientRegistry.get("test.custom")
        assert retrieved == my_grad

        # 清理
        GradientRegistry.unregister("test.custom")
        print("✓ 自定义梯度注册测试通过")


# =============================================================================
# IR 基础结构测试
# =============================================================================


class TestIRBasicStructure:
    """测试 IR 基础结构"""

    def test_import_pypto(self):
        """测试导入 pypto"""
        import pypto.language as pl
        from pypto import ir, DataType

        assert pl is not None
        assert ir is not None
        print("✓ pypto 导入成功")

    def test_create_const_expr(self):
        """测试创建常量表达式"""
        from pypto import ir, DataType

        span = ir.Span("test.py", 1, 1, 1, 10)
        const_int = ir.ConstInt(10, DataType.INT64, span)
        const_float = ir.ConstFloat(3.14, DataType.FP32, span)

        assert isinstance(const_int, ir.ConstInt)
        assert isinstance(const_float, ir.ConstFloat)
        print("✓ 常量表达式创建成功")

    def test_create_var(self):
        """测试创建变量"""
        from pypto import ir, DataType

        span = ir.Span("test.py", 1, 1, 1, 10)
        var = ir.Var("x", ir.ScalarType(DataType.FP32), span)

        assert isinstance(var, ir.Var)
        assert var.name_hint == "x"
        print("✓ 变量创建成功")

    def test_create_call(self):
        """测试创建 Call 表达式"""
        from pypto import ir, DataType

        span = ir.Span("test.py", 1, 1, 1, 10)

        a = ir.Var("a", ir.ScalarType(DataType.FP32), span)
        b = ir.Var("b", ir.ScalarType(DataType.FP32), span)

        call = ir.Call(ir.Op("add"), [a, b], {}, ir.ScalarType(DataType.FP32), span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "add"
        assert len(call.args) == 2
        print("✓ Call 表达式创建成功")


# =============================================================================
# 运行所有测试
# =============================================================================


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("PyPTO 自动微分测试")
    print("=" * 70)

    # 数值验证
    print("\n[1] 梯度数值验证")
    numerical = TestGradientNumericalVerification()
    numerical.test_add_gradient()
    numerical.test_mul_gradient()
    numerical.test_gradient_accumulation()
    numerical.test_matmul_gradient()
    numerical.test_loop_gradient()

    # 注册测试
    print("\n[2] 梯度规则注册")
    registry = TestGradientRegistry()
    registry.test_registry_exists()
    registry.test_builtin_ops_registered()
    registry.test_get_gradient_rule()
    registry.test_register_custom_gradient()

    # IR 结构
    print("\n[3] IR 基础结构")
    ir_test = TestIRBasicStructure()
    ir_test.test_import_pypto()
    ir_test.test_create_const_expr()
    ir_test.test_create_var()
    ir_test.test_create_call()

    print("\n" + "=" * 70)
    print("所有测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
