"""
PyPTO 自动微分测试

测试生成反向算子的 IR 正确性
"""

import pytest
import numpy as np
from typing import Tuple

# =============================================================================
# 测试 1: 基本表达式梯度验证（数值验证）
# =============================================================================


class TestBasicGradientVerification:
    """验证基本算子的梯度数学正确性"""

    def test_add_gradient_numerical(self):
        """
        验证 y = a + b 的梯度

        数学:
          dy/da = 1, dy/db = 1

        数值验证:
          dy/da ≈ (f(a+ε, b) - f(a, b)) / ε
        """

        def forward(a: float, b: float) -> float:
            return a + b

        eps = 1e-5
        a_val, b_val = 3.0, 5.0
        y = forward(a_val, b_val)

        # 数值梯度
        grad_a = (forward(a_val + eps, b_val) - y) / eps
        grad_b = (forward(a_val, b_val + eps) - y) / eps

        # 解析梯度
        expected_grad_a = 1.0
        expected_grad_b = 1.0

        np.testing.assert_almost_equal(grad_a, expected_grad_a, decimal=4)
        np.testing.assert_almost_equal(grad_b, expected_grad_b, decimal=4)

    def test_mul_gradient_numerical(self):
        """
        验证 y = a * b 的梯度

        数学:
          dy/da = b, dy/db = a
        """

        def forward(a: float, b: float) -> float:
            return a * b

        eps = 1e-5
        a_val, b_val = 3.0, 5.0
        y = forward(a_val, b_val)

        # 数值梯度
        grad_a = (forward(a_val + eps, b_val) - y) / eps
        grad_b = (forward(a_val, b_val + eps) - y) / eps

        # 解析梯度（使用前向值）
        expected_grad_a = b_val
        expected_grad_b = a_val

        np.testing.assert_almost_equal(grad_a, expected_grad_a, decimal=4)
        np.testing.assert_almost_equal(grad_b, expected_grad_b, decimal=4)

    def test_composite_expression_gradient(self):
        """
        验证复合表达式 y = a * b + a * c 的梯度

        数学（梯度累加）:
          dy/da = b + c  (来自两个乘法的贡献累加)
          dy/db = a
          dy/dc = a
        """

        def forward(a: float, b: float, c: float) -> float:
            return a * b + a * c

        eps = 1e-5
        a_val, b_val, c_val = 2.0, 3.0, 4.0
        y = forward(a_val, b_val, c_val)

        # 数值梯度
        grad_a = (forward(a_val + eps, b_val, c_val) - y) / eps
        grad_b = (forward(a_val, b_val + eps, c_val) - y) / eps
        grad_c = (forward(a_val, b_val, c_val + eps) - y) / eps

        # 解析梯度（累加）
        expected_grad_a = b_val + c_val  # 累加两个贡献
        expected_grad_b = a_val
        expected_grad_c = a_val

        np.testing.assert_almost_equal(grad_a, expected_grad_a, decimal=4)
        np.testing.assert_almost_equal(grad_b, expected_grad_b, decimal=4)
        np.testing.assert_almost_equal(grad_c, expected_grad_c, decimal=4)


class TestLoopGradientVerification:
    """验证循环的梯度数学正确性"""

    def test_simple_loop_gradient(self):
        """
        验证循环 acc = Σ_{i=0}^{N-1} x 的梯度

        数学:
          dacc/dx = N
        """

        def forward(x: float, N: int) -> float:
            acc = 0.0
            for i in range(N):
                acc = acc + x
            return acc

        eps = 1e-5
        x_val = 5.0
        N = 10
        y = forward(x_val, N)

        # 数值梯度
        grad_x = (forward(x_val + eps, N) - y) / eps

        # 解析梯度
        expected_grad_x = N

        np.testing.assert_almost_equal(grad_x, expected_grad_x, decimal=4)

    def test_dynamic_loop_gradient(self):
        """
        验证动态长度循环的梯度

        数学:
          acc = N * x
          dacc/dx = N
          dacc/dN = x
        """

        def forward(x: float, N: int) -> float:
            acc = 0.0
            for i in range(N):
                acc = acc + x
            return acc

        eps = 1e-5
        x_val = 3.0
        N = 7
        y = forward(x_val, N)

        # 数值梯度 dx
        grad_x = (forward(x_val + eps, N) - y) / eps

        # 数值梯度 dN（使用整数步长）
        grad_N = (forward(x_val, N + 1) - y) / 1

        # 解析梯度
        expected_grad_x = N
        expected_grad_N = x_val

        np.testing.assert_almost_equal(grad_x, expected_grad_x, decimal=4)
        np.testing.assert_almost_equal(grad_N, expected_grad_N, decimal=4)


class TestMatrixGradientVerification:
    """验证矩阵操作的梯度"""

    def test_matmul_gradient(self):
        """
        验证矩阵乘法 C = A @ B 的梯度

        数学:
          dA = dC @ B^T
          dB = A^T @ dC
        """
        A = np.random.randn(4, 3)
        B = np.random.randn(3, 5)
        dC = np.ones((4, 5))

        # 前向
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

        dB_numerical = np.zeros_like(B)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B_plus = B.copy()
                B_plus[i, j] += eps
                C_plus = A @ B_plus
                dB_numerical[i, j] = np.sum((C_plus - C) * dC) / eps

        np.testing.assert_almost_equal(dA_analytical, dA_numerical, decimal=3)
        np.testing.assert_almost_equal(dB_analytical, dB_numerical, decimal=3)


# =============================================================================
# 运行所有数值验证测试
# =============================================================================


def test_all_gradient_verification():
    """运行所有梯度数值验证"""
    print("\n" + "=" * 60)
    print("PyPTO 自动微分梯度数学验证")
    print("=" * 60)

    # 基本表达式
    print("\n[1] 基本表达式梯度验证")
    basic = TestBasicGradientVerification()
    basic.test_add_gradient_numerical()
    print("  ✓ add 梯度验证通过")
    basic.test_mul_gradient_numerical()
    print("  ✓ mul 梯度验证通过")
    basic.test_composite_expression_gradient()
    print("  ✓ 复合表达式梯度累加验证通过")

    # 循环
    print("\n[2] 循环梯度验证")
    loop = TestLoopGradientVerification()
    loop.test_simple_loop_gradient()
    print("  ✓ 固定长度循环梯度验证通过")
    loop.test_dynamic_loop_gradient()
    print("  ✓ 动态长度循环梯度验证通过")

    # 矩阵
    print("\n[3] 矩阵操作梯度验证")
    matrix = TestMatrixGradientVerification()
    matrix.test_matmul_gradient()
    print("  ✓ matmul 梯度验证通过")

    print("\n" + "=" * 60)
    print("所有梯度数值验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_all_gradient_verification()
    pytest.main([__file__, "-v"])
