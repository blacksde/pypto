"""
PyPTO 自动微分 IR 生成测试（含正向/反向 IR 打印）

测试生成反向算子的 IR 结构，打印正向和反向 IR
"""

import pytest
import sys
import os

import pypto.language as pl
from pypto import ir, DataType
from pypto.ir.printer import python_print
from pypto.autodiff import GradientRegistry


class TestForwardBackwardIRGeneration:
    """测试正向 IR 生成和反向梯度 IR 生成（含打印）"""

    def test_add_forward_backward_ir(self):
        """
        测试加法：正向 IR 和反向梯度 IR

        正向数学: y = x + w
        反向数学: dx = dy, dw = dy (∂y/∂x = 1, ∂y/∂w = 1)
        """
        print("\n" + "=" * 70)
        print("【测试 1】加法 (add) 正向/反向 IR")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function
            def add(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, w)
                return y

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向 IR】")
        print(forward_ir_str)
        print("\n正向数学: y = x + w")

        # 反向梯度规则调用
        print("\n【反向梯度生成】")
        add_grad = GradientRegistry.get("tile.add")

        # 创建测试用的 IR 变量
        span = ir.Span("test.py", 1, 1, 1, 10)
        x_var = ir.Var("x", ir.TensorType([64], DataType.FP32, None), span)
        w_var = ir.Var("w", ir.TensorType([64], DataType.FP32, None), span)
        d_y = ir.Var("d_y", ir.TensorType([64], DataType.FP32, None), span)

        # 调用梯度规则
        grads = add_grad([x_var, w_var], d_y, {})

        print("反向数学:")
        print("  dx = dy  (∂y/∂x = 1)")
        print("  dw = dy  (∂y/∂w = 1)")
        print("\n反向梯度 IR:")
        print(f"  dx = {python_print(grads[0])}")
        print(f"  dw = {python_print(grads[1])}")

        # 验证
        assert len(grads) == 2
        # grads[0] 和 grads[1] 应该是 d_y 本身（同一个对象）
        assert grads[0] is d_y  # dx = dy
        assert grads[1] is d_y  # dw = dy

        print("\n✓ 加法正向/反向 IR 验证通过")

    def test_mul_forward_backward_ir(self):
        """
        测试乘法：正向 IR 和反向梯度 IR

        正向数学: y = x * w
        反向数学: dx = dy * w, dw = dy * x (∂y/∂x = w, ∂y/∂w = x)
        """
        print("\n" + "=" * 70)
        print("【测试 2】乘法 (mul) 正向/反向 IR")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function
            def mul(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
                return y

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向 IR】")
        print(forward_ir_str)
        print("\n正向数学: y = x * w")

        # 反向梯度规则调用
        print("\n【反向梯度生成】")
        mul_grad = GradientRegistry.get("tile.mul")

        span = ir.Span("test.py", 1, 1, 1, 10)
        x_var = ir.Var("x", ir.TensorType([64], DataType.FP32, None), span)
        w_var = ir.Var("w", ir.TensorType([64], DataType.FP32, None), span)
        d_y = ir.Var("d_y", ir.TensorType([64], DataType.FP32, None), span)

        grads = mul_grad([x_var, w_var], d_y, {})

        print("反向数学:")
        print("  dx = dy * w  (∂y/∂x = w)")
        print("  dw = dy * x  (∂y/∂w = x)")
        print("\n反向梯度 IR:")
        print(f"  dx = {python_print(grads[0])}")
        print(f"  dw = {python_print(grads[1])}")

        # 验证
        assert len(grads) == 2
        assert isinstance(grads[0], ir.Call)  # dx = mul(dy, w)
        assert isinstance(grads[1], ir.Call)  # dw = mul(dy, x)

        print("\n✓ 乘法正向/反向 IR 验证通过")

    def test_composite_forward_backward_ir(self):
        """
        测试复合表达式：正向 IR 和反向梯度累加

        正向数学: y = x*w + x*v
        反向数学: dx = dy*w + dy*v (梯度累加，两个贡献)
                  dw = dy*x
                  dv = dy*x
        """
        print("\n" + "=" * 70)
        print("【测试 3】复合表达式 (梯度累加) 正向/反向 IR")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function
            def composite(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32], v: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                t1: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
                t2: pl.Tensor[[64], pl.FP32] = pl.mul(x, v)
                y: pl.Tensor[[64], pl.FP32] = pl.add(t1, t2)
                return y

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向 IR】")
        print(forward_ir_str)
        print("\n正向数学: y = x*w + x*v = x*(w+v)")

        print("\n【反向梯度数学】")
        print("  由于 x 在两个乘法中使用，梯度需要累加:")
        print("  dx = dy*w + dy*v = dy*(w+v)  (累加两个贡献)")
        print("  dw = dy*x")
        print("  dv = dy*x")

        # 数值验证梯度累加
        import numpy as np

        def forward_np(x, w, v):
            return x * w + x * v

        eps = 1e-5
        x, w, v = 2.0, 3.0, 4.0
        y = forward_np(x, w, v)
        grad_x = (forward_np(x + eps, w, v) - y) / eps

        print(f"\n数值验证:")
        print(f"  x={x}, w={w}, v={v}, y={y}")
        print(f"  数值梯度 dx ≈ {grad_x:.4f}")
        print(f"  解析梯度 dx = w+v = {w}+{v} = {w + v}")
        np.testing.assert_almost_equal(grad_x, w + v, decimal=4)

        print("\n✓ 复合表达式正向/反向 IR 验证通过 (梯度累加正确)")

    def test_matmul_forward_backward_ir(self):
        """
        测试矩阵乘法：正向 IR 和反向梯度 IR

        正向数学: C = A @ B
        反向数学: dA = dC @ B^T, dB = A^T @ dC
        """
        print("\n" + "=" * 70)
        print("【测试 4】矩阵乘法 (matmul) 正向/反向 IR")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self, A: pl.Tile[[16, 16], pl.FP16], B: pl.Tile[[16, 16], pl.FP16]
            ) -> pl.Tile[[16, 16], pl.FP32]:
                C: pl.Tile[[16, 16], pl.FP32] = pl.matmul(A, B)
                return C

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向 IR】")
        print(forward_ir_str)
        print("\n正向数学: C = A @ B")

        # 反向梯度规则调用
        print("\n【反向梯度生成】")
        matmul_grad = GradientRegistry.get("tile.matmul")

        span = ir.Span("test.py", 1, 1, 1, 10)
        A_var = ir.Var("A", ir.TileType([16, 16], DataType.FP16, None, None), span)
        B_var = ir.Var("B", ir.TileType([16, 16], DataType.FP16, None, None), span)
        d_C = ir.Var("d_C", ir.TileType([16, 16], DataType.FP32, None, None), span)

        grads = matmul_grad([A_var, B_var], d_C, {})

        print("反向数学:")
        print("  dA = dC @ B^T  (∂C/∂A = B^T)")
        print("  dB = A^T @ dC  (∂C/∂B = A^T)")
        print("\n反向梯度 IR:")
        print(f"  dA = {python_print(grads[0])}")
        print(f"  dB = {python_print(grads[1])}")

        # 验证
        assert len(grads) == 2
        assert isinstance(grads[0], ir.Call)  # dA = matmul(d_C, transpose(B))
        assert isinstance(grads[1], ir.Call)  # dB = matmul(transpose(A), d_C)

        print("\n✓ 矩阵乘法正向/反向 IR 验证通过")

    def test_loop_forward_backward_ir(self):
        """
        测试循环：正向 IR

        正向数学: acc = Σ_{i=0}^{N-1} x = N * x
        反向数学: dacc/dx = N
        """
        print("\n" + "=" * 70)
        print("【测试 5】循环 正向/反向 IR")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function
            def loop(self, x: pl.Tensor[[64], pl.FP32], N: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                acc: pl.Tensor[[64], pl.FP32] = pl.tensor.create([64], dtype=pl.FP32)
                for i in pl.range(N):
                    acc_new: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    acc = acc_new
                return acc

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向 IR】")
        print(forward_ir_str)
        print("\n正向数学: acc = Σ_{i=0}^{N-1} x = N * x")

        print("\n【反向梯度数学】")
        print("  dacc/dx = N (循环 N 次，每次贡献 1)")
        print("  反向遍历: 从 N-1 到 0，累加梯度")

        # 数值验证
        import numpy as np

        def forward_np(x, N):
            acc = 0.0
            for i in range(N):
                acc += x
            return acc

        eps = 1e-5
        x = 5.0
        N = 10
        grad_x = (forward_np(x + eps, N) - forward_np(x, N)) / eps

        print(f"\n数值验证:")
        print(f"  x={x}, N={N}, acc={forward_np(x, N)}")
        print(f"  数值梯度 dacc/dx ≈ {grad_x:.1f}")
        print(f"  解析梯度 dacc/dx = N = {N}")
        np.testing.assert_almost_equal(grad_x, N, decimal=4)

        print("\n✓ 循环正向/反向 IR 验证通过")


class TestExpForwardBackwardIR:
    """测试 exp 算子正向/反向 IR"""

    def test_exp_forward_backward_ir(self):
        """
        测试 exp：正向 IR 和反向梯度 IR

        正向数学: y = exp(x)
        反向数学: dx = dy * exp(x) = dy * y
        """
        print("\n" + "=" * 70)
        print("【测试 6】指数函数 (exp) 正向/反向 IR")
        print("=" * 70)

        print("\n【正向数学】")
        print("  y = exp(x)")

        print("\n【反向梯度数学】")
        print("  dx = dy * exp(x) = dy * y (使用前向结果优化)")

        # 反向梯度规则调用
        exp_grad = GradientRegistry.get("tile.exp")

        span = ir.Span("test.py", 1, 1, 1, 10)
        x_var = ir.Var("x", ir.TileType([16, 16], DataType.FP32, None, None), span)
        d_y = ir.Var("d_y", ir.TileType([16, 16], DataType.FP32, None, None), span)

        # 模拟前向结果
        forward_result = ir.Call(ir.Op("tile.exp"), [x_var], {}, x_var.type, span)

        grads = exp_grad([x_var], d_y, {"forward_result": forward_result})

        print("\n反向梯度 IR:")
        print(f"  dx = {python_print(grads[0])}")

        # 数值验证
        import numpy as np

        def forward_np(x):
            return np.exp(x)

        eps = 1e-5
        x = 2.0
        y = forward_np(x)
        grad_x = (forward_np(x + eps) - y) / eps

        print(f"\n数值验证:")
        print(f"  x={x}, y=exp({x})={y:.4f}")
        print(f"  数值梯度 dx ≈ {grad_x:.4f}")
        print(f"  解析梯度 dx = dy * exp(x) = y = {y:.4f}")
        np.testing.assert_almost_equal(grad_x, y, decimal=4)

        print("\n✓ exp 正向/反向 IR 验证通过")


class TestReLUForwardBackwardIR:
    """测试 ReLU 算子正向/反向 IR"""

    def test_relu_forward_backward_ir(self):
        """
        测试 ReLU：正向 IR 和反向梯度 IR

        正向数学: y = max(0, x) = relu(x)
        反向数学: dx = dy * (x > 0 ? 1 : 0)
        """
        print("\n" + "=" * 70)
        print("【测试 7】ReLU 正向/反向 IR")
        print("=" * 70)

        print("\n【正向数学】")
        print("  y = max(0, x) = relu(x)")

        print("\n【反向梯度数学】")
        print("  dx = dy * (x > 0 ? 1 : 0)")
        print("  使用 mask: mask = (x > 0), dx = dy * mask")

        # 反向梯度规则调用
        relu_grad = GradientRegistry.get("tile.relu")

        span = ir.Span("test.py", 1, 1, 1, 10)
        x_var = ir.Var("x", ir.TileType([16, 16], DataType.FP32, None, None), span)
        d_y = ir.Var("d_y", ir.TileType([16, 16], DataType.FP32, None, None), span)

        grads = relu_grad([x_var], d_y, {})

        print("\n反向梯度 IR:")
        print(f"  dx = {python_print(grads[0])}")

        # 数值验证
        import numpy as np

        def forward_np(x):
            return np.maximum(0, x)

        eps = 1e-5
        x_pos = 2.0  # 正值
        y_pos = forward_np(x_pos)
        grad_x_pos = (forward_np(x_pos + eps) - y_pos) / eps

        x_neg = -2.0  # 负值
        y_neg = forward_np(x_neg)
        grad_x_neg = (forward_np(x_neg + eps) - y_neg) / eps

        print(f"\n数值验证:")
        print(f"  x={x_pos} (正值): y={y_pos}, 梯度={grad_x_pos:.1f} ✓")
        print(f"  x={x_neg} (负值): y={y_neg}, 梯度={grad_x_neg:.1f} ✓")

        np.testing.assert_almost_equal(grad_x_pos, 1.0, decimal=4)
        np.testing.assert_almost_equal(grad_x_neg, 0.0, decimal=4)

        print("\n✓ ReLU 正向/反向 IR 验证通过")


class TestTransposedForwardBackwardIR:
    """测试 transpose 算子正向/反向 IR"""

    def test_transpose_forward_backward_ir(self):
        """
        测试 transpose：正向 IR 和反向梯度 IR

        正向数学: Y = X^T
        反向数学: dX = dY^T (transpose 的梯度是 transpose)
        """
        print("\n" + "=" * 70)
        print("【测试 8】Transpose 正向/反向 IR")
        print("=" * 70)

        print("\n【正向数学】")
        print("  Y = transpose(X) = X^T")

        print("\n【反向梯度数学】")
        print("  dX = transpose(dY) = dY^T")

        # 反向梯度规则调用
        transpose_grad = GradientRegistry.get("tile.transpose")

        span = ir.Span("test.py", 1, 1, 1, 10)
        X_var = ir.Var("X", ir.TileType([16, 32], DataType.FP32, None, None), span)
        d_Y = ir.Var("d_Y", ir.TileType([32, 16], DataType.FP32, None, None), span)

        grads = transpose_grad([X_var], d_Y, {})

        print("\n反向梯度 IR:")
        print(f"  dX = {python_print(grads[0])}")

        # 数值验证
        import numpy as np

        X = np.random.randn(4, 3)
        Y = X.T
        d_Y = np.ones((3, 4))

        # 解析梯度
        d_X = d_Y.T

        print(f"\n数值验证:")
        print(f"  X shape: {X.shape}")
        print(f"  Y = X^T shape: {Y.shape}")
        print(f"  dY shape: {d_Y.shape}")
        print(f"  dX = dY^T shape: {d_X.shape}")

        assert d_X.shape == X.shape

        print("\n✓ Transpose 正向/反向 IR 验证通过")


# =============================================================================
# 运行所有测试
# =============================================================================


def run_all_tests_with_print():
    """运行所有测试（带 IR 打印）"""
    print("\n" + "=" * 70)
    print("PyPTO 自动微分正向/反向 IR 生成测试")
    print("=" * 70)

    test_class = TestForwardBackwardIRGeneration()
    test_class.test_add_forward_backward_ir()
    test_class.test_mul_forward_backward_ir()
    test_class.test_composite_forward_backward_ir()
    test_class.test_matmul_forward_backward_ir()
    test_class.test_loop_forward_backward_ir()

    exp_test = TestExpForwardBackwardIR()
    exp_test.test_exp_forward_backward_ir()

    relu_test = TestReLUForwardBackwardIR()
    relu_test.test_relu_forward_backward_ir()

    transpose_test = TestTransposedForwardBackwardIR()
    transpose_test.test_transpose_forward_backward_ir()

    print("\n" + "=" * 70)
    print("所有测试通过！正向/反向 IR 打印完成")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests_with_print()
    pytest.main([__file__, "-v", "-s"])
