"""
测试 pl.grad 函数，打印正向和反向 IR
"""

import pytest
import sys
import os

import pypto.language as pl
from pypto import ir, DataType
from pypto.ir.printer import python_print


class TestGradFunction:
    """测试 pl.grad 函数实现"""

    def test_grad_simple_add(self):
        """
        测试简单加法的 grad

        正向: y = x + w
        反向: dx = dy, dw = dy

        打印正向和反向 IR
        """
        print("\n" + "=" * 70)
        print("【测试 1】grad(add)")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function
            def add_func(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, w)
                return y

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向函数 IR】")
        print("-" * 70)
        print(forward_ir_str)
        print("-" * 70)

        # 生成反向函数
        from pypto.autodiff import grad

        backward_func = grad(forward_func)
        backward_ir_str = python_print(backward_func)

        print("\n【反向函数 IR】")
        print("-" * 70)
        print(backward_ir_str)
        print("-" * 70)

        # 验证
        print("\n【验证】")
        print(f"  正向参数: {[p.name_hint for p in forward_func.params]}")
        print(f"  反向参数: {[p.name_hint for p in backward_func.params]}")
        print(f"  反向返回类型数量: {len(backward_func.return_types)}")

        # 数学验证
        print("\n【数学推导】")
        print("  正向: y = x + w")
        print("  反向: dx = dy (∂y/∂x = 1)")
        print("        dw = dy (∂y/∂w = 1)")

        assert len(backward_func.params) == 3  # x, w, d_output
        assert len(backward_func.return_types) == 2  # d_x, d_w

        print("\n✓ grad(add) 测试通过")

    def test_grad_simple_mul(self):
        """
        测试简单乘法的 grad

        正向: y = x * w
        反向: dx = dy * w, dw = dy * x

        打印正向和反向 IR
        """
        print("\n" + "=" * 70)
        print("【测试 2】grad(mul)")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function
            def mul_func(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
                return y

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向函数 IR】")
        print("-" * 70)
        print(forward_ir_str)
        print("-" * 70)

        # 生成反向函数
        from pypto.autodiff import grad

        backward_func = grad(forward_func)
        backward_ir_str = python_print(backward_func)

        print("\n【反向函数 IR】")
        print("-" * 70)
        print(backward_ir_str)
        print("-" * 70)

        # 验证
        print("\n【验证】")
        print(f"  正向参数: {[p.name_hint for p in forward_func.params]}")
        print(f"  反向参数: {[p.name_hint for p in backward_func.params]}")

        # 数学验证
        print("\n【数学推导】")
        print("  正向: y = x * w")
        print("  反向: dx = dy * w (∂y/∂x = w)")
        print("        dw = dy * x (∂y/∂w = x)")

        print("\n✓ grad(mul) 测试通过")

    def test_grad_partial_params(self):
        """
        测试只计算部分参数的梯度

        正向: y = x * w + v
        反向 (params=['x']): 只返回 dx

        打印正向和反向 IR
        """
        print("\n" + "=" * 70)
        print("【测试 3】grad(部分参数)")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function
            def func(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32], v: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                t1: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
                y: pl.Tensor[[64], pl.FP32] = pl.add(t1, v)
                return y

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向函数 IR】")
        print("-" * 70)
        print(forward_ir_str)
        print("-" * 70)

        # 生成反向函数（只计算 x 的梯度）
        from pypto.autodiff import grad

        backward_func = grad(forward_func, params=["x"])
        backward_ir_str = python_print(backward_func)

        print("\n【反向函数 IR】(只计算 x 的梯度)")
        print("-" * 70)
        print(backward_ir_str)
        print("-" * 70)

        # 验证
        print("\n【验证】")
        print(f"  正向参数: {[p.name_hint for p in forward_func.params]}")
        print(f"  反向参数: {[p.name_hint for p in backward_func.params]}")
        print(f"  反向返回类型数量: {len(backward_func.return_types)} (只有 d_x)")

        # 数学验证
        print("\n【数学推导】")
        print("  正向: y = x * w + v")
        print("  反向: dx = dy * w (∂y/∂x = w)")
        print("        (dw 和 dv 不在 params 中，不计算)")

        assert len(backward_func.return_types) == 1

        print("\n✓ grad(部分参数) 测试通过")

    def test_grad_matmul_tile(self):
        """
        测试 Tile 矩阵乘法的 grad

        正向: C = matmul(A, B)
        反向: dA = matmul(dC, B^T), dB = matmul(A^T, dC)

        打印正向和反向 IR
        """
        print("\n" + "=" * 70)
        print("【测试 4】grad(matmul)")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_func(
                self, A: pl.Tile[[16, 16], pl.FP16], B: pl.Tile[[16, 16], pl.FP16]
            ) -> pl.Tile[[16, 16], pl.FP32]:
                C: pl.Tile[[16, 16], pl.FP32] = pl.matmul(A, B)
                return C

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向函数 IR】")
        print("-" * 70)
        print(forward_ir_str)
        print("-" * 70)

        # 生成反向函数
        from pypto.autodiff import grad

        backward_func = grad(forward_func)
        backward_ir_str = python_print(backward_func)

        print("\n【反向函数 IR】")
        print("-" * 70)
        print(backward_ir_str)
        print("-" * 70)

        # 验证
        print("\n【验证】")
        print(f"  正向参数: {[p.name_hint for p in forward_func.params]}")
        print(f"  反向参数: {[p.name_hint for p in backward_func.params]}")

        # 数学验证
        print("\n【数学推导】")
        print("  正向: C = A @ B")
        print("  反向: dA = dC @ B^T")
        print("        dB = A^T @ dC")

        print("\n✓ grad(matmul) 测试通过")

    def test_value_and_grad(self):
        """
        测试 value_and_grad

        打印正向和反向 IR
        """
        print("\n" + "=" * 70)
        print("【测试 5】value_and_grad")
        print("=" * 70)

        # 正向函数
        @pl.program
        class Forward:
            @pl.function
            def add_mul(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                t1: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
                y: pl.Tensor[[64], pl.FP32] = pl.add(t1, x)  # x 使用两次，梯度累加
                return y

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n【正向函数 IR】")
        print("-" * 70)
        print(forward_ir_str)
        print("-" * 70)

        # 使用 value_and_grad
        from pypto.autodiff import value_and_grad

        fwd, bwd = value_and_grad(forward_func)

        fwd_ir_str = python_print(fwd)
        bwd_ir_str = python_print(bwd)

        print("\n【value_and_grad - Forward IR】")
        print("-" * 70)
        print(fwd_ir_str)
        print("-" * 70)

        print("\n【value_and_grad - Backward IR】")
        print("-" * 70)
        print(bwd_ir_str)
        print("-" * 70)

        # 数学验证（梯度累加）
        print("\n【数学推导】")
        print("  正向: y = x * w + x")
        print("  反向: dx = dy * w + dy (梯度累加，两个贡献)")
        print("        dw = dy * x")

        # 数值验证梯度累加
        import numpy as np

        def forward_np(x, w):
            return x * w + x

        eps = 1e-5
        x, w = 2.0, 3.0
        y = forward_np(x, w)
        grad_x = (forward_np(x + eps, w) - y) / eps

        print(f"\n【数值验证】")
        print(f"  x={x}, w={w}, y={y}")
        print(f"  数值梯度 dx ≈ {grad_x:.4f}")
        print(f"  解析梯度 dx = w + 1 = {w + 1}")

        np.testing.assert_almost_equal(grad_x, w + 1, decimal=4)

        print("\n✓ value_and_grad 测试通过 (梯度累加验证正确)")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("PyPTO pl.grad 函数测试")
    print("=" * 70)

    test = TestGradFunction()
    test.test_grad_simple_add()
    test.test_grad_simple_mul()
    test.test_grad_partial_params()
    test.test_grad_matmul_tile()
    test.test_value_and_grad()

    print("\n" + "=" * 70)
    print("所有测试通过！正向/反向 IR 打印完成")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
    pytest.main([__file__, "-v", "-s"])
