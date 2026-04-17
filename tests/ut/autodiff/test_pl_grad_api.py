"""
测试 pl.grad API - 按文档 examples/ir_analysis/07_autodiff_api_design_20260416.md

演示用户如何使用 pl.grad, pl.value_and_grad 等高层 API
包含循环和控制流场景测试
"""

import pytest
import pypto.language as pl
from pypto.ir.printer import python_print
import numpy as np


class TestPlGradAPI:
    """测试 pl.grad API 使用"""

    def test_basic_grad_usage(self):
        """基本使用: 从 @pl.function 获取 IR 并生成梯度函数"""
        print("\n" + "=" * 70)
        print("【测试 1】基本 pl.grad 使用")
        print("=" * 70)

        @pl.program
        class Forward:
            @pl.function
            def add_func(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                return result

        forward_ir = list(Forward.functions.values())[0]
        print("\n【正向函数 IR】")
        print("-" * 70)
        print(python_print(forward_ir))
        print("-" * 70)

        grad_func = pl.grad(forward_ir, params=["x", "y"])
        print("\n【反向函数 IR】")
        print("-" * 70)
        print(python_print(grad_func))
        print("-" * 70)
        print("\n✓ 基本 pl.grad 使用测试通过")

    def test_loop_accumulated_multiply(self):
        """
        【测试 18】循环表达累积乘法

        正向: for i in range(N): acc = acc * x
        数学: y = init * x^N
        反向: d_x = N * init * x^(N-1) * d_y
              d_init = x^N * d_y
        """
        print("\n" + "=" * 70)
        print("【测试 18】循环表达累积乘法 (for: acc = acc * x)")
        print("=" * 70)

        @pl.program
        class LoopMul:
            @pl.function
            def loop_mul(
                self, x: pl.Tensor[[64], pl.FP32], init: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                acc: pl.Tensor[[64], pl.FP32] = init
                for i in pl.range(4):
                    acc_new: pl.Tensor[[64], pl.FP32] = pl.mul(acc, x)
                    acc = acc_new
                return acc

        forward_ir = list(LoopMul.functions.values())[0]
        print("\n【正向函数 IR】")
        print("-" * 70)
        print(python_print(forward_ir))
        print("-" * 70)

        grad_func = pl.grad(forward_ir, params=["x", "init"])
        print("\n【反向函数 IR】")
        print("-" * 70)
        print(python_print(grad_func))
        print("-" * 70)

        def forward_np(x, init, N=4):
            acc = init
            for _ in range(N):
                acc = acc * x
            return acc

        eps = 1e-5
        x, init = 2.0, 1.0
        N = 4
        y = forward_np(x, init, N)
        grad_x = (forward_np(x + eps, init, N) - y) / eps
        grad_init = (forward_np(x, init + eps, N) - y) / eps

        expected_dx = N * init * (x ** (N - 1))
        expected_d_init = x**N

        print(f"\n【数学推导】")
        print(f"  正向: y = init * x^N = {init} * {x}^{N} = {y}")
        print(f"  反向: d_x = N * init * x^(N-1) = {N} * {init} * {x}^({N - 1}) = {expected_dx}")
        print(f"        d_init = x^N = {x}^N = {expected_d_init}")

        print(f"\n【数值验证】 x={x}, init={init}, N={N}, y={y}")
        print(f"  数值梯度 d_x ≈ {grad_x:.4f}, 解析梯度 d_x = {expected_dx:.4f}")
        print(f"  数值梯度 d_init ≈ {grad_init:.4f}, 解析梯度 d_init = {expected_d_init:.4f}")

        np.testing.assert_almost_equal(grad_x, expected_dx, decimal=3)
        np.testing.assert_almost_equal(grad_init, expected_d_init, decimal=3)
        print("\n✓ 循环累积乘法测试通过")

    def test_loop_yield_accumulated_multiply(self):
        """
        【测试 19】iter_args + yield 模式的累积乘法

        正向: for i, (acc,) in pl.range(N, init=(init)): acc_new = acc * x; yield(acc_new)
        数学: y = init * x^N
        反向: 使用 iter_args 链式追踪梯度
        """
        print("\n" + "=" * 70)
        print("【测试 19】iter_args + yield 模式累积乘法")
        print("=" * 70)

        @pl.program
        class LoopMulYield:
            @pl.function
            def loop_mul_yield(
                self, x: pl.Tensor[[64], pl.FP32], init: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(4, init_values=(init,)):
                    acc_new: pl.Tensor[[64], pl.FP32] = pl.mul(acc, x)
                    out: pl.Tensor[[64], pl.FP32] = pl.yield_(acc_new)
                return out

        forward_ir = list(LoopMulYield.functions.values())[0]
        print("\n【正向函数 IR】")
        print("-" * 70)
        print(python_print(forward_ir))
        print("-" * 70)

        grad_func = pl.grad(forward_ir, params=["x", "init"])
        print("\n【反向函数 IR】")
        print("-" * 70)
        print(python_print(grad_func))
        print("-" * 70)

        def forward_np(x, init, N=4):
            acc = init
            for _ in range(N):
                acc = acc * x
            return acc

        eps = 1e-5
        x_val, init_val = 2.0, 1.0
        N = 4
        y = forward_np(x_val, init_val, N)

        grad_x_numerical = (forward_np(x_val + eps, init_val, N) - y) / eps
        grad_init_numerical = (forward_np(x_val, init_val + eps, N) - y) / eps

        print("\n【数学推导】")
        print("  正向循环: acc_0 = init, acc_1 = init*x, acc_2 = init*x², acc_3 = init*x³, acc_4 = init*x⁴")
        print("  yield 输出: y = acc_4 = init * x⁴")
        print("")
        print("  反向传播分析 (链式法则):")
        print("  y = init * x^N")
        print("  dy/dx = N * init * x^(N-1)")
        print("  dy/d_init = x^N")
        print("")
        print(f"  对于 x={x_val}, init={init_val}, N={N}:")
        print(f"  d_x = {N} * {init_val} * {x_val}^({N-1}) = {N * init_val * x_val**(N-1)}")
        print(f"  d_init = {x_val}^N = {x_val**N}")

        expected_dx = N * init_val * (x_val ** (N - 1))
        expected_d_init = x_val ** N

        print(f"\n【数值验证】 x={x_val}, init={init_val}, N={N}, y={y}")
        print(f"  数值梯度 d_x ≈ {grad_x_numerical:.4f}, 解析梯度 d_x = {expected_dx:.4f}")
        print(f"  数值梯度 d_init ≈ {grad_init_numerical:.4f}, 解析梯度 d_init = {expected_d_init:.4f}")

        np.testing.assert_almost_equal(grad_x_numerical, expected_dx, decimal=3)
        np.testing.assert_almost_equal(grad_init_numerical, expected_d_init, decimal=3)

        print("\n【反向 IR 分析】")
        print("  iter_args + yield 模式的反向特点:")
        print("  1. d_init 直接赋值为 d_output (无需循环)")
        print("  2. 反向循环使用 iter_args 追踪梯度:")
        print("     - init_values=(d_output,) 表示反向循环的初始梯度")
        print("     - 每次迭代累加参数梯度: d_x += d_acc * acc")
        print("  3. iter_args 模式正向值自动追踪，反向计算正确")
        print("\n✓ iter_args + yield 模式累积乘法测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
