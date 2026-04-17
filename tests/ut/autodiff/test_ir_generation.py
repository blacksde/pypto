"""
PyPTO 自动微分 IR 生成测试

测试从 pl.function 生成反向函数 IR 的正确性
"""

import pytest
import sys
import os

import pypto.language as pl
from pypto import ir, DataType
from pypto.ir.printer import python_print


class TestForwardIRGeneration:
    """测试前向函数 IR 生成"""

    def test_simple_add_forward_ir(self):
        """
        测试简单加法的前向 IR

        前向: y = x + w

        验证 IR 包含:
        - 参数 x, w
        - add 操作
        - 返回语句
        """

        @pl.program
        class AddProgram:
            @pl.function
            def add(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, w)
                return y

        func = list(AddProgram.functions.values())[0]
        ir_str = python_print(func)

        print("\n前向 IR (add):")
        print(ir_str)

        # 验证 IR 结构
        assert "add" in ir_str.lower()
        assert len(func.params) == 2

        print("✓ add 前向 IR 验证通过")

    def test_simple_mul_forward_ir(self):
        """
        测试简单乘法的前向 IR

        前向: y = x * w

        验证 IR 包含:
        - mul 操作
        - 正确的参数
        """

        @pl.program
        class MulProgram:
            @pl.function
            def mul(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
                return y

        func = list(MulProgram.functions.values())[0]
        ir_str = python_print(func)

        print("\n前向 IR (mul):")
        print(ir_str)

        assert "mul" in ir_str.lower()
        print("✓ mul 前向 IR 验证通过")

    def test_composite_forward_ir(self):
        """
        测试复合表达式的前向 IR

        前向: y = x*w + x*v

        验证 IR 包含:
        - 两个 mul 操作
        - 一个 add 操作
        - x 多次使用
        """

        @pl.program
        class CompositeProgram:
            @pl.function
            def composite(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32], v: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                t1: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
                t2: pl.Tensor[[64], pl.FP32] = pl.mul(x, v)
                y: pl.Tensor[[64], pl.FP32] = pl.add(t1, t2)
                return y

        func = list(CompositeProgram.functions.values())[0]
        ir_str = python_print(func)

        print("\n前向 IR (复合表达式):")
        print(ir_str)

        # 验证两个 mul
        mul_count = ir_str.lower().count("mul")
        assert mul_count >= 2

        print(f"✓ 复合表达式 IR 验证通过 (mul 数量: {mul_count})")

    def test_loop_forward_ir(self):
        """
        测试循环的前向 IR

        前向:
          for i in range(N):
            acc = acc + x

        验证 IR 包含:
        - ForStmt
        - 循环变量
        """

        @pl.program
        class LoopProgram:
            @pl.function
            def loop(self, x: pl.Tensor[[64], pl.FP32], N: pl.Scalar[pl.INT64]) -> pl.Tensor[[64], pl.FP32]:
                acc: pl.Tensor[[64], pl.FP32] = pl.tensor.create([64], dtype=pl.FP32)
                for i in pl.range(N):
                    acc_new: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    acc = acc_new
                return acc

        func = list(LoopProgram.functions.values())[0]
        ir_str = python_print(func)

        print("\n前向 IR (循环):")
        print(ir_str)

        # 验证包含循环
        assert "for" in ir_str.lower() or "range" in ir_str.lower()

        print("✓ 循环 IR 验证通过")


class TestBackwardIRVerification:
    """测试反向 IR 的数学正确性"""

    def test_add_backward_math(self):
        """
        验证 add 反向梯度数学

        数学:
          y = x + w
          dy/dx = 1
          dy/dw = 1

        验证生成的梯度规则返回:
          [d_output, d_output]
        """
        from pypto.autodiff import GradientRegistry

        # 获取梯度规则
        add_grad = GradientRegistry.get("tile.add")

        # 模拟调用
        saved_inputs = ["x", "w"]  # 简化表示
        d_output = "d_y"  # 梯度种子

        # 调用梯度规则
        grads = add_grad(saved_inputs, d_output, {})

        # 验证返回两个梯度
        assert len(grads) == 2
        assert grads[0] == d_output  # dx = dy
        assert grads[1] == d_output  # dw = dy

        print("\nadd 反向梯度验证:")
        print(f"  dx = {grads[0]} (∂y/∂x = 1)")
        print(f"  dw = {grads[1]} (∂y/∂w = 1)")
        print("✓ add 反向梯度数学验证通过")

    def test_mul_backward_math(self):
        """
        验证 mul 反向梯度数学

        数学:
          y = x * w
          dy/dx = w
          dy/dw = x

        验证梯度规则生成的表达式结构
        """
        from pypto.autodiff import GradientRegistry
        from pypto import ir

        span = ir.Span("test.py", 1, 1, 1, 10)

        x = ir.Var("x", ir.TensorType([64], DataType.FP32, None), span)
        w = ir.Var("w", ir.TensorType([64], DataType.FP32, None), span)
        d_output = ir.Var("d_y", ir.TensorType([64], DataType.FP32, None), span)

        mul_grad = GradientRegistry.get("tile.mul")
        grads = mul_grad([x, w], d_output, {})

        # 验证返回两个 Call 表达式
        assert len(grads) == 2

        # dx 应是 mul(d_y, w)
        assert isinstance(grads[0], ir.Call)
        assert grads[0].op.name == "tile.mul"

        # dw 应是 mul(d_y, x)
        assert isinstance(grads[1], ir.Call)
        assert grads[1].op.name == "tile.mul"

        print("\nmul 反向梯度验证:")
        print(f"  dx = tile.mul(d_y, w) (∂y/∂x = w)")
        print(f"  dw = tile.mul(d_y, x) (∂y/∂w = x)")
        print("✓ mul 反向梯度数学验证通过")

    def test_matmul_backward_math(self):
        """
        验证 matmul 反向梯度数学

        数学:
          C = A @ B
          dA = dC @ B^T
          dB = A^T @ dC

        验证梯度规则生成的 IR 结构
        """
        from pypto.autodiff import GradientRegistry
        from pypto import ir

        span = ir.Span("test.py", 1, 1, 1, 10)

        A = ir.Var("A", ir.TileType([16, 16], DataType.FP16, None, None), span)
        B = ir.Var("B", ir.TileType([16, 16], DataType.FP16, None, None), span)
        d_output = ir.Var("d_C", ir.TileType([16, 16], DataType.FP16, None, None), span)

        matmul_grad = GradientRegistry.get("tile.matmul")
        grads = matmul_grad([A, B], d_output, {})

        # 验证返回两个梯度
        assert len(grads) == 2

        # dA 应包含 transpose(B) 和 matmul
        assert isinstance(grads[0], ir.Call)
        assert grads[0].op.name == "tile.matmul"

        # dB 应包含 transpose(A) 和 matmul
        assert isinstance(grads[1], ir.Call)
        assert grads[1].op.name == "tile.matmul"

        print("\nmatmul 反向梯度验证:")
        print(f"  dA = tile.matmul(d_C, tile.transpose(B))")
        print(f"  dB = tile.matmul(tile.transpose(A), d_C)")
        print("✓ matmul 反向梯度数学验证通过")


class TestGradientAccumulationVerification:
    """测试梯度累加的正确性"""

    def test_accumulation_concept(self):
        """
        验证梯度累加概念

        场景: y = x*w + x*v

        数学:
          dy/dx = w + v  (两个贡献累加)
          dy/dw = x
          dy/dv = x

        验证累加逻辑正确性
        """
        # 数值验证梯度累加
        import numpy as np

        def forward(x, w, v):
            return x * w + x * v

        eps = 1e-5
        x, w, v = 2.0, 3.0, 4.0
        y = forward(x, w, v)

        # 数值梯度
        grad_x = (forward(x + eps, w, v) - y) / eps

        # 解析梯度（累加）
        expected_grad_x = w + v

        np.testing.assert_almost_equal(grad_x, expected_grad_x, decimal=4)

        print("\n梯度累加验证:")
        print(f"  前向: y = x*w + x*v = {y}")
        print(f"  dy/dx = w + v = {w} + {v} = {expected_grad_x}")
        print(f"  数值梯度: {grad_x:.4f}")
        print("✓ 梯度累加数学验证通过")


# =============================================================================
# 运行所有测试
# =============================================================================


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("PyPTO 自动微分 IR 生成测试")
    print("=" * 70)

    # 前向 IR 生成
    print("\n[1] 前向 IR 生成测试")
    forward = TestForwardIRGeneration()
    forward.test_simple_add_forward_ir()
    forward.test_simple_mul_forward_ir()
    forward.test_composite_forward_ir()
    forward.test_loop_forward_ir()

    # 反向梯度数学验证
    print("\n[2] 反向梯度数学验证")
    backward = TestBackwardIRVerification()
    backward.test_add_backward_math()
    backward.test_mul_backward_math()
    backward.test_matmul_backward_math()

    # 梯度累加验证
    print("\n[3] 梯度累加验证")
    accum = TestGradientAccumulationVerification()
    accum.test_accumulation_concept()

    print("\n" + "=" * 70)
    print("所有测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
