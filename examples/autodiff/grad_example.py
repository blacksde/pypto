"""
PyPTO pl.grad API 使用示例

按照 examples/ir_analysis/07_autodiff_api_design_20260416.md 文档设计
"""

import pypto.language as pl
from pypto.ir.printer import python_print


def demo_basic_grad():
    """
    示例 1: 基本 grad 使用

    文档示例:
        @pl.function
        def forward(x, y):
            return pl.add(x, y)

        grad_func = pl.grad(forward, params=['x', 'y'])
    """
    print("\n" + "=" * 70)
    print("示例 1: 基本 grad 使用")
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

    print("\n正向函数 IR:")
    print(python_print(forward_ir))

    grad_func = pl.grad(forward_ir, params=["x", "y"])

    print("\n反向函数 IR:")
    print(python_print(grad_func))

    print("\n数学验证:")
    print("  正向: result = x + y")
    print("  反向: d_x = d_result, d_y = d_result")


def demo_mul_grad():
    """
    示例 2: 乘法梯度

    文档示例:
        @pl.function
        def linear(x, w):
            return pl.mul(x, w)

        grad_linear = pl.grad(linear)
        # d_x = d_output * w, d_w = d_output * x
    """
    print("\n" + "=" * 70)
    print("示例 2: mul 梯度")
    print("=" * 70)

    @pl.program
    class Linear:
        @pl.function
        def linear(
            self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
            return y

    forward_ir = list(Linear.functions.values())[0]

    print("\n正向函数 IR:")
    print(python_print(forward_ir))

    grad_linear = pl.grad(forward_ir)

    print("\n反向函数 IR:")
    print(python_print(grad_linear))

    print("\n数学验证:")
    print("  正向: y = x * w")
    print("  反向: d_x = d_y * w, d_w = d_y * x")


def demo_matmul_grad():
    """
    示例 3: 矩阵乘法梯度

    文档示例:
        @pl.function
        def matmul_func(A, B):
            return pl.matmul(A, B)

        grad_matmul = pl.grad(matmul_func)
        # dA = dC @ B^T, dB = A^T @ dC
    """
    print("\n" + "=" * 70)
    print("示例 3: matmul 梯度")
    print("=" * 70)

    @pl.program
    class MatmulProg:
        @pl.function(type=pl.FunctionType.InCore)
        def matmul_func(
            self, A: pl.Tile[[16, 16], pl.FP16], B: pl.Tile[[16, 16], pl.FP16]
        ) -> pl.Tile[[16, 16], pl.FP32]:
            C: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.matmul(A, B)
            return C

    forward_ir = list(MatmulProg.functions.values())[0]

    print("\n正向函数 IR:")
    print(python_print(forward_ir))

    grad_matmul = pl.grad(forward_ir)

    print("\n反向函数 IR:")
    print(python_print(grad_matmul))

    print("\n数学验证:")
    print("  正向: C = A @ B")
    print("  反向: d_A = d_C @ B^T, d_B = A^T @ d_C")


def demo_value_and_grad():
    """
    示例 4: value_and_grad

    文档示例:
        forward_func, backward_func = pl.value_and_grad(matmul)
    """
    print("\n" + "=" * 70)
    print("示例 4: value_and_grad")
    print("=" * 70)

    @pl.program
    class MyFunc:
        @pl.function
        def my_func(
            self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
            return y

    forward_ir = list(MyFunc.functions.values())[0]

    forward_func, backward_func = pl.value_and_grad(forward_ir)

    print("\nForward IR:")
    print(python_print(forward_func))

    print("\nBackward IR:")
    print(python_print(backward_func))


def demo_gradient_accumulation():
    """
    示例 5: 梯度累加

    当变量在多个表达式中使用时，梯度会累加
    """
    print("\n" + "=" * 70)
    print("示例 5: 梯度累加")
    print("=" * 70)

    @pl.program
    class AccumGrad:
        @pl.function
        def func(
            self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            t1: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
            y: pl.Tensor[[64], pl.FP32] = pl.add(t1, x)
            return y

    forward_ir = list(AccumGrad.functions.values())[0]

    print("\n正向函数 IR (x 使用两次):")
    print(python_print(forward_ir))

    grad_func = pl.grad(forward_ir)

    print("\n反向函数 IR (梯度累加):")
    print(python_print(grad_func))

    print("\n数学验证:")
    print("  正向: y = x * w + x")
    print("  反向: d_x = d_y * w + d_y (两个贡献累加)")
    print("        d_w = d_y * x")


def demo_partial_params():
    """
    示例 6: 只计算部分参数的梯度

    文档示例:
        grad_linear_x_only = pl.grad(linear, params=['x'])
    """
    print("\n" + "=" * 70)
    print("示例 6: 部分参数梯度")
    print("=" * 70)

    @pl.program
    class Linear:
        @pl.function
        def linear(
            self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
        ) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.mul(x, w)
            return y

    forward_ir = list(Linear.functions.values())[0]

    grad_linear_x_only = pl.grad(forward_ir, params=["x"])

    print("\n反向函数 IR (只计算 x 的梯度):")
    print(python_print(grad_linear_x_only))

    print("\n数学验证:")
    print("  正向: y = x * w")
    print("  反向: 只计算 d_x = d_y * w")


def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("PyPTO pl.grad API 使用示例")
    print("按照 examples/ir_analysis/07_autodiff_api_design_20260416.md")
    print("=" * 70)

    demo_basic_grad()
    demo_mul_grad()
    demo_matmul_grad()
    demo_value_and_grad()
    demo_gradient_accumulation()
    demo_partial_params()

    print("\n" + "=" * 70)
    print("所有示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()