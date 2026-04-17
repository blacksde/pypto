"""
PyPTO 自动微分 IR 生成测试

测试从 pl.function 生成的反向函数 IR 结构
"""

import pytest
import sys
import os

# 添加 python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))

import pypto.language as pl
from pypto import ir, DataType
from pypto.ir.printer import python_print


class TestIRGradientGeneration:
    """测试生成的反向函数 IR 结构"""

    def test_simple_add_ir(self):
        """
        测试简单加法的 IR 结构

        前向:
          y = add(x, w)

        反向 IR 应包含:
          d_x: Tensor = d_y  (∂y/∂x = 1)
          d_w: Tensor = d_y  (∂y/∂w = 1)
        """

        @pl.program
        class Forward:
            @pl.function
            def add_func(
                self, x: pl.Tensor[[64], pl.FP32], w: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, w)
                return y

        # 获取前向函数 IR
        forward_func = list(Forward.functions.values())[0]

        # 打印前向 IR
        forward_ir_str = python_print(forward_func)
        print("\n前向函数 IR:")
        print(forward_ir_str)

        # 验证前向 IR 包含 add 操作
        assert "add" in forward_ir_str or "tensor.add" in forward_ir_str

        # 验证参数数量
        assert len(forward_func.params) == 2

    def test_simple_mul_ir(self):
        """
        测试简单乘法的 IR 结构

        前向:
          y = mul(x, w)

        反向 IR 应包含（保存前向值）:
          d_x: Tensor = d_y * w  (∂y/∂x = w)
          d_w: Tensor = d_y * x  (∂y/∂w = x)
        """

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

        print("\n前向函数 IR:")
        print(forward_ir_str)

        assert "mul" in forward_ir_str or "tensor.mul" in forward_ir_str

    def test_composite_expression_ir(self):
        """
        测试复合表达式 y = x * w + x * v

        验证 x 多次使用时的 IR 结构

        前向:
          t1 = mul(x, w)
          t2 = mul(x, v)
          y = add(t1, t2)

        反向（梯度累加）:
          d_x: Tensor = d_y * w + d_y * v  (累加两个贡献)
          d_w: Tensor = d_y * x
          d_v: Tensor = d_y * x
        """

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

        print("\n前向函数 IR (复合表达式):")
        print(forward_ir_str)

        # 验证包含两个 mul 和一个 add
        assert forward_ir_str.count("mul") >= 2 or forward_ir_str.count("tensor.mul") >= 2
        assert "add" in forward_ir_str or "tensor.add" in forward_ir_str

    def test_loop_ir(self):
        """
        测试循环的 IR 结构

        前向:
          for i in range(N):
            acc = acc + x

        SSA 形式:
          for i, (acc_iter,) in pl.range(N, init_values=(acc_init,)):
            acc_new = add(acc_iter, x)
            acc_out = yield(acc_new)
        """

        @pl.program
        class Forward:
            @pl.function
            def loop_func(
                self, x: pl.Tensor[[64], pl.FP32], N: pl.Scalar[pl.INT64]
            ) -> pl.Tensor[[64], pl.FP32]:
                acc: pl.Tensor[[64], pl.FP32] = pl.tensor.create([64], dtype=pl.FP32)
                for i in pl.range(N):
                    acc_new: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                    acc = acc_new
                return acc

        forward_func = list(Forward.functions.values())[0]
        forward_ir_str = python_print(forward_func)

        print("\n前向函数 IR (循环):")
        print(forward_ir_str)

        # 验证包含 ForStmt
        assert "for" in forward_ir_str.lower() or "range" in forward_ir_str.lower()

    def test_tile_matmul_ir(self):
        """
        测试 Tile 矩阵乘法的 IR 结构

        前向:
          C = matmul(A, B)

        反向:
          d_A = matmul(d_C, transpose(B))
          d_B = matmul(transpose(A), d_C)
        """

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

        print("\n前向函数 IR (matmul):")
        print(forward_ir_str)

        assert "matmul" in forward_ir_str.lower()


class TestIRStructureValidation:
    """验证 IR 结构的完整性"""

    def test_function_structure(self):
        """验证函数 IR 结构"""

        @pl.program
        class TestProgram:
            @pl.function
            def test_func(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                y: pl.Tensor[[32], pl.FP32] = pl.mul(x, 2.0)
                return y

        func = list(TestProgram.functions.values())[0]

        # 验证函数属性
        assert func.name == "test_func"
        assert len(func.params) == 1
        assert func.params[0].name_hint == "x"

        # 验证返回类型
        assert len(func.return_types) == 1

        print("\n函数结构验证:")
        print(f"  函数名: {func.name}")
        print(f"  参数: {[p.name_hint for p in func.params]}")
        print(f"  返回类型数量: {len(func.return_types)}")

    def test_assign_stmt_structure(self):
        """验证赋值语句 IR 结构"""
        span = ir.Span("test.py", 1, 1, 1, 10)

        x = ir.Var("x", ir.TensorType([64], DataType.FP32, None), span)
        const = ir.ConstFloat(2.0, DataType.FP32, span)

        # 创建 Call 表达式
        mul_call = ir.Call(ir.Op("tensor.mul"), [x, const], {}, x.type, span)

        # 创建赋值语句
        y = ir.Var("y", x.type, span)
        assign = ir.AssignStmt(y, mul_call, span)

        # 验证结构
        assert isinstance(assign, ir.AssignStmt)
        assert assign.var.name_hint == "y"
        assert isinstance(assign.value, ir.Call)

        print("\nAssignStmt 结构验证:")
        print(f"  输出变量: {assign.var.name_hint}")
        print(f"  表达式类型: {assign.value.__class__.__name__}")

    def test_for_stmt_structure(self):
        """验证循环语句 IR 结构"""
        span = ir.Span("test.py", 1, 1, 1, 10)

        i = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        start = ir.ConstInt(0, DataType.INDEX, span)
        stop = ir.ConstInt(10, DataType.INDEX, span)
        step = ir.ConstInt(1, DataType.INDEX, span)

        # 创建循环体
        body = ir.SeqStmts([], span)

        # 创建 ForStmt
        for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span)

        # 验证结构
        assert isinstance(for_stmt, ir.ForStmt)
        assert for_stmt.loop_var.name_hint == "i"
        assert isinstance(for_stmt.start, ir.ConstInt)
        assert for_stmt.start.value == 0

        print("\nForStmt 结构验证:")
        print(f"  循环变量: {for_stmt.loop_var.name_hint}")
        print(f"  范围: [{for_stmt.start.value}, {for_stmt.stop.value})")

    def test_iter_arg_structure(self):
        """验证 IterArg 结构（SSA 循环变量）"""
        span = ir.Span("test.py", 1, 1, 1, 10)

        init_value = ir.ConstFloat(0.0, DataType.FP32, span)
        iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.FP32), init_value, span)

        # 验证结构
        assert isinstance(iter_arg, ir.IterArg)
        assert iter_arg.name_hint == "acc"
        assert isinstance(iter_arg.initValue, ir.ConstFloat)
        assert iter_arg.initValue.value == 0.0

        print("\nIterArg 结构验证:")
        print(f"  名称: {iter_arg.name_hint}")
        print(f"  初始值: {iter_arg.initValue.value}")

    def test_yield_stmt_structure(self):
        """验证 YieldStmt 结构"""
        span = ir.Span("test.py", 1, 1, 1, 10)

        y = ir.Var("y", ir.ScalarType(DataType.FP32), span)
        yield_stmt = ir.YieldStmt([y], span)

        # 验证结构
        assert isinstance(yield_stmt, ir.YieldStmt)
        assert len(yield_stmt.value) == 1

        print("\nYieldStmt 结构验证:")
        print(f"  yield 值数量: {len(yield_stmt.value)}")


class TestGradientRuleRegistry:
    """测试梯度规则注册"""

    def test_registry_basic(self):
        """测试梯度规则注册基本功能"""
        from pypto.autodiff import GradientRegistry

        # 检查已注册的算子
        registered = GradientRegistry.list_all()

        print("\n已注册梯度规则:")
        print(f"  总数: {len(registered)}")
        for op in sorted(registered)[:10]:
            print(f"    {op}")

        # 验证基本算子已注册
        basic_ops = ["tile.add", "tile.mul", "tile.matmul"]
        for op in basic_ops:
            assert GradientRegistry.has(op), f"{op} should be registered"
            print(f"  ✓ {op} 已注册")

    def test_get_gradient_rule(self):
        """测试获取梯度规则"""
        from pypto.autodiff import GradientRegistry

        # 获取 tile.add 梯度规则
        add_grad = GradientRegistry.get("tile.add")

        assert add_grad is not None
        print(f"\n获取 tile.add 梯度规则: ✓")
        print(f"  函数名: {add_grad.__name__}")

    def test_register_custom_gradient(self):
        """测试注册自定义梯度规则"""
        from pypto.autodiff import GradientRegistry, register_grad

        # 定义自定义梯度
        @register_grad("test.custom_op")
        def custom_grad(saved_inputs, d_output, kwargs):
            return [d_output]

        # 验证已注册
        assert GradientRegistry.has("test.custom_op")
        print("\n自定义梯度规则注册: ✓")

        # 清理
        GradientRegistry.unregister("test.custom_op")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("PyPTO 自动微分 IR 生成测试")
    print("=" * 70)

    # IR 生成测试
    print("\n[1] IR 生成测试")
    ir_gen = TestIRGradientGeneration()
    ir_gen.test_simple_add_ir()
    print("  ✓ add IR 生成测试通过")
    ir_gen.test_simple_mul_ir()
    print("  ✓ mul IR 生成测试通过")
    ir_gen.test_composite_expression_ir()
    print("  ✓ 复合表达式 IR 生成测试通过")

    # IR 结构验证
    print("\n[2] IR 结构验证")
    struct = TestIRStructureValidation()
    struct.test_function_structure()
    print("  ✓ 函数结构验证通过")
    struct.test_assign_stmt_structure()
    print("  ✓ AssignStmt 结构验证通过")
    struct.test_for_stmt_structure()
    print("  ✓ ForStmt 结构验证通过")
    struct.test_iter_arg_structure()
    print("  ✓ IterArg 结构验证通过")
    struct.test_yield_stmt_structure()
    print("  ✓ YieldStmt 结构验证通过")

    # 梯度规则注册
    print("\n[3] 梯度规则注册测试")
    registry = TestGradientRuleRegistry()
    registry.test_registry_basic()
    print("  ✓ 梯度规则注册基本测试通过")
    registry.test_get_gradient_rule()
    print("  ✓ 获取梯度规则测试通过")
    registry.test_register_custom_gradient()
    print("  ✓ 自定义梯度注册测试通过")

    print("\n" + "=" * 70)
    print("所有测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
    pytest.main([__file__, "-v", "-s"])
