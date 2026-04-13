#!/usr/bin/env python3
"""
PyPTO pl.range() 完整示例：从 Python 代码到 IR 生成

这个例子展示了：
1. 如何使用 pl.range() 创建循环
2. 如何从 Python 代码生成 IR
3. 如何打印和分析生成的 IR

运行：python examples/range_example.py
"""

import pypto.language as pl
from pypto.pypto_core import ir


def example_simple_range():
    """示例1：简单的 pl.range() 循环"""
    print("=" * 80)
    print("示例1：简单的 pl.range() 循环")
    print("=" * 80)

    @pl.function
    def simple_loop() -> pl.Tensor[[1], pl.INT32]:
        init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

        for i, (sum_val,) in pl.range(10, init_values=(init,)):
            new_sum: pl.Tensor[[1], pl.INT32] = pl.add(sum_val, i)
            result = pl.yield_(new_sum)

        return result

    print("\n生成的 IR:")
    print("-" * 80)
    print(simple_loop.as_python())
    print("-" * 80)

    # 分析 IR 结构
    print("\nIR 结构分析:")
    print(f"  函数名: {simple_loop.name}")
    print(f"  参数数量: {len(simple_loop.params)}")
    print(f"  返回类型: {simple_loop.return_type}")

    # 查找 ForStmt
    def find_for_stmt(stmt):
        if isinstance(stmt, ir.ForStmt):
            return stmt
        if hasattr(stmt, 'body'):
            result = find_for_stmt(stmt.body)
            if result:
                return result
        if hasattr(stmt, 'stmts'):
            for s in stmt.stmts:
                result = find_for_stmt(s)
                if result:
                    return result
        return None

    for_stmt = find_for_stmt(simple_loop.body)
    if for_stmt:
        print(f"\n  ForStmt 分析:")
        print(f"    循环变量: {for_stmt.loop_var.name}")
        print(f"    循环类型: {for_stmt.kind}")
        print(f"    起始值: {for_stmt.start}")
        print(f"    结束值: {for_stmt.stop}")
        print(f"    步长: {for_stmt.step}")
        print(f"    迭代参数数量: {len(for_stmt.iter_args)}")
        print(f"    返回变量数量: {len(for_stmt.return_vars)}")


def example_range_with_params():
    """示例2：带参数的 pl.range(start, stop, step)"""
    print("\n" + "=" * 80)
    print("示例2：带参数的 pl.range(start, stop, step)")
    print("=" * 80)

    @pl.function
    def range_params() -> pl.Tensor[[1], pl.INT32]:
        init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

        for i, (acc,) in pl.range(0, 10, 2, init_values=(init,)):
            new_acc: pl.Tensor[[1], pl.INT32] = pl.add(acc, i)
            result = pl.yield_(new_acc)

        return result

    print("\n生成的 IR:")
    print("-" * 80)
    print(range_params.as_python())
    print("-" * 80)


def example_nested_range():
    """示例3：嵌套的 pl.range() 循环"""
    print("\n" + "=" * 80)
    print("示例3：嵌套的 pl.range() 循环")
    print("=" * 80)

    @pl.function
    def nested_loops() -> pl.Tensor[[1], pl.INT32]:
        init: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

        for i, (outer,) in pl.range(3, init_values=(init,)):
            for j, (inner,) in pl.range(2, init_values=(outer,)):
                new_inner: pl.Tensor[[1], pl.INT32] = pl.add(inner, 1)
                inner_out = pl.yield_(new_inner)

            outer_out = pl.yield_(inner_out)

        return outer_out

    print("\n生成的 IR:")
    print("-" * 80)
    print(nested_loops.as_python())
    print("-" * 80)


def example_multiple_iter_args():
    """示例4：多个迭代参数的 pl.range()"""
    print("\n" + "=" * 80)
    print("示例4：多个迭代参数的 pl.range()")
    print("=" * 80)

    @pl.function
    def multi_iter() -> pl.Tensor[[1], pl.INT32]:
        init1: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)
        init2: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

        for i, (val1, val2) in pl.range(5, init_values=(init1, init2)):
            new1: pl.Tensor[[1], pl.INT32] = pl.add(val1, i)
            new2: pl.Tensor[[1], pl.INT32] = pl.mul(val2, 2)
            out1, out2 = pl.yield_(new1, new2)

        return out1

    print("\n生成的 IR:")
    print("-" * 80)
    print(multi_iter.as_python())
    print("-" * 80)


def example_simple_no_iter_args():
    """示例5：没有迭代参数的简单循环"""
    print("\n" + "=" * 80)
    print("示例5：没有迭代参数的简单循环")
    print("=" * 80)

    @pl.function
    def simple_no_iter() -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.create_tensor([1], dtype=pl.INT32)

        for i in pl.range(10):
            temp: pl.Tensor[[1], pl.INT32] = pl.add(result, i)
            result = temp

        return result

    print("\n生成的 IR:")
    print("-" * 80)
    print(simple_no_iter.as_python())
    print("-" * 80)


def main():
    """运行所有示例"""
    print("\n" + "🚀 PyPTO pl.range() 完整示例")
    print("从 Python 代码到 IR 生成的完整过程")
    print("=" * 80)

    try:
        example_simple_range()
        example_range_with_params()
        example_nested_range()
        example_multiple_iter_args()
        example_simple_no_iter_args()

        print("\n" + "=" * 80)
        print("✅ 所有示例运行成功！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
