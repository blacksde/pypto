#!/usr/bin/env python3
"""Test trace with nested loops"""

import pypto.language as pl

# 创建一个简单的嵌套循环测试
@pl.function
def nested_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
    init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
    for i, (outer,) in pl.range(3, init_values=(init,)):
        for j, (inner,) in pl.range(2, init_values=(outer,)):
            new_inner: pl.Tensor[[64, 128], pl.FP32] = pl.add(inner, x)
            final_result: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(new_inner)
    return final_result

print('=== Simple nested loops trace ===')
trace_result = pl.trace(nested_func)
print(f'Total operations: {trace_result.count()}')
print(f'All operations: {[op.op_name for op in trace_result.operations]}')

# 详细显示所有操作
print('\n=== All operations detail ===')
for i, op in enumerate(trace_result.operations):
    print(f'[{i}] {op.op_name} ({op.op_type})')
    if op.op_type == 'control_flow':
        print(f'    Args: {op.arg_types}')
        print(f'    Kwargs: {op.kwargs}')
        print(f'    Return: {op.return_type}')
