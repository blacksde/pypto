# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT WITHOUT WARRANTIES OR NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Trace reconstruction module for rebuilding IR functions from trace results.

This module provides functionality to reconstruct IR Function objects from
TraceResult objects, enabling round-trip transformation from trace to IR.

Typical usage:
    import pypto.language as pl
    from pypto.language.grad.trace_reconstruction import reconstruct_from_trace

    @pl.function
    def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
        return result

    trace_result = pl.trace(my_func)
    reconstructed_func = reconstruct_from_trace(trace_result)
"""

from typing import Any

from pypto.ir.builder import IRBuilder
from pypto.pypto_core import DataType, ir
from pypto.pypto_core.ir import Function

from .trace import TraceResult, TraceInfo


class TraceReconstructionError(RuntimeError):
    """Error raised when trace reconstruction fails."""


def reconstruct_from_trace(trace_result: TraceResult) -> Function:
    """Reconstruct an IR Function from a TraceResult.

    This function builds a new IR Function object by processing the
    operations captured in the trace result. The reconstructed function
    can be used for further compilation and execution.

    Args:
        trace_result: TraceResult object containing trace information

    Returns:
        Reconstructed IR Function object

    Raises:
        TraceReconstructionError: If reconstruction fails

    Example:
        >>> trace_result = pl.trace(my_func)
        >>> reconstructed_func = reconstruct_from_trace(trace_result)
        >>> # Use reconstructed_func like any other IR Function
    """
    try:
        builder = IRBuilder()

        with builder.function(
            trace_result.function_name,
            type=trace_result.function_type,
            level=None,
            role=None,
            attrs={},
        ) as f:
            _reconstruct_params_from_trace(f, trace_result)
            _reconstruct_return_types_from_trace(f, trace_result)
            _reconstruct_body(builder, trace_result)

        return f.get_result()

    except Exception as e:
        raise TraceReconstructionError(f"Failed to reconstruct function from trace: {e}") from e


def _extract_attrs(func: Function) -> dict[str, Any]:
    """Extract attributes from a Function object."""
    if hasattr(func, "attrs") and func.attrs:
        return dict(func.attrs)
    return {}


def _reconstruct_params_from_trace(func_builder: Any, trace_result: TraceResult) -> None:
    """Reconstruct function parameters from trace result."""
    defined_vars = set()
    used_vars = {}
    
    for op in trace_result.operations:
        if op.output_var and isinstance(op.output_var, ir.Var) and hasattr(op.output_var, "name_hint"):
            defined_vars.add(op.output_var.name_hint)
        
        for arg in op.args:
            if isinstance(arg, ir.Var) and hasattr(arg, "name_hint"):
                var_name = arg.name_hint
                if var_name not in used_vars and hasattr(arg, "type"):
                    used_vars[var_name] = arg.type
    
    param_map = {name: var_type for name, var_type in used_vars.items() if name not in defined_vars}
    
    for param_name, param_type in param_map.items():
        func_builder.param(param_name, param_type)


def _reconstruct_return_types_from_trace(func_builder: Any, trace_result: TraceResult) -> None:
    """Reconstruct function return types from trace result."""
    pass


def _reconstruct_body(builder: ir.IRBuilder, trace_result: TraceResult) -> None:
    """Reconstruct function body from trace operations.

    This is the core reconstruction logic that processes each operation
    in the trace and rebuilds the corresponding IR statements.

    Args:
        builder: IRBuilder instance for constructing IR
        trace_result: TraceResult containing operations to reconstruct

    Raises:
        TraceReconstructionError: If an operation cannot be reconstructed
    """
    var_map: dict[str, ir.Var] = {}

    for op in trace_result.operations:
        try:
            _reconstruct_operation(builder, op, var_map)
        except Exception as e:
            raise TraceReconstructionError(
                f"Failed to reconstruct operation [{op.index}] {op.op_name}: {e}"
            ) from e


def _reconstruct_operation(builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var]) -> None:
    """Reconstruct a single operation from trace info.

    Args:
        builder: IRBuilder instance
        op: TraceInfo for the operation to reconstruct
        var_map: Dictionary mapping variable names to IR Var objects

    Raises:
        TraceReconstructionError: If the operation cannot be reconstructed
    """
    if op.op_type == "tensor":
        _reconstruct_tensor_op(builder, op, var_map)
    elif op.op_type == "tile":
        _reconstruct_tile_op(builder, op, var_map)
    elif op.op_type == "system":
        _reconstruct_system_op(builder, op, var_map)
    else:
        raise TraceReconstructionError(f"Unknown operation type: {op.op_type}")


_BINARY_OPS = {"tensor.add", "tensor.mul", "tensor.sub", "tensor.div"}
_UNARY_OPS = {"tensor.sqrt", "tensor.exp", "tensor.neg", "tensor.relu", "tensor.sigmoid", "tensor.tanh"}


def _reconstruct_tensor_op(builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var]) -> None:
    """Reconstruct a tensor operation."""
    op_name = op.op_name

    if op_name in _BINARY_OPS:
        _reconstruct_binary_op(builder, op, var_map, op_name.split(".")[1])
    elif op_name in _UNARY_OPS:
        _reconstruct_unary_op(builder, op, var_map, op_name.split(".")[1])
    elif op_name == "tensor.cast":
        _reconstruct_cast_op(builder, op, var_map)
    else:
        _reconstruct_generic_call(builder, op, var_map)


_TILE_BINARY_OPS = {"tile.add", "tile.mul", "tile.sub", "tile.div"}
_TILE_UNARY_OPS = {"tile.neg", "tile.sqrt", "tile.exp"}
_TILE_MEM_OPS = {"tile.load", "tile.store"}


def _reconstruct_tile_op(builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var]) -> None:
    """Reconstruct a tile operation."""
    op_name = op.op_name

    if op_name in _TILE_MEM_OPS:
        if op_name == "tile.load":
            _reconstruct_load_op(builder, op, var_map)
        elif op_name == "tile.store":
            _reconstruct_store_op(builder, op, var_map)
    elif op_name in _TILE_BINARY_OPS:
        _reconstruct_binary_op(builder, op, var_map, op_name.split(".")[1])
    elif op_name in _TILE_UNARY_OPS:
        _reconstruct_unary_op(builder, op, var_map, op_name.split(".")[1])
    else:
        _reconstruct_generic_call(builder, op, var_map)


def _reconstruct_system_op(builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var]) -> None:
    """Reconstruct a system operation."""
    _reconstruct_generic_call(builder, op, var_map)


def _reconstruct_binary_op(
    builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var], op_name: str
) -> None:
    """Reconstruct a binary operation (add, mul, sub, div)."""
    if len(op.args) != 2:
        raise TraceReconstructionError(f"Binary operation {op_name} expected 2 args, got {len(op.args)}")

    lhs = _resolve_arg(op.args[0], var_map)
    rhs = _resolve_arg(op.args[1], var_map)

    span = _create_span(op.source_location)
    dtype = _extract_dtype_from_return_type(op.return_type)

    if op_name == "add":
        result_expr = ir.Add(lhs, rhs, dtype, span)
    elif op_name == "mul":
        result_expr = ir.Mul(lhs, rhs, dtype, span)
    elif op_name == "sub":
        result_expr = ir.Sub(lhs, rhs, dtype, span)
    elif op_name == "div":
        result_expr = ir.FloatDiv(lhs, rhs, dtype, span)
    else:
        raise TraceReconstructionError(f"Unknown binary operation: {op_name}")

    if op.output_var and hasattr(op.output_var, "name_hint"):
        output_var = _get_or_create_var(builder, op.output_var, var_map)
        builder.assign(output_var, result_expr, span)


def _reconstruct_unary_op(
    builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var], op_name: str
) -> None:
    """Reconstruct a unary operation (sqrt, exp, etc.)."""
    if len(op.args) != 1:
        raise TraceReconstructionError(f"Unary operation {op_name} expected 1 arg, got {len(op.args)}")

    arg = _resolve_arg(op.args[0], var_map)
    span = _create_span(op.source_location)

    op_obj = ir.Op(f"tensor.{op_name}")
    result_expr = ir.Call(op_obj, [arg], {}, span)

    if op.output_var and hasattr(op.output_var, "name_hint"):
        output_var = _get_or_create_var(builder, op.output_var, var_map)
        builder.assign(output_var, result_expr, span)


def _reconstruct_cast_op(builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var]) -> None:
    """Reconstruct a cast operation."""
    if len(op.args) != 1:
        raise TraceReconstructionError(f"Cast operation expected 1 arg, got {len(op.args)}")

    arg = _resolve_arg(op.args[0], var_map)
    span = _create_span(op.source_location)

    dtype = _extract_dtype_from_kwargs(op.kwargs)
    if dtype is None:
        raise TraceReconstructionError("Cast operation missing dtype in kwargs")

    op_obj = ir.Op("tensor.cast")
    result_expr = ir.Call(op_obj, [arg], {"dtype": dtype}, span)

    if op.output_var and hasattr(op.output_var, "name_hint"):
        output_var = _get_or_create_var(builder, op.output_var, var_map)
        builder.assign(output_var, result_expr, span)


def _reconstruct_load_op(builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var]) -> None:
    """Reconstruct a load operation."""
    if len(op.args) < 1:
        raise TraceReconstructionError(f"Load operation expected at least 1 arg, got {len(op.args)}")

    resolved_args = [_resolve_arg(arg, var_map) for arg in op.args]
    span = _create_span(op.source_location)

    op_obj = ir.Op("tile.load")
    result_expr = ir.Call(op_obj, resolved_args, {}, span)

    if op.output_var and hasattr(op.output_var, "name_hint"):
        output_var = _get_or_create_var(builder, op.output_var, var_map)
        builder.assign(output_var, result_expr, span)


def _reconstruct_store_op(builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var]) -> None:
    """Reconstruct a store operation."""
    if len(op.args) < 2:
        raise TraceReconstructionError(f"Store operation expected at least 2 args, got {len(op.args)}")

    resolved_args = [_resolve_arg(arg, var_map) for arg in op.args]
    span = _create_span(op.source_location)

    op_obj = ir.Op("tile.store")
    result_expr = ir.Call(op_obj, resolved_args, {}, span)

    if op.output_var and hasattr(op.output_var, "name_hint"):
        output_var = _get_or_create_var(builder, op.output_var, var_map)
        builder.assign(output_var, result_expr, span)


def _reconstruct_generic_call(builder: ir.IRBuilder, op: TraceInfo, var_map: dict[str, ir.Var]) -> None:
    """Reconstruct a generic call operation."""
    resolved_args = [_resolve_arg(arg, var_map) for arg in op.args]
    span = _create_span(op.source_location)

    op_obj = ir.Op(op.op_name)
    result_expr = ir.Call(op_obj, resolved_args, op.kwargs, span)

    if op.output_var and hasattr(op.output_var, "name_hint"):
        output_var = _get_or_create_var(builder, op.output_var, var_map)
        builder.assign(output_var, result_expr, span)


def _resolve_arg(arg: ir.Expr, var_map: dict[str, ir.Var]) -> ir.Expr:
    """Resolve an argument expression, replacing variables with tracked ones."""
    if isinstance(arg, ir.Var) and hasattr(arg, "name_hint"):
        var_name = arg.name_hint
        if var_name in var_map:
            return var_map[var_name]
    return arg


def _get_or_create_var(builder: ir.IRBuilder, var_expr: ir.Expr, var_map: dict[str, ir.Var]) -> ir.Var:
    """Get or create a variable from an expression."""
    if not hasattr(var_expr, "name_hint"):
        raise TraceReconstructionError("Cannot create variable without name_hint")

    var_name = var_expr.name_hint
    if var_name in var_map:
        return var_map[var_name]

    var_type = var_expr.type if hasattr(var_expr, "type") else None
    if var_type is None:
        raise TraceReconstructionError(f"Cannot create variable {var_name} without type information")

    new_var = builder.var(var_name, var_type)
    var_map[var_name] = new_var
    return new_var


def _create_span(source_location: tuple[str, int, int] | None) -> ir.Span:
    """Create an IR Span from source location tuple."""
    if source_location is None:
        return ir.Span.unknown()

    file, line, col = source_location
    return ir.Span(file, line, col, line, col)


def _extract_dtype_from_kwargs(kwargs: dict) -> DataType | None:
    """Extract DataType from kwargs."""
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
        if isinstance(dtype, DataType):
            return dtype
    return None


def _extract_dtype_from_return_type(return_type: str) -> DataType:
    """Extract DataType from return type string."""
    dtype_map = {
        "FP16": DataType.FP16,
        "FP32": DataType.FP32,
        "BF16": DataType.BF16,
        "INT8": DataType.INT8,
        "INT16": DataType.INT16,
        "INT32": DataType.INT32,
        "INT64": DataType.INT64,
        "UINT8": DataType.UINT8,
        "UINT16": DataType.UINT16,
        "UINT32": DataType.UINT32,
        "UINT64": DataType.UINT64,
        "BOOL": DataType.BOOL,
    }

    for dtype_name, dtype in dtype_map.items():
        if dtype_name in return_type:
            return dtype

    return DataType.FP32
