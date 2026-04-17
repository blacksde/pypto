# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TensorArray operations for tape-based gradient computation in PyPTO IR."""

from collections.abc import Sequence
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstInt, Expr, MakeTuple, Op, ScalarType, Span, TensorType

from ..utils import _get_span_or_capture, _to_make_tuple


def create(
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    dtype: DataType,
    capacity: int | Expr,
    span: Span | None = None,
) -> Call:
    """Create a new TensorArray for storing intermediate values.

    Used for tape-based gradient computation in reverse mode autodiff.

    Args:
        shape: Shape of each element tensor (list of int/Expr, or MakeTuple)
        dtype: Data type of element tensors
        capacity: Maximum number of elements (loop iterations)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a TensorArray
    """
    actual_span = _get_span_or_capture(span)

    shape_tuple = _to_make_tuple(shape, actual_span)

    if isinstance(capacity, int):
        capacity_expr = ConstInt(capacity, DataType.INT64, actual_span)
    else:
        capacity_expr = capacity

    args = [shape_tuple, capacity_expr]
    kwargs: dict[str, Any] = {"dtype": dtype}
    
    shape_list_for_type = []
    if isinstance(shape, MakeTuple):
        for dim in shape.fields:
            shape_list_for_type.append(dim)
    elif isinstance(shape, Sequence):
        for dim in shape:
            shape_list_for_type.append(dim)
    else:
        shape_list_for_type = list(shape_tuple.fields) if hasattr(shape_tuple, 'fields') else []
    
    element_type = TensorType(shape_list_for_type, dtype)
    
    return Call(Op("tensor_array.create"), args, kwargs, element_type, actual_span)


def push(
    array: Expr,
    value: Expr,
    span: Span | None = None,
) -> Call:
    """Push a value to the TensorArray.

    Appends a tensor value to the end of the array during forward loop execution.

    Args:
        array: TensorArray expression
        value: Tensor value to append
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for the push operation
    """
    actual_span = _get_span_or_capture(span)

    dummy_type = TensorType([], dtype=DataType.FP32)
    
    return Call(Op("tensor_array.push"), [array, value], {}, dummy_type, actual_span)


def get(
    array: Expr,
    index: int | Expr,
    element_type: TensorType,
    span: Span | None = None,
) -> Call:
    """Get a value from the TensorArray at specified index.

    Retrieves a tensor value from the array during reverse loop execution.

    Args:
        array: TensorArray expression
        index: Index of the element to retrieve (0-based)
        element_type: Type of the returned tensor
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning the tensor at the specified index
    """
    actual_span = _get_span_or_capture(span)

    if isinstance(index, int):
        index_expr = ConstInt(index, DataType.INT64, actual_span)
    else:
        index_expr = index

    return Call(Op("tensor_array.get"), [array, index_expr], {}, element_type, actual_span)


def length(
    array: Expr,
    span: Span | None = None,
) -> Call:
    """Get the current length of the TensorArray.

    Args:
        array: TensorArray expression
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression returning the number of elements in the array
    """
    actual_span = _get_span_or_capture(span)

    return Call(Op("tensor_array.length"), [array], {}, ScalarType(DataType.INT64), actual_span)