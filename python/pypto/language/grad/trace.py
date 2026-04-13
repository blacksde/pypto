# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Trace module for collecting and analyzing operations in PyPTO functions.

This module provides functionality to trace operations within pl.function decorated functions,
collecting information about each operation in execution order, and providing tools for
analysis and debugging.

Typical usage:
    import pypto.language as pl

    @pl.function
    def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
        return result

    trace_result = pl.trace(my_func)
    print(trace_result)
    pl.print_trace(trace_result, format="detailed")
"""

from dataclasses import dataclass, field
from typing import Any, List
import json

from pypto.pypto_core.ir import (
    IRVisitor,
    Function,
    Call,
    Expr,
    Var,
    ConstInt,
    ConstFloat,
    ConstBool,
    FunctionType,
)

_SYSTEM_OPS = {
    "tpush_to_aiv",
    "tpush_to_aic",
    "tpop_from_aic",
    "tpop_from_aiv",
    "aic_initialize_pipe",
    "aiv_initialize_pipe",
    "reserve_buffer",
    "import_peer_buffer",
    "tfree_to_aic",
    "tfree_to_aiv",
    "sync_src",
    "sync_dst",
}

_TILE_OPS = {
    "load",
    "store",
    "move",
    "create_tile",
    "fillpad",
    "gemv",
    "gemv_acc",
    "gemv_bias",
    "matmul_bias",
    "addc",
    "subc",
    "addsc",
    "subsc",
    "and_",
    "ands",
    "or_",
    "ors",
    "xor",
    "xors",
    "shl",
    "shls",
    "shr",
    "shrs",
    "cmp",
    "cmps",
    "rem",
    "rems",
    "sel",
    "sels",
    "maxs",
    "mins",
    "prelu",
    "lrelu",
}


@dataclass
class TraceInfo:
    """Information about a single operation in the trace.

    Attributes:
        op_name: Name of the operation (e.g., "add", "matmul")
        op_type: Type of operation ("tensor", "tile", "system", "binary", "unary")
        args: List of argument expressions
        kwargs: Dictionary of keyword arguments
        arg_types: List of type information strings for each argument
        return_type: Return type information string
        source_location: Tuple of (file, line, column) if available
        index: Sequential index in the trace
        output_var: Output variable expression if this operation defines a variable
    """

    op_name: str
    op_type: str
    args: List[Expr]
    kwargs: dict
    arg_types: List[str]
    return_type: str
    source_location: tuple[str, int, int] | None
    index: int
    output_var: Expr | None = None


@dataclass
class TraceResult:
    """Result of tracing a function.

    Attributes:
        function_name: Name of the traced function
        function_type: Type of function (Orchestration, InCore, etc.)
        operations: List of collected operations in execution order
        param_info: List of parameter information strings
        return_types: List of return type information strings
        function: The function object that was traced
    """

    function_name: str
    function_type: FunctionType
    operations: List[TraceInfo] = field(default_factory=list)
    param_info: List[str] = field(default_factory=list)
    return_types: List[str] = field(default_factory=list)
    function: Function | None = None

    def filter_by_op(self, op_name: str) -> List[TraceInfo]:
        """Filter operations by operation name.

        Args:
            op_name: Name of the operation to filter for

        Returns:
            List of TraceInfo objects matching the operation name
        """
        return [op for op in self.operations if op.op_name == op_name]

    def filter_by_type(self, op_type: str) -> List[TraceInfo]:
        """Filter operations by operation type.

        Args:
            op_type: Type of operation to filter for ("tensor", "tile", "system", etc.)

        Returns:
            List of TraceInfo objects matching the operation type
        """
        return [op for op in self.operations if op.op_type == op_type]

    def get_operation(self, index: int) -> TraceInfo | None:
        """Get operation by index.

        Args:
            index: Index of the operation to retrieve

        Returns:
            TraceInfo object if found, None otherwise
        """
        if 0 <= index < len(self.operations):
            return self.operations[index]
        return None

    def count(self) -> int:
        """Get total number of operations in the trace.

        Returns:
            Number of operations in the trace
        """
        return len(self.operations)

    def __str__(self) -> str:
        """Pretty print format for trace result."""
        lines = []
        lines.append(f"Trace for function '{self.function_name}'")
        lines.append(f"Type: {self.function_type}")

        if self.param_info:
            lines.append("Parameters:")
            for param in self.param_info:
                lines.append(f"    {param}")

        if self.return_types:
            lines.append("Return:")
            for ret_type in self.return_types:
                lines.append(f"    {ret_type}")

        lines.append(f"\nOperations ({self.count()} total):")
        for op in self.operations:
            lines.append(f"[{op.index}] {op.op_name}")
            lines.append(f"    Type: {op.op_type}")
            lines.append("    Args:")
            for i, (arg, arg_type) in enumerate(zip(op.args, op.arg_types)):
                lines.append(f"        - {i}: {arg_type}")
            if op.kwargs:
                lines.append("    Kwargs:")
                for k, v in op.kwargs.items():
                    lines.append(f"        {k}: {v}")
            lines.append(f"    Return: {op.return_type}")
            if op.source_location:
                file, line, col = op.source_location
                lines.append(f"    Location: {file}:{line}:{col}")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export trace result as JSON string.

        Returns:
            JSON string representation of the trace
        """
        data = {
            "function_name": self.function_name,
            "function_type": str(self.function_type),
            "param_info": self.param_info,
            "return_types": self.return_types,
            "operations": [],
        }

        for op in self.operations:
            # Convert kwargs to JSON-serializable format
            serializable_kwargs = {}
            for k, v in op.kwargs.items():
                try:
                    # Try to serialize directly
                    json.dumps(v)
                    serializable_kwargs[k] = v
                except (TypeError, ValueError):
                    # Convert to string representation
                    serializable_kwargs[k] = str(v)

            # Convert source location to JSON-serializable format
            serializable_location = None
            if op.source_location:
                try:
                    json.dumps(op.source_location)
                    serializable_location = op.source_location
                except (TypeError, ValueError):
                    serializable_location = [str(x) for x in op.source_location]

            op_data = {
                "index": op.index,
                "op_name": op.op_name,
                "op_type": op.op_type,
                "arg_types": op.arg_types,
                "return_type": op.return_type,
                "kwargs": serializable_kwargs,
                "source_location": serializable_location,
            }
            data["operations"].append(op_data)

        return json.dumps(data, indent=2)


class TraceVisitor(IRVisitor):
    """IR visitor for collecting operation trace information.

    This visitor traverses a function's IR and collects information about
    all operations (Call nodes) in execution order.
    """

    def __init__(self) -> None:
        """Initialize trace visitor."""
        super().__init__()
        self.operations: List[TraceInfo] = []
        self.current_index: int = 0
        self.function_name: str = ""
        self.function_type: FunctionType | None = None
        self.param_info: List[str] = []
        self.return_types: List[str] = []
        self._backward_registered_functions: set = set()
        self._current_output_var: Expr | None = None

    def _is_backward_registered_call(self, op: Call) -> bool:
        """Check if this is a call to a backward registered function.

        Args:
            op: Call node to check

        Returns:
            True if this is a call to a function marked as is_backward_registered
        """
        # Check if the operation is a function call
        if hasattr(op, "op") and hasattr(op.op, "name"):
            op_name = op.op.name
            # Check if this function name is in our backward registered set
            return op_name in self._backward_registered_functions
        return False

    def visit_function(self, func: Function) -> None:
        """Visit a function and initialize trace.

        Args:
            func: Function to trace
        """
        self.function_name = func.name
        self.function_type = func.func_type

        # Check if this function is marked as is_backward_registered
        # If so, add it to our set so we can skip its internal operations
        if hasattr(func, "attrs") and func.attrs:
            if func.attrs.get("is_backward_registered", False):
                self._backward_registered_functions.add(func.name)

        # Collect parameter information
        for param in func.params:
            param_str = f"{param.name_hint}: {param.type}"
            self.param_info.append(param_str)

        # Collect return type information
        for ret_type in func.return_types:
            self.return_types.append(str(ret_type))

        # Visit function body
        if func.body:
            self.visit_stmt(func.body)

    def visit_for_stmt(self, stmt) -> None:
        """Visit a for loop statement and record loop information.
        
        Args:
            stmt: ForStmt node to visit
        """
        # Extract loop kind string
        kind_str = str(stmt.kind).split('.')[-1] if hasattr(stmt, 'kind') else 'Sequential'
        
        # Determine operation name based on kind
        op_name_map = {
            'Sequential': 'range',
            'Parallel': 'parallel',
            'Unroll': 'unroll',
        }
        op_name = op_name_map.get(kind_str, f'for.{kind_str.lower()}')
        
        # Extract loop bounds as arguments
        args = [stmt.start, stmt.stop, stmt.step]
        
        # Extract argument types
        arg_types = [self._get_type_info(arg) for arg in args]
        
        # Extract source location
        source_location = None
        if hasattr(stmt, 'span') and stmt.span:
            span = stmt.span
            if hasattr(span, 'file_path') and hasattr(span, 'line') and hasattr(span, 'column'):
                source_location = (span.file_path, span.line, span.column)
        
        # Create loop info as a special operation
        loop_info = TraceInfo(
            op_name=op_name,
            op_type='control_flow',
            args=args,
            kwargs={
                'iter_args': len(stmt.iter_args) if hasattr(stmt, 'iter_args') else 0,
                'return_vars': len(stmt.return_vars) if hasattr(stmt, 'return_vars') else 0,
                'kind': kind_str,
            },
            arg_types=arg_types,
            return_type=f'ForLoop[{stmt.loop_var}]',
            source_location=source_location,
            index=self.current_index,
            output_var=None,
        )
        self.operations.append(loop_info)
        self.current_index += 1
        
        # Visit loop body to collect operations inside the loop
        if hasattr(stmt, 'body') and stmt.body:
            self.visit_stmt(stmt.body)
    
    def visit_while_stmt(self, stmt) -> None:
        """Visit a while loop statement and record loop information.
        
        Args:
            stmt: WhileStmt node to visit
        """
        # Extract source location
        source_location = None
        if hasattr(stmt, 'span') and stmt.span:
            span = stmt.span
            if hasattr(span, 'file_path') and hasattr(span, 'line') and hasattr(span, 'column'):
                source_location = (span.file_path, span.line, span.column)
        
        # Create while loop info as a special operation
        loop_info = TraceInfo(
            op_name='while',
            op_type='control_flow',
            args=[stmt.condition] if hasattr(stmt, 'condition') else [],
            kwargs={
                'iter_args': len(stmt.iter_args) if hasattr(stmt, 'iter_args') else 0,
                'return_vars': len(stmt.return_vars) if hasattr(stmt, 'return_vars') else 0,
            },
            arg_types=[self._get_type_info(stmt.condition)] if hasattr(stmt, 'condition') else [],
            return_type='WhileLoop',
            source_location=source_location,
            index=self.current_index,
            output_var=None,
        )
        self.operations.append(loop_info)
        self.current_index += 1
        
        # Visit loop body to collect operations inside the loop
        if hasattr(stmt, 'body') and stmt.body:
            self.visit_stmt(stmt.body)
    
    def visit_assign_stmt(self, stmt) -> None:
        """Visit an assignment statement and track the output variable.
        
        Args:
            stmt: AssignStmt node to visit
        """
        # Set current output variable before visiting value
        self._current_output_var = stmt.var
        
        # Visit value (which will visit any calls)
        if stmt.value:
            self.visit_expr(stmt.value)
        
        # Clear current output variable
        self._current_output_var = None

    def visit_call(self, op: Call) -> None:
        """Visit a call operation and collect trace information.

        Args:
            op: Call operation to trace
        """
        # Check if this is a call to a function marked as is_backward_registered
        # If so, skip visiting its internal operations
        if self._is_backward_registered_call(op):
            trace_info = self._extract_op_info(op)
            self.operations.append(trace_info)
            self.current_index += 1
            # Don't visit arguments - treat as independent node
            return

        trace_info = self._extract_op_info(op)
        self.operations.append(trace_info)
        self.current_index += 1

        # Continue visiting arguments
        for arg in op.args:
            self.visit_expr(arg)

    def _extract_op_info(self, call: Call) -> TraceInfo:
        """Extract trace information from a Call node.

        Args:
            call: Call node to extract information from

        Returns:
            TraceInfo object with operation details
        """
        op_name = call.op.name if hasattr(call.op, "name") else str(call.op)
        op_type = self._determine_op_type(call)

        # Extract argument types
        arg_types = []
        for arg in call.args:
            arg_types.append(self._get_type_info(arg))

        # Extract return type
        return_type = self._get_type_info(call) if hasattr(call, "type") else "unknown"

        # Extract source location
        source_location = None
        if hasattr(call, "span") and call.span:
            span = call.span
            if hasattr(span, "file_path") and hasattr(span, "line") and hasattr(span, "column"):
                source_location = (span.file_path, span.line, span.column)

        return TraceInfo(
            op_name=op_name,
            op_type=op_type,
            args=list(call.args),
            kwargs=dict(call.kwargs) if hasattr(call, "kwargs") else {},
            arg_types=arg_types,
            return_type=return_type,
            source_location=source_location,
            index=self.current_index,
            output_var=self._current_output_var,
        )

    def _determine_op_type(self, call: Call) -> str:
        """Determine the type of operation.

        Args:
            call: Call node to classify

        Returns:
            Operation type string
        """
        op_name = call.op.name if hasattr(call.op, "name") else str(call.op)

        if op_name in _SYSTEM_OPS:
            return "system"

        if op_name in _TILE_OPS:
            return "tile"
        
        return "tensor"
    
    def _get_type_info(self, expr: Expr) -> str:
        """Get type information string for an expression.
        
        Args:
            expr: Expression to get type info for
            
        Returns:
            Type information string
        """
        if isinstance(expr, Var):
            if hasattr(expr, "type") and expr.type:
                return f"{expr.name_hint}: {expr.type}"
            return f"{expr.name_hint}: unknown"
        
        elif isinstance(expr, ConstInt):
            return f"ConstInt({expr.value})"
        
        elif isinstance(expr, ConstFloat):
            return f"ConstFloat({expr.value})"
        
        elif isinstance(expr, ConstBool):
            return f"ConstBool({expr.value})"
        
        elif isinstance(expr, Call):
            op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
            if hasattr(expr, "type") and expr.type:
                return f"Call({op_name}): {expr.type}"
            return f"Call({op_name}): unknown"
        
        else:
            if hasattr(expr, "type") and expr.type:
                return f"{type(expr).__name__}: {expr.type}"
            return type(expr).__name__
 
def trace(func: Function) -> TraceResult:
    """Trace a function and collect operation information.

    This function traverses the IR of a pl.function decorated function and
    collects information about all operations in execution order.

    Args:
        func: Function object to trace (result of @pl.function decorator)

    Returns:
        TraceResult object containing collected operation information

    Raises:
        TypeError: If func is not a Function object

    Example:
        @pl.function
        def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
            return result

        trace_result = pl.trace(my_func)
        print(trace_result)
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected Function object, got {type(func).__name__}")

    visitor = TraceVisitor()
    visitor.visit_function(func)

    return TraceResult(
        function_name=visitor.function_name,
        function_type=visitor.function_type,
        operations=visitor.operations,
        param_info=visitor.param_info,
        return_types=visitor.return_types,
        function=func,
    )


def print_trace(
    trace_result: TraceResult, format: str = "text", output: str = "console", filename: str | None = None
) -> None:
    """Print trace results in various formats.

    Args:
        trace_result: TraceResult object to print
        format: Output format ("text", "json", "detailed")
        output: Output destination ("console" or "file")
        filename: Filename for file output (required if output="file")

    Raises:
        ValueError: If output="file" but filename is not provided
        ValueError: If format is not recognized

    Example:
        trace_result = pl.trace(my_func)
        pl.print_trace(trace_result, format="text")
        pl.print_trace(trace_result, format="json")
        pl.print_trace(trace_result, format="detailed", output="file", filename="trace.txt")
    """
    if output == "file" and filename is None:
        raise ValueError("filename must be provided when output='file'")

    if format == "text":
        content = str(trace_result)
    elif format == "json":
        content = trace_result.to_json()
    elif format == "detailed":
        content = _format_detailed(trace_result)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'text', 'json', or 'detailed'")

    if output == "console":
        print(content)
    elif output == "file":
        with open(filename, "w") as f:
            f.write(content)
    else:
        raise ValueError(f"Unknown output: {output}. Use 'console' or 'file'")


def _format_detailed(trace_result: TraceResult) -> str:
    """Format trace result in detailed verbose format.

    Args:
        trace_result: TraceResult object to format

    Returns:
        Detailed formatted string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"DETAILED TRACE FOR FUNCTION: {trace_result.function_name}")
    lines.append("=" * 80)
    lines.append(f"Function Type: {trace_result.function_type}")
    lines.append(f"Total Operations: {trace_result.count()}")
    lines.append("")

    if trace_result.param_info:
        lines.append("-" * 80)
        lines.append("PARAMETERS:")
        lines.append("-" * 80)
        for i, param in enumerate(trace_result.param_info):
            lines.append(f"  [{i}] {param}")
        lines.append("")

    if trace_result.return_types:
        lines.append("-" * 80)
        lines.append("RETURN TYPES:")
        lines.append("-" * 80)
        for i, ret_type in enumerate(trace_result.return_types):
            lines.append(f"  [{i}] {ret_type}")
        lines.append("")

    lines.append("-" * 80)
    lines.append("OPERATIONS:")
    lines.append("-" * 80)

    for op in trace_result.operations:
        lines.append("")
        lines.append(f"  [{op.index}] Operation: {op.op_name}")
        lines.append(f"      Type: {op.op_type}")
        lines.append(f"      Return Type: {op.return_type}")

        if op.source_location:
            file, line, col = op.source_location
            lines.append(f"      Source: {file}:{line}:{col}")

        lines.append("      Arguments:")
        for i, (arg, arg_type) in enumerate(zip(op.args, op.arg_types)):
            lines.append(f"        [{i}] {arg_type}")

        if op.kwargs:
            lines.append("      Keyword Arguments:")
            for k, v in op.kwargs.items():
                lines.append(f"        {k}: {v}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("TRACE SUMMARY")
    lines.append("=" * 80)

    # Count operations by type
    op_counts = {}
    for op in trace_result.operations:
        op_counts[op.op_type] = op_counts.get(op.op_type, 0) + 1

    lines.append(f"  Total Operations: {trace_result.count()}")
    lines.append("  Operations by Type:")
    for op_type, count in sorted(op_counts.items()):
        lines.append(f"    {op_type}: {count}")

    # Count unique operations
    unique_ops = set(op.op_name for op in trace_result.operations)
    lines.append(f"  Unique Operations: {len(unique_ops)}")

    return "\n".join(lines)