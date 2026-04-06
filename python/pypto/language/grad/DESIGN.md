# PyPTO Trace Functionality Design Document

## Overview

This document describes the design and implementation of the `pl.trace` functionality for PyPTO, which enables collecting and analyzing operations within `pl.function` decorated functions.

## Goals

1. **Operation Collection**: Collect all operations (`pl.op`) within a function in execution order
2. **Information Capture**: Capture detailed information about each operation including:
   - Operation name and type
   - Arguments and their types
   - Return type
   - Source location (file, line, column)
3. **Analysis Support**: Provide filtering and querying capabilities
4. **Multiple Output Formats**: Support text, JSON, and detailed output formats
5. **Integration**: Seamlessly integrate with existing IR visitor pattern

## Architecture

### Core Components

#### 1. TraceInfo Data Class

**Location**: `python/pypto/language/grad/trace.py`

**Purpose**: Store information about a single operation in the trace.

**Attributes**:
- `op_name: str` - Name of the operation (e.g., "add", "matmul")
- `op_type: str` - Type category ("tensor", "tile", "system", "binary", "unary")
- `args: List[Expr]` - Argument expressions
- `kwargs: dict` - Keyword arguments
- `arg_types: List[str]` - Type information for each argument
- `return_type: str` - Return type information
- `source_location: tuple[str, int, int] | None` - (file, line, column)
- `index: int` - Sequential index in trace

**Design Rationale**:
- Uses `@dataclass` for clean, immutable data structure
- Stores original `Expr` objects for potential further analysis
- Provides string representations for types to avoid complex serialization

#### 2. TraceResult Class

**Purpose**: Main container for trace results with analysis methods.

**Attributes**:
- `function_name: str` - Name of traced function
- `function_type: FunctionType` - Type of function (Orchestration, InCore, etc.)
- `operations: List[TraceInfo]` - Collected operations in order
- `param_info: List[str]` - Parameter information
- `return_types: List[str]` - Return type information

**Methods**:
- `filter_by_op(op_name: str) -> List[TraceInfo]` - Filter by operation name
- `filter_by_type(op_type: str) -> List[TraceInfo]` - Filter by operation type
- `get_operation(index: int) -> TraceInfo | None` - Get operation by index
- `count() -> int` - Get total number of operations
- `__str__() -> str` - Pretty print format
- `to_json() -> str` - JSON export

**Design Rationale**:
- Provides both data storage and analysis capabilities
- Immutable operations list ensures trace integrity
- Multiple output formats for different use cases

#### 3. TraceVisitor Class

**Purpose**: IR visitor that traverses function IR and collects operation information.

**Inheritance**: Extends `IRVisitor` from `pypto.pypto_core.ir`

**Attributes**:
- `operations: List[TraceInfo]` - Collected operations
- `current_index: int` - Counter for operation ordering
- `function_name: str` - Name of function being traced
- `function_type: FunctionType | None` - Type of function
- `param_info: List[str]` - Parameter information
- `return_types: List[str]` - Return type information

**Key Methods**:
- `visit_function(func: Function) -> None` - Initialize trace and traverse body
- `visit_call(op: Call) -> None` - Capture operation information
- `_extract_op_info(call: Call) -> TraceInfo` - Extract operation details
- `_determine_op_type(call: Call) -> str` - Classify operation type
- `_get_type_info(expr: Expr) -> str` - Extract type information

**Design Rationale**:
- Leverages existing `IRVisitor` pattern for consistent IR traversal
- Maintains execution order through sequential indexing
- Separates concerns between traversal and information extraction

### Public API

#### trace(func: Function) -> TraceResult

**Purpose**: Main entry point for tracing a function.

**Parameters**:
- `func: Function` - Function object to trace (result of `@pl.function` decorator)

**Returns**:
- `TraceResult` object containing collected operation information

**Raises**:
- `TypeError` - If func is not a Function object

**Example**:
```python
@pl.function
def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
    result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
    return result

trace_result = pl.trace(my_func)
print(trace_result)
```

#### print_trace(trace_result, format, output, filename)

**Purpose**: Print trace results in various formats.

**Parameters**:
- `trace_result: TraceResult` - Trace result to print
- `format: str` - Output format ("text", "json", "detailed")
- `output: str` - Output destination ("console" or "file")
- `filename: str | None` - Filename for file output

**Raises**:
- `ValueError` - If format or output is invalid
- `ValueError` - If output="file" but filename is None

**Example**:
```python
trace_result = pl.trace(my_func)
pl.print_trace(trace_result, format="text")
pl.print_trace(trace_result, format="json")
pl.print_trace(trace_result, format="detailed", output="file", filename="trace.txt")
```

## Operation Type Classification

The trace system classifies operations into three main categories:

### 1. Tensor Operations
High-level tensor operations used in orchestration functions:
- `add`, `sub`, `mul`, `div`, `exp`, `sqrt`, etc.
- `matmul`, `matmul_acc`, `concat`, `reshape`, etc.
- `create_tensor`, `assemble`, `dim`, etc.

### 2. Tile Operations
Tile-level operations used in in-core functions:
- `load`, `store`, `move`
- `addc`, `subc`, `addsc`, `subsc`
- `gemv`, `gemv_acc`, `gemv_bias`
- `and_`, `or_`, `xor`, and bitwise operations
- `cmp`, `rem`, `sel`, etc.

### 3. System Operations
Hardware synchronization and cross-core communication:
- `tpush_to_aiv`, `tpush_to_aic`
- `tpop_from_aic`, `tpop_from_aiv`
- `aic_initialize_pipe`, `aiv_initialize_pipe`
- `reserve_buffer`, `import_peer_buffer`
- `tfree_to_aic`, `tfree_to_aiv`

## Type Information Extraction

The trace system extracts type information from various expression types:

### Variable Expressions
Format: `{var_name}: {type}`
Example: `x: Tensor[[64, 128], FP16]`

### Constant Expressions
Format: `Const{Type}({value})`
Examples:
- `ConstInt(42)`
- `ConstFloat(3.14)`
- `ConstBool(True)`

### Call Expressions
Format: `Call({op_name}): {return_type}`
Example: `Call(add): Tensor[[64, 128], FP32]`

## Output Formats

### 1. Text Format (Default)
Human-readable summary format:
```
Trace for function 'my_function'
Type: Orchestration
Parameters:
    x: Tensor[[64, 128], FP16]
Return:
    Tensor[[64, 128], FP32]

Operations (1 total):
[0] add
    Type: tensor
    Args:
        - 0: x: Tensor[[64, 128], FP16]
        - 1: x: Tensor[[64, 128], FP16]
    Return: Tensor[[64, 128], FP32]
    Location: example.py:10:5
```

### 2. JSON Format
Machine-readable JSON format:
```json
{
  "function_name": "my_function",
  "function_type": "Orchestration",
  "param_info": ["x: Tensor[[64, 128], FP16]"],
  "return_types": ["Tensor[[64, 128], FP32]"],
  "operations": [
    {
      "index": 0,
      "op_name": "add",
      "op_type": "tensor",
      "arg_types": ["x: Tensor[[64, 128], FP16]", "x: Tensor[[64, 128], FP16]"],
      "return_type": "Tensor[[64, 128], FP32]",
      "kwargs": {},
      "source_location": ["example.py", 10, 5]
    }
  ]
}
```

### 3. Detailed Format
Verbose format with summary statistics:
```
================================================================================
DETAILED TRACE FOR FUNCTION: my_function
================================================================================
Function Type: Orchestration
Total Operations: 1

--------------------------------------------------------------------------------
PARAMETERS:
--------------------------------------------------------------------------------
  [0] x: Tensor[[64, 128], FP16]

--------------------------------------------------------------------------------
RETURN TYPES:
--------------------------------------------------------------------------------
  [0] Tensor[[64, 128], FP32]

--------------------------------------------------------------------------------
OPERATIONS:
--------------------------------------------------------------------------------

  [0] Operation: add
        Type: tensor
        Return Type: Tensor[[64, 128], FP32]
        Source: example.py:10:5
        Arguments:
          [0] x: Tensor[[64, 128], FP16]
          [1] x: Tensor[[64, 128], FP16]

================================================================================
TRACE SUMMARY
================================================================================
  Total Operations: 1
  Operations by Type:
    tensor: 1
  Unique Operations: 1
```

## Integration Points

### 1. Language Module Integration

**File**: `python/pypto/language/__init__.py`

**Changes**:
```python
from .grad.trace import trace, print_trace, TraceResult, TraceInfo

__all__.extend([
    "trace",
    "print_trace",
    "TraceResult",
    "TraceInfo",
])
```

### 2. Module Structure

**Directory**: `python/pypto/language/grad/`

**Purpose**: Contains gradient and analysis-related functionality.

**Contents**:
- `trace.py` - Trace implementation
- `__init__.py` - Module initialization (optional)

## Testing Strategy

### Unit Tests

**File**: `tests/ut/language/test_trace.py`

**Test Cases**:

1. **Basic Function Tracing**
   - Test simple function with few operations
   - Verify operation count and names
   - Check parameter and return type information

2. **Complex Control Flow**
   - Functions with for loops
   - Functions with if conditionals
   - Nested control structures
   - Verify operations are collected in correct order

3. **Operation Type Classification**
   - Test tensor operations
   - Test tile operations
   - Test system operations
   - Verify correct type classification

4. **Filtering Operations**
   - Test `filter_by_op()` method
   - Test `filter_by_type()` method
   - Test `get_operation()` method
   - Verify filtering logic

5. **Output Formats**
   - Test text format output
   - Test JSON format output
   - Test detailed format output
   - Verify file output functionality

6. **Error Handling**
   - Test tracing non-Function objects
   - Test invalid format strings
   - Test file output without filename
   - Verify proper error messages

7. **Edge Cases**
   - Functions with no operations
   - Functions with many operations
   - Functions with complex expressions
   - Functions with constant arguments

### Example Test

```python
import pypto.language as pl

@pl.function
def simple_add(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
    result: pl.Tensor[[64, 128], pl.FP32] = pl.add(x, x)
    return result

def test_trace_simple_function():
    trace_result = pl.trace(simple_add)

    # Verify basic information
    assert trace_result.function_name == "simple_add"
    assert trace_result.count() == 1

    # Verify operation details
    op = trace_result.operations[0]
    assert op.op_name == "add"
    assert op.op_type == "tensor"
    assert len(op.args) == 2

    # Test filtering
    add_ops = trace_result.filter_by_op("add")
    assert len(add_ops) == 1

    # Test printing
    text_output = str(trace_result)
    assert "add" in text_output

    # Test JSON export
    import json
    json_data = json.loads(trace_result.to_json())
    assert json_data["function_name"] == "simple_add"
    assert len(json_data["operations"]) == 1
```

## Future Enhancements

### 1. Advanced Analysis
- **Operation Graph**: Build dependency graph between operations
- **Complexity Metrics**: Compute FLOPs, memory usage estimates
- **Pattern Detection**: Identify common operation patterns

### 2. Trace Comparison
- **Diff Traces**: Compare traces between two functions
- **Similarity Metrics**: Compute operation similarity scores
- **Change Detection**: Identify operation additions/removals

### 3. Visualization
- **Call Graphs**: Generate visual operation dependency graphs
- **Timeline Views**: Show operation execution order
- **Heat Maps**: Visualize operation density by type

### 4. Performance Integration
- **Timing Information**: Add execution time to trace
- **Memory Profiling**: Track memory allocation/deallocation
- **Hardware Counters**: Include performance counter data

### 5. Scope-Aware Tracing
- **Loop Context**: Track operations within specific loops
- **Conditional Context**: Distinguish operations in different branches
- **Scope Filtering**: Trace operations within specific scopes

## Implementation Checklist

- [x] Create `python/pypto/language/grad/` directory
- [x] Implement `TraceInfo` data class
- [x] Implement `TraceResult` class with analysis methods
- [x] Implement `TraceVisitor` extending `IRVisitor`
- [x] Implement `trace()` public API function
- [x] Implement `print_trace()` with multiple formats
- [x] Add comprehensive type information extraction
- [x] Implement operation type classification
- [ ] Update `python/pypto/language/__init__.py` exports
- [ ] Create `tests/ut/language/test_trace.py` with test cases
- [ ] Run tests and validate functionality
- [ ] Update documentation
- [ ] Add examples to documentation

## Design Principles

1. **Non-Invasive**: Trace functionality should not modify the original IR
2. **Extensible**: Easy to add new operation types and analysis methods
3. **Type-Safe**: Leverage PyPTO's type system for accurate information
4. **Performant**: Efficient traversal with minimal overhead
5. **User-Friendly**: Clear, well-formatted output for debugging
6. **Testable**: Comprehensive test coverage for all functionality

## Conclusion

The `pl.trace` functionality provides a powerful tool for analyzing and debugging PyPTO functions. By leveraging the existing IR visitor pattern and providing multiple output formats, it offers both programmatic access and human-readable insights into function operations. The design is extensible and can be enhanced with additional analysis capabilities in the future.