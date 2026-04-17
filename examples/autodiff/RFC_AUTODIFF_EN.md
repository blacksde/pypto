# RFC: Source-to-Source Automatic Differentiation for PyPTO

- **RFC Number**: RFC-001
- **Author**: PyPTO Team
- **Date**: 2026-04-17
- **Status**: Draft
- **Category**: Core Feature
- **Target**: PyPTO v0.1.0

---

## Summary

This RFC proposes adding **Source-to-Source Automatic Differentiation (AD)** capability to PyPTO. The system will generate gradient functions directly from forward computation IR, enabling efficient gradient computation for deep learning and scientific computing applications.

Key features:
- **Reverse-mode AD**: Efficient for functions with many inputs and few outputs
- **IR-level transformation**: Generate gradient IR directly, no runtime tape
- **Gradient rule registry**: Extensible system for custom operators
- **Control flow support**: Handle loops and conditional branches correctly

---

## Motivation

### Background

PyPTO is a tile-based programming framework for high-performance computing, targeting GPU/accelerator optimization. In deep learning and scientific computing, gradient computation is essential for:

1. **Deep Learning Training**: Backpropagation requires efficient gradient computation
2. **Optimization Problems**: Gradient descent algorithms need derivatives
3. **Scientific Computing**: Automatic differentiation reduces manual derivation burden

Current frameworks (PyTorch, TensorFlow) use runtime tape-based AD, which:
- Introduces runtime overhead
- Requires storing intermediate values during execution
- Cannot optimize gradient computation ahead of time

### Why Source-to-Source AD?

PyPTO's IR-level approach enables:

| Approach | Runtime Tape | Source-to-Source |
|----------|--------------|------------------|
| Execution | Dynamic | Static |
| Optimization | Limited | Full compiler optimization |
| Memory | Store all intermediates | Selective storage |
| Control flow | Runtime tracing | IR-level handling |

**Benefits**:
- **Compiler optimization**: Gradient IR can be optimized like any other code
- **Memory efficiency**: Selective forward value storage (only when needed)
- **Loop handling**: SSA-based loop gradients without runtime tracing
- **Performance**: No runtime overhead for gradient construction

### Goals

1. Support gradient computation for tile/tensor operations
2. Handle loops (SSA iter_args + Tape modes)
3. Handle conditional branches (Phi node handling)
4. Extensible gradient rule registration system
5. Mathematical correctness verification

---

## Detailed Design

### 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Automatic Differentiation Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Forward Function IR                                        │
│       │                                                     │
│       ├─> 1. Create gradient parameters                     │
│       │      (forward_params + d_output)                    │
│       │                                                     │
│       ├─> 2. Collect forward values to save                 │
│       │      - Analyze backward statements                  │
│       │      - Identify loop-scoped vs non-loop vars        │
│       │                                                     │
│       ├─> 3. Reverse traversal of forward body              │
│       │      - Process each statement                       │
│       │      - Generate backward gradient statements        │
│       │      - Handle control flow (ForStmt, IfStmt)        │
│       │                                                     │
│       ├─> 4. Generate forward compute statements            │
│       │      - Recompute non-loop intermediates             │
│       │                                                     │
│       ├─> 5. Build backward function body                   │
│       │      - Concat forward + backward statements         │
│       │      - Add return statement                         │
│       │                                                     │
│       └─> Gradient Function IR                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Mathematical Foundation

**Chain Rule**:

$$\frac{\partial y}{\partial x} = \prod_{i=n}^{1} \frac{\partial f_i}{\partial f_{i-1}}$$

**Reverse Mode**: For composite function $y = f_n(f_{n-1}(...f_1(x)))$, compute gradients from output to input:

```
Forward:
  v₁ = f₁(x₁, x₂)
  v₂ = f₂(v₁, x₂)
  y  = f₃(v₂)

Backward:
  v̄₂ = ȳ · ∂f₃/∂v₂
  v̄₁ = v̄₂ · ∂f₂/∂v₁
  x̄₂ = v̄₂ · ∂f₂/∂x₂ + v̄₁ · ∂f₁/∂x₂
  x̄₁ = v̄₁ · ∂f₁/∂x₁
```

**Gradient Accumulation**: When a variable is used multiple times:

$$\frac{\partial y}{\partial x} = \sum_{paths} \frac{\partial y}{\partial x}_{path}$$

### 3. IR Node Handling

#### 3.1 Statement Types

| Node | Forward | Backward |
|------|---------|----------|
| `AssignStmt` | `y = op(a, b)` | `d_a += grad_op(dy, a, b)` |
| `SeqStmts` | Sequential execution | Reverse traversal |
| `ForStmt` | Loop execution | Reverse loop + Tape |
| `IfStmt` | Branch execution | Branch structure preserved |
| `YieldStmt` | SSA loop state | Gradient propagation |
| `ReturnStmt` | Return values | Gradient initialization |

#### 3.2 Loop Handling (Key Challenge)

**Two Modes**:

| Mode | Condition | Method |
|------|-----------|--------|
| iter_args + yield | SSA reduce pattern | `_reverse_for_iter_args` |
| Regular loop | No iter_args/yield | `_reverse_for_with_tape` |

**SSA Reduce Pattern**:
```
Forward:
  for i, (acc,) in range(N, init=(acc_init)):
    acc_new = f(acc_iter, x)
    acc = yield(acc_new)

Backward:
  for i_rev, (d_acc,) in range(N, init=(d_output)):
    # Get saved forward values from tape
    saved = tape[i_rev]
    # Compute gradients
    d_x += grad(d_acc, saved)
    d_acc_prev = yield(grad_acc)
```

**Tape Mode**:
- Create TensorArray to store loop values
- Forward: push values each iteration
- Backward: get values for gradient computation

#### 3.3 Conditional Branch Handling

```
Forward:
  if cond:
    y = f1(x)
  else:
    y = f2(x)

Backward:
  if cond:  # Use same condition
    d_x = grad_f1(d_y)
  else:
    d_x = grad_f2(d_y)
```

**Key**: Backward preserves branch structure, uses forward condition.

### 4. Gradient Rule Registration

#### 4.1 Registry Architecture

```python
class GradientRegistry:
    _registry: Dict[str, Callable] = {}
    _categories: Dict[str, str] = {}
    
    @classmethod
    def register(op_name, grad_func, category):
        _registry[op_name] = grad_func
    
    @classmethod
    def get(op_name) -> Callable:
        return _registry.get(op_name)
```

#### 4.2 Gradient Rule Signature

```python
def grad_rule(
    saved_inputs: List[ir.Expr],
    d_output: ir.Expr,
    kwargs: Dict[str, Any]
) -> List[ir.Expr]:
    """
    Compute gradients for operator inputs
    
    Args:
        saved_inputs: Forward input values (saved for backward)
        d_output: Output gradient (from downstream)
        kwargs: Additional info (e.g., forward_result)
    
    Returns:
        List of gradient expressions for each input
    """
```

#### 4.3 Built-in Operators

| Operator | Gradient Math |
|----------|---------------|
| `tile.add` | `ā=ȳ, b̄=ȳ` |
| `tile.mul` | `ā=ȳ·b, b̄=ȳ·a` |
| `tile.matmul` | `Ā=ȳ@B^T, B̄=A^T@ȳ` |
| `tile.exp` | `ā=ȳ·exp(a)` |
| `tile.relu` | `ā=ȳ·sign(a>0)` |
| `tile.sin` | `ā=ȳ·cos(a)` |

**Example**:
```python
@register_grad('tile.mul', 'tile')
def tile_mul_grad(saved_inputs, d_output, kwargs):
    a, b = saved_inputs
    da = ir.Call(ir.Op('tile.mul'), [d_output, b], ...)
    db = ir.Call(ir.Op('tile.mul'), [d_output, a], ...)
    return [da, db]
```

### 5. API Design

#### 5.1 High-level API

```python
# Generate gradient function
grad_func = pl.grad(forward_func, params=['x', 'w'])
# grad_func(x, w, d_output) -> (d_x, d_w)

# Generate forward + backward pair
forward, backward = pl.value_and_grad(forward_func)
# forward(x, w) -> y
# backward(x, w, d_y) -> (d_x, d_w)
```

#### 5.2 Custom Gradient Registration

```python
# Decorator approach
@register_grad('my.custom_op', 'user')
def custom_grad(saved_inputs, d_output, kwargs):
    return [grad_expr1, grad_expr2]

# Direct registration
GradientRegistry.register('my.op', grad_func, 'user')
```

#### 5.3 Query API

```python
GradientRegistry.list_all()           # List all registered ops
GradientRegistry.list_by_category('tile')  # List by category
GradientRegistry.has('tile.add')      # Check if registered
GradientRegistry.get('tile.add')      # Get gradient rule
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

| Task | Description | Priority |
|------|-------------|----------|
| Gradient Registry | Implement `GradientRegistry` class | High |
| Core Gradient Rules | Register 15 built-in operators | High |
| Basic IR Generation | Forward IR to backward IR | High |
| Testing Framework | Numerical verification tests | High |

**Deliverables**:
- `gradient_registry.py` (registry + decorator)
- `grad_rules.py` (15 built-in rules)
- `reverse_mode.py` (core builder)
- Test suite with numerical verification

### Phase 2: Control Flow Support (Week 3-4)

| Task | Description | Priority |
|------|-------------|----------|
| ForStmt Handling | SSA iter_args + Tape modes | High |
| IfStmt Handling | Branch preservation | Medium |
| YieldStmt Handling | SSA loop state | Medium |
| Loop Tape Analysis | Identify vars needing tape | High |

**Deliverables**:
- Loop gradient generation
- Tape-based loop handling
- Branch gradient handling
- Loop test cases

### Phase 3: Advanced Features (Week 5-6)

| Task | Description | Priority |
|------|-------------|----------|
| Gradient Accumulation | Multiple usage handling | Medium |
| Forward Value Management | Selective recomputation | Medium |
| Nested Call Handling | Recursive gradient computation | Medium |
| Operator Name Mapping | tile/tensor variants | Low |

**Deliverables**:
- Gradient accumulation for multi-use vars
- Efficient forward value strategy
- Nested expression gradients

### Phase 4: Documentation & Integration (Week 7-8)

| Task | Description | Priority |
|------|-------------|----------|
| API Documentation | User guide + examples | High |
| Design Documentation | RFC + technical doc | High |
| Integration Testing | End-to-end tests | High |
| Performance Benchmarking | Compare with PyTorch | Medium |

**Deliverables**:
- User documentation
- Developer documentation
- Integration tests
- Performance benchmarks

### Phase 5: Community Contribution (Week 9-10)

| Task | Description | Priority |
|------|-------------|----------|
| RFC Review Process | Community review workflow | High |
| Example Programs | Tutorial examples | Medium |
| Contribution Guide | How to add gradients | Medium |
| Issue Template | Bug report template | Low |

**Deliverables**:
- Community review process
- Example programs
- Contribution guide
- Issue templates

---

## Drawbacks

### Limitations

1. **Non-differentiable operations**: argmax, sort, comparison ops produce no gradients
2. **Dynamic iteration count**: Requires forward value storage (Tape overhead)
3. **High-order derivatives**: Currently supports only first-order gradients
4. **Control flow dependency**: Branch conditions need forward values saved

### Memory Considerations

- Tape mode stores loop intermediates in TensorArray
- Memory grows with iteration count
- Alternative: recomputation strategy (trading compute for memory)

---

## Alternatives Considered

### 1. Runtime Tape-based AD (PyTorch-style)

**Pros**:
- Dynamic control flow support
- Mature ecosystem

**Cons**:
- Runtime overhead
- Cannot optimize gradient code ahead of time
- Memory-intensive (stores all intermediates)

**Decision**: Source-to-source chosen for compiler optimization benefits

### 2. Forward-mode AD

**Pros**:
- Simpler implementation
- Efficient for few inputs, many outputs

**Cons**:
- Not suitable for deep learning (many parameters)
- Less efficient for our target use case

**Decision**: Reverse-mode chosen for DL efficiency

### 3. Manual Gradient Rules Only

**Pros**:
- Maximum control
- No automatic system complexity

**Cons**:
- Error-prone manual derivation
- Limited extensibility
- Poor developer experience

**Decision**: Automatic + extensible registry chosen

---

## Unresolved Questions

1. **Checkpoint strategy**: How to balance recomputation vs. storage?
   - Proposed: User-configurable checkpoint policy
   - Need: Performance benchmarks to validate

2. **Higher-order derivatives**: Support for second-order (Hessian)?
   - Current: First-order only
   - Future: Recursive application of AD

3. **Operator coverage**: Full coverage of tile/tensor ops?
   - Current: 15 core operators
   - Need: Community contribution for extended ops

4. **Performance comparison**: Benchmark against PyTorch/JAX?
   - Need: Systematic benchmarking
   - Metrics: Compute time, memory usage

---

## References

1. Baydin, A. G., et al. "Automatic differentiation in machine learning: a survey." JMLR 2018
2. PyTorch Autograd: https://pytorch.org/docs/stable/autograd.html
3. JAX Automatic Differentiation: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
4. PyPTO IR Design: `examples/ir_analysis/01_ir_node_types_20260416.md`
5. PyPTO Autodiff Design: `examples/autodiff/autodiff_design_doc.md`

---

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Gradient Registry | ✅ Complete | `gradient_registry.py` |
| Core Gradient Rules | ✅ Complete | `grad_rules.py` (15 ops) |
| Reverse Mode Builder | ✅ Complete | `reverse_mode.py` |
| Loop Handling | ✅ Complete | SSA + Tape modes |
| Branch Handling | ✅ Complete | IfStmt preserved |
| Numerical Tests | ✅ Complete | Error < 1e-4 |
| API Functions | ✅ Complete | `grad`, `value_and_grad` |

---

## Conclusion

Source-to-source automatic differentiation provides significant advantages for PyPTO's tile-based programming model:

- **Compiler optimization**: Gradient IR is optimized like any other code
- **Memory efficiency**: Selective storage, not runtime tape
- **Performance**: No runtime gradient construction overhead
- **Extensibility**: Community can contribute gradient rules

This RFC proposes a comprehensive implementation roadmap with phased delivery, enabling PyPTO to support gradient computation for deep learning and scientific computing applications.