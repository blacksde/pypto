"""
PyPTO 自动微分模块实现骨架

文件结构:
python/pypto/autodiff/
├── __init__.py           # 高层 API
├── reverse_mode.py       # 反向模式核心
├── forward_with_tape.py  # 前向带 tape
├── liveness.py           # 活性分析
├── gradient_registry.py  # 梯度规则注册
├── grad_rules.py         # 内置梯度规则
├── jacobian.py           # Jacobian 计算
├── vjp.py                # VJP 实现
├── custom_grad.py        # 自定义梯度
├── checkpoint.py         # 内存优化
└── utils.py              # 辅助工具
"""

# =============================================================================
# __init__.py - 高层 API
# =============================================================================

from typing import Union, List, Optional, Tuple, Callable
from pypto import ir
from pypto.ir import Function

__all__ = [
    'grad',
    'value_and_grad',
    'jacobian',
    'vjp',
    'CustomGrad',
    'stop_gradient',
    'checkpoint',
    'register_grad',
    'GradientRegistry',
]


def grad(
    func: Union[Function, Callable],
    params: Optional[List[str]] = None,
    grad_output: Optional[str] = None,
    name: Optional[str] = None
) -> Function:
    """
    Generate gradient function for forward function.
    
    Args:
        func: Forward function (IR Function or @pl.function decorated)
        params: List of parameter names to compute gradients (default: all)
        grad_output: Output variable name for gradient seed (default: first output)
        name: Name for generated gradient function
    
    Returns:
        Gradient function IR
    
    Example:
        >>> @pl.function
        >>> def add(x, y):
        >>>     return pl.add(x, y)
        >>> 
        >>> grad_add = pl.grad(add)
        >>> # grad_add(x, y, d_output) -> (d_x, d_y)
    """
    func_ir = _ensure_ir_function(func)
    
    if params is None:
        params = [p.name_hint for p in func_ir.params]
    
    param_vars = _find_param_vars(func_ir, params)
    
    from .reverse_mode import ReverseModeMutator
    mutator = ReverseModeMutator(param_vars, grad_output)
    backward_ir = mutator.differentiate(func_ir)
    
    if name:
        backward_ir = ir.Function(
            name,
            backward_ir.params,
            backward_ir.return_types,
            backward_ir.body,
            backward_ir.span,
            backward_ir.func_type
        )
    
    return backward_ir


def value_and_grad(
    func: Union[Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> Tuple[Function, Function]:
    """
    Generate forward and backward function pair.
    
    Args:
        func: Forward function
        params: Parameter names for gradients
        name: Base name for functions
    
    Returns:
        (forward_with_tape, backward) tuple
    
    Example:
        >>> forward, backward = pl.value_and_grad(matmul)
        >>> # forward(A, B) -> C, tape
        >>> # backward(tape, d_C) -> (d_A, d_B)
    """
    func_ir = _ensure_ir_function(func)
    
    if params is None:
        params = [p.name_hint for p in func_ir.params]
    
    param_vars = _find_param_vars(func_ir, params)
    
    from .forward_with_tape import ForwardWithTapeMutator
    from .reverse_mode import ReverseModeMutator
    
    forward_mutator = ForwardWithTapeMutator(param_vars)
    forward_ir = forward_mutator.transform(func_ir)
    
    backward_mutator = ReverseModeMutator(param_vars)
    backward_ir = backward_mutator.differentiate(func_ir)
    
    return forward_ir, backward_ir


def jacobian(
    func: Union[Function, Callable],
    params: Optional[List[str]] = None,
    name: Optional[str] = None
) -> Function:
    """
    Generate Jacobian computation function.
    
    For multi-output functions, computes J where J[i,j] = d(output_i)/d(param_j).
    
    Args:
        func: Forward function
        params: Parameter names
        name: Function name
    
    Returns:
        Jacobian function
    
    Example:
        >>> @pl.function
        >>> def f(x):
        >>>     return pl.mul(x, 2), pl.add(x, 1)
        >>> 
        >>> J = pl.jacobian(f)
        >>> # J(x) -> [[2], [1]] (gradient matrix)
    """
    func_ir = _ensure_ir_function(func)
    
    if params is None:
        params = [p.name_hint for p in func_ir.params]
    
    from .jacobian import JacobianBuilder
    builder = JacobianBuilder(func_ir, params)
    return builder.build()


def vjp(
    func: Union[Function, Callable],
    params: Optional[List[str]] = None
) -> Tuple[Function, Callable]:
    """
    Generate Vector-Jacobian Product.
    
    VJP(v, J) = v @ J is the core primitive for reverse-mode AD.
    
    Args:
        func: Forward function
        params: Parameter names
    
    Returns:
        (forward, vjp_closure) tuple
    
    Example:
        >>> forward, vjp_fn = pl.vjp(f)
        >>> output, vjp_closure = forward(x)
        >>> grads = vjp_closure(v)  # computes v @ J
    """
    func_ir = _ensure_ir_function(func)
    
    if params is None:
        params = [p.name_hint for p in func_ir.params]
    
    param_vars = _find_param_vars(func_ir, params)
    
    from .vjp import VJPBuilder
    builder = VJPBuilder(func_ir, param_vars)
    return builder.build()


def stop_gradient(expr: ir.Expr) -> ir.Expr:
    """
    Stop gradient propagation.
    
    The expression value is computed in forward pass,
    but treated as constant during gradient computation.
    
    Args:
        expr: Expression to stop gradient
    
    Returns:
        Wrapped expression
    
    Example:
        >>> @pl.function
        >>> def f(x):
        >>>     x_const = pl.stop_gradient(x)
        >>>     return pl.mul(x, x_const)
        >>> # Only first x contributes to gradient
    """
    return ir.Call(
        ir.Op("autodiff.stop_gradient"),
        [expr],
        {},
        expr.type,
        expr.span
    )


def checkpoint(
    func: Union[Function, Callable],
    checkpoint_policy: str = 'auto'
) -> Function:
    """
    Apply memory-efficient checkpointing.
    
    Args:
        func: Forward function
        checkpoint_policy: 'auto', 'full', 'none', or 'selective'
    
    Returns:
        Checkpointed function
    
    Example:
        >>> @pl.function
        >>> def large_net(x):
        >>>     h1 = layer1(x)
        >>>     h2 = layer2(h1)
        >>>     return h2
        >>> 
        >>> checkpointed = pl.checkpoint(large_net)
        >>> grad_func = pl.grad(checkpointed)  # Uses less memory
    """
    func_ir = _ensure_ir_function(func)
    
    from .checkpoint import Checkpointer
    checkpointer = Checkpointer(func_ir, policy=checkpoint_policy)
    return checkpointer.transform()


def register_grad(op_name: str) -> Callable:
    """
    Decorator to register custom gradient rule.
    
    Args:
        op_name: Operator name (e.g., 'tile.matmul')
    
    Returns:
        Decorator function
    
    Example:
        >>> @pl.register_grad('tile.custom_op')
        >>> def custom_grad(saved_inputs, d_output, kwargs):
        >>>     a, b = saved_inputs
        >>>     return [pl.mul(d_output, b), pl.mul(d_output, a)]
    """
    def decorator(grad_func: Callable) -> Callable:
        from .gradient_registry import GradientRegistry
        GradientRegistry.register(op_name, grad_func)
        return grad_func
    return decorator


# Export GradientRegistry
from .gradient_registry import GradientRegistry


# Helper functions

def _ensure_ir_function(func: Union[Function, Callable]) -> Function:
    """Ensure input is IR Function."""
    if isinstance(func, Function):
        return func
    elif callable(func):
        if hasattr(func, '__pypto_ir__'):
            return func.__pypto_ir__
        raise ValueError("Function must be decorated with @pl.function")
    raise TypeError(f"Expected Function or Callable, got {type(func)}")


def _find_param_vars(func_ir: Function, param_names: List[str]) -> List[ir.Var]:
    """Find parameter IR variables by name."""
    param_vars = []
    for name in param_names:
        for param in func_ir.params:
            if param.name_hint == name:
                param_vars.append(param)
                break
        else:
            raise ValueError(f"Parameter '{name}' not found")
    return param_vars


# =============================================================================
# reverse_mode.py - 反向模式核心
# =============================================================================

"""
核心实现见 05_autodiff_design 文档

关键类:
- ReverseModeMutator: IR 变换器
- GradientAccumulator: 梯度累加管理
- TapeRecord: 前向值记录
"""

# =============================================================================
# gradient_registry.py - 梯度规则注册表
# =============================================================================

from typing import Dict, Callable, Optional, List

class GradientRegistry:
    """Global registry for operator gradient rules."""
    
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, op_name: str, grad_func: Callable):
        """Register gradient rule."""
        cls._registry[op_name] = grad_func
    
    @classmethod
    def get(cls, op_name: str) -> Optional[Callable]:
        """Get gradient rule."""
        return cls._registry.get(op_name)
    
    @classmethod
    def has(cls, op_name: str) -> bool:
        """Check if registered."""
        return op_name in cls._registry
    
    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered ops."""
        return list(cls._registry.keys())


# =============================================================================
# grad_rules.py - 内置梯度规则
# =============================================================================

"""
注册所有内置算子的梯度规则

规则签名:
def grad_rule(saved_inputs: List[ir.Expr], d_output: ir.Expr, kwargs: Dict) -> List[ir.Expr]:
    return [d_input1, d_input2, ...]

已注册算子:
- tile.add, tile.sub, tile.mul, tile.div
- tile.exp, tile.log, tile.sqrt
- tile.relu, tile.sigmoid
- tile.matmul, tile.transpose
- tile.sin, tile.cos
- tensor.add, tensor.mul, ...
"""

# =============================================================================
# custom_grad.py - 自定义梯度
# =============================================================================

from typing import Callable, Optional
from pypto import ir

def CustomGrad(
    forward_func: Callable,
    backward_func: Callable,
    name: Optional[str] = None
) -> ir.Op:
    """
    Create custom gradient operator.
    
    Args:
        forward_func: Forward computation
        backward_func: Gradient computation
        name: Operator name
    
    Returns:
        Custom gradient Op
    
    Example:
        >>> def sigmoid_fwd(x):
        >>>     return pl.div(1, pl.add(1, pl.exp(pl.neg(x))))
        >>> 
        >>> def sigmoid_bwd(x, d_out):
        >>>     s = sigmoid_fwd(x)
        >>>     return pl.mul(d_out, pl.mul(s, pl.sub(1, s)))
        >>> 
        >>> sigmoid = pl.CustomGrad(sigmoid_fwd, sigmoid_bwd)
    """
    if name is None:
        name = f"custom.{forward_func.__name__}"
    
    op = ir.Op(name, {})
    
    # Register gradient rule
    from .gradient_registry import GradientRegistry
    
    def grad_rule(saved_inputs, d_output, kwargs):
        return backward_func(*saved_inputs, d_output)
    
    GradientRegistry.register(name, grad_rule)
    
    return op


# =============================================================================
# checkpoint.py - 内存优化
# =============================================================================

class Checkpointer:
    """
    Apply checkpointing strategy to reduce memory usage.
    
    Strategies:
    - 'full': Save all intermediate values
    - 'none': Recompute everything in backward (slow but minimal memory)
    - 'auto': Balance between memory and computation
    - 'selective': User-specified checkpoint points
    """
    
    def __init__(self, func_ir: ir.Function, policy: str = 'auto'):
        self.func_ir = func_ir
        self.policy = policy
    
    def transform(self) -> ir.Function:
        """Apply checkpointing transformation."""
        if self.policy == 'full':
            return self.func_ir  # No change
        
        if self.policy == 'none':
            # Mark all values for recomputation
            return self._apply_none_policy()
        
        if self.policy == 'auto':
            # Automatically select checkpoint points
            return self._apply_auto_policy()
        
        if self.policy == 'selective':
            return self._apply_selective_policy()
        
        raise ValueError(f"Unknown checkpoint policy: {self.policy}")
    
    def _apply_none_policy(self) -> ir.Function:
        """Recompute everything."""
        # Implementation details...
        pass
    
    def _apply_auto_policy(self) -> ir.Function:
        """Auto-select checkpoints."""
        # Use heuristic: checkpoint every N layers
        pass
    
    def _apply_selective_policy(self) -> ir.Function:
        """User-specified checkpoints."""
        pass


# =============================================================================
# liveness.py - 活性分析
# =============================================================================

"""
分析哪些变量需要梯度、需要保存

关键分析:
1. 从输出反向传播需求
2. 确定需要保存的中间值
3. 计算梯度变量映射
"""

# =============================================================================
# utils.py - 辅助工具
# =============================================================================

def create_grad_var(var: ir.Var, suffix: str = 'd_') -> ir.Var:
    """Create gradient variable from original."""
    return ir.Var(
        f"{suffix}{var.name_hint}",
        var.type,
        var.span
    )


def create_saved_var(var: ir.Var, suffix: str = 'saved_') -> ir.Var:
    """Create saved variable for forward tape."""
    return ir.Var(
        f"{suffix}{var.name_hint}",
        var.type,
        var.span
    )


def is_float_type(type: ir.Type) -> bool:
    """Check if type is float."""
    if isinstance(type, ir.ScalarType):
        return type.dtype.IsFloat()
    if isinstance(type, ir.TileType) or isinstance(type, ir.TensorType):
        return type.dtype.IsFloat()
    return False


def zero_value(type: ir.Type, span: ir.Span) -> ir.Expr:
    """Create zero value for type."""
    if isinstance(type, ir.ScalarType):
        if type.dtype.IsFloat():
            return ir.ConstFloat(0.0, type.dtype, span)
        return ir.ConstInt(0, type.dtype, span)
    
    # For tensor/tile types, create zeros op
    if isinstance(type, ir.TileType):
        return ir.Call(ir.Op("tile.zeros"), [], {"shape": type.shape, "dtype": type.dtype}, type, span)
    
    if isinstance(type, ir.TensorType):
        return ir.Call(ir.Op("tensor.zeros"), [], {"shape": type.shape, "dtype": type.dtype}, type, span)
    
    return ir.ConstFloat(0.0, ir.DataType.FP32, span)