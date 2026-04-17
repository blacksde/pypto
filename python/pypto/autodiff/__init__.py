"""
PyPTO 自动微分模块

提供 source-to-source 自动微分功能
"""

from .gradient_registry import GradientRegistry, register_grad
from .grad_rules import register_all_builtin_rules

# 注册内置梯度规则
register_all_builtin_rules()

# 导出高层 API
from .reverse_mode import grad, value_and_grad

__all__ = [
    "GradientRegistry",
    "register_grad",
    "grad",
    "value_and_grad",
]
