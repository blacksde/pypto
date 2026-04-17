"""
梯度规则注册表包装

用于反向模式构建器访问梯度规则
"""

from .gradient_registry import GradientRegistry


__all__ = ["GradientRegistry"]
