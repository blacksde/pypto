"""
梯度规则注册表

全局存储所有算子的反向梯度规则
"""

from typing import Dict, Callable, Optional, List, Any


class GradientRegistry:
    """
    全局梯度规则注册表

    每个算子的梯度规则是一个 Callable:
        def grad_rule(saved_inputs, d_output, kwargs) -> List[Expr]:
            return [grad_expr1, grad_expr2, ...]

    存储结构:
        op_name -> grad_func
    """

    _registry: Dict[str, Callable] = {}
    _categories: Dict[str, str] = {}

    @classmethod
    def register(cls, op_name: str, grad_func: Callable, category: str = "builtin") -> None:
        """
        注册算子梯度规则

        Args:
            op_name: 算子名称
            grad_func: 梯度计算函数
            category: 分类
        """
        if op_name in cls._registry:
            import warnings

            warnings.warn(f"Overwriting gradient rule for '{op_name}'")

        cls._registry[op_name] = grad_func
        cls._categories[op_name] = category

    @classmethod
    def get(cls, op_name: str) -> Optional[Callable]:
        """获取梯度规则"""
        return cls._registry.get(op_name)

    @classmethod
    def has(cls, op_name: str) -> bool:
        """检查是否已注册"""
        return op_name in cls._registry

    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册算子"""
        return list(cls._registry.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """列出指定分类的算子"""
        return [op for op, cat in cls._categories.items() if cat == category]

    @classmethod
    def unregister(cls, op_name: str) -> bool:
        """移除注册"""
        if op_name in cls._registry:
            del cls._registry[op_name]
            del cls._categories[op_name]
            return True
        return False

    @classmethod
    def get_info(cls, op_name: str) -> Optional[Dict[str, Any]]:
        """获取注册信息"""
        if op_name not in cls._registry:
            return None
        return {
            "op_name": op_name,
            "category": cls._categories[op_name],
            "grad_func": cls._registry[op_name],
        }


def register_grad(op_name: str, category: str = "user"):
    """
    梯度规则注册装饰器

    Args:
        op_name: 算子名称
        category: 分类

    Example:
        @register_grad('tile.add', 'tile')
        def add_grad(saved_inputs, d_output, kwargs):
            return [d_output, d_output]
    """

    def decorator(grad_func: Callable) -> Callable:
        GradientRegistry.register(op_name, grad_func, category)
        return grad_func

    return decorator
