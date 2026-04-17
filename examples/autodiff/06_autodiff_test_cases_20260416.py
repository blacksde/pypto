"""
PyPTO 自动微分测试用例

验证数学正确性的测试：
1. 基本表达式梯度
2. 复合表达式梯度
3. 固定长度循环梯度
4. 变长循环梯度
5. 嵌套循环梯度
6. Phi 节点（条件分支）梯度

注：本测试仅使用数值验证，不依赖 PyPTO 库
"""

import numpy as np
from typing import Tuple

# =============================================================================
# 测试 1: 基本表达式梯度验证
# =============================================================================

class TestBasicGradients:
    """验证基本算术操作的梯度正确性"""
    
    def test_add_gradient(self):
        """
        y = a + b
        
        数学:
          dy/da = 1
          dy/db = 1
        
        数值验证:
          dy/da ≈ (f(a+ε, b) - f(a, b)) / ε
        """
        def forward(a: float, b: float) -> float:
            return a + b
        
        def numerical_grad(a: float, b: float, eps: float = 1e-5) -> Tuple[float, float]:
            y = forward(a, b)
            grad_a = (forward(a + eps, b) - y) / eps
            grad_b = (forward(a, b + eps) - y) / eps
            return grad_a, grad_b
        
        # 测试值
        a_val, b_val = 3.0, 5.0
        
        # 解析梯度
        analytical_grad_a = 1.0
        analytical_grad_b = 1.0
        
        # 数值梯度
        num_grad_a, num_grad_b = numerical_grad(a_val, b_val)
        
        # 验证
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_b, num_grad_b, decimal=4)
        
        print("✓ test_add_gradient 通过")
    
    def test_mul_gradient(self):
        """
        y = a * b
        
        数学:
          dy/da = b
          dy/db = a
        
        反向需要保存前向的 a 和 b 值
        """
        def forward(a: float, b: float) -> float:
            return a * b
        
        def numerical_grad(a: float, b: float, eps: float = 1e-5) -> Tuple[float, float]:
            y = forward(a, b)
            grad_a = (forward(a + eps, b) - y) / eps
            grad_b = (forward(a, b + eps) - y) / eps
            return grad_a, grad_b
        
        a_val, b_val = 3.0, 5.0
        
        # 解析梯度（使用前向值）
        analytical_grad_a = b_val  # dy/da = b
        analytical_grad_b = a_val  # dy/db = a
        
        # 数值梯度
        num_grad_a, num_grad_b = numerical_grad(a_val, b_val)
        
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_b, num_grad_b, decimal=4)
        
        print("✓ test_mul_gradient 通过")
    
    def test_div_gradient(self):
        """
        y = a / b
        
        数学:
          dy/da = 1/b
          dy/db = -a/b²
        
        反向需要保存前向的 a 和 b 值
        """
        def forward(a: float, b: float) -> float:
            return a / b
        
        def numerical_grad(a: float, b: float, eps: float = 1e-5) -> Tuple[float, float]:
            y = forward(a, b)
            grad_a = (forward(a + eps, b) - y) / eps
            grad_b = (forward(a, b + eps) - y) / eps
            return grad_a, grad_b
        
        a_val, b_val = 6.0, 3.0
        
        # 解析梯度
        analytical_grad_a = 1.0 / b_val
        analytical_grad_b = -a_val / (b_val ** 2)
        
        # 数值梯度
        num_grad_a, num_grad_b = numerical_grad(a_val, b_val)
        
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_b, num_grad_b, decimal=4)
        
        print("✓ test_div_gradient 通过")
    
    def test_exp_gradient(self):
        """
        y = exp(a)
        
        数学:
          dy/da = exp(a)
        
        反向可以使用前向结果 y 本身
        """
        def forward(a: float) -> float:
            return np.exp(a)
        
        def numerical_grad(a: float, eps: float = 1e-5) -> float:
            y = forward(a)
            grad = (forward(a + eps) - y) / eps
            return grad
        
        a_val = 2.0
        
        # 解析梯度 = exp(a)
        analytical_grad = np.exp(a_val)
        
        # 数值梯度
        num_grad = numerical_grad(a_val)
        
        np.testing.assert_almost_equal(analytical_grad, num_grad, decimal=4)
        
        print("✓ test_exp_gradient 通过")


# =============================================================================
# 测试 2: 复合表达式梯度
# =============================================================================

class TestCompoundExpressions:
    """验证复合表达式的梯度传播"""
    
    def test_chain_rule(self):
        """
        y = (a * b) + c
        
        链式法则:
          temp = a * b
          y = temp + c
        
          dy/dtemp = 1
          dy/dc = 1
          dtemp/da = b
          dtemp/db = a
        
          dy/da = dy/dtemp * dtemp/da = b
          dy/db = dy/dtemp * dtemp/db = a
          dy/dc = 1
        
        反向遍历:
          从 y 开始，反向传播到 a, b, c
        """
        def forward(a: float, b: float, c: float) -> float:
            temp = a * b
            y = temp + c
            return y
        
        def numerical_grad(a: float, b: float, c: float, eps: float = 1e-5) -> Tuple[float, float, float]:
            y = forward(a, b, c)
            grad_a = (forward(a + eps, b, c) - y) / eps
            grad_b = (forward(a, b + eps, c) - y) / eps
            grad_c = (forward(a, b, c + eps) - y) / eps
            return grad_a, grad_b, grad_c
        
        a_val, b_val, c_val = 2.0, 3.0, 4.0
        
        # 解析梯度
        analytical_grad_a = b_val  # dy/da = b
        analytical_grad_b = a_val  # dy/db = a
        analytical_grad_c = 1.0    # dy/dc = 1
        
        # 数值梯度
        num_grad_a, num_grad_b, num_grad_c = numerical_grad(a_val, b_val, c_val)
        
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_b, num_grad_b, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_c, num_grad_c, decimal=4)
        
        print("✓ test_chain_rule 通过")
    
    def test_nested_composition(self):
        """
        y = ((a + b) * c) - d
        
        分解:
          t1 = a + b
          t2 = t1 * c
          y = t2 - d
        
        梯度:
          dy/dt2 = 1
          dy/dd = -1
          dt2/dt1 = c
          dt2/dc = t1 = a + b
          dt1/da = 1
          dt1/db = 1
        
          dy/da = 1 * c * 1 = c
          dy/db = 1 * c * 1 = c
          dy/dc = 1 * (a + b) = a + b
          dy/dd = -1
        """
        def forward(a: float, b: float, c: float, d: float) -> float:
            return (a + b) * c - d
        
        def numerical_grad(a: float, b: float, c: float, d: float, eps: float = 1e-5):
            y = forward(a, b, c, d)
            return (
                (forward(a + eps, b, c, d) - y) / eps,
                (forward(a, b + eps, c, d) - y) / eps,
                (forward(a, b, c + eps, d) - y) / eps,
                (forward(a, b, c, d + eps) - y) / eps,
            )
        
        a_val, b_val, c_val, d_val = 1.0, 2.0, 3.0, 4.0
        
        # 解析梯度
        analytical_grad = (c_val, c_val, a_val + b_val, -1.0)
        
        # 数值梯度
        num_grad = numerical_grad(a_val, b_val, c_val, d_val)
        
        for i, (an, num) in enumerate(zip(analytical_grad, num_grad)):
            np.testing.assert_almost_equal(an, num, decimal=4)
        
        print("✓ test_nested_composition 通过")


# =============================================================================
# 测试 2.5: 梯度累加（变量多次使用）
# =============================================================================

class TestGradientAccumulation:
    """
    验证变量多次使用时的梯度累加
    
    关键场景：
    - 同一变量在表达式中多次出现
    - 梯度需要累加所有贡献路径
    - 循环中变量多次更新
    """
    
    def test_simple_reuse(self):
        """
        y = a * b + a * c
        
        a 被使用两次，梯度需要累加：
        
        数学:
          y = a * b + a * c = a * (b + c)
        
          dy/da = b + c  (累加两个贡献)
          dy/db = a
          dy/dc = a
        
        反向传播:
          ȳ -> ā_1 = ȳ * b (来自 a * b)
          ȳ -> ā_2 = ȳ * c (来自 a * c)
          ā = ā_1 + ā_2 = ȳ * (b + c)
        
        关键: 梯度累加器必须正确合并多个贡献
        """
        def forward(a: float, b: float, c: float) -> float:
            return a * b + a * c
        
        def numerical_grad(a: float, b: float, c: float, eps: float = 1e-5) -> Tuple[float, float, float]:
            y = forward(a, b, c)
            grad_a = (forward(a + eps, b, c) - y) / eps
            grad_b = (forward(a, b + eps, c) - y) / eps
            grad_c = (forward(a, b, c + eps) - y) / eps
            return grad_a, grad_b, grad_c
        
        a_val, b_val, c_val = 2.0, 3.0, 4.0
        
        # 解析梯度（累加）
        analytical_grad_a = b_val + c_val  # 累加两个贡献
        analytical_grad_b = a_val
        analytical_grad_c = a_val
        
        # 数值梯度
        num_grad_a, num_grad_b, num_grad_c = numerical_grad(a_val, b_val, c_val)
        
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_b, num_grad_b, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_c, num_grad_c, decimal=4)
        
        print(f"✓ test_simple_reuse 通过 (a 梯度累加: {b_val} + {c_val} = {analytical_grad_a})")
    
    def test_complex_reuse(self):
        """
        y = a * a + a * b
        
        a 被使用三次（两次在 a*a，一次在 a*b）
        
        数学:
          y = a² + a * b
        
          dy/da = 2a + b  (来自 a² 的 2a + 来自 a*b 的 b)
          dy/db = a
        
        反向传播细节:
          term1 = a * a:
            ā += ȳ * a (第一个 a)
            ā += ȳ * a (第二个 a)
            共贡献: 2 * ȳ * a
          
          term2 = a * b:
            ā += ȳ * b
          
          总: ā = 2a + b
        """
        def forward(a: float, b: float) -> float:
            return a * a + a * b
        
        def numerical_grad(a: float, b: float, eps: float = 1e-5) -> Tuple[float, float]:
            y = forward(a, b)
            grad_a = (forward(a + eps, b) - y) / eps
            grad_b = (forward(a, b + eps) - y) / eps
            return grad_a, grad_b
        
        a_val, b_val = 3.0, 2.0
        
        # 解析梯度
        analytical_grad_a = 2 * a_val + b_val  # 2a 来自 a²，b 来自 a*b
        analytical_grad_b = a_val
        
        # 数值梯度
        num_grad_a, num_grad_b = numerical_grad(a_val, b_val)
        
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_b, num_grad_b, decimal=4)
        
        print(f"✓ test_complex_reuse 通过 (a 梯度累加: 2*{a_val} + {b_val} = {analytical_grad_a})")
    
    def test_deep_reuse(self):
        """
        y = (a + b) * (a + c)
        
        a 在两个加法中都使用
        
        数学:
          t1 = a + b
          t2 = a + c
          y = t1 * t2
        
          dy/dt1 = t2 = a + c
          dy/dt2 = t1 = a + b
        
          dt1/da = 1, dt2/da = 1
        
          dy/da = (a + c) + (a + b) = 2a + b + c (累加)
          dy/db = a + c
          dy/dc = a + b
        
        反向:
          ā += ȳ * (a + c) (来自 t1 路径)
          ā += ȳ * (a + b) (来自 t2 路径)
        """
        def forward(a: float, b: float, c: float) -> float:
            t1 = a + b
            t2 = a + c
            return t1 * t2
        
        def numerical_grad(a: float, b: float, c: float, eps: float = 1e-5) -> Tuple[float, float, float]:
            y = forward(a, b, c)
            grad_a = (forward(a + eps, b, c) - y) / eps
            grad_b = (forward(a, b + eps, c) - y) / eps
            grad_c = (forward(a, b, c + eps) - y) / eps
            return grad_a, grad_b, grad_c
        
        a_val, b_val, c_val = 2.0, 3.0, 4.0
        
        # 解析梯度
        analytical_grad_a = (a_val + c_val) + (a_val + b_val)  # 累加
        analytical_grad_b = a_val + c_val
        analytical_grad_c = a_val + b_val
        
        # 数值梯度
        num_grad_a, num_grad_b, num_grad_c = numerical_grad(a_val, b_val, c_val)
        
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_b, num_grad_b, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_c, num_grad_c, decimal=4)
        
        print(f"✓ test_deep_reuse 通过 (a 梯度累加: {a_val+c_val} + {a_val+b_val} = {analytical_grad_a})")
    
    def test_loop_accumulation(self):
        """
        acc = 0
        for i = 0 to N:
          acc = acc + x
          temp = x * i  # x 再次使用
          acc = acc + temp
        
        x 在每次循环中被使用两次
        
        数学:
          acc_N = Σ_{i=0}^{N-1} (x + x*i) = Σ_{i=0}^{N-1} x*(1+i) = x * Σ_{i=0}^{N-1} (i+1) = x * N*(N+1)/2
        
          dacc/dx = N*(N+1)/2
        
        反向:
          每次迭代 x 收到两个梯度贡献:
          - 来自 acc + x: 贡献 1
          - 来自 x * i: 贡献 i
          总贡献 = 1 + i
          
          累加所有迭代: Σ_{i=0}^{N-1} (1+i) = N*(N+1)/2
        """
        def forward(x: float, N: int) -> float:
            acc = 0.0
            for i in range(N):
                acc = acc + x           # 第一次使用 x
                temp = x * i            # 第二次使用 x
                acc = acc + temp        # 累加 temp
            return acc
        
        def numerical_grad(x: float, N: int, eps: float = 1e-5) -> float:
            y = forward(x, N)
            grad = (forward(x + eps, N) - y) / eps
            return grad
        
        x_val = 1.0
        N = 5
        
        # 解析梯度: x * N*(N+1)/2
        analytical_grad = N * (N + 1) // 2
        
        # 数值梯度
        num_grad = numerical_grad(x_val, N)
        
        np.testing.assert_almost_equal(analytical_grad, num_grad, decimal=4)
        
        print(f"✓ test_loop_accumulation 通过 (x 梯度累加: Σ(1+i) = {analytical_grad})")
    
    def test_cross_statement_accumulation(self):
        """
        跨语句的梯度累加
        
        y1 = a * b
        y2 = a * c
        y = y1 + y2
        
        a 在两个语句中使用
        
        数学:
          dy/da = dy/dy1 * dy1/da + dy/dy2 * dy2/da
                 = 1 * b + 1 * c = b + c
        
        这是反向模式中典型的跨语句累加场景：
        反向遍历时，先处理 y = y1 + y2，再处理 y1 和 y2
        梯度通过累加器合并
        """
        def forward(a: float, b: float, c: float) -> float:
            y1 = a * b
            y2 = a * c
            y = y1 + y2
            return y
        
        def numerical_grad(a: float, b: float, c: float, eps: float = 1e-5) -> Tuple[float, float, float]:
            y = forward(a, b, c)
            grad_a = (forward(a + eps, b, c) - y) / eps
            grad_b = (forward(a, b + eps, c) - y) / eps
            grad_c = (forward(a, b, c + eps) - y) / eps
            return grad_a, grad_b, grad_c
        
        a_val, b_val, c_val = 2.0, 3.0, 4.0
        
        # 解析梯度
        analytical_grad_a = b_val + c_val
        analytical_grad_b = a_val
        analytical_grad_c = a_val
        
        # 数值梯度
        num_grad_a, num_grad_b, num_grad_c = numerical_grad(a_val, b_val, c_val)
        
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_b, num_grad_b, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_c, num_grad_c, decimal=4)
        
        print(f"✓ test_cross_statement_accumulation 通过")
    
    def test_multiple_output_accumulation(self):
        """
        多输出场景的梯度累加
        
        y1 = a * b
        y2 = a * c
        
        求两个输出对 a 的梯度之和
        
        数学:
          dy1/da = b
          dy2/da = c
          dy1+dy2/da = b + c
        
        这对应于多任务学习场景，一个参数影响多个输出
        """
        def forward(a: float, b: float, c: float) -> Tuple[float, float]:
            y1 = a * b
            y2 = a * c
            return y1, y2
        
        def numerical_grad_sum(a: float, b: float, c: float, eps: float = 1e-5) -> float:
            y1, y2 = forward(a, b, c)
            y_sum = y1 + y2
            y1_plus, y2_plus = forward(a + eps, b, c)
            y_sum_plus = y1_plus + y2_plus
            grad_a = (y_sum_plus - y_sum) / eps
            return grad_a
        
        a_val, b_val, c_val = 2.0, 3.0, 4.0
        
        # 解析梯度（两个输出的梯度之和）
        analytical_grad_a = b_val + c_val
        
        # 数值梯度
        num_grad_a = numerical_grad_sum(a_val, b_val, c_val)
        
        np.testing.assert_almost_equal(analytical_grad_a, num_grad_a, decimal=4)
        
        print(f"✓ test_multiple_output_accumulation 通过")


# =============================================================================
# 测试 3: 固定长度循环梯度
# =============================================================================

class TestFixedLoopGradients:
    """验证固定长度 SSA 循环的梯度正确性"""
    
    def test_simple_accumulate(self):
        """
        acc = 0
        for i = 0 to N:
          acc = acc + x
        
        结果: acc = N * x
        
        梯度:
          dacc/dx = N
        
        反向循环:
          for i = N-1 to 0:
            dx += dacc_prev * 1
            dacc_prev = dacc_current
        """
        def forward(x: float, N: int) -> float:
            acc = 0.0
            for i in range(N):
                acc = acc + x
            return acc
        
        def numerical_grad(x: float, N: int, eps: float = 1e-5) -> float:
            y = forward(x, N)
            grad = (forward(x + eps, N) - y) / eps
            return grad
        
        x_val = 5.0
        N = 10
        
        # 解析梯度
        analytical_grad = N
        
        # 数值梯度
        num_grad = numerical_grad(x_val, N)
        
        np.testing.assert_almost_equal(analytical_grad, num_grad, decimal=4)
        
        print("✓ test_simple_accumulate 通过")
    
    def test_weighted_accumulate(self):
        """
        acc = 0
        for i = 0 to N:
          acc = acc + x * i
        
        结果: acc = x * Σ_{i=0}^{N-1} i = x * N*(N-1)/2
        
        梯度:
          dacc/dx = Σ_{i=0}^{N-1} i = N*(N-1)/2
        
        反向:
          每次迭代贡献 x 的梯度 = i
        """
        def forward(x: float, N: int) -> float:
            acc = 0.0
            for i in range(N):
                acc = acc + x * i
            return acc
        
        def numerical_grad(x: float, N: int, eps: float = 1e-5) -> float:
            y = forward(x, N)
            grad = (forward(x + eps, N) - y) / eps
            return grad
        
        x_val = 2.0
        N = 10
        
        # 解析梯度
        analytical_grad = N * (N - 1) // 2
        
        # 数值梯度
        num_grad = numerical_grad(x_val, N)
        
        np.testing.assert_almost_equal(analytical_grad, num_grad, decimal=4)
        
        print("✓ test_weighted_accumulate 通过")
    
    def test_multiplicative_loop(self):
        """
        acc = 1
        for i = 0 to N:
          acc = acc * x
        
        结果: acc = x^N
        
        梯度:
          dacc/dx = N * x^(N-1)
        
        反向:
          每次迭代贡献梯度，需要累积
        """
        def forward(x: float, N: int) -> float:
            acc = 1.0
            for i in range(N):
                acc = acc * x
            return acc
        
        def numerical_grad(x: float, N: int, eps: float = 1e-5) -> float:
            y = forward(x, N)
            grad = (forward(x + eps, N) - y) / eps
            return grad
        
        x_val = 2.0
        N = 5
        
        # 解析梯度: d(x^N)/dx = N * x^(N-1)
        analytical_grad = N * (x_val ** (N - 1))
        
        # 数值梯度
        num_grad = numerical_grad(x_val, N)
        
        np.testing.assert_almost_equal(analytical_grad, num_grad, decimal=3)
        
        print("✓ test_multiplicative_loop 通过")


# =============================================================================
# 测试 4: 变长循环梯度
# =============================================================================

class TestDynamicLoopGradients:
    """验证动态迭代次数的循环梯度"""
    
    def test_dynamic_counter(self):
        """
        acc = 0
        for i = 0 to N (动态):
          acc = acc + x
        
        结果: acc = N * x
        
        梯度:
          dacc/dx = N
          dacc/dN = x (如果 N 是输入)
        
        反向使用相同的 N 值迭代
        """
        def forward(x: float, N: int) -> float:
            acc = 0.0
            for i in range(N):
                acc = acc + x
            return acc
        
        def numerical_grad_x(x: float, N: int, eps: float = 1e-5) -> float:
            y = forward(x, N)
            grad = (forward(x + eps, N) - y) / eps
            return grad
        
        def numerical_grad_N(x: float, N: int, eps: int = 1) -> float:
            y = forward(x, N)
            grad = (forward(x, N + eps) - y) / eps
            return grad
        
        x_val = 3.0
        N_val = 7
        
        # 解析梯度
        analytical_grad_x = N_val
        analytical_grad_N = x_val
        
        # 数值梯度
        num_grad_x = numerical_grad_x(x_val, N_val)
        num_grad_N = numerical_grad_N(x_val, N_val)
        
        np.testing.assert_almost_equal(analytical_grad_x, num_grad_x, decimal=4)
        np.testing.assert_almost_equal(analytical_grad_N, num_grad_N, decimal=4)
        
        print("✓ test_dynamic_counter 通过")
    
    def test_dynamic_conditional_loop(self):
        """
        while acc < threshold:
          acc = acc + x
        
        循环次数取决于阈值
        
        梯度需要反向遍历，迭代次数由前向确定
        """
        def forward(x: float, threshold: float) -> Tuple[float, int]:
            acc = 0.0
            iterations = 0
            while acc < threshold:
                acc = acc + x
                iterations += 1
            return acc, iterations
        
        def numerical_grad_x(x: float, threshold: float, eps: float = 1e-5) -> float:
            y, N = forward(x, threshold)
            y_plus, _ = forward(x + eps, threshold)
            # 注意：增加 x 可能改变迭代次数
            grad = (y_plus - y) / eps
            return grad
        
        x_val = 1.0
        threshold = 10.0
        
        y, N = forward(x_val, threshold)
        
        # 解析梯度（假设迭代次数不变）
        analytical_grad = N
        
        # 数值梯度（可能因迭代次数变化而有误差）
        num_grad = numerical_grad_x(x_val, threshold)
        
        # 使用较大的容差（迭代次数可能变化）
        np.testing.assert_almost_equal(analytical_grad, num_grad, decimal=1)
        
        print(f"✓ test_dynamic_conditional_loop 通过 (N={N}, grad_x={num_grad:.2f})")


# =============================================================================
# 测试 5: 嵌套循环梯度
# =============================================================================

class TestNestedLoopGradients:
    """验证嵌套循环的梯度正确性"""
    
    def test_nested_sum(self):
        """
        result = 0
        for i = 0 to M:
          for j = 0 to N:
            result = result + x
        
        结果: result = M * N * x
        
        梯度:
          dresult/dx = M * N
        
        反向需要两层反向循环
        """
        def forward(x: float, M: int, N: int) -> float:
            result = 0.0
            for i in range(M):
                for j in range(N):
                    result = result + x
            return result
        
        def numerical_grad(x: float, M: int, N: int, eps: float = 1e-5) -> float:
            y = forward(x, M, N)
            grad = (forward(x + eps, M, N) - y) / eps
            return grad
        
        x_val = 1.5
        M, N = 3, 4
        
        # 解析梯度
        analytical_grad = M * N
        
        # 数值梯度
        num_grad = numerical_grad(x_val, M, N)
        
        np.testing.assert_almost_equal(analytical_grad, num_grad, decimal=4)
        
        print("✓ test_nested_sum 通过")
    
    def test_nested_dependency(self):
        """
        result = x
        for i = 0 to M:
          for j = 0 to N:
            result = result * 2
        
        结果: result = x * 2^(M*N)
        
        梯度:
          dresult/dx = 2^(M*N)
        
        反向:
          外层循环梯度乘以内层累积因子
        """
        def forward(x: float, M: int, N: int) -> float:
            result = x
            for i in range(M):
                for j in range(N):
                    result = result * 2
            return result
        
        def numerical_grad(x: float, M: int, N: int, eps: float = 1e-5) -> float:
            y = forward(x, M, N)
            grad = (forward(x + eps, M, N) - y) / eps
            return grad
        
        x_val = 1.0
        M, N = 2, 3
        
        # 解析梯度
        analytical_grad = 2 ** (M * N)
        
        # 数值梯度
        num_grad = numerical_grad(x_val, M, N)
        
        np.testing.assert_almost_equal(analytical_grad, num_grad, decimal=3)
        
        print(f"✓ test_nested_dependency 通过 (grad={num_grad:.2f})")


# =============================================================================
# 测试 6: Phi 节点（条件分支）梯度
# =============================================================================

class TestPhiNodeGradients:
    """验证条件分支（Phi 节点）的梯度正确性"""
    
    def test_simple_branch(self):
        """
        if flag:
          y = x * 2
        else:
          y = x + 1
        
        梯度:
          if flag:
            dy/dx = 2
          else:
            dy/dx = 1
        
        反向使用前向保存的条件值
        """
        def forward(x: float, flag: bool) -> float:
            if flag:
                return x * 2
            else:
                return x + 1
        
        def numerical_grad(x: float, flag: bool, eps: float = 1e-5) -> float:
            y = forward(x, flag)
            grad = (forward(x + eps, flag) - y) / eps
            return grad
        
        # 测试 True 分支
        x_val = 5.0
        analytical_grad_true = 2.0
        num_grad_true = numerical_grad(x_val, True)
        np.testing.assert_almost_equal(analytical_grad_true, num_grad_true, decimal=4)
        
        # 测试 False 分支
        analytical_grad_false = 1.0
        num_grad_false = numerical_grad(x_val, False)
        np.testing.assert_almost_equal(analytical_grad_false, num_grad_false, decimal=4)
        
        print("✓ test_simple_branch 通过")
    
    def test_branch_with_loop(self):
        """
        for i = 0 to N:
          if flag:
            acc = acc + x * 2
          else:
            acc = acc + x
        
        梯度:
          if flag:
            dacc/dx = 2 * N
          else:
            dacc/dx = N
        
        Phi 节点在循环内，反向需要处理
        """
        def forward(x: float, flag: bool, N: int) -> float:
            acc = 0.0
            for i in range(N):
                if flag:
                    acc = acc + x * 2
                else:
                    acc = acc + x
            return acc
        
        def numerical_grad(x: float, flag: bool, N: int, eps: float = 1e-5) -> float:
            y = forward(x, flag, N)
            grad = (forward(x + eps, flag, N) - y) / eps
            return grad
        
        x_val = 1.0
        N = 5
        
        # True 分支
        analytical_grad_true = 2 * N
        num_grad_true = numerical_grad(x_val, True, N)
        np.testing.assert_almost_equal(analytical_grad_true, num_grad_true, decimal=4)
        
        # False 分支
        analytical_grad_false = N
        num_grad_false = numerical_grad(x_val, False, N)
        np.testing.assert_almost_equal(analytical_grad_false, num_grad_false, decimal=4)
        
        print("✓ test_branch_with_loop 通过")
    
    def test_nested_phi(self):
        """
        if flag1:
          if flag2:
            y = x * 4
          else:
            y = x * 2
        else:
          y = x
        
        梯度取决于所有条件的组合
        """
        def forward(x: float, flag1: bool, flag2: bool) -> float:
            if flag1:
                if flag2:
                    return x * 4
                else:
                    return x * 2
            else:
                return x
        
        def numerical_grad(x: float, flag1: bool, flag2: bool, eps: float = 1e-5) -> float:
            y = forward(x, flag1, flag2)
            grad = (forward(x + eps, flag1, flag2) - y) / eps
            return grad
        
        x_val = 1.0
        
        # 所有组合
        cases = [
            (True, True, 4.0),
            (True, False, 2.0),
            (False, True, 1.0),  # flag2 无效
            (False, False, 1.0),
        ]
        
        for flag1, flag2, expected_grad in cases:
            num_grad = numerical_grad(x_val, flag1, flag2)
            np.testing.assert_almost_equal(expected_grad, num_grad, decimal=4)
        
        print("✓ test_nested_phi 通过")


# =============================================================================
# 测试 7: 矩阵操作梯度
# =============================================================================

class TestMatrixGradients:
    """验证矩阵/瓦片操作的梯度"""
    
    def test_matmul_gradient(self):
        """
        C = A @ B
        
        梯度:
          dA = dC @ B.T
          dB = A.T @ dC
        
        需要保存前向的 A 和 B
        """
        A = np.random.randn(4, 3)
        B = np.random.randn(3, 5)
        dC = np.ones((4, 5))
        
        # 前向
        C = A @ B
        
        # 解析梯度
        dA_analytical = dC @ B.T
        dB_analytical = A.T @ dC
        
        # 数值梯度
        eps = 1e-5
        
        dA_numerical = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_plus = A.copy()
                A_plus[i, j] += eps
                C_plus = A_plus @ B
                dA_numerical[i, j] = np.sum((C_plus - C) * dC) / eps
        
        dB_numerical = np.zeros_like(B)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B_plus = B.copy()
                B_plus[i, j] += eps
                C_plus = A @ B_plus
                dB_numerical[i, j] = np.sum((C_plus - C) * dC) / eps
        
        np.testing.assert_almost_equal(dA_analytical, dA_numerical, decimal=3)
        np.testing.assert_almost_equal(dB_analytical, dB_numerical, decimal=3)
        
        print("✓ test_matmul_gradient 通过")


# =============================================================================
# 运行所有测试
# =============================================================================

def run_all_tests():
    """运行所有梯度验证测试"""
    print("=" * 60)
    print("PyPTO 自动微分数学正确性验证")
    print("=" * 60)
    
    # 基本表达式
    print("\n[1] 基本表达式梯度")
    test_basic = TestBasicGradients()
    test_basic.test_add_gradient()
    test_basic.test_mul_gradient()
    test_basic.test_div_gradient()
    test_basic.test_exp_gradient()
    
    # 复合表达式
    print("\n[2] 复合表达式梯度")
    test_compound = TestCompoundExpressions()
    test_compound.test_chain_rule()
    test_compound.test_nested_composition()
    
    # 梯度累加（变量多次使用）
    print("\n[2.5] 梯度累加（变量多次使用）")
    test_accum = TestGradientAccumulation()
    test_accum.test_simple_reuse()
    test_accum.test_complex_reuse()
    test_accum.test_deep_reuse()
    test_accum.test_loop_accumulation()
    test_accum.test_cross_statement_accumulation()
    test_accum.test_multiple_output_accumulation()
    
    # 固定长度循环
    print("\n[3] 固定长度循环梯度")
    test_fixed = TestFixedLoopGradients()
    test_fixed.test_simple_accumulate()
    test_fixed.test_weighted_accumulate()
    test_fixed.test_multiplicative_loop()
    
    # 变长循环
    print("\n[4] 变长循环梯度")
    test_dynamic = TestDynamicLoopGradients()
    test_dynamic.test_dynamic_counter()
    test_dynamic.test_dynamic_conditional_loop()
    
    # 嵌套循环
    print("\n[5] 嵌套循环梯度")
    test_nested = TestNestedLoopGradients()
    test_nested.test_nested_sum()
    test_nested.test_nested_dependency()
    
    # Phi 节点
    print("\n[6] Phi 节点梯度")
    test_phi = TestPhiNodeGradients()
    test_phi.test_simple_branch()
    test_phi.test_branch_with_loop()
    test_phi.test_nested_phi()
    
    # 矩阵操作
    print("\n[7] 矩阵操作梯度")
    test_matrix = TestMatrixGradients()
    test_matrix.test_matmul_gradient()
    
    print("\n" + "=" * 60)
    print("所有测试通过！自动微分方案数学正确性验证成功")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()