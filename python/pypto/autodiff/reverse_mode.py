"""
PyPTO 自动微分核心实现

实现 pl.grad, pl.value_and_grad 等高层 API
"""

from typing import Union, List, Optional, Dict, Any, Callable, Tuple
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Span,
    Function,
    Var,
    Type,
    Expr,
    AssignStmt,
    ReturnStmt,
    SeqStmts,
    Call,
    Op,
    ConstInt,
    ConstFloat,
    ScalarType,
    TileType,
    TensorType,
    UnknownType,
    Add,
    Mul,
    Neg,
    ForStmt,
    IfStmt,
    YieldStmt,
    IterArg,
    ForKind,
)
from pypto.ir.printer import python_print
from pypto.ir.utils import _to_make_tuple

ir = _ir_core


class ReverseModeBuilder:
    """
    反向模式梯度函数构建器

    从前向函数 IR 构建反向梯度函数 IR
    """

    def __init__(self, forward_func: ir.Function, grad_params: List[str], grad_output: Optional[str] = None):
        self.forward_func = forward_func
        self.grad_params = grad_params
        self.grad_output = grad_output

        # 梯度变量映射
        self.grad_var_map: Dict[ir.Var, ir.Var] = {}

        # 保存的前向值
        self.saved_values: Dict[ir.Var, ir.Var] = {}

        # 累积梯度
        self.gradient_accumulator: Dict[ir.Var, List[ir.Expr]] = {}

        # Span
        self.span = forward_func.span

    def build(self) -> ir.Function:
        """
        构建反向梯度函数

        Returns:
            反向梯度函数 IR
        """
        # 1. 创建梯度参数（前向参数 + 梯度种子）
        backward_params = list(self.forward_func.params)

        # 添加梯度种子参数
        if self.grad_output is None:
            # 默认：第一个输出的梯度
            output_type = self.forward_func.return_types[0]
            d_output = ir.Var("d_output", output_type, self.span)
            backward_params.append(d_output)
        else:
            # TODO: 支持指定输出
            output_type = self.forward_func.return_types[0]
            d_output = ir.Var("d_output", output_type, self.span)
            backward_params.append(d_output)

        # 2. 创建梯度变量
        param_vars = {p.name_hint: p for p in self.forward_func.params}
        for param_name in self.grad_params:
            if param_name in param_vars:
                param_var = param_vars[param_name]
                grad_var = ir.Var(f"d_{param_name}", param_var.type, self.span)
                self.grad_var_map[param_var] = grad_var

        # 3. 构建反向语句
        backward_stmts = self._build_backward_stmts(d_output)

        # 4. 收集反向中引用的正向中间值并生成计算语句
        forward_saved_vars = self._collect_saved_forward_vars(backward_stmts)

        # 不将正向中间值作为参数，而是在反向函数内部重新计算它们
        forward_compute_stmts = self._generate_forward_compute_stmts(forward_saved_vars)

        # 5. 将正向计算语句插入到反向语句之前
        backward_stmts = forward_compute_stmts + backward_stmts

        # 6. 构建返回语句
        return_vars = []
        for param_name in self.grad_params:
            if param_name in param_vars:
                param_var = param_vars[param_name]
                if param_var in self.grad_var_map:
                    return_vars.append(self.grad_var_map[param_var])

        # 添加返回语句
        if return_vars:
            backward_stmts.append(ir.ReturnStmt(return_vars, self.span))

        # 7. 构建函数体
        backward_body = ir.SeqStmts(backward_stmts, self.span)

        # 8. 构建返回类型
        backward_return_types = []
        for param_name in self.grad_params:
            if param_name in param_vars:
                backward_return_types.append(param_vars[param_name].type)

        # 9. 创建反向函数
        backward_func = ir.Function(
            f"grad_{self.forward_func.name}", backward_params, backward_return_types, backward_body, self.span
        )

        return backward_func

    def _collect_saved_forward_vars(self, stmts: List[ir.Stmt]) -> List[ir.Var]:
        """
        收集反向语句中引用的正向中间值（包括循环内变量）

        Args:
            stmts: 反向语句列表

        Returns:
            需要保存的正向变量列表（包含类型信息）
        """
        saved_var_names = set()
        param_names = {p.name_hint for p in self.forward_func.params}
        grad_var_names = {gv.name_hint for gv in self.grad_var_map.values()}
        allowed_var_names = param_names | grad_var_names | {"d_output", "d_y"}
        forward_var_types = {}

        def collect_forward_var_types(stmt: ir.Stmt):
            if isinstance(stmt, ir.AssignStmt):
                if isinstance(stmt.var, ir.Var):
                    forward_var_types[stmt.var.name_hint] = stmt.var.type
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    collect_forward_var_types(s)
            elif isinstance(stmt, ir.ForStmt):
                collect_forward_var_types(stmt.body)
            elif isinstance(stmt, ir.IfStmt):
                collect_forward_var_types(stmt.then_body)
                if stmt.else_body:
                    collect_forward_var_types(stmt.else_body)

        collect_forward_var_types(self.forward_func.body)

        def collect_from_expr(expr: ir.Expr):
            if isinstance(expr, ir.Var):
                var_name = expr.name_hint
                if var_name not in allowed_var_names and var_name not in saved_var_names:
                    if var_name in forward_var_types:
                        saved_var_names.add(var_name)
            elif isinstance(expr, ir.Call):
                for arg in expr.args:
                    collect_from_expr(arg)
            elif isinstance(expr, ir.Add) or isinstance(expr, ir.Mul):
                collect_from_expr(expr.lhs)
                collect_from_expr(expr.rhs)
            elif isinstance(expr, ir.Neg):
                collect_from_expr(expr.operand)

        def collect_from_stmt(stmt: ir.Stmt):
            if isinstance(stmt, ir.AssignStmt):
                collect_from_expr(stmt.value)
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    collect_from_stmt(s)
            elif isinstance(stmt, ir.ForStmt):
                collect_from_stmt(stmt.body)
            elif isinstance(stmt, ir.IfStmt):
                collect_from_stmt(stmt.then_body)
                if stmt.else_body:
                    collect_from_stmt(stmt.else_body)
            elif isinstance(stmt, ir.ReturnStmt):
                for v in stmt.value:
                    collect_from_expr(v)

        for stmt in stmts:
            collect_from_stmt(stmt)

        saved_vars = []
        for name in saved_var_names:
            if name in forward_var_types:
                saved_vars.append(ir.Var(name, forward_var_types[name], self.span))

        return saved_vars

    def _generate_forward_compute_stmts(self, saved_vars: List[ir.Var]) -> List[ir.Stmt]:
        """
        为反向函数生成计算正向中间值的语句

        只处理循环外定义的变量，循环内变量由 Tape 模式处理

        Args:
            saved_vars: 需要重新计算的中间变量列表

        Returns:
            计算中间变量的语句列表
        """
        if not saved_vars:
            return []

        saved_var_names = {v.name_hint for v in saved_vars}
        param_names = {p.name_hint for p in self.forward_func.params}

        loop_defined_vars = set()

        def collect_loop_vars(stmt: ir.Stmt):
            if isinstance(stmt, ir.ForStmt):

                def collect_body(body_stmt):
                    if isinstance(body_stmt, ir.AssignStmt):
                        if isinstance(body_stmt.var, ir.Var):
                            loop_defined_vars.add(body_stmt.var.name_hint)
                    elif isinstance(body_stmt, ir.SeqStmts):
                        for s in body_stmt.stmts:
                            collect_body(s)

                collect_body(stmt.body)
                collect_loop_vars(stmt.body)
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    collect_loop_vars(s)
            elif isinstance(stmt, ir.IfStmt):
                collect_loop_vars(stmt.then_body)
                if stmt.else_body:
                    collect_loop_vars(stmt.else_body)

        collect_loop_vars(self.forward_func.body)

        non_loop_vars = saved_var_names - loop_defined_vars
        forward_assignments: Dict[str, ir.AssignStmt] = {}

        def collect_forward_assignments(stmt: ir.Stmt, in_loop: bool = False):
            if isinstance(stmt, ir.AssignStmt):
                if isinstance(stmt.var, ir.Var):
                    var_name = stmt.var.name_hint
                    if var_name in non_loop_vars and not in_loop:
                        forward_assignments[var_name] = stmt
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    collect_forward_assignments(s, in_loop)
            elif isinstance(stmt, ir.ForStmt):
                collect_forward_assignments(stmt.body, in_loop=True)
            elif isinstance(stmt, ir.IfStmt):
                collect_forward_assignments(stmt.then_body, in_loop)
                if stmt.else_body:
                    collect_forward_assignments(stmt.else_body, in_loop)

        collect_forward_assignments(self.forward_func.body)

        def get_dependencies(expr: ir.Expr) -> set:
            deps = set()
            if isinstance(expr, ir.Var):
                var_name = expr.name_hint
                if var_name not in param_names and var_name in non_loop_vars:
                    deps.add(var_name)
            elif isinstance(expr, ir.Call):
                for arg in expr.args:
                    deps.update(get_dependencies(arg))
            elif isinstance(expr, ir.Add) or isinstance(expr, ir.Mul):
                deps.update(get_dependencies(expr.lhs))
                deps.update(get_dependencies(expr.rhs))
            elif isinstance(expr, ir.Neg):
                deps.update(get_dependencies(expr.operand))
            return deps

        ordered_stmts: List[ir.Stmt] = []
        processed_vars = set(param_names)

        remaining_vars = non_loop_vars - param_names
        while len(processed_vars) < len(param_names) + len(remaining_vars):
            progress = False
            for var_name in remaining_vars:
                if var_name in processed_vars:
                    continue
                stmt = forward_assignments.get(var_name)
                if stmt is None:
                    processed_vars.add(var_name)
                    progress = True
                    continue
                deps = get_dependencies(stmt.value)
                if all(d in processed_vars for d in deps):
                    ordered_stmts.append(stmt)
                    processed_vars.add(var_name)
                    progress = True
            if not progress:
                break

        return ordered_stmts

    def _build_backward_stmts(self, d_output: ir.Var) -> List[ir.Stmt]:
        """
        构建反向语句列表

        Args:
            d_output: 输出梯度种子

        Returns:
            反向语句列表
        """
        stmts = []

        # 初始化梯度变量为零（用于循环内累加）
        # 只为需要在循环内累加的参数初始化
        for param_var, grad_var in self.grad_var_map.items():
            # 判断是否需要初始化：如果在循环内累加，则需要初始化为零
            # 对于 iter_args + yield 模式，init 的梯度直接赋值，不需要初始化
            forward_body = self.forward_func.body
            needs_init = self._param_needs_init_in_loop(param_var, forward_body)
            if needs_init:
                zero = self._create_zero(param_var.type)
                stmts.append(ir.AssignStmt(grad_var, zero, self.span))

        # 反向遍历前向函数体（使用内联版本直接生成反向语句）
        forward_body = self.forward_func.body
        if isinstance(forward_body, ir.SeqStmts):
            for stmt in reversed(forward_body.stmts):
                stmts.extend(self._reverse_stmt_inline(stmt, d_output))
        elif isinstance(forward_body, ir.AssignStmt):
            stmts.extend(self._reverse_assign_inline(forward_body, d_output))
        elif isinstance(forward_body, ir.ForStmt):
            stmts.extend(self._reverse_for(forward_body, d_output))
        elif isinstance(forward_body, ir.IfStmt):
            stmts.extend(self._reverse_if_inline(forward_body, d_output))
        elif isinstance(forward_body, ir.YieldStmt):
            stmts.extend(self._reverse_yield_inline(forward_body, d_output))

        return stmts

    def _param_needs_init_in_loop(self, param_var: ir.Var, body: ir.Stmt) -> bool:
        """
        判断参数是否需要在循环内初始化为零

        如果参数在循环内被多次使用（需要累加梯度），则需要初始化
        如果参数是 iter_args 的 init，并且使用 yield，则不需要初始化（直接赋值）

        Args:
            param_var: 参数变量
            body: 函数体

        Returns:
            是否需要初始化
        """

        # 递归检查 body 中是否是 iter_args 的 init
        def check_for_stmt(stmt):
            if isinstance(stmt, ir.ForStmt):
                for iter_arg in stmt.iter_args:
                    if isinstance(iter_arg, ir.IterArg):
                        init_expr = iter_arg.initValue
                        # 检查是否是同一个变量（通过名称或对象）
                        if isinstance(init_expr, ir.Var):
                            if init_expr.name_hint == param_var.name_hint:
                                # iter_args + yield 模式，init 直接赋值，不需要初始化
                                if self._has_yield_in_body(stmt.body):
                                    return False
                # 检查嵌套循环
                return check_for_stmt(stmt.body)
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    result = check_for_stmt(s)
                    if result is False:
                        return False
            elif isinstance(stmt, ir.IfStmt):
                result = check_for_stmt(stmt.then_body)
                if result is False:
                    return False
                if stmt.else_body:
                    result = check_for_stmt(stmt.else_body)
                    if result is False:
                        return False
            return True

        return check_for_stmt(body)

    def _reverse_stmt(self, stmt: ir.Stmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理语句

        Args:
            stmt: 前向语句
            d_output: 当前梯度

        Returns:
            反向语句列表
        """
        if isinstance(stmt, ir.AssignStmt):
            return self._reverse_assign(stmt, d_output)
        elif isinstance(stmt, ir.ReturnStmt):
            return []
        elif isinstance(stmt, ir.SeqStmts):
            result = []
            for s in reversed(stmt.stmts):
                result.extend(self._reverse_stmt(s, d_output))
            return result
        elif isinstance(stmt, ir.ForStmt):
            return self._reverse_for(stmt, d_output)
        elif isinstance(stmt, ir.IfStmt):
            return self._reverse_if(stmt, d_output)
        elif isinstance(stmt, ir.YieldStmt):
            return self._reverse_yield(stmt, d_output)
        return []

    def _reverse_for(self, stmt: ir.ForStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理 ForStmt

        对于循环，反向需要：
        1. 处理 iter_args 的初始值梯度（如果有）
        2. 生成反向循环
        3. 如果使用 iter_args + yield，反向也需要 iter_args + yield

        Args:
            stmt: ForStmt 语句
            d_output: 当前梯度

        Returns:
            反向语句列表
        """
        stmts = []

        has_iter_args = len(stmt.iter_args) > 0
        has_yield = self._has_yield_in_body(stmt.body)

        if has_iter_args and has_yield:
            stmts.extend(self._reverse_for_iter_args(stmt, d_output))
        else:
            stmts.extend(self._reverse_for_with_tape(stmt, d_output))

        return stmts

    def _reverse_for_with_tape(self, stmt: ir.ForStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        使用 Tape 模式处理普通循环的反向传播

        对于循环内变量在梯度表达式中被引用的情况：
        1. 创建 TensorArray 保存每次迭代的中间值
        2. 正向循环中 push 每次迭代的值
        3. 反向循环中 get 对应迭代的值计算梯度

        Args:
            stmt: ForStmt 语句
            d_output: 输出梯度

        Returns:
            反向语句列表
        """
        from pypto.ir.op import tensor_array

        stmts = []

        for iter_arg in stmt.iter_args:
            if isinstance(iter_arg, ir.IterArg):
                init_expr = iter_arg.initValue
                if isinstance(init_expr, ir.Var):
                    if init_expr in self.grad_var_map:
                        grad_var = self.grad_var_map[init_expr]
                        stmts.append(ir.AssignStmt(grad_var, d_output, self.span))

        vars_needing_tape = self._identify_vars_needing_tape(stmt.body, stmt)

        tape_vars = {}
        if vars_needing_tape:
            for var_name, var_info in vars_needing_tape.items():
                var_type = var_info["type"]
                tape_var_type = ir.TensorType(
                    var_type.shape if hasattr(var_type, "shape") else [],
                    var_type.dtype if hasattr(var_type, "dtype") else DataType.FP32,
                )
                tape_var = ir.Var(f"tape_{var_name}", tape_var_type, self.span)

                shape_list = []
                if hasattr(var_type, "shape"):
                    for dim in var_type.shape:
                        if isinstance(dim, ir.ConstInt):
                            shape_list.append(dim.value)
                        else:
                            shape_list.append(dim)

                capacity_val = stmt.stop
                if isinstance(stmt.stop, ir.ConstInt):
                    capacity_val = stmt.stop.value

                create_call = tensor_array.create(
                    shape_list if shape_list else [],
                    var_type.dtype if hasattr(var_type, "dtype") else DataType.FP32,
                    capacity_val,
                    self.span,
                )
                stmts.append(ir.AssignStmt(tape_var, create_call, self.span))
                tape_vars[var_name] = {
                    "tape_var": tape_var,
                    "init_expr": var_info.get("init_expr"),
                    "init_type": var_info.get("init_type"),
                }

        forward_loop_stmts = []

        vars_defined_before_loop = {p.name_hint for p in self.forward_func.params}
        for iter_arg in stmt.iter_args:
            if isinstance(iter_arg, ir.IterArg):
                init_expr = iter_arg.initValue
                if isinstance(init_expr, ir.Var):
                    vars_defined_before_loop.add(init_expr.name_hint)

        init_stmts = []
        if isinstance(self.forward_func.body, ir.SeqStmts):
            for s in self.forward_func.body.stmts:
                if isinstance(s, ir.ForStmt):
                    break
                if isinstance(s, ir.AssignStmt) and isinstance(s.var, ir.Var):
                    init_stmts.append(s)
                    vars_defined_before_loop.add(s.var.name_hint)
                    if s.var.name_hint in vars_needing_tape:
                        tape_vars[s.var.name_hint]["init_expr"] = s.value
                        tape_vars[s.var.name_hint]["init_type"] = s.var.type

        stmts.extend(init_stmts)

        loop_body_copy = self._copy_loop_body_with_tape(stmt.body, tape_vars)

        forward_loop = ir.ForStmt(
            stmt.loop_var,
            stmt.start,
            stmt.stop,
            stmt.step if stmt.step else ir.ConstInt(1, DataType.INT64, self.span),
            stmt.iter_args,
            ir.SeqStmts(loop_body_copy, self.span),
            stmt.return_vars if stmt.return_vars else [],
            self.span,
            kind=ir.ForKind.Sequential,
        )
        stmts.append(forward_loop)

        for param_var, grad_var in self.grad_var_map.items():
            forward_body = self.forward_func.body
            needs_init = self._param_needs_init_in_loop(param_var, forward_body)
            if needs_init:
                zero = self._create_zero(param_var.type)
                stmts.append(ir.AssignStmt(grad_var, zero, self.span))

        reverse_loop_stmts = []

        tape_var_mapping = {}
        for var_name, tape_info in tape_vars.items():
            tape_var = tape_info["tape_var"]
            var_type = vars_needing_tape[var_name]["type"]
            element_type = ir.TensorType(
                var_type.shape if hasattr(var_type, "shape") else [],
                var_type.dtype if hasattr(var_type, "dtype") else DataType.FP32,
            )
            get_call = tensor_array.get(tape_var, stmt.loop_var, element_type, self.span)
            saved_var = ir.Var(var_name, var_type, self.span)
            reverse_loop_stmts.append(ir.AssignStmt(saved_var, get_call, self.span))
            tape_var_mapping[var_name] = saved_var

        reverse_loop_stmts.extend(
            self._reverse_loop_body_with_tape_vars(stmt.body, d_output, tape_var_mapping)
        )

        if reverse_loop_stmts:
            reverse_loop = ir.ForStmt(
                stmt.loop_var,
                stmt.start,
                stmt.stop,
                stmt.step if stmt.step else ir.ConstInt(1, DataType.INT64, self.span),
                [],
                ir.SeqStmts(reverse_loop_stmts, self.span),
                [],
                self.span,
                kind=ir.ForKind.Sequential,
            )
            stmts.append(reverse_loop)

        return stmts

    def _identify_vars_needing_tape(self, body: ir.Stmt, for_stmt: ir.ForStmt) -> Dict[str, Dict[str, Any]]:
        """
        分析梯度表达式，识别需要保存的循环内变量

        Args:
            body: 循环体语句
            for_stmt: ForStmt 语句

        Returns:
            需要保存的变量字典 {var_name: {"type": type, "init_expr": expr, "init_type": type}}
        """
        param_names = {p.name_hint for p in self.forward_func.params}
        grad_var_names = {gv.name_hint for gv in self.grad_var_map.values()}
        allowed_var_names = param_names | grad_var_names | {"d_output", "d_y"}

        loop_defined_vars = set()
        var_types = {}
        var_init_exprs = {}

        def collect_loop_vars(stmt: ir.Stmt, in_loop: bool = False):
            if isinstance(stmt, ir.AssignStmt):
                if isinstance(stmt.var, ir.Var):
                    var_types[stmt.var.name_hint] = stmt.var.type
                    if in_loop:
                        loop_defined_vars.add(stmt.var.name_hint)
                    if isinstance(stmt.value, ir.Var):
                        var_init_exprs[stmt.var.name_hint] = stmt.value
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    collect_loop_vars(s, in_loop)
            elif isinstance(stmt, ir.ForStmt):
                collect_loop_vars(stmt.body, in_loop=True)
            elif isinstance(stmt, ir.IfStmt):
                collect_loop_vars(stmt.then_body, in_loop)
                if stmt.else_body:
                    collect_loop_vars(stmt.else_body, in_loop)

        collect_loop_vars(body, in_loop=True)

        vars_needed = {}

        def check_expr_for_loop_vars(expr: ir.Expr):
            if isinstance(expr, ir.Var):
                var_name = expr.name_hint
                if var_name in loop_defined_vars:
                    if var_name not in vars_needed:
                        vars_needed[var_name] = {
                            "type": var_types.get(var_name),
                            "init_expr": var_init_exprs.get(var_name),
                            "init_type": var_types.get(var_name) if var_init_exprs.get(var_name) else None,
                        }
            elif isinstance(expr, ir.Call):
                for arg in expr.args:
                    check_expr_for_loop_vars(arg)
            elif isinstance(expr, ir.Add) or isinstance(expr, ir.Mul):
                check_expr_for_loop_vars(expr.lhs)
                check_expr_for_loop_vars(expr.rhs)
            elif isinstance(expr, ir.Neg):
                check_expr_for_loop_vars(expr.operand)

        def check_stmt(stmt: ir.Stmt):
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                call = stmt.value
                from .gradient_registry import GradientRegistry

                op_name = call.op.name
                if not GradientRegistry.has(op_name):
                    parts = op_name.split(".")
                    if len(parts) >= 2:
                        tile_op = f"tile.{parts[-1]}"
                        tensor_op = f"tensor.{parts[-1]}"
                        if GradientRegistry.has(tile_op):
                            op_name = tile_op
                        elif GradientRegistry.has(tensor_op):
                            op_name = tensor_op

                grad_rule = GradientRegistry.get(op_name)
                if grad_rule:
                    saved_inputs = list(call.args)
                    kwargs = {"forward_result": call}
                    try:
                        d_output_var = ir.Var(
                            "d_output", call.type if hasattr(call, "type") else None, self.span
                        )
                        grad_exprs = grad_rule(saved_inputs, d_output_var, kwargs)
                        for grad_expr in grad_exprs:
                            check_expr_for_loop_vars(grad_expr)
                    except Exception:
                        pass
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    check_stmt(s)
            elif isinstance(stmt, ir.IfStmt):
                check_stmt(stmt.then_body)
                if stmt.else_body:
                    check_stmt(stmt.else_body)

        check_stmt(body)

        return vars_needed

    def _copy_loop_body_with_tape(self, body: ir.Stmt, tape_vars: Dict[str, Dict[str, Any]]) -> List[ir.Stmt]:
        """
        复制循环体，并在每次迭代后添加 tape push 操作

        Args:
            body: 循环体语句
            tape_vars: 需要保存到 tape 的变量字典

        Returns:
            复制的循环体语句列表（包含 push 操作）
        """
        from pypto.ir.op import tensor_array

        stmts = []

        if isinstance(body, ir.SeqStmts):
            for s in body.stmts:
                stmts.append(s)
        elif isinstance(body, ir.AssignStmt):
            stmts.append(body)
        elif isinstance(body, ir.IfStmt):
            stmts.append(body)

        for var_name, tape_info in tape_vars.items():
            tape_var = tape_info["tape_var"]
            var_type = tape_info.get("init_type") or tape_info.get("type")
            if var_type is None:
                var_type = ir.UnknownType()
            push_call = tensor_array.push(tape_var, ir.Var(var_name, var_type, self.span), self.span)
            stmts.append(ir.AssignStmt(ir.Var("_", ir.UnknownType(), self.span), push_call, self.span))

        return stmts

    def _has_yield_in_body(self, body: ir.Stmt) -> bool:
        """检查循环体是否包含 yield"""
        if isinstance(body, ir.YieldStmt):
            return True
        elif isinstance(body, ir.SeqStmts):
            for stmt in body.stmts:
                if self._has_yield_in_body(stmt):
                    return True
        elif isinstance(body, ir.IfStmt):
            if self._has_yield_in_body(body.then_body):
                return True
            if body.else_body and self._has_yield_in_body(body.else_body):
                return True
        return False

    def _reverse_for_iter_args(self, stmt: ir.ForStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        处理 iter_args + yield 模式的反向

        对于 iter_args + yield 模式（SSA reduce）：
        - 正向：for i, (acc,) in range(N, init=(init)): yield(acc_new)
        - 数学：y = init * x^N
        - 反向：d_init 直接赋值，d_x 在循环内累加
        - 需要在反向循环中追踪正向 iter_args 的值

        Args:
            stmt: ForStmt 语句
            d_output: 输出梯度

        Returns:
            反向语句列表
        """
        from pypto.ir.op import tensor_array

        stmts = []

        for iter_arg in stmt.iter_args:
            if isinstance(iter_arg, ir.IterArg):
                init_expr = iter_arg.initValue
                if isinstance(init_expr, ir.Var) and init_expr in self.grad_var_map:
                    grad_var = self.grad_var_map[init_expr]
                    stmts.append(ir.AssignStmt(grad_var, d_output, self.span))

        vars_needing_tape = self._identify_vars_needing_tape_iter_args(stmt.body, stmt)

        tape_vars = {}
        if vars_needing_tape:
            for var_name, var_info in vars_needing_tape.items():
                var_type = var_info["type"]
                tape_var_type = ir.TensorType(
                    var_type.shape if hasattr(var_type, "shape") else [],
                    var_type.dtype if hasattr(var_type, "dtype") else DataType.FP32,
                )
                tape_var = ir.Var(f"tape_{var_name}", tape_var_type, self.span)

                shape_list = []
                if hasattr(var_type, "shape"):
                    for dim in var_type.shape:
                        if isinstance(dim, ir.ConstInt):
                            shape_list.append(dim.value)
                        else:
                            shape_list.append(dim)

                capacity_val = stmt.stop
                if isinstance(stmt.stop, ir.ConstInt):
                    capacity_val = stmt.stop.value

                create_call = tensor_array.create(
                    shape_list if shape_list else [],
                    var_type.dtype if hasattr(var_type, "dtype") else DataType.FP32,
                    capacity_val,
                    self.span,
                )
                stmts.append(ir.AssignStmt(tape_var, create_call, self.span))
                tape_vars[var_name] = {
                    "tape_var": tape_var,
                    "init_expr": None,
                    "init_type": var_type,
                }

        init_stmts = []
        if isinstance(self.forward_func.body, ir.SeqStmts):
            for s in self.forward_func.body.stmts:
                if isinstance(s, ir.ForStmt):
                    break
                if isinstance(s, ir.AssignStmt) and isinstance(s.var, ir.Var):
                    init_stmts.append(s)
                    if s.var.name_hint in vars_needing_tape:
                        vars_needing_tape[s.var.name_hint]["init_expr"] = s.value
                        vars_needing_tape[s.var.name_hint]["init_type"] = s.var.type

        stmts.extend(init_stmts)

        forward_loop_body = self._copy_loop_body_with_tape(stmt.body, tape_vars)

        forward_loop = ir.ForStmt(
            stmt.loop_var,
            stmt.start,
            stmt.stop,
            stmt.step if stmt.step else ir.ConstInt(1, DataType.INT64, self.span),
            stmt.iter_args,
            ir.SeqStmts(forward_loop_body, self.span),
            stmt.return_vars if stmt.return_vars else [],
            self.span,
            kind=ir.ForKind.Sequential,
        )
        stmts.append(forward_loop)

        for param_var, grad_var in self.grad_var_map.items():
            forward_body = self.forward_func.body
            needs_init = self._param_needs_init_in_loop(param_var, forward_body)
            if needs_init:
                zero = self._create_zero(param_var.type)
                stmts.append(ir.AssignStmt(grad_var, zero, self.span))

        reverse_iter_args = []
        for iter_arg in stmt.iter_args:
            if isinstance(iter_arg, ir.IterArg):
                d_iter_var = ir.Var(f"d_{iter_arg.name_hint}", iter_arg.type, self.span)
                reverse_iter_args.append(
                    ir.IterArg(d_iter_var.name_hint, d_iter_var.type, d_output, self.span)
                )

        loop_body_stmts = []

        for var_name, tape_info in tape_vars.items():
            var_type = vars_needing_tape[var_name].get("type")
            if var_type is None:
                var_type = ir.UnknownType()
            element_type = ir.TensorType(
                var_type.shape if hasattr(var_type, "shape") else [],
                var_type.dtype if hasattr(var_type, "dtype") else DataType.FP32,
            )
            tape_var = tape_info["tape_var"]
            get_call = tensor_array.get(tape_var, stmt.loop_var, element_type, self.span)
            saved_var = ir.Var(var_name, var_type, self.span)
            loop_body_stmts.append(ir.AssignStmt(saved_var, get_call, self.span))

        tape_var_mapping = {}
        for var_name, tape_info in tape_vars.items():
            var_type = vars_needing_tape[var_name].get("type")
            if var_type is None:
                var_type = ir.UnknownType()
            saved_var = ir.Var(var_name, var_type, self.span)
            tape_var_mapping[var_name] = saved_var

        if isinstance(stmt.body, ir.SeqStmts):
            for body_stmt in reversed(stmt.body.stmts):
                loop_body_stmts.extend(
                    self._reverse_loop_body_iter_args_with_tape(
                        body_stmt, reverse_iter_args, stmt.iter_args, tape_var_mapping
                    )
                )
        elif isinstance(stmt.body, ir.AssignStmt):
            loop_body_stmts.extend(
                self._reverse_assign_iter_args_with_tape(
                    stmt.body, reverse_iter_args, stmt.iter_args, tape_var_mapping
                )
            )
        elif isinstance(stmt.body, ir.YieldStmt):
            pass

        if loop_body_stmts:
            reverse_loop = ir.ForStmt(
                stmt.loop_var,
                stmt.start,
                stmt.stop,
                stmt.step if stmt.step else ir.ConstInt(1, DataType.INT64, self.span),
                reverse_iter_args,
                ir.SeqStmts(loop_body_stmts, self.span),
                [],
                self.span,
                kind=ir.ForKind.Sequential,
            )
            stmts.append(reverse_loop)

        return stmts

    def _identify_vars_needing_tape_iter_args(
        self, body: ir.Stmt, for_stmt: ir.ForStmt
    ) -> Dict[str, Dict[str, Any]]:
        """
        分析 iter_args 模式的梯度表达式，识别需要保存的循环内变量

        Args:
            body: 循环体语句
            for_stmt: ForStmt 语句

        Returns:
            需要保存的变量字典
        """
        param_names = {p.name_hint for p in self.forward_func.params}
        grad_var_names = {gv.name_hint for gv in self.grad_var_map.values()}
        allowed_var_names = param_names | grad_var_names | {"d_output", "d_y"}

        iter_arg_names = set()
        for iter_arg in for_stmt.iter_args:
            if isinstance(iter_arg, ir.IterArg):
                iter_arg_names.add(iter_arg.name_hint)

        loop_defined_vars = set()
        var_types = {}

        def collect_loop_vars(stmt: ir.Stmt):
            if isinstance(stmt, ir.AssignStmt):
                if isinstance(stmt.var, ir.Var):
                    var_types[stmt.var.name_hint] = stmt.var.type
                    if stmt.var.name_hint not in iter_arg_names:
                        loop_defined_vars.add(stmt.var.name_hint)
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    collect_loop_vars(s)
            elif isinstance(stmt, ir.IfStmt):
                collect_loop_vars(stmt.then_body)
                if stmt.else_body:
                    collect_loop_vars(stmt.else_body)

        collect_loop_vars(body)

        vars_needed = {}

        def check_expr_for_loop_vars(expr: ir.Expr):
            if isinstance(expr, ir.Var):
                var_name = expr.name_hint
                if var_name in loop_defined_vars or var_name in iter_arg_names:
                    if var_name not in vars_needed:
                        vars_needed[var_name] = {
                            "type": var_types.get(var_name),
                            "init_expr": None,
                            "init_type": None,
                        }
            elif isinstance(expr, ir.Call):
                for arg in expr.args:
                    check_expr_for_loop_vars(arg)
            elif isinstance(expr, ir.Add) or isinstance(expr, ir.Mul):
                check_expr_for_loop_vars(expr.lhs)
                check_expr_for_loop_vars(expr.rhs)
            elif isinstance(expr, ir.Neg):
                check_expr_for_loop_vars(expr.operand)

        def check_stmt(stmt: ir.Stmt):
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                call = stmt.value
                from .gradient_registry import GradientRegistry

                op_name = call.op.name
                if not GradientRegistry.has(op_name):
                    parts = op_name.split(".")
                    if len(parts) >= 2:
                        tile_op = f"tile.{parts[-1]}"
                        tensor_op = f"tensor.{parts[-1]}"
                        if GradientRegistry.has(tile_op):
                            op_name = tile_op
                        elif GradientRegistry.has(tensor_op):
                            op_name = tensor_op

                grad_rule = GradientRegistry.get(op_name)
                if grad_rule:
                    saved_inputs = list(call.args)
                    kwargs = {"forward_result": call}
                    try:
                        d_output_var = ir.Var(
                            "d_output", call.type if hasattr(call, "type") else None, self.span
                        )
                        grad_exprs = grad_rule(saved_inputs, d_output_var, kwargs)
                        for grad_expr in grad_exprs:
                            check_expr_for_loop_vars(grad_expr)
                    except Exception:
                        pass
            elif isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    check_stmt(s)
            elif isinstance(stmt, ir.IfStmt):
                check_stmt(stmt.then_body)
                if stmt.else_body:
                    check_stmt(stmt.else_body)

        check_stmt(body)

        return vars_needed

    def _reverse_loop_body_iter_args_with_tape(
        self,
        body: ir.Stmt,
        reverse_iter_args: List[ir.IterArg],
        forward_iter_args: List[ir.IterArg],
        tape_var_mapping: Dict[str, ir.Var],
    ) -> List[ir.Stmt]:
        """处理 iter_args 模式下的循环体反向（使用 tape 变量）"""
        stmts = []
        if isinstance(body, ir.SeqStmts):
            for stmt in reversed(body.stmts):
                stmts.extend(
                    self._reverse_loop_body_iter_args_with_tape(
                        stmt, reverse_iter_args, forward_iter_args, tape_var_mapping
                    )
                )
        elif isinstance(body, ir.AssignStmt):
            stmts.extend(
                self._reverse_assign_iter_args_with_tape(
                    body, reverse_iter_args, forward_iter_args, tape_var_mapping
                )
            )
        elif isinstance(body, ir.YieldStmt):
            pass
        return stmts

    def _reverse_assign_iter_args_with_tape(
        self,
        stmt: ir.AssignStmt,
        reverse_iter_args: List[ir.IterArg],
        forward_iter_args: List[ir.IterArg],
        tape_var_mapping: Dict[str, ir.Var],
    ) -> List[ir.Stmt]:
        """处理 iter_args 模式下的赋值反向（使用 tape 变量）"""
        stmts = []

        if isinstance(stmt.value, ir.Call):
            call = stmt.value
            from .gradient_registry import GradientRegistry

            op_name = call.op.name
            if not GradientRegistry.has(op_name):
                parts = op_name.split(".")
                if len(parts) >= 2:
                    if GradientRegistry.has(f"tile.{parts[-1]}"):
                        op_name = f"tile.{parts[-1]}"
                    elif GradientRegistry.has(f"tensor.{parts[-1]}"):
                        op_name = f"tensor.{parts[-1]}"

            grad_rule = GradientRegistry.get(op_name)
            if grad_rule is None:
                return stmts

            if reverse_iter_args:
                d_iter_var = ir.Var(reverse_iter_args[0].name_hint, reverse_iter_args[0].type, self.span)
                saved_inputs = list(call.args)
                kwargs = {"forward_result": call}

                try:
                    grad_exprs = grad_rule(saved_inputs, d_iter_var, kwargs)
                except Exception:
                    return stmts

                for arg, grad_expr in zip(call.args, grad_exprs):
                    replaced_grad_expr = self._replace_vars_in_expr(grad_expr, tape_var_mapping)
                    if isinstance(arg, ir.Var) and arg in self.grad_var_map:
                        grad_var = self.grad_var_map[arg]
                        add_expr = self._create_add(grad_var, replaced_grad_expr)
                        stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))

        return stmts

    def _reverse_loop_body(self, body: ir.Stmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理循环体

        对于循环体内的操作，只累加参数的梯度，忽略临时变量

        Args:
            body: 循环体语句
            d_output: 每次迭代的梯度（函数输出梯度）

        Returns:
            反向语句列表
        """
        stmts = []

        if isinstance(body, ir.SeqStmts):
            for stmt in reversed(body.stmts):
                stmts.extend(self._reverse_loop_body(stmt, d_output))
        elif isinstance(body, ir.AssignStmt):
            stmts.extend(self._reverse_assign_loop_simple(body, d_output))
        elif isinstance(body, ir.IfStmt):
            stmts.extend(self._reverse_if_loop_simple(body, d_output))
        elif isinstance(body, ir.YieldStmt):
            stmts.extend(self._reverse_yield_loop_simple(body, d_output))

        return stmts

    def _reverse_loop_body_with_tape_vars(
        self, body: ir.Stmt, d_output: ir.Var, tape_var_mapping: Dict[str, ir.Var]
    ) -> List[ir.Stmt]:
        """
        反向处理循环体，使用 tape 变量替换

        Args:
            body: 循环体语句
            d_output: 每次迭代的梯度
            tape_var_mapping: 从变量名到 tape 变量的映射

        Returns:
            反向语句列表
        """
        stmts = []

        if isinstance(body, ir.SeqStmts):
            for stmt in reversed(body.stmts):
                stmts.extend(self._reverse_loop_body_with_tape_vars(stmt, d_output, tape_var_mapping))
        elif isinstance(body, ir.AssignStmt):
            stmts.extend(self._reverse_assign_loop_with_tape(body, d_output, tape_var_mapping))
        elif isinstance(body, ir.IfStmt):
            stmts.extend(self._reverse_if_loop_with_tape(body, d_output, tape_var_mapping))
        elif isinstance(body, ir.YieldStmt):
            stmts.extend(self._reverse_yield_loop_with_tape(body, d_output, tape_var_mapping))

        return stmts

    def _replace_vars_in_expr(self, expr: ir.Expr, tape_var_mapping: Dict[str, ir.Var]) -> ir.Expr:
        """
        替换表达式中的变量引用

        Args:
            expr: 原始表达式
            tape_var_mapping: 变量名到新变量的映射

        Returns:
            替换后的表达式
        """
        if isinstance(expr, ir.Var):
            var_name = expr.name_hint
            if var_name in tape_var_mapping:
                return tape_var_mapping[var_name]
            return expr
        elif isinstance(expr, ir.Call):
            new_args = [self._replace_vars_in_expr(arg, tape_var_mapping) for arg in expr.args]
            return ir.Call(expr.op, new_args, expr.kwargs, expr.type, expr.span)
        elif isinstance(expr, ir.Add) or isinstance(expr, ir.Mul):
            new_lhs = self._replace_vars_in_expr(expr.lhs, tape_var_mapping)
            new_rhs = self._replace_vars_in_expr(expr.rhs, tape_var_mapping)
            if isinstance(expr, ir.Add):
                return ir.Add(new_lhs, new_rhs, expr.dtype, expr.span)
            else:
                return ir.Mul(new_lhs, new_rhs, expr.dtype, expr.span)
        elif isinstance(expr, ir.Neg):
            new_operand = self._replace_vars_in_expr(expr.operand, tape_var_mapping)
            return ir.Neg(new_operand, expr.dtype, expr.span)
        return expr

    def _reverse_assign_loop_with_tape(
        self, stmt: ir.AssignStmt, d_output: ir.Var, tape_var_mapping: Dict[str, ir.Var]
    ) -> List[ir.Stmt]:
        """
        反向处理循环体内的赋值语句（使用 tape 变量）

        Args:
            stmt: 赋值语句
            d_output: 每次迭代的梯度
            tape_var_mapping: 变量名到 tape 变量的映射
        """
        stmts = []

        if isinstance(stmt.value, ir.Call):
            call = stmt.value
            from .gradient_registry import GradientRegistry

            op_name = call.op.name
            if not GradientRegistry.has(op_name):
                parts = op_name.split(".")
                if len(parts) >= 2:
                    tile_op = f"tile.{parts[-1]}"
                    tensor_op = f"tensor.{parts[-1]}"
                    if GradientRegistry.has(tile_op):
                        op_name = tile_op
                    elif GradientRegistry.has(tensor_op):
                        op_name = tensor_op

            grad_rule = GradientRegistry.get(op_name)
            if grad_rule is None:
                return stmts

            saved_inputs = list(call.args)
            kwargs = {"forward_result": call}

            try:
                grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
            except Exception:
                return stmts

            for arg, grad_expr in zip(call.args, grad_exprs):
                replaced_grad_expr = self._replace_vars_in_expr(grad_expr, tape_var_mapping)
                if isinstance(arg, ir.Var) and arg in self.grad_var_map:
                    grad_var = self.grad_var_map[arg]
                    add_expr = self._create_add(grad_var, replaced_grad_expr)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
                elif isinstance(arg, ir.Call):
                    stmts.extend(self._reverse_nested_call(arg, replaced_grad_expr))

        elif isinstance(stmt.value, ir.Var):
            src_var = stmt.value
            if src_var in self.grad_var_map:
                grad_var = self.grad_var_map[src_var]
                add_expr = self._create_add(grad_var, d_output)
                stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))

        return stmts

    def _reverse_if_loop_with_tape(
        self, stmt: ir.IfStmt, d_output: ir.Var, tape_var_mapping: Dict[str, ir.Var]
    ) -> List[ir.Stmt]:
        """反向处理循环体内的 if 语句（使用 tape 变量）"""
        then_stmts = []
        then_body = stmt.then_body
        if isinstance(then_body, ir.SeqStmts):
            for body_stmt in reversed(then_body.stmts):
                then_stmts.extend(
                    self._reverse_loop_body_with_tape_vars(body_stmt, d_output, tape_var_mapping)
                )
        elif isinstance(then_body, ir.AssignStmt):
            then_stmts.extend(self._reverse_assign_loop_with_tape(then_body, d_output, tape_var_mapping))
        elif isinstance(then_body, ir.YieldStmt):
            then_stmts.extend(self._reverse_yield_loop_with_tape(then_body, d_output, tape_var_mapping))

        else_stmts = []
        if stmt.else_body is not None:
            else_body = stmt.else_body
            if isinstance(else_body, ir.SeqStmts):
                for body_stmt in reversed(else_body.stmts):
                    else_stmts.extend(
                        self._reverse_loop_body_with_tape_vars(body_stmt, d_output, tape_var_mapping)
                    )
            elif isinstance(else_body, ir.AssignStmt):
                else_stmts.extend(self._reverse_assign_loop_with_tape(else_body, d_output, tape_var_mapping))
            elif isinstance(else_body, ir.YieldStmt):
                else_stmts.extend(self._reverse_yield_loop_with_tape(else_body, d_output, tape_var_mapping))

        then_body_ir = ir.SeqStmts(then_stmts, self.span) if then_stmts else ir.SeqStmts([], self.span)
        else_body_ir = ir.SeqStmts(else_stmts, self.span) if else_stmts else None

        reverse_if = ir.IfStmt(
            stmt.condition,
            then_body_ir,
            else_body_ir,
            stmt.return_vars if stmt.return_vars else [],
            self.span,
        )

        return [reverse_if]

    def _reverse_yield_loop_with_tape(
        self, stmt: ir.YieldStmt, d_output: ir.Var, tape_var_mapping: Dict[str, ir.Var]
    ) -> List[ir.Stmt]:
        """反向处理循环体内的 yield 语句（使用 tape 变量）"""
        stmts = []
        for value_expr in stmt.value:
            if isinstance(value_expr, ir.Call):
                replaced_call = self._replace_vars_in_expr(value_expr, tape_var_mapping)
                stmts.extend(self._reverse_call_inline(replaced_call, d_output, None))
            elif isinstance(value_expr, ir.Var):
                if value_expr in self.grad_var_map:
                    grad_var = self.grad_var_map[value_expr]
                    add_expr = self._create_add(grad_var, d_output)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
                elif value_expr.name_hint in tape_var_mapping:
                    tape_var = tape_var_mapping[value_expr.name_hint]
                    if tape_var in self.grad_var_map:
                        grad_var = self.grad_var_map[tape_var]
                        add_expr = self._create_add(grad_var, d_output)
                        stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))

        return stmts

    def _reverse_assign_loop_simple(self, stmt: ir.AssignStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理循环体内的赋值语句（简化版本）

        只累加参数的梯度，使用 d_output 作为梯度

        Args:
            stmt: 赋值语句
            d_output: 每次迭代的梯度
        """
        stmts = []

        if isinstance(stmt.value, ir.Call):
            call = stmt.value
            from .gradient_registry import GradientRegistry

            op_name = call.op.name
            if not GradientRegistry.has(op_name):
                parts = op_name.split(".")
                if len(parts) >= 2:
                    tile_op = f"tile.{parts[-1]}"
                    tensor_op = f"tensor.{parts[-1]}"
                    if GradientRegistry.has(tile_op):
                        op_name = tile_op
                    elif GradientRegistry.has(tensor_op):
                        op_name = tensor_op

            grad_rule = GradientRegistry.get(op_name)
            if grad_rule is None:
                return stmts

            saved_inputs = list(call.args)
            kwargs = {"forward_result": call}

            try:
                grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
            except Exception:
                return stmts

            for arg, grad_expr in zip(call.args, grad_exprs):
                if isinstance(arg, ir.Var) and arg in self.grad_var_map:
                    grad_var = self.grad_var_map[arg]
                    add_expr = self._create_add(grad_var, grad_expr)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
                elif isinstance(arg, ir.Call):
                    stmts.extend(self._reverse_nested_call(arg, grad_expr))

        elif isinstance(stmt.value, ir.Var):
            src_var = stmt.value
            if src_var in self.grad_var_map:
                grad_var = self.grad_var_map[src_var]
                add_expr = self._create_add(grad_var, d_output)
                stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))

        return stmts

    def _reverse_nested_call(self, call: ir.Call, d_output: ir.Expr) -> List[ir.Stmt]:
        """
        处理嵌套调用的梯度

        例如：add(acc, muls(x, 2.0))
        - add 的梯度：[d_acc, d_muls_result]
        - muls 的梯度：[d_x * 2.0]

        Args:
            call: 嵌套的 Call 表达式
            d_output: 该嵌套调用的输出梯度

        Returns:
            反向语句列表
        """
        stmts = []
        from .gradient_registry import GradientRegistry

        op_name = call.op.name
        if not GradientRegistry.has(op_name):
            parts = op_name.split(".")
            if len(parts) >= 2:
                tile_op = f"tile.{parts[-1]}"
                tensor_op = f"tensor.{parts[-1]}"
                if GradientRegistry.has(tile_op):
                    op_name = tile_op
                elif GradientRegistry.has(tensor_op):
                    op_name = tensor_op

        grad_rule = GradientRegistry.get(op_name)
        if grad_rule is None:
            return stmts

        saved_inputs = list(call.args)
        kwargs = {"forward_result": call}

        try:
            grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
        except Exception:
            return stmts

        # 处理嵌套调用的参数梯度
        for arg, grad_expr in zip(call.args, grad_exprs):
            if isinstance(arg, ir.Var):
                if arg in self.grad_var_map:
                    # 输入参数：累加梯度
                    grad_var = self.grad_var_map[arg]
                    add_expr = self._create_add(grad_var, grad_expr)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
                else:
                    # 临时变量：直接赋值（不初始化为零）
                    grad_var = self._get_or_create_grad_var(arg)
                    stmts.append(ir.AssignStmt(grad_var, grad_expr, self.span))
            elif isinstance(arg, ir.Call):
                # 多层嵌套：继续递归
                stmts.extend(self._reverse_nested_call(arg, grad_expr))

        return stmts

    def _reverse_if_loop_simple(self, stmt: ir.IfStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """反向处理循环体内的 if 语句（保持分支结构）"""
        # 生成反向 then_body
        then_stmts = []
        then_body = stmt.then_body
        if isinstance(then_body, ir.SeqStmts):
            for body_stmt in reversed(then_body.stmts):
                then_stmts.extend(self._reverse_loop_body(body_stmt, d_output))
        elif isinstance(then_body, ir.AssignStmt):
            then_stmts.extend(self._reverse_assign_loop_simple(then_body, d_output))
        elif isinstance(then_body, ir.YieldStmt):
            then_stmts.extend(self._reverse_yield_loop_simple(then_body, d_output))

        # 生成反向 else_body（如果有）
        else_stmts = []
        if stmt.else_body is not None:
            else_body = stmt.else_body
            if isinstance(else_body, ir.SeqStmts):
                for body_stmt in reversed(else_body.stmts):
                    else_stmts.extend(self._reverse_loop_body(body_stmt, d_output))
            elif isinstance(else_body, ir.AssignStmt):
                else_stmts.extend(self._reverse_assign_loop_simple(else_body, d_output))
            elif isinstance(else_body, ir.YieldStmt):
                else_stmts.extend(self._reverse_yield_loop_simple(else_body, d_output))

        # 创建反向 IfStmt（保持分支结构）
        then_body_ir = ir.SeqStmts(then_stmts, self.span) if then_stmts else ir.SeqStmts([], self.span)
        else_body_ir = ir.SeqStmts(else_stmts, self.span) if else_stmts else None

        reverse_if = ir.IfStmt(
            stmt.condition,
            then_body_ir,
            else_body_ir,
            stmt.return_vars if stmt.return_vars else [],
            self.span,
        )

        return [reverse_if]

    def _reverse_yield_loop_simple(self, stmt: ir.YieldStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """反向处理循环体内的 yield 语句（简化版本）"""
        stmts = []
        for value_expr in stmt.value:
            if isinstance(value_expr, ir.Call):
                # 处理 Call
                call = value_expr
                from .gradient_registry import GradientRegistry

                op_name = call.op.name
                if not GradientRegistry.has(op_name):
                    parts = op_name.split(".")
                    if len(parts) >= 2:
                        tile_op = f"tile.{parts[-1]}"
                        tensor_op = f"tensor.{parts[-1]}"
                        if GradientRegistry.has(tile_op):
                            op_name = tile_op
                        elif GradientRegistry.has(tensor_op):
                            op_name = tensor_op

                grad_rule = GradientRegistry.get(op_name)
                if grad_rule is None:
                    continue

                saved_inputs = list(call.args)
                kwargs = {"forward_result": call}

                try:
                    grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
                except Exception:
                    continue

                for arg, grad_expr in zip(call.args, grad_exprs):
                    if isinstance(arg, ir.Var) and arg in self.grad_var_map:
                        grad_var = self.grad_var_map[arg]
                        add_expr = self._create_add(grad_var, grad_expr)
                        stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))

            elif isinstance(value_expr, ir.Var):
                if value_expr in self.grad_var_map:
                    grad_var = self.grad_var_map[value_expr]
                    add_expr = self._create_add(grad_var, d_output)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))

        return stmts

    def _collect_loop_vars(self, body: ir.Stmt) -> List[ir.Var]:
        """收集循环体内的所有变量"""
        vars_list = []
        if isinstance(body, ir.SeqStmts):
            for stmt in body.stmts:
                vars_list.extend(self._collect_loop_vars(stmt))
        elif isinstance(body, ir.AssignStmt):
            if isinstance(body.var, ir.Var):
                vars_list.append(body.var)
            if isinstance(body.value, ir.Var):
                vars_list.append(body.value)
            elif isinstance(body.value, ir.Call):
                for arg in body.value.args:
                    if isinstance(arg, ir.Var):
                        vars_list.append(arg)
        elif isinstance(body, ir.IfStmt):
            vars_list.extend(self._collect_loop_vars(body.then_body))
            if body.else_body:
                vars_list.extend(self._collect_loop_vars(body.else_body))
        elif isinstance(body, ir.YieldStmt):
            for val in body.value:
                if isinstance(val, ir.Var):
                    vars_list.append(val)
                elif isinstance(val, ir.Call):
                    for arg in val.args:
                        if isinstance(arg, ir.Var):
                            vars_list.append(arg)
        return vars_list

    def _reverse_stmt_loop(
        self, stmt: ir.Stmt, d_output: ir.Var, loop_grad_vars: Dict[ir.Var, ir.Var]
    ) -> List[ir.Stmt]:
        """反向处理循环体内的语句"""
        if isinstance(stmt, ir.AssignStmt):
            return self._reverse_assign_loop(stmt, d_output, loop_grad_vars)
        elif isinstance(stmt, ir.ReturnStmt):
            return []
        elif isinstance(stmt, ir.SeqStmts):
            result = []
            for s in reversed(stmt.stmts):
                result.extend(self._reverse_stmt_loop(s, d_output, loop_grad_vars))
            return result
        elif isinstance(stmt, ir.ForStmt):
            return self._reverse_for(stmt, d_output)
        elif isinstance(stmt, ir.IfStmt):
            return self._reverse_if_loop(stmt, d_output, loop_grad_vars)
        elif isinstance(stmt, ir.YieldStmt):
            return self._reverse_yield_loop(stmt, d_output, loop_grad_vars)
        return []

    def _reverse_assign_loop(
        self, stmt: ir.AssignStmt, d_output: ir.Var, loop_grad_vars: Dict[ir.Var, ir.Var]
    ) -> List[ir.Stmt]:
        """反向处理循环体内的赋值语句"""
        stmts = []

        if isinstance(stmt.value, ir.Call):
            call = stmt.value
            # 获取输出的梯度变量
            if stmt.var in loop_grad_vars:
                out_grad = loop_grad_vars[stmt.var]
            else:
                out_grad = d_output

            # 获取梯度规则
            from .gradient_registry import GradientRegistry

            op_name = call.op.name
            if not GradientRegistry.has(op_name):
                parts = op_name.split(".")
                if len(parts) >= 2:
                    tile_op = f"tile.{parts[-1]}"
                    tensor_op = f"tensor.{parts[-1]}"
                    if GradientRegistry.has(tile_op):
                        op_name = tile_op
                    elif GradientRegistry.has(tensor_op):
                        op_name = tensor_op

            grad_rule = GradientRegistry.get(op_name)
            if grad_rule is None:
                return stmts

            saved_inputs = list(call.args)
            kwargs = {"forward_result": call}

            try:
                grad_exprs = grad_rule(saved_inputs, out_grad, kwargs)
            except Exception:
                return stmts

            # 累加梯度到输入
            for arg, grad_expr in zip(call.args, grad_exprs):
                if isinstance(arg, ir.Var):
                    if arg in self.grad_var_map:
                        grad_var = self.grad_var_map[arg]
                        add_expr = self._create_add(grad_var, grad_expr)
                        stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
                    elif arg in loop_grad_vars:
                        grad_var = loop_grad_vars[arg]
                        add_expr = self._create_add(grad_var, grad_expr)
                        stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))

        elif isinstance(stmt.value, ir.Var):
            src_var = stmt.value
            # 获取输出的梯度
            if stmt.var in loop_grad_vars:
                out_grad = loop_grad_vars[stmt.var]
            else:
                out_grad = d_output

            # 累加到输入
            if src_var in self.grad_var_map:
                grad_var = self.grad_var_map[src_var]
                add_expr = self._create_add(grad_var, out_grad)
                stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
            elif src_var in loop_grad_vars:
                grad_var = loop_grad_vars[src_var]
                add_expr = self._create_add(grad_var, out_grad)
                stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))

        return stmts

    def _reverse_if_loop(
        self, stmt: ir.IfStmt, d_output: ir.Var, loop_grad_vars: Dict[ir.Var, ir.Var]
    ) -> List[ir.Stmt]:
        """反向处理循环体内的 if 语句（保持分支结构）"""
        # 生成反向 then_body
        then_stmts = []
        then_body = stmt.then_body
        if isinstance(then_body, ir.SeqStmts):
            for body_stmt in reversed(then_body.stmts):
                then_stmts.extend(self._reverse_stmt_loop(body_stmt, d_output, loop_grad_vars))
        elif isinstance(then_body, ir.AssignStmt):
            then_stmts.extend(self._reverse_assign_loop(then_body, d_output, loop_grad_vars))
        elif isinstance(then_body, ir.YieldStmt):
            then_stmts.extend(self._reverse_yield_loop(then_body, d_output, loop_grad_vars))

        # 生成反向 else_body（如果有）
        else_stmts = []
        if stmt.else_body is not None:
            else_body = stmt.else_body
            if isinstance(else_body, ir.SeqStmts):
                for body_stmt in reversed(else_body.stmts):
                    else_stmts.extend(self._reverse_stmt_loop(body_stmt, d_output, loop_grad_vars))
            elif isinstance(else_body, ir.AssignStmt):
                else_stmts.extend(self._reverse_assign_loop(else_body, d_output, loop_grad_vars))
            elif isinstance(else_body, ir.YieldStmt):
                else_stmts.extend(self._reverse_yield_loop(else_body, d_output, loop_grad_vars))

        # 创建反向 IfStmt（保持分支结构）
        then_body_ir = ir.SeqStmts(then_stmts, self.span) if then_stmts else ir.SeqStmts([], self.span)
        else_body_ir = ir.SeqStmts(else_stmts, self.span) if else_stmts else None

        reverse_if = ir.IfStmt(
            stmt.condition,
            then_body_ir,
            else_body_ir,
            stmt.return_vars if stmt.return_vars else [],
            self.span,
        )

        return [reverse_if]

    def _reverse_yield_loop(
        self, stmt: ir.YieldStmt, d_output: ir.Var, loop_grad_vars: Dict[ir.Var, ir.Var]
    ) -> List[ir.Stmt]:
        """反向处理循环体内的 yield 语句"""
        stmts = []
        for value_expr in stmt.value:
            if isinstance(value_expr, ir.Call):
                stmts.extend(self._reverse_call_inline(value_expr, d_output, None))
            elif isinstance(value_expr, ir.Var):
                if value_expr in self.grad_var_map:
                    grad_var = self.grad_var_map[value_expr]
                    add_expr = self._create_add(grad_var, d_output)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
                elif value_expr in loop_grad_vars:
                    grad_var = loop_grad_vars[value_expr]
                    add_expr = self._create_add(grad_var, d_output)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
        return stmts

    def _reverse_stmt_inline(self, stmt: ir.Stmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理语句（内联版本，不添加到累加器）

        Args:
            stmt: 前向语句
            d_output: 当前梯度

        Returns:
            反向语句列表（直接生成赋值语句，不使用累加器）
        """
        if isinstance(stmt, ir.AssignStmt):
            return self._reverse_assign_inline(stmt, d_output)
        elif isinstance(stmt, ir.ReturnStmt):
            return []
        elif isinstance(stmt, ir.SeqStmts):
            result = []
            for s in reversed(stmt.stmts):
                result.extend(self._reverse_stmt_inline(s, d_output))
            return result
        elif isinstance(stmt, ir.ForStmt):
            return self._reverse_for(stmt, d_output)
        elif isinstance(stmt, ir.IfStmt):
            return self._reverse_if_inline(stmt, d_output)
        elif isinstance(stmt, ir.YieldStmt):
            return self._reverse_yield_inline(stmt, d_output)
        return []

    def _reverse_assign_inline(self, stmt: ir.AssignStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理赋值语句（内联版本）

        输入参数：累加梯度（因为可能被多次使用）
        临时变量：直接赋值（不需要初始化为零）

        Args:
            stmt: 赋值语句
            d_output: 输出梯度

        Returns:
            反向语句列表
        """
        stmts = []
        grad = d_output

        if isinstance(stmt.value, ir.Call):
            call = stmt.value
            stmts.extend(self._reverse_call_inline(call, grad, stmt.var))
        elif isinstance(stmt.value, ir.Var):
            src_var = stmt.value
            if src_var in self.grad_var_map:
                # 输入参数：累加梯度
                grad_var = self.grad_var_map[src_var]
                add_expr = self._create_add(grad_var, grad)
                stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
            else:
                # 临时变量：直接赋值（不初始化为零）
                grad_var = self._get_or_create_grad_var(src_var)
                stmts.append(ir.AssignStmt(grad_var, grad, self.span))
        elif isinstance(stmt.value, ir.ConstInt) or isinstance(stmt.value, ir.ConstFloat):
            pass

        return stmts

    def _reverse_call_inline(self, call: ir.Call, d_output: ir.Expr, output_var: ir.Var) -> List[ir.Stmt]:
        """
        反向处理 Call 表达式（内联版本）

        Args:
            call: Call 表达式
            d_output: 输出梯度
            output_var: 输出变量

        Returns:
            反向语句列表
        """
        from .gradient_registry import GradientRegistry

        stmts = []

        op_name = call.op.name
        if not GradientRegistry.has(op_name):
            if "tensor" in op_name or "tile" in op_name:
                parts = op_name.split(".")
                if len(parts) >= 2:
                    tile_op = f"tile.{parts[-1]}"
                    tensor_op = f"tensor.{parts[-1]}"
                    if GradientRegistry.has(tile_op):
                        op_name = tile_op
                    elif GradientRegistry.has(tensor_op):
                        op_name = tensor_op

        grad_rule = GradientRegistry.get(op_name)
        if grad_rule is None:
            return stmts

        saved_inputs = list(call.args)
        kwargs = {"forward_result": call}

        try:
            grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
        except Exception:
            return stmts

        # 处理每个参数的梯度
        for arg, grad_expr in zip(call.args, grad_exprs):
            if isinstance(arg, ir.Var):
                if arg in self.grad_var_map:
                    # 输入参数：累加梯度
                    grad_var = self.grad_var_map[arg]
                    add_expr = self._create_add(grad_var, grad_expr)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
                else:
                    # 临时变量：直接赋值（不初始化为零）
                    grad_var = self._get_or_create_grad_var(arg)
                    stmts.append(ir.AssignStmt(grad_var, grad_expr, self.span))

        return stmts

    def _reverse_if_inline(self, stmt: ir.IfStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """反向处理 IfStmt（内联版本，保持分支结构）"""
        # 生成反向 then_body
        then_stmts = []
        then_body = stmt.then_body
        if isinstance(then_body, ir.SeqStmts):
            for body_stmt in reversed(then_body.stmts):
                then_stmts.extend(self._reverse_stmt_inline(body_stmt, d_output))
        elif isinstance(then_body, ir.AssignStmt):
            then_stmts.extend(self._reverse_assign_inline(then_body, d_output))
        elif isinstance(then_body, ir.YieldStmt):
            then_stmts.extend(self._reverse_yield_inline(then_body, d_output))

        # 生成反向 else_body（如果有）
        else_stmts = []
        if stmt.else_body is not None:
            else_body = stmt.else_body
            if isinstance(else_body, ir.SeqStmts):
                for body_stmt in reversed(else_body.stmts):
                    else_stmts.extend(self._reverse_stmt_inline(body_stmt, d_output))
            elif isinstance(else_body, ir.AssignStmt):
                else_stmts.extend(self._reverse_assign_inline(else_body, d_output))
            elif isinstance(else_body, ir.YieldStmt):
                else_stmts.extend(self._reverse_yield_inline(else_body, d_output))

        # 创建反向 IfStmt（保持分支结构）
        then_body_ir = ir.SeqStmts(then_stmts, self.span) if then_stmts else ir.SeqStmts([], self.span)
        else_body_ir = ir.SeqStmts(else_stmts, self.span) if else_stmts else None

        reverse_if = ir.IfStmt(
            stmt.condition,
            then_body_ir,
            else_body_ir,
            stmt.return_vars if stmt.return_vars else [],
            self.span,
        )

        return [reverse_if]

    def _reverse_yield_inline(self, stmt: ir.YieldStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """反向处理 YieldStmt（内联版本）"""
        stmts = []
        for value_expr in stmt.value:
            if isinstance(value_expr, ir.Call):
                stmts.extend(self._reverse_call_inline(value_expr, d_output, None))
            elif isinstance(value_expr, ir.Var):
                if value_expr in self.grad_var_map:
                    # 输入参数：累加梯度
                    grad_var = self.grad_var_map[value_expr]
                    add_expr = self._create_add(grad_var, d_output)
                    stmts.append(ir.AssignStmt(grad_var, add_expr, self.span))
                else:
                    # 临时变量：直接赋值（不初始化为零）
                    grad_var = self._get_or_create_grad_var(value_expr)
                    stmts.append(ir.AssignStmt(grad_var, d_output, self.span))
        return stmts

    def _get_or_create_grad_var(self, var: ir.Var) -> ir.Var:
        """获取或创建梯度变量"""
        if var in self.grad_var_map:
            return self.grad_var_map[var]
        # 创建新的梯度变量
        grad_name = f"d_{var.name_hint}"
        grad_var = ir.Var(grad_name, var.type, self.span)
        self.grad_var_map[var] = grad_var
        return grad_var

    def _reverse_if(self, stmt: ir.IfStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理 IfStmt

        对于条件分支，反向应该保持分支结构：
        - 正向: if cond: y = op1(x) else: y = op2(x)
        - 反向: if cond: d_x = grad_op1(d_y, x) else: d_x = grad_op2(d_y, x)

        Args:
            stmt: IfStmt 语句
            d_output: 当前梯度

        Returns:
            反向语句列表
        """
        # 生成反向 then_body
        then_stmts = []
        then_body = stmt.then_body
        if isinstance(then_body, ir.SeqStmts):
            for body_stmt in reversed(then_body.stmts):
                then_stmts.extend(self._reverse_stmt(body_stmt, d_output))
        elif isinstance(then_body, ir.AssignStmt):
            then_stmts.extend(self._reverse_assign(then_body, d_output))
        elif isinstance(then_body, ir.YieldStmt):
            then_stmts.extend(self._reverse_yield(then_body, d_output))

        # 生成反向 else_body（如果有）
        else_stmts = []
        if stmt.else_body is not None:
            else_body = stmt.else_body
            if isinstance(else_body, ir.SeqStmts):
                for body_stmt in reversed(else_body.stmts):
                    else_stmts.extend(self._reverse_stmt(body_stmt, d_output))
            elif isinstance(else_body, ir.AssignStmt):
                else_stmts.extend(self._reverse_assign(else_body, d_output))
            elif isinstance(else_body, ir.YieldStmt):
                else_stmts.extend(self._reverse_yield(else_body, d_output))

        # 创建反向 IfStmt（保持分支结构）
        then_body_ir = ir.SeqStmts(then_stmts, self.span) if then_stmts else ir.SeqStmts([], self.span)
        else_body_ir = ir.SeqStmts(else_stmts, self.span) if else_stmts else None

        reverse_if = ir.IfStmt(
            stmt.condition,
            then_body_ir,
            else_body_ir,
            stmt.return_vars if stmt.return_vars else [],
            self.span,
        )

        return [reverse_if]

    def _reverse_yield(self, stmt: ir.YieldStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理 YieldStmt

        yield 语句返回循环/控制流的中间状态
        反向时梯度从 yield 输出反向传播到 yield 前的表达式

        Args:
            stmt: YieldStmt 语句
            d_output: 当前梯度

        Returns:
            反向语句列表
        """
        stmts = []

        # yield 的值是输出，梯度传播到 yield 前的表达式
        # 如果 yield 有多个值，每个值对应一个梯度
        for value_expr in stmt.value:
            if isinstance(value_expr, ir.Call):
                stmts.extend(self._reverse_call(value_expr, d_output))
            elif isinstance(value_expr, ir.Var):
                self._accumulate_gradient(value_expr, d_output)

        return stmts

    def _reverse_assign(self, stmt: ir.AssignStmt, d_output: ir.Var) -> List[ir.Stmt]:
        """
        反向处理赋值语句

        Args:
            stmt: 赋值语句
            d_output: 输出梯度

        Returns:
            反向语句列表
        """
        stmts = []

        # 获取输出变量的梯度
        output_grad = self.gradient_accumulator.get(stmt.var, [d_output])
        if output_grad:
            # 合并梯度
            if len(output_grad) == 1:
                grad = output_grad[0]
            else:
                # 累加多个梯度贡献
                grad = output_grad[0]
                for g in output_grad[1:]:
                    grad = self._create_add(grad, g)
        else:
            return []

        # 处理表达式
        if isinstance(stmt.value, ir.Call):
            call = stmt.value
            stmts.extend(self._reverse_call(call, grad))
        elif isinstance(stmt.value, ir.Var):
            # y = x, dx += dy
            self._accumulate_gradient(stmt.value, grad)
        elif isinstance(stmt.value, ir.ConstInt) or isinstance(stmt.value, ir.ConstFloat):
            # 常量：无梯度
            pass

        return stmts

    def _reverse_call(self, call: ir.Call, d_output: ir.Expr) -> List[ir.Stmt]:
        """
        反向处理 Call 表达式

        Args:
            call: Call 表达式
            d_output: 输出梯度

        Returns:
            反向语句列表
        """
        from .gradient_registry import GradientRegistry

        stmts = []

        # 获取梯度规则
        op_name = call.op.name

        # 尝试获取 tile 或 tensor 版本
        if not GradientRegistry.has(op_name):
            # 尝试 tile.xxx 或 tensor.xxx
            if "tensor" in op_name or "tile" in op_name:
                parts = op_name.split(".")
                if len(parts) >= 2:
                    tile_op = f"tile.{parts[-1]}"
                    tensor_op = f"tensor.{parts[-1]}"
                    if GradientRegistry.has(tile_op):
                        op_name = tile_op
                    elif GradientRegistry.has(tensor_op):
                        op_name = tensor_op

        grad_rule = GradientRegistry.get(op_name)

        if grad_rule is None:
            # 未注册的算子：跳过或报错
            return stmts

        # 准备输入
        saved_inputs = []
        for arg in call.args:
            if isinstance(arg, ir.Var):
                saved_inputs.append(arg)
            else:
                saved_inputs.append(arg)

        # 准备 kwargs
        kwargs = {}
        kwargs["forward_result"] = call

        # 调用梯度规则
        try:
            grad_exprs = grad_rule(saved_inputs, d_output, kwargs)
        except Exception as e:
            return stmts

        # 累加梯度到输入变量
        for arg, grad_expr in zip(call.args, grad_exprs):
            if isinstance(arg, ir.Var):
                self._accumulate_gradient(arg, grad_expr)

        return stmts

    def _accumulate_gradient(self, var: ir.Var, grad_expr: ir.Expr):
        """
        累加梯度到变量

        Args:
            var: 变量
            grad_expr: 梯度表达式
        """
        if var not in self.gradient_accumulator:
            self.gradient_accumulator[var] = []
        self.gradient_accumulator[var].append(grad_expr)

    def _create_add(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
        """
        创建加法表达式

        Args:
            lhs: 左操作数
            rhs: 右操作数

        Returns:
            加法表达式
        """
        # 简化：如果 rhs 是零，直接返回 lhs
        if isinstance(rhs, ir.ConstFloat) and rhs.value == 0.0:
            return lhs
        if isinstance(rhs, ir.ConstInt) and rhs.value == 0:
            return lhs

        # 创建 Call 表达式
        if isinstance(lhs.type, ir.TileType):
            return ir.Call(ir.Op("tile.add"), [lhs, rhs], {}, lhs.type, self.span)
        elif isinstance(lhs.type, ir.TensorType):
            return ir.Call(ir.Op("tensor.add"), [lhs, rhs], {}, lhs.type, self.span)
        elif isinstance(lhs.type, ir.ScalarType):
            return ir.Add(lhs, rhs, lhs.type.dtype, self.span)

        return lhs

    def _create_zero(self, type: ir.Type) -> ir.Expr:
        """
        创建零值

        Args:
            type: 类型

        Returns:
            零值表达式
        """
        if isinstance(type, ir.ScalarType):
            dtype = type.dtype
            if dtype in [DataType.FP16, DataType.FP32, DataType.FP64]:
                return ir.ConstFloat(0.0, dtype, self.span)
            return ir.ConstInt(0, dtype, self.span)

        if isinstance(type, ir.TileType):
            shape_tuple = _to_make_tuple(type.shape, self.span)
            value_expr = ir.ConstFloat(0.0, type.dtype, self.span)
            return _ir_core.create_op_call(
                "tile.full", [shape_tuple, value_expr], {"dtype": type.dtype}, self.span
            )

        if isinstance(type, ir.TensorType):
            shape_tuple = _to_make_tuple(type.shape, self.span)
            value_expr = ir.ConstFloat(0.0, type.dtype, self.span)
            return _ir_core.create_op_call(
                "tensor.full", [shape_tuple, value_expr], {"dtype": type.dtype}, self.span
            )

        return ir.ConstFloat(0.0, DataType.FP32, self.span)


def grad(
    func: Union[ir.Function, Any],
    params: Optional[List[str]] = None,
    grad_output: Optional[str] = None,
    name: Optional[str] = None,
) -> ir.Function:
    """
    Generate gradient function for a forward function.

    Args:
        func: Forward function (IR Function or @pl.function decorated)
        params: List of parameter names to compute gradients (default: all inputs)
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
    # 获取 IR Function
    if hasattr(func, "__pypto_ir__"):
        func_ir = func.__pypto_ir__
    elif isinstance(func, ir.Function):
        func_ir = func
    elif hasattr(func, "functions"):
        # pl.program class
        func_ir = list(func.functions.values())[0]
    else:
        raise TypeError(f"Expected Function or Callable, got {type(func)}")

    # 确定参数
    if params is None:
        params = [p.name_hint for p in func_ir.params]

    # 构建反向函数
    builder = ReverseModeBuilder(func_ir, params, grad_output)
    backward_ir = builder.build()

    # 设置名称
    if name:
        backward_ir = ir.Function(
            name, backward_ir.params, backward_ir.return_types, backward_ir.body, backward_ir.span
        )

    return backward_ir


def value_and_grad(
    func: Union[ir.Function, Any], params: Optional[List[str]] = None, name: Optional[str] = None
) -> Tuple[ir.Function, ir.Function]:
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
    # 获取 IR Function
    if hasattr(func, "__pypto_ir__"):
        func_ir = func.__pypto_ir__
    elif isinstance(func, ir.Function):
        func_ir = func
    elif hasattr(func, "functions"):
        func_ir = list(func.functions.values())[0]
    else:
        raise TypeError(f"Expected Function or Callable, got {type(func)}")

    if params is None:
        params = [p.name_hint for p in func_ir.params]

    # 前向函数（保持原样）
    forward_ir = func_ir

    # 反向函数
    backward_ir = grad(func_ir, params, name=name)

    return forward_ir, backward_ir
