# PyPTO IR 遍历与修改分析

## 1. Visitor 模式实现

PyPTO 通过三个关键类实现经典的访问者模式用于 IR 遍历。

### 1.1 基础 Functor 类

**文件**: `include/pypto/ir/transforms/base/functor.h`

| 类 | 描述 |
|----|------|
| `ExprFunctor<R, Args...>` | 表达式访问者的模板基类，通过 dynamic_cast 实现类型分发 |
| `StmtFunctor<R, Args...>` | 语句访问者的模板基类 |
| `IRFunctor<R, Args...>` | 统一的表达式和语句遍历 functor |

分发机制使用宏 `EXPR_FUNCTOR_DISPATCH(OpType)` 和 `STMT_FUNCTOR_DISPATCH(OpType)` 路由到具体的 `VisitExpr_`/`VisitStmt_` 方法。

### 1.2 IRVisitor（只读遍历）

**文件**: `include/pypto/ir/transforms/base/visitor.h`

- **目的**: 只读 IR 遍历
- **继承**: 继承 `IRFunctor<void>`
- **关键特性**:
  - 默认实现递归遍历所有子节点
  - 提供分组处理器: `VisitVarLike_`, `VisitBinaryExpr_`, `VisitUnaryExpr_`
  - 顶层入口: `VisitProgram`, `VisitFunction`
  - 处理 9 种叶子表达式类型、23 种二元操作、5 种一元操作、11 种语句类型

```cpp
class IRVisitor : public IRFunctor<void> {
  virtual void VisitProgram(const ProgramPtr& program);
  virtual void VisitFunction(const FunctionPtr& func);
  void VisitExpr(const ExprPtr& expr);
  void VisitStmt(const StmtPtr& stmt);
  
  // 表达式处理器: VisitExpr_(VarPtr), VisitExpr_(CallPtr) 等
  // 语句处理器: VisitStmt_(AssignStmtPtr), VisitStmt_(ForStmtPtr) 等
  
  // 便捷方法
  virtual void VisitVarLike_(const VarPtr& op);  // Var & IterArg
  virtual void VisitBinaryExpr_(const BinaryExprPtr& op);
  virtual void VisitUnaryExpr_(const UnaryExprPtr& op);
};
```

### 1.3 Python TraceVisitor 示例

**文件**: `python/pypto/language/grad/trace.py`

```python
class TraceVisitor(IRVisitor):
    """用于收集操作追踪信息的 IR visitor"""
    
    def visit_function(self, func: Function) -> None:
        self.function_name = func.name
        # 访问函数体
        if func.body:
            self.visit_stmt(func.body)
    
    def visit_for_stmt(self, stmt) -> None:
        # ForStmt 的自定义处理
        # 访问循环体以收集循环内的操作
        if hasattr(stmt, 'body') and stmt.body:
            self.visit_stmt(stmt.body)
```

---

## 2. Mutator 模式实现

### 2.1 IRMutator（写时复制转换）

**文件**: `include/pypto/ir/transforms/base/mutator.h`

- **目的**: 带写时复制语义的 IR 转换
- **继承**: 继承 `ExprFunctor<ExprPtr>` 和 `StmtFunctor<StmtPtr>`
- **关键特性**:
  - 未修改时返回原指针（写时复制优化）
  - 仅在子节点改变时重建节点
  - 提供 `var_remap_` 用于变异时的指针重映射
  - 支持与 IRVisitor 类似的分组处理器

```cpp
class IRMutator : public ExprFunctor<ExprPtr>, public StmtFunctor<StmtPtr> {
public:
  ProgramPtr VisitProgram(const ProgramPtr& program);
  FunctionPtr VisitFunction(const FunctionPtr& func);
  ExprPtr VisitExpr(const ExprPtr& expr);
  StmtPtr VisitStmt(const StmtPtr& stmt);
  
  // 表达式变换器返回 ExprPtr
  // 语句变换器返回 StmtPtr
  // 写时复制: 未改变时返回原指针
  
  std::unordered_map<const Expr*, ExprPtr> var_remap_;  // 变量重映射
};
```

### 2.2 IRMutatorWithAnalyzer

**文件**: `include/pypto/ir/arith/ir_mutator_with_analyzer.h`

- **目的**: 需要算术分析的 mutator 基类
- **关键特性**: 自动将 ForStmt 循环变量绑定到迭代范围，实现范围感知简化

### 2.3 示例: FlattenCallExprMutator

**文件**: `src/ir/transforms/flatten_call_expr_pass.cpp`

```cpp
class FlattenCallExprMutator : public IRMutator {
 private:
  int temp_var_counter_ = 0;
  std::vector<StmtPtr> pending_stmts_;  // 累积提取的语句
  
  ExprPtr ExtractCallToTemp(const ExprPtr& expr) {
    // 创建临时变量
    auto temp_var = std::make_shared<Var>(temp_name, expr->GetType(), expr->span_);
    auto assign = std::make_shared<AssignStmt>(temp_var, expr, expr->span_);
    pending_stmts_.push_back(assign);
    return temp_var;
  }
  
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // 在每个变换后的语句前插入待处理语句
  }
};
```

---

## 3. Pass 基础设施

### 3.1 核心 Pass 类

**文件**: `include/pypto/ir/transforms/passes.h`

**PassImpl** (内部基类):

```cpp
class PassImpl {
  virtual ProgramPtr operator()(const ProgramPtr& program) = 0;
  virtual std::string GetName() const;
  virtual IRPropertySet GetRequiredProperties() const;
  virtual IRPropertySet GetProducedProperties() const;
  virtual IRPropertySet GetInvalidatedProperties() const;
};
```

**Pass** (Pimpl 封装):
- 工厂函数: `CreateFunctionPass`, `CreateProgramPass`
- 如果有活动的 PassContext 则执行插桩

**PassPipeline**:
- 有序的 pass 序列
- 带自动属性验证执行
- 可配置阶段的警告检查

### 3.2 Pass Context

**文件**: `include/pypto/ir/transforms/pass_context.h`

| 类 | 功能 |
|----|------|
| `PassInstrument` | pass 前后回调的抽象基类 |
| `VerificationInstrument` | pass 前后的属性验证 |
| `CallbackInstrument` | 用于 IR 导出、日志的用户回调 |
| `ReportInstrument` | 在指定 pass 后生成报告 |
| `WarningInstrument` | 运行警告检查 |
| `PassContext` | 线程局部栈用于上下文嵌套 |

### 3.3 Python PassManager

**文件**: `python/pypto/ir/pass_manager.py`

```python
class PassManager:
    _strategy_passes: dict[OptimizationStrategy, list[PassSpec]] = {}
    
    def run_passes(self, input_ir: Program, dump_ir: bool = False) -> Program:
        # 使用 C++ PassPipeline 进行属性追踪执行
        return self._pipeline.run(input_ir)
```

---

## 4. 内置变换 Pass

**目录**: `src/ir/transforms/`

| Pass 名称 | 目的 | 使用 |
|-----------|------|------|
| `SimplifyExpr` | 代数简化 | IRMutatorWithAnalyzer + RewriteSimplifier |
| `FlattenCallExpr` | 三地址码转换 | IRMutator with pending_stmts_ |
| `ConvertToSSA` | SSA 形式转换 | IRMutator |
| `CtrlFlowTransform` | break/continue 到结构化流 | IRMutator |
| `UnrollLoops` | 循环展开 | IRMutator with substitution |
| `SplitChunkedLoops` | 分块循环拆分 | IRMutator |
| `MemoryReuse` | 基于生命周期的内存复用 | IRVisitor + analysis |
| `ConvertTensorToTileOps` | Tensor 到 Tile op 转换 | IRMutator + OpConversionRegistry |
| `ExpandMixedKernel` | Cube/Vector kernel 拆分 | IRMutator + analysis |

---

## 5. 遍历工具

### 5.1 transform_utils

**文件**: `include/pypto/ir/transforms/utils/transform_utils.h`

```cpp
namespace transform_utils {
  // 按指针标识替换变量
  ExprPtr Substitute(const ExprPtr& expr, 
      const std::unordered_map<const Var*, VarPtr>& var_map);
  StmtPtr Substitute(const StmtPtr& body, 
      const std::unordered_map<const Var*, VarPtr>& var_map);
  
  // 查找 yield 语句
  YieldStmtPtr FindYieldStmt(const StmtPtr& body);
  YieldStmtPtr GetLastYieldStmt(const StmtPtr& body);
  
  // 展平语句序列
  std::vector<StmtPtr> FlattenToStmts(const StmtPtr& stmt);
  
  // 收集定义点
  void CollectDefVars(const StmtPtr& stmt, std::vector<VarPtr>& result);
  
  // Op 分类
  bool IsComputeTensorOp(const std::string& op_name);
}
```

### 5.2 ParentStmtAnalysis

**文件**: `include/pypto/ir/transforms/utils/parent_stmt_analysis.h`

```cpp
class ParentStmtAnalysis : public IRVisitor {
  void BuildMap(const FunctionPtr& func);
  StmtPtr GetParent(const StmtPtr& stmt) const;
  bool HasParent(const StmtPtr& stmt) const;
private:
  std::unordered_map<StmtPtr, StmtPtr> parent_map_;
  StmtPtr current_parent_;  // 遍历时的当前父节点
};
```

---

## 6. 重写/模式匹配

### 6.1 RewriteSimplifier

**文件**: `src/ir/arith/rewrite_simplify.h`

- 用于代数简化的模式匹配重写引擎
- 继承 `ExprFunctor<ExprPtr>`
- 关键特性:
  - 递归重写最大深度 `kMaxRecursiveDepth`
  - 通过 `EnterConstraint` 进行约束感知简化
  - 通过 `var_map_` 进行变量替换
  - 基于边界的比较证明

### 6.2 Analyzer

**文件**: `include/pypto/ir/arith/analyzer.h`

```cpp
class Analyzer {
  ConstIntBoundAnalyzer const_int_bound;    // 整数边界分析
  ModularSetAnalyzer modular_set;           // 模集分析
  RewriteSimplifier rewrite_simplify;       // 重写简化器
  TransitiveComparisonAnalyzer transitive_cmp;  // 传递比较分析
  IntSetAnalyzer int_set;                   // 整数集合分析
  
  void Bind(const VarPtr& var, const ExprPtr& bound_value);
  void Unbind(const VarPtr& var);
  ExprPtr Simplify(const ExprPtr& expr);
  std::function<void()> GetConstraintContext(const ExprPtr& constraint);
};
```

---

## 7. Python 绑定

**文件**: `python/bindings/modules/functor.cpp`

- Trampoline 类 `PyIRVisitor` 和 `PyIRMutator` 启用 Python 子类化
- 所有 visit 方法通过 `base_*` 转发器暴露用于调用父类实现

```python
class MyVisitor(IRVisitor):
    def visit_for_stmt(self, stmt):
        # 自定义处理
        self.visit_stmt(stmt.body)  # 继续遍历
```

---

## 8. 架构总结图

```
IR 遍历与修改架构:
┌─────────────────────────────────────────────────────────────┐
│                    PassPipeline                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Pass 1      │→ │ Pass 2      │→ │ Pass N      │          │
│  │ (Program)   │  │ (Function)  │  │ (Program)   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         ↓               ↓               ↓                   │
│  ┌─────────────────────────────────────────────┐            │
│  │              PassContext                      │            │
│  │  ┌──────────────┐ ┌──────────────┐           │            │
│  │  │ Verification │ │ Callback     │           │            │
│  │  │ Instrument   │ │ Instrument   │           │            │
│  │  └──────────────┘ └──────────────┘           │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                  IR Transformation                           │
│  ┌─────────────────┐       ┌─────────────────┐              │
│  │ IRVisitor       │       │ IRMutator       │              │
│  │ (只读遍历)       │       │ (写时复制)       │              │
│  └─────────────────┘       └─────────────────┘              │
│         ↓                           ↓                        │
│  ┌─────────────────────────────────────────────┐            │
│  │            ExprFunctor / StmtFunctor         │            │
│  │         (通过 As<T> 实现类型分发)             │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                 Traversal Utilities                          │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐      │
│  │ Substitute    │ │ FlattenToStmts│ │ ParentStmt    │      │
│  │ (变量替换)     │ │ (展平语句)     │ │ Analysis      │      │
│  └───────────────┘ └───────────────┘ └───────────────┘      │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                 Arithmetic Analysis                          │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐      │
│  │ Analyzer      │ │ Rewrite       │ │ IRMutatorWith │      │
│  │ (分析器)       │ │ Simplifier    │ │ Analyzer      │      │
│  └───────────────┘ └───────────────┘ └───────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 关键设计模式

| 模式 | 实现 |
|------|------|
| **Visitor 模式** | 通过 functor 基类实现双重分发 |
| **Pimpl 模式** | Pass 类通过 PassImpl 隐藏实现 |
| **写时复制** | IRMutator 未改变时返回原指针 |
| **线程局部上下文** | PassContext 栈启用嵌套插桩 |
| **属性追踪** | Pass 声明 required/produced/invalidated 属性用于验证 |

---

## 10. 使用示例

### 10.1 自定义 Visitor

```python
# Python
from pypto.ir import IRVisitor

class MyVisitor(IRVisitor):
    def visit_call(self, op):
        print(f"Found call to: {op.op.name}")
        # 继续遍历参数
        for arg in op.args:
            self.visit_expr(arg)
    
    def visit_for_stmt(self, stmt):
        print(f"For loop: {stmt.loop_var.name_hint}")
        self.visit_stmt(stmt.body)

# 使用
visitor = MyVisitor()
visitor.visit_function(func)
```

### 10.2 自定义 Mutator

```python
# Python
from pypto.ir import IRMutator

class MyMutator(IRMutator):
    def visit_add(self, op):
        # 简化 x + 0 -> x
        if isinstance(op.right, ConstInt) and op.right.value == 0:
            return op.left
        return self.base_visit_add(op)

# 使用
mutator = MyMutator()
new_func = mutator.visit_function(func)
```

### 10.3 C++ Pass 示例

```cpp
class MyPass : public PassImpl {
public:
  std::string GetName() const override { return "MyPass"; }
  
  ProgramPtr operator()(const ProgramPtr& program) override {
    auto mutator = MyMutator();
    auto result = std::make_shared<Program>(*program);
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_funcs;
    for (const auto& [gv, func] : program->functions_) {
      new_funcs[gv] = mutator.VisitFunction(func);
    }
    result->functions_ = std::move(new_funcs);
    return result;
  }
};
```

---

## 11. 文件位置总结

| 组件 | 文件 |
|------|------|
| Functor 基类 | `include/pypto/ir/transforms/base/functor.h` |
| IRVisitor | `include/pypto/ir/transforms/base/visitor.h` |
| IRMutator | `include/pypto/ir/transforms/base/mutator.h` |
| Pass 基础设施 | `include/pypto/ir/transforms/passes.h` |
| PassContext | `include/pypto/ir/transforms/pass_context.h` |
| 内置 Pass | `src/ir/transforms/*.cpp` |
| Transform 工具 | `include/pypto/ir/transforms/utils/transform_utils.h` |
| ParentStmtAnalysis | `include/pypto/ir/transforms/utils/parent_stmt_analysis.h` |
| Analyzer | `include/pypto/ir/arith/analyzer.h` |
| RewriteSimplifier | `src/ir/arith/rewrite_simplify.h` |
| Python 绑定 | `python/bindings/modules/functor.cpp` |
| Python PassManager | `python/pypto/ir/pass_manager.py` |