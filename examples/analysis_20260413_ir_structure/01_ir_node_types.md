# PyPTO IR 节点种类分析

## 1. 核心基类

### 1.1 IRNode (所有 IR 节点的基类)

**文件**: `include/pypto/ir/core.h`

```cpp
class IRNode {
 public:
  explicit IRNode(Span s) : span_(std::move(s)) {}
  virtual ~IRNode() = default;
  
  // 禁止拷贝/移动 - 所有 IR 节点都是不可变的
  IRNode(IRNode&&) = delete;
  IRNode& operator=(IRNode&&) = delete;
  
  [[nodiscard]] virtual ObjectKind GetKind() const = 0;
  [[nodiscard]] virtual std::string TypeName() const { return "IRNode"; }
  
  Span span_;  // 源码位置
};
```

### 1.2 ObjectKind 枚举

`ObjectKind` 枚举列举了所有 IR 节点类型：

| 分类 | 节点类型 |
|------|----------|
| **基类** | `IRNode`, `Expr`, `Stmt`, `Type` |
| **表达式** | `Var`, `IterArg`, `MemRef`, `Call`, `MakeTuple`, `TupleGetItemExpr`, `ConstInt`, `ConstFloat`, `ConstBool` |
| **二元表达式** | `Add`, `Sub`, `Mul`, `FloorDiv`, `FloorMod`, `FloatDiv`, `Min`, `Max`, `Pow`, `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`, `And`, `Or`, `Xor`, `BitAnd`, `BitOr`, `BitXor`, `BitShiftLeft`, `BitShiftRight` |
| **一元表达式** | `Abs`, `Neg`, `Not`, `BitNot`, `Cast` |
| **语句** | `AssignStmt`, `IfStmt`, `YieldStmt`, `ReturnStmt`, `ForStmt`, `WhileStmt`, `ScopeStmt`, `SeqStmts`, `EvalStmt`, `BreakStmt`, `ContinueStmt` |
| **类型** | `UnknownType`, `MemRefType`, `ScalarType`, `ShapedType`, `TensorType`, `TileType`, `TupleType` |
| **容器** | `Function`, `Program` |
| **操作** | `Op`, `GlobalVar` |

---

## 2. 表达式 IR 节点

**文件**: `include/pypto/ir/expr.h`

### 2.1 Expr 基类

```cpp
class Expr : public IRNode {
 protected:
  TypePtr type_;  // 表达式结果的类型
 public:
  explicit Expr(Span s, TypePtr type = GetUnknownType());
  [[nodiscard]] const TypePtr& GetType() const { return type_; }
};
```

### 2.2 表达式类型

| 类名 | 继承 | 描述 | 关键字段 |
|------|------|------|----------|
| `Expr` | `IRNode` | 所有表达式的基类 | `type_` |
| `Var` | `Expr` | 变量引用 | `unique_id_`, `name_hint_` |
| `IterArg` | `Var` | 循环携带变量 | `initValue_` |
| `MemRef` | `Var` | 内存引用变量 | `addr_`, `size_`, `id_` |
| `Call` | `Expr` | 函数调用 | `OpPtr`, `args`, `kwargs` |
| `MakeTuple` | `Expr` | 元组构造 | `elements_` |
| `TupleGetItemExpr` | `Expr` | 元组元素访问 | `tuple_`, `index_` |

### 2.3 Op 类 (非 IRNode)

```cpp
class Op {
 public:
  std::string name_;
  // 属性类型注册方法
  template <typename T> void SetAttrType(const std::string& key);
  std::type_index GetAttrType(const std::string& key);
  bool HasAttr(const std::string& key);
};

class GlobalVar : public Op {
  // 程序中函数的引用
};
```

---

## 3. 标量表达式 IR 节点

**文件**: `include/pypto/ir/scalar_expr.h`

### 3.1 ScalarExpr 基类

```cpp
class ScalarExpr : public Expr {
 public:
  DataType dtype_;
  ScalarExpr(Span s, DataType dtype);
};
```

### 3.2 常量类型

| 类名 | 描述 | 字段 |
|------|------|------|
| `ConstInt` | 整数常量 | `int64_t value_` |
| `ConstFloat` | 浮点常量 | `double value_` |
| `ConstBool` | 布尔常量 | `bool value_` |

### 3.3 二元表达式

继承自 `BinaryExpr`，所有都具有 `ExprPtr left_` 和 `ExprPtr right_` 操作数。

支持的二元操作：
- 算术: `Add`, `Sub`, `Mul`, `FloorDiv`, `FloorMod`, `FloatDiv`, `Min`, `Max`, `Pow`
- 比较: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`
- 逻辑: `And`, `Or`, `Xor`
- 位运算: `BitAnd`, `BitOr`, `BitXor`, `BitShiftLeft`, `BitShiftRight`

### 3.4 一元表达式

继承自 `UnaryExpr`，所有都具有 `ExprPtr operand_`。

支持的一元操作：`Abs`, `Neg`, `Not`, `BitNot`, `Cast`

---

## 4. 内存引用 IR 节点

**文件**: `include/pypto/ir/memref.h`

```cpp
class MemRef : public Var {
 public:
  ExprPtr addr_;   // 起始地址表达式
  uint64_t size_;  // 字节大小
  uint64_t id_;    // 唯一标识符
  // 构造函数生成名称如 "mem_123" 或 "mem_vec_7"
};
```

---

## 5. 语句 IR 节点

**文件**: `include/pypto/ir/stmt.h`

### 5.1 Stmt 基类

```cpp
class Stmt : public IRNode {
 public:
  explicit Stmt(Span s);
};
```

### 5.2 语句类型

| 类名 | 描述 | 关键字段 |
|------|------|----------|
| `AssignStmt` | 变量赋值 | `VarPtr var_`, `ExprPtr value_` |
| `IfStmt` | 条件分支 | `ExprPtr condition_`, `StmtPtr then_body_`, `optional<StmtPtr> else_body_`, `vector<VarPtr> return_vars_` |
| `ForStmt` | for 循环 | `VarPtr loop_var_`, `ExprPtr start_/stop_/step_`, `vector<IterArgPtr> iter_args_`, `StmtPtr body_`, `vector<VarPtr> return_vars_`, `ForKind kind_` |
| `WhileStmt` | while 循环 | `ExprPtr condition_`, `vector<IterArgPtr> iter_args_`, `StmtPtr body_`, `vector<VarPtr> return_vars_` |
| `ScopeStmt` | 作用域区域 | `ScopeKind scope_kind_`, `StmtPtr body_` |
| `SeqStmts` | 语句序列 | `vector<StmtPtr> stmts_` |
| `YieldStmt` | yield 返回值 | `vector<ExprPtr> value_` |
| `ReturnStmt` | 函数返回 | `vector<ExprPtr> value_` |
| `EvalStmt` | 表达式求值 | `ExprPtr expr_` |
| `BreakStmt` | 循环 break | (无字段) |
| `ContinueStmt` | 循环 continue | (无字段) |

### 5.3 相关枚举

- `ForKind`: `Sequential`, `Parallel`, `Unroll`
- `ScopeKind`: `InCore`, `AutoInCore`, `Cluster`, `Hierarchy`
- `ChunkPolicy`: `LeadingFull`
- `LoopOrigin`: `Original`, `ChunkOuter`, `ChunkInner`, `ChunkRemainder`
- `SplitMode`: `None`, `UpDown`, `LeftRight`

---

## 6. 类型 IR 节点

**文件**: `include/pypto/ir/type.h`

### 6.1 Type 基类

```cpp
class Type {
 public:
  virtual ~Type() = default;
  [[nodiscard]] virtual ObjectKind GetKind() const = 0;
  [[nodiscard]] virtual std::string TypeName() const;
};
```

### 6.2 类型类

| 类名 | 描述 | 关键字段 |
|------|------|----------|
| `Type` | 所有类型的基类 | (抽象) |
| `UnknownType` | 未知/未指定类型 | (单例) |
| `ScalarType` | 标量值类型 | `DataType dtype_` |
| `ShapedType` | 张量/Tile 基类 | `DataType dtype_`, `vector<ExprPtr> shape_`, `optional<MemRefPtr> memref_` |
| `TensorType` | 张量类型 | 继承 `ShapedType`，添加 `optional<TensorView> tensor_view_` |
| `TileType` | Tile 类型 | 继承 `ShapedType`，添加 `optional<TileView> tile_view_`, `optional<MemorySpace> memory_space_` |
| `TupleType` | 类型元组 | `vector<TypePtr> types_` |
| `MemRefType` | 内存引用类型 | (单例) |

### 6.3 View 结构

- `TensorView`: `stride`, `layout` (ND/DN/NZ), `valid_shape`
- `TileView`: `valid_shape`, `stride`, `start_offset`, `blayout`, `slayout`, `fractal`, `pad`

---

## 7. 容器 IR 节点

### 7.1 Function

**文件**: `include/pypto/ir/function.h`

```cpp
class Function : public IRNode {
 public:
  std::string name_;
  FunctionType func_type_;  // Opaque, Orchestration, InCore, AIC, AIV, Group
  optional<Level> level_;   // 层次级别
  optional<Role> role_;     // Orchestrator 或 Worker
  vector<pair<string, any>> attrs_;
  vector<VarPtr> params_;
  vector<ParamDirection> param_directions_;  // In, Out, InOut
  vector<TypePtr> return_types_;
  StmtPtr body_;
};
```

### 7.2 Program

**文件**: `include/pypto/ir/program.h`

```cpp
class Program : public IRNode {
 public:
  std::string name_;
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions_;
};
```

---

## 8. 继承层次总览

```
IRNode (基类)
├── Expr (表达式)
│   ├── Var (变量)
│   │   ├── IterArg (循环携带变量)
│   │   └── MemRef (内存引用)
│   ├── ConstInt, ConstFloat, ConstBool (常量)
│   ├── Call (函数调用)
│   ├── MakeTuple (元组构造)
│   ├── TupleGetItemExpr (元组访问)
│   ├── BinaryExpr (抽象)
│   │   └── Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow,
│   │       Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor,
│   │       BitShiftLeft, BitShiftRight
│   └── UnaryExpr (抽象)
│       └── Abs, Neg, Not, BitNot, Cast
├── Stmt (语句)
│   ├── AssignStmt
│   ├── IfStmt
│   ├── ForStmt
│   ├── WhileStmt
│   ├── ScopeStmt
│   ├── SeqStmts
│   ├── YieldStmt
│   ├── ReturnStmt
│   ├── EvalStmt
│   ├── BreakStmt
│   └── ContinueStmt
├── Function
└── Program

Type (独立层次)
├── UnknownType
├── ScalarType
├── ShapedType
│   ├── TensorType
│   └── TileType
├── TupleType
└── MemRefType

Op (非 IRNode)
└── GlobalVar
```

---

## 9. Python 绑定

**文件**: `python/bindings/modules/ir.cpp`

所有 IR 节点类通过 nanobind 暴露给 Python：
- `IRNode`, `Expr`, `Stmt`, `Function`, `Program`
- 所有表达式类型
- 所有语句类型
- `Op`, `GlobalVar`
- `IRBuilder`

**类型存根**: `python/pypto/pypto_core/ir.pyi`
- 所有 IR 类的完整 Python 类型注解
- `IRVisitor` 和 `IRMutator` 类及其所有 `visit_*` 方法