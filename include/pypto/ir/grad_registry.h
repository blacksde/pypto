/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file grad_registry.h
 * @brief Gradient function registry for automatic differentiation
 *
 * This file provides a gradient function registry system that enables
 * storing and retrieving gradient computation functions for automatic
 * differentiation.
 */

#ifndef PYPTO_IR_GRAD_REGISTRY_H_
#define PYPTO_IR_GRAD_REGISTRY_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {

/**
 * @brief Gradient computation function type
 *
 * A gradient function takes the original operator inputs and the gradient
 * of the output, and returns the gradients for each input.
 *
 * @param inputs Vector of input expressions to the original operator
 * @param grad_output Gradient of the operator's output
 * @return Vector of gradients for each input
 */
using GradFunc = std::function<std::vector<ExprPtr>(const std::vector<ExprPtr>& inputs, ExprPtr grad_output)>;

/**
 * @brief Gradient function registry (singleton)
 *
 * Manages registration and lookup of gradient computation functions
 * for automatic differentiation.
 *
 * Thread-safety: The registry is not thread-safe during registration.
 * Register all gradient functions during initialization before concurrent access.
 */
class GradRegistry {
 public:
  // Disable copy and move
  GradRegistry(const GradRegistry&) = delete;
  GradRegistry& operator=(const GradRegistry&) = delete;
  GradRegistry(GradRegistry&&) = delete;
  GradRegistry& operator=(GradRegistry&&) = delete;

  /**
   * @brief Get the singleton instance
   *
   * @return Reference to the global gradient registry
   */
  static GradRegistry& GetInstance();

  /**
   * @brief Register a gradient computation function
   *
   * @param op_name Name of the operator
   * @param grad_func Gradient computation function
   * @throws ValueError if gradient function is already registered for this operator
   */
  void RegisterGrad(const std::string& op_name, GradFunc grad_func);

  /**
   * @brief Get the gradient function for an operator
   *
   * @param op_name Name of the operator
   * @return Gradient function, or nullptr if not registered
   */
  [[nodiscard]] GradFunc GetGrad(const std::string& op_name) const;

  /**
   * @brief Check if an operator has a registered gradient
   *
   * @param op_name Name of the operator
   * @return true if gradient is registered
   */
  [[nodiscard]] bool HasGrad(const std::string& op_name) const;

 private:
  GradRegistry() = default;
  ~GradRegistry() = default;

  std::unordered_map<std::string, GradFunc> grad_registry_;
};

/**
 * @brief Helper macro for gradient function registration
 *
 * Use this macro to register gradient functions in initialization code:
 * @code
 * REGISTER_GRAD("tensor.add", [](const auto& inputs, auto grad_output) {
 *     return {grad_output, grad_output};
 * });
 * @endcode
 */
#define REGISTER_GRAD(OpName, GradFunc) \
  ::pypto::ir::GradRegistry::GetInstance().RegisterGrad(OpName, GradFunc)

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_GRAD_REGISTRY_H_
