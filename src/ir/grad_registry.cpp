/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pypto/ir/grad_registry.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"

namespace pypto {
namespace ir {

GradRegistry&GradRegistry::GetInstance() {
  static GradRegistry instance;
  return instance;
}

void GradRegistry::RegisterGrad(const std::string& op_name, GradFunc grad_func) {
  CHECK(grad_registry_.find(op_name) == grad_registry_.end())
      << "Gradient function for operator '" + op_name + "' is already registered";
  grad_registry_[op_name] = std::move(grad_func);
}

GradFunc GradRegistry::GetGrad(const std::string& op_name) const {
  auto it = grad_registry_.find(op_name);
  if (it == grad_registry_.end()) {
    return nullptr;
  }
  return it->second;
}

bool GradRegistry::HasGrad(const std::string& op_name) const {
  return grad_registry_.find(op_name) != grad_registry_.end();
}

}  // namespace ir
}  // namespace pypto
