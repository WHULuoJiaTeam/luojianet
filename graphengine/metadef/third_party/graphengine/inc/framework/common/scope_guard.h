/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INC_FRAMEWORK_COMMON_SCOPE_GUARD_H_
#define INC_FRAMEWORK_COMMON_SCOPE_GUARD_H_

#include <functional>
#include <iostream>

/// Usage:
/// Acquire Resource 1
/// MAKE_GUARD([&] { Release Resource 1 })
/// Acquire Resource 2
// MAKE_GUARD([&] { Release Resource 2 })
#define GE_MAKE_GUARD(var, callback) ScopeGuard make_guard_##var(callback)
#define GE_DISMISS_GUARD(var) make_guard_##var.Dismiss()

namespace ge {
class GE_FUNC_VISIBILITY ScopeGuard {
 public:
  // Noncopyable
  ScopeGuard(ScopeGuard const &) = delete;
  ScopeGuard &operator=(ScopeGuard const &) = delete;

  explicit ScopeGuard(const std::function<void()> &on_exit_scope) : on_exit_scope_(on_exit_scope), dismissed_(false) {}

  ~ScopeGuard() {
    if (!dismissed_) {
      if (on_exit_scope_ != nullptr) {
        try {
          on_exit_scope_();
        } catch (std::bad_function_call &e) { }
          catch (...) { }
      }
    }
  }

  void Dismiss() { dismissed_ = true; }

 private:
  std::function<void()> on_exit_scope_;
  bool dismissed_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_SCOPE_GUARD_H_
