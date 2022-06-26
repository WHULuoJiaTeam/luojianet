/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_BASE_H_
#define MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_BASE_H_
#include <set>
#include <mutex>
#include <memory>

#include "ir/anf.h"

namespace mindspore {
class FuncGraphBase;
using FuncGraphBasePtr = std::shared_ptr<FuncGraphBase>;
class MS_CORE_API FuncGraphLoopBreaker {
 public:
  ~FuncGraphLoopBreaker();

  static FuncGraphLoopBreaker &Inst();

  void RegFuncGraphBase(FuncGraphBase *graph) {
    std::lock_guard<std::mutex> lock_set(func_mutex_);
    (void)func_set_.insert(graph);
  }
  void UnRegFuncGraphBase(FuncGraphBase *graph) {
    std::lock_guard<std::mutex> lock_set(func_mutex_);
    (void)func_set_.erase(graph);
  }

  void BreakLoop();

 private:
  FuncGraphLoopBreaker() = default;
  std::set<FuncGraphBase *> func_set_;
  std::mutex func_mutex_;
};

class FuncGraphBase : public Value {
 public:
  FuncGraphBase() {
    FuncGraphLoopBreaker::Inst().RegFuncGraphBase(this);
    reg_flg = true;
  }

  ~FuncGraphBase() override {
    if (reg_flg) {
      FuncGraphLoopBreaker::Inst().UnRegFuncGraphBase(this);
    }
  }
  MS_DECLARE_PARENT(FuncGraphBase, Value);

  // Clear the member of FuncGraph to break loop
  virtual void DoBreakLoop() = 0;

 protected:
  friend FuncGraphLoopBreaker;
  bool reg_flg = false;
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_BASE_H_
