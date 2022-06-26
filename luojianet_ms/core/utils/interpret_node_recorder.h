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

#ifndef LUOJIANET_MS_CORE_UTILS_InterpretNodeRecorder_H_
#define LUOJIANET_MS_CORE_UTILS_InterpretNodeRecorder_H_

#include <string>
#include "utils/hash_set.h"
#include "utils/visible.h"

namespace luojianet_ms {
class MS_CORE_API InterpretNodeRecorder {
 public:
  explicit InterpretNodeRecorder(InterpretNodeRecorder &&) = delete;
  explicit InterpretNodeRecorder(const InterpretNodeRecorder &) = delete;
  InterpretNodeRecorder &operator=(const InterpretNodeRecorder &) = delete;
  InterpretNodeRecorder &operator=(InterpretNodeRecorder &&) = delete;
  static InterpretNodeRecorder &GetInstance();

  void PushLineInfo(const std::string &line) { (void)interpret_nodes_lines_.emplace(line); }

  const luojianet_ms::HashSet<std::string> &LineInfos() const { return interpret_nodes_lines_; }

  void Clear() { interpret_nodes_lines_.clear(); }

 protected:
  InterpretNodeRecorder() = default;
  virtual ~InterpretNodeRecorder() = default;

 private:
  luojianet_ms::HashSet<std::string> interpret_nodes_lines_;
};
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CORE_UTILS_InterpretNodeRecorder_H_
