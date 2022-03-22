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

#ifndef LUOJIANET_MS_CCSRC_PARALLEL_EXECUTOR_STUB_H
#define LUOJIANET_MS_CCSRC_PARALLEL_EXECUTOR_STUB_H

#include <string>
#include <vector>
#include <memory>

namespace luojianet_ms {
namespace parallel {
class Executor {
 public:
  Executor(const std::string &device_name, uint32_t device_id) : device_name_(device_name), device_id_(device_id) {}
  ~Executor() = default;
  bool CreateCommGroup(const std::string &group_name, std::vector<uint32_t> ranks) const { return true; }
  bool DestroyCommGroup(const std::string &group_name) const { return true; }

 private:
  std::string device_name_;
  uint32_t device_id_;
};
}  // namespace parallel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_PARALLEL_EXECUTOR_STUB_H
