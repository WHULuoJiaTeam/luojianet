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

#include "ge_running_env/info_store_holder.h"
FAKE_NS_BEGIN

namespace {
std::string GenStoreName() {
  static int store_id = 0;
  return "store_" + std::to_string(store_id++);
}
}  // namespace

InfoStoreHolder::InfoStoreHolder(const std::string& kernel_lib_name) : kernel_lib_name_(kernel_lib_name) {}

InfoStoreHolder::InfoStoreHolder() : kernel_lib_name_(GenStoreName()) {}

void InfoStoreHolder::RegistOp(std::string op_type) {
  OpInfo default_op_info = {.engine = engine_name_,
                            .opKernelLib = kernel_lib_name_,
                            .computeCost = 0,
                            .flagPartial = false,
                            .flagAsync = false,
                            .isAtomic = false};

  auto iter = op_info_map_.find(op_type);
  if (iter == op_info_map_.end()) {
    op_info_map_.emplace(op_type, default_op_info);
  }
}

void InfoStoreHolder::EngineName(std::string engine_name) { engine_name_ = engine_name; }

std::string InfoStoreHolder::GetLibName() { return kernel_lib_name_; }

FAKE_NS_END
