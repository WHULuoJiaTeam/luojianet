/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef H7992249B_058D_40A1_94EA_52BBCB76434E
#define H7992249B_058D_40A1_94EA_52BBCB76434E

#include "fake_ns.h"
#include "common/opskernel/ops_kernel_info_types.h"

FAKE_NS_BEGIN

struct InfoStoreHolder {
  InfoStoreHolder();
  InfoStoreHolder(const std::string&);
  void EngineName(std::string engine_name);
  void RegistOp(std::string op_type);
  std::string GetLibName();

 protected:
  std::map<std::string, ge::OpInfo> op_info_map_;
  std::string kernel_lib_name_;
  std::string engine_name_;
};

FAKE_NS_END

#endif
