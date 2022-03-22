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

#include "common/tbe_kernel_store.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {

TBEKernelStore::TBEKernelStore() {}

void TBEKernelStore::AddTBEKernel(const TBEKernelPtr &kernel) {
  AddKernel(kernel);
}

void TBEKernelStore::LoadTBEKernelBinToOpDesc(const std::shared_ptr<ge::OpDesc> &op_desc) const {
  if (op_desc != nullptr) {
    auto kernel_bin = FindKernel(op_desc->GetName());
    if (kernel_bin != nullptr) {
      GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin),
                      GELOGW("LoadKernelTBEBinToOpDesc: SetExtAttr for kernel_bin failed");)
      GELOGI("Load tbe kernel:%s, %zu", kernel_bin->GetName().c_str(), kernel_bin->GetBinDataSize());

      std::string atomic_kernel_name;
      (void) AttrUtils::GetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, atomic_kernel_name);
      if (!atomic_kernel_name.empty()) {
        GELOGI("Get atomic kernel name is %s.", atomic_kernel_name.c_str());
        auto atomic_kernel_bin = FindKernel(atomic_kernel_name);
        GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, atomic_kernel_bin),
                        GELOGW("LoadKernelTBEBinToOpDesc: SetExtAttr for atomic kernel_bin failed");)
      }
    }
  }
}
}  // namespace ge
