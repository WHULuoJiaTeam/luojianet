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

#ifndef GE_COMMON_KERNEL_STORE_H_
#define GE_COMMON_KERNEL_STORE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <securec.h>
#include <utility>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/fmk_types.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"

namespace ge {
using KernelBin = ge::OpKernelBin;
using KernelBinPtr = std::shared_ptr<ge::OpKernelBin>;
using CustAICPUKernel = ge::OpKernelBin;
using CustAICPUKernelPtr = std::shared_ptr<ge::OpKernelBin>;
using TBEKernel = ge::OpKernelBin;
using TBEKernelPtr = std::shared_ptr<ge::OpKernelBin>;

const uint32_t kKernelItemMagic = 0x5d776efd;

struct KernelStoreItemHead {
  uint32_t magic;
  uint32_t name_len;
  uint32_t bin_len;
};

class KernelStore {
 public:
  KernelStore() = default;
  virtual ~KernelStore() = default;
  virtual bool Build();

  virtual bool Load(const uint8_t *data, const size_t &len);

  virtual const uint8_t *Data() const;
  virtual size_t DataSize() const;
  virtual void AddKernel(const KernelBinPtr &kernel);
  virtual KernelBinPtr FindKernel(const std::string &name) const;

 private:
  std::unordered_map<std::string, KernelBinPtr> kernels_;
  std::vector<uint8_t> buffer_;
};
}  // namespace ge

#endif  // GE_COMMON_KERNEL_STORE_H_
