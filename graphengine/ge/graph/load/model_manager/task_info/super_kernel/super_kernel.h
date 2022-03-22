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

#ifndef SUPER_KERNEL_H
#define SUPER_KERNEL_H

#include "framework/common/fmk_error_codes.h"
#include "framework/common/debug/log.h"
#include "runtime/rt.h"

namespace ge {
namespace skt {
class SuperKernel {
 private:
  void *device_args_addr_ = nullptr;
  const void *func_stub_;
  void *dev_nav_table_;
  uint64_t nav_table_size_;
  uint32_t block_dim_;

 public:
  SuperKernel(const void *stub, void *ptr, uint64_t sz, uint32_t dim)
      : func_stub_(stub), dev_nav_table_(ptr), nav_table_size_(sz), block_dim_(dim) {}
  ~SuperKernel() = default;
  Status Launch(rtStream_t stream, uint32_t dump_flag);
  const void *GetFuncStub() const { return func_stub_; }
  uint64_t GetNavTableSize() const { return nav_table_size_; }
  uint32_t GetBlockDim() const { return block_dim_; }
  void *GetNavTablePtr() const { return dev_nav_table_; }
  void *GetDeviceArgsPtr() const { return device_args_addr_; }
};
}  // namespace skt
}  // namespace ge
#endif
