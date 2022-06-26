/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include "common/common_test.h"
#define private public
#define protected public
#include "plugin/device/cpu/kernel/unique_with_pad_cpu_kernel.h"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
class UniqueWithPadCpuKernelTest : public UT::Common {
 public:
  UniqueWithPadCpuKernelTest() : unique_with_pad_(std::make_shared<UniqueWithPadCpuKernelMod>()) {}

  void SetUp() override {
    unique_with_pad_->input_size_ = 10;
    unique_with_pad_->dtype_ = kNumberTypeInt64;
    inputs_.clear();
    workspace_.clear();
    outputs_.clear();
  }

  AddressPtr CreateKernelAddress(void *addr, size_t size) {
    auto kernel_addr = std::make_shared<Address>();
    kernel_addr->addr = addr;
    kernel_addr->size = size;
    return kernel_addr;
  }

  void CreateAddress(size_t type_size) {
    inputs_.push_back(CreateKernelAddress(x_.data(), x_.size() * type_size));
    inputs_.push_back(CreateKernelAddress(&pad_dim_, type_size));
    outputs_.push_back(CreateKernelAddress(out_.data(), out_.size() * type_size));
    outputs_.push_back(CreateKernelAddress(idx_.data(), idx_.size() * type_size));
    workspace_.push_back(CreateKernelAddress(workspace_idx_.data(), workspace_idx_.size() * type_size));
    workspace_.push_back(CreateKernelAddress(workspace_idx_.data(), workspace_idx_.size() * type_size));
    workspace_.push_back(CreateKernelAddress(workspace_idx_.data(), workspace_idx_.size() * type_size));
  }

  std::vector<int64_t> x_;
  int64_t pad_dim_;
  std::vector<int64_t> out_;
  std::vector<int64_t> idx_;
  std::vector<int64_t> workspace_idx_;
  std::vector<AddressPtr> inputs_;
  std::vector<AddressPtr> workspace_;
  std::vector<AddressPtr> outputs_;
  std::shared_ptr<UniqueWithPadCpuKernelMod> unique_with_pad_;
};

TEST_F(UniqueWithPadCpuKernelTest, compute_test) {
  x_ = {1, 1, 5, 5, 4, 4, 3, 3, 2, 2};
  pad_dim_ = 8;
  out_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  idx_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  workspace_idx_ = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  CreateAddress(sizeof(int64_t));
  unique_with_pad_->Launch(inputs_, workspace_, outputs_);

  // check compute result
  std::vector<int64_t> expect_out{1, 5, 4, 3, 2, 8, 8, 8, 8, 8};
  std::vector<int64_t> expect_idx{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
  EXPECT_TRUE(out_ == expect_out);
  EXPECT_TRUE(idx_ == expect_idx);
}
}  // namespace kernel
}  // namespace mindspore
