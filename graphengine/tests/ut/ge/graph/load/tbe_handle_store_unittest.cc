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

#include <gtest/gtest.h>

#define protected public
#define private public
#include "graph/load/model_manager/tbe_handle_store.h"
#include "runtime/kernel.h"
#undef protected
#undef private

namespace ge {
class UtestTBEHandleStore : public testing::Test {
 protected:
  void SetUp() {
    TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
    kernel_store.kernels_.clear();
  }

  void TearDown() {
    TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
    kernel_store.kernels_.clear();
  }
};

TEST_F(UtestTBEHandleStore, test_store_tbe_handle) {
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();

  // not in store, can`t find.
  void *handle = nullptr;
  std::string tbe_name0("tbe_kernel_key0");
  EXPECT_FALSE(kernel_store.FindTBEHandle(tbe_name0, handle));
  EXPECT_EQ(handle, nullptr);

  // store first, size is 1, num is 1.
  std::string tbe_name1("tbe_kernel_key1");
  void *tbe_handle1 = (void *)0x12345678;
  std::shared_ptr<OpKernelBin> tbe_kernel = std::shared_ptr<OpKernelBin>();
  kernel_store.StoreTBEHandle(tbe_name1, tbe_handle1, tbe_kernel);
  EXPECT_EQ(kernel_store.kernels_.size(), 1);

  EXPECT_TRUE(kernel_store.FindTBEHandle(tbe_name1, handle));
  EXPECT_EQ(handle, tbe_handle1);

  auto it = kernel_store.kernels_.find(tbe_name1);
  EXPECT_NE(it, kernel_store.kernels_.end());
  TbeHandleInfo &info1 = it->second;
  EXPECT_EQ(info1.handle(), tbe_handle1);
  EXPECT_EQ(info1.used_num(), 1);

  // store second, size is 1, num is 2.
  kernel_store.StoreTBEHandle(tbe_name1, tbe_handle1, tbe_kernel);
  EXPECT_EQ(kernel_store.kernels_.size(), 1);

  EXPECT_TRUE(kernel_store.FindTBEHandle(tbe_name1, handle));
  EXPECT_EQ(handle, tbe_handle1);

  it = kernel_store.kernels_.find(tbe_name1);
  EXPECT_NE(it, kernel_store.kernels_.end());
  TbeHandleInfo &info2 = it->second;
  EXPECT_EQ(info2.handle(), tbe_handle1);
  EXPECT_EQ(info2.used_num(), 2);

  // store other, size is 2, num is 2, num is 1.
  std::string tbe_name2("tbe_kernel_key2");
  void *tbe_handle2 = (void *)0x22345678;
  kernel_store.StoreTBEHandle(tbe_name2, tbe_handle2, tbe_kernel);
  EXPECT_EQ(kernel_store.kernels_.size(), 2);

  EXPECT_TRUE(kernel_store.FindTBEHandle(tbe_name2, handle));
  EXPECT_EQ(handle, tbe_handle2);
  EXPECT_TRUE(kernel_store.FindTBEHandle(tbe_name1, handle));
  EXPECT_EQ(handle, tbe_handle1);

  it = kernel_store.kernels_.find(tbe_name1);
  EXPECT_NE(it, kernel_store.kernels_.end());
  TbeHandleInfo &info3 = it->second;
  EXPECT_EQ(info3.handle(), tbe_handle1);
  EXPECT_EQ(info3.used_num(), 2);

  it = kernel_store.kernels_.find(tbe_name2);
  EXPECT_NE(it, kernel_store.kernels_.end());
  TbeHandleInfo &info4 = it->second;
  EXPECT_EQ(info4.handle(), tbe_handle2);
  EXPECT_EQ(info4.used_num(), 1);

  // For Refer
  kernel_store.ReferTBEHandle(tbe_name0);
  EXPECT_EQ(kernel_store.kernels_.size(), 2);

  kernel_store.ReferTBEHandle(tbe_name1);
  EXPECT_EQ(kernel_store.kernels_.size(), 2);

  // For Erase.
  std::map<std::string, uint32_t> names0 = {{tbe_name0, 1}};
  kernel_store.EraseTBEHandle(names0);
  EXPECT_EQ(kernel_store.kernels_.size(), 2);

  std::map<std::string, uint32_t> names1 = {{tbe_name1, 1}};
  kernel_store.EraseTBEHandle(names1);
  EXPECT_EQ(kernel_store.kernels_.size(), 2);

  std::map<std::string, uint32_t> names2 = {{tbe_name1, 2}, {tbe_name2, 1}};
  kernel_store.EraseTBEHandle(names2);
  EXPECT_EQ(kernel_store.kernels_.size(), 0);
}

TEST_F(UtestTBEHandleStore, test_tbe_handle_info) {
  void *tbe_handle = (void *)0x12345678;
  std::shared_ptr<OpKernelBin> tbe_kernel = std::shared_ptr<OpKernelBin>();
  TbeHandleInfo info(tbe_handle, tbe_kernel);
  EXPECT_EQ(info.used_num(), 0);

  info.used_dec();
  EXPECT_EQ(info.used_num(), 0);

  info.used_inc(std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(info.used_num(), std::numeric_limits<uint32_t>::max());

  info.used_inc();
  EXPECT_EQ(info.used_num(), std::numeric_limits<uint32_t>::max());
}
}  // namespace ge
