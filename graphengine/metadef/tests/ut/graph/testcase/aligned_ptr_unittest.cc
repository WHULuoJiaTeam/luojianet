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

#include <gtest/gtest.h>
#include "graph/utils/mem_utils.h"
#include <memory>

#define private public
#define protected public
#include "graph/aligned_ptr.h"
#undef private
#undef protected
#include "graph/aligned_ptr.h"

namespace ge
{
  class UtestAlignedPtr : public testing::Test {
   protected:
   void SetUp() {}
   void TearDown() {}
  };
  
  TEST_F(UtestAlignedPtr, Reset_success) {
   auto aligned_ptr = MakeShared<AlignedPtr>(6);
   uint64_t aligned_data = reinterpret_cast<uintptr_t>(aligned_ptr->Get());
   uint64_t base_data = reinterpret_cast<uintptr_t>(aligned_ptr->base_.get());
   auto output = aligned_ptr->Reset();
   uint64_t output_data = reinterpret_cast<uintptr_t>(output.get());
   uint64_t output_base =reinterpret_cast<uintptr_t>(aligned_ptr->base_.get());
   ASSERT_EQ(aligned_data, output_data);
   ASSERT_EQ(output_base, 0);
  }

  TEST_F(UtestAlignedPtr, BuildFromData_success) {
   auto deleter = [](uint8_t *ptr) {
     delete ptr;
     ptr = nullptr;
   };
   uint8_t* data_ptr = new uint8_t(10);
   auto aligned_ptr = AlignedPtr::BuildFromData(data_ptr, deleter);
   uint8_t result = *(aligned_ptr->Get());
   ASSERT_EQ(result, 10);
  }
}
