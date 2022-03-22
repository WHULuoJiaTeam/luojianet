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

#include <memory>
#include <vector>

#define private public
#define protected public
#include "graph/object_pool.h"
#undef private
#undef protected
#include "graph/ge_tensor.h"

using std::vector;
namespace ge {
class UTObjectPool : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTObjectPool, Add) {
  ObjectPool<GeTensor, 10> object_pool_;
  ASSERT_TRUE(object_pool_.IsEmpty());

  auto ge_tensor = object_pool_.Acquire();
  GeTensorDesc tensor_desc(GeShape({10}));
  ge_tensor->SetTensorDesc(tensor_desc);

  float dt[10] = {1.0f};
  auto deleter = [](const uint8_t *ptr) {

  };
  ge_tensor->SetData((uint8_t *)&dt, sizeof(dt), deleter);
  object_pool_.Release(std::move(ge_tensor));
  ASSERT_EQ(object_pool_.handlers_.size(), 1);
}

TEST_F(UTObjectPool, UniqueToShared) {
  ObjectPool<GeTensor, 10> object_pool_;
  auto ge_tensor = object_pool_.Acquire();
  GeTensorDesc tensor_desc(GeShape({10}));
  ge_tensor->SetTensorDesc(tensor_desc);

  float dt[10] = {1.0f};
  auto deleter = [](const uint8_t *ptr) {

  };
  ge_tensor->SetData((uint8_t *)&dt, sizeof(dt), deleter);

  {
    std::shared_ptr<GeTensor> shared_tensor(ge_tensor.get(), [](GeTensor *){});
  }
  ASSERT_NE(ge_tensor, nullptr);
  object_pool_.Release(std::move(ge_tensor));
  ASSERT_EQ(object_pool_.handlers_.size(), 1);
}

TEST_F(UTObjectPool, GetFromFull) {
  ObjectPool<GeTensor, 1> object_pool_;

  auto ge_tensor = object_pool_.Acquire();
  GeTensorDesc tensor_desc(GeShape({10}));
  ge_tensor->SetTensorDesc(tensor_desc);
  float dt[10] = {1.0f};
  auto deleter = [](const uint8_t *ptr) {
  };
  ge_tensor->SetData((uint8_t *)&dt, sizeof(dt), deleter);
  object_pool_.Release(std::move(ge_tensor));

  ASSERT_TRUE(object_pool_.IsFull());
  auto tmp = object_pool_.Acquire();
  ASSERT_TRUE(object_pool_.IsEmpty());
}


TEST_F(UTObjectPool, AutoRelease) {
  ObjectPool<GeTensor, 10> object_pool_;
  auto ge_tensor = object_pool_.Acquire();
  GeTensorDesc tensor_desc(GeShape({10}));
  ge_tensor->SetTensorDesc(tensor_desc);

  float dt[10] = {1.0f};
  auto deleter = [](const uint8_t *ptr) {
  };
  ge_tensor->SetData((uint8_t *)&dt, sizeof(dt), deleter);
  {
    std::queue<std::unique_ptr<GeTensor>> shared_tensors_;
    shared_tensors_.push(std::move(ge_tensor));

  }
  ASSERT_EQ(ge_tensor, nullptr);
  object_pool_.Release(std::move(ge_tensor));
  ASSERT_TRUE(object_pool_.IsEmpty());
}
}
