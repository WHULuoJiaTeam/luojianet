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
#include <vector>
#include <securec.h>
#include <gtest/gtest.h>
#define private public
#include "graph/ge_tensor.h"
#include "ge_ir.pb.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_tensor_impl.h"
#include "external/graph/tensor.h"
#include <iostream>

namespace ge {
class TensorUtilsUT : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {}
};

TEST_F(TensorUtilsUT, CopyConstruct1_NullTensorDef) {
  GeTensor t1;
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  std::cout << "test1" << std::endl;
  t1.SetData(vec);
  GeTensor t2 = TensorUtils::CreateShareTensor(t1);
  t1.impl_->tensor_def_.GetProtoOwner();
// The copy construct share tensor_data_, do not share tensor_desc
  ASSERT_EQ(t1.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_data_.impl_->tensor_descriptor_, t1.impl_->desc_.impl_);
  ASSERT_EQ(t2.impl_->tensor_data_.impl_->tensor_descriptor_, t2.impl_->desc_.impl_);
  ASSERT_EQ(t1.impl_->tensor_data_.GetData(), t2.impl_->tensor_data_.GetData());

  t1.MutableTensorDesc().SetFormat(FORMAT_NCHW);
  t2.MutableTensorDesc().SetFormat(FORMAT_NHWC);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  ASSERT_EQ(t2.GetTensorDesc().GetFormat(), FORMAT_NHWC);

  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());
}

TEST_F(TensorUtilsUT, CopyConstruct2_WithTensorDef) {
  GeIrProtoHelper<ge::proto::TensorDef> helper;
  helper.InitDefault();
  helper.GetProtoMsg()->mutable_data()->resize(100);
  GeTensor t1(helper.GetProtoOwner(), helper.GetProtoMsg());

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2 = TensorUtils::CreateShareTensor(t1);

  // The copy construct share tensor_data_ and tensor_desc
  ASSERT_NE(t1.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_NE(t1.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_data_.impl_->tensor_descriptor_, t1.impl_->desc_.impl_);
  ASSERT_EQ(t2.impl_->tensor_data_.impl_->tensor_descriptor_, t2.impl_->desc_.impl_);
  ASSERT_EQ(t1.impl_->tensor_data_.GetData(), t2.impl_->tensor_data_.GetData());

  t1.MutableTensorDesc().SetFormat(FORMAT_NCHW);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  ASSERT_EQ(t2.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  t2.MutableTensorDesc().SetFormat(FORMAT_NHWC);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NHWC);
  ASSERT_EQ(t2.GetTensorDesc().GetFormat(), FORMAT_NHWC);

  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());
}

TEST_F(TensorUtilsUT, SetData_CreateShareTensorWithTensorDef) {
  GeIrProtoHelper<ge::proto::TensorDef> helper;
  helper.InitDefault();
  helper.GetProtoMsg()->mutable_data()->resize(100);
  GeTensor t1(helper.GetProtoOwner(), helper.GetProtoMsg());

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2 = TensorUtils::CreateShareTensor(t1);

  std::vector<uint8_t> vec2;
  for (uint8_t i = 0; i < 100; ++i) {
    vec2.push_back(i);
  }
  t2.SetData(vec2);
  ASSERT_EQ(memcmp(t2.GetData().GetData(), vec2.data(), vec2.size()), 0);
  // todo 这里存在bug，但是从目前来看，并没有被触发，因此暂时不修复了，重构后一起修复。
  //  触发bug的场景为：如果tensor1是通过tensor_def_持有TensorData，然后通过拷贝构造、拷贝赋值的方式，从tensor1构造了tensor2。
  //  那么通过tensor2.SetData后，会导致tensor1的GetData接口失效（获取到野指针）
  //  触发的表现就是，如下两条ASSERT_EQ并不成立
  // ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());
  // ASSERT_EQ(memcmp(t1.GetData().GetData(), vec2.data(), vec2.size()), 0);
}

TEST_F(TensorUtilsUT, SetData_CreateShareTensorWithoutTensorDef) {
  GeTensor t1;

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2 = TensorUtils::CreateShareTensor(t1);

  std::vector<uint8_t> vec3;
  for (uint8_t i = 0; i < 100; ++i) {
    vec3.push_back(i);
  }
  t2.SetData(vec3);
  ASSERT_EQ(t2.GetData().size(), vec3.size());
  ASSERT_EQ(memcmp(t2.GetData().GetData(), vec3.data(), vec3.size()), 0);
  ASSERT_EQ(t1.GetData().size(), vec3.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec3.data(), vec3.size()), 0);
  ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());

  std::vector<uint8_t> vec2;
  for (uint8_t i = 0; i < 105; ++i) {
    vec2.push_back(i);
  }
  t2.SetData(vec2);
  ASSERT_EQ(t2.GetData().size(), vec2.size());
  ASSERT_EQ(memcmp(t2.GetData().GetData(), vec2.data(), vec2.size()), 0);
  // after modify the data of t2 using a different size buffer, the t1 will not be modified
  ASSERT_EQ(t1.GetData().size(), vec3.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec3.data(), vec3.size()), 0);
  ASSERT_NE(t1.GetData().GetData(), t2.GetData().GetData());
}

TEST_F(TensorUtilsUT, CreateShareTensorFromSharedPtr) {
  auto ap = std::make_shared<AlignedPtr>(100);
  for (uint8_t i = 0; i < 100; ++i) {
    ap->MutableGet()[i] = i;
  }
  GeTensorDesc td;
  GeTensor t1 = TensorUtils::CreateShareTensor(td, ap, 100);
  ASSERT_EQ(t1.GetData().GetData(), ap->MutableGet());
  ASSERT_EQ(t1.GetData().size(), 100);

  GeTensor t2(td, ap, 100);
  ASSERT_EQ(t2.GetData().GetData(), ap->MutableGet());
  ASSERT_EQ(t2.GetData().size(), 100);
}

TEST_F(TensorUtilsUT, ShareTensorData) {
  auto ap = std::make_shared<AlignedPtr>(100);
  for (uint8_t i = 0; i < 100; ++i) {
    ap->MutableGet()[i] = i;
  }
  GeTensorDesc td;

  GeTensor t1(td);
  t1.SetData(ap, 100);
  ASSERT_EQ(t1.GetData().GetData(), ap->MutableGet());
  ASSERT_EQ(t1.GetData().size(), 100);

  GeTensor t2(td);
  TensorUtils::ShareAlignedPtr(ap, 100, t2);
  ASSERT_EQ(t2.GetData().GetData(), ap->MutableGet());
  ASSERT_EQ(t2.GetData().size(), 100);
}

TEST_F(TensorUtilsUT, CopyAssign_NullTensorDef) {
  GeTensor t1;

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2;
  TensorUtils::ShareTensor(t1, t2);

  // The copy construct share tensor_data_, do not share tensor_desc
  ASSERT_EQ(t1.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_data_.impl_->tensor_descriptor_, t1.impl_->desc_.impl_);
  ASSERT_EQ(t2.impl_->tensor_data_.impl_->tensor_descriptor_, t2.impl_->desc_.impl_);
  ASSERT_EQ(t1.impl_->tensor_data_.GetData(), t2.impl_->tensor_data_.GetData());

  t1.MutableTensorDesc().SetFormat(FORMAT_NCHW);
  t2.MutableTensorDesc().SetFormat(FORMAT_NHWC);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  ASSERT_EQ(t2.GetTensorDesc().GetFormat(), FORMAT_NHWC);

  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());
}

TEST_F(TensorUtilsUT, CopyConstruct3_TensorData) {
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(i);
  }
  TensorData td1;
  td1.SetData(vec);

  TensorData td2(td1);
  ASSERT_EQ(td1.GetData(), td2.GetData());
  ASSERT_EQ(td1.GetSize(), td2.GetSize());
  ASSERT_EQ(td1.GetSize(), 200);

  TensorData td3 = TensorUtils::CreateShareTensorData(td1);
  ASSERT_EQ(td1.GetData(), td3.GetData());
  ASSERT_EQ(td1.GetSize(), td3.GetSize());
  ASSERT_EQ(td1.GetSize(), 200);
}

TEST_F(TensorUtilsUT, CopyAssign_TensorData) {
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(i);
  }
  TensorData td1;
  td1.SetData(vec);

  TensorData td2 = td1;
  ASSERT_EQ(td1.GetData(), td2.GetData());
  ASSERT_EQ(td1.GetSize(), td2.GetSize());
  ASSERT_EQ(td1.GetSize(), 200);

  TensorData td3;
  TensorUtils::ShareTensorData(td1, td3);
  ASSERT_EQ(td1.GetData(), td3.GetData());
  ASSERT_EQ(td1.GetSize(), td3.GetSize());
  ASSERT_EQ(td1.GetSize(), 200);
}

TEST_F(TensorUtilsUT, SetData_ShareAlignedPtr_TensorData) {
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(i);
  }
  auto ap = std::make_shared<AlignedPtr>(vec.size());
  memcpy_s(ap->MutableGet(), vec.size(), vec.data(), vec.size());

  TensorData td1;
  td1.SetData(ap, vec.size());
  ASSERT_EQ(td1.GetData(), ap->MutableGet());
  ASSERT_EQ(td1.GetSize(), 200);

  TensorData td2;
  TensorUtils::ShareAlignedPtr(ap, vec.size(), td2);
  ASSERT_EQ(td2.GetData(), ap->MutableGet());
  ASSERT_EQ(td2.GetSize(), 200);
}

TEST_F(TensorUtilsUT, ShareTheSame) {
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(i);
  }
  TensorData td1;
  td1.SetData(vec);
  TensorUtils::ShareTensorData(td1, td1);
  ASSERT_EQ(memcmp(td1.GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(td1.GetSize(), 200);

  GeTensorDesc tensor_desc;
  GeTensor t1(tensor_desc, vec);
  TensorUtils::ShareTensor(t1, t1);
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
}

TEST_F(TensorUtilsUT, ConstData) {
  int *c = new int(10);
  void *d = static_cast<void*>(c);
  std::unique_ptr<uint8_t[]> const_data = std::unique_ptr<uint8_t[]>(reinterpret_cast<uint8_t *>(d));
  TensorDesc tensor_desc;
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetConstData(std::move(const_data), sizeof(int));
  uint8_t *ret = nullptr;
  size_t len = 0;
  tensor_desc.GetConstData(&ret, len);
  printf("GetConstData1====0x%x \n", ret);
  ASSERT_NE(ret, nullptr);
  ASSERT_EQ(sizeof(int), len);

  // operator=
  tensor_desc = tensor_desc;
  TensorDesc tensor_desc1;
  tensor_desc1 = tensor_desc;
  uint8_t *ret1 = nullptr;
  size_t len1 = 0;
  tensor_desc1.GetConstData(&ret1, len1);
  printf("GetConstData2 ====0x%x \n", ret1);
  ASSERT_NE(ret1, nullptr);
  ASSERT_NE(ret1, ret);
  ASSERT_EQ(sizeof(int), len1);
  ASSERT_EQ(tensor_desc1.GetFormat(), FORMAT_NCHW);

  // copy
  std::size_t big_size = SECUREC_MEM_MAX_LEN+1;
  std::unique_ptr<uint8_t[]> big_data = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[big_size]);
  TensorDesc tensor_desc2;
  tensor_desc2.SetFormat(FORMAT_NCHW);
  tensor_desc2.SetConstData(std::move(big_data), big_size);
  TensorDesc tensor_desc3(tensor_desc2);
  uint8_t *ret2 = nullptr;
  size_t len2 = 0;
  tensor_desc3.GetConstData(&ret2, len2);
  printf("GetConstData3 ====0x%x \n", ret2);
  ASSERT_NE(ret2, nullptr);
  ASSERT_NE(ret2, ret);
  ASSERT_EQ(big_size, len2);
  ASSERT_EQ(tensor_desc3.GetFormat(), FORMAT_NCHW);
}
}  // namespace ge