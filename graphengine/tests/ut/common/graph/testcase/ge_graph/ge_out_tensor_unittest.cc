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
#include <iostream>

#include "graph/ge_attr_value.h"
#include "graph/tensor.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/tensor_utils.h"

using namespace std;
using namespace ge;

class UtestGeOutTensor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeOutTensor, shape) {
  Shape a;
  EXPECT_EQ(a.GetDim(0), 0);
  EXPECT_EQ(a.GetShapeSize(), 0);
  EXPECT_EQ(a.SetDim(0, 0), GRAPH_FAILED);

  vector<int64_t> vec({1, 2, 3, 4});
  Shape b(vec);
  Shape c({1, 2, 3, 4});
  EXPECT_EQ(c.GetDimNum(), 4);
  EXPECT_EQ(c.GetDim(2), 3);
  EXPECT_EQ(c.GetDim(5), 0);
  EXPECT_EQ(c.SetDim(10, 0), GRAPH_FAILED);

  EXPECT_EQ(c.SetDim(2, 2), GRAPH_SUCCESS);
  EXPECT_EQ(c.GetDim(2), 2);
  vector<int64_t> vec1 = c.GetDims();
  EXPECT_EQ(c.GetDim(0), vec1[0]);
  EXPECT_EQ(c.GetDim(1), vec1[1]);
  EXPECT_EQ(c.GetDim(2), vec1[2]);
  EXPECT_EQ(c.GetDim(3), vec1[3]);

  EXPECT_EQ(c.GetShapeSize(), 16);
}

TEST_F(UtestGeOutTensor, tensor_desc) {
  TensorDesc a;
  Shape s({1, 2, 3, 4});
  TensorDesc b(s);
  Shape s1 = b.GetShape();
  EXPECT_EQ(s1.GetDim(0), s.GetDim(0));
  auto shape_m1 = b.GetShape();
  shape_m1.SetDim(0, 2);
  b.SetShape(shape_m1);
  EXPECT_EQ(b.GetShape().GetDim(0), 2);
  Shape s2({3, 2, 3, 4});
  b.SetShape(s2);
  EXPECT_EQ(b.GetShape().GetDim(0), 3);

  EXPECT_EQ(b.GetFormat(), FORMAT_NCHW);
  b.SetFormat(FORMAT_RESERVED);
  EXPECT_EQ(b.GetFormat(), FORMAT_RESERVED);

  EXPECT_EQ(b.GetDataType(), DT_FLOAT);
  b.SetDataType(DT_INT8);
  EXPECT_EQ(b.GetDataType(), DT_INT8);

  TensorDesc c;
  c.Update(Shape({1}), FORMAT_NCHW);
  c.Update(s, FORMAT_NCHW);
  c.SetSize(1);

  TensorDesc d;
  d = c;  // Clone;
  EXPECT_EQ(d.GetSize(), 1);
  d.SetSize(12);
  EXPECT_EQ(d.GetSize(), 12);

  TensorDesc e = c;
  EXPECT_EQ(e.GetSize(), 1);

  TensorDesc f = c;
  EXPECT_EQ(f.GetSize(), 1);
}

TEST_F(UtestGeOutTensor, tensor) {
  Shape s({1, 2, 3, 4});
  TensorDesc tensor_desc(s);
  std::vector<uint8_t> data({1, 2, 3, 4});
  Tensor a;
  Tensor b(tensor_desc);
  Tensor c(tensor_desc, data);
  Tensor d(tensor_desc, data.data(), data.size());

  ASSERT_EQ(a.GetSize(), 0);
  ASSERT_EQ(b.GetSize(), 0);
  ASSERT_EQ(c.GetSize(), data.size());
  ASSERT_EQ(d.GetSize(), data.size());
  EXPECT_EQ(c.GetData()[0], uint8_t(1));
  EXPECT_EQ(c.GetData()[1], uint8_t(2));
  EXPECT_EQ(d.GetData()[2], uint8_t(3));
  EXPECT_EQ(d.GetData()[3], uint8_t(4));
  EXPECT_EQ(d.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(b.GetTensorDesc().GetShape().GetDim(0), 1);
  EXPECT_EQ(c.GetTensorDesc().GetShape().GetDim(1), 2);
  EXPECT_EQ(d.GetTensorDesc().GetShape().GetDim(2), 3);

  Shape s1 = b.GetTensorDesc().GetShape();
  EXPECT_EQ(s1.GetDim(0), 1);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_FLOAT);
  EXPECT_EQ(b.GetTensorDesc().GetFormat(), FORMAT_NCHW);

  auto tensor_desc_m1 = b.GetTensorDesc();
  tensor_desc_m1.SetDataType(DT_INT8);
  b.SetTensorDesc(tensor_desc_m1);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_INT8);
  EXPECT_EQ(b.GetTensorDesc().GetFormat(), FORMAT_NCHW);

  EXPECT_EQ(b.GetTensorDesc().GetSize(), 0);
  auto tensor_desc_m2 = b.GetTensorDesc();
  tensor_desc_m2.SetFormat(FORMAT_NC1HWC0);
  tensor_desc_m2.SetSize(112);
  b.SetTensorDesc(tensor_desc_m2);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_INT8);
  EXPECT_EQ(b.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(b.GetTensorDesc().GetSize(), 112);

  auto data1 = c.GetData();
  c.SetData(data);
  c.SetData(data.data(), data.size());
  EXPECT_EQ(c.GetSize(), data.size());
  EXPECT_EQ(c.GetData()[0], uint8_t(1));
  EXPECT_EQ(c.GetData()[1], uint8_t(2));
  EXPECT_EQ(c.GetData()[2], uint8_t(3));
  EXPECT_EQ(c.GetData()[3], uint8_t(4));

  Tensor e(std::move(tensor_desc), std::move(data));
  EXPECT_EQ(e.GetSize(), data.size());
  EXPECT_EQ(e.GetData()[2], uint8_t(3));

  Tensor f = e.Clone();
  e.GetData()[2] = 5;
  EXPECT_EQ(e.GetData()[2], uint8_t(5));
  EXPECT_EQ(f.GetSize(), data.size());
  EXPECT_EQ(f.GetData()[2], uint8_t(3));
}

TEST_F(UtestGeOutTensor, test_shape_copy) {
  Shape shape;
  EXPECT_EQ(shape.GetDimNum(), 0);

  Shape shape2 = shape;
  EXPECT_EQ(shape2.GetDimNum(), 0);

  Shape shape3({1, 2, 3});
  shape2 = shape3;
  EXPECT_EQ(shape2.GetDimNum(), 3);
  EXPECT_EQ(shape3.GetDimNum(), 3);
}

TEST_F(UtestGeOutTensor, test_tensor_adapter_as_ge_tensor) {
  TensorDesc tensor_desc(Shape({2, 3, 4, 5}), FORMAT_NC1HWC0, DT_FLOAT16);
  tensor_desc.SetSize(120);
  vector<uint8_t> data = {3, 4, 5, 6, 7, 8};
  Tensor tensor(tensor_desc, data);

  GeTensor ge_tensor = TensorAdapter::AsGeTensor(tensor);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetDataType(), DT_FLOAT16);
  uint32_t size = 0;
  TensorUtils::GetSize(ge_tensor.GetTensorDesc(), size);
  EXPECT_EQ(size, 120);
  auto dims = ge_tensor.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims.size(), 4);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[3], 5);
  EXPECT_EQ(ge_tensor.GetData().GetSize(), 6);
  EXPECT_EQ(ge_tensor.GetData().GetData()[0], 3);
  EXPECT_EQ(ge_tensor.GetData().GetData()[5], 8);

  auto ge_tensor_ptr = TensorAdapter::AsGeTensorPtr(tensor);
  EXPECT_EQ(ge_tensor_ptr->GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(ge_tensor_ptr->GetTensorDesc().GetDataType(), DT_FLOAT16);

  const Tensor tensor2 = tensor;
  const GeTensor ge_tensor2 = TensorAdapter::AsGeTensor(tensor2);
  EXPECT_EQ(ge_tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(ge_tensor2.GetTensorDesc().GetDataType(), DT_FLOAT16);
  TensorUtils::GetSize(ge_tensor2.GetTensorDesc(), size);
  EXPECT_EQ(size, 120);
  auto dims2 = ge_tensor2.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims2.size(), 4);
  EXPECT_EQ(dims2[0], 2);
  EXPECT_EQ(dims2[3], 5);
  EXPECT_EQ(ge_tensor2.GetData().GetSize(), 6);
  EXPECT_EQ(ge_tensor2.GetData().GetData()[0], 3);
  EXPECT_EQ(ge_tensor2.GetData().GetData()[5], 8);

  auto ge_tensor_ptr2 = TensorAdapter::AsGeTensorPtr(tensor2);
  EXPECT_EQ(ge_tensor_ptr2->GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(ge_tensor_ptr2->GetTensorDesc().GetDataType(), DT_FLOAT16);

  // modify format
  ge_tensor.MutableTensorDesc().SetFormat(FORMAT_NC1C0HWPAD);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(ge_tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);

  EXPECT_EQ(ge_tensor_ptr->GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(ge_tensor_ptr2->GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);

  // modify datatype
  tensor_desc.SetDataType(DT_INT32);
  tensor.SetTensorDesc(tensor_desc);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(tensor2.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(ge_tensor2.GetTensorDesc().GetDataType(), DT_INT32);

  EXPECT_EQ(ge_tensor_ptr->GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(ge_tensor_ptr2->GetTensorDesc().GetDataType(), DT_INT32);
}

TEST_F(UtestGeOutTensor, test_tensor_adapter_as_tensor) {
  GeTensorDesc ge_tensor_desc(GeShape({2, 3, 4, 5}), FORMAT_NC1HWC0, DT_FLOAT16);
  TensorUtils::SetSize(ge_tensor_desc, 120);
  vector<uint8_t> data = {3, 4, 5, 6, 7, 8};
  GeTensor ge_tensor(ge_tensor_desc, data);

  Tensor tensor = TensorAdapter::AsTensor(ge_tensor);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_FLOAT16);
  EXPECT_EQ(tensor.GetTensorDesc().GetSize(), 120);

  auto dims = tensor.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims.size(), 4);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[3], 5);
  EXPECT_EQ(tensor.GetSize(), 6);
  EXPECT_EQ(tensor.GetData()[0], 3);
  EXPECT_EQ(tensor.GetData()[5], 8);

  const GeTensor ge_tensor2 = ge_tensor;
  const Tensor tensor2 = TensorAdapter::AsTensor(ge_tensor2);
  EXPECT_EQ(tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(tensor2.GetTensorDesc().GetDataType(), DT_FLOAT16);
  EXPECT_EQ(tensor2.GetTensorDesc().GetSize(), 120);
  auto dims2 = tensor2.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims2.size(), 4);
  EXPECT_EQ(dims2[0], 2);
  EXPECT_EQ(dims2[3], 5);
  EXPECT_EQ(tensor2.GetSize(), 6);
  EXPECT_EQ(tensor2.GetData()[0], 3);
  EXPECT_EQ(tensor2.GetData()[5], 8);

  // modify format
  ge_tensor.MutableTensorDesc().SetFormat(FORMAT_NC1C0HWPAD);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(ge_tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor2.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);

  // modify datatype
  auto tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(ge_tensor_desc);
  tensor_desc.SetDataType(DT_INT32);
  tensor.SetTensorDesc(tensor_desc);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(tensor2.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(ge_tensor2.GetTensorDesc().GetDataType(), DT_INT32);
}

TEST_F(UtestGeOutTensor, test_tensor_adapter_transfer2_ge_tensor) {
  TensorDesc tensor_desc(Shape({2, 3, 4, 5}), FORMAT_NC1HWC0, DT_FLOAT16);
  tensor_desc.SetSize(120);
  vector<uint8_t> data = {3, 4, 5, 6, 7, 8};
  Tensor tensor(tensor_desc, data);

  auto get_tensor_ptr = TensorAdapter::Tensor2GeTensor(tensor);

  EXPECT_EQ(get_tensor_ptr->GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(get_tensor_ptr->GetTensorDesc().GetDataType(), DT_FLOAT16);
  uint32_t size = 0;
  TensorUtils::GetSize(get_tensor_ptr->GetTensorDesc(), size);
  EXPECT_EQ(size, 120);
  auto dims = get_tensor_ptr->GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims.size(), 4);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[3], 5);
  EXPECT_EQ(get_tensor_ptr->GetData().GetSize(), 6);
  EXPECT_EQ(get_tensor_ptr->GetData().GetData()[0], 3);
  EXPECT_EQ(get_tensor_ptr->GetData().GetData()[5], 8);

  // modify format
  get_tensor_ptr->MutableTensorDesc().SetFormat(FORMAT_NC1C0HWPAD);
  EXPECT_EQ(get_tensor_ptr->GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);  // copy, not change

  // modify datatype
  tensor_desc.SetDataType(DT_INT32);
  tensor.SetTensorDesc(tensor_desc);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(get_tensor_ptr->GetTensorDesc().GetDataType(), DT_FLOAT16);  // copy, not change
}

TEST_F(UtestGeOutTensor, test_tensor_adapter_transfer2_tensor) {
  GeTensorDesc ge_tensor_desc(GeShape({2, 3, 4, 5}), FORMAT_NC1HWC0, DT_FLOAT16);
  TensorUtils::SetSize(ge_tensor_desc, 120);
  vector<uint8_t> data = {3, 4, 5, 6, 7, 8};
  GeTensor ge_tensor(ge_tensor_desc, data);

  Tensor tensor = TensorAdapter::GeTensor2Tensor(std::make_shared<GeTensor>(ge_tensor));
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_FLOAT16);
  EXPECT_EQ(tensor.GetTensorDesc().GetSize(), 120);

  auto dims = tensor.GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(dims.size(), 4);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[3], 5);
  EXPECT_EQ(tensor.GetSize(), 6);
  EXPECT_EQ(tensor.GetData()[0], 3);
  EXPECT_EQ(tensor.GetData()[5], 8);

  // modify format
  ge_tensor.MutableTensorDesc().SetFormat(FORMAT_NC1C0HWPAD);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(tensor.GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);  // copy, not change

  // modify datatype
  auto tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(ge_tensor_desc);
  tensor_desc.SetDataType(DT_INT32);
  tensor.SetTensorDesc(tensor_desc);
  EXPECT_EQ(tensor.GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetDataType(), DT_FLOAT16);  // copy, not change
}
