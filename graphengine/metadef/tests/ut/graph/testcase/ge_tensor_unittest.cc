/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include <string>

#define private public
#define protected public
#include "graph/ge_tensor.h"

#include "graph/ge_attr_value.h"
#include "graph/tensor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_tensor_impl.h"
#undef private
#undef protected

using namespace std;
using namespace ge;

class UtestGeTensor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeTensor, origin_shape_format) {
  GeTensorDesc a;
  GeShape shape({1, 2, 3, 4});
  a.SetOriginShape(shape);
  a.SetOriginFormat(FORMAT_NCHW);
  EXPECT_EQ(a.GetOriginShape().GetShapeSize(), 24);
  EXPECT_EQ(a.GetOriginFormat(), FORMAT_NCHW);
}

TEST_F(UtestGeTensor, get_shape_size) {
  vector<int64_t> vec2{-1, 1, 2, 4};
  Shape shape2(vec2);
  shape2.GetShapeSize();

  vector<int64_t> vec3{-1, 2, 4, INT64_MAX};
  Shape shape3(vec3);
  shape3.GetShapeSize();

  vector<int64_t> vec4{-1, 2, 4, INT64_MAX};
  Shape shape4(vec4);
  shape4.GetShapeSize();

  vector<int64_t> vec1{1, 2, 3, 4};
  Shape shape1(vec1);
  EXPECT_EQ(shape1.GetShapeSize(), 24);
}

TEST_F(UtestGeTensor, shape) {
  GeShape a;
  EXPECT_EQ(a.GetDim(0), 0);
  EXPECT_EQ(a.GetShapeSize(), 0);
  EXPECT_EQ(a.SetDim(0, 0), GRAPH_FAILED);

  vector<int64_t> vec({1, 2, 3, 4});
  GeShape b(vec);
  GeShape c({1, 2, 3, 4});
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

TEST_F(UtestGeTensor, ge_shape_to_string1) {
  GeShape shape1({1, 2, 3, 4});
  EXPECT_EQ(shape1.ToString(), "1,2,3,4");
  GeShape shape2;
  EXPECT_EQ(shape2.ToString(), "");
}

TEST_F(UtestGeTensor, tensor_desc) {
  GeTensorDesc a;
  GeShape s({1, 2, 3, 4});
  GeTensorDesc b(s, FORMAT_NCHW);
  GeShape s1 = b.GetShape();
  EXPECT_EQ(s1.GetDim(0), s.GetDim(0));
  b.MutableShape().SetDim(0, 2);
  EXPECT_EQ(b.GetShape().GetDim(0), 2);
  GeShape s2({3, 2, 3, 4});
  b.SetShape(s2);
  EXPECT_EQ(b.GetShape().GetDim(0), 3);

  EXPECT_EQ(b.GetFormat(), FORMAT_NCHW);
  b.SetFormat(FORMAT_RESERVED);
  EXPECT_EQ(b.GetFormat(), FORMAT_RESERVED);

  EXPECT_EQ(b.GetDataType(), DT_FLOAT);
  b.SetDataType(DT_INT8);
  EXPECT_EQ(b.GetDataType(), DT_INT8);

  GeTensorDesc c;
  c.Update(GeShape({1}), FORMAT_NCHW);
  c.Update(s, FORMAT_NCHW);
  uint32_t size1 = 1;
  TensorUtils::SetSize(c, size1);
  GeTensorDesc d;
  d = c.Clone();
  GeTensorDesc e = c;
  int64_t size2 = 0;
  EXPECT_EQ(TensorUtils::GetSize(e, size2), GRAPH_SUCCESS);
  EXPECT_EQ(size2, 1);

  GeTensorDesc f = c;
  size2 = 0;
  EXPECT_EQ(TensorUtils::GetSize(f, size2), GRAPH_SUCCESS);
  EXPECT_EQ(size2, 1);
  EXPECT_EQ(c.IsValid(), GRAPH_SUCCESS);
  c.Update(GeShape(), FORMAT_RESERVED, DT_UNDEFINED);
  EXPECT_EQ(c.IsValid(), GRAPH_PARAM_INVALID);
}

TEST_F(UtestGeTensor, tensor) {
  GeShape s({1, 2, 3, 4});
  GeTensorDesc tensor_desc(s);
  std::vector<uint8_t> data({1, 2, 3, 4});
  GeTensor a;
  GeTensor b(tensor_desc);
  GeTensor c(tensor_desc, data);
  GeTensor d(tensor_desc, data.data(), data.size());

  GeShape s1 = b.GetTensorDesc().GetShape();
  EXPECT_EQ(s1.GetDim(0), 1);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_FLOAT);
  b.MutableTensorDesc().SetDataType(DT_INT8);
  EXPECT_EQ(b.GetTensorDesc().GetDataType(), DT_INT8);
  b.SetTensorDesc(tensor_desc);

  auto data1 = c.GetData();
  c.SetData(data);
  c.SetData(data.data(), data.size());
  EXPECT_EQ(c.GetData()[0], uint8_t(1));
  EXPECT_EQ(c.GetData()[1], uint8_t(2));
  EXPECT_EQ(c.MutableData().GetData()[2], uint8_t(3));
  EXPECT_EQ(c.MutableData().GetData()[3], uint8_t(4));

  GeTensor e(std::move(tensor_desc), std::move(data));
  EXPECT_EQ(e.GetData().GetSize(), data.size());
  EXPECT_EQ(e.GetData()[2], uint8_t(3));

  GeTensor f = e.Clone();
  e.MutableData().data()[2] = 5;
  EXPECT_EQ(e.GetData().data()[2], uint8_t(5));
  EXPECT_EQ(f.GetData().GetSize(), data.size());
  EXPECT_EQ(f.GetData()[2], uint8_t(3));
}

TEST_F(UtestGeTensor, test_shape_copy_move) {
  GeShape shape(nullptr, nullptr);
  EXPECT_EQ(shape.GetDimNum(), 0);

  GeShape shape2 = shape;
  EXPECT_EQ(shape2.GetDimNum(), 0);

  GeShape shape3({1, 2, 3});
  shape2 = shape3;
  EXPECT_EQ(shape2.GetDimNum(), 3);
  EXPECT_EQ(shape3.GetDimNum(), 3);

  GeShape shape4 = std::move(shape3);
  EXPECT_EQ(shape4.GetDimNum(), 3);
  EXPECT_EQ(shape3.GetDimNum(), 3);

  GeShape shape5;
  EXPECT_EQ(shape5.GetDimNum(), 0);
  shape5 = std::move(shape4);
  EXPECT_EQ(shape5.GetDimNum(), 3);
  EXPECT_EQ(shape4.GetDimNum(), 3);
}

TEST_F(UtestGeTensor, test_tensor_null_proto) {
  ProtoMsgOwner msg_owner;
  GeTensor tensor(msg_owner, nullptr);
  EXPECT_EQ(tensor.GetData().size(), 0);
  EXPECT_EQ(tensor.MutableData().size(), 0);
  EXPECT_EQ(tensor.SetData(Buffer(100)), GRAPH_SUCCESS);

  TensorUtils::SetWeightSize(tensor.MutableTensorDesc(), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor), 100);

  auto tensor_ptr = std::make_shared<GeTensor>(msg_owner, nullptr);
  TensorUtils::SetWeightSize(tensor_ptr->MutableTensorDesc(), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor_ptr), 100);

  GeTensor tensor1 = tensor;
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor1), 100);
}

TEST_F(UtestGeTensor, test_tensor_utils_weight_size) {
  GeTensor tensor;
  EXPECT_EQ(tensor.GetData().size(), 0);
  EXPECT_EQ(tensor.MutableData().size(), 0);
  EXPECT_EQ(tensor.SetData(Buffer(100)), GRAPH_SUCCESS);

  TensorUtils::SetWeightSize(tensor.MutableTensorDesc(), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor), 100);

  uint8_t buffer[100];
  EXPECT_TRUE(TensorUtils::GetWeightAddr(tensor, buffer) != nullptr);

  auto tensor_ptr = std::make_shared<GeTensor>();
  TensorUtils::SetWeightSize(tensor_ptr->MutableTensorDesc(), 100);
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor_ptr), 100);
  EXPECT_TRUE(TensorUtils::GetWeightAddr(tensor_ptr, buffer) != nullptr);

  GeTensor tensor1 = tensor;
  EXPECT_EQ(TensorUtils::GetWeightSize(tensor1), 100);

  GeTensor tensor2(GeTensorDesc(), Buffer(100));
  EXPECT_EQ(tensor2.GetData().size(), 100);
  EXPECT_EQ(tensor2.MutableData().size(), 100);

  GeTensor tensor3;
  tensor3 = tensor2;
  EXPECT_EQ(tensor3.GetData().size(), 100);
  EXPECT_EQ(tensor3.MutableData().size(), 100);

  TensorUtils::SetDataOffset(tensor3.MutableTensorDesc(), 20);
  EXPECT_EQ(TensorUtils::GetWeightAddr(tensor3, buffer), buffer + 20);
}

TEST_F(UtestGeTensor, test_tensor_valid) {
  // Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data)
  Shape shape({1, 1, 1});
  TensorDesc tensor_desc(shape);
  std::vector<uint8_t> data({1, 2, 3, 4});
  Tensor tensor1(tensor_desc, data);
  EXPECT_EQ(tensor1.IsValid(), GRAPH_SUCCESS);

  // Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size)
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  uint32_t size2 = 3 * 3 * 3 * 4;
  uint8_t data2[3 * 3 * 3 * 4] = {0};
  Tensor tensor2(tensor_desc2, data2, size2);
  EXPECT_EQ(tensor2.IsValid(), GRAPH_SUCCESS);

  // Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
  Tensor tensor3(std::move(tensor_desc), std::move(data));
  EXPECT_EQ(tensor3.IsValid(), GRAPH_SUCCESS);

  // DT_UNDEFINED
  TensorDesc tensor_desc3(Shape({3, 3, 3}), FORMAT_NCHW, DT_UNDEFINED);
  Tensor tensor4(tensor_desc3, data2, size2);
  EXPECT_EQ(tensor4.IsValid(), GRAPH_SUCCESS);

  // Tensor()
  Tensor tensor5;
  EXPECT_EQ(tensor5.IsValid(), GRAPH_SUCCESS);
  tensor5.SetTensorDesc(tensor_desc);
  tensor5.SetData(data);
  EXPECT_EQ(tensor5.IsValid(), GRAPH_SUCCESS);

  // scalar 1
  uint8_t data6[4] = {1, 2, 3, 4};
  Tensor tensor6;
  tensor6.SetData(data6, 4);
  EXPECT_EQ(tensor6.IsValid(), GRAPH_SUCCESS);

  // scalar 2
  TensorDesc tensor_desc7(Shape(), FORMAT_NCHW, DT_FLOAT);
  float data7 = 2;
  Tensor tensor7(tensor_desc7, (uint8_t *)&data7, sizeof(float));
  EXPECT_EQ(tensor7.IsValid(), GRAPH_SUCCESS);

  // string scalar
  TensorDesc tensor_desc8(Shape(), FORMAT_NCHW, DT_STRING);
  Tensor tensor8;
  tensor8.SetTensorDesc(tensor_desc8);
  string data8 = "A handsome boy write this code";
  EXPECT_EQ(tensor8.SetData(data8), GRAPH_SUCCESS);
  EXPECT_EQ(tensor8.IsValid(), GRAPH_SUCCESS);

  // string vector
  TensorDesc tensor_desc9(Shape({2}), FORMAT_NCHW, DT_STRING);
  vector<string> data9 = {"A handsome boy write this code", "very handsome"};
  Tensor tensor9(tensor_desc9);
  EXPECT_EQ(tensor9.SetData(data9), GRAPH_SUCCESS);
  EXPECT_EQ(tensor9.IsValid(), GRAPH_SUCCESS);

  vector<string> empty_data9;
  EXPECT_EQ(tensor9.SetData(empty_data9), GRAPH_FAILED);
}

TEST_F(UtestGeTensor, test_tensor_invalid) {
  // Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data)
  Shape shape({1, 1, 1});
  TensorDesc tensor_desc(shape);
  std::vector<uint8_t> data({1, 2, 3, 4, 5});
  Tensor tensor1(tensor_desc, data);
  EXPECT_EQ(tensor1.IsValid(), GRAPH_FAILED);

  // Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size)
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  uint32_t size2 = 3 * 3 * 3;
  uint8_t data2[3 * 3 * 3] = {0};
  Tensor tensor2(tensor_desc2, data2, size2);
  EXPECT_EQ(tensor2.IsValid(), GRAPH_FAILED);

  // Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
  Tensor tensor3(std::move(tensor_desc), std::move(data));
  EXPECT_EQ(tensor3.IsValid(), GRAPH_FAILED);

  // Tensor()
  Tensor tensor4;
  tensor4.SetTensorDesc(tensor_desc);
  EXPECT_EQ(tensor4.IsValid(), GRAPH_FAILED);
  tensor4.SetData(data);
  EXPECT_EQ(tensor4.IsValid(), GRAPH_FAILED);

  Tensor tensor5;
  tensor5.SetData(data);
  EXPECT_EQ(tensor5.IsValid(), GRAPH_FAILED);
  tensor5.SetTensorDesc(tensor_desc);
  EXPECT_EQ(tensor5.IsValid(), GRAPH_FAILED);

  // scalar
  TensorDesc tensor_desc6(Shape(), FORMAT_NCHW, DT_FLOAT);
  uint8_t data6 = 2;
  Tensor tensor6(tensor_desc6, &data6, 1);
  EXPECT_EQ(tensor6.IsValid(), GRAPH_FAILED);
}
