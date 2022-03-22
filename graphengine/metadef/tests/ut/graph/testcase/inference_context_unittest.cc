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
#include "graph/ge_error_codes.h"
#include "graph/inference_context.h"
#include "graph/resource_context_mgr.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "graph/utils/transformer_utils.h"
#include "external/graph/types.h"

namespace ge {
namespace {
struct TestResourceContext : ResourceContext {
  std::vector<GeShape> shapes;
  std::string resource_type;
};
}
class TestInferenceConext : public testing::Test {
 protected:
 ComputeGraphPtr graph_;
  void SetUp() {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    builder.AddNode("TensorArrayWrite", "TensorArrayWrite", 1, 1);
    builder.AddNode("TensorArrayRead", "TensorArrayRead", 1, 1);
    graph_ = builder.GetGraph();
  }

  void TearDown() {}
};

TEST_F(TestInferenceConext, TestSetAndGetResourceContext) {
  ResourceContextMgr resource_context_mgr;
  InferenceContextPtr write_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create(&resource_context_mgr));
  InferenceContextPtr read_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create(&resource_context_mgr));

  // simulate write op
  const char* resource_key = "123";
  std::vector<GeShape> resource_shapes = {GeShape({1,1,2,3})};
  TestResourceContext *resource_context = new TestResourceContext();
  resource_context->shapes = resource_shapes;
  resource_context->resource_type = "normal";
  // test resource key empty, return fail
  auto ret = write_inference_context->SetResourceContext(AscendString(nullptr), resource_context);
  ASSERT_EQ(ret, GRAPH_PARAM_INVALID);

  write_inference_context->SetResourceContext(AscendString(resource_key), resource_context);

  // simulate read op
  TestResourceContext *test_reousce_context =
      dynamic_cast<TestResourceContext *>(read_inference_context->GetResourceContext(resource_key));

  // check result
  auto ret_shape = test_reousce_context->shapes.at(0);
  auto ret_type = test_reousce_context->resource_type;
  ASSERT_EQ(ret_shape.GetDims(), resource_context->shapes.at(0).GetDims());
  ASSERT_EQ(ret_type, resource_context->resource_type);
}

TEST_F(TestInferenceConext, TestRegisterAndGetReiledOnResource) {
  InferenceContextPtr read_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());

  // simulate read_op register relied resource
  const char* resource_key = "456";
  read_inference_context->RegisterReliedOnResourceKey(AscendString(resource_key));

  // simulate read_op register empty relied resource
  auto ret = read_inference_context->RegisterReliedOnResourceKey(AscendString(nullptr));
  ASSERT_EQ(ret, GRAPH_PARAM_INVALID);

  auto reiled_keys = read_inference_context->GetReliedOnResourceKeys();
  // check result
  ASSERT_EQ(reiled_keys.empty(), false);
  ASSERT_EQ(*reiled_keys.begin(), resource_key);
}

TEST_F(TestInferenceConext, TestAddChangeResourceAndGet) {
  InferenceContextPtr write_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());

  // simulate write node add changed resource
  const char* resource_key = "789";
  write_inference_context->AddChangedResourceKey(AscendString(resource_key));

  // simulate write node add empty changed resource
  auto ret = write_inference_context->AddChangedResourceKey(AscendString(nullptr));
  ASSERT_EQ(ret, GRAPH_PARAM_INVALID);

  auto changed_keys = write_inference_context->GetChangedResourceKeys();
  // check result
  ASSERT_EQ(changed_keys.empty(), false);
  ASSERT_EQ(*(changed_keys.begin()), resource_key);

  // clear changed_key
  write_inference_context->ClearChangedResourceKeys();
  changed_keys = write_inference_context->GetChangedResourceKeys();
  // check result
  ASSERT_EQ(changed_keys.empty(), true);
}

TEST_F(TestInferenceConext, transformer_util) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("tmp", "tmp");
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetShape(GeShape(std::vector<int64_t>{1, 1}));
  tensor_desc.SetOriginShape(GeShape(std::vector<int64_t>{1, 1, 1, 1}));
  tensor_desc.SetFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT16);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  std::unique_ptr<NodeShapeTransUtils> transformer(new (std::nothrow) NodeShapeTransUtils(op_desc));
  transformer->Init();
  ASSERT_EQ(transformer->CatchFormatAndShape(), true);
  ASSERT_EQ(transformer->UpdateFormatAndShape(), true);
}

TEST_F(TestInferenceConext, ShapeAndType) {
  ShapeAndType SAndT;

  Shape shape;
  DataType data_type;

  shape = SAndT.GetShape();
  //ASSERT_NE(shape, NULL);
  data_type = SAndT.GetDataType();
  //ASSERT_NE(data_type, NULL);

  ShapeAndType SAndT2(shape, data_type);

  SAndT2.SetShape(shape);
  SAndT2.SetType(data_type);
}

TEST_F(TestInferenceConext, SetGetInputHandleShapesAndTypes) {
  InferenceContextPtr write_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());

  std::vector<std::vector<ShapeAndType>> input_handle_shapes_and_types;
  std::vector<std::vector<ShapeAndType>> input_handle_shapes_and_types_2;

  write_inference_context->SetInputHandleShapesAndTypes(std::move(input_handle_shapes_and_types));
  input_handle_shapes_and_types_2 = write_inference_context->GetInputHandleShapesAndTypes();
  ASSERT_EQ(input_handle_shapes_and_types_2.empty(), true);
}

TEST_F(TestInferenceConext, SetGetOutputHandleShapesAndTypes) {
  InferenceContextPtr write_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());

  std::vector<std::vector<ShapeAndType>> output_handle_shapes_and_types;
  std::vector<std::vector<ShapeAndType>> output_handle_shapes_and_types_2;

  write_inference_context->SetOutputHandleShapesAndTypes(output_handle_shapes_and_types);
  write_inference_context->SetOutputHandleShapesAndTypes(std::move(output_handle_shapes_and_types));
  output_handle_shapes_and_types_2 = write_inference_context->GetOutputHandleShapesAndTypes();
  ASSERT_EQ(output_handle_shapes_and_types_2.empty(), true);
}

TEST_F(TestInferenceConext, SetGetMarks) {
  InferenceContextPtr write_inference_context = std::shared_ptr<InferenceContext>(InferenceContext::Create());

  const std::vector<AscendString> marks;
  std::vector<AscendString> marks_2;
  write_inference_context->SetMarks(marks);
  write_inference_context->GetMarks(marks_2);
  ASSERT_EQ(marks, marks_2);
}

} // namespace ge
