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

#define protected public
#define private public
#include "graph/op_desc.h"
#include "graph/op_desc_impl.h"
#include "graph/ge_tensor.h"
#include "graph/utils/ge_ir_utils.h"
#undef private
#undef protected
#include "graph/utils/transformer_utils.h"
#include "graph/common_error_codes.h"
#include "graph/operator_factory_impl.h"

namespace ge {
class UtestOpDesc : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpDesc, TestCommonVerifyOnDummyShape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({-3}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  EXPECT_EQ(GRAPH_SUCCESS, op_desc->CommonVerify());
}

TEST_F(UtestOpDesc, TestOpDescGetSetTensorDesc) {
  GeTensorDesc desc(GeShape(), FORMAT_NCHW, DT_INT32);
  OpDesc op_desc("foo", "Foo");
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddInputDesc("x", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("y", desc));

  EXPECT_EQ(op_desc.GetInputDesc("x"), desc);
  EXPECT_EQ(op_desc.GetOutputDesc("y"), desc);
}

TEST_F(UtestOpDesc, TestNodeShapeTransUtils) {

  NodeShapeTransUtils transformer1(nullptr);
  EXPECT_NE(transformer1.Init(), true);

  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1, 1, 16, 16}));
  tensor_desc->SetFormat(FORMAT_FRACTAL_NZ);
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetOriginFormat(FORMAT_ND);

  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());
  NodeShapeTransUtils transformer2(op_desc);
  EXPECT_EQ(transformer2.Init(), true);
  EXPECT_EQ(transformer2.CatchFormatAndShape(), true);
  EXPECT_EQ(transformer2.UpdateFormatAndShape(), true);


  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  NodeShapeTransUtils transformer3(op_desc);
  EXPECT_EQ(transformer3.Init(), true);
  EXPECT_EQ(transformer3.CatchFormatAndShape(), true);
  EXPECT_EQ(transformer3.UpdateFormatAndShape(), true);


  EXPECT_EQ(GRAPH_SUCCESS, op_desc->CommonVerify());
}

TEST_F(UtestOpDesc, IndexOutOfRange) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());

  EXPECT_NE(nullptr, op_desc->MutableInputDesc(0));
  EXPECT_EQ(nullptr, op_desc->MutableInputDesc(1));
  EXPECT_EQ(nullptr, op_desc->MutableInputDesc(999));
}

TEST_F(UtestOpDesc, SerializeMetadata) {
  OpDescImpl impl;
  impl.meta_data_.inputs_.emplace_back("input");
  impl.meta_data_.input_names_.emplace_back("names");
  impl.meta_data_.src_names_.push_back("src");
  impl.meta_data_.dst_names_.push_back("dst");
  impl.meta_data_.dst_indexes_.push_back(2);
  impl.meta_data_.src_indexes_.push_back(2);
  impl.meta_data_.input_offsets_.push_back(987654321);
  impl.meta_data_.output_offsets_.push_back(987654321);
  impl.meta_data_.workspaces.push_back(222);
  impl.meta_data_.workspace_bytes_list_.push_back(111);
  impl.meta_data_.is_input_consts_.push_back(false);

  proto::OpDef def;
  impl.SerializeMetaDataToOpDef(&def);
  EXPECT_EQ(def.input(0), "input");
  EXPECT_EQ(def.input_name(0), "names");
  EXPECT_EQ(def.src_name(0), "src");
  EXPECT_EQ(def.dst_name(0), "dst");
  EXPECT_EQ(def.dst_index(0), 2);
  EXPECT_EQ(def.src_index(0), 2);
  EXPECT_EQ(def.input_i(0), 987654321);
  EXPECT_EQ(def.output_i(0), 987654321);
  EXPECT_EQ(def.workspace(0), 222);
  EXPECT_EQ(def.workspace_bytes(0), 111);
  EXPECT_EQ(def.is_input_const(0), false);
}

TEST_F(UtestOpDesc, DeSerializeMetadata) {
  proto::OpDef def;
  def.add_input("input");
  def.add_input_name("names");
  def.add_src_name("src");
  def.add_dst_name("dst");
  def.add_dst_index(2);
  def.add_src_index(2);
  def.add_input_i(987654321);
  def.add_output_i(987654321);
  def.add_workspace(222);
  def.add_workspace_bytes(222);
  def.add_is_input_const(false);
  OpDescImpl impl;
  impl.DeSerializeOpDefToMetaData(def);
  EXPECT_EQ(impl.meta_data_.inputs_.size(), 1);
  EXPECT_EQ(impl.meta_data_.inputs_[0], "input");
  EXPECT_EQ(impl.meta_data_.input_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.input_names_[0], "names");
  EXPECT_EQ(impl.meta_data_.src_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.src_names_[0], "src");
  EXPECT_EQ(impl.meta_data_.dst_names_.size(), 1);
  EXPECT_EQ(impl.meta_data_.dst_names_[0], "dst");
  EXPECT_EQ(impl.meta_data_.dst_indexes_.size(), 1);
  EXPECT_EQ(impl.meta_data_.dst_indexes_[0], 2);
  EXPECT_EQ(impl.meta_data_.src_indexes_.size(), 1);
  EXPECT_EQ(impl.meta_data_.src_indexes_[0], 2);
  EXPECT_EQ(impl.meta_data_.input_offsets_.size(), 1);
  EXPECT_EQ(impl.meta_data_.input_offsets_[0], 987654321);
  EXPECT_EQ(impl.meta_data_.output_offsets_.size(), 1);
  EXPECT_EQ(impl.meta_data_.output_offsets_[0], 987654321);
  EXPECT_EQ(impl.meta_data_.workspaces.size(), 1);
  EXPECT_EQ(impl.meta_data_.workspaces[0], 222);
  EXPECT_EQ(impl.meta_data_.workspace_bytes_list_.size(), 1);
  EXPECT_EQ(impl.meta_data_.workspace_bytes_list_[0], 222);
  EXPECT_EQ(impl.meta_data_.is_input_consts_.size(), 1);
  EXPECT_EQ(impl.meta_data_.is_input_consts_[0], false);

  OpDescImpl impl1;
  impl1.DeSerializeOpDefToMetaData(def);
  EXPECT_TRUE(impl1.OpDescAttrsAreEqual(impl));
}

TEST_F(UtestOpDesc, AddDescForward) {
  GeTensorDesc desc(GeShape(), FORMAT_NCHW, DT_INT32);
  OpDesc op_desc("foo", "Foo");
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("x", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("y", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDesc("z", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddOutputDescForward("t", 2));

  EXPECT_EQ(5, op_desc.GetOutputsSize());

  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddInputDesc("x", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddInputDesc("y", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddInputDesc("z", desc));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc.AddInputDescForward("t", 2));

  EXPECT_EQ(5, op_desc.GetInputsSize());
}

TEST_F(UtestOpDesc, AddInputDesc1_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  EXPECT_EQ(op_desc->AddInputDesc(0, tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddInputDesc(0, tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, AddInputDesc2_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  EXPECT_EQ(op_desc->AddInputDesc("input_desc1", tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddInputDesc("input_desc1", tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, AddInputDescMiddle_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddInputDesc("input_desc1", tensor_desc->Clone());
  op_desc->AddInputDesc("input_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->AddInputDescMiddle("input_desc3", 1, 1), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddInputDescMiddle("input_desc4", 1, 4), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, AddOutputDescMiddle_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddOutputDesc("output_desc1", tensor_desc->Clone());
  op_desc->AddOutputDesc("output_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->AddOutputDescMiddle("output_desc3", 1, 1), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddOutputDescMiddle("output_desc4", 1, 4), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, UpdateInputDesc_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  op_desc->AddInputDesc("input_desc1", tensor_desc->Clone());
  op_desc->AddInputDesc("input_desc2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->UpdateInputDesc(1, tensor_desc->Clone()), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateInputDesc(4, tensor_desc->Clone()), GRAPH_FAILED);
}

TEST_F(UtestOpDesc, AddInputDescForward_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddInputDesc(tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddInputDescForward("test", 1), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, AddOutputDescForward_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOutputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddOutputDescForward("test", 1), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, AddOptionalInputDesc_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->AddOptionalInputDesc("test", tensor_desc->Clone()), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, OpDescMembersAreEqual_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc("input_desc", tensor_desc->Clone());
  op_desc1->AddOutputDesc("output_desc", tensor_desc->Clone());
  op_desc1->AddOptionalInputDesc("optional_input", tensor_desc->Clone());
  op_desc1->SetOpEngineName("DNN_VM_HOST_CPU");
  op_desc1->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  auto op_desc2 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc("input_desc_diff", tensor_desc->Clone());
  op_desc1->AddOutputDesc("output_desc", tensor_desc->Clone());
  op_desc1->AddOptionalInputDesc("optional_input", tensor_desc->Clone());
  op_desc1->SetOpEngineName("DNN_VM_HOST_CPU");
  op_desc1->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  auto op_desc3 = op_desc1;

  EXPECT_EQ(op_desc1->OpDescMembersAreEqual(*(op_desc3)), true);
  EXPECT_EQ(op_desc1->OpDescMembersAreEqual(*(op_desc2)), false);
}

TEST_F(UtestOpDesc, OpDescGenTensorDescsAreEqual_success) {
  auto tensor_desc1 = std::make_shared<GeTensorDesc>();
  tensor_desc1->SetShape(GeShape({1}));
  tensor_desc1->SetFormat(FORMAT_NCHW);
  tensor_desc1->SetDataType(DT_FLOAT);

  auto tensor_desc2 = std::make_shared<GeTensorDesc>();
  tensor_desc2->SetShape(GeShape({-1}));
  tensor_desc2->SetFormat(FORMAT_NHWC);
  tensor_desc2->SetDataType(DT_INT32);

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddInputDesc(tensor_desc1->Clone());
  auto op_desc2 = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc2->AddInputDesc(tensor_desc2->Clone());
  op_desc1->AddOutputDesc(tensor_desc1->Clone());
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc2->AddOutputDesc(tensor_desc2->Clone());
  auto op_desc3 = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc2)), false);
  op_desc3->AddInputDesc(tensor_desc1->Clone());
  op_desc3->AddOutputDesc(tensor_desc2->Clone());
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc3)), false);
  EXPECT_EQ(op_desc1->OpDescGenTensorDescsAreEqual(*(op_desc1)), true);
}

TEST_F(UtestOpDesc, InputIsSet_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->InputIsSet("input_test"), false);
  op_desc->AddInputDesc("input_test",tensor_desc->Clone());
  EXPECT_EQ(op_desc->InputIsSet("input_test"), true);
}

TEST_F(UtestOpDesc, MutableInputDesc_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("input_test1",tensor_desc->Clone());
  EXPECT_EQ(op_desc->MutableInputDesc("input_test"), nullptr);
  EXPECT_NE(op_desc->MutableInputDesc("input_test1"), nullptr);
}

TEST_F(UtestOpDesc, Get_SetOpKernelLibName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");
  EXPECT_EQ(op_desc->GetOpKernelLibName(), "DNN_VM_RTS_OP_STORE");
}

TEST_F(UtestOpDesc, Get_SetOpEngineName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetOpEngineName("DNN_VM_HOST_CPU");
  EXPECT_EQ(op_desc->GetOpEngineName(), "DNN_VM_HOST_CPU");
}

TEST_F(UtestOpDesc, GetAllOutputsDescSize_sucess) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOutputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());
  EXPECT_EQ(op_desc->GetAllOutputsDescSize(), 2);
}

TEST_F(UtestOpDesc, AddDynamicInputDescByIndex_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("input_test1",tensor_desc->Clone());
  op_desc->AddInputDesc("input_test2",tensor_desc->Clone());
  EXPECT_EQ(op_desc->AddDynamicInputDescByIndex("input_test2", 1, 1), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, IsOptionalInput_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddOptionalInputDesc("optional_test", tensor_desc->Clone());
  op_desc->AddInputDesc("input_test", tensor_desc->Clone());
  EXPECT_EQ(op_desc->IsOptionalInput("input_test"), false);
  EXPECT_EQ(op_desc->IsOptionalInput("optional_test"), true);
}

TEST_F(UtestOpDesc, GetAllOutputName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddOutputDesc("output1", tensor_desc->Clone());
  op_desc->AddOutputDesc("output2", tensor_desc->Clone());
  std::map<std::string, uint32_t> all_output;
  all_output = op_desc->GetAllOutputName();
  EXPECT_EQ(all_output.size(), 2);
  EXPECT_EQ(all_output["output1"], 0);
  EXPECT_EQ(all_output["output2"], 1);
}

TEST_F(UtestOpDesc, UpdateInputName_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT); 
  auto op_desc = std::make_shared<OpDesc>();

  op_desc->AddInputDesc("name1", tensor_desc->Clone());
  op_desc->AddInputDesc("name2", tensor_desc->Clone());

  std::map<std::string, uint32_t> input_name_idx;
  input_name_idx.insert(pair<std::string, uint32_t>("update_name1", 0));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), false);
  input_name_idx.insert(pair<std::string, uint32_t>("update_name2", 1));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), true);
  auto all_input_name = op_desc->GetAllInputName();
  EXPECT_EQ(input_name_idx, all_input_name);
  input_name_idx.insert(pair<std::string, uint32_t>("update_name3", 2));
  EXPECT_EQ(op_desc->UpdateInputName(input_name_idx), true);
}

TEST_F(UtestOpDesc, UpdateOutputName_success) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT); 
  auto op_desc = std::make_shared<OpDesc>();

  op_desc->AddOutputDesc("name1", tensor_desc->Clone());
  op_desc->AddOutputDesc("name2", tensor_desc->Clone());

  std::map<std::string, uint32_t> output_name_idx;
  output_name_idx.insert(pair<std::string, uint32_t>("update_name1", 0));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), false);
  output_name_idx.insert(pair<std::string, uint32_t>("update_name2", 1));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), true);
  auto all_output_name = op_desc->GetAllOutputName();
  EXPECT_EQ(output_name_idx, all_output_name);
  output_name_idx.insert(pair<std::string, uint32_t>("update_name3", 2));
  EXPECT_EQ(op_desc->UpdateOutputName(output_name_idx), true);
}

TEST_F(UtestOpDesc, GetInferFunc_success) {
  auto op_desc = std::make_shared<OpDesc>();
  const auto add_func = [](Operator &op) { 
    return GRAPH_SUCCESS; 
  };
  op_desc->AddInferFunc(add_func);

  Operator op;
  auto func = op_desc->GetInferFunc();
  EXPECT_EQ(func == nullptr, false);
  EXPECT_EQ(func(op), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, GetInferFormatFunc_success) {
  auto op_desc = std::make_shared<OpDesc>();
  const auto add_func = [](Operator &op) { 
    return GRAPH_SUCCESS; 
  };
  op_desc->AddInferFormatFunc(add_func);

  Operator op;
  auto func = op_desc->GetInferFormatFunc();
  EXPECT_EQ(func == nullptr, false);
  EXPECT_EQ(func(op), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, InferDataSlice_success) {
  auto op_desc = std::make_shared<OpDesc>();
  const auto func = [](Operator &op) { return GRAPH_SUCCESS; };
  EXPECT_EQ(op_desc->InferDataSlice(), NO_DEPENDENCE_FUNC);
  const auto infer_data_slice_func = [](Operator &op) { 
    return GRAPH_SUCCESS; 
  };
  auto op = std::make_shared<Operator>();
  op_desc->SetType("test");
  OperatorFactoryImpl::RegisterInferDataSliceFunc("test",infer_data_slice_func);
  EXPECT_EQ(op_desc->InferDataSlice(), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, InferShapeAndType_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->InferShapeAndType(), GRAPH_SUCCESS);
  const auto add_func = [](Operator &op) { 
    return GRAPH_SUCCESS;
  };
  op_desc->AddInferFunc(add_func);
  EXPECT_EQ(op_desc->InferShapeAndType(), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, OpVerify_success) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(op_desc->OpVerify(), GRAPH_SUCCESS);
  const auto verify_func = [](Operator &op) { 
    return GRAPH_SUCCESS;
  };
  op_desc->AddVerifierFunc(verify_func);
  EXPECT_EQ(op_desc->OpVerify(), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, GetValidInputIndexByName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddInputDesc("name1", tensor_desc->Clone());
  op_desc->AddInputDesc("name2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->GetValidInputIndexByName("name1"), 0);
  EXPECT_EQ(op_desc->GetValidInputIndexByName("name2"), 1);
}

TEST_F(UtestOpDesc, GetValidInputNameByIndex_success) {
  auto op_desc = std::make_shared<OpDesc>("verify", "Rule");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddInputDesc("name1", tensor_desc->Clone());
  op_desc->AddInputDesc("name2", tensor_desc->Clone());

  EXPECT_EQ(op_desc->GetValidInputNameByIndex(0), "name1");
  EXPECT_EQ(op_desc->GetValidInputNameByIndex(1), "name2");
}

TEST_F(UtestOpDesc, GetStreamId_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->SetStreamId(1);
  EXPECT_EQ(op_desc->GetStreamId(), 1);
}

TEST_F(UtestOpDesc, Set_GetInputName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> input_name {"name1", "name2"};
  op_desc->SetInputName(input_name);
  auto get_input_name = op_desc->GetInputName();
  EXPECT_EQ(get_input_name.size(), 2);
  EXPECT_EQ(get_input_name[0], "name1");
  EXPECT_EQ(get_input_name[1], "name2");
}

TEST_F(UtestOpDesc, GetSrcName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> src_name {"src"};
  op_desc->SetSrcName(src_name);
  auto get_src_name = op_desc->GetSrcName();
  EXPECT_EQ(get_src_name.size(), 1);
  EXPECT_EQ(get_src_name[0], "src");
}

TEST_F(UtestOpDesc, GetSrcIndex_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> src_index{2};
  op_desc->SetSrcIndex(src_index);
  auto get_src_index = op_desc->GetSrcIndex();
  EXPECT_EQ(get_src_index.size(), 1);
  EXPECT_EQ(get_src_index[0], 2);
}

TEST_F(UtestOpDesc, GetInputOffset_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> input_offset{987654321};
  op_desc->SetInputOffset(input_offset);
  auto get_input_offset = op_desc->GetInputOffset();
  EXPECT_EQ(get_input_offset.size(), 1);
  EXPECT_EQ(get_input_offset[0], 987654321);
}

TEST_F(UtestOpDesc, GetOutputOffset_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> output_offset{987654321};
  op_desc->SetOutputOffset(output_offset);
  auto get_output_offset = op_desc->GetOutputOffset();
  EXPECT_EQ(get_output_offset.size(), 1);
  EXPECT_EQ(get_output_offset[0], 987654321);
}

TEST_F(UtestOpDesc, GetDstName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<std::string> dst_name{"dst"};
  op_desc->SetDstName(dst_name);
  auto get_dst_name = op_desc->GetDstName();
  EXPECT_EQ(get_dst_name.size(), 1);
  EXPECT_EQ(get_dst_name[0], "dst");
}

TEST_F(UtestOpDesc, GetDstIndex_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> dst_index{2};
  op_desc->SetDstIndex(dst_index);
  auto get_dst_index = op_desc->GetDstIndex();
  EXPECT_EQ(get_dst_index.size(), 1);
  EXPECT_EQ(get_dst_index[0], 2);
}

TEST_F(UtestOpDesc, Set_GetOpInferDepends_success) {
  auto op_desc = std::make_shared<OpDesc>("verify", "Rule");
  std::vector<std::string> depend_names {"depend_name1", "depend_name2"};
  op_desc->SetOpInferDepends(depend_names);
  auto get_depend_names = op_desc->GetOpInferDepends();
  EXPECT_EQ(get_depend_names.size(), 2);
  EXPECT_EQ(get_depend_names[0], "depend_name1");
  EXPECT_EQ(get_depend_names[1], "depend_name2");
}

TEST_F(UtestOpDesc, GetWorkspace_success) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<int64_t> workspace{222};
  op_desc->SetWorkspace(workspace);
  auto get_workspace = op_desc->GetWorkspace();
  EXPECT_EQ(get_workspace.size(), 1);
  EXPECT_EQ(get_workspace[0], 222);
}

TEST_F(UtestOpDesc, RestoreInputNameIdx_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddInputDesc("name1", tensor_desc->Clone());
  EXPECT_EQ(op_desc->RestoreInputNameIdx("name2",1), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, RestoreOutputNameIdx_success) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);

  op_desc->AddOutputDesc("name1", tensor_desc->Clone());
  EXPECT_EQ(op_desc->RestoreOutputNameIdx("name2",1), GRAPH_SUCCESS);
}

TEST_F(UtestOpDesc, GetSubgraphNameByInstanceName_success) {
  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddSubgraphName("subgraph");
  op_desc->SetSubgraphInstanceName(0, "subgraph");
  std::string subname("");
  EXPECT_EQ(op_desc->GetSubgraphNameByInstanceName("subgraph", subname), GRAPH_SUCCESS);
  EXPECT_EQ(subname, "subgraph");

  auto op_desc1 = std::make_shared<OpDesc>();
  op_desc1->AddSubgraphName("subgraph1");
  op_desc1->SetSubgraphInstanceName(0, "sub");
  EXPECT_EQ(op_desc1->GetSubgraphNameByInstanceName("sub", subname), GRAPH_SUCCESS);
  EXPECT_EQ(subname, "subgraph1");
}
}
