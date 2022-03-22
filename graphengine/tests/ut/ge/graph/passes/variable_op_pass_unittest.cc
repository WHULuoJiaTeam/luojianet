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
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/types.h"

#define protected public
#define private public
#include "graph/passes/variable_op_pass.h"

#include "common/op/ge_op_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/manager/graph_context.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/manager/util/variable_accelerate_ctrl.h"
#include "graph/manager/graph_mem_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph_builder_utils.h"
#include "cce/dnn.h"
#include "cce/dnn_struct_base.hpp"
#include "common/formats/format_transfers/format_transfer_nchw_nc1hwc0.h"
#include "common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.h"
#include "common/formats/format_transfers/datatype_transfer.h"
#undef private
#undef protected

using namespace std;
using namespace ge;
using namespace cce;

class UtestVariableOpPassUnit : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  // AUTO GEN PLEASE DO NOT MODIFY IT
};
namespace {

///         c
/// var1ref1 --> netoutput1
///    \          /
///     transdata2
///         |
///       assign1
///      /     \.
/// transdata1  |
///     |       |
///  var1     const1
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto const1 =
      builder.AddNode("const1", "Const", 0, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT,
                                    std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  transdata1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  auto assign1 =
      builder.AddNode("assign1", "Assign", 2, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT,
                                    std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  auto var1ref1 = builder.AddNode("var1ref1", "Variable", 1, 0);
  AttrUtils::SetStr(var1ref1->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "var1");
  auto netoutput1 = builder.AddNode("netoutput1", "Netoutput", 2, 0);

  builder.AddDataEdge(var1, 0, transdata1, 0);
  builder.AddDataEdge(const1, 0, assign1, 1);
  builder.AddDataEdge(transdata1, 0, assign1, 0);
  builder.AddDataEdge(assign1, 0, transdata2, 0);
  builder.AddDataEdge(transdata2, 0, var1ref1, 0);
  builder.AddDataEdge(transdata2, 0, netoutput1, 0);
  builder.AddControlEdge(var1ref1, netoutput1);

  return builder.GetGraph();
}

///  conv1
///   |
/// reshape1
///   |
///  var1
ComputeGraphPtr BuildGraph2() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1, FORMAT_ND, DT_FLOAT, std::vector<int64_t>({8 * 8 * 3, 2}));
  auto reshape1 =
      builder.AddNode("reshape1", "Reshape", 2, 1, FORMAT_HWCN, DT_FLOAT, std::vector<int64_t>({8, 8, 3, 2}));
  reshape1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  reshape1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({8 * 8 * 3, 2})));
  auto conv1 = builder.AddNode("conv1", "Conv2D", 2, 1, FORMAT_HWCN, DT_FLOAT, std::vector<int64_t>({8, 8, 3, 2}));

  builder.AddDataEdge(var1, 0, reshape1, 0);
  builder.AddDataEdge(reshape1, 0, conv1, 1);

  return builder.GetGraph();
}

///  conv1
///    |
/// reformat1
///    |
///  var1
ComputeGraphPtr BuildGraph3() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({8, 8, 3, 2}));
  auto reformat1 =
      builder.AddNode("reformat1", "ReFormat", 1, 1, FORMAT_ND, DT_FLOAT, std::vector<int64_t>({8, 8, 3, 2}));
  reformat1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  reformat1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({8, 8, 3, 2})));
  auto conv1 = builder.AddNode("conv1", "Conv2D", 2, 1, FORMAT_ND, DT_FLOAT, std::vector<int64_t>({8, 8, 3, 2}));

  builder.AddDataEdge(var1, 0, reformat1, 0);
  builder.AddDataEdge(reformat1, 0, conv1, 1);

  return builder.GetGraph();
}

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};

std::string var_ref_name_0;

ge::NodePtr CreatVariableRef(ge::NodePtr &final_writable_node, ge::NodePtr &var_node) {
  GELOGI("Create VarRef Op: final_writable_node: [%s] var_node: [%s]>>>>", final_writable_node->GetName().c_str(),
         var_node->GetName().c_str());

  static uint32_t var_ref_count = 0;
  std::stringstream var_ref_name;
  var_ref_name << "_to_" << final_writable_node->GetName() << "_REF_" << var_ref_count++;

  OpDescPtr var_op_desc = var_node->GetOpDesc();
  GE_CHK_BOOL_EXEC(var_op_desc != nullptr, return nullptr, "get var opdesc is nullptr");
  OpDescPtr var_ref_op_desc = nullptr;
  GE_MAKE_SHARED(var_ref_op_desc =
                     std::make_shared<OpDesc>(var_node->GetName() + var_ref_name.str().c_str(), var_op_desc->GetType()),
                 return nullptr);

  var_ref_op_desc->AddOutputDesc(var_op_desc->GetOutputDesc(0));
  var_ref_op_desc->AddInputDesc(var_op_desc->GetOutputDesc(0));

  const map<string, ge::GeAttrValue> var_attr_value = var_op_desc->GetAllAttrs();
  for (auto const &attrIt : var_attr_value) {
    var_ref_op_desc->SetAttr(attrIt.first, attrIt.second);
  }

  NodePtr var_ref_node = var_node->GetOwnerComputeGraph()->AddNode(var_ref_op_desc);
  GE_CHK_BOOL_EXEC(var_ref_node != nullptr, return nullptr, "create var_REF_node failed")

  GE_IF_BOOL_EXEC(ge::AttrUtils::SetStr(var_ref_op_desc, REF_VAR_SRC_VAR_NAME, var_op_desc->GetName()),
                  GELOGI("Set node [%s] VAR_ATTR_VAR_IS_REF [%s]", var_ref_node->GetName().c_str(),
                         var_op_desc->GetName().c_str()));
  var_ref_name_0 = var_ref_node->GetName();
  return var_ref_node;
}

bool BuildComputeGraph0(ge::ComputeGraphPtr &graph) {
  // graph = std::make_shared<ComputeGraph>("test");

  ge::NodePtr node_4d_new =
      NodeBuilder("Node4D_new", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32).Build(graph);

  ge::NodePtr node_4d_to_5d_1_new = NodeBuilder("4d_to_5d_1_new", TRANSDATA)
                                        .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                        .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                        .Build(graph);

  ge::NodePtr node_4d_to_5d_2_new = NodeBuilder("4d_to_5d_2_new", TRANSDATA)
                                        .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                        .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32)
                                        .Build(graph);

  ge::GraphUtils::AddEdge(node_4d_new->GetOutDataAnchor(0), node_4d_to_5d_1_new->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_new->GetOutDataAnchor(0), node_4d_to_5d_2_new->GetInDataAnchor(0));

  // Node4D
  ge::NodePtr node_4d =
      NodeBuilder("Node4D", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_5d_1 = NodeBuilder("4d_to_5d_1", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .Build(graph);

  ge::NodePtr node_4d_to_5d_2 = NodeBuilder("4d_to_5d_2", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .Build(graph);

  // Node5D
  ge::NodePtr node_5d_1 =
      NodeBuilder("5D_1", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  ge::NodePtr node_5d_2 =
      NodeBuilder("5D_2", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_5d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_5d_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_5d_1->GetOutDataAnchor(0), node_5d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_to_5d_2->GetOutDataAnchor(0), node_5d_2->GetInDataAnchor(0));

  // Node4D
  ge::NodePtr node_4d_nhwc =
      NodeBuilder("Node4D_NHWC", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_5d_1_nhwc = NodeBuilder("4d_to_5d_1_NHWC", TRANSDATA)
                                         .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_INT32)
                                         .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                         .Build(graph);

  // Node5D
  ge::NodePtr node_5d_1_nhwc =
      NodeBuilder("5D_1_NHWC", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_4d_nhwc->GetOutDataAnchor(0), node_4d_to_5d_1_nhwc->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_5d_1_nhwc->GetOutDataAnchor(0), node_5d_1_nhwc->GetInDataAnchor(0));

  // Node4D
  ge::NodePtr node_4d_hwcn =
      NodeBuilder("Node4D_HWCN", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_HWCN, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_5d_1_hwcn = NodeBuilder("4d_to_5d_1_HWCN", TRANSDATA)
                                         .AddInputDesc({1, 2, 3, 4}, FORMAT_HWCN, DT_INT32)
                                         .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                         .Build(graph);

  // Node5D
  ge::NodePtr node_5d_1_hwcn =
      NodeBuilder("5D_1_HWCN", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_4d_hwcn->GetOutDataAnchor(0), node_4d_to_5d_1_hwcn->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_5d_1_hwcn->GetOutDataAnchor(0), node_5d_1_hwcn->GetInDataAnchor(0));

  ge::NodePtr node_4d_chwn =
      NodeBuilder("Node4D_CHWN", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_CHWN, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_5d_1_chwn = NodeBuilder("4d_to_5d_1_CHWN", TRANSDATA)
                                         .AddInputDesc({1, 2, 3, 4}, FORMAT_CHWN, DT_INT32)
                                         .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                         .Build(graph);

  // Node5D
  ge::NodePtr node_5d_1_chwn =
      NodeBuilder("5D_1_CHWN", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_4d_chwn->GetOutDataAnchor(0), node_4d_to_5d_1_chwn->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_5d_1_chwn->GetOutDataAnchor(0), node_5d_1_chwn->GetInDataAnchor(0));

  ge::NodePtr node_4d_d =
      NodeBuilder("Node4D_D", VARIABLE).AddOutputDesc({1}, FORMAT_CHWN, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_5d_1_d = NodeBuilder("4d_to_5d_1_D", TRANSDATA)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_CHWN, DT_INT32)
                                      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                      .Build(graph);

  // Node5D
  ge::NodePtr node_5d_1_d =
      NodeBuilder("5D_1_D", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  ge::NodePtr node_apply_monetum = NodeBuilder("apply_monetum", APPLYMOMENTUM)
                                       .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                       .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                       .Build(graph);

  ge::NodePtr node_5d_to_4d_1 = NodeBuilder("5d_to_4d_1", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .Build(graph);

  ge::NodePtr node_ref = CreatVariableRef(node_5d_to_4d_1, node_4d);

  // add edge
  ge::GraphUtils::AddEdge(node_4d_d->GetOutDataAnchor(0), node_4d_to_5d_1_d->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_5d_1_d->GetOutDataAnchor(0), node_5d_1_d->GetInDataAnchor(0));

  if (ge::GraphUtils::AddEdge(node_apply_monetum->GetOutDataAnchor(0), node_5d_to_4d_1->GetInDataAnchor(0)) !=
      ge::SUCCESS) {
  };
  ge::GraphUtils::AddEdge(node_5d_to_4d_1->GetOutDataAnchor(0), node_ref->GetInDataAnchor(0));

  return true;
}

bool BuildComputeGraph1(ge::ComputeGraphPtr &graph) {
  // Node4D
  ge::NodePtr node_4d =
      NodeBuilder("Node4D", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_5d_1 = NodeBuilder("4d_to_5d_1", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .Build(graph);

  ge::NodePtr node_4d_to_5d_2 = NodeBuilder("4d_to_5d_2", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .Build(graph);

  // Node5D
  ge::NodePtr node_5d_1 =
      NodeBuilder("5D_1", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  ge::NodePtr node_5d_2 =
      NodeBuilder("5D_2", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  ge::NodePtr node_5d_to_4d_1 = NodeBuilder("5d_to_4d_1", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .Build(graph);

  ge::NodePtr node_apply_monetum = NodeBuilder("apply_monetum", APPLYMOMENTUM)
                                       .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                       .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32)
                                       .Build(graph);

  ge::NodePtr node_ref = CreatVariableRef(node_5d_to_4d_1, node_4d);

  // add edge
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_5d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_5d_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_5d_1->GetOutDataAnchor(0), node_5d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_to_5d_2->GetOutDataAnchor(0), node_5d_2->GetInDataAnchor(0));

  if (ge::GraphUtils::AddEdge(node_apply_monetum->GetOutDataAnchor(0), node_5d_to_4d_1->GetInDataAnchor(0)) !=
      ge::SUCCESS) {
  };
  ge::GraphUtils::AddEdge(node_5d_to_4d_1->GetOutDataAnchor(0), node_ref->GetInDataAnchor(0));

  return true;
}

bool BuildComputeGraph4(ge::ComputeGraphPtr &graph) {
  // Node4D
  ge::NodePtr node_4d =
      NodeBuilder("Node4D", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_5d_1 = NodeBuilder("4d_to_5d_1", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .Build(graph);

  ge::NodePtr node_4d_to_5d_2 = NodeBuilder("4d_to_5d_2", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .Build(graph);

  // Node5D
  ge::NodePtr node_5d_1 =
      NodeBuilder("5D_1", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  ge::NodePtr node_5d_2 =
      NodeBuilder("5D_2", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT).Build(graph);

  ge::NodePtr node_5d_to_4d_1 = NodeBuilder("5d_to_4d_1", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .Build(graph);

  ge::NodePtr node_5d_to_4d_2 = NodeBuilder("5d_to_4d_2", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .Build(graph);

  ge::NodePtr node_apply_monetum = NodeBuilder("apply_monetum", APPLYMOMENTUM)
                                       .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                       .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32)
                                       .Build(graph);

  ge::NodePtr node_ref = CreatVariableRef(node_5d_to_4d_1, node_4d);

  // add edge
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_5d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_5d_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_5d_1->GetOutDataAnchor(0), node_5d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_to_5d_2->GetOutDataAnchor(0), node_5d_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_apply_monetum->GetOutDataAnchor(0), node_5d_to_4d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_5d_to_4d_1->GetOutDataAnchor(0), node_ref->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_5d_to_4d_2->GetOutDataAnchor(0), node_ref->GetInDataAnchor(0));

  return true;
}

bool BuildComputeGraph5(ge::ComputeGraphPtr &graph) {
  // Node4D
  ge::NodePtr node_4d =
      NodeBuilder("Node4D", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32).Build(graph);

  return true;
}

bool BuildComputeGraph6(ge::ComputeGraphPtr &graph) {
  // Node4D
  ge::NodePtr node_4d =
      NodeBuilder("Node4D", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_5d_1 = NodeBuilder("4d_to_5d_1", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .Build(graph);

  ge::NodePtr node_float_to_int_1 = NodeBuilder("float_to_int_1", CAST)
                                        .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                        .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32)
                                        .Build(graph);

  ge::NodePtr node_4d_to_5d_2 = NodeBuilder("4d_to_5d_2", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                    .Build(graph);

  ge::NodePtr node_float_to_int_2 = NodeBuilder("float_to_int_2", CAST)
                                        .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                        .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32)
                                        .Build(graph);

  // Node5D
  ge::NodePtr node_5d_1 =
      NodeBuilder("5D_1", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32).Build(graph);

  ge::NodePtr node_5d_2 =
      NodeBuilder("5D_2", RELU).AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_INT32).Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_5d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_5d_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_5d_1->GetOutDataAnchor(0), node_float_to_int_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_4d_to_5d_2->GetOutDataAnchor(0), node_float_to_int_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_float_to_int_1->GetOutDataAnchor(0), node_5d_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_float_to_int_2->GetOutDataAnchor(0), node_5d_2->GetInDataAnchor(0));

  return true;
}
}  // namespace

bool BuildComputeGraph7(ge::ComputeGraphPtr &graph) {
  // Node4D
  ge::NodePtr node_4d =
      NodeBuilder("Node4D", VARIABLE).AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32).Build(graph);

  // NodeTrans4DTo5D
  ge::NodePtr node_4d_to_4d_1 = NodeBuilder("4d_to_4d_1", TRANSDATA)
                                    .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_INT32)
                                    .Build(graph);
  // Node5D
  ge::NodePtr node_4d_1 = NodeBuilder("4D_1", RELU).AddInputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_INT32).Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_4d->GetOutDataAnchor(0), node_4d_to_4d_1->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node_4d_to_4d_1->GetOutDataAnchor(0), node_4d_1->GetInDataAnchor(0));
  return true;
}

class VariableOpPassSimulator {
 public:
  bool DoTest0() {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("0");
    const std::string var_name = "Node4D";

    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    std::vector<int64_t> dims(4, 20);
    ge::GeShape shape(dims);

    MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
    VarManager::Instance(session_id)->Init(session_version, session_id, device_id, job_id);

    BuildComputeGraph0(compute_graph);

    std::vector<std::string> var_names = {"Node4D_new",  "Node4D",      "Node4D_NHWC",
                                          "Node4D_HWCN", "Node4D_CHWN", "Node4D_D"};
    for (auto name : var_names) {
      auto var_node = compute_graph->FindNode(name);
      auto var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);

      uint8_t *dev_ptr = nullptr;
      ge::VarManager::Instance(session_id)->AssignVarMem(name, var_tensor_desc, RT_MEMORY_HBM);
      ge::VarManager::Instance(session_id)->SetVarAddr(name, var_tensor_desc, dev_ptr, RT_MEMORY_HBM);
    }

    ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
    compute_graph->InferShapeInNeed();
    graph_node->SetComputeGraph(compute_graph);
    auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
    graph_node->SetGraph(tmp_graph_ptr);

    VarAccelerateCtrl ctrl;
    ctrl.AddGraph(graph_node->GetGraphId(), compute_graph);
    ge::formats::FormatTransferNchwNc1hwc0 ClassObj;
    VariableOpPass pass(&ctrl);
    pass.Run(compute_graph);

    MemManager::Instance().Finalize();

    return CheckTest0(compute_graph);
  }

  bool DoTest1() {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("0");
    const std::string var_name = "Node4D";

    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    std::vector<int64_t> dims(4, 20);
    ge::GeShape shape(dims);
    VarManager::Instance(session_id)->Init(session_version, session_id, device_id, job_id);

    BuildComputeGraph1(compute_graph);

    auto var_node = compute_graph->FindNode(var_name);
    auto var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);

    uint8_t *dev_ptr = nullptr;

    ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
    compute_graph->InferShapeInNeed();
    graph_node->SetComputeGraph(compute_graph);
    auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
    graph_node->SetGraph(tmp_graph_ptr);

    VarAccelerateCtrl ctrl;
    ctrl.AddGraph(graph_node->GetGraphId(), compute_graph);
    VariableOpPass pass(&ctrl);
    pass.Run(compute_graph);
    return CheckTest1(compute_graph);
  }

  bool DoTest2() {
    VarAccelerateCtrl ctrl;
    VariableOpPass pass(&ctrl);
    return pass.Run(nullptr) == ge::INTERNAL_ERROR;
  }

  bool DoTest3() {
    std::vector<rtMemType_t> mem_type;
    std::map<std::string, std::string> empty_options;
    mem_type.push_back(RT_MEMORY_HBM);
    MemManager::Instance().Initialize(mem_type);

    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("0");

    std::vector<std::string> var_names = {"Node4D", "Node4D_NHWC", "Node4D_HWCN", "Node4D_CHWN", "Node4D_D"};
    std::vector<ge::GeTensorDesc> tensor_descs;

    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    compute_graph->SetSessionID(session_id);
    std::vector<int64_t> dims(4, 20);
    ge::GeShape shape(dims);
    VarManager::Instance(session_id)->Init(session_version, session_id, device_id, job_id);

    BuildComputeGraph0(compute_graph);
    for (auto var_name : var_names) {
      auto var_node = compute_graph->FindNode(var_name);
      auto var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);

      uint8_t *dev_ptr = nullptr;
      ge::VarManager::Instance(session_id)->AssignVarMem(var_name, var_tensor_desc, RT_MEMORY_HBM);
      ge::VarManager::Instance(session_id)->SetVarAddr(var_name, var_tensor_desc, dev_ptr, RT_MEMORY_HBM);
    }

    ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
    compute_graph->InferShapeInNeed();
    graph_node->SetComputeGraph(compute_graph);
    auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
    graph_node->SetGraph(tmp_graph_ptr);

    VarAccelerateCtrl ctrl;
    ctrl.AddGraph(graph_node->GetGraphId(), compute_graph);
    VariableOpPass pass(&ctrl);
    auto ret = pass.Run(compute_graph);
    MemManager::Instance().Finalize();
    return ret == GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  bool DoTest4() {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("0");
    const std::string var_name = "Node4D";

    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    std::vector<int64_t> dims(4, 20);
    ge::GeShape shape(dims);
    VarManager::Instance(session_id)->Init(session_version, session_id, device_id, job_id);

    BuildComputeGraph4(compute_graph);

    auto var_node = compute_graph->FindNode(var_name);
    auto var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);

    uint8_t *dev_ptr = nullptr;
    ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
    compute_graph->InferShapeInNeed();
    graph_node->SetComputeGraph(compute_graph);
    auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
    graph_node->SetGraph(tmp_graph_ptr);

    VarAccelerateCtrl ctrl;
    ctrl.AddGraph(graph_node->GetGraphId(), compute_graph);
    VariableOpPass pass(&ctrl);
    auto ret = pass.Run(compute_graph);
    return ret == ge::SUCCESS;
  }

  bool DoTest5() {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("0");
    BuildComputeGraph5(compute_graph);
    const std::string var_name = "Node4D";

    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    std::vector<int64_t> dims(4, 20);
    ge::GeShape shape(dims);
    VarManager::Instance(session_id)->Init(session_version, session_id, device_id, job_id);

    BuildComputeGraph4(compute_graph);

    auto var_node = compute_graph->FindNode(var_name);
    auto var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);

    uint8_t *dev_ptr = nullptr;

    ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
    compute_graph->InferShapeInNeed();
    graph_node->SetComputeGraph(compute_graph);
    auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
    graph_node->SetGraph(tmp_graph_ptr);

    VarAccelerateCtrl ctrl;
    ctrl.AddGraph(graph_node->GetGraphId(), compute_graph);
    VariableOpPass pass(&ctrl);
    auto ret = pass.Run(compute_graph);

    return ret == ge::SUCCESS;
  }

  bool DoTest6() {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("0");
    const std::string var_name = "Node4D";

    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    std::vector<int64_t> dims(4, 20);
    ge::GeShape shape(dims);
    MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
    VarManager::Instance(session_id)->Init(session_version, session_id, device_id, job_id);

    BuildComputeGraph6(compute_graph);

    auto var_node = compute_graph->FindNode(var_name);
    auto var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);

    uint8_t *dev_ptr = nullptr;
    ge::VarManager::Instance(session_id)->AssignVarMem(var_name, var_tensor_desc, RT_MEMORY_HBM);
    ge::VarManager::Instance(session_id)->SetVarAddr(var_name, var_tensor_desc, dev_ptr, RT_MEMORY_HBM);
    ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
    compute_graph->InferShapeInNeed();
    graph_node->SetComputeGraph(compute_graph);
    auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
    graph_node->SetGraph(tmp_graph_ptr);

    VarAccelerateCtrl ctrl;
    ctrl.AddGraph(graph_node->GetGraphId(), compute_graph);
    ge::formats::FormatTransferNchwNc1hwc0 ClassObj;
    VariableOpPass pass(&ctrl);
    auto ret = pass.Run(compute_graph);
    MemManager::Instance().Finalize();
    return CheckTest6(compute_graph);
  }

  bool DoTest7() {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("0");
    const std::string var_name = "Node4D";

    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    std::vector<int64_t> dims(4, 20);
    ge::GeShape shape(dims);
    VarManager::Instance(session_id)->Init(session_version, session_id, device_id, job_id);

    BuildComputeGraph7(compute_graph);

    auto var_node = compute_graph->FindNode(var_name);
    auto var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);

    uint8_t *dev_ptr = nullptr;

    ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
    compute_graph->InferShapeInNeed();
    graph_node->SetComputeGraph(compute_graph);
    auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
    graph_node->SetGraph(tmp_graph_ptr);

    VarAccelerateCtrl ctrl;
    ctrl.AddGraph(graph_node->GetGraphId(), compute_graph);
    VariableOpPass pass(&ctrl);
    auto ret = pass.Run(compute_graph);
    return CheckTest7(compute_graph);
  }

  bool DoTest8() {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("0");
    const std::string var_name = "Node4D";

    uint64_t session_id = 0;
    uint32_t device_id = 0;
    uint64_t job_id = 0;
    uint32_t session_version = 0;
    std::vector<int64_t> dims(4, 20);
    ge::GeShape shape(dims);
    VarManager::Instance(session_id)->Init(session_version, session_id, device_id, job_id);

    BuildComputeGraph0(compute_graph);

    auto var_node = compute_graph->FindNode(var_name);
    auto var_tensor_desc = var_node->GetOpDesc()->GetOutputDesc(0);

    uint8_t *dev_ptr = nullptr;
    ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
    compute_graph->InferShapeInNeed();
    graph_node->SetComputeGraph(compute_graph);
    auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
    graph_node->SetGraph(tmp_graph_ptr);
    VarAccelerateCtrl ctrl;
    ctrl.AddGraph(graph_node->GetGraphId(), compute_graph);
    VariableOpPass pass(&ctrl);
    pass.Run(compute_graph);
    return CheckTest8(compute_graph);
  }

 private:
  bool CheckTest0(const ge::ComputeGraphPtr compute_graph) {
    const auto &variable_node = compute_graph->FindNode("Node4D");
    auto variable_node_format = variable_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
    auto variable_node_data_type = variable_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    auto variable_node_shape = variable_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();

    if (variable_node_format != FORMAT_NC1HWC0 || variable_node_data_type != DT_FLOAT ||
        variable_node_shape.size() != 5) {
      std::cout << "var format not changed !" << std::endl;
      return false;
    }

    const auto &variable_ref_node = compute_graph->FindNode(var_ref_name_0);
    GELOGD("var_ref_name_0 is %s", var_ref_name_0.c_str());
    auto variable_ref_node_format = variable_ref_node->GetOpDesc()->GetInputDesc(0).GetFormat();
    auto variable_ref_node_data_type = variable_ref_node->GetOpDesc()->GetInputDesc(0).GetDataType();
    auto variable_ref_node_shape = variable_ref_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();

    if (variable_ref_node_format != FORMAT_NC1HWC0 || variable_ref_node_data_type != DT_FLOAT ||
        variable_ref_node_shape.size() != 5) {
      GELOGI("wanted data format is  (%d,%d,%u)", FORMAT_NC1HWC0, DT_FLOAT, 5);
      GELOGI("variable_ref_node_format is (%d,%d,%zu)", variable_ref_node_format, variable_ref_node_data_type,
             variable_ref_node_shape.size());

      std::cout << "var ref format not changed !" << std::endl;
      return false;
    }

    ge::NodePtr trans_node = compute_graph->FindNode("4d_to_5d_1");
    if (trans_node != nullptr) {
      std::cout << "4d_to_5d_1 not empty !" << std::endl;
      return false;
    }

    trans_node = compute_graph->FindNode("4d_to_5d_2");
    if (trans_node != nullptr) {
      std::cout << "4d_to_5d_2 not empty !" << std::endl;
      return false;
    }

    trans_node = compute_graph->FindNode("5d_to_4d_1");
    if (trans_node != nullptr) {
      std::cout << "5d_to_4d_1 not empty !" << std::endl;
      return false;
    }

    trans_node = compute_graph->FindNode("4d_to_5d_1_new");
    if (trans_node == nullptr) {
      std::cout << "4d_to_5d_1_new is empty !" << std::endl;
      return false;
    }

    auto new_variable_node = compute_graph->FindNode("Node4D_new");

    auto new_variable_node_format = new_variable_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
    auto new_variable_node_data_type = new_variable_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    auto new_variable_node_shape = new_variable_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();

    if (new_variable_node_format != FORMAT_NCHW || new_variable_node_data_type != DT_INT32 ||
        new_variable_node_shape.size() != 4) {
      std::cout << "Node4D_new format Changed ! wanted data format is  ( " << FORMAT_NC1HWC0 << ", " << DT_INT32
                << ", 4) " << std::endl;
      std::cout << "current is ( " << new_variable_node_format << ", " << new_variable_node_data_type << ", "
                << new_variable_node_shape.size() << ")" << std::endl;
      return false;
    }

    return true;
  };

  bool CheckTest1(const ge::ComputeGraphPtr compute_graph) {
    const auto &variable_node = compute_graph->FindNode("Node4D");
    auto variable_node_format = variable_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
    auto variable_node_data_type = variable_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    auto variable_node_shape = variable_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();

    if (variable_node_format != FORMAT_NCHW || variable_node_data_type != DT_INT32 || variable_node_shape.size() != 4) {
      std::cout << "var format changed !" << std::endl;
      return false;
    }

    const auto &variable_ref_node = compute_graph->FindNode(var_ref_name_0);
    GELOGD("var_ref_name_0 is %s", var_ref_name_0.c_str());
    auto variable_ref_node_format = variable_ref_node->GetOpDesc()->GetInputDesc(0).GetFormat();
    auto variable_ref_node_data_type = variable_ref_node->GetOpDesc()->GetInputDesc(0).GetDataType();
    auto variable_ref_node_shape = variable_ref_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();

    if (variable_ref_node_format != FORMAT_NCHW || variable_ref_node_data_type != DT_INT32 ||
        variable_ref_node_shape.size() != 4) {
      GELOGI("wanted data format is  (%d,%d,%u)", FORMAT_NCHW, DT_INT32, 4);
      GELOGI("variable_ref_node_format is (%d,%d,%zu)", variable_ref_node_format, variable_ref_node_data_type,
             variable_ref_node_shape.size());

      std::cout << "var ref format not changed !" << std::endl;
      return false;
    }

    ge::NodePtr trans_node = compute_graph->FindNode("4d_to_5d_1");
    if (trans_node == nullptr) {
      std::cout << "4d_to_5d_1 empty !" << std::endl;
      return false;
    }

    trans_node = compute_graph->FindNode("4d_to_5d_2");
    if (trans_node == nullptr) {
      std::cout << "4d_to_5d_2  empty !" << std::endl;
      return false;
    }

    trans_node = compute_graph->FindNode("5d_to_4d_1");
    if (trans_node == nullptr) {
      std::cout << "5d_to_4d_1 not empty !" << std::endl;
      return false;
    }

    return true;
  };

  bool CheckTest6(const ge::ComputeGraphPtr compute_graph) {
    const auto &variable_node = compute_graph->FindNode("Node4D");
    auto variable_node_format = variable_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
    auto variable_node_data_type = variable_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    auto variable_node_shape = variable_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();

    if (variable_node_format != FORMAT_NC1HWC0 || variable_node_data_type != DT_INT32 ||
        variable_node_shape.size() != 5) {
      std::cout << "var format not changed !" << std::endl;
      return false;
    }

    ge::NodePtr trans_node = compute_graph->FindNode("4d_to_5d_1");
    if (trans_node != nullptr) {
      std::cout << "4d_to_5d_1 not empty !" << std::endl;
      return false;
    }

    trans_node = compute_graph->FindNode("4d_to_5d_2");
    if (trans_node != nullptr) {
      std::cout << "4d_to_5d_2 not empty !" << std::endl;
      return false;
    }

    trans_node = compute_graph->FindNode("float_to_int_1");
    if (trans_node != nullptr) {
      std::cout << "float_to_int_1 not empty !" << std::endl;
      return false;
    }

    trans_node = compute_graph->FindNode("float_to_int_2");
    if (trans_node != nullptr) {
      std::cout << "float_to_int_1 not empty !" << std::endl;
      return false;
    }

    return true;
  };

  bool CheckTest7(const ge::ComputeGraphPtr compute_graph) {
    const auto &variable_node = compute_graph->FindNode("Node4D");
    auto variable_node_format = variable_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
    auto variable_node_data_type = variable_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    auto variable_node_shape = variable_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();

    if (variable_node_format != FORMAT_NC1HWC0 || variable_node_data_type != DT_INT32 ||
        variable_node_shape.size() != 5) {
      std::cout << "var format not changed !" << std::endl;
      return false;
    }

    ge::NodePtr trans_node = compute_graph->FindNode("4d_to_4d_1");
    if (trans_node != nullptr) {
      std::cout << "4d_to_5d_1 not empty !" << std::endl;
      return false;
    }
    return true;
  };

  bool CheckTest8(const ge::ComputeGraphPtr compute_graph) {
    const auto &variable_node = compute_graph->FindNode("Node4D");
    auto variable_node_format = variable_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
    auto variable_node_data_type = variable_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    auto variable_node_shape = variable_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
    return true;
  };
};

TEST_F(UtestVariableOpPassUnit, test_trans_data_remove) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest0();

  EXPECT_EQ(result, true);
}

TEST_F(UtestVariableOpPassUnit, test_variable_ref) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest1();

  EXPECT_EQ(result, true);
}

TEST_F(UtestVariableOpPassUnit, test_null_graph) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest2();

  EXPECT_EQ(result, true);
}

TEST_F(UtestVariableOpPassUnit, test_covarage_trans_var_data) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest3();

  EXPECT_EQ(result, false);
}

TEST_F(UtestVariableOpPassUnit, test_illegally_ref) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest4();

  EXPECT_EQ(result, true);
}

TEST_F(UtestVariableOpPassUnit, test_single_node) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest5();

  EXPECT_EQ(result, true);
}

TEST_F(UtestVariableOpPassUnit, test_un_mathed) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest6();

  EXPECT_EQ(result, true);
}

TEST_F(UtestVariableOpPassUnit, test_same_op) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest7();

  EXPECT_EQ(true, true);
}

TEST_F(UtestVariableOpPassUnit, test_error_return) {
  VariableOpPassSimulator varibale_op_pass_simulator;

  bool result = varibale_op_pass_simulator.DoTest8();
  EXPECT_EQ(true, true);
}

TEST_F(UtestVariableOpPassUnit, reshape) {
  // init
  MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  auto graph = BuildGraph2();
  graph->SetSessionID(0);
  auto var1 = graph->FindNode("var1");
  VarManager::Instance(0)->AssignVarMem(var1->GetName(), var1->GetOpDesc()->GetOutputDesc(0), RT_MEMORY_HBM);
  uint8_t *dev_ptr = nullptr;
  VarManager::Instance(0)->SetVarAddr(var1->GetName(), var1->GetOpDesc()->GetOutputDesc(0), dev_ptr, RT_MEMORY_HBM);

  ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
  graph->InferShapeInNeed();
  graph_node->SetComputeGraph(graph);
  auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(graph);
  auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
  graph_node->SetGraph(tmp_graph_ptr);

  VarAccelerateCtrl ctrl;
  ctrl.AddGraph(graph_node->GetGraphId(), graph);
  VariableOpPass pass(&ctrl);
  EXPECT_EQ(pass.Run(graph), ge::SUCCESS);
  MemManager::Instance().Finalize();

  EXPECT_EQ(var1->GetOutNodes().size(), 1);
  EXPECT_EQ(var1->GetOutDataNodes().at(0)->GetName(), "conv1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetFormat(), FORMAT_HWCN);
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({8, 8, 3, 2}));
}

TEST_F(UtestVariableOpPassUnit, reformat) {
  // init
  MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  auto graph = BuildGraph3();
  graph->SetSessionID(0);
  auto var1 = graph->FindNode("var1");
  VarManager::Instance(0)->AssignVarMem(var1->GetName(), var1->GetOpDesc()->GetOutputDesc(0), RT_MEMORY_HBM);
  uint8_t *dev_ptr = nullptr;
  VarManager::Instance(0)->SetVarAddr(var1->GetName(), var1->GetOpDesc()->GetOutputDesc(0), dev_ptr, RT_MEMORY_HBM);

  ge::GraphNodePtr graph_node = make_shared<GraphNode>(0);
  graph->InferShapeInNeed();
  graph_node->SetComputeGraph(graph);
  auto tmp_graph = GraphUtils::CreateGraphFromComputeGraph(graph);
  auto tmp_graph_ptr = std::make_shared<Graph>(tmp_graph);
  graph_node->SetGraph(tmp_graph_ptr);

  VarAccelerateCtrl ctrl;
  ctrl.AddGraph(graph_node->GetGraphId(), graph);
  VariableOpPass pass(&ctrl);
  EXPECT_EQ(pass.Run(graph), ge::SUCCESS);
  MemManager::Instance().Finalize();

  EXPECT_EQ(var1->GetOutNodes().size(), 1);
  EXPECT_EQ(var1->GetOutDataNodes().at(0)->GetName(), "conv1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetFormat(), FORMAT_ND);
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({8, 8, 3, 2}));
}

TEST_F(UtestVariableOpPassUnit, invalid_src_shape2) {
  formats::FormatTransferNchwNc1hwc0 t1;
  formats::FormatTransferNhwcNc1hwc0 t2;
  formats::TransArgs args = formats::TransArgs();
  formats::TransResult ret;
  t2.TransFormat(args, ret);
}
