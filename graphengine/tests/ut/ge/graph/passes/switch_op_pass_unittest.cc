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

#include "omg/omg_inner_types.h"
#define protected public
#define private public

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/graph.h"
#include "inc/pass_manager.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

class UtestGraphPassesSwitchOpPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
  void make_graph(ComputeGraphPtr graph, bool match = true) {
    GeTensorDesc bool_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc int_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
    GeTensorDesc scalar_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

    auto xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
    xOpDef->AddOutputDesc(scalar_tensor_desc);
    auto xNode = graph->AddNode(xOpDef);

    auto yOpDef = std::make_shared<OpDesc>("y", VARIABLEV2);
    yOpDef->AddOutputDesc(scalar_tensor_desc);
    auto yNode = graph->AddNode(yOpDef);

    auto zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
    zOpDef->AddOutputDesc(scalar_tensor_desc);
    auto zNode = graph->AddNode(zOpDef);

    auto condOpDef = std::make_shared<OpDesc>("Less", "Less");
    condOpDef->AddInputDesc(scalar_tensor_desc);
    condOpDef->AddInputDesc(scalar_tensor_desc);
    condOpDef->AddOutputDesc(bool_tensor_desc);
    auto condNode = graph->AddNode(condOpDef);

    auto switch_op_def1 = std::make_shared<OpDesc>("Add/Switch", SWITCH);
    switch_op_def1->AddInputDesc(scalar_tensor_desc);
    switch_op_def1->AddInputDesc(bool_tensor_desc);
    switch_op_def1->AddOutputDesc(scalar_tensor_desc);
    switch_op_def1->AddOutputDesc(scalar_tensor_desc);
    auto switch_node1 = graph->AddNode(switch_op_def1);

    auto switch_op_def2 = std::make_shared<OpDesc>("Add/Switch_1", SWITCH);
    switch_op_def2->AddInputDesc(scalar_tensor_desc);
    switch_op_def2->AddInputDesc(bool_tensor_desc);
    switch_op_def2->AddOutputDesc(scalar_tensor_desc);
    switch_op_def2->AddOutputDesc(scalar_tensor_desc);
    auto switch_node2 = graph->AddNode(switch_op_def2);

    auto switch_op_def3 = std::make_shared<OpDesc>("Square/Switch", SWITCH);
    switch_op_def3->AddInputDesc(scalar_tensor_desc);
    switch_op_def3->AddInputDesc(bool_tensor_desc);
    switch_op_def3->AddOutputDesc(scalar_tensor_desc);
    switch_op_def3->AddOutputDesc(scalar_tensor_desc);
    auto switch_node3 = graph->AddNode(switch_op_def3);

    auto addOpDef = std::make_shared<OpDesc>("Add", "ADD");
    addOpDef->AddInputDesc(scalar_tensor_desc);
    addOpDef->AddInputDesc(scalar_tensor_desc);
    addOpDef->AddOutputDesc(scalar_tensor_desc);
    auto addNode = graph->AddNode(addOpDef);

    auto mergeOpDef = std::make_shared<OpDesc>("Merge", "Merge");
    mergeOpDef->AddInputDesc(scalar_tensor_desc);
    mergeOpDef->AddInputDesc(scalar_tensor_desc);
    mergeOpDef->AddOutputDesc(scalar_tensor_desc);
    mergeOpDef->AddOutputDesc(int_tensor_desc);
    auto mergeNode = graph->AddNode(mergeOpDef);

    auto output_op_def = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    output_op_def->AddInputDesc(scalar_tensor_desc);
    output_op_def->AddOutputDesc(scalar_tensor_desc);
    auto output_node = graph->AddNode(output_op_def);

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switch_node3->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switch_node3->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switch_node1->GetOutDataAnchor(1), addNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switch_node2->GetOutDataAnchor(1), addNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(addNode->GetOutDataAnchor(0), mergeNode->GetInDataAnchor(1));
    (void)GraphUtils::AddEdge(switch_node3->GetOutDataAnchor(0), mergeNode->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(mergeNode->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void make_graph_const(ComputeGraphPtr graph, bool match = true) {
    // resnet50 PolynomialDecay
    GeTensorDesc scalar_tensor_desc(GeShape({1, 1, 1, 1}));
    GeTensorDesc bool_tensor_desc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc int_tensor_desc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_INT32);

    auto xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
    xOpDef->AddOutputDesc(scalar_tensor_desc);
    auto xNode = graph->AddNode(xOpDef);

    auto yOpDef = std::make_shared<OpDesc>("y", "Const");
    yOpDef->AddOutputDesc(scalar_tensor_desc);
    auto yNode = graph->AddNode(yOpDef);

    auto zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
    zOpDef->AddOutputDesc(scalar_tensor_desc);
    auto zNode = graph->AddNode(zOpDef);

    auto constOpDef = std::make_shared<OpDesc>("Const", "Const");
    constOpDef->AddOutputDesc(scalar_tensor_desc);
    auto constNode = graph->AddNode(constOpDef);

    auto condOpDef = std::make_shared<OpDesc>("Equal", "Equal");
    condOpDef->AddInputDesc(scalar_tensor_desc);
    condOpDef->AddInputDesc(scalar_tensor_desc);
    condOpDef->AddOutputDesc(bool_tensor_desc);
    auto condNode = graph->AddNode(condOpDef);

    auto identityOpDef = std::make_shared<OpDesc>("identity", "Identity");
    identityOpDef->AddInputDesc(bool_tensor_desc);
    identityOpDef->AddOutputDesc(bool_tensor_desc);
    auto identityNode = graph->AddNode(identityOpDef);

    auto switch_op_def1 = std::make_shared<OpDesc>("Switch", SWITCH);
    switch_op_def1->AddInputDesc(bool_tensor_desc);
    switch_op_def1->AddInputDesc(bool_tensor_desc);
    switch_op_def1->AddOutputDesc(bool_tensor_desc);
    switch_op_def1->AddOutputDesc(bool_tensor_desc);
    auto switch_node1 = graph->AddNode(switch_op_def1);

    auto tIdentityOpDef = std::make_shared<OpDesc>("switch_t", "Identity");
    tIdentityOpDef->AddInputDesc(scalar_tensor_desc);
    tIdentityOpDef->AddOutputDesc(scalar_tensor_desc);
    auto tIdentityNode = graph->AddNode(tIdentityOpDef);

    auto fIdentityOpDef = std::make_shared<OpDesc>("switch_f", "Identity");
    fIdentityOpDef->AddInputDesc(scalar_tensor_desc);
    fIdentityOpDef->AddOutputDesc(scalar_tensor_desc);
    auto fIdentityNode = graph->AddNode(fIdentityOpDef);

    auto switch_op_def2 = std::make_shared<OpDesc>("Switch_1", SWITCH);
    switch_op_def2->AddInputDesc(scalar_tensor_desc);
    switch_op_def2->AddInputDesc(bool_tensor_desc);
    switch_op_def2->AddOutputDesc(scalar_tensor_desc);
    switch_op_def2->AddOutputDesc(scalar_tensor_desc);
    auto switch_node2 = graph->AddNode(switch_op_def2);

    auto mulOpDef = std::make_shared<OpDesc>("truediv", "Mul");
    mulOpDef->AddInputDesc(scalar_tensor_desc);
    mulOpDef->AddInputDesc(scalar_tensor_desc);
    mulOpDef->AddOutputDesc(scalar_tensor_desc);
    auto mulNode = graph->AddNode(mulOpDef);

    auto ceilOpDef = std::make_shared<OpDesc>("Ceil", "Ceil");
    ceilOpDef->AddInputDesc(scalar_tensor_desc);
    ceilOpDef->AddOutputDesc(scalar_tensor_desc);
    auto ceilNode = graph->AddNode(ceilOpDef);

    auto mergeOpDef = std::make_shared<OpDesc>("Merge", "Merge");
    mergeOpDef->AddInputDesc(scalar_tensor_desc);
    mergeOpDef->AddInputDesc(scalar_tensor_desc);
    mergeOpDef->AddOutputDesc(scalar_tensor_desc);
    mergeOpDef->AddOutputDesc(int_tensor_desc);
    auto mergeNode = graph->AddNode(mergeOpDef);

    auto output_op_def = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    output_op_def->AddInputDesc(scalar_tensor_desc);
    output_op_def->AddOutputDesc(scalar_tensor_desc);
    auto output_node = graph->AddNode(output_op_def);

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), identityNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(identityNode->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(identityNode->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switch_node1->GetOutDataAnchor(0), fIdentityNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switch_node1->GetOutDataAnchor(1), tIdentityNode->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(fIdentityNode->GetOutControlAnchor(), zNode->GetInControlAnchor());
    (void)GraphUtils::AddEdge(tIdentityNode->GetOutControlAnchor(), constNode->GetInControlAnchor());

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(identityNode->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), mulNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switch_node2->GetOutDataAnchor(0), mulNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0), ceilNode->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(constNode->GetOutDataAnchor(0), mergeNode->GetInDataAnchor(1));
    (void)GraphUtils::AddEdge(ceilNode->GetOutDataAnchor(0), mergeNode->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(mergeNode->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void make_graph_cyclic_dependence(ComputeGraphPtr graph, bool match = true) {
    GeTensorDesc scalar_tensor_desc(GeShape({1, 1, 1, 1}));
    GeTensorDesc bool_tensor_desc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc int_tensor_desc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_INT32);

    auto xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
    xOpDef->AddOutputDesc(scalar_tensor_desc);
    auto xNode = graph->AddNode(xOpDef);

    auto yOpDef = std::make_shared<OpDesc>("y", VARIABLEV2);
    yOpDef->AddOutputDesc(scalar_tensor_desc);
    auto yNode = graph->AddNode(yOpDef);

    auto zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
    zOpDef->AddOutputDesc(scalar_tensor_desc);
    auto zNode = graph->AddNode(zOpDef);

    auto condOpDef = std::make_shared<OpDesc>("Less", "Less");
    condOpDef->AddInputDesc(scalar_tensor_desc);
    condOpDef->AddInputDesc(scalar_tensor_desc);
    condOpDef->AddOutputDesc(bool_tensor_desc);
    auto condNode = graph->AddNode(condOpDef);

    auto switch_op_def1 = std::make_shared<OpDesc>("Switch_f_1", SWITCH);
    switch_op_def1->AddInputDesc(scalar_tensor_desc);
    switch_op_def1->AddInputDesc(bool_tensor_desc);
    switch_op_def1->AddOutputDesc(scalar_tensor_desc);
    switch_op_def1->AddOutputDesc(scalar_tensor_desc);
    auto switch_node1 = graph->AddNode(switch_op_def1);

    auto switch_op_def2 = std::make_shared<OpDesc>("Switch_t_1", SWITCH);
    switch_op_def2->AddInputDesc(scalar_tensor_desc);
    switch_op_def2->AddInputDesc(bool_tensor_desc);
    switch_op_def2->AddOutputDesc(scalar_tensor_desc);
    switch_op_def2->AddOutputDesc(scalar_tensor_desc);
    auto switch_node2 = graph->AddNode(switch_op_def2);

    auto switch_op_def3 = std::make_shared<OpDesc>("Switch_f_2", SWITCH);
    switch_op_def3->AddInputDesc(scalar_tensor_desc);
    switch_op_def3->AddInputDesc(bool_tensor_desc);
    switch_op_def3->AddOutputDesc(scalar_tensor_desc);
    switch_op_def3->AddOutputDesc(scalar_tensor_desc);
    auto switch_node3 = graph->AddNode(switch_op_def3);

    auto switch_op_def4 = std::make_shared<OpDesc>("Switch_t_2", SWITCH);
    switch_op_def4->AddInputDesc(scalar_tensor_desc);
    switch_op_def4->AddInputDesc(bool_tensor_desc);
    switch_op_def4->AddOutputDesc(scalar_tensor_desc);
    switch_op_def4->AddOutputDesc(scalar_tensor_desc);
    auto switch_node4 = graph->AddNode(switch_op_def4);

    auto squareOpDef1 = std::make_shared<OpDesc>("Square1", "Square");
    squareOpDef1->AddInputDesc(scalar_tensor_desc);
    squareOpDef1->AddOutputDesc(scalar_tensor_desc);
    auto squareNode1 = graph->AddNode(squareOpDef1);

    auto squareOpDef2 = std::make_shared<OpDesc>("Square2", "Square");
    squareOpDef2->AddInputDesc(scalar_tensor_desc);
    squareOpDef2->AddOutputDesc(scalar_tensor_desc);
    auto squareNode2 = graph->AddNode(squareOpDef2);

    auto squareOpDef3 = std::make_shared<OpDesc>("Square3", "Square");
    squareOpDef3->AddInputDesc(scalar_tensor_desc);
    squareOpDef3->AddOutputDesc(scalar_tensor_desc);
    auto squareNode3 = graph->AddNode(squareOpDef3);

    auto squareOpDef4 = std::make_shared<OpDesc>("Square4", "Square");
    squareOpDef4->AddInputDesc(scalar_tensor_desc);
    squareOpDef4->AddOutputDesc(scalar_tensor_desc);
    auto squareNode4 = graph->AddNode(squareOpDef4);

    auto merge_op_def1 = std::make_shared<OpDesc>("Merge1", "Merge");
    merge_op_def1->AddInputDesc(scalar_tensor_desc);
    merge_op_def1->AddInputDesc(scalar_tensor_desc);
    merge_op_def1->AddOutputDesc(scalar_tensor_desc);
    merge_op_def1->AddOutputDesc(int_tensor_desc);
    auto merge_node1 = graph->AddNode(merge_op_def1);

    auto merge_op_def2 = std::make_shared<OpDesc>("Merge2", "Merge");
    merge_op_def2->AddInputDesc(scalar_tensor_desc);
    merge_op_def2->AddInputDesc(scalar_tensor_desc);
    merge_op_def2->AddOutputDesc(scalar_tensor_desc);
    merge_op_def2->AddOutputDesc(int_tensor_desc);
    auto merge_node2 = graph->AddNode(merge_op_def2);

    auto output_op_def = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    output_op_def->AddInputDesc(scalar_tensor_desc);
    output_op_def->AddOutputDesc(scalar_tensor_desc);
    auto output_node = graph->AddNode(output_op_def);

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), condNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switch_node1->GetOutDataAnchor(0), squareNode1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switch_node2->GetOutDataAnchor(1), squareNode2->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(squareNode1->GetOutDataAnchor(0), merge_node1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(squareNode2->GetOutDataAnchor(0), merge_node1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(merge_node1->GetOutDataAnchor(0), switch_node3->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switch_node3->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switch_node4->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(condNode->GetOutDataAnchor(0), switch_node4->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switch_node3->GetOutDataAnchor(0), squareNode3->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switch_node4->GetOutDataAnchor(1), squareNode4->GetInDataAnchor(0));

    (void)GraphUtils::AddEdge(squareNode3->GetOutDataAnchor(0), merge_node2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(squareNode4->GetOutDataAnchor(0), merge_node2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(merge_node2->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void make_graph_case(ComputeGraphPtr graph, bool match = true) {
    GeTensorDesc scalar_tensor_desc(GeShape({1, 1, 1, 1}));
    GeTensorDesc bool_tensor_desc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_BOOL);
    GeTensorDesc int_tensor_desc(GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_INT32);

    auto xOpDef = std::make_shared<OpDesc>("x", VARIABLEV2);
    xOpDef->AddOutputDesc(scalar_tensor_desc);
    auto xNode = graph->AddNode(xOpDef);

    auto yOpDef = std::make_shared<OpDesc>("y", VARIABLEV2);
    yOpDef->AddOutputDesc(scalar_tensor_desc);
    auto yNode = graph->AddNode(yOpDef);

    auto zOpDef = std::make_shared<OpDesc>("z", VARIABLEV2);
    zOpDef->AddOutputDesc(scalar_tensor_desc);
    auto zNode = graph->AddNode(zOpDef);

    auto greater_op_def = std::make_shared<OpDesc>("Greater", "Greater");
    greater_op_def->AddInputDesc(scalar_tensor_desc);
    greater_op_def->AddInputDesc(scalar_tensor_desc);
    greater_op_def->AddOutputDesc(bool_tensor_desc);
    auto greaterNode = graph->AddNode(greater_op_def);

    auto less_op_def = std::make_shared<OpDesc>("Less", "Less");
    less_op_def->AddInputDesc(scalar_tensor_desc);
    less_op_def->AddInputDesc(scalar_tensor_desc);
    less_op_def->AddOutputDesc(bool_tensor_desc);
    auto less_node = graph->AddNode(less_op_def);

    auto switch_op_def1 = std::make_shared<OpDesc>("greater/Switch_t", SWITCH);
    switch_op_def1->AddInputDesc(bool_tensor_desc);
    switch_op_def1->AddInputDesc(bool_tensor_desc);
    switch_op_def1->AddOutputDesc(bool_tensor_desc);
    switch_op_def1->AddOutputDesc(bool_tensor_desc);
    auto switch_node1 = graph->AddNode(switch_op_def1);

    auto switch_op_def2 = std::make_shared<OpDesc>("greater/Switch_f", SWITCH);
    switch_op_def2->AddInputDesc(scalar_tensor_desc);
    switch_op_def2->AddInputDesc(bool_tensor_desc);
    switch_op_def2->AddOutputDesc(scalar_tensor_desc);
    switch_op_def2->AddOutputDesc(scalar_tensor_desc);
    auto switch_node2 = graph->AddNode(switch_op_def2);

    auto switch_op_def3 = std::make_shared<OpDesc>("less/Switch_t", SWITCH);
    switch_op_def3->AddInputDesc(scalar_tensor_desc);
    switch_op_def3->AddInputDesc(bool_tensor_desc);
    switch_op_def3->AddOutputDesc(scalar_tensor_desc);
    switch_op_def3->AddOutputDesc(scalar_tensor_desc);
    auto switch_node3 = graph->AddNode(switch_op_def3);

    auto switch_op_def4 = std::make_shared<OpDesc>("less/Switch_f", SWITCH);
    switch_op_def4->AddInputDesc(scalar_tensor_desc);
    switch_op_def4->AddInputDesc(bool_tensor_desc);
    switch_op_def4->AddOutputDesc(scalar_tensor_desc);
    switch_op_def4->AddOutputDesc(scalar_tensor_desc);
    auto switch_node4 = graph->AddNode(switch_op_def4);

    auto merge_op_def1 = std::make_shared<OpDesc>("Merge1", "Merge");
    merge_op_def1->AddInputDesc(scalar_tensor_desc);
    merge_op_def1->AddInputDesc(scalar_tensor_desc);
    merge_op_def1->AddOutputDesc(scalar_tensor_desc);
    merge_op_def1->AddOutputDesc(int_tensor_desc);
    auto merge_node1 = graph->AddNode(merge_op_def1);

    auto merge_op_def2 = std::make_shared<OpDesc>("Merge2", "Merge");
    merge_op_def2->AddInputDesc(scalar_tensor_desc);
    merge_op_def2->AddInputDesc(scalar_tensor_desc);
    merge_op_def2->AddOutputDesc(scalar_tensor_desc);
    merge_op_def2->AddOutputDesc(int_tensor_desc);
    auto merge_node2 = graph->AddNode(merge_op_def2);

    auto output_op_def = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    output_op_def->AddInputDesc(scalar_tensor_desc);
    output_op_def->AddOutputDesc(scalar_tensor_desc);
    auto output_node = graph->AddNode(output_op_def);

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), greaterNode->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), greaterNode->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), less_node->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), less_node->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(xNode->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(greaterNode->GetOutDataAnchor(0), switch_node1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(less_node->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(greaterNode->GetOutDataAnchor(0), switch_node2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(yNode->GetOutDataAnchor(0), switch_node3->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switch_node2->GetOutDataAnchor(0), switch_node3->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(zNode->GetOutDataAnchor(0), switch_node4->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switch_node2->GetOutDataAnchor(0), switch_node4->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switch_node3->GetOutDataAnchor(1), merge_node1->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(switch_node4->GetOutDataAnchor(0), merge_node1->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(switch_node1->GetOutDataAnchor(1), merge_node2->GetInDataAnchor(0));
    (void)GraphUtils::AddEdge(merge_node1->GetOutDataAnchor(0), merge_node2->GetInDataAnchor(1));

    (void)GraphUtils::AddEdge(merge_node2->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }
};
