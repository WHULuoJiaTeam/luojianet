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
#include <iostream>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph_builder_utils.h"

namespace ge {
namespace {
class SubInDataAnchor : public InDataAnchor
{
public:

  SubInDataAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubInDataAnchor();

  bool EncaEq(const AnchorPtr anchor);
  bool EncaIsTypeOf(const TYPE type);
  void SetImpNull();
  template <class T>
  static Anchor::TYPE EncaTypeOf() {
    return Anchor::TypeOf<T>();
  }

};

SubInDataAnchor::SubInDataAnchor(const NodePtr &owner_node, const int32_t idx):InDataAnchor(owner_node,idx){}

SubInDataAnchor::~SubInDataAnchor() = default;

bool SubInDataAnchor::EncaEq(const AnchorPtr anchor){
  return Equal(anchor);
}

bool SubInDataAnchor::EncaIsTypeOf(const TYPE type){
  return IsTypeOf(type);
}

void SubInDataAnchor::SetImpNull(){
  impl_ = nullptr;
}

class SubOutDataAnchor : public OutDataAnchor
{
public:

  SubOutDataAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubOutDataAnchor();

  bool EncaEq(const AnchorPtr anchor);
  bool EncaIsTypeOf(const TYPE type);
  void SetImpNull();

};

SubOutDataAnchor::SubOutDataAnchor(const NodePtr &owner_node, const int32_t idx):OutDataAnchor(owner_node,idx){}

SubOutDataAnchor::~SubOutDataAnchor() = default;

bool SubOutDataAnchor::EncaEq(const AnchorPtr anchor){
  return Equal(anchor);
}

bool SubOutDataAnchor::EncaIsTypeOf(const TYPE type){
  return IsTypeOf(type);
}

void SubOutDataAnchor::SetImpNull(){
  impl_ = nullptr;
}

class SubInControlAnchor : public InControlAnchor
{
public:

  SubInControlAnchor(const NodePtr &owner_node);
  SubInControlAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubInControlAnchor();

  bool EncaEq(const AnchorPtr anchor);
  bool EncaIsTypeOf(const TYPE type);
  void SetImpNull();

};

SubInControlAnchor::SubInControlAnchor(const NodePtr &owner_node):InControlAnchor(owner_node){}
SubInControlAnchor::SubInControlAnchor(const NodePtr &owner_node, const int32_t idx):InControlAnchor(owner_node,idx){}

SubInControlAnchor::~SubInControlAnchor() = default;

bool SubInControlAnchor::EncaEq(const AnchorPtr anchor){
  return Equal(anchor);
}

bool SubInControlAnchor::EncaIsTypeOf(const TYPE type){
  return IsTypeOf(type);
}

void SubInControlAnchor::SetImpNull(){
  impl_ = nullptr;
}

class SubOutControlAnchor : public OutControlAnchor
{
public:

  SubOutControlAnchor(const NodePtr &owner_node);

  SubOutControlAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubOutControlAnchor();

  bool EncaEq(const AnchorPtr anchor);
  bool EncaIsTypeOf(const TYPE type);
  void SetImpNull();

};

SubOutControlAnchor::SubOutControlAnchor(const NodePtr &owner_node):OutControlAnchor(owner_node){}
SubOutControlAnchor::SubOutControlAnchor(const NodePtr &owner_node, const int32_t idx):OutControlAnchor(owner_node,idx){}

SubOutControlAnchor::~SubOutControlAnchor() = default;

bool SubOutControlAnchor::EncaEq(const AnchorPtr anchor){
  return Equal(anchor);
}

bool SubOutControlAnchor::EncaIsTypeOf(const TYPE type){
  return IsTypeOf(type);
}

void SubOutControlAnchor::SetImpNull(){
  impl_ = nullptr;
}

}

using SubInDataAnchorPtr = std::shared_ptr<SubInDataAnchor>;
using SubOutDataAnchorPtr = std::shared_ptr<SubOutDataAnchor>;
using SubInControlAnchorPtr = std::shared_ptr<SubInControlAnchor>;
using SubOutControlAnchorPtr = std::shared_ptr<SubOutControlAnchor>;

class AnchorUt : public testing::Test {};

TEST_F(AnchorUt, SubInDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  SubInDataAnchorPtr in_anch = std::make_shared<SubInDataAnchor>(node, 111);
  in_anch->SetIdx(222);
  EXPECT_EQ(in_anch->GetIdx(), 222);
  EXPECT_EQ(in_anch->EncaEq(Anchor::DynamicAnchorCast<Anchor>(in_anch)), true);
  EXPECT_EQ(in_anch->GetPeerAnchorsSize(),0);
  EXPECT_EQ(in_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_EQ(in_anch->GetOwnerNode(),node);
  EXPECT_EQ(in_anch->IsLinkedWith(nullptr), false);
  EXPECT_EQ(in_anch->GetPeerOutAnchor(), nullptr);
  EXPECT_EQ(in_anch->LinkFrom(nullptr), GRAPH_FAILED);
  auto node2 = builder.AddNode("Data", "Data", 2, 2);
  OutDataAnchorPtr peer = std::make_shared<OutDataAnchor>(node2, 22);
  EXPECT_EQ(in_anch->LinkFrom(peer), GRAPH_SUCCESS);
  EXPECT_EQ(in_anch->IsLinkedWith(peer), true);
  EXPECT_EQ(in_anch->GetPeerAnchorsSize(),1);
  EXPECT_EQ(in_anch->GetPeerAnchors().size(),1);
  EXPECT_NE(in_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_NE(in_anch->GetOwnerNode(),nullptr);
  auto node3 = builder.AddNode("Data", "Data", 3, 3);
  SubInDataAnchorPtr first = std::make_shared<SubInDataAnchor>(node3, 33);
  auto node4 = builder.AddNode("Data", "Data", 4, 4);
  OutDataAnchorPtr second = std::make_shared<OutDataAnchor>(node4, 44);
  EXPECT_EQ(in_anch->ReplacePeer(peer,first,second),GRAPH_SUCCESS);

  auto node5 = builder.AddNode("Data", "Data", 5, 5);
  OutDataAnchorPtr oa5 = std::make_shared<OutDataAnchor>(node5, 55);
  auto node6 = builder.AddNode("Data", "Data", 6, 6);
  SubInDataAnchorPtr ia6 = std::make_shared<SubInDataAnchor>(node, 66);
  EXPECT_EQ(ia6->LinkFrom(oa5), GRAPH_SUCCESS);
  EXPECT_EQ(ia6->Unlink(oa5), GRAPH_SUCCESS);

  EXPECT_EQ(in_anch->Unlink(nullptr),GRAPH_FAILED);
  EXPECT_EQ(in_anch->EncaEq(nullptr),false);
  EXPECT_EQ(in_anch->EncaIsTypeOf("nnn"),false);
  EXPECT_NE(in_anch->DynamicAnchorCast<InDataAnchor>(in_anch),nullptr);
  EXPECT_EQ(in_anch->DynamicAnchorCast<OutDataAnchor>(in_anch),nullptr);
}


TEST_F(AnchorUt, SubOutDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  SubOutDataAnchorPtr out_anch = std::make_shared<SubOutDataAnchor>(node, 111);
  out_anch->SetIdx(222);
  EXPECT_EQ(out_anch->GetIdx(), 222);
  EXPECT_EQ(out_anch->GetPeerAnchorsSize(),0);
  EXPECT_EQ(out_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_EQ(out_anch->GetOwnerNode(),node);
  EXPECT_EQ(out_anch->IsLinkedWith(nullptr), false);
  auto node2 = builder.AddNode("Data", "Data", 2, 2);
  InDataAnchorPtr peer = std::make_shared<InDataAnchor>(node2, 22);
  EXPECT_EQ(out_anch->LinkTo(peer), GRAPH_SUCCESS);
  auto node3 = builder.AddNode("Data", "Data", 3, 3);
  InControlAnchorPtr peerctr = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(peerctr), GRAPH_SUCCESS);
  EXPECT_EQ(peerctr->GetPeerOutDataAnchors().size(), 1);
  EXPECT_EQ(out_anch->GetPeerAnchorsSize(),2);
  EXPECT_EQ(out_anch->GetPeerAnchors().size(),2);
  EXPECT_NE(out_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_NE(out_anch->GetOwnerNode(),nullptr);
  auto node22 = builder.AddNode("Data", "Data", 22, 22);
  SubInDataAnchorPtr peerd2 = std::make_shared<SubInDataAnchor>(node2, 222);
  peerd2->SetImpNull();
  EXPECT_EQ(out_anch->LinkTo(peerd2), GRAPH_FAILED);
  auto node33 = builder.AddNode("Data", "Data", 33, 33);
  SubInControlAnchorPtr peerctr2 = std::make_shared<SubInControlAnchor>(node3, 333);
  peerctr2->SetImpNull();
  EXPECT_EQ(out_anch->LinkTo(peerctr2), GRAPH_FAILED);

  EXPECT_EQ(out_anch->Unlink(nullptr),GRAPH_FAILED);
  EXPECT_EQ(out_anch->EncaEq(nullptr),false);
  EXPECT_EQ(out_anch->EncaIsTypeOf("nnn"),false);
  out_anch->SetImpNull();
  auto nodelast = builder.AddNode("Data", "Data", 23, 23);
  SubInDataAnchorPtr peerd23 = std::make_shared<SubInDataAnchor>(nodelast, 223);
  EXPECT_EQ(out_anch->LinkTo(peerd23), GRAPH_FAILED);
}


TEST_F(AnchorUt, SubInControlAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 111, 1);
  SubInControlAnchorPtr in_canch0 = std::make_shared<SubInControlAnchor>(node0);
  EXPECT_NE(in_canch0, nullptr);
  auto node = builder.AddNode("Data", "Data", 1, 1);
  SubInControlAnchorPtr inc_anch = std::make_shared<SubInControlAnchor>(node, 111);
  inc_anch->SetIdx(222);
  EXPECT_EQ(inc_anch->GetIdx(), 222);
  EXPECT_EQ(inc_anch->GetPeerAnchorsSize(),0);
  EXPECT_EQ(inc_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_EQ(inc_anch->GetOwnerNode(),node);
  EXPECT_EQ(inc_anch->IsLinkedWith(nullptr), false);
  EXPECT_EQ(inc_anch->LinkFrom(nullptr), GRAPH_FAILED);
  auto node2 = builder.AddNode("Data", "Data", 2, 2);
  OutControlAnchorPtr peer = std::make_shared<OutControlAnchor>(node2, 22);
  EXPECT_EQ(inc_anch->LinkFrom(peer), GRAPH_SUCCESS);
  EXPECT_EQ(inc_anch->IsPeerOutAnchorsEmpty(),false);
  EXPECT_EQ(inc_anch->GetPeerAnchorsSize(),1);
  EXPECT_EQ(inc_anch->GetPeerAnchors().size(),1);
  EXPECT_EQ(inc_anch->GetPeerOutDataAnchors().size(), 0);
  EXPECT_NE(inc_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_NE(inc_anch->GetOwnerNode(),nullptr);
  auto node3 = builder.AddNode("Data", "Data", 3, 3);
  SubInControlAnchorPtr first = std::make_shared<SubInControlAnchor>(node3, 33);
  auto node4 = builder.AddNode("Data", "Data", 4, 4);
  OutControlAnchorPtr second = std::make_shared<OutControlAnchor>(node4, 44);
  EXPECT_EQ(inc_anch->ReplacePeer(peer,first,second),GRAPH_SUCCESS);
  EXPECT_EQ(inc_anch->Unlink(nullptr),GRAPH_FAILED);
  EXPECT_EQ(inc_anch->EncaEq(nullptr),false);
  EXPECT_EQ(inc_anch->EncaIsTypeOf("nnn"),false);
  inc_anch->UnlinkAll();
}


TEST_F(AnchorUt, SubOutControlAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node0 = builder.AddNode("Data", "Data", 111, 1);
  SubOutControlAnchorPtr out_canch0 = std::make_shared<SubOutControlAnchor>(node0);
  EXPECT_NE(out_canch0, nullptr);
  auto node = builder.AddNode("Data", "Data", 1, 1);
  SubOutControlAnchorPtr outc_anch = std::make_shared<SubOutControlAnchor>(node, 111);
  outc_anch->SetIdx(222);
  EXPECT_EQ(outc_anch->GetIdx(), 222);
  EXPECT_EQ(outc_anch->GetPeerAnchorsSize(),0);
  EXPECT_EQ(outc_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_EQ(outc_anch->GetOwnerNode(),node);
  EXPECT_EQ(outc_anch->IsLinkedWith(nullptr), false);
  auto node2 = builder.AddNode("Data", "Data", 2, 2);
  InDataAnchorPtr peer = std::make_shared<InDataAnchor>(node2, 22);
  EXPECT_EQ(outc_anch->LinkTo(peer), GRAPH_SUCCESS);
  auto node3 = builder.AddNode("Data", "Data", 3, 3);
  InControlAnchorPtr peerctr = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(outc_anch->LinkTo(peerctr), GRAPH_SUCCESS);
  EXPECT_EQ(outc_anch->GetPeerAnchorsSize(),2);
  EXPECT_EQ(outc_anch->GetPeerAnchors().size(),2);
  EXPECT_NE(outc_anch->GetFirstPeerAnchor(),nullptr);
  EXPECT_NE(outc_anch->GetOwnerNode(),nullptr);
  auto node22 = builder.AddNode("Data", "Data", 22, 22);
  SubInDataAnchorPtr peerd2 = std::make_shared<SubInDataAnchor>(node2, 222);
  peerd2->SetImpNull();
  EXPECT_EQ(outc_anch->LinkTo(peerd2), GRAPH_FAILED);
  auto node33 = builder.AddNode("Data", "Data", 33, 33);
  SubInControlAnchorPtr peerctr2 = std::make_shared<SubInControlAnchor>(node3, 333);
  peerctr2->SetImpNull();
  EXPECT_EQ(outc_anch->LinkTo(peerctr2), GRAPH_FAILED);
  EXPECT_NE(outc_anch->GetPeerInControlAnchors().size(), 0);
  EXPECT_NE(outc_anch->GetPeerInDataAnchors().size(), 0);

  EXPECT_EQ(outc_anch->Unlink(nullptr),GRAPH_FAILED);
  EXPECT_EQ(outc_anch->EncaEq(nullptr),false);
  EXPECT_EQ(outc_anch->EncaIsTypeOf("nnn"),false);
  outc_anch->SetImpNull();
  auto nodelast = builder.AddNode("Data", "Data", 23, 23);
  SubInDataAnchorPtr peerd23 = std::make_shared<SubInDataAnchor>(nodelast, 223);
  EXPECT_EQ(outc_anch->LinkTo(peerd23), GRAPH_FAILED);
}


}  // namespace ge