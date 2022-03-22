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
#include "test_structs.h"
#include "func_counter.h"
#include "graph/detail/attributes_holder.h"
#include "graph/ge_attr_value.h"
#include "graph/any_value.h"
#include "ge_ir.pb.h"
#undef private
#undef protected

namespace ge {
namespace {

class SubAttrStore : public AttrStore {
public:
  bool SetAnyValueByName(const std::string &name, const AnyValue &value);

};


bool SubAttrStore::SetAnyValueByName(const std::string &name, const AnyValue &value){
  return false;
}

class SubAttrHolder : public AttrHolder {
public:
  SubAttrHolder();
  virtual ~SubAttrHolder() = default;


protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

public:
  SubAttrStore attrs_;
};

SubAttrHolder::SubAttrHolder(){
  attrs_ = SubAttrStore();
}

ProtoAttrMap &SubAttrHolder::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &SubAttrHolder::GetAttrMap() const {
  return attrs_;
}

}

void oper(AnyValue::OperateType ot, const AnyValue *av, void *out){
  return;
}

class AttrHolderUt : public testing::Test {};

TEST_F(AttrHolderUt, All) {

  GeIrProtoHelper<proto::TensorDescriptor> helper1;
  helper1.InitDefault();

  GeIrProtoHelper<proto::ShapeDef> helper2;
  helper2.InitDefault();

  GeIrProtoHelper<proto::NamedAttrs> helper3;
  helper3.InitDefault();

  GeIrProtoHelper<proto::ModelDef> helper4;
  helper4.InitDefault();

  GeIrProtoHelper<proto::OpDef> helper5;
  helper5.InitDefault();

  GeIrProtoHelper<proto::GraphDef> helper6;
  helper6.InitDefault();

}

TEST_F(AttrHolderUt, Plus) {

  SubAttrHolder sub_attr_hodler = SubAttrHolder();
  AnyValue av = AnyValue::CreateFrom<int>(1);
  av.operate_ = oper;
  EXPECT_EQ(sub_attr_hodler.SetAttr("name", av), GRAPH_SUCCESS);
  av.operate_ = nullptr;
  EXPECT_EQ(sub_attr_hodler.TrySetAttr("name", av), GRAPH_FAILED);
  EXPECT_EQ(sub_attr_hodler.AddRequiredAttr("name"), GRAPH_FAILED);
}


}  // namespace ge