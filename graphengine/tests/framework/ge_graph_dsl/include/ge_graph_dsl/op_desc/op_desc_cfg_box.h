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

#ifndef HF55B1FFE_C64C_4671_8A25_A57DDD5D1280
#define HF55B1FFE_C64C_4671_8A25_A57DDD5D1280

#include "easy_graph/graph/node_id.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/op_desc/op_box.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg.h"
#include "graph/ge_attr_value.h"
#include "graph/op_desc.h"

GE_NS_BEGIN

struct OpDescCfgBox : OpBox, private OpDescCfg {
  OpDescCfgBox(const OpType &opType);
  OpDescCfgBox &InCnt(int in_cnt);
  OpDescCfgBox &OutCnt(int out_cnt);
  OpDescCfgBox &ParentNodeIndex(int node_index);
  OpDescCfgBox &TensorDesc(Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                           std::vector<int64_t> shape = {1, 1, 224, 224});
  OpDescCfgBox &Weight(GeTensorPtr &);

  template <typename Type>
  OpDescCfgBox &Attr(const std::string &name, Type &&value) {
    auto attrvalue = ge::GeAttrValue::CreateFrom<Type>(std::forward<Type>(value));
    attrs_.emplace(std::make_pair(name, attrvalue));
    return *this;
  }

  template <typename Type>
  OpDescCfgBox &Attr(const std::string &name, Type &value) {
    auto attrvalue = ge::GeAttrValue::CreateFrom<Type>(value);
    attrs_.emplace(std::make_pair(name, attrvalue));
    return *this;
  }

  OpDescCfgBox &Attr(const std::string &name, int value);
  OpDescCfgBox &Attr(const std::string &name, const char *value);
  OpDescPtr Build(const ::EG_NS::NodeId &id) const override;

 private:
  void UpdateAttrs(OpDescPtr &) const;
  std::map<std::string, GeAttrValue> attrs_;
};

#define OP_CFG(optype) ::GE_NS::OpDescCfgBox(optype)

GE_NS_END

#endif /* HF55B1FFE_C64C_4671_8A25_A57DDD5D1280 */
