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

#include "ge_graph_dsl/op_desc/op_desc_cfg_repo.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg.h"

GE_NS_BEGIN

namespace {

#define OP_CFG(optype, ...)                   \
  {                                           \
    optype, OpDescCfg { optype, __VA_ARGS__ } \
  }

static std::map<OpType, OpDescCfg> cfg_repo{OP_CFG(DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(ADD, 2, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(ENTER, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(MERGE, 2, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(LESS, 2, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(LOOPCOND, 1, 1, FORMAT_NCHW, DT_BOOL, {1, 1, 224, 224}),
                                            OP_CFG(SWITCH, 2, 2, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(EXIT, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(NEXTITERATION, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(NETOUTPUT, 2, 2, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(VARIABLE, 1, 1)};
}  // namespace

const OpDescCfg *OpDescCfgRepo::FindBy(const OpType &id) {
  auto it = cfg_repo.find(id);
  if (it == cfg_repo.end()) {
    return nullptr;
  }
  return &(it->second);
}

GE_NS_END
