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

#ifndef DPICO_MAPPER_OP_MAPPER_H_
#define DPICO_MAPPER_OP_MAPPER_H_

#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <memory>
#include "common/check_base.h"
#include "common/fetch_content.h"
#include "mindapi/base/base.h"
#include "mindapi/ir/anf.h"
#include "include/errorcode.h"
#include "op/base_operator.h"
#include "op/recurrent_operator.h"
#include "mindapi/base/logging.h"
#include "ops/op_name.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;
namespace mindspore {
namespace dpico {
using BaseOperatorPtr = std::unique_ptr<mapper::BaseOperator>;
class OpMapper {
 public:
  explicit OpMapper(std::string node_name) : name(std::move(node_name)) {}
  virtual ~OpMapper() = default;
  virtual STATUS Map(const api::CNodePtr &node, std::vector<BaseOperatorPtr> *base_operators,
                     const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) = 0;

 private:
  const std::string name;
};
using OpMapperPtr = std::shared_ptr<OpMapper>;

STATUS SetCommonAttr(const api::CNodePtr &node, mapper::BaseOperator *base_operator,
                     const api::CNodePtrList &output_cnodes);
STATUS SetConvFcDataInfo(const api::CNodePtr &cnode, mapper::BaseOperator *base_operator);
STATUS SetRecurrentDataInfo(const api::CNodePtr &cnode, mapper::RecurrentOperator *recurrent_operator);
STATUS PushOfflineArgs(const api::CNodePtr &cnode, mapper::BaseOperator *base_operator, size_t offline_args_size);
}  // namespace dpico
}  // namespace mindspore

#endif  // DPICO_MAPPER_OP_MAPPER_H_
