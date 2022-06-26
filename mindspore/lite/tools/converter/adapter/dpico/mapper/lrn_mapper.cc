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

#include "mapper/lrn_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "ops/lrn.h"
#include "op/lrn_operator.h"

namespace mindspore {
namespace dpico {
STATUS LrnMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                      const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto lrn_prim = api::utils::cast<api::SharedPtr<ops::LRN>>(prim);
  MS_ASSERT(lrn_prim != nullptr);

  auto lrn_operator = std::make_unique<mapper::LrnOperator>();
  if (lrn_operator == nullptr) {
    MS_LOG(ERROR) << "lrn_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, lrn_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  lrn_operator->SetOpType(mapper::OpType::LRN);
  auto local_size = lrn_prim->get_depth_radius() * 2 + 1;
  lrn_operator->SetLrnLocalSize(local_size);
  lrn_operator->SetLrnAlpha(lrn_prim->get_alpha() * local_size);
  lrn_operator->SetLrnBeta(lrn_prim->get_beta());
  if (lrn_prim->GetAttr(kLrnK) != nullptr) {
    lrn_operator->SetLrnK(api::GetValue<float>(lrn_prim->GetAttr(kLrnK)));
  }
  if (PushOfflineArgs(cnode, lrn_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(lrn_operator));
  return RET_OK;
}
REG_MAPPER(LRN, LrnMapper)
}  // namespace dpico
}  // namespace mindspore
