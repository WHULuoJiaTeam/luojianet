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

#include "mapper/lstm_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/lstm_operator.h"

namespace mindspore {
namespace dpico {
STATUS LstmMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                       const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto lstm_operator = std::make_unique<mapper::LstmOperator>();
  if (lstm_operator == nullptr) {
    MS_LOG(ERROR) << "lstm_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, lstm_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (prim->GetAttr(kNumOutput) != nullptr) {
    lstm_operator->SetRecurrentNumOutput(static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kNumOutput))));
  }
  if (prim->GetAttr(kExposeHidden) != nullptr) {
    lstm_operator->SetRecurrentExposeHidden(api::GetValue<bool>(prim->GetAttr(kExposeHidden)));
  }
  if (prim->GetAttr(kOutputLastFrameFlag) != nullptr) {
    lstm_operator->SetOutputLastFrameFlag(api::GetValue<bool>(prim->GetAttr(kOutputLastFrameFlag)));
  }
  if (prim->GetAttr(kInitialHOnlineFlag) != nullptr) {
    lstm_operator->SetInitialHOnlineFlag(api::GetValue<bool>(prim->GetAttr(kInitialHOnlineFlag)));
  }
  if (prim->GetAttr(kUseDefaultInitialHFlag) != nullptr) {
    lstm_operator->SetUseDefaultInitialHFlag(api::GetValue<bool>(prim->GetAttr(kUseDefaultInitialHFlag)));
  }
  if (prim->GetAttr(kInitialCOnlineFlag) != nullptr) {
    lstm_operator->SetInitialCOnlineFlag(api::GetValue<bool>(prim->GetAttr(kInitialCOnlineFlag)));
  }
  if (prim->GetAttr(kUseDefaultInitialCFlag) != nullptr) {
    lstm_operator->SetUseDefaultInitialCFlag(api::GetValue<bool>(prim->GetAttr(kUseDefaultInitialCFlag)));
  }
  if (prim->GetAttr(kKeepDirectionDimFlag) != nullptr) {
    lstm_operator->SetKeepDirectionDimFlag(api::GetValue<bool>(prim->GetAttr(kKeepDirectionDimFlag)));
  }
  if (prim->GetAttr(kPeepHoleFlag) != nullptr) {
    lstm_operator->SetPeepholeFlag(api::GetValue<bool>(prim->GetAttr(kPeepHoleFlag)));
  }
  if (prim->GetAttr(kLstmWeightOrderIofcFlag) != nullptr) {
    lstm_operator->SetLstmWeightOrderIofcFlag(api::GetValue<bool>(prim->GetAttr(kLstmWeightOrderIofcFlag)));
  }
  if (prim->GetAttr(kSequenceLensOnlineFlag) != nullptr) {
    lstm_operator->SetSequenceLensOnlineFlag(api::GetValue<bool>(prim->GetAttr(kSequenceLensOnlineFlag)));
  }

  if (SetRecurrentDataInfo(cnode, lstm_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set lstm data info failed.";
    return RET_ERROR;
  }

  base_operators->push_back(std::move(lstm_operator));
  return RET_OK;
}
REG_MAPPER(Lstm, LstmMapper)
}  // namespace dpico
}  // namespace mindspore
