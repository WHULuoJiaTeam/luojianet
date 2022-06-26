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

#include "frontend/parallel/ops_info/virtual_output_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/step_parallel.h"
#include "include/common/utils/parallel_context.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status VirtualOutputInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategys stra = strategy->GetInputDim();
  if (stra.size() != 1) {
    MS_LOG(ERROR) << name_ << ": Strategys size must be  1.";
    return FAILED;
  }
  Dimensions strategy_first = stra.at(0);
  for (auto dim = strategy_first.begin() + 1; dim != strategy_first.end(); ++dim) {
    if (*dim != 1) {
      MS_LOG(ERROR) << name_ << ": All dimension except the first dimension of the strategy must be 1.";
      return FAILED;
    }
  }
  if (!strategy_first.empty()) {
    shard_num_ = strategy_first[0];
  }
  return SUCCESS;
}

Status VirtualOutputInfo::GenerateStrategies(int64_t stage_id) {
  StrategyPtr sp;
  Strategys strategy;
  bool full_batch = ParallelContext::GetInstance()->full_batch();
  size_t total_dev_num;
  if (full_batch) {
    total_dev_num = 1;
  } else {
    total_dev_num = LongToSize(stage_device_size_);
  }

  if (total_dev_num == 0) {
    MS_LOG(ERROR) << name_ << ": The total devices num is 0";
    return FAILED;
  }

  for (auto &shape : inputs_shape_) {
    Shape temp;
    if (!shape.empty()) {
      if (LongToSize(shape[0]) % total_dev_num == 0) {
        (void)temp.emplace_back(SizeToLong(total_dev_num));
      } else {
        (void)temp.emplace_back(1);
      }
      (void)temp.insert(temp.end(), shape.size() - 1, 1);
    }
    strategy.push_back(temp);
  }
  sp = std::make_shared<Strategy>(stage_id, strategy);
  if (SetCostUnderStrategy(sp) == SUCCESS) {
    MS_LOG(INFO) << name_ << ": Successfully dataset strategy.";
    PrintStrategy(sp);
  } else {
    MS_LOG(ERROR) << name_ << ": Generating dataset strategy failed.";
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
