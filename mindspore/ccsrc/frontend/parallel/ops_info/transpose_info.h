/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TRANSPOSE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TRANSPOSE_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
/*
 * parallel class for Transpose Primitive
 */
class TransposeInfo : public OperatorInfo {
 public:
  TransposeInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<TransposeCost>()) {}
  ~TransposeInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;

 private:
  Status ComputeAxis();
  std::vector<int64_t> axis_v_;
  Dimensions input_strategy_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TRANSPOSE_INFO_H_
