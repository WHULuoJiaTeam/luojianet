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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GETNEXT_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GETNEXT_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class GetNextInfo : public OperatorInfo {
 public:
  GetNextInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<GetNextCost>()) {}
  ~GetNextInfo() override = default;

  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status GetAttrs() override;
  Status InferTensorMap() override;
  Status InferTensorLayout(TensorLayouts *outputs_layout);
  Status InferTensorInfo() override;  // The GetNext() has not input, so need override the InferTensorInfo().
  Status InferDevMatrixShape() override;
  Status InferMirrorOps() override { return SUCCESS; }
  Status InferForwardCommunication() override { return SUCCESS; }
  void InferReplaceOps() override;
  Status GetAttrTypes();
  Status GetAttrShapes();
  Status GetAttrOutPutNum();
  Status InferAsLossDivisor() override { return SUCCESS; }

 private:
  int64_t dev_num_ = 1;
  std::vector<std::string> types_;
  Shapes shapes_;
  int64_t output_num_ = 0;
  int64_t shard_num_ = 1;
  std::string shared_name_;
  Strategys dataset_strategy_;
  Shape dev_matrix_shape_origin_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_GETNEXT_INFO_H_
