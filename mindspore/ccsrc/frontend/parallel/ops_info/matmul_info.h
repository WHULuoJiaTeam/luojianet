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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MATMUL_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MATMUL_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class MatMulBase : public OperatorInfo {
 public:
  MatMulBase(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<MatMulCost>()) {}
  ~MatMulBase() override = default;

  // Generate all strategies and the corresponding cost for this MatMul operator
  Status GenerateStrategies(int64_t stage_id) override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  Status PrepareStrategy(int64_t stage_id, size_t dev_num, Dimensions combined_partitions, size_t input0_shape_size,
                         size_t input1_shape_size, StrategyPtr *sp);

  Status SwapLastTwoElements(Shape *shape);

 protected:
  Status InferForwardCommunication() override;
  Status InferTensorInfo() override;  // the forward_reduce_scatter mode need to override this function
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout);
  void InitTensorInfoForCost(std::vector<TensorInfo> *);
  Status GenerateStrategiesBase(int64_t stage_id, size_t dev_num, const Shape &input0_shape, Shape input1_shape,
                                std::vector<StrategyPtr> *const sp_vector);
  Status GenerateStrategiesNotPower2(int64_t stage_id, size_t dev_num_not_2_power,
                                     const std::vector<StrategyPtr> &sp_vector_2_power_part);
  Status CheckForTensorSliceValid() const;
  Status GetAttrs() override;

  bool transpose_a_ = false;
  bool transpose_b_ = false;
  bool forward_reduce_scatter_ = false;
  int64_t field_size_ = 0;
  size_t mat_a_dimension_ = 0;
  size_t mat_b_dimension_ = 0;
  Shape origin_dev_matrix_shape_;
};

class MatMul : public MatMulBase {
 public:
  MatMul(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : MatMulBase(name, inputs_shape, outputs_shape, attrs) {}
  ~MatMul() override = default;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status CheckOutputStrategy(const StrategyPtr &out_strategy) override;
};

class MatMulInfo : public MatMul {
 public:
  MatMulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : MatMul(name, inputs_shape, outputs_shape, attrs) {}
  ~MatMulInfo() override = default;
};

class BatchMatMulInfo : public MatMul {
 public:
  BatchMatMulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : MatMul(name, inputs_shape, outputs_shape, attrs) {}
  ~BatchMatMulInfo() override = default;

  std::shared_ptr<Strategys> GenerateBatchStrategies() override;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_MATMUL_INFO_H_
