/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_BATCH_TO_SPACE_H_
#define MINDSPORE_CORE_OPS_BATCH_TO_SPACE_H_

#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchToSpace = "BatchToSpace";
/// \brief Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.
/// Refer to Python API @ref mindspore.ops.BatchToSpace for more details.
class MIND_API BatchToSpace : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BatchToSpace);
  /// \brief Constructor.
  BatchToSpace() : BaseOperator(kNameBatchToSpace) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.BatchToSpace for the inputs.
  void Init(const std::vector<int64_t> &block_size, const std::vector<std::vector<int64_t>> &crops);
  /// \brief Set block_size.
  void set_block_size(const std::vector<int64_t> &block_size);
  /// \brief Set crops.
  void set_crops(const std::vector<std::vector<int64_t>> &crops);
  /// \brief Get block_size.
  ///
  /// \return block_size.
  std::vector<int64_t> get_block_size() const;
  /// \brief Get crops.
  ///
  /// \return crops.
  std::vector<std::vector<int64_t>> get_crops() const;
};

abstract::AbstractBasePtr BatchToSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_TO_SPACE_H_
