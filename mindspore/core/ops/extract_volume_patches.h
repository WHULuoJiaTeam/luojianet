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

#ifndef MINDSPORE_CORE_OPS_EXTRACT_VOLUME_PATCHES_H_
#define MINDSPORE_CORE_OPS_EXTRACT_VOLUME_PATCHES_H_
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <string>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameExtractVolumePatches = "ExtractVolumePatches";
/// \brief Extract patches from input and put them in the "depth" output dimension.
/// Refer to Python API @ref mindspore.ops.ExtractVolumePatches for more details.
class MIND_API ExtractVolumePatches : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ExtractVolumePatches);
  /// \brief Constructor.
  ExtractVolumePatches() : BaseOperator(kNameExtractVolumePatches) { InitIOName({"x"}, {"y"}); }
};

abstract::AbstractBasePtr ExtractVolumePatchesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimExtractVolumePatchesPtr = std::shared_ptr<ExtractVolumePatches>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EXTRACT_VOLUME_PATCHES_H_
