/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_ARGMIN_FUSION_H_
#define MINDSPORE_CORE_OPS_ARGMIN_FUSION_H_
#include <vector>
#include <memory>

#include "ops/arg_min.h"

namespace mindspore {
namespace ops {
constexpr auto kNameArgMinFusion = "ArgMinFusion";
/// \brief ArgMinFusion defined ArgMin operator prototype of lite.
class MIND_API ArgMinFusion : public ArgMin {
 public:
  MIND_API_BASE_MEMBER(ArgMinFusion);
  /// \brief Constructor.
  ArgMinFusion() : ArgMin(kNameArgMinFusion) { InitIOName({"x"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] keep_dims Define a boolean value to indicate the dimension of output is equal to that of input or not.
  /// \param[in] out_max_value Define a boolean value to indicate whether to output minimum value.
  /// \param[in] top_k Define the number of minimum value along with axis.
  /// \param[in] axis Define where the argmin operation applies to.
  void Init(bool keep_dims, bool out_max_value, int64_t top_k, int64_t axis = -1);

  /// \brief Method to set keep_dims attribute.
  ///
  /// \param[in] keep_dims Define a boolean value to indicate the dimension of output is equal to that of input or not.
  void set_keep_dims(const bool keep_dims);

  /// \brief Method to set out_max_value attribute.
  ///
  /// \param[in] out_max_value Define a boolean value to indicate whether to output minimum value.
  void set_out_max_value(bool out_max_value);

  /// \brief Method to set top_k attribute.
  ///
  /// \param[in] top_k Define the number of minimum value along with axis.
  void set_top_k(int64_t top_k);

  /// \brief Method to get keep_dims attribute.
  ///
  /// \return a boolean value to indicate the dimension of output is equal to that of input or not.
  bool get_keep_dims() const;

  /// \brief Method to get out_max_value attribute.
  ///
  /// \return a boolean value to indicate whether to output minimum value.
  bool get_out_max_value() const;

  /// \brief Method to get top_k attribute.
  ///
  /// \return the number of minimum value along with axis.
  int64_t get_top_k() const;
};
abstract::AbstractBasePtr ArgMinFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimArgMinFusion = std::shared_ptr<ArgMinFusion>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ARGMINTOPKMAXVALUE_H_
