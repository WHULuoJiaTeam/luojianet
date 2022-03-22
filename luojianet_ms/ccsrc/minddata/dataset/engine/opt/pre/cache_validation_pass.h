/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_CACHE_VALIDATION_PASS_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_CACHE_VALIDATION_PASS_

#include <memory>
#include <stack>
#include <utility>
#include "minddata/dataset/engine/opt/pass.h"

namespace luojianet_ms {
namespace dataset {

/// \class CacheValidationPass cache_validation_pass.h
/// \brief This is a NodePass who's job is to catch invalid tree configurations related to cache and generate failures.
class CacheValidationPass : public IRNodePass {
 public:
  /// \brief Constructor
  CacheValidationPass();

  /// \brief Destructor
  ~CacheValidationPass() = default;

  /// \brief Returns an error if BatchNode exists under a cache
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<BatchNode> node, bool *const modified) override;

  /// \brief Returns an error if ConcatNode exists under a cache
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<ConcatNode> node, bool *const modified) override;

  /// \brief Returns an error if FilterNode exists under a cache
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<FilterNode> node, bool *const modified) override;

  /// \brief Returns an error if SkipNode exists under a cache
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<SkipNode> node, bool *const modified) override;

  /// \brief Returns an error if TakeNode exists under a cache
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<TakeNode> node, bool *const modified) override;

  /// \brief Returns an error if ZipNode exists under a cache
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<ZipNode> node, bool *const modified) override;

  /// \brief Returns an error if MapNode with non-deterministic tensor operations exists under a cache
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<MapNode> node, bool *const modified) override;

  /// \brief Returns an error if there is a cache over another cache
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) override;

  /// \brief Identifies and block repeat under cache scenarios
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Identifies the subtree above this node as not being cached
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<TFRecordNode> node, bool *const modified) override;

  /// \brief Identifies the subtree above this node as not being cached
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) override;

 private:
  bool is_cached_;
  bool is_mappable_;
};
}  // namespace dataset
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_CACHE_VALIDATION_PASS_
