/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_ASCEND_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_ASCEND_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"

namespace mindspore {
namespace dataset {
// Transform operations for performing computer vision.
namespace vision {
/* ##################################### API class ###########################################*/

/// \brief Decode and resize JPEG image using the hardware algorithm of
///     Ascend series chip DVPP module.
class MS_API DvppDecodeResizeJpeg final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] resize Parameter vector of two integers for each dimension, with respect to H,W order.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto dvpp_op = vision::DvppDecodeResizeJpeg({255, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({dvpp_op},   // operations
  ///                            {"image"});  // input columns
  /// \endcode
  explicit DvppDecodeResizeJpeg(const std::vector<uint32_t> &resize);

  /// \brief Destructor.
  ~DvppDecodeResizeJpeg() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode, resize and crop JPEG image using the hardware algorithm of
///     Ascend series chip DVPP module.
class MS_API DvppDecodeResizeCropJpeg final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] crop Parameter vector of two integers for each dimension after final crop, with respect to H,W order.
  /// \param[in] resize Parameter vector of two integers for each dimension after resize, with respect to H,W order.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto dvpp_op = vision::DvppDecodeResizeCropJpeg({50, 50}, {100, 100});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({dvpp_op},   // operations
  ///                            {"image"});  // input columns
  /// \endcode
  DvppDecodeResizeCropJpeg(const std::vector<uint32_t> &crop, const std::vector<uint32_t> &resize);

  /// \brief Destructor.
  ~DvppDecodeResizeCropJpeg() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode PNG image using the hardware algorithm of
///     Ascend series chip DVPP module.
class MS_API DvppDecodePng final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto dvpp_op = vision::DvppDecodePng();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({dvpp_op},   // operations
  ///                            {"image"});  // input columns
  /// \endcode
  DvppDecodePng();

  /// \brief Destructor.
  ~DvppDecodePng() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_ASCEND_H_
