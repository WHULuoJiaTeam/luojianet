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

#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_CROP_JPEG_OP_H
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_CROP_JPEG_OP_H

#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "luojianet_ms/core/utils/log_adapter.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/device_resource.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "minddata/dataset/kernels/image/dvpp/utils/MDAclProcess.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ResourceManager.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace luojianet_ms {
namespace dataset {
class DvppCropJpegOp : public TensorOp {
 public:
  DvppCropJpegOp(int32_t crop_height, int32_t crop_width) : crop_height_(crop_height), crop_width_(crop_width) {}

  /// \brief Destructor
  ~DvppCropJpegOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kDvppCropJpegOp; }

  Status SetAscendResource(const std::shared_ptr<DeviceResource> &resource) override;

 private:
  uint32_t crop_height_;
  uint32_t crop_width_;

  std::shared_ptr<MDAclProcess> processor_;
};
}  // namespace dataset
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_CROP_JPEG_OP_H
