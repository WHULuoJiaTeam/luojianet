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

#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_CORE_DETENSOR_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_CORE_DETENSOR_H_
#include <string>
#include <vector>
#include <memory>
#include "include/api/status.h"
#include "include/api/types.h"
#if defined(ENABLE_ANDROID) || defined(ENABLE_LITE_TENSOR)
#include "luojianet_ms/lite/src/cxx_api/tensor/tensor_impl.h"
#else
#include "luojianet_ms/core/ir/api_tensor_impl.h"
#endif
#include "minddata/dataset/core/tensor.h"

namespace luojianet_ms {
namespace dataset {
class DETensor : public luojianet_ms::MSTensor::Impl {
 public:
  DETensor() = default;
  ~DETensor() = default;
  explicit DETensor(std::shared_ptr<dataset::Tensor> tensor_impl);
#ifndef ENABLE_ANDROID
  DETensor(std::shared_ptr<dataset::DeviceTensor> device_tensor_impl, bool is_device);
#endif
  const std::string &Name() const override;

  enum luojianet_ms::DataType DataType() const override;

  size_t DataSize() const override;

  const std::vector<int64_t> &Shape() const override;

  std::shared_ptr<const void> Data() const override;

  void *MutableData() override;

  bool IsDevice() const override;

  std::shared_ptr<luojianet_ms::MSTensor::Impl> Clone() const override;

 private:
  std::shared_ptr<dataset::Tensor> tensor_impl_;
#ifndef ENABLE_ANDROID
  std::shared_ptr<dataset::DeviceTensor> device_tensor_impl_;
#endif
  bool is_device_;
  std::string name_;
  enum luojianet_ms::DataType type_;
  std::vector<int64_t> shape_;
};
}  // namespace dataset
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_CORE_DETENSOR_H_
