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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ASCEND310_KERNEL_CUSTOM_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ASCEND310_KERNEL_CUSTOM_H_

#include <vector>
#include <string>
#include <memory>
#include "src/runtime/kernel/ascend/src/acl_model_options.h"
#include "src/runtime/kernel/ascend/src/model_infer.h"
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/kernel.h"
#include "include/errorcode.h"

namespace luojianet_ms::kernel {
namespace acl {
using luojianet_ms::lite::STATUS;

class CustomAscendKernel : public kernel::Kernel {
 public:
  CustomAscendKernel(const std::vector<luojianet_ms::MSTensor> &inputs, const std::vector<luojianet_ms::MSTensor> &outputs,
                     const luojianet_ms::schema::Primitive *primitive, const luojianet_ms::Context *ctx);
  ~CustomAscendKernel() override;

  STATUS Prepare() override;
  STATUS ReSize() override;
  STATUS Execute() override;

 private:
  void RecordInputDataIndex();
  STATUS PrepareModelInfer();
  STATUS ProcDynamicInput(std::vector<luojianet_ms::MSTensor> *input);
  STATUS GetRealBatchSize(std::vector<luojianet_ms::MSTensor> *inputs, int32_t *batch_size);
  STATUS GetRealImageSize(std::vector<luojianet_ms::MSTensor> *inputs, int32_t *image_size, int32_t num);

  bool load_model_;
  AclModelOptions acl_options_;
  std::shared_ptr<ModelInfer> model_infer_;
  size_t InputDataIndex_;
};
}  // namespace acl
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ASCEND310_KERNEL_CUSTOM_H_
