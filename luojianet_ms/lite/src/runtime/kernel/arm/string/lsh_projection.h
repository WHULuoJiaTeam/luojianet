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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_STRING_LSH_PROJECTION_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_STRING_LSH_PROJECTION_H_

#include <vector>

#include "nnacl/lsh_projection_parameter.h"
#include "src/inner_kernel.h"

namespace luojianet_ms::kernel {
class LshProjectionCPUKernel : public InnerKernel {
 public:
  LshProjectionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<LshProjectionParameter *>(parameter);
  }
  ~LshProjectionCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  int MallocKeys();
  void FreeKeys();
  int GetSignBit(const int32_t *feature, const float *weight, float seed, const LshProjectionParameter *para,
                 char *hash_buff);
  void LshProjectionSparse(const float *hashSeed, const int32_t *feature, const float *weight, int32_t *output,
                           const LshProjectionParameter *param, int32_t start, int32_t end, char *hash_buff);
  void LshProjectionDense(const float *hashSeed, const int32_t *feature, const float *weight, int32_t *output,
                          const LshProjectionParameter *param, int32_t start, int32_t end, char *hash_buff);
  LshProjectionParameter *param_ = nullptr;
  float *hash_seed_ = nullptr;
  int32_t *feature_ = nullptr;
  float *weight_ = nullptr;
  int32_t *output_ = nullptr;
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_STRING_LSH_PROJECTION_H_
