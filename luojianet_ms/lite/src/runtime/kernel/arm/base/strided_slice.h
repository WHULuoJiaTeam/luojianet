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

#ifndef LUOJIANET_MS_LITE_SRC_BACKEND_ARM_BASE_STRIDED_SLICE_H_
#define LUOJIANET_MS_LITE_SRC_BACKEND_ARM_BASE_STRIDED_SLICE_H_

#include <vector>
#include "nnacl/fp32/strided_slice_fp32.h"
#include "src/inner_kernel.h"

namespace luojianet_ms::kernel {
class StridedSliceCPUKernel : public InnerKernel {
 public:
  StridedSliceCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                        const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<StridedSliceParameter *>(parameter);
  }
  ~StridedSliceCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  bool MatchFastPattern();
  void InitFastRunParam();
  int NormalRun();
  int FastRun();
  int FastRunImpl(int task_id);

 private:
  StridedSliceParameter *param_ = nullptr;
  uint8_t *input_ptr_ = nullptr;
  uint8_t *output_ptr_ = nullptr;
  int split_axis_{-1};
  int inner_{1};
  int outer_{1};
  int cal_num_per_thread_{1};
  size_t inner_size_{1};
  bool fast_run_{false};
  bool parallel_on_split_axis_{false};
  bool parallel_on_outer_{false};
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_BACKEND_ARM_BASE_STRIDED_SLICE_H_
