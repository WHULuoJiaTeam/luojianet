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

#ifndef LUOJIANET_MS_LITE_TOOLS_BENCHMARK_NNIE_PROPOSAL_PROPOSAL_FP32_H_
#define LUOJIANET_MS_LITE_TOOLS_BENCHMARK_NNIE_PROPOSAL_PROPOSAL_FP32_H_

#include <vector>
#include "schema/model_generated.h"
#include "include/context.h"
#include "include/api/kernel.h"
#include "src/proposal.h"

using luojianet_ms::kernel::Kernel;
namespace luojianet_ms {
namespace proposal {
class ProposalCPUKernel : public Kernel {
 public:
  ProposalCPUKernel(const std::vector<luojianet_ms::MSTensor> &inputs, const std::vector<luojianet_ms::MSTensor> &outputs,
                    const luojianet_ms::schema::Primitive *primitive, const luojianet_ms::Context *ctx, int id,
                    int image_height, int image_width)
      : Kernel(inputs, outputs, primitive, ctx), id_(id), image_height_(image_height), image_weight_(image_width) {}

  ~ProposalCPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Execute() override;

 private:
  proposal::ProposalParam proposal_param_ = {0};
  int64_t id_;
  int64_t image_height_;
  int64_t image_weight_;
};
}  // namespace proposal
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LITE_TOOLS_BENCHMARK_NNIE_PROPOSAL_PROPOSAL_FP32_H_
