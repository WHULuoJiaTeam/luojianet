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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/conv_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateDeconvParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto deconv2d_prim = primitive->value_as_DeConv2D();
  if (deconv2d_prim == nullptr) {
    MS_LOG(ERROR) << "deconv2d_prim is nullptr";
    return nullptr;
  }
  auto *conv_param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(conv_param, 0, sizeof(ConvParameter));
  conv_param->op_parameter_.type_ = schema::PrimitiveType_Conv2dTransposeFusion;
  conv_param->group_ = 1;

  conv_param->kernel_h_ = deconv2d_prim->kernelH();
  conv_param->kernel_w_ = deconv2d_prim->kernelW();
  conv_param->stride_h_ = deconv2d_prim->strideH();
  conv_param->stride_w_ = deconv2d_prim->strideW();

  conv_param->pad_u_ = deconv2d_prim->padUp();
  conv_param->pad_d_ = deconv2d_prim->padDown();
  conv_param->pad_l_ = deconv2d_prim->padLeft();
  conv_param->pad_r_ = deconv2d_prim->padRight();
  conv_param->dilation_h_ = deconv2d_prim->dilateH();
  conv_param->dilation_w_ = deconv2d_prim->dilateW();
  auto act_type = deconv2d_prim->activationType();
  switch (act_type) {
    case schema::v0::ActivationType_RELU:
      conv_param->act_type_ = ActType_Relu;
      break;
    case schema::v0::ActivationType_RELU6:
      conv_param->act_type_ = ActType_Relu6;
      break;
    default:
      conv_param->act_type_ = ActType_No;
      break;
  }
  auto pad_mode = deconv2d_prim->padMode();
  switch (pad_mode) {
    case schema::v0::PadMode_SAME_UPPER:
      conv_param->pad_mode_ = Pad_same;
      break;
    case schema::v0::PadMode_VALID:
      conv_param->pad_mode_ = Pad_valid;
      break;
    default:
      conv_param->pad_mode_ = Pad_pad;
      break;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}
}  // namespace

Registry g_deConv2DV0ParameterRegistry(schema::v0::PrimitiveType_DeConv2D, PopulateDeconvParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
