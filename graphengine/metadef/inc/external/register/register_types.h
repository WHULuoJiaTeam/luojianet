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

#ifndef INC_EXTERNAL_REGISTER_REGISTER_TYPES_H_
#define INC_EXTERNAL_REGISTER_REGISTER_TYPES_H_

#if(defined(HOST_VISIBILITY)) && (defined(__GNUC__))
#define FMK_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define FMK_FUNC_HOST_VISIBILITY
#endif
#if(defined(DEV_VISIBILITY)) && (defined(__GNUC__))
#define FMK_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define FMK_FUNC_DEV_VISIBILITY
#endif
#ifdef __GNUC__
#define ATTRIBUTED_DEPRECATED(replacement) __attribute__((deprecated("Please use " #replacement " instead.")))
#else
#define ATTRIBUTED_DEPRECATED(replacement) __declspec(deprecated("Please use " #replacement " instead."))
#endif
namespace domi {

///
/// @ingroup domi
/// @brief original tensor type
///
typedef enum tagDomiTensorFormat {
  DOMI_TENSOR_NCHW = 0,   // < NCHW
  DOMI_TENSOR_NHWC,       // < NHWC
  DOMI_TENSOR_ND,         // < Nd Tensor
  DOMI_TENSOR_NC1HWC0,    // < NC1HWC0
  DOMI_TENSOR_FRACTAL_Z,  // < FRACTAL_Z
  DOMI_TENSOR_NC1C0HWPAD,
  DOMI_TENSOR_NHWC1C0,
  DOMI_TENSOR_FSR_NCHW,
  DOMI_TENSOR_FRACTAL_DECONV,
  DOMI_TENSOR_BN_WEIGHT,
  DOMI_TENSOR_CHWN,         // Android NN Depth CONV
  DOMI_TENSOR_FILTER_HWCK,  // filter input tensor format
  DOMI_TENSOR_NDHWC,
  DOMI_TENSOR_NCDHW,
  DOMI_TENSOR_DHWCN,  // 3D filter input tensor format
  DOMI_TENSOR_DHWNC,
  DOMI_TENSOR_RESERVED
} domiTensorFormat_t;
}  // namespace domi

#endif  // INC_EXTERNAL_REGISTER_REGISTER_TYPES_H_
