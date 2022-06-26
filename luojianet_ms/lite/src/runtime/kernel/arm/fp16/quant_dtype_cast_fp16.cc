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
#include "src/runtime/kernel/arm/fp16/quant_dtype_cast_fp16.h"
#include <vector>
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include "nnacl/fp16/quant_dtype_cast_fp16.h"
#include "src/kernel_registry.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

using luojianet_ms::kernel::KERNEL_ARCH;
using luojianet_ms::lite::KernelRegistrar;
using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_OK;
using luojianet_ms::lite::RET_PARAM_INVALID;
using luojianet_ms::schema::PrimitiveType_QuantDTypeCast;

namespace luojianet_ms::kernel {
int QuantDTypeCastFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  CHECK_NULL_RETURN(in_tensor);
  CHECK_NULL_RETURN(out_tensor);
  auto param = reinterpret_cast<QuantDTypeCastParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  if (param->dstT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeFloat16 || out_tensor->data_type() != kNumberTypeInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
    int_to_float_ = false;
    is_uint8_ = false;
  } else if (param->srcT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeInt8 || out_tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
    int_to_float_ = true;
    is_uint8_ = false;
  } else if (param->dstT == kNumberTypeUInt8) {
    if (in_tensor->data_type() != kNumberTypeFloat16 || out_tensor->data_type() != kNumberTypeUInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
    int_to_float_ = false;
    is_uint8_ = true;
  } else if (param->srcT == kNumberTypeUInt8) {
    if (in_tensor->data_type() != kNumberTypeUInt8 || out_tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      return RET_ERROR;
    }
    int_to_float_ = true;
    is_uint8_ = true;
  } else {
    MS_LOG(ERROR) << "param data type not supported:"
                  << " src: " << param->srcT << " dst: " << param->dstT;
    return RET_PARAM_INVALID;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int QuantDTypeCastFp16CPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  num_unit_ = static_cast<int>(in_tensor->ElementsNum());
  thread_n_num_ = MSMIN(ms_context_->thread_num_, num_unit_);
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  return RET_OK;
}

int QuantDTypeCastFp16CPUKernel::QuantDTypeCast(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  if (in_tensors_.front()->quant_params().empty() && out_tensors_.front()->quant_params().empty()) {
    MS_LOG(ERROR) << "QuantDTypeCast need quantization parameters which is not found.";
    return RET_ERROR;
  }
  auto quant_arg = !out_tensors_.front()->quant_params().empty() ? out_tensors_.front()->quant_params().front()
                                                                 : in_tensors_.front()->quant_params().front();
  int ret;
  MS_ASSERT(float16_ptr_ != nullptr);
  if (!is_uint8_) {
    MS_ASSERT(int8_ptr_ != nullptr);
    if (int_to_float_) {
      ret = DoDequantizeInt8ToFp16(int8_ptr_ + thread_offset, float16_ptr_ + thread_offset, quant_arg.scale,
                                   quant_arg.zeroPoint, num_unit_thread);
    } else {
      ret = DoQuantizeFp16ToInt8(float16_ptr_ + thread_offset, int8_ptr_ + thread_offset, quant_arg.scale,
                                 quant_arg.zeroPoint, num_unit_thread);
    }
  } else {
    // uint8
    MS_ASSERT(uint8_ptr_ != nullptr);
    if (int_to_float_) {
      ret = DoDequantizeUInt8ToFp16(uint8_ptr_ + thread_offset, float16_ptr_ + thread_offset, quant_arg.scale,
                                    quant_arg.zeroPoint, num_unit_thread);
    } else {
      ret = DoQuantizeFp16ToUInt8(float16_ptr_ + thread_offset, uint8_ptr_ + thread_offset, quant_arg.scale,
                                  quant_arg.zeroPoint, num_unit_thread);
    }
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCastFp16 error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantDTypeCastFP16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<QuantDTypeCastFp16CPUKernel *>(cdata);
  auto ret = g_kernel->QuantDTypeCast(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCastFP16Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantDTypeCastFp16CPUKernel::Run() {
  if (in_tensors_.at(0)->data_type() == TypeId::kNumberTypeInt8 &&
      out_tensors_.at(0)->data_type() == TypeId::kNumberTypeFloat16) {
    int8_ptr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
    float16_ptr_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
    CHECK_NULL_RETURN(int8_ptr_);
    CHECK_NULL_RETURN(float16_ptr_);
  } else if (in_tensors_.at(0)->data_type() == TypeId::kNumberTypeFloat16 &&
             out_tensors_.at(0)->data_type() == TypeId::kNumberTypeInt8) {
    float16_ptr_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
    int8_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());
    CHECK_NULL_RETURN(float16_ptr_);
    CHECK_NULL_RETURN(int8_ptr_);
  } else if (in_tensors_.at(0)->data_type() == TypeId::kNumberTypeUInt8 &&
             out_tensors_.at(0)->data_type() == TypeId::kNumberTypeFloat16) {
    uint8_ptr_ = reinterpret_cast<uint8_t *>(in_tensors_.at(0)->data());
    float16_ptr_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
    CHECK_NULL_RETURN(uint8_ptr_);
    CHECK_NULL_RETURN(float16_ptr_);
  } else if (in_tensors_.at(0)->data_type() == TypeId::kNumberTypeFloat16 &&
             out_tensors_.at(0)->data_type() == TypeId::kNumberTypeUInt8) {
    float16_ptr_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
    uint8_ptr_ = reinterpret_cast<uint8_t *>(out_tensors_.at(0)->data());
    CHECK_NULL_RETURN(float16_ptr_);
    CHECK_NULL_RETURN(uint8_ptr_);
  } else {
    MS_LOG(ERROR) << "QuantDTypeCastFp16 not support input or output type";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(this->ms_context_, QuantDTypeCastFP16Run, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::InnerKernel *CpuQuantDTypeCastFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                        const std::vector<lite::Tensor *> &outputs,
                                                        OpParameter *opParameter, const lite::Context *ctx,
                                                        const kernel::KernelKey &desc) {
  auto in_tensor = inputs.front();
  auto out_tensor = outputs.front();
  auto param = reinterpret_cast<QuantDTypeCastParameter *>(opParameter);
  if (param->dstT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeFloat16 || out_tensor->data_type() != kNumberTypeInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      free(opParameter);
      return nullptr;
    }
  } else if (param->srcT == kNumberTypeInt8) {
    if (in_tensor->data_type() != kNumberTypeInt8 || out_tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      free(opParameter);
      return nullptr;
    }
  } else if (param->dstT == kNumberTypeUInt8) {
    if (in_tensor->data_type() != kNumberTypeFloat16 || out_tensor->data_type() != kNumberTypeUInt8) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      free(opParameter);
      return nullptr;
    }
  } else if (param->srcT == kNumberTypeUInt8) {
    if (in_tensor->data_type() != kNumberTypeUInt8 || out_tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << "param data type and tensor data type do not match.";
      free(opParameter);
      return nullptr;
    }
  } else {
    MS_LOG(ERROR) << "param data type not supported:"
                  << " src: " << param->srcT << " dst: " << param->dstT;
    free(opParameter);
    return nullptr;
  }

  kernel::InnerKernel *kernel = new (std::nothrow)
    QuantDTypeCastFp16CPUKernel(opParameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new QuantDTypeCastFp16CPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_QuantDTypeCast, CpuQuantDTypeCastFp16KernelCreator)
}  // namespace luojianet_ms::kernel
