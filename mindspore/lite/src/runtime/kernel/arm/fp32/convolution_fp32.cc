/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/convolution_fp32.h"
#ifdef SERVER_INFERENCE
#include "src/pack_weight_manager.h"
#endif
#include "include/errorcode.h"
#include "nnacl/common_func.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/conv_common_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
#ifdef ENABLE_AVX
#define OC_BLOCK C16NUM
#elif defined(ENABLE_ARM32)
#define OC_BLOCK C4NUM
#else
#define OC_BLOCK C8NUM
#endif
int ConvolutionCPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]->MutableData());
#ifdef ENABLE_AVX
  int unit_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C6NUM * thread_count_;
#elif defined(ENABLE_SSE)
  int unit_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C4NUM * thread_count_;
#else
  int unit_size =
    conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C12NUM * thread_count_;
#endif
  packed_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed input failed.";
    return RET_ERROR;
  }

  col_major_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (col_major_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_major_input_ failed.";
    return RET_ERROR;
  }

#ifdef ENABLE_AVX
  if (conv_param_->output_channel_ % OC_BLOCK != 0 && out_tensors_[0]->format() == NC4HW4) {
    output_need_align_ = true;
    int oc_algin = UP_DIV(conv_param_->output_channel_, OC_BLOCK);
    int pack_output_size =
      conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * OC_BLOCK * oc_algin;
    tmp_output_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_output_size * sizeof(float)));
    if (tmp_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc tmp_output_ buffer is failed.";
      return RET_ERROR;
    }
  }
#endif
  return RET_OK;
}

int ConvolutionCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (op_parameter_->is_train_session_) {
    auto filter_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(filter_tensor);
    size_t in_channel = filter_tensor->Channel();
    size_t out_channel = filter_tensor->Batch();
    size_t oc_block_num = UP_ROUND(out_channel, OC_BLOCK);
    size_t kernel_plane = filter_tensor->Height() * filter_tensor->Width();
    size_t pack_weight_size = oc_block_num * in_channel * kernel_plane;
    set_workspace_size(pack_weight_size * sizeof(float));
  }
  auto ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }
  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv base init failed.";
    return ret;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::RunImpl(int task_id) {
  auto ori_input_data = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data());
  if (out_tensors_[0]->format() != NC4HW4) {
    ConvFp32(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
             reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
  } else {
#if defined(ENABLE_ARM64) || defined(ENABLE_AVX)
    ConvFp32OutNC4HW4(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                      reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
#else
    ConvFp32(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
             reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
#endif
  }
  return RET_OK;
}

int ConvolutionImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv = reinterpret_cast<ConvolutionCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  if (!output_need_align_) {
    tmp_output_ = output_addr;
  }
  if (RepackWeight() != RET_OK) {
    FreeTmpBuffer();
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }
  ret = ParallelLaunch(this->ms_context_, ConvolutionImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << ret << "]";
  }
#ifdef ENABLE_AVX
  if (output_need_align_) {
    PackNC8HW8AlignedToNC8HW8NotAlignedFp32(tmp_output_, output_addr, conv_param_->output_batch_,
                                            conv_param_->output_h_ * conv_param_->output_w_,
                                            conv_param_->output_channel_);
  }
#endif
  FreeTmpBuffer();
  return ret;
}

void ConvolutionCPUKernel::PackWeight() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int32_t in_channel = filter_tensor->Channel();
  if (in_channel < 0) {
    MS_LOG(ERROR) << "get channel from filter_tensor failed.";
    return;
  }
  int32_t out_channel = filter_tensor->Batch();
  if (out_channel < 0) {
    MS_LOG(ERROR) << "get batch from filter_tensor failed.";
    return;
  }
  int32_t kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  if (kernel_plane < 0) {
    MS_LOG(ERROR) << "get height and width from filter_tensor failed.";
    return;
  }
  void *origin_weight = (op_parameter_->is_train_session_) ? filter_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
#ifdef ENABLE_AVX
  RowMajor2Col16Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), out_channel,
                      in_channel * kernel_plane);
#elif defined(ENABLE_ARM32)
  RowMajor2Col4Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), out_channel,
                     in_channel * kernel_plane);
#else
  RowMajor2Col8Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), out_channel,
                     in_channel * kernel_plane);
#endif
}

int ConvolutionCPUKernel::MallocWeightBiasData() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int32_t in_channel = filter_tensor->Channel();
  int32_t out_channel = filter_tensor->Batch();
  MS_CHECK_TRUE_RET(in_channel > 0 && out_channel > 0, RET_ERROR);
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  size_t oc_block_num = UP_ROUND(out_channel, OC_BLOCK);
  size_t kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  size_t pack_weight_size = oc_block_num * in_channel * kernel_plane;
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
#ifdef SERVER_INFERENCE
    auto packed = lite::PackWeightManager::GetInstance()->GetPackedTensor(
      in_tensors_[1], static_cast<size_t>(pack_weight_size) * sizeof(float));
    packed_weight_ = packed.second;
    weight_is_packed_ = packed.first;
    if (weight_is_packed_ == lite::MALLOC && packed_weight_ == nullptr) {
      packed_weight_ = malloc(pack_weight_size * sizeof(float));
      memset(packed_weight_, 0, pack_weight_size * sizeof(float));
    }
#else
    packed_weight_ = malloc(pack_weight_size * sizeof(float));
#endif
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "malloc packed weight failed.";
      return RET_ERROR;
    }
#ifndef SERVER_INFERENCE
    memset(packed_weight_, 0, pack_weight_size * sizeof(float));
#endif
  }

  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, oc_block_num * sizeof(float));
    bias_data_ = malloc(oc_block_num * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, oc_block_num * sizeof(float));
  return RET_OK;
}
}  // namespace mindspore::kernel
