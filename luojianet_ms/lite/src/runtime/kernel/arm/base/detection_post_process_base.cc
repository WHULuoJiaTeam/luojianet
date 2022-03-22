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
#include "src/runtime/kernel/arm/base/detection_post_process_base.h"
#include <cfloat>
#include <cmath>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"

using luojianet_ms::kernel::KERNEL_ARCH;
using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_NULL_PTR;
using luojianet_ms::lite::RET_OK;
using luojianet_ms::schema::PrimitiveType_DetectionPostProcess;

namespace luojianet_ms::kernel {
void PartialArgSort(const float *scores, int *indexes, int num_to_sort, int num_values) {
  std::partial_sort(indexes, indexes + num_to_sort, indexes + num_values, [&scores](const int i, const int j) {
    if (std::abs(scores[i] - scores[j]) < FLT_EPSILON) {
      return i < j;
    }
    return scores[i] > scores[j];
  });
}

int DetectionPostProcessBaseCPUKernel::Prepare() {
  params_->decoded_boxes_ = nullptr;
  params_->nms_candidate_ = nullptr;
  params_->indexes_ = nullptr;
  params_->scores_ = nullptr;
  params_->all_class_indexes_ = nullptr;
  params_->all_class_scores_ = nullptr;
  params_->single_class_indexes_ = nullptr;
  params_->selected_ = nullptr;
  params_->anchors_ = nullptr;
  auto anchor_tensor = in_tensors_.at(2);
  MS_CHECK_GT(anchor_tensor->ElementsNum(), 0, RET_ERROR);
  CHECK_NULL_RETURN(anchor_tensor->data());
  if (anchor_tensor->data_type() == kNumberTypeFloat32 || anchor_tensor->data_type() == kNumberTypeFloat) {
    params_->anchors_ = new (std::nothrow) float[anchor_tensor->ElementsNum()];
    if (params_->anchors_ == nullptr) {
      MS_LOG(ERROR) << "Malloc anchor failed";
      return RET_ERROR;
    }
    MS_CHECK_FALSE(anchor_tensor->Size() == 0, RET_ERROR);
    memcpy(params_->anchors_, anchor_tensor->data(), anchor_tensor->Size());
#ifndef OP_INT8_CLIP
  } else if (anchor_tensor->data_type() == kNumberTypeInt8) {
    auto quant_param = anchor_tensor->quant_params().front();
    auto anchor_int8 = reinterpret_cast<int8_t *>(anchor_tensor->data());
    auto anchor_fp32 = new (std::nothrow) float[anchor_tensor->ElementsNum()];
    if (anchor_fp32 == nullptr) {
      MS_LOG(ERROR) << "Malloc anchor failed";
      return RET_ERROR;
    }
    DoDequantizeInt8ToFp32(anchor_int8, anchor_fp32, quant_param.scale, quant_param.zeroPoint,
                           anchor_tensor->ElementsNum());
    params_->anchors_ = anchor_fp32;
  } else if (anchor_tensor->data_type() == kNumberTypeUInt8) {
    auto quant_param = anchor_tensor->quant_params().front();
    auto anchor_uint8 = reinterpret_cast<uint8_t *>(anchor_tensor->data());
    auto anchor_fp32 = new (std::nothrow) float[anchor_tensor->ElementsNum()];
    if (anchor_fp32 == nullptr) {
      MS_LOG(ERROR) << "Malloc anchor failed";
      return RET_ERROR;
    }
    DoDequantizeUInt8ToFp32(anchor_uint8, anchor_fp32, quant_param.scale, quant_param.zeroPoint,
                            anchor_tensor->ElementsNum());
    params_->anchors_ = anchor_fp32;
#endif
  } else {
    MS_LOG(ERROR) << "unsupported anchor data type " << anchor_tensor->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

DetectionPostProcessBaseCPUKernel::~DetectionPostProcessBaseCPUKernel() { delete[](params_->anchors_); }

int DetectionPostProcessBaseCPUKernel::ReSize() { return RET_OK; }

int NmsMultiClassesFastCoreRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto KernelData = reinterpret_cast<DetectionPostProcessBaseCPUKernel *>(cdata);
  int ret = NmsMultiClassesFastCore(KernelData->num_boxes_, KernelData->num_classes_with_bg_, KernelData->input_scores_,
                                    PartialArgSort, KernelData->params_, task_id, KernelData->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NmsMultiClassesFastCore error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void DetectionPostProcessBaseCPUKernel::FreeAllocatedBuffer() {
  if (params_->decoded_boxes_ != nullptr) {
    ms_context_->allocator->Free(params_->decoded_boxes_);
    params_->decoded_boxes_ = nullptr;
  }
  if (params_->nms_candidate_ != nullptr) {
    ms_context_->allocator->Free(params_->nms_candidate_);
    params_->nms_candidate_ = nullptr;
  }
  if (params_->indexes_ != nullptr) {
    ms_context_->allocator->Free(params_->indexes_);
    params_->indexes_ = nullptr;
  }
  if (params_->scores_ != nullptr) {
    ms_context_->allocator->Free(params_->scores_);
    params_->scores_ = nullptr;
  }
  if (params_->all_class_indexes_ != nullptr) {
    ms_context_->allocator->Free(params_->all_class_indexes_);
    params_->all_class_indexes_ = nullptr;
  }
  if (params_->all_class_scores_ != nullptr) {
    ms_context_->allocator->Free(params_->all_class_scores_);
    params_->all_class_scores_ = nullptr;
  }
  if (params_->single_class_indexes_ != nullptr) {
    ms_context_->allocator->Free(params_->single_class_indexes_);
    params_->single_class_indexes_ = nullptr;
  }
  if (params_->selected_ != nullptr) {
    ms_context_->allocator->Free(params_->selected_);
    params_->selected_ = nullptr;
  }
}

int DetectionPostProcessBaseCPUKernel::ParamInit() {
  num_boxes_ = in_tensors_.at(0)->shape().at(1);
  num_classes_with_bg_ = in_tensors_.at(1)->shape().at(2);
  params_->decoded_boxes_ = ms_context_->allocator->Malloc(num_boxes_ * DIMENSION_4D * sizeof(float));
  if (params_->decoded_boxes_ == nullptr) {
    MS_LOG(ERROR) << "malloc params->decoded_boxes_ failed.";
    FreeAllocatedBuffer();
    return RET_ERROR;
  }
  params_->nms_candidate_ = ms_context_->allocator->Malloc(num_boxes_ * sizeof(uint8_t));
  if (params_->nms_candidate_ == nullptr) {
    MS_LOG(ERROR) << "malloc params->nms_candidate_ failed.";
    FreeAllocatedBuffer();
    return RET_ERROR;
  }
  params_->selected_ = ms_context_->allocator->Malloc(num_boxes_ * sizeof(int));
  if (params_->selected_ == nullptr) {
    MS_LOG(ERROR) << "malloc params->selected_ failed.";
    FreeAllocatedBuffer();
    return RET_ERROR;
  }
  params_->single_class_indexes_ = ms_context_->allocator->Malloc(num_boxes_ * sizeof(int));
  if (params_->single_class_indexes_ == nullptr) {
    MS_LOG(ERROR) << "malloc params->single_class_indexes_ failed.";
    FreeAllocatedBuffer();
    return RET_ERROR;
  }

  if (params_->use_regular_nms_) {
    params_->scores_ = ms_context_->allocator->Malloc((num_boxes_ + params_->max_detections_) * sizeof(float));
    if (params_->scores_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->scores_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
    params_->indexes_ = ms_context_->allocator->Malloc((num_boxes_ + params_->max_detections_) * sizeof(int));
    if (params_->indexes_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->indexes_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
    params_->all_class_scores_ =
      ms_context_->allocator->Malloc((num_boxes_ + params_->max_detections_) * sizeof(float));
    if (params_->all_class_scores_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->all_class_scores_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
    params_->all_class_indexes_ = ms_context_->allocator->Malloc((num_boxes_ + params_->max_detections_) * sizeof(int));
    if (params_->all_class_indexes_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->all_class_indexes_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
  } else {
    params_->scores_ = ms_context_->allocator->Malloc(num_boxes_ * sizeof(float));
    if (params_->scores_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->scores_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
    params_->indexes_ = ms_context_->allocator->Malloc(num_boxes_ * params_->num_classes_ * sizeof(int));
    if (!params_->indexes_) {
      MS_LOG(ERROR) << "malloc params->indexes_ failed.";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int DetectionPostProcessBaseCPUKernel::Run() {
  MS_ASSERT(ms_context_->allocator != nullptr);
  int status = GetInputData();
  if (status != RET_OK) {
    return status;
  }
  auto output_boxes = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  auto output_classes = reinterpret_cast<float *>(out_tensors_.at(1)->data());
  auto output_scores = reinterpret_cast<float *>(out_tensors_.at(2)->data());
  auto output_num = reinterpret_cast<float *>(out_tensors_.at(3)->data());
  if (output_boxes == nullptr || output_classes == nullptr || output_scores == nullptr || output_num == nullptr) {
    return RET_NULL_PTR;
  }

  if (ParamInit() != RET_OK) {
    MS_LOG(ERROR) << "ParamInit error";
    return status;
  }

  status = DecodeBoxes(num_boxes_, input_boxes_, params_->anchors_, params_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DecodeBoxes error";
    FreeAllocatedBuffer();
    return status;
  }

  if (params_->use_regular_nms_) {
    status = DetectionPostProcessRegular(num_boxes_, num_classes_with_bg_, input_scores_, output_boxes, output_classes,
                                         output_scores, output_num, PartialArgSort, params_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DetectionPostProcessRegular error error_code[" << status << "]";
      FreeAllocatedBuffer();
      return status;
    }
  } else {
    status = ParallelLaunch(this->ms_context_, NmsMultiClassesFastCoreRun, this, op_parameter_->thread_num_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "NmsMultiClassesFastCoreRun error error_code[" << status << "]";
      FreeAllocatedBuffer();
      return status;
    }
    status = DetectionPostProcessFast(num_boxes_, num_classes_with_bg_, input_scores_,
                                      reinterpret_cast<float *>(params_->decoded_boxes_), output_boxes, output_classes,
                                      output_scores, output_num, PartialArgSort, params_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DetectionPostProcessFast error error_code[" << status << "]";
      FreeAllocatedBuffer();
      return status;
    }
  }
  FreeAllocatedBuffer();
  return RET_OK;
}
}  // namespace luojianet_ms::kernel
