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

#include "src/common/tensor_util.h"
#include <algorithm>
#include <unordered_map>
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#ifdef ENABLE_FP16
#include "src/runtime/kernel/arm/fp16/fp16_op_handler.h"
#endif

namespace luojianet_ms {
namespace lite {
int OutputTensor2TensorC(const std::vector<lite::Tensor *> &tensors, std::vector<TensorC *> *tensors_c) {
  MS_ASSERT(tensors_c != nullptr);
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto *tensor_c = static_cast<TensorC *>(malloc(sizeof(TensorC)));
    if (tensor_c == nullptr) {
      MS_LOG(ERROR) << "malloc tensor fail!";
      return RET_ERROR;
    }
    tensor_c->data_type_ = kNumberTypeFloat32;
    tensor_c->format_ = tensors[i]->format();
    tensor_c->data_ = nullptr;
    tensor_c->shape_size_ = 0;
    tensors_c->push_back(tensor_c);
  }
  return RET_OK;
}

void FreeAllTensorC(std::vector<TensorC *> *tensors_in) {
  if (tensors_in == nullptr) {
    return;
  }
  for (auto &i : *tensors_in) {
    if (i == nullptr) {
      continue;
    }
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    if (i->data_type_ == kObjectTypeTensorType) {
      TensorListC *tensorListC = reinterpret_cast<TensorListC *>(i);
      FreeTensorListC(tensorListC);
      tensorListC = nullptr;
    } else {
#endif
      free(i);
      i = nullptr;
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    }
#endif
  }
  tensors_in->clear();
}

int Tensor2TensorC(const Tensor *src, TensorC *dst) {
  MS_CHECK_TRUE_RET(src != nullptr && dst != nullptr, RET_ERROR);
  dst->is_ready_ = src->IsReady();
  dst->format_ = static_cast<int>(src->format());
  dst->data_ = src->data();
  dst->data_type_ = src->data_type();
  dst->shape_size_ = src->shape().size();
  if (dst->shape_size_ > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "tensor shape size " << dst->shape_size_ << " is larger than max shape size " << MAX_SHAPE_SIZE;
    return RET_ERROR;
  }
  for (size_t i = 0; i < dst->shape_size_; i++) {
    dst->shape_[i] = src->shape().at(i);
  }
  return RET_OK;
}

int TensorC2Tensor(const TensorC *src, Tensor *dst) {
  MS_CHECK_TRUE_RET(src != nullptr && dst != nullptr, RET_ERROR);
  dst->set_format(static_cast<luojianet_ms::Format>(src->format_));
  dst->set_data_type(static_cast<TypeId>(src->data_type_));  // get data during the runtime period
  dst->set_shape(std::vector<int>(src->shape_, src->shape_ + src->shape_size_));
  if (src->data_ != nullptr) {
    auto data = dst->MutableData();
    MS_CHECK_TRUE_RET(data != nullptr, RET_ERROR);
    memcpy(data, src->data_, dst->Size());
    dst->set_category(CONST_TENSOR);
    free(src->data_);
  }
  return RET_OK;
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP
void FreeTensorListC(TensorListC *tensorlist_c) {
  MS_ASSERT(tensorlist_c != nullptr);
  if (tensorlist_c->tensors_ != nullptr) {
    free(tensorlist_c->tensors_);
    tensorlist_c->tensors_ = nullptr;
  }
  free(tensorlist_c);
}

int TensorList2TensorListC(TensorList *src, TensorListC *dst) {
  MS_CHECK_TRUE_RET(src != nullptr && dst != nullptr, RET_ERROR);
  dst->is_ready_ = src->IsReady();
  dst->data_type_ = static_cast<TypeIdC>(src->data_type());
  dst->format_ = static_cast<int>(src->format());
  dst->shape_value_ = src->shape().empty() ? 0 : src->shape().front();
  dst->element_num_ = src->shape().empty() ? 0 : src->tensors().size();

  if ((dst->element_num_ != 0 && SIZE_MAX / dst->element_num_ < sizeof(TensorC)) ||
      dst->element_num_ * sizeof(TensorC) > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "data size error.";
    return RET_ERROR;
  }
  dst->tensors_ = reinterpret_cast<TensorC *>(malloc(dst->element_num_ * sizeof(TensorC)));
  if (dst->tensors_ == nullptr) {
    return RET_ERROR;
  }
  memset(dst->tensors_, 0, dst->element_num_ * sizeof(TensorC));
  for (size_t i = 0; i < dst->element_num_; i++) {
    auto ret = Tensor2TensorC(src->tensors().at(i), &dst->tensors_[i]);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Tensor to TensorC failed.";
      return ret;
    }
  }

  dst->tensors_data_type_ = src->tensors_data_type();
  dst->element_shape_size_ = src->element_shape().size();
  for (size_t i = 0; i < dst->element_shape_size_; i++) {
    dst->element_shape_[i] = src->element_shape().at(i);
  }
  dst->max_elements_num_ = src->max_elements_num();
  return NNACL_OK;
}

int TensorListC2TensorList(const TensorListC *src, TensorList *dst) {
  MS_CHECK_TRUE_RET(src != nullptr && dst != nullptr, RET_ERROR);
  dst->set_data_type(static_cast<TypeId>(src->data_type_));
  dst->set_format(static_cast<luojianet_ms::Format>(src->format_));
  dst->set_shape(std::vector<int>(1, src->element_num_));
  dst->set_tensors_data_type(static_cast<TypeId>(src->tensors_data_type_));

  // Set Tensors
  for (size_t i = 0; i < src->element_num_; i++) {
    auto ret = TensorC2Tensor(&src->tensors_[i], dst->GetTensor(static_cast<int>(i)));
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "TensorC2Tensor failed";
      return ret;
    }
  }

  dst->set_element_shape(std::vector<int>(src->element_shape_, src->element_shape_ + src->element_shape_size_));
  dst->set_max_elements_num(src->max_elements_num_);
  return RET_OK;
}

#endif

int GenerateOutTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &outputs,
                       std::vector<TensorC *> *out_tensor_c) {
  MS_CHECK_TRUE_RET(out_tensor_c != nullptr && parameter != nullptr, RET_ERROR);
  if (parameter->type_ == luojianet_ms::schema::PrimitiveType_TensorListFromTensor ||
      parameter->type_ == luojianet_ms::schema::PrimitiveType_TensorListReserve ||
      parameter->type_ == luojianet_ms::schema::PrimitiveType_TensorListSetItem) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    // TensorListC ->TensorC
    MS_CHECK_TRUE_RET(!outputs.empty() && outputs.front()->data_type() == TypeId::kObjectTypeTensorType, RET_ERROR);
    auto *tensor_list_c = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
    if (tensor_list_c == nullptr) {
      return RET_ERROR;
    }
    memset(tensor_list_c, 0, sizeof(TensorListC));
    out_tensor_c->push_back(reinterpret_cast<TensorC *const>(tensor_list_c));
    return RET_OK;
#else
    MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
    return RET_ERROR;
#endif
  } else {
    return OutputTensor2TensorC(outputs, out_tensor_c);
  }
}

int GenerateInTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &inputs,
                      std::vector<TensorC *> *in_tensor_c) {
  MS_CHECK_TRUE_RET(in_tensor_c != nullptr, RET_ERROR);
  int ret = RET_OK;
  for (auto input : inputs) {
    if (input->data_type() == kObjectTypeTensorType) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
      // Tensor ->TensorList -> TensorListC -> TensorC
      auto *tensor_list = reinterpret_cast<TensorList *>(input);
      auto *tensor_list_c = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
      if (tensor_list_c == nullptr) {
        ret = RET_NULL_PTR;
        break;
      }
      memset(tensor_list_c, 0, sizeof(TensorListC));
      ret = TensorList2TensorListC(tensor_list, tensor_list_c);
      if (ret != RET_OK) {
        free(tensor_list_c->tensors_);
        free(tensor_list_c);
        return NNACL_ERR;
      }
      in_tensor_c->push_back(reinterpret_cast<TensorC *>(tensor_list_c));
#else
      MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
      return RET_NOT_SUPPORT;
#endif
    } else {
      // Tensor -> TensorC
      auto *tensor_c = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
      if (tensor_c == nullptr) {
        ret = RET_NULL_PTR;
        break;
      }
      ret = Tensor2TensorC(input, tensor_c);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Tensor to TensorC failed.";
        free(tensor_c);
        return ret;
      }
      in_tensor_c->emplace_back(tensor_c);
    }
  }
  return ret;
}

int CheckTensorsInvalid(const std::vector<Tensor *> &tensors) {
  for (auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (tensor->data_type() != kObjectTypeTensorType && tensor->data() == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor data is nullptr " << tensor->tensor_name();
      return RET_ERROR;
    }
    const auto &shape = tensor->shape();
    bool valid = all_of(shape.begin(), shape.end(), [](int i) { return i >= 0; });
    if (!valid) {
      MS_LOG(ERROR) << "The shape of tensor contains negative dimension,"
                    << "check the model and assign the input shape with method Resize().";
      return RET_ERROR;
    }
    if (tensor->format() != luojianet_ms::NHWC && tensor->format() != luojianet_ms::NCHW) {
      MS_LOG(ERROR) << "model input's format may be changed, which should be NHWC or NCHW";
      return RET_FORMAT_ERR;
    }
    if (tensor->data() == nullptr) {
      MS_LOG(ERROR) << "tensor data should be filled before run op";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

std::string ShapeToString(const std::vector<int> &shape) {
  std::string result = "[";
  int max_size = 40;
  result.reserve(max_size);
  for (size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

int CheckGraphInputShapes(const std::vector<Tensor *> &inputs,
                          const std::unordered_map<Tensor *, std::vector<int>> &input_shape_map) {
  for (const auto input : inputs) {
    MS_CHECK_TRUE_MSG(input != nullptr, RET_ERROR, "graph input tensor is nullptr.");
    if (input_shape_map.find(input) == input_shape_map.end()) {
      MS_LOG(ERROR) << "can't find " << input->tensor_name() << " in input_shape_map";
      return RET_ERROR;
    }
    if (!input_shape_map.at(input).empty() && input_shape_map.at(input) != input->shape()) {
#ifndef ENABLE_LITE_ACL
      MS_LOG(ERROR) << "graph input:" << input->tensor_name()
                    << " shape has been illegally modified, please modify the input shape with method Resize().";
      return RET_ERROR;
#else
      MS_LOG(WARNING) << "Please check graph input " << input->tensor_name()
                      << " shape:" << ShapeToString(input->shape())
                      << " has been modified by DVPP method to shape:" << ShapeToString(input_shape_map.at(input))
                      << "."
                      << "If not, the modification is illegal, please modify the input shape with method Resize().";
#endif
    }
  }
  return RET_OK;
}

std::vector<luojianet_ms::MSTensor> LiteTensorsToMSTensors(const std::vector<lite::Tensor *> &lite_tensors) {
  std::vector<luojianet_ms::MSTensor> tensors;
  std::transform(lite_tensors.begin(), lite_tensors.end(), std::back_inserter(tensors),
                 [](lite::Tensor *tensor) { return luojianet_ms::MSTensor(std::make_shared<LiteTensorImpl>(tensor)); });

  return tensors;
}

void MoveCommonTensorData(Tensor *dst_tensor, Tensor *src_tensor) {
  MS_ASSERT(src_tensor != dst_tensor);
  if (src_tensor->data() == dst_tensor->data()) {
    MS_LOG(DEBUG) << "no need to move data.";
    return;
  }
  dst_tensor->FreeData();
  dst_tensor->ResetRefCount();
  dst_tensor->set_allocator(src_tensor->allocator());

  src_tensor->allocator()->IncRefCount(src_tensor->data(), dst_tensor->ref_count());

  if (src_tensor->data() != nullptr) {
    dst_tensor->set_data(src_tensor->MutableData()); /* using MutableData to sync GPU data */
  }

  dst_tensor->set_own_data(src_tensor->own_data());
  src_tensor->DecRefCount();
}

void MoveTensorData(Tensor *dst_tensor, Tensor *src_tensor) {
  if (src_tensor == dst_tensor) {
    MS_LOG(INFO) << "no need to move.";
    return;
  }
  MS_ASSERT(src_tensor->allocator() != nullptr);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  if (src_tensor->data_type() == kObjectTypeTensorType) {
    MoveTensorListTensorData(reinterpret_cast<TensorList *>(dst_tensor), reinterpret_cast<TensorList *>(src_tensor));
  } else {
    MoveCommonTensorData(dst_tensor, src_tensor);
  }
#else
  MoveCommonTensorData(dst_tensor, src_tensor);
#endif
  return;
}

void SetCommonTensorData(Tensor *dst_tensor, Tensor *src_tensor) {
  dst_tensor->set_data(src_tensor->data());
  dst_tensor->set_own_data(false);
}

void SetTensorData(Tensor *dst_tensor, Tensor *src_tensor) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  if (src_tensor->data_type() == kObjectTypeTensorType) {
    SetTensorListTensorData(reinterpret_cast<TensorList *>(dst_tensor), reinterpret_cast<TensorList *>(src_tensor));
  } else {
    SetCommonTensorData(dst_tensor, src_tensor);
  }
#else
  SetCommonTensorData(dst_tensor, src_tensor);
#endif
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP
void MoveTensorListTensorData(TensorList *dst_tensorlist, TensorList *src_tensorlist) {
  MS_ASSERT(src_tensorlist != nullptr);
  MS_ASSERT(dst_tensorlist != nullptr);
  dst_tensorlist->FreeData();
  dst_tensorlist->ResetRefCount();
  dst_tensorlist->set_allocator(src_tensorlist->allocator());

  auto src_tensorlist_tensors_size = src_tensorlist->tensors().size();
  auto dst_tensorlist_tensors_size = dst_tensorlist->tensors().size();
  if (src_tensorlist_tensors_size != dst_tensorlist_tensors_size) {
    MS_LOG(ERROR) << "src tensorlist: " << src_tensorlist->tensor_name()
                  << " tesnors size: " << src_tensorlist_tensors_size
                  << " vs dst tensorlist: " << dst_tensorlist->tensor_name()
                  << " tensors size: " << dst_tensorlist_tensors_size;
    return;
  }

  dst_tensorlist->set_own_data(src_tensorlist->own_data());
  for (size_t i = 0; i < src_tensorlist_tensors_size; ++i) {
    auto &src_tensor = src_tensorlist->tensors()[i];
    auto &dst_tensor = dst_tensorlist->tensors()[i];

    if (src_tensor->allocator() != nullptr) {
      src_tensor->allocator()->IncRefCount(src_tensor->data(), dst_tensor->ref_count());
    }
    dst_tensor->set_own_data(src_tensor->own_data());
    if (src_tensor->data() != nullptr) {
      dst_tensor->set_data(src_tensor->MutableData()); /* using MutableData to sync GPU data */
    }
    dst_tensor->set_shape(src_tensor->shape());
  }

  if (src_tensorlist->IsConst() || src_tensorlist->IsGraphInput()) {
    dst_tensorlist->set_own_data(false);
  } else {
    src_tensorlist->DecRefCount();
  }
}

void SetTensorListTensorData(TensorList *dst_tensor_list, TensorList *src_tensor_list) {
  dst_tensor_list->FreeTensorListData();
  dst_tensor_list->set_own_data(false);
  dst_tensor_list->set_tensors(src_tensor_list->tensors());
  dst_tensor_list->set_tensors_data_type(src_tensor_list->tensors_data_type());
  dst_tensor_list->set_element_shape(src_tensor_list->element_shape());
}
#endif

void SetTensorShape(Tensor *dst, Tensor *src) {
  dst->set_shape(src->shape());
  dst->set_format(src->format());
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP
void SetTensorListShape(Tensor *dst, Tensor *src) {
  auto input_tensorlist = reinterpret_cast<TensorList *>(dst);
  auto input_data_tensorlist = reinterpret_cast<TensorList *>(src);
  if (input_data_tensorlist == nullptr || input_tensorlist == nullptr) {
    MS_LOG(ERROR) << "cast to tensorlist failed.";
    return;
  }
  input_tensorlist->FreeTensorListData();
  input_tensorlist->set_element_shape(input_data_tensorlist->element_shape());
  input_tensorlist->set_shape(input_data_tensorlist->shape());
  std::vector<std::vector<int>> tensor_shape{};
  std::transform(input_data_tensorlist->tensors().begin(), input_data_tensorlist->tensors().end(),
                 std::back_inserter(tensor_shape), [](const Tensor *tensor_item) { return tensor_item->shape(); });
  input_tensorlist->MallocTensorListData(input_data_tensorlist->tensors_data_type(), tensor_shape);
}
#endif

bool NeedCastData(Tensor *dst_tensor, Tensor *src_tensor) {
  if (dst_tensor->data_type() != kObjectTypeTensorType && src_tensor->data_type() != kObjectTypeTensorType &&
      dst_tensor->data_type() != src_tensor->data_type()) {
    return true;
  }
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  if (dst_tensor->data_type() == kObjectTypeTensorType && src_tensor->data_type() == kObjectTypeTensorType &&
      reinterpret_cast<TensorList *>(dst_tensor)->tensors_data_type() !=
        reinterpret_cast<TensorList *>(src_tensor)->tensors_data_type()) {
    return true;
  }
#endif
  return false;
}

int CastTensorData(Tensor *dst, Tensor *src, bool support_fp16) {
  int ret = RET_OK;
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  if (src->data_type() != kObjectTypeTensorType) {
    ret = CastCommonTensorData(dst, src, support_fp16);
  } else {
    ret =
      CastTensorListTensorData(reinterpret_cast<TensorList *>(dst), reinterpret_cast<TensorList *>(src), support_fp16);
  }
#else
  ret = CastCommonTensorData(dst, src, support_fp16);
#endif
  src->DecRefCount();
  return ret;
}

int CastCommonTensorData(Tensor *dst, Tensor *src, bool support_fp16) {
  dst->ReallocData();
  dst->ResetRefCount();
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  if (dst->shape() != src->shape()) {
    MS_LOG(ERROR) << "dst tensor: " << dst->tensor_name() << " shape: " << dst->shape() << " vs "
                  << "src tensor: " << src->tensor_name() << " shape: " << src->shape();
    return RET_PARAM_INVALID;
  }
  auto dst_data = dst->MutableData(); /* using MutableData to sync GPU data */
  auto src_data = src->MutableData();
  auto src_nums_size = src->ElementsNum();
  auto dst_data_type = static_cast<int>(dst->data_type());
  auto src_data_type = static_cast<int>(src->data_type());
  if (dst_data_type == kNumberTypeFloat32 && src_data_type == kNumberTypeFloat16) {
    Float16ToFloat32_fp16_handler(src_data, dst_data, src_nums_size, support_fp16);
  } else if (dst_data_type == kNumberTypeFloat16 && src_data_type == kNumberTypeFloat32) {
    Float32ToFloat16_fp16_handler(src_data, dst_data, src_nums_size, support_fp16);
  } else {
    MS_LOG(ERROR) << "not support dst_data_type: " << dst_data_type << " src_data_type: " << src_data_type;
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
#endif
  return RET_ERROR;
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP
int CastTensorListTensorData(TensorList *dst_tensorlist, TensorList *src_tensorlist, bool support_fp16) {
  MS_ASSERT(src_tensorlist != nullptr);
  MS_ASSERT(dst_tensorlist != nullptr);
  dst_tensorlist->set_shape(src_tensorlist->shape());
  std::vector<std::vector<int>> tensors_shapes{};
  tensors_shapes.resize(src_tensorlist->tensors().size());
  for (size_t i = 0; i < tensors_shapes.size(); ++i) {
    tensors_shapes[i] = src_tensorlist->tensors()[i]->shape();
  }
  if (src_tensorlist->tensors_data_type() == kNumberTypeFloat16) {
    dst_tensorlist->MallocTensorListData(kNumberTypeFloat32, tensors_shapes);
  }
  if (src_tensorlist->tensors_data_type() == kNumberTypeFloat32) {
    dst_tensorlist->MallocTensorListData(kNumberTypeFloat16, tensors_shapes);
  }
  dst_tensorlist->set_allocator(src_tensorlist->allocator());
  dst_tensorlist->ResetRefCount();

  for (size_t i = 0; i < src_tensorlist->tensors().size(); ++i) {
    auto &src_tensor = src_tensorlist->tensors()[i];
    auto &dst_tensor = dst_tensorlist->tensors()[i];
    CastCommonTensorData(dst_tensor, src_tensor, support_fp16);
  }
  return RET_OK;
}
#endif
}  // namespace lite
}  // namespace luojianet_ms
