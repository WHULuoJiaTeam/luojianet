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

#include "src/train/train_session.h"
#include <sys/stat.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>
#include <set>
#include "include/errorcode.h"
#include "src/lite_model.h"
#include "src/lite_kernel_util.h"
#include "src/tensor.h"
#include "src/kernel_registry.h"
#include "src/common/prim_util.h"
#include "src/common/tensor_util.h"
#include "src/common/utils.h"
#include "src/train/optimizer_kernel.h"
#include "src/train/train_utils.h"
#include "src/train/train_export.h"
#include "src/train/opt_allocator.h"
#include "src/train/static_allocator.h"
#include "src/train/train_populate_parameter.h"
#include "src/train/train_populate_parameter_v0.h"

namespace luojianet_ms {
namespace lite {
const char *kGradName = "Gradients";
const char *kOptimizerName = "optimizer";

TrainSession::TrainSession() {
  is_train_session_ = true;
  InitCallBack();
}

int TrainSession::TrainInit(InnerContext *context, const TrainCfg *train_cfg) {
  if (train_cfg != nullptr) {
    if (train_cfg->mix_precision_cfg_.loss_scale_ <= 0) {
      MS_LOG(ERROR) << "illegal loss scale configuration";
      return RET_NULL_PTR;
    }
    cfg_ = *train_cfg;
  }
  if (context == nullptr) {
    MS_LOG(ERROR) << "context cannot be nullptr";
    return RET_NULL_PTR;
  }
  allocator_ = context->allocator;
  return lite::LiteSession::Init(context);
}

std::vector<CreatorOp> TrainSession::ReplaceOps() {
  const std::vector<CreatorOp> replace = {
    // currently no ops are Hijacked by TrainSession
  };
  luojianet_ms::lite::KernelRegistry *reg = luojianet_ms::lite::KernelRegistry::GetInstance();
  std::vector<CreatorOp> results;
  for (auto v : replace) {
    const CreatorOp cl = make_tuple(std::get<0>(v), reg->GetCreator(std::get<0>(v)));
    results.push_back(cl);
    reg->RegKernel(std::get<0>(v), std::get<1>(v));
  }
  return results;
}

void TrainSession::RestoreOps(const std::vector<CreatorOp> &restore) {
  luojianet_ms::lite::KernelRegistry *reg = luojianet_ms::lite::KernelRegistry::GetInstance();
  for (auto v : restore) {
    reg->RegKernel(std::get<0>(v), std::get<1>(v));
  }
}

int TrainSession::AllocWorkSpace() {
  size_t workspace_size = 0;
  for (auto kernel : this->train_kernels_) {
    if (workspace_size < static_cast<kernel::InnerKernel *>(kernel->kernel())->workspace_size()) {
      workspace_size = static_cast<kernel::InnerKernel *>(kernel->kernel())->workspace_size();
    }
  }
  workspace_ = malloc(workspace_size);
  if (workspace_ == nullptr) {
    MS_LOG(ERROR) << "cannot allocate " << workspace_size << " for workspace";
    return RET_ERROR;
  }
  for (auto kernel : this->train_kernels_) {
    static_cast<kernel::InnerKernel *>(kernel->kernel())->set_workspace(workspace_);
  }
  return RET_OK;
}

void TrainSession::FreeWorkSpace() {
  if (workspace_ != nullptr) {
    free(workspace_);
    workspace_ = nullptr;
  }
  for (auto kernel : this->train_kernels_) {
    static_cast<kernel::InnerKernel *>(kernel->kernel())->FreeWorkspace();
  }
}

int TrainSession::InitCallBack() {
  sched_mix_precision_callback_ = [&](const Model::Node *node) {
    if (!context_->IsCpuFloat16Enabled()) {
      return false;
    }
    bool force_fp16 = false;
    auto node_type = GetPrimitiveType(node->primitive_, SCHEMA_VERSION::SCHEMA_CUR);
    if (node_type == schema::PrimitiveType_Cast) {
      schema::Tensor *tensor = model_.get()->all_tensors_.at(node->input_indices_[0]);
      if (tensor->dataType() == kNumberTypeFloat16) {
        force_fp16 = true;
      } else if (tensor->dataType() == kNumberTypeFloat32) {
        return false;
      }
    } else {
      auto in_size = node->input_indices_.size();
      for (std::size_t k = 0; k < in_size; k++) {
        schema::Tensor *tensor = model_->all_tensors_.at(node->input_indices_[k]);
        if ((tensor->dataType() == kNumberTypeFloat16) && (tensor->nodeType() == NodeType_ValueNode)) {
          force_fp16 = true;
          break;
        }
      }
    }

    const auto &node_name = node->name_;
    bool is_fp16 = true;
    if (!force_fp16) {
      // optimizer runs in fp32
      if (node_name.find(kOptimizerName) != std::string::npos) {
        is_fp16 = false;
      }
      // loss function runs in fp32
      auto v = get_loss_name();
      for (auto &s : v) {
        if (node_name.find(s) != std::string::npos) {
          is_fp16 = false;
          break;
        }
      }
      // run bn according to user configuration
      if ((cfg_.mix_precision_cfg_.keep_batchnorm_fp32_) &&
          (node_type == schema::PrimitiveType_FusedBatchNorm || node_type == schema::PrimitiveType_BatchNorm ||
           node_type == schema::PrimitiveType_BatchNormGrad)) {
        is_fp16 = false;
      }
    }
    MS_LOG(DEBUG) << "Debug: " << node_name << ((is_fp16) ? " fp16" : " fp32");
    return is_fp16;
  };
  return RET_OK;
}

static int ReshapeWeightTensor(Tensor *orig_tensor, tensor::MSTensor *new_tensor) {
  if (orig_tensor->data_type() != new_tensor->data_type()) {
    MS_LOG(ERROR) << "Cannot reshape tensor of different type: " << new_tensor->tensor_name();
    return RET_PARAM_INVALID;
  }

  if (orig_tensor->category() != lite::Category::CONST_TENSOR) {
    MS_LOG(ERROR) << "Cannot reshape non const tensor: " << new_tensor->tensor_name();
    return RET_ERROR;
  }

  auto orig_size = orig_tensor->Size();
  uint8_t *new_data = reinterpret_cast<uint8_t *>(new_tensor->data());
  if (new_data == nullptr) {
    // Copy original data into new_tensor
    new_data = reinterpret_cast<uint8_t *>(new_tensor->MutableData());
    if (new_data == nullptr) {
      MS_LOG(ERROR) << "Allocation of Data Failed" << new_tensor->tensor_name();
      return RET_ERROR;
    }
    if (orig_size == 0) {
      MS_LOG(ERROR) << "Operation failed: Both new tensors and original one have no data";
      return RET_ERROR;
    }
    uint8_t *orig_data = reinterpret_cast<uint8_t *>(orig_tensor->data());
    for (unsigned int loc = 0; loc < new_tensor->Size(); loc++) {
      new_data[loc] = orig_data[loc % orig_size];
    }
  }

  orig_tensor->FreeData();
  orig_tensor->set_data(nullptr);
  orig_tensor->set_shape(new_tensor->shape());

  uint8_t *dst_data = reinterpret_cast<uint8_t *>(orig_tensor->MutableData());
  if (dst_data == nullptr) {
    MS_LOG(ERROR) << "Allocation of Data Failed";
    return RET_ERROR;
  }
  std::copy(new_data, new_data + orig_tensor->Size(), dst_data);
  return RET_OK;
}

int TrainSession::UpdateWeights(std::vector<tensor::MSTensor *> modify_tensors) {
  unsigned int num_of_found_tensors = 0;
  for (auto tensor : tensors_) {
    for (auto modify : modify_tensors) {
      if (modify == nullptr) {
        MS_LOG(ERROR) << "Tensor is nullptr";
        return RET_PARAM_INVALID;
      }
      if (modify->tensor_name() == tensor->tensor_name()) {
        auto ret = ReshapeWeightTensor(tensor, modify);
        num_of_found_tensors++;
        if (ret != RET_OK) {
          return ret;
        }
        break;
      }
    }
  }
  if (num_of_found_tensors != modify_tensors.size()) {
    MS_LOG(ERROR) << "Did not find all the given tensors in the model";
    return RET_ERROR;
  }
  auto ret = ReSizeKernels(kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize kernels fail!";
    return ret;
  }

  bool is_eval = IsEval();
  ret = Train();  // This will trigger proper Allocation of static data;
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "General failure occurred during Update of Weights";
    return ret;
  }
  if (is_eval) {
    ret = Eval();
  }
  return ret;
}

int TrainSession::AllocTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  if (!IS_STATIC_ALLOCATOR(allocator_)) return RET_OK;
  OptAllocator allocator;
  std::unordered_map<lite::Tensor *, int> ref_count;
  std::unordered_map<lite::Tensor *, size_t> offset_map;
  int counter = 0;
  uint32_t input_idx = 0;
  for (auto &kernel : kernels) {
    for (size_t i = 0; i < kernel->out_tensors().size(); i++) {
      auto tensor = kernel->out_tensors().at(i);
      bool in_place = false;
      if (counter != 0) {
        in_place = IsInPlaceTensor(kernel, i, ref_count, &input_idx);
      }
      counter++;
      size_t offset;
      if (in_place) {
        offset = GetInplaceTensorOffset(kernel, offset_map, &ref_count, input_idx);
      } else {
        size_t size = tensor->Size();
        offset = allocator.Malloc(size);
      }
      offset_map[tensor] = offset;
      ref_count[tensor] = tensor->init_ref_count();
    }
    for (auto tensor : kernel->in_tensors()) {
      if (tensor->category() == lite::Category::VAR) {
        int count = ref_count[tensor] - 1;
        ref_count[tensor] = count;
        if (count == 0) {
          allocator.Free(offset_map[tensor]);
        }
      }
    }
  }
  // Set Tensor data
  auto size = allocator.total_size();
  if (size > tensors_data_size_) {
    free(tensors_data_);
    tensors_data_ = nullptr;
  }
  if (tensors_data_ == nullptr) {
    auto buf = malloc(size);
    if (buf == nullptr) {
      MS_LOG(ERROR) << "cannot allocate buffer size" << size;
      return RET_ERROR;
    }
    StaticAllocator *alloc = reinterpret_cast<StaticAllocator *>(allocator_.get());
    alloc->SetContex(buf, size);
    tensors_data_ = buf;
    tensors_data_size_ = size;
  }
  for (auto kernel : train_kernels_) {
    for (auto tensor : kernel->out_tensors()) {
      auto it = offset_map.find(tensor);
      if (it != offset_map.end()) {
        tensor->set_data(reinterpret_cast<void *>(reinterpret_cast<char *>(tensors_data_) + it->second));
      }
    }
  }
  return RET_OK;
}

int TrainSession::CompileGraph(lite::Model *model) { return lite::RET_ERROR; }

int TrainSession::CompileTrainGraph(std::shared_ptr<Model> model) {
  model_ = model;
  auto restore = ReplaceOps();
  sched_cb_ = std::make_unique<SchedulerCb>(sched_mix_precision_callback_);
  if (sched_cb_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create SchedulerCb node";
    return RET_ERROR;
  }

#ifdef ENABLE_V0
  if (reinterpret_cast<LiteModel *>(model_.get())->GetSchemaVersion() == SCHEMA_VERSION::SCHEMA_V0) {
    kernel::PopulateTrainV0Parameters();
  }
#endif
  if (reinterpret_cast<LiteModel *>(model_.get())->GetSchemaVersion() == SCHEMA_VERSION::SCHEMA_CUR) {
    kernel::PopulateTrainParameters();
  }

  auto ret = lite::LiteSession::CompileGraph(model_.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to compile train model";
    return RET_ERROR;
  }
  orig_output_node_map_ = output_node_map_;
  orig_output_tensor_map_ = output_tensor_map_;
  orig_output_tensor_names_ = output_tensor_names_;
  for (auto inTensor : inputs_) inTensor->MutableData();
  RestoreOps(restore);
  CompileTrainKernels();      // Prepare a list of train kernels
  CompileOptimizedKernels();  // Prepare a list of kernels which are optimized (weight update step)
  CompileTrainOutputs();      // prepare outputs in train mode
  CompileEvalOutputs();       // prepare outputs in eval mode
  // Prepare a list of eval kernels
  if (CompileInferenceKernels() != RET_OK) {
    MS_LOG(ERROR) << "CompileInferenceKernels failed.";
    return RET_ERROR;
  }
  ret = AllocWorkSpace();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to allocate space";
    return RET_ERROR;
  }
  ret = AllocTensors(train_kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to allocate space";
    return RET_ERROR;
  }
  return RET_OK;
}

TrainSession::~TrainSession() {
  FreeWorkSpace();
  if (tensors_data_ != nullptr) {
    free(tensors_data_);
    tensors_data_ = nullptr;
  }
}

int TrainSession::ExecKernels(const KernelCallBack &before, const KernelCallBack &after,
                              const std::vector<kernel::LiteKernel *> &run_kernels) {
  for (auto *kernel : run_kernels) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Execute(before, after);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "Execute kernel failed, name: " << kernel->name();
      return ret;
    }
  }
  return RET_OK;
}

void TrainSession::RestoreTensorData() {
  for (auto &restored_origin_tensor : restored_origin_tensors_) {
    auto *origin_tensor = restored_origin_tensor.first;
    auto *restored_tensor = restored_origin_tensor.second;
    MS_ASSERT(origin_tensor != nullptr);
    MS_ASSERT(restored_tensor != nullptr);

    bool own_data = restored_tensor->own_data();
    if (origin_tensor->data() == nullptr) {
      restored_tensor->FreeData();
    } else {
      origin_tensor->FreeData();
    }
    origin_tensor->set_data_type(restored_tensor->data_type());
    origin_tensor->set_data(restored_tensor->data());
    origin_tensor->set_own_data(own_data);
  }
}

void TrainSession::FreeRestoreTensors() {
  for (auto &restored_origin_tensor : restored_origin_tensors_) {
    auto *restored_tensor = restored_origin_tensor.second;
    restored_tensor->set_data(nullptr);
    delete (restored_tensor);
  }
  restored_origin_tensors_.clear();
}

bool TrainSession::IsLossTensor(Tensor *tensor) {
  MS_ASSERT(tensor != nullptr);
  bool isLoss = false;
  auto t_n = tensor->tensor_name();
  auto v = get_loss_name();
  for (auto &s : v) {
    if (t_n.find(s) != std::string::npos) {
      isLoss = true;
      break;
    }
  }
  return isLoss;
}

bool TrainSession::AllInputsNeedScale(kernel::LiteKernel *kernel) {
  auto type = kernel->type();
  bool is_scale = false;
  switch (type) {
    case schema::PrimitiveType_AbsGrad:
    case schema::PrimitiveType_AddFusion:
    case schema::PrimitiveType_SubFusion:
    case schema::PrimitiveType_AddN:
      for (auto &tensor : kernel->in_tensors()) {
        is_scale = is_scale || tensor->IsScale();
      }
      return (is_scale);
    default:
      return false;
  }
  return false;
}

int TrainSession::MixPrecisionPreProcess(kernel::LiteKernel *kernel, float scale) {
  auto kernel_type = kernel->desc().data_type;
  auto all_scale = AllInputsNeedScale(kernel);

  for (auto &tensor : kernel->in_tensors()) {
    if ((tensor->IsScale() == false) && ((!IsLossKernel(kernel) && IsLossTensor(tensor)) || (all_scale == true))) {
      ScaleTensor(tensor, scale);
    }
    // adjust tensor data type
    if (tensor->data_type() != kernel_type) {
      auto restore_tensor = CastTensor(tensor, kernel_type, this->context_->device_and_pkg_support_fp16());
      if (restore_tensor != nullptr) {
        restored_origin_tensors_[tensor] = restore_tensor;
      }
    }
  }
  return RET_OK;
}

int TrainSession::MixPrecisionPostProcess(kernel::LiteKernel *kernel) {
  RestoreTensorData();
  FreeRestoreTensors();

  float scale = 1.0f;
  auto all_scale = AllInputsNeedScale(kernel);
  for (auto &tensor : kernel->in_tensors()) {
    if (tensor->IsScale()) {
      scale *= tensor->get_scale();
      if (all_scale) {
        break;
      }
    }
  }
  for (auto &tensor : kernel->out_tensors()) {
    tensor->set_scale(scale);
  }

  for (auto &tensor : kernel->in_tensors()) {
    if ((tensor->IsScale() == true) && ((!IsLossKernel(kernel) && IsLossTensor(tensor)) || (all_scale == true))) {
      ScaleTensor(tensor, 1.0f / scale);
    }
  }
  return RET_OK;
}

int TrainSession::MixPrecisionExecKernels(const KernelCallBack &before, const KernelCallBack &after,
                                          const std::vector<kernel::LiteKernel *> &run_kernels) {
  float scale = cfg_.mix_precision_cfg_.loss_scale_;
  for (auto *kernel : run_kernels) {
    MS_ASSERT(kernel != nullptr);
    auto ret = MixPrecisionPreProcess(kernel, scale);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MixPrecisionPreProcess failed.";
      return RET_ERROR;
    }
    ret = kernel->Execute(before, after);
    if (RET_OK != ret) {
      MixPrecisionPostProcess(kernel);
      // decrease loss scale in case of nan or inf
      if (ret == RET_OUT_OF_TENSOR_RANGE) {
        bool is_dynamic_scale = cfg_.mix_precision_cfg_.dynamic_loss_scale_;
        cfg_.mix_precision_cfg_.loss_scale_ = std::max(((is_dynamic_scale) ? (scale / 2.f) : scale), 1.0f);
        num_of_not_nan_iter_ = 0;
        return RET_OK;
      }
      MS_LOG(ERROR) << "Execute kernel failed, name: " << kernel->name();
      return ret;
    }
    MixPrecisionPostProcess(kernel);
  }
  // increase dynamic loss scale if pass pass threshold
  if (cfg_.mix_precision_cfg_.dynamic_loss_scale_) {
    num_of_not_nan_iter_++;
    if (num_of_not_nan_iter_ >= cfg_.mix_precision_cfg_.num_of_not_nan_iter_th_) {
      cfg_.mix_precision_cfg_.loss_scale_ = std::min((cfg_.mix_precision_cfg_.loss_scale_ * 2.0f), 65536.0f);
      num_of_not_nan_iter_ = 0;
    }
  }

  // cast output to FP32
  if (train_mode_ == false) {
    for (auto t : this->outputs_) {
      if (t->data_type() == kNumberTypeFloat16) {
        auto restore = CastTensor(t, kNumberTypeFloat32, this->context_->device_and_pkg_support_fp16());
        delete restore;
      }
    }
  }
  return RET_OK;
}

int TrainSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  // check inputs
  auto ret = CheckTensorsInvalid(inputs_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CheckInputs failed";
    return ret;
  }

  // build out tensor
  this->outputs_.clear();
  for (auto &ms_tensors : output_node_map_) {
    for (auto &ms_tensor : ms_tensors.second) {
      auto lite_tensor = static_cast<lite::Tensor *>(ms_tensor);
      this->outputs_.push_back(lite_tensor);
    }
  }

  if (this->context_ == nullptr) {
    MS_LOG(ERROR) << "context is null";
    return lite::RET_NULL_PTR;
  }
  auto &run_kernels = (train_mode_) ? train_kernels_ : inference_kernels_;
  if (context_->IsCpuFloat16Enabled()) {
    ret = MixPrecisionExecKernels(before, after, run_kernels);
  } else {
    ret = ExecKernels(before, after, run_kernels);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to run model kernels";
    return ret;
  }

  if (train_mode_ && virtual_batch_multiplier_) {
    virtual_batch_idx_++;
    if (virtual_batch_idx_ >= virtual_batch_multiplier_) {
      virtual_batch_idx_ = 0;
      ret = OptimizerStep();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "failed to optimize model weights";
        return ret;
      }
    }
  }
  return RET_OK;
}

int TrainSession::Train() {
  // shift kernels to train mode
  train_mode_ = true;
  virtual_batch_idx_ = 0;
  for (auto &kernel : this->train_kernels_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Train();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << kernel->name() << " failed to set train mode";
      return RET_ERROR;
    }
  }
  // set train outputs
  output_node_map_ = train_output_node_map_;
  output_tensor_map_ = train_output_tensor_map_;
  output_tensor_names_ = train_output_tensor_names_;
  kernel::LiteKernelUtil::InitTensorInitRefCount(train_kernels_);
  for (auto &ms_tensors : eval_output_node_map_) {  // Allow to look at prediction also during training
    for (auto &ms_tensor : ms_tensors.second) {
      lite::Tensor *lite_tensor = static_cast<lite::Tensor *>(ms_tensor);
      lite_tensor->set_init_ref_count(lite_tensor->init_ref_count() + 1);
    }
  }
  // allocate tensors
  auto ret = AllocTensors(train_kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to allocate tensor space";
    return RET_ERROR;
  }
  return RET_OK;
}

int TrainSession::Eval() {
  // shift kernels to eval mode
  train_mode_ = false;
  virtual_batch_idx_ = 0;
  for (auto &kernel : this->train_kernels_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Eval();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << kernel->name() << " failed to set eval mode";
      return RET_ERROR;
    }
  }
  // set eval outputs
  output_node_map_ = eval_output_node_map_;
  output_tensor_map_ = eval_output_tensor_map_;
  output_tensor_names_ = eval_output_tensor_names_;
  kernel::LiteKernelUtil::InitTensorInitRefCount(inference_kernels_);
  for (auto &ms_tensors : eval_output_node_map_) {
    for (auto &ms_tensor : ms_tensors.second) {
      lite::Tensor *lite_tensor = static_cast<lite::Tensor *>(ms_tensor);
      lite_tensor->set_init_ref_count(lite_tensor->init_ref_count() + 1);
    }
  }
  auto ret = AllocTensors(inference_kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to allocate space";
    return RET_ERROR;
  }
  return RET_OK;
}

void TrainSession::CompileEvalOutputs() {
  eval_output_node_map_.clear();
  eval_output_tensor_map_.clear();
  eval_output_tensor_names_.clear();
  for (auto kernel : this->train_kernels_) {
    if (IsLossKernel(kernel) && !(IsGradKernel(kernel))) {
      for (auto in_kernel : kernel->in_kernels()) {
        if (IsLossKernel(in_kernel) || IsGradKernel(in_kernel)) continue;
        bool is_loss = false;
        for (auto in_in_kernel : in_kernel->in_kernels()) {
          if (IsLossKernel(in_in_kernel)) is_loss = true;
        }
        if (is_loss) continue;
        // insert if not already in
        if (eval_output_node_map_.find(in_kernel->name()) == eval_output_node_map_.end()) {
          auto *ms_tensor = in_kernel->out_tensors().at(0);
          if (ms_tensor != nullptr) {
            ms_tensor->set_init_ref_count(ms_tensor->init_ref_count() + 1);
            eval_output_node_map_[in_kernel->name()].emplace_back(ms_tensor);
            auto index = TSFindTensor(tensors_, ms_tensor);
            if (index != tensors_.size()) {
              if (!ms_tensor->tensor_name().empty()) {
                eval_output_tensor_map_.insert(std::make_pair(ms_tensor->tensor_name(), ms_tensor));
                eval_output_tensor_names_.emplace_back(ms_tensor->tensor_name());
              } else {
                eval_output_tensor_map_.insert(std::make_pair(std::to_string(index), ms_tensor));
                eval_output_tensor_names_.emplace_back(std::to_string(index));
              }
            }
          }
        }
      }
    }
  }
  if (eval_output_node_map_.empty()) eval_output_node_map_ = orig_output_node_map_;
  if (eval_output_tensor_map_.empty()) eval_output_tensor_map_ = orig_output_tensor_map_;
  if (eval_output_tensor_names_.empty()) eval_output_tensor_names_ = orig_output_tensor_names_;
}

void TrainSession::CompileTrainOutputs() {
  train_output_node_map_.clear();
  train_output_tensor_map_.clear();
  train_output_tensor_names_.clear();
  for (auto kernel : this->train_kernels_) {
    if (orig_output_node_map_.find(kernel->name()) == orig_output_node_map_.end()) continue;
    // Mask out optimizer out tensors
    if (IsMaskOutput(kernel)) continue;
    // insert if not already in
    if (train_output_node_map_.find(kernel->name()) == train_output_node_map_.end()) {
      auto *ms_tensor = kernel->out_tensors().at(0);
      if (ms_tensor != nullptr) {
        train_output_node_map_[kernel->name()].emplace_back(ms_tensor);
        auto index = TSFindTensor(tensors_, ms_tensor);
        if (index != tensors_.size()) {
          if (!ms_tensor->tensor_name().empty()) {
            train_output_tensor_map_.insert(std::make_pair(ms_tensor->tensor_name(), ms_tensor));
            train_output_tensor_names_.emplace_back(ms_tensor->tensor_name());
          } else {
            train_output_tensor_map_.insert(std::make_pair(std::to_string(index), ms_tensor));
            train_output_tensor_names_.emplace_back(std::to_string(index));
          }
        }
      }
    }
  }
  if (train_output_node_map_.empty()) train_output_node_map_ = orig_output_node_map_;
  if (train_output_tensor_map_.empty()) train_output_tensor_map_ = orig_output_tensor_map_;
  if (train_output_tensor_names_.empty()) train_output_tensor_names_ = orig_output_tensor_names_;
}

void TrainSession::BuildInferenceKernelsRecursive(kernel::LiteKernel *kernel, std::vector<kernel::LiteKernel *> *v) {
  MS_ASSERT(kernel != nullptr);
  MS_ASSERT(v != nullptr);
  if (std::find(v->begin(), v->end(), kernel) == v->end()) {  // kernel is not already in vector
    for (auto in_node : kernel->in_kernels()) {
      BuildInferenceKernelsRecursive(in_node, v);
    }
    if (!IsLossKernel(kernel)) v->push_back(kernel);
  }
}

void TrainSession::CompileTrainKernels() {
  train_kernels_.clear();
  std::vector<kernel::LiteKernel *> train_kernels;
  for (auto ori_kernel : kernels_) {
    if (ori_kernel->subgraph_type() == kernel::kNotSubGraph) {
      train_kernels.push_back(ori_kernel);
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(ori_kernel);
      std::copy(sub_graph->nodes().begin(), sub_graph->nodes().end(), std::back_inserter(train_kernels));
    }
  }
  // For LSTM GPU operators are synchronized internally hence we need to add sync mechanizm to execution graph
  for (auto k : train_kernels) {
    if (k->type() == schema::PrimitiveType_LSTMGradWeight) {
      // Find PrimitiveType_LSTMGradData that matches this PrimitiveType_LSTMGradWeight
      for (auto mk : train_kernels) {
        if (mk->type() == schema::PrimitiveType_LSTMGradData) {
          if (k->in_tensors().at(C2NUM)->tensor_name() == mk->in_tensors().at(0)->tensor_name()) {
            mk->AddOutKernel(k);
            k->AddInKernel(mk);
          }
        }
      }
    }
  }
  std::queue<kernel::LiteKernel *> queue;
  for (auto k : train_kernels) {
    auto in_kernels = k->in_kernels();
    if (in_kernels.size() == 0) {
      queue.push(k);
    }
  }
  std::unordered_map<kernel::LiteKernel *, int> map;
  while (queue.size()) {
    // pop first element
    auto k = queue.front();
    train_kernels_.push_back(k);
    queue.pop();
    for (auto &ok : k->out_kernels()) {
      auto cnt_iter = map.find(ok);
      if (cnt_iter == map.end()) {
        int ref_cnt = ok->in_kernels().size();
        map[ok] = ref_cnt;
      }
      cnt_iter = map.find(ok);
      auto ref_cnt = cnt_iter->second - 1;
      map[ok] = ref_cnt;
      if (ref_cnt <= 0) {
        queue.push(cnt_iter->first);
      }
    }
  }
}

int TrainSession::CompileInferenceKernels() {
  inference_kernels_.clear();
  for (auto item : eval_output_node_map_) {
    std::string kernel_name = item.first;
    auto kernel = TSFindKernel(train_kernels_, kernel_name);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "kernel is nullptr";
      return RET_ERROR;
    }
    BuildInferenceKernelsRecursive(kernel, &inference_kernels_);
  }
  if (inference_kernels_.size() == 0) {
    inference_kernels_ = this->train_kernels_;
  }
  return RET_OK;
}

void TrainSession::CompileOptimizedKernels() {
  std::vector<lite::Tensor *> out_tensor;
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      std::copy(kernel->in_tensors().begin(), kernel->in_tensors().end(), std::back_inserter(out_tensor));
      if (cfg_.accumulate_gradients_) {
        auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
        auto ret = optimizer->SetOptimizerMode(kernel::WeightUpdateMode::ACCUMULATE_GRADS);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "SetOptimizerMode failed.";
          return;
        }
      }
    }
  }

  for (auto kernel : this->train_kernels_) {
    if (!IsOptimizer(kernel)) {
      for (auto it : kernel->in_tensors()) {
        if (std::find(out_tensor.begin(), out_tensor.end(), it) != out_tensor.end()) {
          kernel->SetTrainable(true);
          break;
        }
      }
    }
  }
}

int TrainSession::SetLearningRate(float learning_rate) {
  if (learning_rate < 0.0f) {
    MS_LOG(ERROR) << "learning rate should more than 0";
    return RET_ERROR;
  }
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      auto ret = optimizer->SetLearningRate(learning_rate);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to set learning rate";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

float TrainSession::GetLearningRate() {
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      return optimizer->GetLearningRate();
    }
  }
  return 0.0;
}

std::vector<tensor::MSTensor *> TrainSession::GetOptimizerParams() const {
  std::vector<tensor::MSTensor *> params;
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      auto kernelParams = optimizer->GetOptimizerParams();
      for (size_t ix = 0; ix < kernelParams.size(); ix++) {
        auto kernelParam = kernelParams[ix];
        auto name = kernelParam->tensor_name();
        bool found = false;
        for (size_t iy = 0; iy < params.size(); iy++) {
          if (params[iy]->tensor_name() == name) {
            found = true;
            break;
          }
        }
        if (!found) {
          params.push_back(kernelParam);
        }
      }
    }
  }
  return params;
}

int TrainSession::SetOptimizerParams(const std::vector<tensor::MSTensor *> &params) {
  for (size_t ix = 0; ix < params.size(); ix++) {
    auto param = params[ix];
    if (param == nullptr) {
      MS_LOG(ERROR) << "Param tensor is null.";
      return RET_ERROR;
    }
    bool found = false;
    for (auto kernel : this->train_kernels_) {
      if (IsOptimizer(kernel)) {
        auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
        found = optimizer->SetOptimizerParams(param);
        if (found) break;
      }
    }
    if (!found) {
      MS_LOG(ERROR) << "Tensor " << param->tensor_name() << " with " << param->ElementsNum() << " elelmts and type "
                    << param->data_type() << " is not a valid params tensor";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

std::vector<tensor::MSTensor *> TrainSession::GetGradients() const {
  std::vector<tensor::MSTensor *> gradients;
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      auto kernelGradint = optimizer->GetGradients();
      if (kernelGradint != nullptr) {
        gradients.push_back(kernelGradint);
      }
    }
  }
  return gradients;
}

int TrainSession::ApplyGradients(const std::vector<tensor::MSTensor *> &gradients) {
  auto current_gradients = GetGradients();
  if (current_gradients.size() != gradients.size()) {
    MS_LOG(ERROR) << "gradients vector has wrong size " << gradients.size() << " instead of "
                  << current_gradients.size();
    return RET_ERROR;
  }
  for (size_t ix = 0; ix < gradients.size(); ix++) {
    auto gradient = gradients[ix];
    if (gradient == nullptr) {
      MS_LOG(ERROR) << "gradient tensor is null.";
      return RET_ERROR;
    }
    bool found = false;
    for (size_t iy = 0; iy < current_gradients.size(); iy++) {
      auto current_gradient = current_gradients[iy];
      if (current_gradient->tensor_name() == gradient->tensor_name()) {
        found = true;
        if (current_gradient->Size() == gradient->Size()) {
          std::copy(static_cast<uint8_t *>(gradient->data()),
                    static_cast<uint8_t *>(gradient->data()) + gradient->Size(),
                    static_cast<uint8_t *>(current_gradient->MutableData()));
        } else {
          MS_LOG(ERROR) << "gradient tensor " << gradient->tensor_name() << " has wrong size " << gradient->Size()
                        << " instead of " << current_gradient->Size();
          return RET_ERROR;
        }
        break;
      }
    }
    if (!found) {
      MS_LOG(ERROR) << "gradient tensor " << gradient->tensor_name() << " not found";
      return RET_ERROR;
    }
  }
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      if (optimizer->set_grad_sum_valid() != RET_OK) {
        MS_LOG(ERROR) << "set grad sum valid failed.";
        return RET_ERROR;
      }
      auto ret = optimizer->OptimizerStep();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "failed to optimize model weights";
        return ret;
      }
    }
  }
  for (size_t ix = 0; ix < current_gradients.size(); ix++) {
    delete current_gradients[ix];
  }
  return RET_OK;
}

int TrainSession::AdminSetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  auto mod =
    (virtual_batch_multiplier <= 1) ? kernel::WeightUpdateMode::NORMAL : kernel::WeightUpdateMode::VIRTUAL_BATCH;
  virtual_batch_multiplier_ = (virtual_batch_multiplier <= 1) ? 0 : virtual_batch_multiplier;
  virtual_batch_idx_ = 0;

  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      if (optimizer->get_optimizer_mode() != kernel::WeightUpdateMode::NORMAL &&
          optimizer->get_optimizer_mode() != kernel::WeightUpdateMode::VIRTUAL_BATCH) {
        MS_LOG(ERROR) << kernel->name() << " failed to set optimizer mode, conflict with accumulate grads";
        return RET_ERROR;
      }
      auto ret = optimizer->SetOptimizerMode(mod);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to set optimizer mode";
        return RET_ERROR;
      }
      if (mod == kernel::WeightUpdateMode::VIRTUAL_BATCH) {
        lr = (lr < 0.0f) ? (optimizer->GetLearningRate() / static_cast<float>(virtual_batch_multiplier_)) : lr;
        ret = optimizer->SetLearningRate(lr);
      } else {
        ret = optimizer->RestoreDefaultLearningRate();
      }
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to set learning rate";
        return RET_ERROR;
      }
    }

    if (IsBN(kernel) && kernel->IsTrainable()) {
      auto batchnorm = static_cast<kernel::InnerKernel *>(kernel->kernel());
      auto ret = batchnorm->SetupVirtualBatch(virtual_batch_multiplier_, momentum);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to set momentum";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
int TrainSession::SetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  int tmp = (virtual_batch_multiplier <= 1) ? 0 : virtual_batch_multiplier;
  if (tmp != 0 && virtual_batch_multiplier_ != 0) {
    AdminSetupVirtualBatch(0, lr, momentum);
  }
  return AdminSetupVirtualBatch(virtual_batch_multiplier, lr, momentum);
}

int TrainSession::OptimizerStep() {
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      auto ret = optimizer->OptimizerStep();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to do optimize step";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

bool TrainSession::IsLossKernel(const kernel::LiteKernel *kernel) const {
  bool isLoss = false;
  for (auto &s : cfg_.loss_name_) {
    if (kernel->name().find(s) != std::string::npos) {
      isLoss = true;
      break;
    }
  }
  return isLoss;
}

bool TrainSession::IsGradKernel(const kernel::LiteKernel *kernel) const {
  return kernel->name().find(kGradName) != std::string::npos;
}

bool TrainSession::IsOptimizer(kernel::LiteKernel *kernel) const {
  return ((kernel->type() == schema::PrimitiveType_Adam) || (kernel->type() == schema::PrimitiveType_SGD) ||
          (kernel->type() == schema::PrimitiveType_ApplyMomentum));
}

bool TrainSession::IsMaskOutput(kernel::LiteKernel *kernel) const {
  return (IsOptimizer(kernel) || (kernel->type() == schema::PrimitiveType_Assign));
}

bool TrainSession::IsBN(kernel::LiteKernel *kernel) const {
  return ((kernel->type() == schema::PrimitiveType_BatchNorm) ||
          (kernel->type() == schema::PrimitiveType_FusedBatchNorm));
}

int TrainSession::Resize(const std::vector<tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims) {
  FreeWorkSpace();
  if (tensors_data_ != nullptr) {
    free(tensors_data_);
    tensors_data_ = nullptr;
  }
  auto ret = lite::LiteSession::Resize(inputs, dims);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "train resize input failed.";
    return RET_ERROR;
  }
  ret = AllocWorkSpace();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to allocate space";
    return RET_ERROR;
  }
  ret = AllocTensors(train_kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "train alloc failed after resize.";
    return RET_ERROR;
  }
  return RET_OK;
}

int TrainSession::FindUseInTensorKernel(std::vector<kernel::LiteKernel *> *use_in_tensor_kernels,
                                        const std::vector<lite::Tensor *> &kernel_in_tensors,
                                        const std::vector<kernel::LiteKernel *> &inference_kernels) {
  for (size_t i = 0; i < inference_kernels.size(); i++) {
    for (size_t j = 0; j < kernel_in_tensors.size(); j++) {
      if (IsContain(inference_kernels[i]->out_tensors(), kernel_in_tensors[j])) {
        use_in_tensor_kernels->push_back(inference_kernels[i]);
      }
    }
  }
  return RET_OK;
}

int TrainSession::FindExportKernels(std::vector<kernel::LiteKernel *> *export_kernels,
                                    const std::vector<std::string> &export_output_tensor_names,
                                    const std::vector<kernel::LiteKernel *> &inference_kernels) {
  std::vector<std::string> all_kernel_name = {};
  std::transform(inference_kernels.begin(), inference_kernels.end(), std::back_inserter(all_kernel_name),
                 [](kernel::LiteKernel *kernel) { return kernel->name(); });
  std::queue<std::string> need_kernel_names;
  // Find the kernel name according to the tensor name
  for (auto &kernel : inference_kernels) {
    if (std::any_of(kernel->out_tensors().begin(), kernel->out_tensors().end(), [&](lite::Tensor *out_tensor) {
          return IsContain(export_output_tensor_names, out_tensor->tensor_name());
        })) {
      need_kernel_names.push(kernel->name());
    }
  }
  if (need_kernel_names.size() == 0) {
    MS_LOG(ERROR) << "can not find tensor";
    return RET_ERROR;
  }
  // find all kernel
  while (!need_kernel_names.empty()) {
    auto kernel_name = need_kernel_names.front();
    need_kernel_names.pop();
    auto it = find(all_kernel_name.begin(), all_kernel_name.end(), kernel_name);
    if (it == all_kernel_name.end()) {
      MS_LOG(ERROR) << "not find kernel name in export trained model.";
      return RET_ERROR;
    }
    auto kernel = inference_kernels[it - all_kernel_name.begin()];
    if (!IsContain(*export_kernels, kernel)) {
      export_kernels->push_back(kernel);
    }
    auto kernel_in_tensors = kernel->in_tensors();
    std::vector<kernel::LiteKernel *> use_in_tensor_kernels;
    auto status = FindUseInTensorKernel(&use_in_tensor_kernels, kernel_in_tensors, inference_kernels);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "FindUseInTensorKernel failed.";
      return RET_ERROR;
    }
    for (size_t i = 0; i < use_in_tensor_kernels.size(); i++) {
      need_kernel_names.push(use_in_tensor_kernels[i]->name());
    }
  }
  return RET_OK;
}

int TrainSession::Export(const std::string &file_name, ModelType model_type, QuantizationType quant_type,
                         FormatType format, std::vector<std::string> out_put_tensor_name) {
  MS_CHECK_FALSE_MSG(file_name.empty(), RET_ERROR, "File name cannot be empty");
  MS_CHECK_FALSE_MSG(model_type > luojianet_ms::lite::MT_INFERENCE || model_type < luojianet_ms::lite::MT_TRAIN, RET_ERROR,
                     "Export model type parameter error");
  MS_CHECK_FALSE_MSG(quant_type < luojianet_ms::lite::QT_DEFAULT || quant_type > luojianet_ms::lite::QT_WEIGHT, RET_ERROR,
                     "Export quant type parameter error");
  MS_CHECK_FALSE_MSG(format != FT_FLATBUFFERS, RET_ERROR, "File name cannot be empty");

  bool orig_train_state = IsTrain();
  Eval();
  TrainExport texport(file_name);
  int status = texport.ExportInit(model_.get()->name_, model_.get()->version_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "cannot init export";
    return status;
  }

  if (!out_put_tensor_name.empty() && model_type == MT_INFERENCE) {
    std::vector<kernel::LiteKernel *> export_kernels = {};
    status = FindExportKernels(&export_kernels, out_put_tensor_name, inference_kernels_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "FindExportKernels failed.";
      return RET_ERROR;
    }
    status = texport.ExportNet(export_kernels, tensors_, out_put_tensor_name, model_.get(), quant_type);
  } else {
    status = texport.ExportNet((model_type == MT_TRAIN) ? train_kernels_ : inference_kernels_, tensors_,
                               (model_type == MT_TRAIN) ? train_output_tensor_names_ : eval_output_tensor_names_,
                               model_.get(), quant_type);
  }

  if (status != RET_OK) {
    MS_LOG(ERROR) << "cannot export Network";
    return status;
  }
  if (model_type == MT_INFERENCE) {
    status = texport.TrainModelDrop();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "TrainModelDrop failed.";
      return status;
    }
    status = texport.TrainModelFusion();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "TrainModelFusion failed.";
      return status;
    }
  }
  status = texport.SaveToFile();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "failed to save to " << file_name;
    return status;
  }
  if (orig_train_state) Train();
  return status;
}
std::vector<tensor::MSTensor *> TrainSession::GetFeatureMaps() const {
  std::vector<tensor::MSTensor *> features;
  for (auto cur_tensor : this->tensors_) {
    if (cur_tensor->IsConst() && cur_tensor->data_type() == kNumberTypeFloat32) {
      features.push_back(cur_tensor);
    }
  }
  return features;
}

int TrainSession::UpdateFeatureMaps(const std::vector<tensor::MSTensor *> &features_map) {
  for (auto feature : features_map) {
    bool find = false;
    for (auto tensor : tensors_) {
      if (!tensor->IsConst() || tensor->data_type() != kNumberTypeFloat32) {
        continue;
      }
      if (feature->tensor_name() != tensor->tensor_name()) {
        continue;
      }
      if (feature->Size() != tensor->Size()) {
        MS_LOG(ERROR) << "feature name:" << feature->tensor_name() << ",len diff:"
                      << "old is:" << tensor->Size() << "new is:" << feature->Size();
        return RET_ERROR;
      }
      find = true;
      memcpy(tensor->data(), feature->data(), tensor->Size());
    }
    if (!find) {
      MS_LOG(ERROR) << "cannot find feature:" << feature->tensor_name() << ",update failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

std::set<schema::PrimitiveType> inPlaceSupportedKernels = {
  luojianet_ms::schema::PrimitiveType_Activation, luojianet_ms::schema::PrimitiveType_ActivationGrad,
  luojianet_ms::schema::PrimitiveType_Reshape,    luojianet_ms::schema::PrimitiveType_DivFusion,
  luojianet_ms::schema::PrimitiveType_AddFusion,  luojianet_ms::schema::PrimitiveType_SubFusion,
  luojianet_ms::schema::PrimitiveType_RealDiv,    luojianet_ms::schema::PrimitiveType_Select,
  luojianet_ms::schema::PrimitiveType_BiasAdd,    luojianet_ms::schema::PrimitiveType_BiasAddGrad,
  luojianet_ms::schema::PrimitiveType_Sqrt,       luojianet_ms::schema::PrimitiveType_Abs,
  luojianet_ms::schema::PrimitiveType_Cos,        luojianet_ms::schema::PrimitiveType_Log,
  luojianet_ms::schema::PrimitiveType_Square,     luojianet_ms::schema::PrimitiveType_Rsqrt,
  luojianet_ms::schema::PrimitiveType_Sin,        luojianet_ms::schema::PrimitiveType_LogicalNot,
  luojianet_ms::schema::PrimitiveType_LogicalAnd, luojianet_ms::schema::PrimitiveType_LogicalOr,
  luojianet_ms::schema::PrimitiveType_Floor,      luojianet_ms::schema::PrimitiveType_Ceil,
  luojianet_ms::schema::PrimitiveType_Round,      luojianet_ms::schema::PrimitiveType_Neg,
  luojianet_ms::schema::PrimitiveType_Reciprocal, luojianet_ms::schema::PrimitiveType_Erf,
  luojianet_ms::schema::PrimitiveType_Maximum,    luojianet_ms::schema::PrimitiveType_Minimum,
  luojianet_ms::schema::PrimitiveType_FloorDiv,   luojianet_ms::schema::PrimitiveType_FloorMod,
  luojianet_ms::schema::PrimitiveType_Eltwise,    luojianet_ms::schema::PrimitiveType_SquaredDifference,
  luojianet_ms::schema::PrimitiveType_ExpandDims, luojianet_ms::schema::PrimitiveType_Cast,
  luojianet_ms::schema::PrimitiveType_Flatten,    luojianet_ms::schema::PrimitiveType_FlattenGrad,
  luojianet_ms::schema::PrimitiveType_Squeeze,    luojianet_ms::schema::PrimitiveType_Unsqueeze};

bool TrainSession::IsInPlaceKernel(kernel::LiteKernel *kernel) {
  if (inPlaceSupportedKernels.find(kernel->type()) != inPlaceSupportedKernels.end() &&
      !(kernel->type() == luojianet_ms::schema::PrimitiveType_Activation &&
        kernel->op_parameter()->type_ == schema::ActivationType_SIGMOID)) {
    return true;
  }
  return false;
}

bool TrainSession::IsInPlaceTensor(kernel::LiteKernel *kernel, uint32_t idx,
                                   const std::unordered_map<lite::Tensor *, int> &ref_count, uint32_t *input_idx) {
  if (IsInPlaceKernel(kernel)) {
    auto out_tensor = kernel->out_tensors().at(idx);
    for (size_t i = 0; i < kernel->in_tensors().size(); i++) {
      auto tensor = kernel->in_tensors().at(i);
      if ((tensor->category() == lite::Category::VAR) && (ref_count.find(tensor) != ref_count.end()) &&
          (tensor->init_ref_count() == 1 || (tensor->init_ref_count() > 1 && ref_count.at(tensor) == 1)) &&
          (out_tensor->Size() == tensor->Size())) {
        *input_idx = static_cast<uint32_t>(i);
        return true;
      }
    }
  }
  return false;
}

size_t TrainSession::GetInplaceTensorOffset(kernel::LiteKernel *kernel,
                                            const std::unordered_map<lite::Tensor *, size_t> &offset_map,
                                            std::unordered_map<lite::Tensor *, int> *ref_count, uint32_t input_idx) {
  auto tensor = kernel->in_tensors().at(input_idx);
  ref_count->at(tensor) = ref_count->at(tensor) + 1;
  return offset_map.at(tensor);
}
}  // namespace lite

session::LiteSession *session::TrainSession::CreateTrainSession(const std::string &fn, const lite::Context *context,
                                                                bool train_mode, const lite::TrainCfg *cfg) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context cannot be nullptr";
    return nullptr;
  }
  auto session = std::make_unique<lite::TrainSession>();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return nullptr;
  }
  if (context->allocator == nullptr) {
    const_cast<lite::Context *>(context)->allocator = std::make_shared<StaticAllocator>();
    if (context->allocator == nullptr) {
      MS_LOG(ERROR) << " cannot convert to static allocation";
    }
  }

  auto *inner_context = new (std::nothrow) luojianet_ms::lite::InnerContext(context);
  auto ret = session->TrainInit(inner_context, cfg);
  if (ret != luojianet_ms::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    return nullptr;
  }

  std::string filename = fn;
  if (filename.substr(filename.find_last_of(".") + 1) != "ms") {
    filename = filename + ".ms";
  }

  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(filename.c_str()));
  if (model == nullptr) {
    MS_LOG(ERROR) << "create model for train session failed " << filename;
    return nullptr;
  }

  ret = session->CompileTrainGraph(model);
  if (ret != luojianet_ms::lite::RET_OK) {
    MS_LOG(ERROR) << "Compiling Train Graph session failed";
    return nullptr;
  }

  if (train_mode) {
    ret = session->Train();
  } else {
    ret = session->Eval();
  }
  if (ret != luojianet_ms::lite::RET_OK) {
    MS_LOG(ERROR) << "Could not switch to Train Modei " << train_mode;
    return nullptr;
  }
  return session.release();
}
}  // namespace luojianet_ms
