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
#include "tools/converter/micro/coder/opcoders/op_coder_builder.h"
#include <vector>
#include <memory>
#include "tools/converter/micro/coder/allocator/allocator.h"
#include "src/common/prim_util.h"
#include "src/common/version_manager.h"
#include "src/ops/populate/populate_register.h"
#include "tools/converter/micro/coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
std::unique_ptr<OperatorCoder> OpCoderBuilder::build(int schema_version) {
  MS_CHECK_PTR_RET_NULL(node_->primitive_);
  int primitive_type = GetPrimitiveType(node_->primitive_, schema_version);
  CoderKey coder_key(target_, data_type_, primitive_type);
  CoderCreatorFunc creator_func = OpCoderFactory::GetInstance()->FindOpCoder(coder_key);
  if (creator_func == nullptr) {
    MS_LOG(ERROR) << "caught unsupported layer: " << node_->name_;
    return nullptr;
  }
  if (inputs_.empty() || outputs_.empty()) {
    MS_LOG(ERROR) << "input tensors or output tensors are empty";
    return nullptr;
  } else {
    MS_CHECK_PTR_RET_NULL(inputs_.at(kInputIndex));
    MS_CHECK_PTR_RET_NULL(outputs_.at(kOutputIndex));
  }
  std::unique_ptr<OperatorCoder> op_coder =
    creator_func(inputs_, outputs_, node_, node_index_++, target_, schema_version);
  if (op_coder == nullptr) {
    MS_LOG(ERROR) << "create op_coder failed: " << node_->name_ << " primitive type: "
                  << mindspore::schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(primitive_type))
                  << " code_target: " << target_ << " data_type: " << EnumNameDataType(data_type_);
    return nullptr;
  }
  op_coder->set_input_tensor_indices(input_indices_);
  op_coder->set_output_tensor_indices(output_indices_);
  int thread_num = support_parallel_ ? kMaxThreadNumSupported : 1;
  if ((primitive_type == schema::PrimitiveType_FullConnection) && data_type_ == kNumberTypeFloat32) {
    thread_num = 1;  // cur cannot support parallel run
  }
  op_coder->set_thread_num(thread_num);
  if (primitive_type != schema::PrimitiveType_Custom) {
    parameter_->thread_num_ = thread_num;
  }
  op_coder->set_parameter(parameter_);
  op_coder->set_type(primitive_type);
  return op_coder;
}

OpCoderBuilder &OpCoderBuilder::inputs(const std::vector<Tensor *> &inputs) {
  this->inputs_ = inputs;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::outputs(const std::vector<Tensor *> &outputs) {
  this->outputs_ = outputs;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::node(const Model::Node *node) {
  this->node_ = node;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::parameter(OpParameter *parameter) {
  this->parameter_ = parameter;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::data_type(TypeId data_type) {
  this->data_type_ = data_type;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::mode(CodeMode mode) {
  this->mode_ = mode;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::input_indices(const std::vector<uint32_t> &indices) {
  this->input_indices_ = indices;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::output_indices(const std::vector<uint32_t> &indices) {
  this->output_indices_ = indices;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::target(Target target) {
  this->target_ = target;
  return *this;
}

OpCoderBuilder &OpCoderBuilder::support_parallel(bool parallel) {
  support_parallel_ = parallel;
  return *this;
}

void OpCoderBuilder::Reset() {}
}  // namespace mindspore::lite::micro
