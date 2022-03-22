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

#include "external/graph/inference_context.h"
#include "debug/ge_util.h"
#include "debug/ge_log.h"
#include "graph/ge_context.h"
#include "graph/resource_context_mgr.h"

namespace ge {
class ShapeAndTypeImpl {
 public:
  ShapeAndTypeImpl() = default;
  ~ShapeAndTypeImpl() = default;

  ShapeAndTypeImpl(const Shape &shape, const DataType data_type) : shape_(shape), data_type_(data_type) {}

  Shape shape_;
  DataType data_type_ = DT_UNDEFINED;
};

struct InnerInferenceContext {
  // For deliver to op in pair, help to support dynamic shape
  std::vector<std::string> marks;
  std::vector<std::vector<ShapeAndType>> input_handle_shapes_and_types;
  std::vector<std::vector<ShapeAndType>> output_handle_shapes_and_types;
  // For write op , if reousce changed, push to here
  std::set<AscendString> changed_resource_keys;
  // For read op, register relied resource
  std::set<AscendString> relied_resource_keys;
  ResourceContextMgr *resource_context_mgr = nullptr;
};

ShapeAndType::ShapeAndType() { shape_and_type_impl_ = ComGraphMakeShared<ShapeAndTypeImpl>(); }

ShapeAndType::ShapeAndType(const Shape &shape, DataType data_type) {
  shape_and_type_impl_ = ComGraphMakeShared<ShapeAndTypeImpl>(shape, data_type);
}

void ShapeAndType::SetShape(const Shape &shape) {
  if (shape_and_type_impl_ != nullptr) {
    shape_and_type_impl_->shape_ = shape;
  }
}

void ShapeAndType::SetType(DataType data_type) {
  if (shape_and_type_impl_ != nullptr) {
    shape_and_type_impl_->data_type_ = data_type;
  }
}

Shape ShapeAndType::GetShape() const {
  if (shape_and_type_impl_ != nullptr) {
    return shape_and_type_impl_->shape_;
  }
  return Shape();
}

DataType ShapeAndType::GetDataType() const {
  if (shape_and_type_impl_ != nullptr) {
    return shape_and_type_impl_->data_type_;
  }
  return DT_UNDEFINED;
}

InferenceContext::InferenceContext(std::unique_ptr<InnerInferenceContext> &inner_context) {
  inner_inference_context_ = std::move(inner_context);
}

std::unique_ptr<InferenceContext> InferenceContext::Create(void *resource_context_mgr) {
  std::unique_ptr<InnerInferenceContext> inner_context =
      std::unique_ptr<InnerInferenceContext>(new (std::nothrow) InnerInferenceContext());
  if (inner_context == nullptr) {
    return nullptr;
  }
  inner_context->resource_context_mgr = reinterpret_cast<ResourceContextMgr *>(resource_context_mgr);

  return std::unique_ptr<InferenceContext>(new (std::nothrow) InferenceContext(inner_context));
}

void InferenceContext::SetInputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types) {
  inner_inference_context_->input_handle_shapes_and_types.swap(shapes_and_types);
}

const std::vector<std::vector<ShapeAndType>> &InferenceContext::GetInputHandleShapesAndTypes() const {
  return inner_inference_context_->input_handle_shapes_and_types;
}

const std::vector<std::vector<ShapeAndType>> &InferenceContext::GetOutputHandleShapesAndTypes() const {
  return inner_inference_context_->output_handle_shapes_and_types;
}

void InferenceContext::SetOutputHandleShapesAndTypes(const std::vector<std::vector<ShapeAndType>> &shapes_and_types) {
  inner_inference_context_->output_handle_shapes_and_types = shapes_and_types;
}

void InferenceContext::SetOutputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types) {
  inner_inference_context_->output_handle_shapes_and_types.swap(shapes_and_types);
}

void InferenceContext::SetMarks(const std::vector<std::string> &marks) { inner_inference_context_->marks = marks; }

void InferenceContext::SetMarks(const std::vector<AscendString> &marks) {
  std::vector<std::string> impl_marks;
  for (const auto &mark : marks) {
    if (mark.GetString() != nullptr) {
      impl_marks.emplace_back(mark.GetString());
    }
  }
  inner_inference_context_->marks = impl_marks;
}

const std::vector<std::string> &InferenceContext::GetMarks() const { return inner_inference_context_->marks; }

void InferenceContext::GetMarks(std::vector<AscendString> &marks) const {
  for (auto &str_mark : inner_inference_context_->marks) {
    marks.emplace_back(str_mark.c_str());
  }
}

ResourceContext *InferenceContext::GetResourceContext(const ge::AscendString &key) {
  if (inner_inference_context_->resource_context_mgr == nullptr) {
    return nullptr;
  }
  return inner_inference_context_->resource_context_mgr->GetResourceContext(key.GetString());
}

graphStatus InferenceContext::SetResourceContext(const ge::AscendString &key, ResourceContext *resource_context) {
  if (key.GetString() == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "Resource key is null, invalid param.");
    return GRAPH_PARAM_INVALID;
  }
  if (inner_inference_context_->resource_context_mgr == nullptr) {
    GELOGE(GRAPH_FAILED, "No resource context mgr exist, resource context can not deliver in graph."
                         "Please check session initialized success or not.");
    return GRAPH_FAILED;
  }
  (void)inner_inference_context_->resource_context_mgr->SetResourceContext(key.GetString(), resource_context);
  return GRAPH_SUCCESS;
}

graphStatus InferenceContext::AddChangedResourceKey(const ge::AscendString &key) {
  if (key.GetString() == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "Resource key is null, invalid param.");
    return GRAPH_PARAM_INVALID;
  }
  (void)inner_inference_context_->changed_resource_keys.insert(key.GetString());
  return GRAPH_SUCCESS;
}

void InferenceContext::ClearChangedResourceKeys() {
  inner_inference_context_->changed_resource_keys.clear();
}

const std::set<ge::AscendString> &InferenceContext::GetChangedResourceKeys() const {
  return inner_inference_context_->changed_resource_keys;
}

graphStatus InferenceContext::RegisterReliedOnResourceKey(const ge::AscendString &key) {
  if (key.GetString() == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "Resource key is null, invalid param.");
    return GRAPH_PARAM_INVALID;
  }
  (void)inner_inference_context_->relied_resource_keys.insert(key.GetString());
  return GRAPH_SUCCESS;
}

const std::set<ge::AscendString> &InferenceContext::GetReliedOnResourceKeys() const {
  return inner_inference_context_->relied_resource_keys;
}
}  // namespace ge
