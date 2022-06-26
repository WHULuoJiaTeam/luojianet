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
#include "src/ops/compat/attr_transfer_common.h"

namespace mindspore {
namespace lite {
int TransferFillToAttr(Model::Node *node, std::vector<schema::Tensor *> *dst_tensors,
                       std::vector<char *> *const tensor_bufs) {
  if (node == nullptr || node->primitive_ == nullptr || dst_tensors == nullptr || tensor_bufs == nullptr) {
    MS_LOG(ERROR) << "the parameter of this function is nullptr.";
    return RET_ERROR;
  }
  if (node->input_indices_.size() != 1) {
    MS_LOG(DEBUG) << "fill don't need to convert attr to tensor.";
    return RET_OK;
  }
  dst_tensors->clear();
  auto prim = reinterpret_cast<const schema::v0::Primitive *>(node->primitive_);
  MS_ASSERT(prim != nullptr);
  auto param = prim->value_as_Fill();
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return RET_ERROR;
  }
  auto dims_attr = param->dims();
  if (dims_attr == nullptr) {
    MS_LOG(ERROR) << "dims_attr is nullptr";
    return RET_ERROR;
  }
  std::vector<int> dims = std::vector<int>(dims_attr->begin(), dims_attr->end());
  auto dims_tensor = AttrToTensor(dims.data(), dims.size(), true, kNumberTypeInt32, tensor_bufs);
  if (dims_tensor == nullptr) {
    MS_LOG(ERROR) << "attr tensor is nullptr, transform is failed.";
    return RET_NULL_PTR;
  }
  dst_tensors->push_back(dims_tensor);
  return RET_OK;
}

Register FillTransferRegistry(SCHEMA_VERSION::SCHEMA_V0, schema::v0::PrimitiveType_Fill, TransferFillToAttr);
}  // namespace lite
}  // namespace mindspore
