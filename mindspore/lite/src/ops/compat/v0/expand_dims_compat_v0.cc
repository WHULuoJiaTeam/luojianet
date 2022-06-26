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
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
int TransferExpandDimsAttr(Model::Node *node, std::vector<schema::Tensor *> *dst_tensors,
                           std::vector<char *> *const tensor_bufs) {
  if (node == nullptr || node->primitive_ == nullptr || dst_tensors == nullptr || tensor_bufs == nullptr) {
    MS_LOG(ERROR) << "the parameter of this function is nullptr.";
    return RET_ERROR;
  }
  MS_CHECK_TRUE_RET(node->input_indices_.size() == 1, RET_ERROR);
  MS_CHECK_TRUE_RET(dst_tensors->size() == 0, RET_ERROR);

  auto prim = reinterpret_cast<const schema::v0::Primitive *>(node->primitive_);
  MS_ASSERT(prim != nullptr);
  auto param = prim->value_as_ExpandDims();
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return RET_ERROR;
  }
  int32_t dim = param->dim();
  auto dim_tensor = AttrToTensor(&dim, 1, false, kNumberTypeInt32, tensor_bufs);
  if (dim_tensor == nullptr) {
    MS_LOG(ERROR) << "transfer expand dim tensor failed.";
    return RET_NULL_PTR;
  }
  dst_tensors->push_back(dim_tensor);
  return RET_OK;
}

Register ExpandDimsTransferRegistry(SCHEMA_VERSION::SCHEMA_V0, schema::v0::PrimitiveType_ExpandDims,
                                    TransferExpandDimsAttr);
}  // namespace lite
}  // namespace mindspore
