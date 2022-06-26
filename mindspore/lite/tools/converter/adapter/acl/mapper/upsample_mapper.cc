/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/upsample_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kUpsample = "Upsample";
constexpr size_t kScaleMinNum = 2;
constexpr size_t kInputNum = 3;
}  // namespace

STATUS UpsampleMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  if (cnode->inputs().size() != kInputNum) {
    MS_LOG(ERROR) << "Upsample input num should be " << kInputNum << ", real size: " << cnode->inputs().size();
    return RET_ERROR;
  }
  TypeId type_id;
  if (opt::GetDataTypeFromAnfNode(cnode->inputs()[kInputNum - 1], &type_id) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed.";
    return RET_ERROR;
  }
  if (type_id == kNumberTypeFloat32) {
    if (AttrAdjust(src_prim, value_node) != RET_OK) {
      MS_LOG(ERROR) << "Upsample attr adjust failed.";
      return RET_ERROR;
    }
    if (RemoveConstInput(cnode) != RET_OK) {
      MS_LOG(ERROR) << "Upsample remove const input failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS UpsampleMapper::AttrAdjust(const PrimitivePtr &src_prim, const ValueNodePtr &val_node) {
  MS_CHECK_TRUE_MSG(src_prim != nullptr, RET_ERROR, "src_prim is nullptr.");
  MS_CHECK_TRUE_MSG(val_node != nullptr, RET_ERROR, "val_node is nullptr.");
  auto attr_val = src_prim->GetAttr("scale");
  CHECK_NULL_RETURN(attr_val);
  std::vector<float> scale = opt::CastToFloat(attr_val);
  if (scale.size() < kScaleMinNum) {
    MS_LOG(ERROR) << "Scale size must not be less than " << kScaleMinNum << ", real size: " << scale.size();
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "The scale value: " << scale[1];
  auto dst_prim = std::make_shared<acl::Upsample>();
  CHECK_NULL_RETURN(dst_prim);
  float attr_scale = 1;
  dst_prim->AddAttr("scale", MakeValue(attr_scale));
  int64_t stride_h = static_cast<int64_t>(scale[1]);
  int64_t stride_w = stride_h;
  dst_prim->AddAttr("stride_h", MakeValue(stride_h));
  dst_prim->AddAttr("stride_w", MakeValue(stride_w));
  val_node->set_value(dst_prim);
  return RET_OK;
}

STATUS UpsampleMapper::RemoveConstInput(const CNodePtr &cnode) {
  std::vector<AnfNodePtr> inputs{cnode->inputs().begin(), cnode->inputs().end() - 1};
  cnode->set_inputs(inputs);
  auto redundant_input = cnode->inputs()[kInputNum - 1];
  auto graph = cnode->func_graph();
  CHECK_NULL_RETURN(graph);
  graph->DropNode(redundant_input);
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kUpsample, UpsampleMapper)
}  // namespace lite
}  // namespace mindspore
