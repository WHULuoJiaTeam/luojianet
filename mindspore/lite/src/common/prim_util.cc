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

#include "src/common/prim_util.h"
#include <set>
#include "nnacl/op_base.h"
#include "schema/model_generated.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#ifdef ENABLE_V0
#include "schema/model_v0_generated.h"
#endif

namespace mindspore {
namespace lite {
static std::set<schema::PrimitiveType> tensor_list_ops = {
  schema::PrimitiveType_TensorListFromTensor, schema::PrimitiveType_TensorListGetItem,
  schema::PrimitiveType_TensorListReserve, schema::PrimitiveType_TensorListSetItem,
  schema::PrimitiveType_TensorListStack};
#ifdef ENABLE_V0
static std::set<schema::v0::PrimitiveType> v0_tensor_list_ops = {
  schema::v0::PrimitiveType_TensorListFromTensor, schema::v0::PrimitiveType_TensorListGetItem,
  schema::v0::PrimitiveType_TensorListReserve, schema::v0::PrimitiveType_TensorListSetItem,
  schema::v0::PrimitiveType_TensorListStack};
#endif

int GetPrimitiveType(const void *primitive, int schema_version) {
  if (primitive == nullptr) {
    return -1;
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    return static_cast<const schema::v0::Primitive *>(primitive)->value_type();
  }
#endif
  return static_cast<const schema::Primitive *>(primitive)->value_type();
}

const char *GetPrimitiveTypeName(const void *primitive, int schema_version) {
  if (primitive == nullptr) {
    return "NONE";
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    return schema::v0::EnumNamePrimitiveType(static_cast<const schema::v0::Primitive *>(primitive)->value_type());
  }
#endif
  return schema::EnumNamePrimitiveType(static_cast<const schema::Primitive *>(primitive)->value_type());
}

const char *PrimitiveCurVersionTypeName(int type) {
  return schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type));
}

int GenPrimVersionKey(int primitive_type, int schema_version) { return primitive_type * 1000 + schema_version; }

bool IsPartialNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_PartialFusion;
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    return reinterpret_cast<const schema::v0::Primitive *>(primitive)->value_type() ==
           schema::v0::PrimitiveType_Partial;
  }
#endif
  return false;
}

bool IsCallNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_Call;
  }
  return false;
}

bool IsSwitchNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_Switch;
  }
  return false;
}

bool IsSwitchLayerNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_SwitchLayer;
  }
  return false;
}

bool IsCustomNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    return reinterpret_cast<const schema::Primitive *>(primitive)->value_type() == schema::PrimitiveType_Custom;
  }
  return false;
}

bool IsTensorListNode(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primtive cannot be nullptr");
  if (schema_version == SCHEMA_CUR) {
    if (tensor_list_ops.find(reinterpret_cast<const schema::Primitive *>(primitive)->value_type()) !=
        tensor_list_ops.end()) {
      return true;
    }
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    if (v0_tensor_list_ops.find(reinterpret_cast<const schema::v0::Primitive *>(primitive)->value_type()) !=
        v0_tensor_list_ops.end()) {
      return true;
    }
  }
#endif
  return false;
}

int GetPartialGraphIndex(const void *primitive, int schema_version) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, -1, "primtive cannot be nullptr");
  int index = -1;
  if (schema_version == SCHEMA_CUR) {
    auto partial_fusion = reinterpret_cast<const schema::Primitive *>(primitive)->value_as_PartialFusion();
    if (partial_fusion == nullptr) {
      return -1;
    }
    index = partial_fusion->sub_graph_index();
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_V0) {
    auto partial = reinterpret_cast<const schema::v0::Primitive *>(primitive)->value_as_Partial();
    if (partial == nullptr) {
      return -1;
    }
    index = partial->subGraphIndex();
  }
#endif
  return index;
}
}  // namespace lite
}  // namespace mindspore
