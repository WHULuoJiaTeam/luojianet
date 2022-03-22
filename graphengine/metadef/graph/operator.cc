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

#include <cstdint>
#include <algorithm>
#include <mutex>
#include <queue>
#include <set>
#include "external/graph/operator_factory.h"
#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "external/graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/constant_utils.h"
#include "tensor_type_impl.h"
#include "op_io.h"
#include "operator_impl.h"
#include "external/graph/operator.h"

using std::enable_shared_from_this;
using std::make_pair;
using std::shared_ptr;
using std::to_string;

#define OP_ATTR_SET_IMP(ArgType, AttrUtilsFun)                                                                         \
  Operator &ge::Operator::SetAttr(const std::string &name, ArgType attr_value) {                                       \
    if ((operator_impl_ == nullptr) || (operator_impl_->GetOpDescImpl() == nullptr)) {                                 \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");                   \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());                         \
      return *this;                                                                                                    \
    }                                                                                                                  \
    if (!ge::AttrUtils::Set##AttrUtilsFun(operator_impl_->GetOpDescImpl(), name, attr_value)) {                        \
      GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());                                                     \
    }                                                                                                                  \
    return *this;                                                                                                      \
  }                                                                                                                    \
  Operator &ge::Operator::SetAttr(const char_t *name, ArgType attr_value) {                                            \
    if (name == nullptr) {                                                                                             \
      REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");                                           \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator attr name is nullptr.");                                           \
      return *this;                                                                                                    \
    }                                                                                                                  \
    const std::string op_name = name;                                                                                  \
    if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {                                     \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");                  \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", op_name.c_str());                      \
      return *this;                                                                                                    \
    }                                                                                                                  \
    if (!ge::AttrUtils::Set##AttrUtilsFun(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {                     \
      GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());                                                  \
    }                                                                                                                  \
    return *this;                                                                                                      \
  }

#define OP_ATTR_GET_IMP(ArgType, AttrUtilsFun)                                                                         \
  graphStatus ge::Operator::GetAttr(const std::string &name, ArgType attr_value) const {                               \
    if ((operator_impl_ == nullptr) || (operator_impl_->GetOpDescImpl() == nullptr)) {                                 \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");                   \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());                         \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    if (!ge::AttrUtils::Get##AttrUtilsFun(operator_impl_->GetOpDescImpl(), name, attr_value)) {                        \
      GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());                                                     \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    return GRAPH_SUCCESS;                                                                                              \
  }                                                                                                                    \
  graphStatus ge::Operator::GetAttr(const char_t *name, ArgType attr_value) const {                                    \
    if (name == nullptr) {                                                                                             \
      REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");                                            \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator attr name is nullptr.");                                           \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    if ((operator_impl_ == nullptr) || (operator_impl_->GetOpDescImpl() == nullptr)) {                                 \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");                   \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name);                                 \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    const std::string op_name = name;                                                                                  \
    if (!ge::AttrUtils::Get##AttrUtilsFun(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {                     \
      GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());                                                  \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    return GRAPH_SUCCESS;                                                                                              \
  }

#define OP_ATTR_REG_IMP(ArgType, AttrUtilsFun)                                                                         \
  void ge::Operator::AttrRegister(const std::string &name, ArgType attr_value) {                                       \
    if ((operator_impl_ == nullptr) || (operator_impl_->GetOpDescImpl() == nullptr)) {                                 \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");                   \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());                         \
      return;                                                                                                          \
    }                                                                                                                  \
    if (!ge::AttrUtils::Set##AttrUtilsFun(operator_impl_->GetOpDescImpl(), name, attr_value)) {                        \
      GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());                                                \
    }                                                                                                                  \
  }

namespace ge {
TensorType::TensorType(DataType dt) {
  tensor_type_impl_ = ComGraphMakeShared<TensorTypeImpl>();
  if (tensor_type_impl_ != nullptr) {
    tensor_type_impl_->dt_vec_.push_back(dt);
  }
}

TensorType::TensorType(const std::initializer_list<DataType> &initial_types) {
  tensor_type_impl_ = ComGraphMakeShared<TensorTypeImpl>();
  if (tensor_type_impl_ != nullptr) {
    tensor_type_impl_->dt_vec_ = initial_types;
  }
}

// Used to manage OperatorImpl instances created by ge api.
class OperatorKeeper {
public:
  static OperatorKeeper &GetInstance() {
    static OperatorKeeper instance;
    return instance;
  }
  void CheckInOperator(const OperatorImplPtr &op_impl) {
    if (op_impl) {
      const std::lock_guard<std::mutex> lock(mutex_);
      (void)(operators_.insert(op_impl));
    }
  }
  void CheckOutOperator(const OperatorImplPtr &op_impl) {
    if (op_impl) {
      const std::lock_guard<std::mutex> lock(mutex_);
      (void)(operators_.erase(op_impl));
    }
  }

private:
  OperatorKeeper() = default;
  ~OperatorKeeper() {
    for (const auto &iter : operators_) {
      if (iter) {
        iter->ClearInputLinks();
        iter->ClearOutputLinks();
      }
    }
  }
  std::set<OperatorImplPtr> operators_;
  std::mutex mutex_;
};

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator OpDescUtils::CreateOperatorFromNode(ge::ConstNodePtr node_ptr) {
  const ge::OperatorImplPtr operator_impl_ptr = ComGraphMakeShared<OperatorImpl>(node_ptr);
  if (operator_impl_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
    GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
    return Operator("default");
  }
  return operator_impl_ptr->ToOperator();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::CopyOperators(const ComputeGraphPtr &dst_compute_graph,
                           const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                           const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                           const std::map<std::string, ge::Operator> &src_op_list,
                           std::map<std::string, ge::Operator> &dst_op_list) {
  GE_CHECK_NOTNULL(dst_compute_graph);

  std::map<OperatorImplPtr, NodePtr> all_node_info;

  for (const auto &itr : src_op_list) {
    auto name = itr.first;
    const ge::Operator &src_op = itr.second;
    GE_CHECK_NOTNULL(src_op.operator_impl_);
    const OperatorImplPtr scr_op_impl_ptr = src_op.operator_impl_;
    GE_CHECK_NOTNULL(scr_op_impl_ptr->op_desc_);
    ge::Operator dst_op;
    OpDescPtr dst_op_desc = nullptr;
    if (scr_op_impl_ptr->node_ == nullptr) {
      // cannot find op_desc in compute graph, need creat new op_desc
      // otherwise use existing op_desc
      const auto it = op_desc_old_2_new.find(scr_op_impl_ptr->op_desc_);
      if (it != op_desc_old_2_new.end()) {
        dst_op_desc = it->second;
      } else {
        dst_op_desc = AttrUtils::CopyOpDesc(scr_op_impl_ptr->op_desc_);
        if (dst_op_desc == nullptr) {
          REPORT_CALL_ERROR("E19999", "CopyOpDesc from %s failed", scr_op_impl_ptr->op_desc_->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Copy][OpDesc] from %s failed", scr_op_impl_ptr->op_desc_->GetName().c_str());
          return GRAPH_FAILED;
        }
        dst_op_desc->CopyAttrsFrom(*scr_op_impl_ptr->op_desc_);
        dst_op_desc->SetName(scr_op_impl_ptr->op_desc_->GetName());
      }
      dst_op = CreateOperatorFromOpDesc(dst_op_desc);
    } else {
      const auto original_op_desc = scr_op_impl_ptr->node_->GetOpDesc();
      if (scr_op_impl_ptr->op_desc_ != original_op_desc) {
        REPORT_INNER_ERROR("E19999", "node and op_desc of operator are not equal.");
        GELOGE(GRAPH_FAILED, "[Check][Param] node and op_desc of operator are not equal.");
        return GRAPH_FAILED;
      }
      NodePtr dst_node = nullptr;
      // cannot find node in compute graph, need creat new node
      // otherwise use existing node and op_desc
      const auto it = node_old_2_new.find(scr_op_impl_ptr->node_);
      if (it != node_old_2_new.end()) {
        dst_node = it->second;
      } else {
        dst_op_desc = AttrUtils::CopyOpDesc(scr_op_impl_ptr->op_desc_);
        if (dst_op_desc == nullptr) {
          REPORT_CALL_ERROR("E19999", "CopyOpDesc from %s failed", scr_op_impl_ptr->op_desc_->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Copy][OpDesc] from %s failed", scr_op_impl_ptr->op_desc_->GetName().c_str());
          return GRAPH_FAILED;
        }
        dst_op_desc->CopyAttrsFrom(*scr_op_impl_ptr->op_desc_);
        dst_op_desc->SetName(scr_op_impl_ptr->op_desc_->GetName());
        dst_node = NodeUtils::CreatNodeWithoutGraph(dst_op_desc);
        GE_CHECK_NOTNULL(dst_node);
        // to do link egdes
      }
      dst_op = CreateOperatorFromNode(dst_node);
      (void)(all_node_info.emplace(dst_op.GetOperatorImplPtr(), dst_node));
    }
    dst_op.operator_impl_->subgraph_names_to_builders_ = src_op.operator_impl_->subgraph_names_to_builders_;
    (void)(dst_op_list.emplace(name, dst_op));
  }

  dst_compute_graph->SetAllNodesInfo(all_node_info);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::CopyOperatorLinks(const std::map<std::string, ge::Operator> &src_op_list,
                               std::map<std::string, ge::Operator> &dst_op_list) {
  for (const auto &it : src_op_list) {
    auto &src_op = it.second;
    const auto op_name = it.first;
    auto &dst_op = dst_op_list[op_name];
    const OperatorImplPtr src_impl_ptr = src_op.GetOperatorImplPtr();
    const OperatorImplPtr dst_impl_ptr = dst_op.GetOperatorImplPtr();
    for (const auto &itr : src_impl_ptr->input_link_) {
      const std::string dst_name = itr.first;
      const OpIO &op_io = itr.second;
      const OperatorImplPtr input_impl_ptr = op_io.GetOwner();
      GE_CHECK_NOTNULL(input_impl_ptr);
      const auto iter = dst_op_list.find(input_impl_ptr->GetName());
      if (iter == dst_op_list.end()) {
        REPORT_INNER_ERROR("E19999", "Find dst operator:%s failed", input_impl_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] Find dst operator:%s failed", input_impl_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      auto &input_op = iter->second;
      (void)(dst_op.SetInput(dst_name, input_op));
    }

    for (const auto &itr : src_impl_ptr->control_input_link_) {
      const OperatorImplPtr input_ctrl_impl_ptr = itr.lock();
      GE_CHECK_NOTNULL(input_ctrl_impl_ptr);
      const auto iter = dst_op_list.find(input_ctrl_impl_ptr->GetName());
      if (iter == dst_op_list.end()) {
        REPORT_INNER_ERROR("E19999", "Find dst ctrl operator:%s failed", input_ctrl_impl_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] Find dst ctrl operator:%s failed", input_ctrl_impl_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      auto &ctrl_input_op = iter->second;
      (void)(dst_op.AddControlInput(ctrl_input_op));
    }
  }
  return GRAPH_SUCCESS;
}

Operator::Operator(const std::string &type) {
  static uint32_t index = 0U;
  std::string name = type + "_" + std::to_string(index++);
  operator_impl_ = ComGraphMakeShared<OperatorImpl>(name, type);
  if (operator_impl_ == nullptr) {
    GELOGW("[Check][Param] Make OperatorImpl failed");
  }
  OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
}

Operator::Operator(const char_t *type) {
  if (type != nullptr) {
    std::string op_type = type;
    static uint32_t index = 0U;
    std::string name = op_type + "_" + std::to_string(index++);
    operator_impl_ = ComGraphMakeShared<OperatorImpl>(name, op_type);
    if (operator_impl_ == nullptr) {
      GELOGW("[Check][Param] Make OperatorImpl failed");
    }
    OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
  } else {
    GELOGW("[Check][Param] Operator type is nullptr");
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator OpDescUtils::CreateOperatorFromOpDesc(OpDescPtr op_desc) {
  std::shared_ptr<OperatorImpl> operator_impl_ptr;
  operator_impl_ptr = ComGraphMakeShared<OperatorImpl>(op_desc);
  if (operator_impl_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
    GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
    return Operator("default");
  }
  OperatorKeeper::GetInstance().CheckInOperator(operator_impl_ptr);
  return operator_impl_ptr->ToOperator();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescUtils::GetOpDescFromOperator(const Operator &oprt) {
  return OperatorImpl::GetOpDesc(oprt);
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const std::string &name, const std::string &type) {
  operator_impl_ = ComGraphMakeShared<OperatorImpl>(name, type);
  if (operator_impl_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
    GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
    return;
  }
  OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const AscendString &name, const AscendString &type) {
  if ((name.GetString() != nullptr) && (type.GetString() != nullptr)) {
    std::string op_name = name.GetString();
    std::string op_type = type.GetString();
    operator_impl_ = ComGraphMakeShared<OperatorImpl>(op_name, op_type);
    if (operator_impl_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
      GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
      return;
    }
    OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
  } else {
    GELOGW("[Check][Param] Operator input parameter is nullptr");
  }
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const char_t *name, const char_t *type) {
  if ((name != nullptr) && (type != nullptr)) {
    std::string op_name = name;
    std::string op_type = type;
    operator_impl_ = ComGraphMakeShared<OperatorImpl>(op_name, op_type);
    if (operator_impl_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
      GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
      return;
    }
    OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
  } else {
    GELOGW("[Check][Param] Operator input parameter is nullptr");
  }
}

Operator::Operator(ge::OperatorImplPtr &&op_impl) { operator_impl_ = std::move(op_impl); }

bool Operator::IsEmpty() const {
  if (operator_impl_ == nullptr) {
    return true;
  }
  return false;
}

std::string Operator::GetName() const {
  if (operator_impl_ != nullptr) {
    return operator_impl_->GetName();
  }
  return "";
}

graphStatus Operator::GetName(AscendString &name) const {
  if (operator_impl_ != nullptr) {
    const std::string op_name = operator_impl_->GetName();
    name = op_name.c_str();
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_HOST_VISIBILITY Operator &Operator::SetInput(const std::string &dst_name, const ge::Operator &src_oprt) {
  // Describe the connection relationship between operators, no create action
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator impl is nullptr, check invalid.");
                   return *this, "[Check][Param] operator impl is nullptr.");
  operator_impl_->SetInputImpl(dst_name, src_oprt);
  return *this;
}

GE_FUNC_HOST_VISIBILITY Operator &Operator::SetInput(const char_t *dst_name, const ge::Operator &src_oprt) {
  GE_CHK_BOOL_EXEC(dst_name != nullptr, REPORT_INNER_ERROR("E19999", "param dst name is nullptr, check invalid");
                   return *this, "[Check][Param] Operator dst name is nullptr.");
  // Describe the connection relationship between operators, no create action
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return *this, "[Check][Param] Operator impl is nullptr.");
  const std::string dst_op_name = dst_name;
  operator_impl_->SetInputImpl(dst_op_name, src_oprt);
  return *this;
}

Operator &Operator::SetInput(const std::string &dst_name, const ge::OutHandler &out_handler) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return *this, "[Check][Param] operator impl is nullptr.");
  operator_impl_->SetInputImpl(dst_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(const char *dst_name, const ge::OutHandler &out_handler) {
  if (dst_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dst_name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator dst_name is nullptr.");
    return *this;
  }
  std::string op_dst_name = dst_name;
  return SetInput(op_dst_name, out_handler);
}

Operator &Operator::SetInput(const std::string &dst_name, const ge::Operator &src_oprt, const std::string &name) {
  const auto out_handler = src_oprt.GetOutput(name);
  GE_CHK_BOOL_EXEC(out_handler != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOutput by name:%s failed, out_handler is nullptr.", name.c_str());
                   return *this, "[Get][Output] by name:%s failed, out_handler is nullptr.", name.c_str());
  (void)SetInput(dst_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(const char_t *dst_name, const ge::Operator &src_oprt, const char_t *name) {
  GE_CHK_BOOL_EXEC(dst_name != nullptr, REPORT_INNER_ERROR("E19999", "param dst_name is nullptr, check invalid.");
                   return *this, "[Check][Param] Dst name is nullptr.");
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
                   return *this, "[Check][Param] Name is nullptr.");
  const std::string op_name = name;
  const std::string dst_op_name = dst_name;
  const auto out_handler = src_oprt.GetOutput(op_name);
  GE_CHK_BOOL_EXEC(out_handler != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOutput by name:%s failed, out_handler is nullptr.",
                                      op_name.c_str());
                   return *this, "[Get][Output] by name:%s failed, out_handler is nullptr.", op_name.c_str());
  (void)SetInput(dst_op_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(const std::string &dst_name, const ge::Operator &src_oprt, uint32_t index) {
  const auto out_handler = src_oprt.GetOutput(index);
  GE_CHK_BOOL_EXEC(out_handler != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOutput by index:%u failed, out_handler is nullptr.", index);
                   return *this, "[Get][Output] by index:%u failed, out_handler is nullptr.", index);
  (void)SetInput(dst_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(const char_t *dst_name, const ge::Operator &src_oprt, uint32_t index) {
  GE_CHK_BOOL_EXEC(dst_name != nullptr, REPORT_INNER_ERROR("E19999", "param dst_name is nullptr, check invalid");
                   return *this, "[Check][Param] Dst name is nullptr.");
  const auto out_handler = src_oprt.GetOutput(index);
  GE_CHK_BOOL_EXEC(out_handler != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOutput by index:%u failed, out_handler is nullptr.", index);
                   return *this, "[Get][Output] by index:%u failed, out_handler is nullptr.", index);
  const std::string op_dst_name = dst_name;
  (void)SetInput(dst_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(uint32_t dst_index, const Operator &src_oprt, uint32_t src_index) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    const char_t *invalid_obj_name = (operator_impl_ == nullptr ? "operator" : "op desc");
    REPORT_INNER_ERROR("E19999", "%s impl is nullptr, check invalid.", invalid_obj_name);
    GELOGE(ge::FAILED, "[Check][Param] %s impl is nullptr.", invalid_obj_name);
    return *this;
  }
  std::string dst_name = operator_impl_->GetOpDescImpl()->GetInputNameByIndex(dst_index);
  if (dst_name.empty()) {
    REPORT_INNER_ERROR("E19999", "Set by dst_index:%u failed, dst_index is invalid.", dst_index);
    GELOGE(ge::FAILED, "[GetInputNameByIndex] by index:%u failed, dst_index is invalid.", dst_index);
    return *this;
  }
  return SetInput(dst_name.c_str(), src_oprt, src_index);
}

Operator &Operator::AddControlInput(const Operator &src_oprt) {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
    return *this;
  }
  operator_impl_->AddControlInputImp(src_oprt);
  return *this;
}

graphStatus Operator::GetInputConstData(const std::string &dst_name, Tensor &data) const {
  GE_CHECK_NOTNULL(operator_impl_);
  const graphStatus ret = operator_impl_->GetInputConstData(dst_name, data);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("[Get][ConstInput] %s get input const data failed", dst_name.c_str());
    return ret;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputConstData(const char_t *dst_name, Tensor &data) const {
  GE_CHECK_NOTNULL(dst_name);
  GE_CHECK_NOTNULL(operator_impl_);
  const std::string op_dst_name = dst_name;
  const graphStatus ret = operator_impl_->GetInputConstData(op_dst_name, data);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("[Get][ConstInput] %s get input const data failed", op_dst_name.c_str());
    return ret;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputConstDataOut(const std::string &dst_name, Tensor &data) const {
  GE_CHECK_NOTNULL(operator_impl_);
  if (operator_impl_->GetInputConstDataOut(dst_name, data) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "%s get input const data out failed", dst_name.c_str());
    GELOGE(GRAPH_FAILED, "[Get][Tensor] %s get input const data out failed", dst_name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputConstDataOut(const char *dst_name, Tensor &data) const {
  if (dst_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dst_name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator dst_name is nullptr.");
    return GRAPH_FAILED;
  }
  std::string op_dst_name = dst_name;
  return GetInputConstDataOut(op_dst_name, data);
}

std::shared_ptr<const Node>  Operator::GetNode() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return nullptr, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetNode();
}

TensorDesc Operator::GetInputDesc(const std::string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(name));
}

TensorDesc Operator::GetInputDescByName(const char_t *name) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(op_name));
}

void Operator::SetInferenceContext(const InferenceContextPtr &inference_context) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  operator_impl_->SetInferenceContext(inference_context);
}

InferenceContextPtr Operator::GetInferenceContext() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return nullptr, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetInferenceContext();
}

TensorDesc Operator::GetInputDesc(uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(index));
}

graphStatus Operator::TryGetInputDesc(const std::string &name, TensorDesc &tensor_desc) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  const auto check = operator_impl_->InputIsSet(name);
  if (check) {
    tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(name));
  }
  return check ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Operator::TryGetInputDesc(const char_t *name, TensorDesc &tensor_desc) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  const auto check = operator_impl_->InputIsSet(op_name);
  if (check) {
    tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(op_name));
  }
  return check ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Operator::UpdateInputDesc(const std::string &name, const ge::TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->UpdateInputDesc(name, TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::UpdateInputDesc(const char_t *name, const ge::TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  return operator_impl_->UpdateInputDesc(op_name, TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

OutHandler Operator::GetOutput(const std::string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return nullptr, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetOutput(name);
}

OutHandler Operator::GetOutput(const char *name) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return nullptr;
  }
  std::string op_name = name;
  return GetOutput(op_name);
}

OutHandler Operator::GetOutput(uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return nullptr, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetOutput(index);
}

TensorDesc Operator::GetOutputDesc(const std::string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(name));
}

TensorDesc Operator::GetOutputDescByName(const char_t *name) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(op_name));
}

TensorDesc Operator::GetOutputDesc(uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(index));
}

graphStatus Operator::UpdateOutputDesc(const std::string &name, const ge::TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->UpdateOutputDesc(name, TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::UpdateOutputDesc(const char_t *name, const ge::TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  return operator_impl_->UpdateOutputDesc(op_name, TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

TensorDesc Operator::GetDynamicInputDesc(const std::string &name, uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(name + std::to_string(index)));
}

TensorDesc Operator::GetDynamicInputDesc(const char_t *name, uint32_t index) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(op_name + std::to_string(index)));
}

graphStatus Operator::UpdateDynamicInputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->UpdateInputDesc(name + std::to_string(index),
                                         TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::UpdateDynamicInputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  const std::string op_name = name;
  return operator_impl_->UpdateInputDesc(op_name + std::to_string(index),
                                         TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

TensorDesc Operator::GetDynamicOutputDesc(const std::string &name, uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(name + std::to_string(index)));
}

TensorDesc Operator::GetDynamicOutputDesc(const char_t *name, uint32_t index) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(op_name + std::to_string(index)));
}

graphStatus Operator::UpdateDynamicOutputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->UpdateOutputDesc(name + std::to_string(index),
                                          TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::UpdateDynamicOutputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  return operator_impl_->UpdateOutputDesc(op_name + std::to_string(index),
                                          TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::InferShapeAndType() {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return GRAPH_FAILED, "[Get][OpDescImpl] is nullptr.");

  return operator_impl_->GetOpDescImpl()->CallInferFunc(*this);
}

graphStatus Operator::VerifyAllAttr(bool disable_common_verifier) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return GRAPH_FAILED, "[Get][OpDescImpl] is nullptr.");

  if ((!disable_common_verifier) && (static_cast<graphStatus>(Operator::VerifyAll()) == GRAPH_FAILED)) {
    return GRAPH_FAILED;
  } else {
    return (graphStatus)operator_impl_->GetOpDescImpl()->OpVerify();
  }
}

GE_FUNC_HOST_VISIBILITY size_t Operator::GetInputsSize() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0UL, "[Check][Param] OperatorImpl_ is nullptr");
  return static_cast<size_t>(operator_impl_->GetInputsSize());
}

GE_FUNC_HOST_VISIBILITY size_t Operator::GetOutputsSize() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0UL, "[Check][Param] OperatorImpl_ is nullptr");
  return static_cast<size_t>(operator_impl_->GetOutputsSize());
}

// According to op get the attrs name and type
namespace {
const std::map<AnyValue::ValueType, std::string> kAttrTypesMap = {
    {AnyValue::VT_NONE, "VT_STRING"},
    {AnyValue::VT_STRING, "VT_STRING"},
    {AnyValue::VT_FLOAT, "VT_FLOAT"},
    {AnyValue::VT_BOOL, "VT_BOOL"},
    {AnyValue::VT_INT, "VT_INT"},
    {AnyValue::VT_TENSOR_DESC, "VT_TENSOR_DESC"},
    {AnyValue::VT_TENSOR, "VT_TENSOR"},
    {AnyValue::VT_BYTES, "VT_BYTES"},
    {AnyValue::VT_GRAPH, "VT_GRAPH"},
    {AnyValue::VT_NAMED_ATTRS, "VT_NAMED_ATTRS"},
    {AnyValue::VT_LIST_LIST_INT, "VT_LIST_LIST_INT"},
    {AnyValue::VT_DATA_TYPE, "VT_DATA_TYPE"},
    {AnyValue::VT_LIST_STRING, "VT_LIST_STRING"},
    {AnyValue::VT_LIST_FLOAT, "VT_LIST_FLOAT"},
    {AnyValue::VT_LIST_BOOL, "VT_LIST_BOOL"},
    {AnyValue::VT_LIST_INT, "VT_LIST_INT"},
    {AnyValue::VT_LIST_TENSOR_DESC, "VT_LIST_TENSOR_DESC"},
    {AnyValue::VT_LIST_TENSOR, "VT_LIST_TENSOR"},
    {AnyValue::VT_LIST_BYTES, "VT_LIST_BYTES"},
    {AnyValue::VT_GRAPH, "VT_GRAPH"},
    {AnyValue::VT_LIST_NAMED_ATTRS, "VT_LIST_NAMED_ATTRS"},
    {AnyValue::VT_LIST_DATA_TYPE, "VT_LIST_DATA_TYPE"},
};
} // namespace
const std::map<std::string, std::string> Operator::GetAllAttrNamesAndTypes() const {
  std::map<std::string, std::string> attr_types;

  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return attr_types, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return attr_types, "[Get][OpDescImpl] is nullptr.");
  std::map<std::string, AnyValue> attr_map = operator_impl_->GetOpDescImpl()->GetAllAttrs();

  for (auto &iter : attr_map) {
    const std::string name = iter.first;
    const AnyValue::ValueType type = iter.second.GetValueType();

    const auto iter2 = kAttrTypesMap.find(type);
    if (iter2 != kAttrTypesMap.end()) {
      attr_types[name] = iter2->second;
    }
  }

  return attr_types;
}

graphStatus Operator::GetAllAttrNamesAndTypes(std::map<AscendString, AscendString> &attr_name_types) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return GRAPH_FAILED, "[Get][OpDescImpl] is nullptr.");
  std::map<std::string, AnyValue> attr_map = operator_impl_->GetOpDescImpl()->GetAllAttrs();

  std::map<std::string, AnyValue>::iterator iter;
  for (iter = attr_map.begin(); iter != attr_map.end(); ++iter) {
    std::string name = iter->first;
    AnyValue::ValueType type = iter->second.GetValueType();

    auto iter2 = kAttrTypesMap.find(type);
    if (iter2 != kAttrTypesMap.end()) {
      AscendString temp(name.c_str());
      attr_name_types[temp] = AscendString(iter2->second.c_str());
    }
  }

  return GRAPH_SUCCESS;
}

void Operator::InputRegister(const std::string &name) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  (void)operator_impl_->GetOpDescImpl()->AddInputDesc(name, GeTensorDesc());
}

void Operator::InputRegister(const char *name) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  InputRegister(op_name);
}

void Operator::OptionalInputRegister(const std::string &name) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddOptionalInputDesc(name,
                                                              GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
}

void Operator::OptionalInputRegister(const char *name) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  OptionalInputRegister(op_name);
}

void Operator::InferFuncRegister(const std::function<graphStatus(Operator &)> &func) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddInferFunc(func);
}

void Operator::InferFormatFuncRegister(const std::function<graphStatus(Operator &)> &func) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddInferFormatFunc(func);
}

void Operator::VerifierFuncRegister(const std::function<graphStatus(Operator &)> &func) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddVerifierFunc(func);
}

void Operator::OutputRegister(const std::string &name) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddOutputDesc(name, GeTensorDesc());
}

void Operator::OutputRegister(const char *name) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  OutputRegister(op_name);
}

void Operator::DynamicInputRegister(const std::string &name, const unsigned int num, bool is_push_back) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_INPUT_TD_NUM(name), num),
                   REPORT_INNER_ERROR("E19999", "set attr %s to op:%s failed.", name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return, "[Set][Int] %s to op:%s failed", name.c_str(),
                   operator_impl_->GetOpDescImpl()->GetName().c_str());
  (void)operator_impl_->GetOpDescImpl()->AddDynamicInputDesc(name, num, is_push_back);
}

void Operator::DynamicInputRegister(const char *name, const unsigned int num, bool is_push_back) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  DynamicInputRegister(op_name, num, is_push_back);
}

void Operator::DynamicInputRegisterByIndex(const std::string &name, const unsigned int num, size_t index) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  (void)(operator_impl_->GetOpDescImpl()->AddDynamicInputDescByIndex(name, num, index));
}

void Operator::DynamicInputRegisterByIndex(const char *name, const unsigned int num, size_t index) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  DynamicInputRegisterByIndex(op_name, num, index);
}

int Operator::GetDynamicInputNum(const std::string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return 0, "[Get][OpDescImpl] is nullptr.");
  int32_t num = 0;
  GE_CHK_BOOL_EXEC(AttrUtils::GetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_INPUT_TD_NUM(name), num),
                   REPORT_INNER_ERROR("E19999", "get attr %s failed, op:%s.", name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return num, "[Get][Init] %s failed", name.c_str());
  return num;
}

int32_t Operator::GetDynamicInputNum(const char_t *name) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return 0, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return 0, "[Get][OpDescImpl] is nullptr.");
  std::string op_name = name;
  int32_t num = 0;
  GE_CHK_BOOL_EXEC(AttrUtils::GetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_INPUT_TD_NUM(op_name), num),
                   REPORT_INNER_ERROR("E19999", "get attr %s failed, op:%s.", op_name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return num, "[Get][Int] %s failed", op_name.c_str());
  return num;
}

void Operator::DynamicOutputRegister(const std::string &name, const uint32_t num, bool is_push_back) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_OUTPUT_TD_NUM(name), num),
                   REPORT_INNER_ERROR("E19999", "set attr %s to op:%s failed.", name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return, "[Set][Int] %s to op:%s failed", name.c_str(),
                   operator_impl_->GetOpDescImpl()->GetName().c_str());
  (void)operator_impl_->GetOpDescImpl()->AddDynamicOutputDesc(name, num, is_push_back);
}

void Operator::DynamicOutputRegister(const char *name, const unsigned int num, bool is_push_back) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  DynamicOutputRegister(op_name, num, is_push_back);
}

int Operator::GetDynamicOutputNum(const std::string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return 0, "[Get][OpDescImpl] is nullptr.");
  int32_t num = 0;
  GE_CHK_BOOL_EXEC(AttrUtils::GetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_OUTPUT_TD_NUM(name), num),
                   REPORT_INNER_ERROR("E19999", "get attr %s failed, op:%s.", name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return num, "[Get][Init] %s failed", name.c_str());
  return num;
}

int32_t Operator::GetDynamicOutputNum(const char_t *name) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return 0, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return 0, "[Get][OpDescImpl] is nullptr.");
  std::string op_name = name;
  int32_t num = 0;
  GE_CHK_BOOL_EXEC(AttrUtils::GetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_OUTPUT_TD_NUM(op_name), num),
                   REPORT_INNER_ERROR("E19999", "get attr %s failed, op:%s.", op_name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return num, "[Get][Init] %s failed", op_name.c_str());
  return num;
}

void Operator::RequiredAttrRegister(const std::string &name) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  (void)(operator_impl_->GetOpDescImpl()->AddRequiredAttr(name));
}

void Operator::RequiredAttrRegister(const char *name) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  RequiredAttrRegister(op_name);
}

graphStatus Operator::VerifyAll() {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return GRAPH_FAILED, "[Get][OpDescImpl] is nullptr.");

  // Check all inputs defined
  for (const std::string &iname : operator_impl_->GetOpDescImpl()->GetAllInputNames()) {
    GE_CHK_BOOL_RET_STATUS(operator_impl_->GetOpDescImpl()->IsOptionalInput(iname) || operator_impl_->InputIsSet(iname),
                           GRAPH_FAILED, "[Check][Param] operator input %s is not linked.", iname.c_str());
    std::vector<int64_t> ishape = operator_impl_->GetOpDescImpl()->GetInputDesc(iname).GetShape().GetDims();
    for (int64_t dim : ishape) {
      GE_CHK_BOOL_RET_STATUS(dim > 0, GRAPH_FAILED,
                             "[Check][Param] operator input %s shape contains negative or zero dimension, "
                             "node:%s, index:%d.",
                             iname.c_str(), operator_impl_->GetOpDescImpl()->GetName().c_str(),
                             operator_impl_->GetOpDescImpl()->GetInputIndexByName(iname));
    }
  }
  // Check all attributes defined
  const auto all_attributes = operator_impl_->GetOpDescImpl()->GetAllAttrs();
  for (const auto &name : operator_impl_->GetOpDescImpl()->GetAllAttrNames()) {
    GE_CHK_BOOL_RET_STATUS(all_attributes.find(name) != all_attributes.end(), GRAPH_FAILED,
                           "[Check][Param] operator attribute %s is empty.", name.c_str());
  }

  return GRAPH_SUCCESS;
}

std::string Operator::GetOpType() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return "Data", "[Check][Param] operator impl is nullptr.");
  return OperatorImpl::GetOpDesc(*this)->GetType();
}

graphStatus Operator::GetOpType(AscendString &type) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  std::string op_type = OperatorImpl::GetOpDesc(*this)->GetType();
  type = op_type.c_str();
  return GRAPH_SUCCESS;
}

Operator &Operator::SetInput(const std::string &dst_name, uint32_t dst_index, const ge::Operator &src_oprt) {
  std::string dynamic_dst_name = DYNAMIN_INPUT_NAME(dst_name, dst_index);
  return SetInput(dynamic_dst_name, src_oprt);
}

Operator &Operator::SetInput(const char *dst_name, uint32_t dst_index, const ge::Operator &src_oprt) {
  if (dst_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dst_name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator dst_name is nullptr.");
    return *this;
  }
  std::string op_dst_name = dst_name;
  return SetInput(op_dst_name, dst_index, src_oprt);
}

Operator &Operator::SetInput(const std::string &dst_name, uint32_t dst_index, const ge::Operator &src_oprt,
                             const std::string &name) {
  std::string dynamic_dst_name = DYNAMIN_INPUT_NAME(dst_name, dst_index);
  return SetInput(dynamic_dst_name, src_oprt, name);
}

Operator &Operator::SetInput(const char *dst_name, uint32_t dst_index, const ge::Operator &src_oprt,
                             const char *name) {
  if (dst_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dst_name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator dst_name is nullptr.");
    return *this;
  }
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  std::string op_dst_name = dst_name;
  std::string op_name = name;
  return SetInput(op_dst_name, dst_index, src_oprt, op_name);
}

OperatorImplPtr Operator::GetOperatorImplPtr() const { return operator_impl_; }

void Operator::BreakConnect() const {
  if (operator_impl_ == nullptr) {
    GELOGW("[Check][Param] operator impl is nullptr");
    return;
  }
  operator_impl_->ClearInputLinks();
  operator_impl_->ClearOutputLinks();
  OperatorKeeper::GetInstance().CheckOutOperator(operator_impl_);
}

OP_ATTR_SET_IMP(int64_t, Int)
OP_ATTR_SET_IMP(int32_t, Int)
OP_ATTR_SET_IMP(uint32_t, Int)
OP_ATTR_GET_IMP(int64_t &, Int)
OP_ATTR_GET_IMP(int32_t &, Int)
OP_ATTR_GET_IMP(uint32_t &, Int)
OP_ATTR_SET_IMP(const std::vector<int64_t> &, ListInt)
OP_ATTR_SET_IMP(const std::vector<int32_t> &, ListInt)
OP_ATTR_SET_IMP(const std::vector<uint32_t> &, ListInt)
OP_ATTR_SET_IMP(std::initializer_list<int64_t> &&, ListInt)
OP_ATTR_GET_IMP(std::vector<int64_t> &, ListInt)
OP_ATTR_GET_IMP(std::vector<int32_t> &, ListInt)
OP_ATTR_GET_IMP(std::vector<uint32_t> &, ListInt)
OP_ATTR_GET_IMP(std::vector<std::vector<int64_t>> &, ListListInt)
OP_ATTR_SET_IMP(const std::vector<std::vector<int64_t>> &, ListListInt)

OP_ATTR_SET_IMP(float32_t, Float)
OP_ATTR_GET_IMP(float32_t &, Float)
OP_ATTR_SET_IMP(const std::vector<float32_t> &, ListFloat)
OP_ATTR_GET_IMP(std::vector<float32_t> &, ListFloat)

OP_ATTR_SET_IMP(bool, Bool)
OP_ATTR_GET_IMP(bool &, Bool)
OP_ATTR_SET_IMP(const std::vector<bool> &, ListBool)
OP_ATTR_GET_IMP(std::vector<bool> &, ListBool)

OP_ATTR_SET_IMP(const ge::NamedAttrs &, NamedAttrs)
OP_ATTR_GET_IMP(ge::NamedAttrs &, NamedAttrs)
OP_ATTR_SET_IMP(const std::vector<ge::NamedAttrs> &, ListNamedAttrs)
OP_ATTR_GET_IMP(std::vector<ge::NamedAttrs> &, ListNamedAttrs)

OP_ATTR_REG_IMP(int64_t, Int)
OP_ATTR_REG_IMP(const std::vector<int64_t> &, ListInt)
OP_ATTR_REG_IMP(float, Float)
OP_ATTR_REG_IMP(const std::vector<float> &, ListFloat)
OP_ATTR_REG_IMP(bool, Bool)
OP_ATTR_REG_IMP(const std::vector<bool> &, ListBool)
OP_ATTR_REG_IMP(const std::vector<std::vector<int64_t>> &, ListListInt)
OP_ATTR_REG_IMP(const ge::NamedAttrs &, NamedAttrs)
OP_ATTR_REG_IMP(const std::vector<ge::NamedAttrs> &, ListNamedAttrs)

void Operator::AttrRegister(const std::string &name, const std::string &attr_value) {
  AttrRegister(name.c_str(), AscendString(attr_value.c_str()));
}

void Operator::AttrRegister(const std::string &name, const std::vector<std::string> &attr_value) {
  std::vector<AscendString> attr_values;
  std::transform(attr_value.begin(), attr_value.end(), attr_values.begin(),
                 [](const std::string &val) {
                   return AscendString(val.c_str());
                 });
  AttrRegister(name.c_str(), attr_values);
}

void Operator::AttrRegister(const char *name, int64_t attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const char *name, const vector<int64_t> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const char *name, float attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const char *name, const vector<float> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const char *name, const char *attr_value) {
  if ((name == nullptr) || (attr_value == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr or attr_value is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator input parameters is nullptr.");
    return;
  }
  std::string op_name = name;
  std::string op_attr_value = attr_value;
  AttrRegister(op_name, op_attr_value);
}

void Operator::AttrRegister(const char *name, bool attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const char *name, const vector<bool> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const char *name, const vector<vector<int64_t>> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const char *name, const NamedAttrs &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const char *name, const vector<NamedAttrs> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const std::string &name, const AscendString &attr_value) {
  if (attr_value.GetString() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Attr %s register param is invalid.", name.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Attr %s register param is invalid.", name.c_str());
    return;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  const std::string str_attr_value = attr_value.GetString();
  if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), name, str_attr_value)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const char *name, const AscendString &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  if (attr_value.GetString() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Attr %s register param is invalid.", name);
    GELOGE(GRAPH_FAILED, "[Check][Param] Attr %s register param is invalid.", name);
    return;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", op_name.c_str());
    return;
  }
  std::string str_attr_value = attr_value.GetString();
  if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), op_name, str_attr_value)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", op_name.c_str());
  }
}

void Operator::AttrRegister(const std::string &name, const std::vector<AscendString> &attr_value) {
  std::vector<std::string> str_attr_values;
  for (auto &val : attr_value) {
    if (val.GetString() == nullptr) {
      REPORT_INNER_ERROR("E19999", "Attr %s register value is invalid.", name.c_str());
      GELOGE(GRAPH_FAILED, "Attr %s register value is invalid.", name.c_str());
      return;
    }
    str_attr_values.emplace_back(val.GetString());
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  if (!AttrUtils::SetListStr(operator_impl_->GetOpDescImpl(), name, str_attr_values)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const char *name, const std::vector<AscendString> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

Operator &Operator::SetAttr(const std::string &name, const std::string &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
      GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
      return *this;
    }
    if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), name, attr_value)) {
      GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
    }
    return *this;
}

graphStatus Operator::GetAttr(const std::string &name, std::string &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
      GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
      return GRAPH_FAILED;
    }
    if (!AttrUtils::GetStr(operator_impl_->GetOpDescImpl(), name, attr_value)) {
      GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const std::vector<std::string> &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  if (!AttrUtils::SetListStr(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const std::string &name, std::vector<std::string> &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  if (!AttrUtils::GetListStr(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const char_t *name, const char_t *attr_value) {
  if ((name == nullptr) || (attr_value == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr or attr_value is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator input parameters is nullptr.");
    return *this;
  }

  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  const std::string op_name = name;
  const std::string op_attr_value = attr_value;
  if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char_t *name, const AscendString &attr_value) {
  if ((name == nullptr) || (attr_value.GetString() == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Operator input parameters is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator input parameters is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  const std::string op_name = name;
  const std::string op_attr_value = attr_value.GetString();
  if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const char_t *name, AscendString &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "Operator input parameters name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator input parameters is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  const std::string op_name = name;
  std::string op_attr_value;
  if (!AttrUtils::GetStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  attr_value = AscendString(op_attr_value.c_str());
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const char_t *name, const std::vector<AscendString> &attr_values) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "Operator input parameters name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return  *this;
  }
  std::vector<std::string> op_attr_values;
  for (auto &attr_value : attr_values) {
    if (attr_value.GetString() == nullptr) {
      REPORT_INNER_ERROR("E19999", "Operator ascend std::string name is nullptr, check invalid");
      GELOGE(GRAPH_FAILED, "[Check][Param] Operator ascend std::string name is nullptr.");
      return  *this;
    }
    op_attr_values.emplace_back(attr_value.GetString());
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  const std::string op_name = name;
  if (!AttrUtils::SetListStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_values)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<AscendString> &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "Operator input parameters name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  const std::string op_name = name;
  std::vector<std::string> op_attr_values;
  if (!AttrUtils::GetListStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_values)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  for (auto &op_attr_value : op_attr_values) {
    attr_value.emplace_back(AscendString(op_attr_value.c_str()));
  }
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const Tensor &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  const GeTensor tensor = TensorAdapter::AsGeTensor(attr_value);
  if (!AttrUtils::SetTensor(operator_impl_->GetOpDescImpl(), name, tensor)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char_t *name, const Tensor &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name);
    return *this;
  }
  const std::string op_name = name;
  const GeTensor tensor = TensorAdapter::AsGeTensor(attr_value);
  if (!AttrUtils::SetTensor(operator_impl_->GetOpDescImpl(), op_name, tensor)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const std::string &name, const std::vector<Tensor> &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  std::vector<GeTensor> val_list;
  for (const auto &item : attr_value) {
    const auto tensor = TensorAdapter::AsGeTensor(item);
    val_list.push_back(tensor);
  }
  if (!AttrUtils::SetListTensor(operator_impl_->GetOpDescImpl(), name, val_list)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char_t *name, const std::vector<Tensor> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  const std::string op_name = name;
  std::vector<GeTensor> val_list;
  for (const auto &item : attr_value) {
    const auto tensor = TensorAdapter::AsGeTensor(item);
    val_list.push_back(tensor);
  }
  if (!AttrUtils::SetListTensor(operator_impl_->GetOpDescImpl(), op_name, val_list)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const std::string &name, Tensor &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  ConstGeTensorPtr tensor;
  if (!AttrUtils::GetTensor(operator_impl_->GetOpDescImpl(), name, tensor)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  attr_value = TensorAdapter::GeTensor2Tensor(tensor);
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, Tensor &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  const std::string op_name = name;
  ConstGeTensorPtr tensor;
  if (!AttrUtils::GetTensor(operator_impl_->GetOpDescImpl(), op_name, tensor)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  attr_value = TensorAdapter::GeTensor2Tensor(tensor);
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const std::string &name, std::vector<Tensor> &attr_value) const {
  attr_value.clear();
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  std::vector<ConstGeTensorPtr> val_list;
  if (!AttrUtils::GetListTensor(operator_impl_->GetOpDescImpl(), name, val_list)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  for (auto &tensor : val_list) {
    attr_value.push_back(TensorAdapter::GeTensor2Tensor(tensor));
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<Tensor> &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  attr_value.clear();
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  const std::string op_name = name;
  std::vector<ConstGeTensorPtr> val_list;
  if (!AttrUtils::GetListTensor(operator_impl_->GetOpDescImpl(), op_name, val_list)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  for (auto &tensor : val_list) {
    attr_value.push_back(TensorAdapter::GeTensor2Tensor(tensor));
  }
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const OpBytes &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  if (!AttrUtils::SetZeroCopyBytes(operator_impl_->GetOpDescImpl(), name,
                                   Buffer::CopyFrom(attr_value.data(), attr_value.size()))) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char_t *name, const OpBytes &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  const std::string op_name = name;
  if (!AttrUtils::SetZeroCopyBytes(operator_impl_->GetOpDescImpl(), op_name,
                                   Buffer::CopyFrom(attr_value.data(), attr_value.size()))) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const std::string &name, OpBytes &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  Buffer buffer;
  if (!AttrUtils::GetZeroCopyBytes(operator_impl_->GetOpDescImpl(), name, buffer)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  attr_value.clear();
  if (buffer.data() == nullptr) {
    REPORT_CALL_ERROR("E19999", "buffer data is null, op:%s", operator_impl_->GetOpDescImpl()->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] buffer data is null.");
    return GRAPH_FAILED;
  }
  attr_value.assign(buffer.data(), buffer.data() + buffer.size());
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, OpBytes &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  const std::string op_name = name;
  Buffer buffer;
  if (!AttrUtils::GetZeroCopyBytes(operator_impl_->GetOpDescImpl(), op_name, buffer)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  attr_value.clear();
  if (buffer.data() == nullptr) {
    REPORT_CALL_ERROR("E19999", "buffer data is null, op:%s", operator_impl_->GetOpDescImpl()->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] buffer data is null.");
    return GRAPH_FAILED;
  }
  attr_value.assign(buffer.data(), buffer.data() + buffer.size());
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, ge::AttrValue &&attr_value) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return *this, "[Check][Param] Operator impl is nullptr.");
  (void)operator_impl_->SetAttr(name, std::move(attr_value.impl->geAttrValue_));
  return *this;
}

Operator &Operator::SetAttr(const char_t *name, ge::AttrValue &&attr_value) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
                   return *this, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return *this, "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  (void)operator_impl_->SetAttr(op_name, std::move(attr_value.impl->geAttrValue_));
  return *this;
}

graphStatus Operator::GetAttr(const std::string &name, ge::AttrValue &attr_value) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetAttr(name, attr_value.impl->geAttrValue_);
}

graphStatus Operator::GetAttr(const char_t *name, ge::AttrValue &attr_value) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  const std::string op_name = name;
  return operator_impl_->GetAttr(op_name, attr_value.impl->geAttrValue_);
}

Operator &Operator::SetAttr(const std::string &name, const std::vector<ge::DataType> &attr_value) {
  if ((operator_impl_ == nullptr) || (!operator_impl_->GetOpDescImpl())) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  if (!AttrUtils::SetListDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char_t *name, const std::vector<ge::DataType> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if ((operator_impl_ == nullptr) || (!operator_impl_->GetOpDescImpl())) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  const std::string op_name = name;
  if (!AttrUtils::SetListDataType(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const std::string &name, std::vector<ge::DataType> &attr_value) const {
  attr_value.clear();
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  if (!AttrUtils::GetListDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, std::vector<ge::DataType> &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  attr_value.clear();
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  const std::string op_name = name;
  if (!AttrUtils::GetListDataType(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const std::string &name, const ge::DataType &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  if (!AttrUtils::SetDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char_t *name, const ge::DataType &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  const std::string op_name = name;
  if (!AttrUtils::SetDataType(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const std::string &name, ge::DataType &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  if (!AttrUtils::GetDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char_t *name, ge::DataType &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  const std::string op_name = name;
  if (!AttrUtils::GetDataType(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

void Operator::AttrRegister(const std::string &name, const std::vector<ge::DataType> &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  if (!AttrUtils::SetListDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const char *name, const std::vector<ge::DataType> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const std::string &name, const ge::DataType &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  if (!AttrUtils::SetDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const char *name, const ge::DataType &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const std::string &name, const Tensor &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  const GeTensor &tensor = TensorAdapter::AsGeTensor(attr_value);
  if (!AttrUtils::SetTensor(operator_impl_->GetOpDescImpl(), name, tensor)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const char *name, const Tensor &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const std::string &name, const std::vector<Tensor> &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  std::vector<GeTensor> val_list;
  for (const auto &item : attr_value) {
    val_list.push_back(TensorAdapter::AsGeTensor(item));
  }
  if (!AttrUtils::SetListTensor(operator_impl_->GetOpDescImpl(), name, val_list)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const char *name, const vector<Tensor> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::AttrRegister(const std::string &name, const OpBytes &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  if (!AttrUtils::SetZeroCopyBytes(operator_impl_->GetOpDescImpl(), name,
                                   Buffer::CopyFrom(attr_value.data(), attr_value.size()))) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const char *name, const OpBytes &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = name;
  AttrRegister(op_name, attr_value);
}

void Operator::SubgraphRegister(const std::string &ir_name, bool dynamic) {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", ir_name.c_str());
    return;
  }
  operator_impl_->SubgraphRegister(ir_name, dynamic ? static_cast<bool>(kDynamic) : static_cast<bool>(kStatic));
}

void Operator::SubgraphRegister(const char *ir_name, bool dynamic) {
  if (ir_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = ir_name;
  SubgraphRegister(op_name, dynamic);
}

void Operator::SubgraphCountRegister(const std::string &ir_name, uint32_t count) {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", ir_name.c_str());
    return;
  }
  operator_impl_->SubgraphCountRegister(ir_name, count);
}

void Operator::SubgraphCountRegister(const char *ir_name, uint32_t count) {
  if (ir_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return;
  }
  std::string op_name = ir_name;
  SubgraphCountRegister(op_name, count);
}

void Operator::SetSubgraphBuilder(const std::string &ir_name, uint32_t index, const SubgraphBuilder &builder) {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", ir_name.c_str());
    return;
  }
  operator_impl_->SetSubgraphBuilder(ir_name, index, builder);
}

void Operator::SetSubgraphBuilder(const char *ir_name, uint32_t index, const SubgraphBuilder &builder) {
  if (ir_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param ir_name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator ir_name is nullptr.");
    return;
  }
  std::string op_ir_name = ir_name;
  SetSubgraphBuilder(op_ir_name, index, builder);
}

std::vector<std::string> Operator::GetSubgraphNames() const {
  return operator_impl_->GetSubgraphNames();
}

graphStatus Operator::GetSubgraphNames(std::vector<AscendString> &names) const {
  std::vector<std::string> subgraph_names = operator_impl_->GetSubgraphNames();
  for (auto &subgraph_name : subgraph_names) {
    names.emplace_back(subgraph_name.c_str());
  }
  return GRAPH_SUCCESS;
}

SubgraphBuilder Operator::GetDynamicSubgraphBuilder(const std::string &name, uint32_t index) const {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
    return nullptr;
  }
  return operator_impl_->GetSubgraphBuilder(name, index);
}

SubgraphBuilder Operator::GetDynamicSubgraphBuilder(const char_t *name, uint32_t index) const {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
    return nullptr;
  }
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param ir_name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return nullptr;
  }
  const std::string op_ir_name = name;
  return operator_impl_->GetSubgraphBuilder(op_ir_name, index);
}

SubgraphBuilder Operator::GetSubgraphBuilder(const std::string &name) const {
  return GetDynamicSubgraphBuilder(name, 0U);
}

SubgraphBuilder Operator::GetSubgraphBuilder(const char_t *name) const {
  std::string graph_ir_name;
  if (name != nullptr) {
    graph_ir_name = name;
  }
  return GetDynamicSubgraphBuilder(graph_ir_name, 0U);
}

Graph Operator::GetSubgraphImpl(const std::string &name) const {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GE_LOGE("[Check][Param] Failed to get subgraph %s, the operator impl is null", name.c_str());
    return Graph("");
  }
  const auto op_desc = OpDescUtils::GetOpDescFromOperator(*this);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s, because the op_desc is nullptr.", name.c_str());
    GE_LOGE("[Get][OpDesc] Failed to get subgraph %s, the op_desc is null", name.c_str());
    return Graph("");
  }
  const auto &subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  const auto iter = subgraph_names_to_index.find(name);
  if (iter == subgraph_names_to_index.end()) {
    REPORT_INNER_ERROR("E19999", "Failed to get subgraph %s, the name may be invalid", name.c_str());
    GE_LOGE("[Check][Param] Failed to get subgraph %s, the name may be invalid", name.c_str());
    return Graph("");
  }
  const auto subgraph_instance_name = op_desc->GetSubgraphInstanceName(iter->second);
  if (subgraph_instance_name.empty()) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s index %u, the subgraph may not be added",
                      name.c_str(), iter->second);
    GE_LOGE("[Get][Subgraph] %s index %u failed, because the subgraph may not be added",
            name.c_str(), iter->second);
    return Graph("");
  }

  const auto node = operator_impl_->GetNode();
  if (node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s, because the node is null", name.c_str());
    GE_LOGE("[Get][Node] Failed to get subgraph %s, because the node is null", name.c_str());
    return Graph("");
  }
  const auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s, because can not find the root graph,node:%s",
                      name.c_str(), node->GetName().c_str());
    GE_LOGE("[Get][Subgraph] subgraph %s failed, because can not find the root graph", name.c_str());
    return Graph("");
  }
  const auto subgraph = root_graph->GetSubgraph(subgraph_instance_name);
  if (subgraph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s index %u, because can not find the instance %s "
                      "from the root graph", name.c_str(), iter->second, subgraph_instance_name.c_str());
    GE_LOGE("[Get][Subgraph] %s index %u failed, because can not find the instance %s from the root graph",
            name.c_str(), iter->second, subgraph_instance_name.c_str());
    return Graph("");
  }
  return GraphUtils::CreateGraphFromComputeGraph(subgraph);
}

Graph Operator::GetSubgraphImpl(const char *name) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return Graph("");
  }
  std::string op_name = name;
  return GetSubgraphImpl(op_name);
}

Graph Operator::GetSubgraph(const std::string &name) const {
  return GetSubgraphImpl(name);
}

Graph Operator::GetSubgraph(const char_t *name) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Get subgraph failed, name is nullptr.");
    return Graph("");
  }
  const std::string op_name = name;
  return GetSubgraphImpl(op_name);
}

Graph Operator::GetDynamicSubgraph(const std::string &name, uint32_t index) const {
  return GetSubgraph(name + std::to_string(index));
}

Graph Operator::GetDynamicSubgraph(const char_t *name, uint32_t index) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return Graph("");
  }
  const std::string op_name = name;
  return GetSubgraph(op_name + std::to_string(index));
}

size_t Operator::GetSubgraphNamesCount() const {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GE_LOGE("[Check][Param] Failed to get subgraph names count, the operator impl is null");
    return 0UL;
  }
  return operator_impl_->GetSubgraphNamesCount();
}

class GraphBuilderImpl {
public:
  explicit GraphBuilderImpl(const std::string &name) : graph_(ComGraphMakeShared<ComputeGraph>(name)) {
    if (graph_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "ComputeGraph make shared failed");
      GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] ComputeGraph make shared failed");
      return;
    }
  }

  ~GraphBuilderImpl() {}

  ComputeGraphPtr BuildGraph(const std::vector<Operator> &inputs) {
    std::vector<OperatorImplPtr> vec_inputs;
    for (auto &it : inputs) {
      const auto src_op_impl = it.operator_impl_;
      GE_CHK_BOOL_EXEC(src_op_impl != nullptr, REPORT_INNER_ERROR("E19999", "src_op_impl is nullptr, check invalid.");
                       return nullptr, "[Check][Param] Operator Impl is null.");
      GE_CHK_BOOL_EXEC(src_op_impl->op_desc_ != nullptr,
                       REPORT_INNER_ERROR("E19999", "impl's opdesc is nullptr, check invalid.");
                       return nullptr, "[Check][Param] Operator impl's opdesc is null.");

      const std::string type = src_op_impl->op_desc_->GetType();
      const auto node_op = ge::OperatorFactory::CreateOperator("node_op", type);
      const auto tensor_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
      node_op.BreakConnect();

      GE_CHK_BOOL_EXEC(tensor_desc != nullptr, continue, "[Get][Opdesc] tensor_desc is null.");
      if (((tensor_desc->GetInputsSize() == 0UL) && (tensor_desc->GetOutputsSize() > 0UL)) || type == DATA ||
          type == VARIABLE || type == INITDATA || type == GETNEXT) {
        vec_inputs.push_back(it.operator_impl_);
      } else {
        GELOGW("[BuildGraph][CheckInput] Input operator should be Data, Variable operator or operator that has output "
               "but no input.");
      }
    }
    GE_CHK_BOOL_EXEC(!vec_inputs.empty(),
                     REPORT_INNER_ERROR("E19999", "User Input do not include operator such as "
                                        "Data, Variable operator or operator that has output but no input.");
                     return nullptr, "[Check][Param] User Input do not include operator such as "
                     "Data, Variable operator or operator that has output but no input.");
    auto ret = WalkAllOperators(vec_inputs);
    GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS, return nullptr, "[Call][WalkAllOperators] failed, ret:%d.", ret);

    ret = AddEdge();
    GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS, return nullptr, "[Add][Edge] failed, ret:%d.", ret);

    return graph_;
  }

  const std::map<OperatorImplPtr, NodePtr> &GetAllNodesInfo() const { return all_nodes_info_; }

private:
  graphStatus WalkAllOperators(const std::vector<OperatorImplPtr> &vec_ops) {
    GE_CHK_BOOL_EXEC(graph_ != nullptr,
                     REPORT_INNER_ERROR("E19999", "graph_ is nullptr, check invalid.");
                     return GRAPH_FAILED, "[Check][Param] graph_ is null.");
    std::queue<std::vector<OperatorImplPtr>> que;
    que.push(vec_ops);
    while (!que.empty()) {
      const auto vec_tem = que.front();
      que.pop();
      for (const auto &op_impl : vec_tem) {
        GE_CHK_BOOL_EXEC(op_impl != nullptr,
                         REPORT_INNER_ERROR("E19999", "op_impl is nullptr, check invalid.");
                         return GRAPH_FAILED, "[Check][Param] Operator Impl is null.");
        GE_CHK_BOOL_EXEC_INFO(all_nodes_info_.find(op_impl) == all_nodes_info_.end(), continue,
                              "This node %s has created.", op_impl->GetName().c_str())
        auto node_ptr = graph_->AddNode(op_impl->op_desc_);
        GE_CHK_BOOL_EXEC(node_ptr != nullptr,
                         REPORT_CALL_ERROR("E19999", "add node failed.");
                         return GRAPH_FAILED, "[Add][Node] failed.");
        (void)(all_nodes_info_.insert(std::make_pair(op_impl, node_ptr)));

        auto &out_links = op_impl->output_links_;
        std::vector<OperatorImplPtr> vec_op_forward{};
        for (const auto &out_link : out_links) {
          for (const auto &op_forward : out_link.second) {
            vec_op_forward.push_back(op_forward.GetOwner());
          }
        }

        auto &out_control_links = op_impl->control_output_link_;
        for (const auto &out_link : out_control_links) {
          vec_op_forward.push_back(out_link.lock());
        }
        que.push(vec_op_forward);

        auto &in_links = op_impl->input_link_;
        std::vector<OperatorImplPtr> vec_op_back_forward{};
        for (const auto &in_link : in_links) {
          vec_op_back_forward.push_back(in_link.second.GetOwner());
        }

        auto &in_control_links = op_impl->control_input_link_;
        for (const auto &in_link : in_control_links) {
          vec_op_back_forward.push_back(in_link.lock());
        }
        que.push(vec_op_back_forward);

        if (WalkAllSubgraphs(node_ptr, op_impl) != GRAPH_SUCCESS) {
          return GRAPH_FAILED;
        }
      }
    }
    return MoveSubgraphToRoot(graph_);
  }

  graphStatus WalkAllSubgraphs(const NodePtr &node, const OperatorImplPtr &op_impl) {
    const std::string name = node->GetName();
    for (auto &name_idx : op_impl->op_desc_->GetSubgraphNameIndexes()) {
      const SubgraphBuilder &builder = op_impl->GetSubgraphBuilder(name_idx.first);
      if (builder == nullptr) {
        GELOGW("[Check][Param] Node %s has no builder", name.c_str());
        continue;
      }

      const Graph graph = builder();  // Build subgraph from user define builder.
      const ComputeGraphPtr &subgraph = GraphUtils::GetComputeGraph(graph);
      GE_CHK_BOOL_EXEC(subgraph != nullptr,
                       REPORT_CALL_ERROR("E19999", "Node: %s, Build graph failed.", name.c_str());
                       return GRAPH_FAILED, "[Get][Graph] Node: %s, Build graph failed.", name.c_str());

      subgraph->SetParentNode(node);
      subgraph->SetParentGraph(graph_);
      if (graph_->AddSubgraph(subgraph->GetName(), subgraph) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }

      if (op_impl->op_desc_->SetSubgraphInstanceName(name_idx.second, subgraph->GetName()) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Failed to set subgraph %s index %u", subgraph->GetName().c_str(), name_idx.second);
        GELOGE(GRAPH_FAILED, "[Set][SubGraph] %s index %u failed", subgraph->GetName().c_str(), name_idx.second);
        return GRAPH_FAILED;
      }
    }

    return GRAPH_SUCCESS;
  }

  graphStatus MoveSubgraphToRoot(const ComputeGraphPtr &graph) const {
    const ComputeGraphPtr &root_graph = GraphUtils::FindRootGraph(graph);
    if (root_graph == nullptr) {
      REPORT_CALL_ERROR("E19999", "failed to find root graph of %s", graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Find][RootGraph] failed for graph:%s.", graph->GetName().c_str());
      return GRAPH_FAILED;
    }

    if (root_graph == graph) {
      const auto subgraphs = graph->GetAllSubgraphs();

      for (auto &subgraph : subgraphs) {
        if (MoveSubgraphToRoot(subgraph) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
      }
    } else {
      const auto subgraphs = graph->GetAllSubgraphs();
      for (auto &subgraph : subgraphs) {
        if (root_graph->AddSubgraph(subgraph->GetName(), subgraph) != GRAPH_SUCCESS) {
          return GRAPH_FAILED;
        }
        graph->RemoveSubgraph(subgraph->GetName());
        if (MoveSubgraphToRoot(subgraph) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
      }
    }

    return GRAPH_SUCCESS;
  }

  graphStatus AddEdge() {
    for (const auto &node_info : all_nodes_info_) {
      const auto src_op_impl_ptr = node_info.first;
      const auto src_node_ptr = node_info.second;

      GE_IF_BOOL_EXEC(src_op_impl_ptr == nullptr || src_node_ptr == nullptr, continue);
      const auto out_links = src_op_impl_ptr->output_links_;
      GE_CHK_BOOL_EXEC(src_op_impl_ptr->op_desc_ != nullptr,
                       REPORT_INNER_ERROR("E19999", "Src operator impl's op_desc is nullptr, check invalid.");
                       return GRAPH_FAILED, "[Check][Param] Src operator impl's op_desc is null.");
      auto &op_desc = src_op_impl_ptr->op_desc_;
      GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
      for (const auto &out : out_links) {
        const auto src_idx = op_desc->GetOutputIndexByName(out.first);
        GE_CHK_BOOL_EXEC(src_idx >= 0,
                         REPORT_INNER_ERROR("E19999", "Find output index by name:%s in op:%s failed",
                                            out.first.c_str(), op_desc->GetName().c_str());
                         return GRAPH_FAILED, "[Get][Index] Find output index by name:%s failed", out.first.c_str());

        const auto src_anchor = src_node_ptr->GetOutDataAnchor(src_idx);
        GE_CHK_BOOL_EXEC(src_anchor != nullptr,
                         REPORT_INNER_ERROR("E19999", "GetOutDataAnchor failed, index:%d, op:%s.",
                                            src_idx, op_desc->GetName().c_str());
                         return GRAPH_FAILED, "[Get][OutDataAnchor] failed, index:%d.", src_idx);

        for (const auto &dst_opio : out.second) {
          const auto dst_node_info = all_nodes_info_.find(dst_opio.GetOwner());
          GE_CHK_BOOL_EXEC(dst_node_info != all_nodes_info_.end(),
                           REPORT_INNER_ERROR("E19999", "Find Dst node failed, op:%s.", op_desc->GetName().c_str());
                           return GRAPH_FAILED, "[Check][Param] Find Dst node failed.");

          GE_IF_BOOL_EXEC(dst_node_info->second == nullptr, continue);

          const auto dst_anchor = dst_node_info->second->GetInDataAnchor(dst_opio.GetIndex());
          GE_CHK_BOOL_EXEC(dst_anchor != nullptr,
                           REPORT_INNER_ERROR("E19999", "GetInDataAnchor failed, index:%d, op:%s",
                                              dst_opio.GetIndex(), op_desc->GetName().c_str());
                           return GRAPH_FAILED, "GetInDataAnchor failed, index:%d", dst_opio.GetIndex());

          const auto ret = GraphUtils::AddEdge(src_anchor, dst_anchor);
          GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS,
                           REPORT_CALL_ERROR("E19999", "add edge from node[%s][%d] to node[%s][%d] failed.",
                                             src_node_ptr->GetName().c_str(), src_anchor->GetIdx(),
                                             dst_node_info->second->GetName().c_str(), dst_anchor->GetIdx());
                           return GRAPH_FAILED, "[Add][Edge] from node[%s][%d] to node[%s][%d] failed.",
                           src_node_ptr->GetName().c_str(), src_anchor->GetIdx(),
                           dst_node_info->second->GetName().c_str(), dst_anchor->GetIdx());
        }
      }
      const auto out_control_anchor = src_node_ptr->GetOutControlAnchor();
      for (const auto &control_out : src_op_impl_ptr->control_output_link_) {
        const auto dst_node_info = all_nodes_info_.find(control_out.lock());
        if (dst_node_info == all_nodes_info_.end()) {
          REPORT_INNER_ERROR("E19999", "Find Dst node failed.");
          GELOGE(GRAPH_FAILED, "[Check][Param] Find Dst node failed.");
          return GRAPH_FAILED;
        }
        GE_IF_BOOL_EXEC(dst_node_info->second == nullptr, continue);
        const auto in_control_anchor = dst_node_info->second->GetInControlAnchor();
        const auto ret = GraphUtils::AddEdge(out_control_anchor, in_control_anchor);
        if (ret != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "add edge failed. srcNode %s:%s, dstNode %s:%s", op_desc->GetName().c_str(),
                            op_desc->GetType().c_str(), dst_node_info->second->GetName().c_str(),
                            dst_node_info->second->GetType().c_str());
          GELOGE(ret, "[Add][Edge] failed. srcNode %s:%s, dstNode %s:%s", op_desc->GetName().c_str(),
                 op_desc->GetType().c_str(), dst_node_info->second->GetName().c_str(),
                 dst_node_info->second->GetType().c_str());
          return ret;
        }
      }
    }
    return GRAPH_SUCCESS;
  }

  ComputeGraphPtr graph_ = nullptr;
  std::map<OperatorImplPtr, NodePtr> all_nodes_info_{};
};

inline bool HasSameNameNode(const ComputeGraphPtr &compute_graph) {
  for (const auto &graph : compute_graph->GetAllSubgraphs()) {
    std::set<std::string> node_names;
    for (auto const &node : graph->GetDirectNode()) {
      const auto result = node_names.insert(node->GetName());
      if (!result.second) {
        REPORT_INNER_ERROR("E19999", "[Check][Param] graph %s has same name node%s",
                           graph->GetName().c_str(), node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] graph %s has same name node%s",
               graph->GetName().c_str(), node->GetName().c_str());
        return true;
      }
    }
  }

  std::set<std::string> node_names;
  for (auto const &node : compute_graph->GetDirectNode()) {
    auto result = node_names.insert(node->GetName());
    if (!result.second) {
      REPORT_INNER_ERROR("E19999", "[Check][Param] graph %s has same name node%s",
                         compute_graph->GetName().c_str(), node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] graph %s has same name node%s",
             compute_graph->GetName().c_str(), node->GetName().c_str());
      return true;
    }
  }
  return false;
}

ComputeGraphPtr GraphUtils::CreateGraphFromOperator(const std::string &name, const std::vector<ge::Operator> &inputs) {
  auto graph_builder_impl = GraphBuilderImpl(name);
  ComputeGraphPtr compute_graph = graph_builder_impl.BuildGraph(inputs);
  GE_CHK_BOOL_EXEC(compute_graph != nullptr,
                   REPORT_INNER_ERROR("E19999", "BuildGraph failed, as return nullptr.");
                   return compute_graph, "[Build][Graph] Computer graph is nullptr");
  compute_graph->SetAllNodesInfo(graph_builder_impl.GetAllNodesInfo());
  if (HasSameNameNode(compute_graph)) {
    GELOGW("[CreateGraph][Check] Nodes with same name exist in one compute graph is not allowed, graph_name: %s",
           name.c_str());
    compute_graph = nullptr;
  }

  return compute_graph;
}

void GraphUtils::BreakConnect(const std::map<OperatorImplPtr, NodePtr> &all_nodes_infos) {
  for (const auto &it : all_nodes_infos) {
    const OperatorImplPtr op_impl = it.first;
    if (op_impl == nullptr) {
      GELOGW("[BreakConnect][Check] Operator impl is null");
      continue;
    }
    op_impl->ClearOutputLinks();
    op_impl->ClearInputLinks();
    OperatorKeeper::GetInstance().CheckOutOperator(op_impl);
  }
}
} // namespace ge
