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

#include "graph/passes/bitcast_pass.h"

#include <vector>
#include <string>
#include "common/ge/ge_util.h"
#include "graph/utils/type_utils.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/formats/utils/formats_trans_utils.h"

namespace ge {
namespace {
const char *const kAttrNameType = "type";
}  // namespace

Status BitcastPass::Run(NodePtr &node) {
  GELOGD("Bitcast running");
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Param [node] must not be null.");
    return PARAM_INVALID;
  }

  if (node->GetType() != BITCAST) {
    return SUCCESS;
  }

  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param op_desc of node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] failed, op_desc of node is nullptr");
    return PARAM_INVALID;
  }
  ge::DataType dst_data_type;
  if (CheckDstDataType(op_desc, dst_data_type) != SUCCESS) {
    return PARAM_INVALID;
  }

  if (CheckOutputShape(op_desc, dst_data_type) != SUCCESS) {
    return PARAM_INVALID;
  }

  return IsolateAndDeleteNode(node, {0});
}

Status BitcastPass::CheckDstDataType(const OpDescPtr op_desc, ge::DataType &dst_data_type) {

  if (!ge::AttrUtils::GetDataType(op_desc, kAttrNameType, dst_data_type)) {
    REPORT_CALL_ERROR("E19999", "Get Attr:%s of op:%s(%s) failed",
                      kAttrNameType, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Get][Attr] %s of op:%s(%s) failed",
           kAttrNameType, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  if (dst_data_type >= ge::DT_UNDEFINED) {
    REPORT_INNER_ERROR("E19999", "Param dst_data_type:%d check invalid, op:%s(%s)",
                       dst_data_type, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] dst_data_type[%s] is not valid, op:%s(%s)",
           TypeUtils::DataTypeToSerialString(dst_data_type).c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  if (op_desc->GetOutputDescPtr(0) == nullptr) {
    REPORT_INNER_ERROR("E19999", "Index 0 ouput desc of op:%s(%s) not exist, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Get][OutputDescPtr] Index 0 ouput desc of op:%s(%s) not exist.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  if (op_desc->GetOutputDescPtr(0)->GetDataType() != dst_data_type) {
    REPORT_INNER_ERROR("E19999", "Index 0 ouput desc of op:%s(%s), it't data type:%s not equal to dst_data_type:%s, "
                       "check invalid", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       TypeUtils::DataTypeToSerialString(dst_data_type).c_str(),
                       TypeUtils::DataTypeToSerialString(op_desc->GetOutputDescPtr(0)->GetDataType()).c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] dst_data_type[%s] is not equal to output_data_type[%s], op:%s(%s).",
           TypeUtils::DataTypeToSerialString(dst_data_type).c_str(),
           TypeUtils::DataTypeToSerialString(op_desc->GetOutputDescPtr(0)->GetDataType()).c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status BitcastPass::CheckOutputShape(const OpDescPtr op_desc, const ge::DataType dst_data_type) {
  const GeTensorDescPtr &input_tensor_desc = op_desc->MutableInputDesc(0);
  const GeTensorDescPtr &output_tensor_desc = op_desc->MutableOutputDesc(0);
  if (input_tensor_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Index 0 input desc of op:%s(%s) not exist, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] input_tensor_desc(index:0) of op:%s(%s) must not be null.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  // get origin data_type and shape
  ge::DataType ori_data_type = input_tensor_desc->GetDataType();
  if (ori_data_type >= ge::DT_UNDEFINED) {
    REPORT_INNER_ERROR("E19999", "ori_data_type:%d of index 0 input desc in op:%s(%s), "
                       "check invalid",
                       ori_data_type, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] ori_data_type[%s] of input desc(index 0) in op:%s(%s) is not valid.",
           TypeUtils::DataTypeToSerialString(ori_data_type).c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  if (ori_data_type == dst_data_type) {
    GELOGW("Origin data type is equal to dest data type.");
    return SUCCESS;
  }

  BitcastPass::kVecInt64 dim_vec(input_tensor_desc->GetShape().GetDims());
  if (CalcAndUpdateShape(dim_vec, ori_data_type, dst_data_type) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Call][CalcAndUpdateShape] failed.");
    return PARAM_INVALID;
  }

  if (dim_vec != output_tensor_desc->GetShape().GetDims()) {
    REPORT_INNER_ERROR("E19999", "Shape:%s of index 0 output desc in op:%s(%s), different from expect shape:%s ,"
                       "check invalid",
                       formats::JoinToString(output_tensor_desc->GetShape().GetDims()).c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), formats::JoinToString(dim_vec).c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Shape:%s of index 0 output desc in op:%s(%s), different from expect shape:%s",
           formats::JoinToString(output_tensor_desc->GetShape().GetDims()).c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), formats::JoinToString(dim_vec).c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status BitcastPass::CalcAndUpdateShape(BitcastPass::kVecInt64 &dim_vec, ge::DataType ori_data_type,
                                       ge::DataType dst_data_type) {
  if (dim_vec.size() == 0) {
    REPORT_INNER_ERROR("E19999", "Param dim_vec is empty, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Param dim_vec is empty.");
    return PARAM_INVALID;
  }
  int64_t ori_data_size = GetSizeByDataType(ori_data_type);
  int64_t dst_data_size = GetSizeByDataType(dst_data_type);

  if (ori_data_size == dst_data_size) {
    return SUCCESS;
  } else if (ori_data_size > dst_data_size) {
    if (ori_data_size % dst_data_size != 0) {
      REPORT_INNER_ERROR("E19999", "size:%ld of ori_data_type:%s is not divisible by size:%ld of dst_data_type:%s ,"
                         "check invalid",
                         ori_data_size, TypeUtils::DataTypeToSerialString(ori_data_type).c_str(),
                         dst_data_size, TypeUtils::DataTypeToSerialString(dst_data_type).c_str());
      GELOGE(PARAM_INVALID,
             "[Check][Param] size:%ld of ori_data_type:%s is not divisible by size:%ld of dst_data_type:%s.",
             ori_data_size, TypeUtils::DataTypeToSerialString(ori_data_type).c_str(),
             dst_data_size, TypeUtils::DataTypeToSerialString(dst_data_type).c_str());
      return PARAM_INVALID;
    }
    dim_vec.push_back(ori_data_size / dst_data_size);
    return SUCCESS;
  } else {
    if (dst_data_size % ori_data_size != 0) {
      REPORT_INNER_ERROR("E19999", "size:%ld of dst_data_type:%s is not divisible by size:%ld of ori_data_type:%s ,"
                         "check invalid",
                         dst_data_size, TypeUtils::DataTypeToSerialString(dst_data_type).c_str(),
                         ori_data_size, TypeUtils::DataTypeToSerialString(ori_data_type).c_str());
      GELOGE(PARAM_INVALID,
             "[Check][Param] size:%ld of dst_data_type:%s is not divisible by size:%ld of ori_data_type:%s.",
             dst_data_size, TypeUtils::DataTypeToSerialString(dst_data_type).c_str(),
             ori_data_size, TypeUtils::DataTypeToSerialString(ori_data_type).c_str());
      return PARAM_INVALID;
    }

    if (dim_vec[dim_vec.size() - 1] != (dst_data_size / ori_data_size)) {
      REPORT_INNER_ERROR("E19999", "The last dim:%ld in param dim_vec is not equal to "
                         "dst_data_size:%ld / ori_data_size:%ld, check invalid",
                         dim_vec[dim_vec.size() - 1], dst_data_size, ori_data_size);
      GELOGE(PARAM_INVALID, "[Check][Param] The last dim:%ld in param dim_vec is not equal to "
             "dst_data_size:%ld / ori_data_size:%ld",
             dim_vec[dim_vec.size() - 1], dst_data_size, ori_data_size);
      return PARAM_INVALID;
    }
    dim_vec.pop_back();
  }
  return SUCCESS;
}

} // namespace ge
