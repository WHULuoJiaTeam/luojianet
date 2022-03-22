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

#include "graph/manager/util/hcom_util.h"

#include "framework/common/debug/log.h"
#include "common/math/math_util.h"
#include "framework/common/op/attr_value_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
Status HcomOmeUtil::GetHcclDataType(const ge::ConstOpDescPtr &op_desc,
                                    std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (CheckKernelHcclInfo(op_desc, kernel_hccl_infos) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][KernelHcclInfo] failed, op:%s(%s).",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  GELOGI("GetHcclDataType start, node[%s], opType[%s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }
  ge::DataType src_data_type = ge::DT_FLOAT;
  for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
    if (op_desc->GetType() == HCOMRECEIVE) {
      bool ret = ge::AttrUtils::GetDataType(op_desc, HCOM_ATTR_DATA_TYPE, src_data_type);
      if (ret == false) {
        REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail", HCOM_ATTR_DATA_TYPE.c_str(),
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(PARAM_INVALID, "[Get][Attr] %s in op:%s(%s) fail", HCOM_ATTR_DATA_TYPE.c_str(),
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return PARAM_INVALID;
      }
    } else {
      auto input_desc_ptr = op_desc->GetInputDescPtr(i);
      GE_CHECK_NOTNULL(input_desc_ptr);
      src_data_type = input_desc_ptr->GetDataType();
    }

    auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
    if (iter == kConstOpHcclDataType.end()) {
      REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s), value data_type:%s, not support in kConstOpHcclDataType now, "
                         "check invalid", HCOM_ATTR_DATA_TYPE.c_str(), op_desc->GetName().c_str(),
                         op_desc->GetType().c_str(), ge::TypeUtils::DataTypeToSerialString(src_data_type).c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in op:%s(%s), value data_type:%s, "
             "not support in kConstOpHcclDataType now", HCOM_ATTR_DATA_TYPE.c_str(), op_desc->GetName().c_str(),
             op_desc->GetType().c_str(), ge::TypeUtils::DataTypeToSerialString(src_data_type).c_str());
      return PARAM_INVALID;
    }

    kernel_hccl_infos[i].dataType = iter->second;
  }
  return SUCCESS;
}

Status HcomOmeUtil::GetHcclTypeSize(HcclDataType data_type, int32_t &size) {
  auto iter = kConstOpHcclDataTypeSize.find(data_type);
  GE_CHK_BOOL_EXEC(iter != kConstOpHcclDataTypeSize.end(), return PARAM_INVALID,
                   "[Check][Param] param data_type:%d not find", data_type);

  size = iter->second;
  return SUCCESS;
}

Status HcomOmeUtil::GetHcomCount(const ge::ConstOpDescPtr &op_desc, HcclDataType data_type, bool is_allgather,
                                 int &count) {
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHCOMOp(op_desc->GetType())) {
    REPORT_INNER_ERROR("E19999", "Op:%s(%s) is not hcom op, check invalid", op_desc->GetName().c_str(),
                       op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Op:%s(%s) is not hcom op", op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  int64_t total_size = 0;
  int64_t align_size = 512;
  int32_t size = 0;
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclTypeSize(data_type, size), "[Get][HcclTypeSize] fail, datatype:%d", data_type);
  if (op_desc->GetType() == HCOMRECEIVE) {
    for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
      int64_t output_size = 0;
      GE_CHECK_NOTNULL(op_desc->GetOutputDescPtr(i));
      GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetOutputDescPtr(i), output_size),
                        "[Get][Size] from TensorDesc failed, op:%s, output index:%zu.", op_desc->GetName().c_str(), i);
      output_size = (output_size + align_size - 1) / align_size * align_size;
      total_size += output_size;
    }
  } else {
    for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
      int64_t input_size = 0;
      int64_t block_size = 0;
      GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(i));
      GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetInputDescPtr(i), input_size),
                        "[Get][Size] from TensorDesc failed, op:%s, input index:%zu", op_desc->GetName().c_str(), i);
      // dynamic shape hccl op get size from output tensor desc
      if (op_desc->HasAttr(ATTR_NAME_IS_UNKNOWN_SHAPE) && (op_desc->GetOutputDescPtr(i) != nullptr)) {
        GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetOutputDescPtr(i), input_size),
                          "[Get][Size] from TensorDesc failed, op:%s, input index:%zu", op_desc->GetName().c_str(), i);
      }

      GE_IF_BOOL_EXEC(
          op_desc->GetType() == HCOMREDUCESCATTER, int32_t rank_size = 0;
          GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetInt(op_desc, HCOM_ATTR_RANK_SIZE, rank_size), PARAM_INVALID,
                                 "[Get][Attr] %s in op:%s(%s) failed", HCOM_ATTR_RANK_SIZE.c_str(),
                                 op_desc->GetName().c_str(), op_desc->GetType().c_str());
          GE_CHK_BOOL_RET_STATUS(rank_size != 0, PARAM_INVALID, "[Check][Param] rank size is zero");
          int64_t shape_size = op_desc->GetInputDescPtr(i)->GetShape().GetShapeSize();
          GE_CHK_STATUS_RET(ge::CheckInt64Uint32MulOverflow(shape_size, size),
                            "[Check][Param] Product of shape size:%ld and size:%d beyond INT64_MAX, op:%s(%s)",
                            shape_size, size, op_desc->GetName().c_str(), op_desc->GetType().c_str());
          block_size = (shape_size * size) / rank_size;
          GE_CHK_STATUS_RET(ge::CheckInt64AddOverflow(total_size, block_size),
                            "[Check][Param] Total size:%ld is beyond the INT64_MAX, op:%s(%s)",
                            total_size, op_desc->GetName().c_str(), op_desc->GetType().c_str());
          total_size = total_size + block_size; continue;);

      int64_t shape_size = op_desc->GetInputDescPtr(i)->GetShape().GetShapeSize();
      GELOGD("hcom util node %s inputsize %ld, shapesize %ld, datasize %d.",
             op_desc->GetName().c_str(), input_size, shape_size, size);
      GE_CHK_STATUS_RET(ge::CheckInt64Int32MulOverflow(shape_size, size),
                        "[Check][Param] Product of shape size:%ld and size:%d beyond INT64_MAX", shape_size, size);
      GE_IF_BOOL_EXEC(is_allgather, block_size = shape_size * size;);
      GE_IF_BOOL_EXEC(!is_allgather, block_size = (input_size + align_size - 1) / align_size * align_size;);
      GE_CHK_STATUS_RET(ge::CheckInt64AddOverflow(total_size, block_size),
                        "[Check][Param] Total size:%ld is beyond the INT64_MAX", total_size);
      total_size = total_size + block_size;
    }
  }

  GE_CHK_BOOL_RET_STATUS(size != 0, PARAM_INVALID, "[Check][Param] Size is zero");
  count = static_cast<int>(total_size / size);

  GE_CHK_BOOL_EXEC(total_size % size == 0, return PARAM_INVALID,
                   "[Check][Param] total_size:%ld is not divisiable by size:%d.", total_size, size);

  return SUCCESS;
}

Status HcomOmeUtil::GetHorovodCount(const ge::ConstOpDescPtr &op_desc,
                                    std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHorovodOp(op_desc->GetType())) {
    REPORT_INNER_ERROR("E19999", "Op:%s(%s) is not horovod op, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Call][IsHorovodOp] failed, Op:%s(%s) is not horovod op",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  int64_t align_size = 512;
  int32_t size = 0;
  for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclTypeSize(static_cast<HcclDataType>(kernel_hccl_infos[i].dataType), size),
                      "[Call][GetHcclTypeSize] fail, op:%s(%s)",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    int64_t input_size = 0;
    int64_t block_size = 0;
    GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(i));
    GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetInputDescPtr(i), input_size),
                      "[Get][Size] from TensorDesc failed, op:%s, input index:%zu", op_desc->GetName().c_str(), i);

    int64_t shape_size = op_desc->GetInputDescPtr(i)->GetShape().GetShapeSize();
    GE_CHK_STATUS_RET(ge::CheckInt64Int32MulOverflow(shape_size, size),
                      "[Check][Param] Product of shape size:%ld and size:%d beyond INT64_MAX", shape_size, size);
    if (kernel_hccl_infos[0].hccl_type == HVDCALLBACKALLGATHER) {
      block_size = shape_size * size;
    } else {
      block_size = (input_size + align_size - 1) / align_size * align_size;
    }

    GE_CHK_BOOL_RET_STATUS(size != 0, PARAM_INVALID, "[Check][Param] Size is zero");
    GE_CHK_BOOL_EXEC(block_size % size == 0, return PARAM_INVALID,
                     "[Check][Param] block_size:%ld is not divisiable by size:%d.", block_size, size);
    kernel_hccl_infos[i].count = static_cast<int>(block_size / size);
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclCount(const ge::ConstOpDescPtr &op_desc,
                                 std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  Status ret;
  ret = CheckKernelHcclInfo(op_desc, kernel_hccl_infos);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][KernelHcclInfo] failed, the number of GETaskKernelHcclInfo is invalid, op:%s(%s).",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  GELOGI("GetHcclCount start, node[%s], opType[%s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (IsHCOMOp(op_desc->GetType())) {
    int32_t count = 0;
    ret = GetHcomCount(op_desc, static_cast<HcclDataType>(kernel_hccl_infos[0].dataType),
                       kernel_hccl_infos[0].hccl_type == HCOMALLGATHER, count);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][GetHcomCount] Node:%s Optype:%s get the Hcom operator hccl count fail.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
    kernel_hccl_infos[0].count = count;
  }

  if (IsHorovodOp(op_desc->GetType())) {
    ret = GetHorovodCount(op_desc, kernel_hccl_infos);
    if (ret != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Call][GetHorovodCount] Node:%s Optype:%s get the Horovod hccl operator count fail.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclOperationType(const ge::ConstOpDescPtr &op_desc, HcclReduceOp &op_type) {
  GE_CHECK_NOTNULL(op_desc);

  if (IsHCOMOp(op_desc->GetType())) {
    std::string hcom_op_type;
    GE_CHK_BOOL_EXEC(ge::AttrUtils::GetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, hcom_op_type),
                     REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail", HCOM_ATTR_REDUCE_TYPE.c_str(),
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     return PARAM_INVALID,
                     "[Get][Attr] %s in op:%s(%s) fail", HCOM_ATTR_REDUCE_TYPE.c_str(),
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());

    if (hcom_op_type == "min") {
      op_type = HCCL_REDUCE_MIN;
    } else if (hcom_op_type == "max") {
      op_type = HCCL_REDUCE_MAX;
    } else if (hcom_op_type == "prod") {
      op_type = HCCL_REDUCE_PROD;
    } else if (hcom_op_type == "sum") {
      op_type = HCCL_REDUCE_SUM;
    } else {
      REPORT_INNER_ERROR("E19999", "Attr:%s in Op:%s(%s), hcom_op_type value:%s is not support now, "
                         "check invalid", HCOM_ATTR_REDUCE_TYPE.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), hcom_op_type.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in Op:%s(%s), hcom_op_type value:%s is not support now",
             HCOM_ATTR_REDUCE_TYPE.c_str(), op_desc->GetName().c_str(),
             op_desc->GetType().c_str(), hcom_op_type.c_str());
      return PARAM_INVALID;
    }
  }

  if (IsHorovodOp(op_desc->GetType())) {
    int64_t horovod_op_type;
    GE_CHK_BOOL_EXEC(ge::AttrUtils::GetInt(op_desc, ATTR_HOROVOD_ATTR_REDUCE_TYPE, horovod_op_type),
                     REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail",
                                        ATTR_HOROVOD_ATTR_REDUCE_TYPE.c_str(),
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     return PARAM_INVALID,
                     "[Get][Attr] %s in op:%s(%s) fail", ATTR_HOROVOD_ATTR_REDUCE_TYPE.c_str(),
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());

    auto iter = kHorovodRedOpToHcclRedOp.find(static_cast<HorovodReduceOp>(horovod_op_type));
    if (iter == kHorovodRedOpToHcclRedOp.end()) {
      REPORT_INNER_ERROR("E19999", "Attr:%s in Op:%s(%s), horovod_op_type value:%ld is not support now, "
                         "check invalid", ATTR_HOROVOD_ATTR_REDUCE_TYPE.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), horovod_op_type);
      GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in Op:%s(%s), horovod_op_type value:%ld is not support now",
             ATTR_HOROVOD_ATTR_REDUCE_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             horovod_op_type);
      return PARAM_INVALID;
    }
    op_type = iter->second;
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclRootId(const ge::ConstOpDescPtr &op_desc, int64_t &root_id) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::GetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id),
                   REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail",
                                      HCOM_ATTR_ROOT_RANK.c_str(),
                                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return PARAM_INVALID,
                   "[Get][Attr] %s in op:%s(%s) fail", HCOM_ATTR_ROOT_RANK.c_str(),
                   op_desc->GetName().c_str(), op_desc->GetType().c_str());

  return SUCCESS;
}

Status HcomOmeUtil::GetAllRootId(const ge::ConstOpDescPtr &op_desc,
                                 std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetType() == HCOMBROADCAST ||
      op_desc->GetType() == HVDCALLBACKBROADCAST || op_desc->GetType() == HCOMREDUCE) {
    GELOGI("GetAllRootId Node[%s] opType[%s] get hccl rootId.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    int64_t root_id = 0;
    Status dmrt = GetHcclRootId(op_desc, root_id);
    if (dmrt != SUCCESS) {
      GELOGE(FAILED, "[Get][HcclRootId] fail! domi error: %u", dmrt);
      return FAILED;
    }

    for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
      kernel_hccl_infos[i].rootId = root_id;
    }
  }
  return SUCCESS;
}

bool HcomOmeUtil::IsHCOMOp(const string &op_type) {
  return (op_type == HCOMALLREDUCE) || (op_type == HCOMALLGATHER) || (op_type == HCOMBROADCAST) ||
         (op_type == HCOMSEND) || (op_type == HCOMRECEIVE) || (op_type == HCOMREDUCESCATTER) || (op_type == HCOMREDUCE);
}

bool HcomOmeUtil::IsHorovodOp(const string &op_type) {
  return (op_type == HVDCALLBACKALLREDUCE) || (op_type == HVDCALLBACKALLGATHER) || (op_type == HVDCALLBACKBROADCAST) ||
         (op_type == HVDWAIT);
}

Status HcomOmeUtil::CheckKernelHcclInfo(const ge::ConstOpDescPtr &op_desc,
                                        std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (IsHCOMOp(op_desc->GetType()) && kernel_hccl_infos.size() != 1) {
    REPORT_INNER_ERROR("E19999", "Op:%s(%s) is not hcom op or param kernel_hccl_infos.size:%zu != 1, "
                       "check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_hccl_infos.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Op:%s(%s) is not hcom op or param kernel_hccl_infos.size:%zu != 1",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_hccl_infos.size());
    return PARAM_INVALID;
  }

  if (IsHorovodOp(op_desc->GetType())) {
    if (op_desc->GetType() == HVDWAIT) {
      return SUCCESS;
    }
    if (kernel_hccl_infos.empty() || op_desc->GetInputsSize() != kernel_hccl_infos.size()) {
      REPORT_INNER_ERROR("E19999", "Param kernel_hccl_infos.size:%zu is empty or not equal to input_desc size:%zu "
                         "in op:%s(%s), check invalid",
                         kernel_hccl_infos.size(), op_desc->GetInputsSize(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(PARAM_INVALID, "Param kernel_hccl_infos.size:%zu is empty or not equal to "
             "input_desc size:%zu in op:%s(%s)", kernel_hccl_infos.size(), op_desc->GetInputsSize(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

void HcomOmeUtil::GetHcclType(const domi::TaskDef &task_def, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  auto hccl_def = task_def.kernel_hccl();
  std::string hccl_type = hccl_def.hccl_type();
  for (size_t i = 0; i < kernel_hccl_infos.size(); i++) {
    kernel_hccl_infos[i].hccl_type = hccl_type;
  }
}

Status HcomOmeUtil::GetHorovodInputs(const ge::ConstOpDescPtr &op_desc,
                                     std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHorovodOp(op_desc->GetType())) {
    return SUCCESS;
  }

  if (CheckKernelHcclInfo(op_desc, kernel_hccl_infos) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][KernelHcclInfo] Node:%s Optype:%s the number of GETaskKernelHcclInfo is invalid.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }

  for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
    ConstGeTensorDescPtr input_desc = op_desc->GetInputDescPtr(i);
    GETaskKernelHcclInfo &kernel_hccl_info = kernel_hccl_infos.at(i);
    kernel_hccl_info.input_name = op_desc->GetInputNameByIndex(i);
    kernel_hccl_info.dims = input_desc->GetShape().GetDims();
  }
  return SUCCESS;
}
}  // namespace ge
