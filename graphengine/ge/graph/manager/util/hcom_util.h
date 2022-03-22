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

#ifndef GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_
#define GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_

#include <map>
#include <string>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/opskernel/ge_task_info.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/op_desc.h"
#include "hccl/hcom.h"
#include "proto/task.pb.h"

namespace ge {
using std::string;
using std::vector;

static std::map<int64_t, HcclDataType> kConstOpHcclDataType = {
    {ge::DT_FLOAT, HCCL_DATA_TYPE_FP32},
    {ge::DT_FLOAT16, HCCL_DATA_TYPE_FP16},
    {ge::DT_INT8, HCCL_DATA_TYPE_INT8},
    {ge::DT_INT32, HCCL_DATA_TYPE_INT32},
    {ge::DT_INT64, HCCL_DATA_TYPE_INT64},
    {ge::DT_UINT64, HCCL_DATA_TYPE_UINT64},
};

static std::map<HcclDataType, int32_t> kConstOpHcclDataTypeSize = {
    {HCCL_DATA_TYPE_FP32, sizeof(float)},
    {HCCL_DATA_TYPE_FP16, sizeof(float) / 2},
    {HCCL_DATA_TYPE_INT8, sizeof(int8_t)},
    {HCCL_DATA_TYPE_INT32, sizeof(int32_t)},
    {HCCL_DATA_TYPE_INT64, sizeof(int64_t)},
    {HCCL_DATA_TYPE_UINT64, sizeof(uint64_t)},
};

static std::map<HorovodReduceOp, HcclReduceOp> kHorovodRedOpToHcclRedOp = {
    {HOROVOD_REDUCE_SUM, HCCL_REDUCE_SUM},           {HOROVOD_REDUCE_MIN, HCCL_REDUCE_MIN},
    {HOROVOD_REDUCE_MAX, HCCL_REDUCE_MAX},           {HOROVOD_REDUCE_PROD, HCCL_REDUCE_PROD},
    {HOROVOD_REDUCE_RESERVED, HCCL_REDUCE_RESERVED},
};

class HcomOmeUtil {
 public:
  ///
  /// @ingroup domi_ome
  /// @brief GetHcclDataType
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcclDataType(const ge::ConstOpDescPtr &op_desc,
                                std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcclTypeSize
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcclTypeSize(HcclDataType data_type, int32_t &size);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcclCount
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcclCount(const ge::ConstOpDescPtr &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcclOperationType
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcclOperationType(const ge::ConstOpDescPtr &op_desc, HcclReduceOp &op_type);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcclRootId
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcclRootId(const ge::ConstOpDescPtr &op_desc, int64_t &root_id);

  ///
  /// @ingroup domi_ome
  /// @brief GetAllRootId
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetAllRootId(const ge::ConstOpDescPtr &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  ///
  /// @ingroup domi_ome
  /// @brief check the op_type whether is hcom operator or not
  /// @return true
  /// @return false
  ///
  static bool IsHCOMOp(const string &op_type);

  ///
  /// @ingroup domi_ome
  /// @brief check the op_type whether is horovod operator or not
  /// @return true
  /// @return false
  ///
  static bool IsHorovodOp(const string &op_type);

  ///
  /// @ingroup domi_ome
  /// @brief GetHcclType
  /// @return void
  ///
  static void GetHcclType(const domi::TaskDef &task_def, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  ///
  /// @ingroup domi_ome
  /// @brief CheckKernelHcclInfo
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status CheckKernelHcclInfo(const ge::ConstOpDescPtr &op_desc,
                                    std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);
  ///
  /// @ingroup domi_ome
  /// @brief GetHorovodInputs
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHorovodInputs(const ge::ConstOpDescPtr &op_desc,
                                 std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);
  ///
  /// @ingroup domi_ome
  /// @brief GetHcomCount
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHcomCount(const ge::ConstOpDescPtr &op_desc, HcclDataType data_type, bool is_allgather,
                             int &count);

 private:
  ///
  /// @ingroup domi_ome
  /// @brief GetHorovodCount
  /// @return SUCCESS
  /// @return FAIL
  ///
  static Status GetHorovodCount(const ge::ConstOpDescPtr &op_desc,
                                std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_
