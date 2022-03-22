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

#ifndef REGISTER_OP_TILING_OP_TILING_CONSTANTS_H_
#define REGISTER_OP_TILING_OP_TILING_CONSTANTS_H_

#include <string>
#include <map>
#include "graph/types.h"

namespace optiling {
static const std::string COMPILE_INFO_JSON = "compile_info_json";
static const std::string COMPILE_INFO_KEY = "compile_info_key";
static const std::string COMPILE_INFO_WORKSPACE_SIZE_LIST = "_workspace_size_list";
static const std::string ATOMIC_COMPILE_INFO_JSON = "_atomic_compile_info_json";
static const std::string ATOMIC_COMPILE_INFO_KEY = "_atomic_compile_info_key";
static const std::string ATTR_NAME_ATOMIC_CLEAN_WORKSPACE = "_optiling_atomic_add_mem_size";
static const std::string ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";
static const std::string OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN = "DynamicAtomicAddrClean";
static const std::string OP_TYPE_AUTO_TILING = "AutoTiling";

static const std::map<ge::DataType, std::string> DATATYPE_STRING_MAP {
    {ge::DT_FLOAT, "float32"},
    {ge::DT_FLOAT16, "float16"},
    {ge::DT_INT8, "int8"},
    {ge::DT_INT16, "int16"},
    {ge::DT_INT32, "int32"},
    {ge::DT_INT64, "int64"},
    {ge::DT_UINT8, "uint8"},
    {ge::DT_UINT16, "uint16"},
    {ge::DT_UINT32, "uint32"},
    {ge::DT_UINT64, "uint64"},
    {ge::DT_BOOL, "bool"},
    {ge::DT_DOUBLE, "double"},
    {ge::DT_DUAL, "dual"},
    {ge::DT_DUAL_SUB_INT8, "dual_sub_int8"},
    {ge::DT_DUAL_SUB_UINT8, "dual_sub_uint8"}
};

}  // namespace optiling

#endif  // REGISTER_OP_TILING_OP_TILING_CONSTANTS_H_
