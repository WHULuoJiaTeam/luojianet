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
#ifndef FRAMEWORK_DOMI_ATC_IR_COMMON_H_
#define FRAMEWORK_DOMI_ATC_IR_COMMON_H_

#include <unistd.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <set>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/preprocess/multi_batch_options.h"

namespace ge {
const char *const kModifyMixlistError = "modify_mixlist is assigned, please ensure that "
                                        "precision_mode is assigned to 'allow_mix_precision'";
static std::set<std::string> caffe_support_input_format = {"NCHW", "ND"};
static std::set<std::string> tf_support_input_format = {"NCHW", "NHWC", "ND", "NCDHW", "NDHWC"};
static std::set<std::string> onnx_support_input_format = {"NCHW", "ND", "NCDHW"};

static std::map<std::string, domiTensorFormat_t> input_format_str_to_geformat = {
    {"ND", domi::DOMI_TENSOR_ND},
    {"NCHW", domi::DOMI_TENSOR_NCHW},
    {"NHWC", domi::DOMI_TENSOR_NHWC},
    {"CHWN", domi::DOMI_TENSOR_CHWN},
    {"NC1HWC0", domi::DOMI_TENSOR_NC1HWC0},
    {"NHWC1C0", domi::DOMI_TENSOR_NHWC1C0},
    {"NCDHW", domi::DOMI_TENSOR_NCDHW},
    {"NDHWC", domi::DOMI_TENSOR_NDHWC}
};
static const std::string kEnableCompressWeightTrue = "1";
static const std::string kEnableCompressWeightFalse = "0";

bool CheckDynamicBatchSizeInputShapeValid(map<string, vector<int64_t>> shape_map,
                                          std::string &dynamic_batch_size);

bool CheckDynamicImagesizeInputShapeValid(map<string, vector<int64_t>> shape_map,
                                          const std::string input_format, std::string &dynamic_image_size);

bool CheckDynamicDimsInputShapeValid(const std::map<std::string, std::vector<int64_t>> &shape_map,
                                     std::string input_format, std::string &dynamic_dims);

bool CheckAndParseDynamicDims(int32_t dynamic_dim_num, std::string &dynamic_dims);

Status CheckDynamicInputParamValid(std::string &dynamic_batch_size, std::string &dynamic_image_size,
                                   std::string &dynamic_dims, const std::string input_shape,
                                   const std::string input_shape_range, const std::string input_format,
                                   bool &is_dynamic_input);

bool ParseInputShape(const std::string &input_shape, std::map<string, std::vector<int64_t>> &shape_map,
                     std::vector<std::pair<string, vector<int64_t>>> &user_shape_map, bool is_dynamic_input = false);
Status ParseInputShapeRange(const std::string &shape_range,
                            std::map<string, std::vector<std::pair<int64_t, int64_t>>> &shape_range_map);
Status ParseInputShapeRange(const std::string &shape_range,
                            std::vector<std::vector<std::pair<int64_t, int64_t>>> &range);

Status CheckOutputTypeParamValid(const std::string output_type);
Status CheckBufferOptimizeParamValid(const std::string buffer_optimize);
Status CheckCompressWeightParamValid(const std::string enable_compress_weight, const std::string compress_weight_conf);
int CheckLogParamValidAndSetLogLevel(const std::string log);
Status CheckInsertOpConfParamValid(const std::string insert_op_conf);
Status CheckDisableReuseMemoryParamValid(const std::string disable_reuse_memory);
Status CheckEnableSingleStreamParamValid(const std::string enable_single_stream);
Status CheckImplmodeParamValid(const std::string &optypelist_for_implmode, std::string &op_select_implmode);
Status CheckModifyMixlistParamValid(const std::map<std::string, std::string> &options);
Status CheckModifyMixlistParamValid(const std::string &precision_mode, const std::string &modify_mixlist);
Status CheckInputFormat(const string &input_format);
Status CheckKeepTypeParamValid(const std::string &keep_dtype);
void PrintOptionMap(std::map<std::string, std::string> &options, std::string tips);
void EraseEndSemicolon(std::string &param);
Status UpdateDataOpShape(const OpDescPtr &op, std::map<std::string, std::vector<int64_t>> &shape_map);
Status UpdateDataOpShapeRange(
    const OpDescPtr &op, const std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> &name_shape_range_map);
Status UpdateDataOpShapeRange(const OpDescPtr &op,
                              const std::vector<std::vector<std::pair<int64_t, int64_t>>> &index_shape_range_map);
Status UpdateDynamicInputShapeRange(const ge::ComputeGraphPtr &compute_graph, const string &input_shape_range);
}
#endif  // FRAMEWORK_DOMI_ATC_IR_COMMON_H_
