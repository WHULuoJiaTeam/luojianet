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

#ifndef INC_FRAMEWORK_OMG_OMG_INNER_TYPES_H_
#define INC_FRAMEWORK_OMG_OMG_INNER_TYPES_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "framework/common/fmk_error_codes.h"
#include "register/register_fmk_types.h"
#include "graph/node.h"

using domi::DOMI_TENSOR_ND;
using domi::DOMI_TENSOR_RESERVED;
using domi::domiTensorFormat_t;
using std::unordered_map;

namespace ge {
/**
 * @ingroup domi_omg
 * @brief run model
 */
enum RunMode {
  GEN_OM_MODEL = 0,    // generate offline model file
  MODEL_TO_JSON = 1,   // convert to JSON file
  ONLY_PRE_CHECK = 3,  // only for pre-check
  PBTXT_TO_JSON = 5,   // pbtxt to json
  DISPLAY_OM_INFO = 6  // display model info
};

struct OmgContext {
  OmgContext() : format(domi::DOMI_TENSOR_ND) {}
  domi::domiTensorFormat_t format;

  // format of the input specified by the command line
  std::unordered_map<std::string, domi::domiTensorFormat_t> input_nodes_format_map;
  std::vector<domi::domiTensorFormat_t> output_formats;

  // user-designate input dims
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  // global input dims
  std::map<std::string, std::vector<int64_t>> input_dims;

  // resolve the mapping between operators with the same name and corresponding network. format e.g.
  // Detectionoutput:SsdDetectiontOutput
  std::map<std::string, std::string> op_conf_map;
  // save the output node of the network. key = operator name, value = index, index indicates the output index of the
  // operator
  std::map<std::string, std::vector<int32_t>> out_nodes_map;
  // user-designate out nodes (this is used for determing the orders)
  std::vector<std::pair<std::string, int32_t>> user_out_nodes;
  // default out nodes (this is used for determing the orders)
  std::vector<std::pair<std::string, int32_t>> default_out_nodes;
  // save the output node of the network, value = topName,
  // tensorName indicates the output name of the operator.
  std::vector<std::string> user_out_tensors;
  // net out nodes (where user_out_nodes or leaf nodes)
  std::vector<std::string> net_out_nodes;
  // net out nodes tensor names(caffe or onnx)
  std::vector<std::string> out_tensor_names;
  // net data nodes tensor names(caffe or onnx)
  std::vector<std::string> data_tensor_names;
  // preferential format used by the entire network
  domi::domiTensorFormat_t net_format = domi::DOMI_TENSOR_RESERVED;
  domi::FrameworkType type = domi::FRAMEWORK_RESERVED;
  RunMode run_mode = RunMode::ONLY_PRE_CHECK;
  bool train_flag = false;

  std::string output_type;

  // Whether to use dynamic batch size or dynamic image size
  bool is_dynamic_input = false;
  std::string dynamic_batch_size;
  std::string dynamic_image_size;
  std::string dynamic_dims;
  std::string dynamic_node_type;
  bool need_multi_batch = false;
  std::vector<NodePtr> data_nodes;
  std::vector<NodePtr> getnext_nosink_nodes;
  bool fuzz_compile_flag = false;
  std::string atc_cmdline;
  bool user_attr_index_valid = false;
  bool is_online_model = false;
};
}  // namespace ge

namespace domi {
/**
 * @ingroup domi_omg
 * @brief get OMG context
 * @return OmgContext context
 */
GE_FUNC_VISIBILITY ge::OmgContext &GetContext();

struct TEBinInfo {
  // It is obsolete. It will be automatically obtained from the binfilename field of the JSON file later.
  // To be compatible with use cases written by previous users, fields are not deleted.(2018.11.21)
  std::string bin_file_path;
  std::string json_file_path;
  std::string ddk_version;
};
}  // namespace domi

#endif  // INC_FRAMEWORK_OMG_OMG_INNER_TYPES_H_
