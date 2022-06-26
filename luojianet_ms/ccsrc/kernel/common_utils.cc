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

#include "kernel/common_utils.h"
#include <unordered_map>
#include <map>
#include <bitset>
#include <iostream>
#include <utility>
#include <fstream>
#include <algorithm>
#include <thread>
#include "nlohmann/json.hpp"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "ir/manager.h"
#include "ir/meta_tensor.h"
#include "base/core_ops.h"
#include "ir/graph_utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "luojianet_ms/ccsrc/include/common/debug/common.h"

namespace luojianet_ms {
namespace kernel {
constexpr char kAxis[] = "axis";
constexpr char kTypeInt32[] = "Int32";
constexpr auto kStridedSliceMaxDims = 8;
const std::unordered_map<std::string, std::string> dtype_shortdtype_map_ = {
  {"float16", "f16"}, {"float32", "f32"}, {"float64", "f64"}, {"int8", "i8"},    {"int16", "i16"},  {"int32", "i32"},
  {"int64", "i64"},   {"uint8", "u8"},    {"uint16", "u16"},  {"uint32", "u32"}, {"uint64", "u64"}, {"bool", "bool"},
};

const std::unordered_map<std::string, size_t> dtype_nbyte_map = {
  {"float16", sizeof(float) / 2},  {"float32", sizeof(float)},  {"float64", sizeof(float) * 2},
  {"int8", sizeof(int) / 4},       {"int16", sizeof(int) / 2},  {"int32", sizeof(int)},
  {"int64", sizeof(int) * 2},      {"uint8", sizeof(int) / 4},  {"uint16", sizeof(int) / 2},
  {"uint32", sizeof(int)},         {"uint64", sizeof(int) * 2}, {"bool", sizeof(char)},
  {"complex64", sizeof(float) * 2}};

// Define all patterns here for different schedule
const std::unordered_map<FusionType, std::string> fusion_type_name_maps = {
  {FusionType::BN_UPDATE_GRAD, "bn_update_grad"},
  {FusionType::BN_GRAD_REDUCE, "bn_grad_reduce"},
  {FusionType::LAYER_NORM_GRAD, "layer_norm_grad"},
  {FusionType::L2LOSS_MUL_ADDN, "l2loss_mul_addn"},
  {FusionType::ELEMWISE, "ElemWise"},
  {FusionType::PURE_BROADCAST, "PureBroadcast"},
  {FusionType::COMMREDUCE, "CommReduce"},
  {FusionType::SEGMENT, "Segment"},
  {FusionType::INPLACE, "Inplace"},
  {FusionType::MATMUL, "Matmul"},
  {FusionType::MATMUL_V2, "Matmul_v2"},
  {FusionType::GEMM, "GEMM"},
  {FusionType::CONV, "Convolution"},
  {FusionType::CONV2D_BACKPROP_INPUT, "Conv2d_backprop_input"},
  {FusionType::CONV2D_BACKPROP_FILTER, "Conv2d_backprop_filter"},
  {FusionType::CONV3D_BACKPROP_INPUT, "Conv3d_backprop_input"},
  {FusionType::CONV3D_BACKPROP_FILTER, "Conv3d_backprop_filter"},
  {FusionType::CUBE_LAYER_NORM, "cube_layer_norm"},
  {FusionType::OPAQUE, "Opaque"},
  {FusionType::BN_REDUCE, "bn_reduce"},
  {FusionType::BN_UPDATE, "bn_update"},
  {FusionType::SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, "softmax_cross_entropy_with_logits"},
  {FusionType::L2_NORMALIZE, "l2_normalize"},
  {FusionType::SOFTMAX, "softmax_pattern"},
  {FusionType::L2_LOSS, "l2_loss"},
  {FusionType::ASCEND_QUANT, "quant"},
  {FusionType::ASCEND_DEQUANT, "dequant"},
  {FusionType::ASCEND_ANTI_QUANT, "anti_quant"},
  {FusionType::STRIDED_READ, "strided_read"},
  {FusionType::STRIDED_WRITE, "strided_write"},
  {FusionType::ASCEND_DEQUANT_S16, "dequant_s16"},
  {FusionType::ASCEND_REQUANT, "requant"},
  {FusionType::ASCEND_REQUANT_S16, "requant_s16"},
  {FusionType::MAX_POOL, "MaxPool"},
  {FusionType::DEPTHWISECONV, "DepthwiseConvolution"},
  {FusionType::CONV3D, "Conv3d"},
  {FusionType::POOL2D, "Pool2d"},
  {FusionType::POOL3D, "Pool3d"},
  {FusionType::READ_SELECT, "read_select"},
  {FusionType::WRITE_SELECT, "write_select"},
  {FusionType::COSINE_EMBEDDING_LOSS, "cosine_embedding_loss"},
  {FusionType::DILATION_PATTERN, "dilation"},
  {FusionType::BROAD_CAST, "Broadcast"},
  {FusionType::BATCH_MATMUL, "BatchMatmul"},
  {FusionType::CONFUSION_TRANSPOSE, "confusiontranspose"},
  {FusionType::DROPOUT_DOMASKV3D, "DropOutDoMaskV3D"},
  {FusionType::UNKNOWN_FUSION_TYPE, ""}};

std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> GetAlignments(const std::string &alignment) {
  auto alignment_iter = MatrixDiag::AlignmentMap.find(alignment);
  if (alignment_iter == MatrixDiag::AlignmentMap.end()) {
    MS_LOG(EXCEPTION) << "For  current kernel, input alignment is invalid: " << alignment
                      << ". please limit it to {RIGHT_LEFT, LEFT_RIGHT, RIGHT_RIGHT, LEFT_LEFT}";
  }
  return alignment_iter->second;
}

int CalDiagOffset(int diag_index, int max_diag_len, int inner_rows, int inner_cols,
                  const std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> &alignment) {
  bool right_align_super_diagonal = (alignment.first == MatrixDiag::RIGHT);
  bool right_align_sub_diagonal = (alignment.second == MatrixDiag::RIGHT);
  const bool right_align =
    (diag_index >= 0 && right_align_super_diagonal) || (diag_index <= 0 && right_align_sub_diagonal);
  const int diag_len = std::min(inner_rows + std::min(0, diag_index), inner_cols - std::max(0, diag_index));
  const int offset = (right_align) ? (max_diag_len - diag_len) : 0;
  return offset;
}

std::string GetFusionNameByType(const kernel::FusionType &type) {
  auto iter = fusion_type_name_maps.find(type);
  if (iter == fusion_type_name_maps.end()) {
    MS_LOG(EXCEPTION) << "Illegal fusion type: " << type;
  }
  return iter->second;
}

FusionType GetFusionTypeByName(const std::string &name) {
  std::string fusion_name_upper = name;
  transform(fusion_name_upper.begin(), fusion_name_upper.end(), fusion_name_upper.begin(), ::toupper);
  auto iter =
    std::find_if(fusion_type_name_maps.begin(), fusion_type_name_maps.end(), [&fusion_name_upper](const auto &it) {
      std::string name_upper = it.second;
      transform(name_upper.begin(), name_upper.end(), name_upper.begin(), ::toupper);
      return fusion_name_upper == name_upper;
    });
  if (iter == fusion_type_name_maps.end()) {
    MS_LOG(EXCEPTION) << "Illegal fusion name: " << name;
  }
  return iter->first;
}

std::string GetCompilerCachePath() { return Common::GetUserDefineCachePath(); }

void KernelMeta::Initialize() {
  auto config_path = GetCompilerCachePath();
  kernel_meta_path_ = config_path + std::string(kAkgKernelMeta);
  FileUtils::CreateNotExistDirs(kernel_meta_path_);
  initialized_ = true;
}

std::string KernelMeta::Search(const std::string &kernel_name) const {
  if (!initialized_) {
    return "";
  }

  auto iter = kernel_meta_map_.find(kernel_name);
  if (iter == kernel_meta_map_.end()) {
    return "";
  } else {
    return iter->second;
  }
}

bool KernelMeta::Insert(const std::string &kernel_name, const std::string &kernel_json) {
  if (!initialized_) {
    return false;
  }
  kernel_meta_map_[kernel_name] = kernel_json;
  return true;
}

bool CheckCache(const std::string &kernel_name) {
  // check cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "Kernel cache is invalid, kernel_name: " << kernel_name;
    return false;
  }
  std::string kernel_json = bin_map->Search(kernel_name);
  bool ret = (!kernel_json.empty());
  if (ret) {
    MS_LOG(INFO) << "Kernel name:" << kernel_name << " has registered.";
  } else {
    MS_LOG(INFO) << "Kernel name:" << kernel_name << " will been registered.";
  }
  return ret;
}

KernelPackPtr SearchCache(const std::string &kernel_name, const std::string &processor) {
  // search cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "kernel cache is invalid, kernel_name: " << kernel_name;
    return nullptr;
  }

  std::string kernel_json = bin_map->Search(kernel_name);
  if (!kernel_json.empty()) {
    KernelPackPtr kernel_pack = std::make_shared<KernelPack>();
    // just a tmp solution.
    if (!kernel_pack->ReadFromJsonFile(kernel_json, processor)) {
      MS_LOG(ERROR) << "Read cache json and bin file failed[" << kernel_json << "].";
      return nullptr;
    } else {
      return kernel_pack;
    }
  } else {
    MS_LOG(INFO) << "The cache kernel not found[" << kernel_name << "].";
    return nullptr;
  }
}

KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor) {
  MS_LOG(INFO) << "Insert cache for kernel:" << kernel_name << ", processr:" << processor;
  KernelMeta *bin_map = KernelMeta::GetInstance();
  std::string kernel_json = bin_map->kernel_meta_path();
  (void)kernel_json.append(kernel_name).append(kJsonSuffix);
  KernelPackPtr kernel_pack = std::make_shared<KernelPack>();
  if (!kernel_pack->ReadFromJsonFile(kernel_json, processor)) {
    MS_LOG(ERROR) << "Read json and bin file failed[" << kernel_json << "].";
    return nullptr;
  }

  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "Kernel cache is invalid, kernel name :" << kernel_name;
    return nullptr;
  }
  if (bin_map->Insert(kernel_name, kernel_json)) {
    MS_LOG(INFO) << "Kernel insert cache success[" << kernel_json << "], kernel name[" << kernel_name << "].";
  }
  return kernel_pack;
}

TypeId DtypeToTypeId(const std::string &dtypes) {
  if (dtypes == "float") {
    return TypeId::kNumberTypeFloat32;
  }
  if (dtypes.empty()) {
    return TypeId::kMetaTypeNone;
  }
  return StringToTypeId(dtypes);
}

std::string Dtype2ShortType(const std::string &dtype) {
  auto iter = dtype_shortdtype_map_.find(dtype);
  if (iter != dtype_shortdtype_map_.end()) {
    return iter->second;
  } else {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype:" << dtype;
  }
}

size_t GetDtypeNbyte(const std::string &dtype) {
  auto iter = dtype_nbyte_map.find(dtype);
  if (iter != dtype_nbyte_map.end()) {
    return iter->second;
  } else {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype:" << dtype;
  }
}

bool SetInputKernelBuilderInfo(const std::vector<std::shared_ptr<OpIOInfo>> &inputs, size_t real_input_num,
                               size_t builder_idex, const std::vector<int64_t> &dyn_input_sizes,
                               const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder) {
  MS_EXCEPTION_IF_NULL(builder);

  std::vector<TypeId> inputs_device_type;
  std::vector<std::string> inputs_format;
  size_t dyn_input_idx = 0;
  size_t kernel_info_index = 0;
  MS_EXCEPTION_IF_NULL(inputs[0]);
  size_t kernel_info_cnt = inputs[0]->dtypes().size();

  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    std::string param_type = input->param_type();
    std::vector<std::string> dtypes = input->dtypes();
    std::vector<std::string> formats = input->formats();
    if (dtypes.size() != kernel_info_cnt || formats.size() != kernel_info_cnt) {
      MS_LOG(DEBUG) << "Set input kernel builder info failed, dtyps size != formats size. dtypes size: "
                    << dtypes.size() << ", formats size : " << formats.size();
      return false;
    }

    if (param_type == "dynamic") {
      if (dyn_input_sizes.empty()) {
        MS_LOG(DEBUG) << "Set input kernel builder info failed, dyn_input_sizes's size is 0 when param_type is dynamic";
        return false;
      }

      for (int64_t t = 0; t < dyn_input_sizes[dyn_input_idx]; t++) {
        kernel_info_index++;
        auto type_id = DtypeToTypeId(dtypes[builder_idex]);
        inputs_device_type.push_back(type_id);
        inputs_format.push_back(formats[builder_idex]);
      }
      dyn_input_idx++;
    } else if (param_type == "required") {
      kernel_info_index++;
      auto type_id = DtypeToTypeId(dtypes[builder_idex]);
      inputs_device_type.push_back(type_id);
      inputs_format.push_back(formats[builder_idex]);
    } else {
      if (kernel_info_index < real_input_num) {
        MS_LOG(INFO) << "Set input kernel builder info, input type is optional, input index is :" << kernel_info_index;
        kernel_info_index++;
        auto type_id = DtypeToTypeId(dtypes[builder_idex]);
        inputs_device_type.push_back(type_id);
        inputs_format.push_back(formats[builder_idex]);
      }
    }
  }

  builder->SetInputsDeviceType(inputs_device_type);
  builder->SetInputsFormat(inputs_format);
  return true;
}

bool SetOutputKernelBuilderInfo(const std::vector<std::shared_ptr<OpIOInfo>> &outputs, size_t builder_idex,
                                const size_t &real_output_num,
                                const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder) {
  // not now but in the next we need to support dynamic output case
  MS_EXCEPTION_IF_NULL(builder);

  size_t output_idx = 0;
  std::vector<TypeId> outputs_device_type;
  std::vector<std::string> outputs_format;
  MS_EXCEPTION_IF_NULL(outputs[0]);
  size_t kernel_info_cnt = outputs[0]->dtypes().size();

  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    if (output_idx >= real_output_num) {
      MS_LOG(DEBUG) << "real_output_num:" << real_output_num << ", output_idx:" << output_idx << " is out of limit!";
      continue;
    }
    size_t output_num = 0;
    if (output->param_type() == "dynamic") {
      if (outputs.size() > 1) {
        MS_EXCEPTION(ArgumentError) << "Dynamic output is unsupported multi output!";
      }
      output_num = real_output_num;
    } else if (output->param_type() == "required") {
      output_num = 1;
    } else {
      if (output_idx < real_output_num) {
        MS_LOG(DEBUG) << "Set output kernel builder info, output type is optional, output index is :" << output_idx;
        output_num = 1;
      }
    }

    for (size_t i = 0; i < output_num; i++) {
      std::vector<std::string> dtypes = output->dtypes();
      std::vector<std::string> formats = output->formats();
      if (dtypes.size() != kernel_info_cnt || formats.size() != kernel_info_cnt) {
        MS_LOG(DEBUG) << "Set output kernel builder info, dtyps size != formats size.";
        return false;
      }
      auto type_id = DtypeToTypeId(dtypes[builder_idex]);
      outputs_device_type.push_back(type_id);
      outputs_format.push_back(formats[builder_idex]);
      output_idx++;
    }
  }

  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_device_type);
  return true;
}

void SetKernelBuildInfo(const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder, Processor processor,
                        const std::shared_ptr<const OpInfo> &op_info_ptr) {
  MS_EXCEPTION_IF_NULL(builder);
  MS_EXCEPTION_IF_NULL(op_info_ptr);

  auto imply_type = op_info_ptr->imply_type();
  builder->SetProcessor(processor);
  std::string fusion_name = op_info_ptr->fusion_type();
  auto fusion_type = GetFusionTypeByName(fusion_name);
  builder->SetFusionType(fusion_type);

  if (imply_type == kAKG) {
    builder->SetKernelType(AKG_KERNEL);
  } else if (imply_type == kGPU) {
    builder->SetKernelType(GPU_KERNEL);
  } else if (imply_type == kCPU) {
    builder->SetKernelType(CPU_KERNEL);
  } else if (imply_type == kAICPU) {
    builder->SetKernelType(AICPU_KERNEL);
  } else {
    builder->SetKernelType(TBE_KERNEL);
  }
}

bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr, Processor processor,
                   std::vector<std::shared_ptr<KernelBuildInfo>> *const kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  size_t real_input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t real_output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  std::vector<std::shared_ptr<OpIOInfo>> inputs = op_info_ptr->inputs_ptr();
  std::vector<std::shared_ptr<OpIOInfo>> outputs = op_info_ptr->outputs_ptr();
  std::vector<int64_t> dyn_input_sizes;
  auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (primitive->GetAttr("dyn_input_sizes") != nullptr) {
    dyn_input_sizes = GetValue<std::vector<int64_t>>(primitive->GetAttr("dyn_input_sizes"));
  }
  if (inputs.size() > 0) {
    if (inputs[0] == nullptr) {
      MS_LOG(EXCEPTION) << "Inputs[0] is nullptr. Op name: " << op_name;
    }
    size_t kernel_info_cnt = inputs[0]->dtypes().size();
    for (size_t j = 0; j < kernel_info_cnt; j++) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);

      if (!SetInputKernelBuilderInfo(inputs, real_input_num, j, dyn_input_sizes, builder)) {
        MS_LOG(DEBUG) << "Parse kernel metadata, set inputs kernel builder info failed. Op name: " << op_name;
        return false;
      }

      if (outputs.size() > 0) {
        if (!SetOutputKernelBuilderInfo(outputs, j, real_output_num, builder)) {
          MS_LOG(DEBUG) << "Parse kernel metadata, set outputs kernel builder info failed. Op name: " << op_name;
          return false;
        }
      }

      kernel_info_list->push_back(builder->Build());
    }
  } else if (outputs.size() > 0) {
    if (outputs[0] == nullptr) {
      MS_LOG(EXCEPTION) << "Outputs[0] is nullptr. Op name: " << op_name;
    }
    size_t kernel_info_cnt = outputs[0]->dtypes().size();
    for (size_t j = 0; j < kernel_info_cnt; j++) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);

      if (!SetOutputKernelBuilderInfo(outputs, j, real_output_num, builder)) {
        MS_LOG(DEBUG) << "Parse kernel metadata, set outputs kernel builder info failed. Op name: " << op_name;
        return false;
      }

      kernel_info_list->push_back(builder->Build());
    }
  } else {
    if (processor == AICPU) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);
      kernel_info_list->push_back(builder->Build());
    }
  }
  return true;
}

void SaveJsonInfo(const std::string &json_name, const std::string &info, const std::string &base_path) {
  std::string path = base_path + json_name + kInfoSuffix;
  auto realpath = Common::CreatePrefixPath(path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream filewrite(realpath.value());
  if (!filewrite.is_open()) {
    MS_LOG(ERROR) << "Open file '" << realpath.value() << "' failed!";
    return;
  }
  filewrite << info << std::endl;
  filewrite.close();
  ChangeFileMode(realpath.value(), S_IRUSR);
}

Processor GetProcessor(const string &processor) {
  if (processor == kProcessorAiCore) return Processor::AICORE;
  if (processor == kProcessorAiCpu) return Processor::AICPU;
  if (processor == kProcessorCuda) return Processor::CUDA;
  MS_LOG(DEBUG) << "Unknown processor type.";
  return Processor::UNKNOWN;
}

std::string GetProcessor(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string device;
  switch (AnfAlgo::GetProcessor(anf_node)) {
    case Processor::AICORE:
      device = kProcessorAiCore;
      break;

    case Processor::AICPU:
      device = kProcessorAiCpu;
      break;

    case Processor::CUDA:
      device = kProcessorCuda;
      break;

    default:
      MS_LOG(DEBUG) << "Unknown processor type.";
      break;
  }
  return device;
}

bool IsSameShape(const std::vector<size_t> &shape_a, const std::vector<size_t> &shape_b) {
  if (shape_a.size() != shape_b.size()) {
    return false;
  }
  for (size_t i = 0; i < shape_a.size(); ++i) {
    if (shape_a[i] != shape_b[i]) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<AnfNodePtr, size_t>> GetOutputIndex(const std::vector<AnfNodePtr> &node_list,
                                                          const std::vector<AnfNodePtr> &input_list,
                                                          const std::vector<AnfNodePtr> &output_list) {
  std::vector<std::pair<AnfNodePtr, size_t>> output_index;
  for (size_t i = 0; i < output_list.size(); ++i) {
    auto const &output = output_list[i];
    MS_EXCEPTION_IF_NULL(output);
    bool found = false;
    auto pree_node = common::AnfAlgo::VisitKernel(output, 0);
    auto pos = std::find(std::begin(node_list), std::end(node_list), pree_node.first);
    if (pos != std::end(node_list)) {
      output_index.push_back(pree_node);
      continue;
    }
    auto ret = std::find(std::begin(input_list), std::end(input_list), pree_node.first);
    if (ret != std::end(input_list)) {
      output_index.push_back(std::make_pair(pree_node.first, 0));
      found = true;
    }
    if (!found) {
      MS_EXCEPTION(ArgumentError) << "Output [" << i << "][" << output->DebugString(2) << "] of ["
                                  << output->func_graph()->ToString() << "] found no related kernel info.";
    }
  }
  return output_index;
}

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list) {
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_lists = TopoSort(func_graph->get_return());
  for (auto const &node : node_lists) {
    if (!AnfUtils::IsRealKernel(node) || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsValueNode<Primitive>(cnode->input(kAnfPrimitiveIndex))) {
      node_list->push_back(node);
    }
  }
}

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                         std::vector<AnfNodePtr> *input_list, std::vector<AnfNodePtr> *output_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(input_list);

  GetValidKernelNodes(func_graph, node_list);

  auto parameters = func_graph->parameters();
  input_list->insert(input_list->begin(), parameters.begin(), parameters.end());

  GetFuncGraphOutputNodes(func_graph, output_list);
}

void GetFuncGraphOutputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *output_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(output_list);
  auto func_output = func_graph->output();
  MS_EXCEPTION_IF_NULL(func_output);
  if (func_output->isa<CNode>()) {
    // multi output.
    auto cnode = func_output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(kAnfPrimitiveIndex);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
      for (size_t input_idx = 1; input_idx < cnode->inputs().size(); ++input_idx) {
        auto input_node = cnode->input(input_idx);
        MS_EXCEPTION_IF_NULL(input_node);
        if (input_node->isa<CNode>() && common::AnfAlgo::GetInputTensorNum(input_node) == 0) {
          continue;
        }
        output_list->push_back(common::AnfAlgo::VisitKernel(input_node, 0).first);
      }
    } else {
      // single output.
      output_list->push_back(common::AnfAlgo::VisitKernel(func_output, 0).first);
    }
  } else {
    // single output.
    output_list->push_back(common::AnfAlgo::VisitKernel(func_output, 0).first);
  }
}

bool IsWeightBoundary(const AnfNodePtr &node) {
  if (node->isa<ValueNode>()) {
    return true;
  }
  if (node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>())) {
    return true;
  }
  return false;
}

std::vector<int64_t> GetReduceAttrAxis(const CNodePtr &cnode) {
  if (common::AnfAlgo::GetInputTensorNum(cnode) != 1 || common::AnfAlgo::GetOutputTensorNum(cnode) != 1) {
    MS_LOG(EXCEPTION) << "The reduce node [" << cnode->DebugString() << "] is not single input or single output."
                      << trace::DumpSourceLines(cnode);
  }
  std::vector<int64_t> axis;
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto axis_attr = primitive->GetAttr(kAxis);
  if (axis_attr == nullptr) {
    MS_LOG(ERROR) << "This node doesn't have axis attr. Node info [" << cnode->DebugString() << "]";
    return std::vector<int64_t>();
  }
  std::vector<int64_t> axis_list;
  if (axis_attr->isa<Int64Imm>()) {
    (void)axis_list.emplace_back(GetValue<int64_t>(axis_attr));
  } else {
    axis_list = GetValue<std::vector<int64_t>>(axis_attr);
  }
  for (const auto &elem : axis_list) {
    if (elem < 0) {
      (void)axis.emplace_back(input_shape.size() + elem);
    } else {
      (void)axis.emplace_back(elem);
    }
  }
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis), cnode);
  return axis;
}

void FillEmptyDims(const CNodePtr &kernel_node, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                   std::vector<int64_t> *stride, std::vector<size_t> *input_shape) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  std::vector<size_t> &_input_shape = *input_shape;
  if (_begin.size() != _end.size() || _begin.size() != _stride.size() || _begin.size() > _input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << common::AnfAlgo::GetCNodeName(kernel_node)
                      << "', the length of 'begin', 'stride' and 'end' should be equal "
                         "and less than or equal to the dimension of 'input_x', but got the length of 'begin': "
                      << _begin.size() << ", the length of 'stride': " << _stride.size()
                      << ", the length of 'end': " << _end.size()
                      << ", the dimension of 'input_x': " << _input_shape.size();
  }

  for (size_t i = 0; i < kStridedSliceMaxDims; i++) {
    if (i >= _input_shape.size()) {
      _input_shape.push_back(1);
    }

    if (i < _begin.size()) {
      int64_t dim = SizeToLong(_input_shape[i]);
      _begin[i] = std::min(_begin[i] < 0 ? std::max(_begin[i] + dim, static_cast<int64_t>(0)) : _begin[i], dim - 1);
    } else {
      _begin.push_back(0);
    }

    if (i < _end.size()) {
      int64_t dim = SizeToLong(_input_shape[i]);
      _end[i] = std::max(_end[i] < 0 ? _end[i] + dim : std::min(_end[i], dim), static_cast<int64_t>(-1));
    } else {
      _end.push_back(i < _input_shape.size() ? SizeToLong(_input_shape[i]) : 1);
    }

    if (i >= _stride.size()) {
      _stride.push_back(1);
    }
  }
}

std::vector<bool> Dec2Bin(const int64_t &mask) {
  auto mask_str = std::bitset<kStridedSliceMaxDims>(mask).to_string();
  int64_t dim_idx = 0;
  std::vector<bool> result(kStridedSliceMaxDims, false);
  for (int64_t i = mask_str.size() - 1; i >= 0; i--) {
    if (mask_str[i] == '1') {
      result[dim_idx] = true;
    }
    dim_idx++;
  }
  return result;
}

void ComputeBeginMask(const CNodePtr &kernel_node, std::vector<int64_t> *begin, const std::vector<int64_t> &stride,
                      const std::vector<size_t> &input_shape) {
  std::vector<int64_t> &_begin = *begin;
  auto begin_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrBeginMask);
  auto begin_mask = Dec2Bin(begin_mask_int);
  for (size_t i = 0; i < begin_mask.size(); i++) {
    if (i < kStridedSliceMaxDims && begin_mask[i]) {
      _begin[i] = stride[i] < 0 ? SizeToLong(input_shape[i]) - 1 : 0;
    }
  }
}

void ComputeEndMask(const CNodePtr &kernel_node, std::vector<int64_t> *end, const std::vector<int64_t> &stride,
                    const std::vector<size_t> &input_shape) {
  std::vector<int64_t> &_end = *end;
  auto end_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrEndMask);
  auto end_mask = Dec2Bin(end_mask_int);
  for (size_t j = 0; j < end_mask.size(); j++) {
    if (j < kStridedSliceMaxDims && end_mask[j]) {
      _end[j] = stride[j] < 0 ? -1 : SizeToLong(input_shape[j]);
    }
  }
}

void ComputeEllipsisMask(const CNodePtr &kernel_node, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                         std::vector<int64_t> *stride, const std::vector<size_t> &input_shape) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto ellipsis_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrEllipsisMask);
  auto ellipsis_mask = Dec2Bin(ellipsis_mask_int);
  for (size_t k = 0; k < ellipsis_mask.size(); k++) {
    if (k < kStridedSliceMaxDims && ellipsis_mask[k]) {
      _begin[k] = 0;
      _end[k] = SizeToLong(input_shape[k]);
      _stride[k] = 1;
    }
  }
}

void ComputNewAxisMask(const CNodePtr &kernel_node, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                       std::vector<int64_t> *stride, const std::vector<size_t> &input_shape) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto new_axis_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrNewAxisMask);
  auto new_axis_mask = Dec2Bin(new_axis_mask_int);
  for (size_t l = 0; l < new_axis_mask.size(); l++) {
    if (l < kStridedSliceMaxDims && new_axis_mask[l]) {
      _begin[l] = 0;
      _end[l] = SizeToLong(input_shape[l]);
      _stride[l] = 1;
    }
  }
}

void ComputShrinkAxisMask(const CNodePtr &kernel_node, const std::vector<int64_t> &begin, std::vector<int64_t> *end,
                          std::vector<int64_t> *stride) {
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto shrink_axis_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrShrinkAxisMask);
  auto shrink_axis_mask = Dec2Bin(shrink_axis_mask_int);
  for (size_t m = 0; m < shrink_axis_mask.size(); m++) {
    if (m < kStridedSliceMaxDims && shrink_axis_mask[m]) {
      _end[m] = _end[m] > begin[m] ? begin[m] + 1 : begin[m] - 1;
      _stride[m] = _end[m] > begin[m] ? 1 : -1;
    }
  }
}

void ParseStrideSliceMasks(const CNodePtr &kernel_node, std::vector<int64_t> *begin, std::vector<int64_t> *end,
                           std::vector<int64_t> *stride, const std::vector<size_t> &input_shape) {
  ComputeBeginMask(kernel_node, begin, *stride, input_shape);
  ComputeEndMask(kernel_node, end, *stride, input_shape);
  ComputeEllipsisMask(kernel_node, begin, end, stride, input_shape);
  ComputNewAxisMask(kernel_node, begin, end, stride, input_shape);
  ComputShrinkAxisMask(kernel_node, *begin, end, stride);
}

std::string GetProcessorStr(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string processor = kProcessorUnknown;
  auto kernel_info = dynamic_cast<device::KernelInfo *>(anf_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  // we may call this before kernel select.
  if (build_info == nullptr) {
    return processor;
  }
  switch (build_info->processor()) {
    case Processor::AICORE:
      processor = kProcessorAiCore;
      break;

    case Processor::AICPU:
      processor = kProcessorAiCpu;
      break;

    case Processor::CUDA:
      processor = kProcessorCuda;
      break;

    default:
      MS_LOG(ERROR) << "Unknown processor type.";
      break;
  }

  return processor;
}

Processor GetProcessorFromContext() {
  kernel::Processor processor = kernel::Processor::UNKNOWN;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_info = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_info == kGPUDevice) {
    processor = kernel::Processor::CUDA;
  } else if (device_info == kAscendDevice) {
    processor = kernel::Processor::AICORE;
  } else if (device_info == kCPUDevice) {
    processor = kernel::Processor::CPU;
  }
  return processor;
}

std::string GetStrProcessorFromContext() {
  auto processor = GetProcessorFromContext();
  string str_processor = kernel::kProcessorUnknown;
  if (processor == kernel::Processor::CUDA) {
    str_processor = kernel::kProcessorCuda;
  } else if (processor == kernel::Processor::AICORE) {
    str_processor = kernel::kProcessorAiCore;
  } else if (processor == kernel::Processor::CPU) {
    str_processor = kernel::kProcessorCpu;
  }
  return str_processor;
}

float Scaling(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}

float ScaleGrid(const int x, const float scale) { return static_cast<float>(x) * scale; }

void ComputeInterpolationWeights(const size_t out_size, const size_t in_size, const float scale,
                                 CachedInterpolation *interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (size_t i = 0; i <= out_size - 1; ++i) {
    const float in = ScaleGrid(i, scale);
    const float in_f = std::floor(in);
    interpolation[i].lower = std::max(static_cast<size_t>(in_f), static_cast<size_t>(0));
    interpolation[i].upper = std::min(static_cast<size_t>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

bool GetShapeSize(const std::vector<size_t> &shape, const TypePtr &type_ptr, int64_t *size_i) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  size_t type_byte = GetTypeByte(type_ptr);
  if (type_byte == 0) {
    return false;
  }
  for (size_t j = 0; j < shape.size(); j++) {
    size_i[0] = LongMulWithOverflowCheck(size_i[0], static_cast<int64_t>(shape[j]));
  }
  size_i[0] = LongMulWithOverflowCheck(size_i[0], SizeToInt(type_byte));
  return true;
}

void CastShapeSizeToLong(const std::vector<size_t> &shape, std::vector<int64_t> *long_shape) {
  MS_EXCEPTION_IF_NULL(long_shape);
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(*long_shape), SizeToLong);
}

void CheckSliceValid(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                     const std::vector<int64_t> &step, const std::vector<int64_t> &input_shape) {
  if (start.size() != stop.size() || start.size() != step.size() || start.size() > input_shape.size()) {
    MS_LOG(EXCEPTION)
      << "TensorCopySlices requires the length of begin, stride and end must be equal and less than input dimension.";
  }

  size_t size = start.size();
  for (size_t i = 0; i < size; ++i) {
    if (stop[i] <= start[i]) {
      MS_LOG(EXCEPTION) << "Invalid slice: (" << start[i] << ", " << stop[i] << " ," << step[i] << ")";
    }
    // Operator need to be generalized in the future. Only support to copy continuous memory now.
    if (step[i] != 1) {
      MS_LOG(EXCEPTION) << "The element in step only support 1, but got:" << step;
    }
  }

  size_t slice_pos = size;
  for (size_t i = 0; i < size; ++i) {
    if (stop[i] - start[i] > 1) {
      slice_pos = i;
      break;
    }
  }

  for (size_t i = slice_pos + 1; i < size; ++i) {
    if (stop[i] - start[i] != input_shape[i]) {
      MS_LOG(EXCEPTION) << "Only support copy continuous memory now. For example tensor[0, 0:100] is fine, "
                           "but tensor[0:100, 0] is not supported.";
    }
  }
}

size_t GetCopySize(const std::vector<int64_t> &dim_offset, const std::vector<int64_t> &start,
                   const std::vector<int64_t> &stop) {
  for (size_t i = 0; i < start.size(); ++i) {
    if (stop[i] - start[i] != 1) {
      return SizetMulWithOverflowCheck(LongToSize(stop[i] - start[i]), LongToSize(dim_offset[i]));
    }
  }
  return LongToSize(dim_offset[start.size() - 1]);
}

std::vector<int64_t> CalDimOffset(const std::vector<int64_t> &input_shape) {
  std::vector<int64_t> dim_offset;
  int64_t offset = 1;
  for (auto iter = input_shape.rbegin(); iter != input_shape.rend(); ++iter) {
    dim_offset.push_back(offset);
    offset = offset * (*iter);
  }
  std::reverse(dim_offset.begin(), dim_offset.end());
  return dim_offset;
}

size_t CalOffset(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                 const std::vector<int64_t> &dim_offset) {
  size_t size = start.size();
  size_t offset = 0;
  for (size_t i = 0; i < size; ++i) {
    offset += SizetMulWithOverflowCheck(LongToSize(dim_offset[i]), LongToSize(start[i]));
    if (stop[i] - start[i] != 1) {
      break;
    }
  }
  return offset;
}

size_t UnitSizeInBytes(const luojianet_ms::TypeId &t) {
  size_t bytes = 0;
  switch (t) {
    case kNumberTypeBool:
    case kNumberTypeInt8:
    case kNumberTypeUInt8:
      bytes = sizeof(int8_t);
      break;
    case kNumberTypeInt16:
    case kNumberTypeUInt16:
    case kNumberTypeFloat16:
      bytes = sizeof(int16_t);
      break;
    case kNumberTypeInt:
    case kNumberTypeUInt:
    case kNumberTypeInt32:
    case kNumberTypeUInt32:
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      bytes = sizeof(int32_t);
      break;
    case kNumberTypeUInt64:
    case kNumberTypeInt64:
    case kNumberTypeFloat64:
      bytes = sizeof(int64_t);
      break;
    case kNumberTypeInt4:
    default:
      MS_LOG(EXCEPTION) << "Invalid types " << t;
  }

  return bytes;
}

KernelAttr &KernelAttr::AddInputAttr(const TypeId &ms_type, const std::string &format) {
  (void)input_type_.emplace_back(ms_type, format);
  return *this;
}

KernelAttr &KernelAttr::AddOutputAttr(const TypeId &ms_type, const std::string &format) {
  (void)output_type_.emplace_back(ms_type, format);
  return *this;
}

KernelAttr &KernelAttr::AddAllSameAttr(const bool &all_same) {
  all_same_ = all_same;
  return *this;
}

KernelAttr &KernelAttr::AddOutInRef(size_t output_index, size_t input_index) {
  out_in_ref_map_[output_index] = input_index;
  return *this;
}

void KernelAttr::SetInputAttrList(const std::vector<DataType> &addr_list) {
  input_type_.assign(addr_list.begin(), addr_list.end());
}

std::ostream &operator<<(std::ostream &os, KernelAttr kernel_attr) {
  std::stringstream ss;
  ss << "[Kernel Attr] all same: " << kernel_attr.GetAllSame();
  size_t input_num = kernel_attr.GetInputSize();
  if (input_num > 0) {
    ss << ", input(";
    for (size_t i = 0; i < input_num; ++i) {
      ss << TypeIdLabel(kernel_attr.GetInputAttr(i).first);
      if (i != input_num - 1) {
        ss << ",";
      }
    }
    ss << ") ";
  }
  size_t output_num = kernel_attr.GetOutputSize();
  if (output_num > 0) {
    ss << ", output(";
    for (size_t i = 0; i < output_num; ++i) {
      ss << TypeIdLabel(kernel_attr.GetOutputAttr(i).first);
      if (i != output_num - 1) {
        ss << ",";
      }
    }
    ss << ").";
  }

  return os << ss.str();
}

std::pair<bool, size_t> MatchKernelAttr(const KernelAttr &kernel_attr,
                                        const std::vector<KernelAttr> &kernel_attr_list) {
  // kernel_attr should not be all same. If so, then return false.
  if (kernel_attr.GetAllSame()) {
    return std::make_pair(false, 0);
  }

  auto input_num = kernel_attr.GetInputSize();
  auto output_num = kernel_attr.GetOutputSize();

  for (size_t index = 0; index < kernel_attr_list.size(); ++index) {
    const auto &cur_kernel_attr = kernel_attr_list[index];
    auto cur_input_num = cur_kernel_attr.GetInputSize();
    auto cur_output_num = cur_kernel_attr.GetOutputSize();
    if (!cur_kernel_attr.GetAllSame() && (input_num != cur_input_num || output_num != cur_output_num)) {
      continue;
    }

    bool mis_match = false;
    for (size_t i = 0; i < input_num; ++i) {
      auto dtype = cur_kernel_attr.GetInputAttr(cur_kernel_attr.GetAllSame() ? 0 : i).first;
      auto format = cur_kernel_attr.GetInputAttr(cur_kernel_attr.GetAllSame() ? 0 : i).second;
      if (kernel_attr.GetInputAttr(i).first != dtype || kernel_attr.GetInputAttr(i).second != format) {
        mis_match = true;
        break;
      }
    }
    if (mis_match) {
      continue;
    }

    for (size_t i = 0; i < output_num; ++i) {
      auto dtype = cur_kernel_attr.GetOutputAttr(cur_kernel_attr.GetAllSame() ? 0 : i).first;
      auto format = cur_kernel_attr.GetOutputAttr(cur_kernel_attr.GetAllSame() ? 0 : i).second;
      if (kernel_attr.GetOutputAttr(i).first != dtype || kernel_attr.GetOutputAttr(i).second != format) {
        mis_match = true;
        break;
      }
    }
    if (!mis_match) {
      return std::make_pair(true, index);
    }
  }

  return std::make_pair(false, 0);
}

KernelAttr GetKernelAttrFromBuildInfo(const KernelBuildInfoPtr &build_info) {
  MS_EXCEPTION_IF_NULL(build_info);
  KernelAttr kernel_attr;
  for (size_t i = 0; i < build_info->GetInputNum(); i++) {
    (void)kernel_attr.AddInputAttr(build_info->GetInputDeviceType(i), build_info->GetInputFormat(i));
  }
  for (size_t j = 0; j < build_info->GetOutputNum(); j++) {
    (void)kernel_attr.AddOutputAttr(build_info->GetOutputDeviceType(j), build_info->GetOutputFormat(j));
  }
  return kernel_attr;
}

KernelAttr GetKernelAttrFromNode(const AnfNodePtr &kernel_node) {
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  return GetKernelAttrFromBuildInfo(build_info);
}
}  // namespace kernel
}  // namespace luojianet_ms
