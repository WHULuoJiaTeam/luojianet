/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_select.h"

#include <map>
#include <memory>
#include <set>
#include <utility>
#include "kernel/common_utils.h"
#include "kernel/oplib/oplib.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynaminc_shape_util.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_build.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/common_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_broadcast_selecter.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_reduce_selecter.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_property_checker.h"
#include "backend/common/optimizer/helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_build_client.h"
#include "nlohmann/json.hpp"
#include "utils/convert_utils_base.h"
#include "include/common/utils/json_operation_utils.h"

namespace mindspore::kernel {
constexpr auto kName = "name";
constexpr auto kDtype = "dtype";
constexpr auto kFormat = "format";
constexpr auto kPrefixInput = "input";
constexpr auto kPrefixOutput = "output";
constexpr char kParamTypeDynamic[] = "dynamic";
constexpr char kParamTypeRequre[] = "required";
constexpr char kParamTypeOptional[] = "optional";
void TbeMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  auto tbe_selecter = TbeKernelSelect(kernel_node, kernel_info_list);
  tbe_selecter.TbeMetadataInfoEx();
}

TbeKernelSelect::TbeKernelSelect(CNodePtr kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list)
    : cnode_ptr_(std::move(kernel_node)), kernel_info_list_(kernel_info_list) {}

void TbeKernelSelect::TbeMetadataInfoEx() {
  MS_EXCEPTION_IF_NULL(cnode_ptr_);
  MS_EXCEPTION_IF_NULL(kernel_info_list_);
  node_name_ = common::AnfAlgo::GetCNodeName(cnode_ptr_);
  full_name_ = cnode_ptr_->fullname_with_scope();

  auto op_info_ptr = tbe::TbeDynamicShapeUtil::FindOp(node_name_, cnode_ptr_);
  if (!op_info_ptr) {
    return;
  }
  if (!TbePropertyChecker::CheckTbeProperties(cnode_ptr_)) {
    MS_LOG(INFO) << "Warning: node(" << full_name_ << ") is not supported by tbe ai_core.";
    return;
  }

  if (op_info_ptr->is_dynamic_format()) {
    GetDynamicFormatPatternKernelInfo(*op_info_ptr);
  } else {
    OpPattern pattern = op_info_ptr->op_pattern();
    if (pattern == kCommonPattern) {
      GetCommonPatternKernelInfo(*op_info_ptr);
    } else if (pattern == kFormatAgnosticPattern) {
      GetAgnosticPatternKernelInfo(*op_info_ptr);
    } else if (pattern == kBroadcastPattern) {
      GetBroadcastPatternKernelInfo(*op_info_ptr);
    } else if (pattern == kReducePattern) {
      GetReducePatternKernelInfo(*op_info_ptr);
    } else {
      MS_LOG(INFO) << "Warning: op pattern is invailed.";
    }
  }
  // check support
  FilterInVaildKernelInfo(*op_info_ptr);
}

void TbeKernelSelect::GetCommonPatternKernelInfo(const OpInfo &op_info) {
  auto dyn_input_sizes = GetNodeDynamicInputs();
  // get real input/output num
  size_t real_input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode_ptr_);
  const auto inputs_info = op_info.inputs_ptr();
  size_t real_output_tensor_num = common::AnfAlgo::GetOutputTensorNum(cnode_ptr_);
  const auto outputs_info = op_info.outputs_ptr();
  if (inputs_info.empty() && outputs_info.empty()) {
    MS_LOG(EXCEPTION) << common::AnfAlgo::GetCNodeName(cnode_ptr_)
                      << "'s op info input & output is null, please check.";
  }
  // create kernel build info from opinfo
  size_t kernel_build_info_num =
    inputs_info.empty() ? outputs_info[0]->dtypes().size() : inputs_info[0]->dtypes().size();
  for (size_t kernel_build_info_index = 0; kernel_build_info_index < kernel_build_info_num; ++kernel_build_info_index) {
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
    SetTbeBuildCommonInfo(op_info, &builder);
    std::vector<std::string> inputs_format;
    std::vector<TypeId> inputs_device_type;
    std::vector<std::string> inputs_reshape_type;
    std::vector<std::string> inputs_value_depend;
    // input
    if (!GenBuilderItem(true, kernel_build_info_index, real_input_tensor_num, inputs_info, dyn_input_sizes,
                        &inputs_format, &inputs_device_type, &inputs_reshape_type, &inputs_value_depend)) {
      break;
    }
    builder.SetInputsDeviceType(inputs_device_type);
    builder.SetInputsFormat(inputs_format);
    builder.SetInputsReshapeType(inputs_reshape_type);
    builder.SetInputsValueDepend(inputs_value_depend);
    // output
    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_device_type;
    std::vector<std::string> outputs_reshape_type;
    std::vector<std::string> outputs_value_depend;
    if (!GenBuilderItem(false, kernel_build_info_index, real_output_tensor_num, outputs_info, dyn_input_sizes,
                        &outputs_format, &outputs_device_type, &outputs_reshape_type, &outputs_value_depend)) {
      break;
    }
    builder.SetOutputsDeviceType(outputs_device_type);
    builder.SetOutputsFormat(outputs_format);
    builder.SetOutputsReshapeType(outputs_reshape_type);
    kernel_info_list_->emplace_back(builder.Build());
  }
}

void TbeKernelSelect::GetDynamicFormatPatternKernelInfo(const OpInfo &op_info) {
  OpInfo op_info_new;
  CreateNewOpInfo(op_info, &op_info_new);
  GetCommonPatternKernelInfo(op_info_new);
}

void TbeKernelSelect::GetAgnosticPatternKernelInfo(const OpInfo &op_info) {
  if (op_info.inputs_ptr().size() != 1) {
    MS_LOG(EXCEPTION) << "AgnosticPattern only support one input.";
  }
  auto format = AnfAlgo::GetPrevNodeOutputFormat(cnode_ptr_, 0);
  if (kOpFormatList.find(format) == kOpFormatList.end()) {
    MS_LOG(INFO) << "Got the unknown format " << format;
    format = kOpFormat_DEFAULT;
  }
  SupportFormat support_format;
  SupportFormatItem input_item;
  SupportFormatItem output_item;
  input_item.assign(op_info.inputs_ptr().size(), format);
  output_item.assign(op_info.outputs_ptr().size(), format);
  support_format.input_format.emplace_back(input_item);
  support_format.output_format.emplace_back(output_item);
  OpInfo op_info_new;
  CreateNewOpInfo(op_info, support_format, &op_info_new);
  GetCommonPatternKernelInfo(op_info_new);
}

void TbeKernelSelect::GetBroadcastPatternKernelInfo(const OpInfo &op_info) {
  auto broadcast_selecter = TbeKernelBroadCastSelecter(cnode_ptr_);
  SupportFormat support_format;
  (void)broadcast_selecter.GetShapeInfo(&support_format);
  (void)broadcast_selecter.IsBroadCastSupport5HD(&support_format);
  (void)broadcast_selecter.IsBroadCastSupportFracZ(&support_format);
  (void)broadcast_selecter.IsBroadCastSupportC1HWNCoC0(&support_format);
  (void)broadcast_selecter.IsBroadCastSupportFracNZ(&support_format);
  (void)broadcast_selecter.IsBroadCastSupportNDC1HWC0(&support_format);
  OpInfo op_info_new;
  CreateNewOpInfo(op_info, support_format, &op_info_new);
  GetCommonPatternKernelInfo(op_info_new);
}

void TbeKernelSelect::GetReducePatternKernelInfo(const OpInfo &op_info) {
  auto reduce_selecter = TbeKernelReduceSelecter(cnode_ptr_);
  SupportFormat support_format;
  (void)reduce_selecter.GetShapeInfo(&support_format);
  (void)reduce_selecter.IsReduceSupport5HD(&support_format);
  (void)reduce_selecter.IsReduceSupportFracZ(&support_format);
  (void)reduce_selecter.IsReduceSupportC1HWNCoC0(&support_format);
  (void)reduce_selecter.IsReduceSupportFracNZ(&support_format);
  OpInfo op_info_new;
  CreateNewOpInfo(op_info, support_format, &op_info_new);
  GetCommonPatternKernelInfo(op_info_new);
}

void TbeKernelSelect::FilterInVaildKernelInfo(const OpInfo &op_info) {
  if (kernel_info_list_->empty()) {
    MS_LOG(INFO) << "Warning: get kernel build info failed. Skip check supported. Op name: " << full_name_;
    return;
  }
  std::vector<std::shared_ptr<KernelBuildInfo>> kernel_info_list;
  auto dynamic_inputs = GetNodeDynamicInputs();
  auto need_check_supported = op_info.need_check_supported();
  for (auto iter = kernel_info_list_->begin(); iter != kernel_info_list_->end(); ++iter) {
    if (!FilterInVaildShape(iter, !dynamic_inputs.empty())) {
      continue;
    }
    if (need_check_supported && !TbeCheckSupported(iter)) {
      continue;
    }
    kernel_info_list.emplace_back(*iter);
  }
  if (kernel_info_list.empty()) {
    MS_LOG(DEBUG) << "After tbe check supported, all valid AI CORE kernel infos were filtered out. Node:" << full_name_;
  }
  (*kernel_info_list_) = kernel_info_list;
}

bool TbeKernelSelect::FilterInVaildShape(const KernelBuildInfoIter &kernel_build_info_iter, bool is_dynamic_input) {
  MS_EXCEPTION_IF_NULL((*kernel_build_info_iter));
  const auto &kernel_build_info_inputs_format = (*kernel_build_info_iter)->GetAllInputFormats();
  // dynamic input just need to check first input, because other inputs copy from 1th input;
  auto iter_num =
    is_dynamic_input && !kernel_build_info_inputs_format.empty() ? 1 : kernel_build_info_inputs_format.size();
  for (size_t i = 0; i < iter_num; ++i) {
    auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode_ptr_, i);
    const auto &format = kernel_build_info_inputs_format.at(i);
    if (!IsShapeMatchFormat(shape, format)) {
      return false;
    }
  }
  const auto &kernel_build_info_outputs_format = (*kernel_build_info_iter)->GetAllOutputFormats();
  for (size_t j = 0; j < kernel_build_info_outputs_format.size(); ++j) {
    auto shape = common::AnfAlgo::GetOutputInferShape(cnode_ptr_, j);
    const auto &format = kernel_build_info_outputs_format[j];
    if (!IsShapeMatchFormat(shape, format)) {
      return false;
    }
  }
  return true;
}

bool TbeKernelSelect::IsShapeMatchFormat(const std::vector<size_t> &shape, const std::string &format) {
  if (format == kOpFormat_DEFAULT) {
    return true;
  }
  static const std::set<std::string> kServerNotSupportFormat = {kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04};
  // if format is default, it remarkes support all format
  if (kOpFormatList.find(format) == kOpFormatList.end()) {
    MS_LOG(EXCEPTION) << "Got the unknown format " << format;
  }
  // server not support format with C04 suffix
  if (std::find(kServerNotSupportFormat.begin(), kServerNotSupportFormat.end(), format) !=
      kServerNotSupportFormat.end()) {
    MS_LOG(INFO) << "Warning: Server not support format with C04 suffix.";
    return false;
  }
  if (format == kOpFormat_FRAC_NZ && shape.size() > kShape2dDims) {
    return true;
  }
  // not support format:
  // 1 3d formats with shape size > 5
  if (k3DFormatSet.find(format) != k3DFormatSet.end() && shape.size() > kShape5dDims) {
    return false;
  }
  return true;
}

bool TbeKernelSelect::TbeCheckSupported(const KernelBuildInfoIter &kernel_build_info_iter) {
  MS_EXCEPTION_IF_NULL((*kernel_build_info_iter));
  // replace kernel_info with current kernel info
  auto kernel_build_info_tmp = AnfAlgo::GetSelectKernelBuildInfo(cnode_ptr_);
  AnfAlgo::SetSelectKernelBuildInfo(*kernel_build_info_iter, cnode_ptr_.get());
  auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
  auto ret = HostCheck::CheckValidDeviceShape(cnode_ptr_) && build_manager.TbeOpCheckSupported(cnode_ptr_);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_tmp, cnode_ptr_.get());
  return ret;
}

void TbeKernelSelect::SetTbeBuildCommonInfo(const mindspore::kernel::OpInfo &op_info,
                                            mindspore::kernel::KernelBuildInfo::KernelBuildInfoBuilder *builder) {
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetProcessor(AICORE);
  std::string fusion_name = op_info.fusion_type();
  auto fusion_type = kernel::GetFusionTypeByName(fusion_name);
  if (fusion_type != UNKNOWN_FUSION_TYPE) {
    builder->SetFusionType(fusion_type);
  }
  builder->SetOpPattern(op_info.op_pattern());
  builder->SetKernelType(TBE_KERNEL);
}

std::vector<int64_t> TbeKernelSelect::GetNodeDynamicInputs() {
  // get dynamic inputs
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode_ptr_);
  MS_EXCEPTION_IF_NULL(primitive);
  std::vector<int64_t> dyn_input_sizes;
  if (primitive->HasAttr(kAttrDynInputSizes)) {
    dyn_input_sizes = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrDynInputSizes));
  }
  return dyn_input_sizes;
}

bool TbeKernelSelect::GenBuilderItem(bool is_input, size_t kernel_build_info_index, size_t real_io_tensor_num,
                                     const std::vector<std::shared_ptr<OpIOInfo>> &ios_info,
                                     const std::vector<int64_t> &dyn_input_sizes, std::vector<std::string> *formats,
                                     std::vector<TypeId> *device_types, std::vector<std::string> *reshape_types,
                                     std::vector<std::string> *value_depends) {
  MS_EXCEPTION_IF_NULL(formats);
  MS_EXCEPTION_IF_NULL(device_types);
  MS_EXCEPTION_IF_NULL(reshape_types);
  MS_EXCEPTION_IF_NULL(value_depends);
  size_t dynamic_input_index = 0;
  size_t real_io_tensor_index = 0;
  size_t io_info_index = 0;
  size_t io_info_num = ios_info.size();
  for (; io_info_index < io_info_num && real_io_tensor_index < real_io_tensor_num; io_info_index++) {
    std::shared_ptr<OpIOInfo> io_info_item = ios_info[io_info_index];
    const auto &kernel_build_info_dtype = io_info_item->dtypes()[kernel_build_info_index];
    std::string kernel_build_info_format;
    if (!io_info_item->formats().empty()) {
      kernel_build_info_format = io_info_item->formats()[kernel_build_info_index];
    }
    const std::string &io_param_type = io_info_item->param_type();
    auto reshape_type = io_info_item->reshape_type();
    auto value_depend = io_info_item->value_depend();
    if (io_param_type == kParamTypeDynamic) {
      // dynamic io
      if (is_input) {
        if (dynamic_input_index >= dyn_input_sizes.size()) {
          MS_LOG(EXCEPTION) << "dyn_input_sizes attr set error, dynamic_input_index: " << dynamic_input_index
                            << ", dyn_input_sizes size: " << dyn_input_sizes.size();
        }
        int64_t dynamic_input_size = dyn_input_sizes[dynamic_input_index];
        for (int64_t i = 0; i < dynamic_input_size; ++i) {
          device_types->emplace_back(tbe::DtypeToTypeId(kernel_build_info_dtype));
          formats->emplace_back(kernel_build_info_format);
          reshape_types->emplace_back(reshape_type);
          value_depends->emplace_back(value_depend);
        }
        dynamic_input_index++;
        real_io_tensor_index = SizetAddWithOverflowCheck(real_io_tensor_index, LongToSize(dynamic_input_size));
      } else {
        if (ios_info.size() != 1) {
          MS_LOG(EXCEPTION) << "if output is dynamic, so output must has one output.";
        }
        for (size_t i = 0; i < real_io_tensor_num; ++i) {
          device_types->emplace_back(tbe::DtypeToTypeId(kernel_build_info_dtype));
          formats->emplace_back(kernel_build_info_format);
          reshape_types->emplace_back(reshape_type);
          value_depends->emplace_back(value_depend);
        }
        real_io_tensor_index = SizetAddWithOverflowCheck(real_io_tensor_index, real_io_tensor_num);
      }
    } else if (io_param_type == kParamTypeRequre || io_param_type == kParamTypeOptional) {
      // require or optional io
      device_types->emplace_back(tbe::DtypeToTypeId(kernel_build_info_dtype));
      formats->emplace_back(kernel_build_info_format);
      reshape_types->emplace_back(reshape_type);
      value_depends->emplace_back(value_depend);
      real_io_tensor_index++;
    } else {
      MS_LOG(EXCEPTION) << "op info's param type is not match: " << io_param_type;
    }
  }

  if (real_io_tensor_index != real_io_tensor_num) {
    std::string io_type = is_input ? "inputs " : "outputs";
    MS_LOG(INFO) << node_name_ << "'s " << io_type << "op io info num: " << io_info_num
                 << ", real io tensor num:" << real_io_tensor_num << "real_io_tensor_index(" << real_io_tensor_index
                 << ") != real_io_tensor_num(" << real_io_tensor_num << ")";
    return false;
  }
  return true;
}

void TbeKernelSelect::CreateNewOpIOInfo(const mindspore::kernel::OpIOInfo &op_io_info,
                                        const std::vector<std::vector<std::string>> &support_format_item, size_t index,
                                        mindspore::kernel::OpIOInfo *op_io_info_new) {
  MS_EXCEPTION_IF_NULL(op_io_info_new);
  op_io_info_new->set_index(op_io_info.index());
  op_io_info_new->set_name(op_io_info.name());
  op_io_info_new->set_param_type(op_io_info.param_type());
  op_io_info_new->set_need_compile(op_io_info.need_compile());
  op_io_info_new->set_reshape_type(op_io_info.reshape_type());
  op_io_info_new->set_shape(op_io_info.shape());
  op_io_info_new->set_value_depend(op_io_info.value_depend());
  // dtype
  std::vector<std::string> dtype_new;
  auto dtype = op_io_info.dtypes();
  for (size_t i = 0; i < support_format_item.size(); ++i) {
    (void)dtype_new.insert(dtype_new.end(), dtype.begin(), dtype.end());
  }
  op_io_info_new->set_dtypes(dtype_new);
  // format
  std::vector<std::string> format_new;
  for (const auto &formats : support_format_item) {
    auto format = formats.at(index);
    for (size_t j = 0; j < dtype.size(); ++j) {
      format_new.emplace_back(format);
    }
  }
  op_io_info_new->set_formats(format_new);
}

std::vector<std::string> TbeKernelSelect::SplitStrToVec(const std::string &op_select_json_item) {
  const std::map<std::string, std::string> kDynamicFormatMap = {
    {"NCHW", "DefaultFormat"}, {"ND", "DefaultFormat"}, {"FRACTAL_Z", "FracZ"}, {"NCDHW", "DefaultFormat"}};
  if (op_select_json_item.empty()) {
    MS_LOG(EXCEPTION) << "Op select ret item is null.";
  }
  const char space = ' ';
  const char sep = ',';
  std::string op_select_tmp = op_select_json_item + ",";
  std::vector<std::string> ret;
  auto begin = op_select_tmp.find_first_not_of(space, 0);
  auto sep_pos = op_select_tmp.find(sep);
  if (begin >= sep_pos) {
    MS_LOG(EXCEPTION) << "Select ret json is error.";
  }
  while (sep_pos != std::string::npos) {
    auto obj = op_select_tmp.substr(begin, sep_pos - begin);
    if (kDynamicFormatMap.find(obj) != kDynamicFormatMap.end()) {
      obj = kDynamicFormatMap.at(obj);
    }
    ret.emplace_back(obj);
    begin = op_select_tmp.find_first_not_of(space, sep_pos + 1);
    sep_pos = op_select_tmp.find(sep, begin);
  }
  return ret;
}

std::string TbeKernelSelect::OpSelectFormat() {
  std::string res_json_str;
  MS_LOG(DEBUG) << "Format select for node:[" << cnode_ptr_->fullname_with_scope() << "].";
  auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
  res_json_str = build_manager.TbeOpSelectFormat(cnode_ptr_);
  return res_json_str;
}

void TbeKernelSelect::CreateNewOpInfo(const mindspore::kernel::OpInfo &op_info, const SupportFormat &support_format,
                                      mindspore::kernel::OpInfo *op_info_new) {
  MS_EXCEPTION_IF_NULL(op_info_new);
  if (support_format.input_format.empty() || support_format.output_format.empty()) {
    MS_LOG(EXCEPTION) << "Support input format and output format size can not be empty, but the input format size is: "
                      << support_format.input_format.size()
                      << ", output format size is: " << support_format.output_format.size();
  }
  if (op_info.inputs_ptr().size() != support_format.input_format[0].size() ||
      op_info.outputs_ptr().size() != support_format.output_format[0].size()) {
    MS_LOG(EXCEPTION) << "BroadCast input/output size not match, op info input size:" << op_info.inputs_ptr().size()
                      << ", input support size: " << support_format.input_format[0].size()
                      << ", op info output size: " << op_info.outputs_ptr().size()
                      << ", output support size: " << support_format.output_format[0].size();
  }
  *op_info_new = op_info;
  op_info_new->ClearInputs();
  op_info_new->ClearOutputs();
  for (size_t i = 0; i < op_info.inputs_ptr().size(); ++i) {
    auto inputs_ptr = op_info.inputs_ptr();
    auto input = inputs_ptr.at(i);
    auto input_new = std::make_shared<OpIOInfo>();
    CreateNewOpIOInfo(*input, support_format.input_format, i, input_new.get());
    op_info_new->add_inputs_ptr(input_new);
  }
  for (size_t j = 0; j < op_info.outputs_ptr().size(); ++j) {
    auto outputs_ptr = op_info.outputs_ptr();
    auto output = outputs_ptr.at(j);
    auto output_new = std::make_shared<OpIOInfo>();
    CreateNewOpIOInfo(*output, support_format.output_format, j, output_new.get());
    op_info_new->add_outputs_ptr(output_new);
  }
}

struct SelectOpIOInfo {
  std::string name;
  std::vector<std::string> dtypes;
  std::vector<std::string> formats;
};

void TbeKernelSelect::CreateNewOpInfo(const mindspore::kernel::OpInfo &op_info,
                                      mindspore::kernel::OpInfo *op_info_new) {
  MS_EXCEPTION_IF_NULL(op_info_new);
  auto op_seclect_json = OpSelectFormat();
  if (!op_seclect_json.empty()) {
    nlohmann::json json_obj;
    if (!ParseJson(op_seclect_json, &json_obj)) {
      MS_LOG(EXCEPTION) << "Parse op_select_json error.";
    }
    if (!json_obj.is_object()) {
      MS_LOG(EXCEPTION) << "JsonStr is not an object, the jsonStr is:" << op_seclect_json;
    }
    std::vector<SelectOpIOInfo> inputs;
    std::vector<SelectOpIOInfo> outputs;
    for (const auto &item : json_obj.items()) {
      const std::string &item_name = item.key();
      bool is_input = (item_name.find(kPrefixInput) != std::string::npos);
      bool is_output = (item_name.find(kPrefixOutput) != std::string::npos);
      if (!is_input && !is_output) {
        MS_LOG(EXCEPTION) << "op select ret json is error.";
      }
      if (is_input) {
        SelectOpIOInfo select_input;
        select_input.name = item.value().at(kName);
        std::string input_dtype_item = item.value().at(kDtype);
        select_input.dtypes = SplitStrToVec(input_dtype_item);
        std::string input_format_item = item.value().at(kFormat);
        select_input.formats = SplitStrToVec(input_format_item);
        inputs.emplace_back(select_input);
      } else {
        SelectOpIOInfo select_output;
        select_output.name = item.value().at(kName);
        std::string input_dtype_item = item.value().at(kDtype);
        select_output.dtypes = SplitStrToVec(input_dtype_item);
        std::string input_format_item = item.value().at(kFormat);
        select_output.formats = SplitStrToVec(input_format_item);
        outputs.emplace_back(select_output);
      }
    }

    if (op_info.inputs_ptr().size() != inputs.size() || op_info.outputs_ptr().size() != outputs.size()) {
      MS_LOG(EXCEPTION) << "select format input/output size not equal, please check register.";
    }

    *op_info_new = op_info;
    op_info_new->ClearInputs();
    op_info_new->ClearOutputs();
    for (size_t i = 0; i < op_info.inputs_ptr().size(); ++i) {
      auto input_new = std::make_shared<OpIOInfo>();
      const auto &inputs_ptr = op_info.inputs_ptr();
      CreateNewOpIOInfo(*(inputs_ptr.at(i)), inputs.at(i).dtypes, inputs.at(i).formats, input_new.get());
      op_info_new->add_inputs_ptr(input_new);
    }
    for (size_t i = 0; i < op_info.outputs_ptr().size(); ++i) {
      auto output_new = std::make_shared<OpIOInfo>();
      const auto &outputs_ptr = op_info.outputs_ptr();
      CreateNewOpIOInfo(*(outputs_ptr.at(i)), outputs.at(i).dtypes, outputs.at(i).formats, output_new.get());
      op_info_new->add_outputs_ptr(output_new);
    }
  }
}

void TbeKernelSelect::CreateNewOpIOInfo(const mindspore::kernel::OpIOInfo &op_io_info,
                                        const std::vector<std::string> &support_dtype,
                                        const std::vector<std::string> &support_format,
                                        mindspore::kernel::OpIOInfo *op_io_info_new) {
  MS_EXCEPTION_IF_NULL(op_io_info_new);
  op_io_info_new->set_index(op_io_info.index());
  op_io_info_new->set_name(op_io_info.name());
  op_io_info_new->set_param_type(op_io_info.param_type());
  op_io_info_new->set_need_compile(op_io_info.need_compile());
  op_io_info_new->set_reshape_type(op_io_info.reshape_type());
  op_io_info_new->set_shape(op_io_info.shape());
  op_io_info_new->set_value_depend(op_io_info.value_depend());
  // dtype  && format
  op_io_info_new->set_dtypes(support_dtype);
  op_io_info_new->set_formats(support_format);
}

void TbeKernelSelect::PrintSupportedFormat(const SupportFormat &support_format) {
  if (support_format.input_format.size() != support_format.output_format.size()) {
    MS_LOG(EXCEPTION) << "Input(" << support_format.input_format.size() << ")Output("
                      << support_format.output_format.size() << ") size not match.";
  }
  for (size_t i = 0; i < support_format.input_format.size(); ++i) {
    auto input_items = support_format.input_format.at(i);
    auto output_items = support_format.output_format.at(i);
    std::string print_str = "[";
    for (const auto &input : input_items) {
      (void)print_str.append(input);
      (void)print_str.append(", ");
    }
    (void)print_str.append("] -->");
    for (const auto &output : output_items) {
      (void)print_str.append(output);
      (void)print_str.append(", ");
    }
    MS_LOG(INFO) << "Support format: " << print_str;
  }
}
}  // namespace mindspore::kernel
