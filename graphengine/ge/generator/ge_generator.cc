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

#include "framework/generator/ge_generator.h"

#include <atomic>

#include "common/ge/ge_util.h"
#include "common/ge/plugin_manager.h"
#include "framework/common/helper/model_helper.h"
#include "framework/common/helper/om_file_helper.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "external/ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/operator_factory_impl.h"
#include "graph/opsproto_manager.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "common/model/ge_model.h"
#include "analyzer/analyzer.h"

using std::map;
using std::string;
using std::vector;

namespace {
const char *const kAttrOpType = "op_type";
const char *const kEngineNameDefault = "default";
const char *const kVectorEngine = "VectorEngine";
const char *const kAIcoreEngine = "AIcoreEngine";
const char *const kFileNameSuffix = "online";
const char *const kAicpuAllshape = "_AllShape";
constexpr char const *kAttrSupportDynamicShape = "support_dynamicshape";
const int64_t kDynamicDimValue = -2;
const int kDefaultDeviceId = 0;
const int kDefaultJobId = 0;
const int32_t kFuzzBuildPattern = 1;

std::map<ge::OpEngineType, std::string> engine_type_map{
    {ge::ENGINE_SYS, kEngineNameDefault},
    {ge::ENGINE_AICORE, kAIcoreEngine},
    {ge::ENGINE_VECTOR, kVectorEngine}};

bool ContainsDynamicInpus(const ge::OpDesc &op_desc) {
  for (auto &tensor_desc : op_desc.GetAllInputsDescPtr()) {
    if (tensor_desc->MutableShape().IsUnknownShape()) {
      GELOGI("Contains unknown shape input. set is_dynamic_input to true.");
      return true;
    }
  }
  return false;
}
// if optional in/out, format is format_reserved and dtype is dt_undefined
bool IsOptional(const ge::GeTensorDesc &tensor_desc) {
  return tensor_desc.GetFormat() == ge::FORMAT_RESERVED && tensor_desc.GetDataType() == ge::DT_UNDEFINED;
}
}  // namespace

namespace ge {
static Status CheckEngineTypeSupport(const NodePtr &node, OpEngineType engine_type) {
  const OpDescPtr &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return PARAM_INVALID);
  if (engine_type == ENGINE_SYS) {
    GELOGI("CheckEngineType: use default engine.");
    return SUCCESS;
  }

  // get op engine name
  string op_engine_name;
  auto iter = engine_type_map.find(engine_type);
  if (iter != engine_type_map.end()) {
    op_engine_name = iter->second;
    GELOGI("CheckEngineType: engine type: %d", static_cast<int>(engine_type));
  } else {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "engine type",
        "it only support default/AIcoreEngine/VectorEngine"});
    GELOGE(FAILED, "[Check][Param] value:%d not support, "
           "only support default/AIcoreEngine/VectorEngine now", static_cast<int>(engine_type));
    return FAILED;
  }

  if (op_desc->HasAttr(ATTR_NAME_UNREGST_OPPATH)) {
    op_desc->SetOpEngineName(op_engine_name);
    op_desc->SetOpKernelLibName(op_engine_name);
    return SUCCESS;
  }
  // set op engine name and opkernelLib. when engine support
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    REPORT_INNER_ERROR("E19999", "get gelib failed, as get instance failed or initflag failed.");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] CheckEngineType failed, as get gelib failed.");
    return FAILED;
  }
  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  std::vector<OpInfo> op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "optype", "it can not find"});
    GELOGE(FAILED, "[Get][OpInfo] by op type %s failed.", op_desc->GetType().c_str());
    return FAILED;
  }
  string kernel_name;
  for (const auto &it : op_infos) {
    if (it.engine == op_engine_name) {
      kernel_name = it.opKernelLib;
      break;
    }
  }
  if (kernel_name.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "engine name" + FmtToStr(op_engine_name), "it can not find"});
    GELOGE(FAILED, "[Check][Param] Can not find ops kernel, engine name:%s. op:%s(%s)",
           op_engine_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  auto &kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
  auto kernel_info_store = kernel_map.find(kernel_name);
  if (kernel_info_store != kernel_map.end()) {
    std::string unsupported_reason;
    if (kernel_info_store->second->CheckSupported(node, unsupported_reason)) {
      op_desc->SetOpEngineName(op_engine_name);
      op_desc->SetOpKernelLibName(kernel_name);
      GELOGI("CheckEngineType:Set OpKernelLibName %s and engine name %s into op_desc %s", kernel_name.c_str(),
             op_engine_name.c_str(), op_desc->GetName().c_str());
      return SUCCESS;
    } else {
      ErrorManager::GetInstance().ATCReportErrMessage(
        "E13002", {"optype", "opskernel", "reason"}, {op_desc->GetType(), kernel_name, unsupported_reason});
      GELOGE(FAILED, "[Call][CheckSupported] failed, Op type %s of ops kernel %s is unsupported, reason:%s",
             op_desc->GetType().c_str(), kernel_name.c_str(), unsupported_reason.c_str());
      return FAILED;
    }
  } else {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E13003", {"opname", "optype"}, {op_desc->GetName(), op_desc->GetType()});
    GELOGE(FAILED, "[Check][Param] Can not find any supported ops kernel info store by kernel_name %s,"
           "op type is %s, op name is %s",
           kernel_name.c_str(), op_desc->GetType().c_str(), op_desc->GetName().c_str());
  }
  return FAILED;
}

static Status AddInputs(const ComputeGraphPtr &graph, const NodePtr &node, const GeTensorDesc &tensor, int32_t index,
                        bool attr, int32_t &data_index) {
  GE_CHECK_NOTNULL_EXEC(graph, return PARAM_INVALID);
  GE_CHECK_NOTNULL_EXEC(node, return PARAM_INVALID);

  auto format = tensor.GetFormat();
  auto data_type = tensor.GetDataType();
  if (format == FORMAT_RESERVED && data_type == DT_UNDEFINED) {
    return SUCCESS;
  }

  string op_type;
  bool is_const = false;
  (void)AttrUtils::GetBool(tensor, CONST_ATTR_NAME_INPUT, is_const);
  if (is_const) {
    GELOGD("Get input[%d] is const", index);
    op_type = CONSTANTOP;
  } else if (!AttrUtils::GetStr(tensor, kAttrOpType, op_type) || op_type.empty()) {
    op_type = DATA;
  }

  string op_name = node->GetName() + "_in_" + std::to_string(index);
  OpDescPtr data_op = MakeShared<ge::OpDesc>(op_name, op_type);
  if (data_op == nullptr) {
    REPORT_CALL_ERROR("E19999", "create OpDesc failed, name:%s", op_name.c_str());
    GELOGE(FAILED, "[Create][OpDesc] failed, name:%s", op_name.c_str());
    return FAILED;
  }
  if (is_const) {
    ConstGeTensorPtr tensor_value;
    if (!AttrUtils::GetTensor(tensor, ge::ATTR_NAME_WEIGHTS, tensor_value)) {
      REPORT_CALL_ERROR("E19999", "get attr %s failed, tensor:%s.",
                        ge::ATTR_NAME_WEIGHTS.c_str(), tensor.GetName().c_str());
      GELOGE(FAILED, "[Get][Attr] %s failed, tensor:%s.", ge::ATTR_NAME_WEIGHTS.c_str(), tensor.GetName().c_str());
      return FAILED;
    }
    if (!AttrUtils::SetTensor(data_op, ge::ATTR_NAME_WEIGHTS, tensor_value)) {
      REPORT_CALL_ERROR("E19999", "set attr %s failed, op:%s.", ge::ATTR_NAME_WEIGHTS.c_str(), op_name.c_str());
      GELOGE(FAILED, "[Set][Attr] %s failed, op:%s.", ge::ATTR_NAME_WEIGHTS.c_str(), op_name.c_str());
      return FAILED;
    }
  }

  (void)AttrUtils::SetBool(data_op, "_is_single_op", true);
  (void)AttrUtils::SetBool(data_op, ATTR_NAME_IS_ORIGINAL_INPUT, true);

  GE_CHK_BOOL_EXEC(data_op->AddInputDesc(tensor) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "AddInputDesc failed for node:%s", data_op->GetName().c_str());
                   return FAILED, "[Add][InputDesc] fail for node:%s", data_op->GetName().c_str());
  GE_CHK_BOOL_EXEC(data_op->AddOutputDesc(tensor) == GRAPH_SUCCESS,
                   REPORT_CALL_ERROR("E19999", "AddOutputDesc failed for node:%s", data_op->GetName().c_str());
                   return FAILED, "[Add][OutputDesc] fail for node:%s", data_op->GetName().c_str());
  if (attr && !is_const) {
    GE_CHK_BOOL_EXEC(AttrUtils::SetInt(data_op, ATTR_NAME_INDEX, data_index),
                     REPORT_CALL_ERROR("E19999", "set attr %s failed for node:%s",
                                       ATTR_NAME_INDEX.c_str(), data_op->GetName().c_str());
                     return FAILED,
                     "[Set][Attr:%s] fail for node:%s", ATTR_NAME_INDEX.c_str(), data_op->GetName().c_str());
    ++data_index;
  }

  ge::NodePtr arg_node = graph->AddNode(data_op);
  GE_CHK_BOOL_EXEC(arg_node != nullptr,
                   REPORT_CALL_ERROR("E19999", "add node:%s to graph:%s failed", data_op->GetName().c_str(),
                                     graph->GetName().c_str());
                   return FAILED, "[Add][Node] Insert Data node:%s fail", data_op->GetName().c_str());

  GE_CHK_STATUS(GraphUtils::AddEdge(arg_node->GetOutDataAnchor(0), node->GetInDataAnchor(index)),
                "[Add][Edge]fail from node:%s to node:%s", data_op->GetName().c_str(), node->GetName().c_str());

  return SUCCESS;
}

static Status AddOutputs(const ComputeGraphPtr &graph, const NodePtr &node, const vector<GeTensor> &outputs) {
  OpDescPtr op_desc = MakeShared<ge::OpDesc>(graph->GetName() + "_" + NODE_NAME_NET_OUTPUT, NETOUTPUT);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "create OpDesc failed, graph:%s", graph->GetName().c_str());
    GELOGE(FAILED, "[Create][OpDesc] failed, graph:%s", graph->GetName().c_str());
    return FAILED;
  }
  (void)AttrUtils::SetBool(op_desc, "_is_single_op", true);
  int32_t count = 0;
  for (const auto &out_desc : outputs) {
    GeTensorDesc tensor = out_desc.GetTensorDesc();
    TensorUtils::SetInputTensor(tensor, true);
    GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(tensor) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "AddInputDesc failed for node:%s", op_desc->GetName().c_str());
                     return FAILED, "[Add][InputDesc]fail for node:%s", op_desc->GetName().c_str());

    TensorUtils::SetInputTensor(tensor, false);
    TensorUtils::SetOutputTensor(tensor, true);
    GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(tensor) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "AddOutputDesc failed for node:%s", op_desc->GetName().c_str());
                     return FAILED, "[Add][OutputDesc]fail for node:%s", op_desc->GetName().c_str());
    count++;
  }
  GE_CHECK_NOTNULL_EXEC(graph, return PARAM_INVALID);
  ge::NodePtr out_node = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(out_node != nullptr,
                   REPORT_CALL_ERROR("E19999", "add node:%s to graph:%u failed.",
                                     op_desc->GetName().c_str(), graph->GetGraphID());
                   return FAILED,
                   "[Add][Node:%s]fail in graph:%u", op_desc->GetName().c_str(), graph->GetGraphID());
  GE_CHECK_NOTNULL_EXEC(node, return PARAM_INVALID);
  for (int32_t i = 0; i < count; ++i) {
    GE_CHK_STATUS(GraphUtils::AddEdge(node->GetOutDataAnchor(i), out_node->GetInDataAnchor(i)),
                  "[Add][Edge]fail from node:%s to node:%s", node->GetName().c_str(), out_node->GetName().c_str());
  }

  return SUCCESS;
}

static void GetOpsProtoPath(string &opsproto_path) {
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    string path = path_env;
    string file_path = RealPath(path.c_str());
    if (file_path.empty()) {
      REPORT_CALL_ERROR("E19999", "File path %s is invalid.", path.c_str());
      GELOGE(FAILED, "[Call][RealPath] File path %s is invalid.", path.c_str());
      return;
    }
    opsproto_path = (path + "/op_proto/custom/" + ":") + (path + "/op_proto/built-in/");
    GELOGI("Get opsproto so path from env : %s", path.c_str());
    return;
  }
  string path_base = PluginManager::GetPath();
  GELOGI("path_base is %s", path_base.c_str());
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  opsproto_path = (path_base + "ops/op_proto/custom/" + ":") + (path_base + "ops/op_proto/built-in/");
}

static Status ResetTensorVecShape(const vector<GeTensor> &inputs, vector<GeTensor> &inputs_dynamic) {
  for (auto input : inputs) {
    auto input_desc = input.GetTensorDesc();
    GeShape shape_ori = input_desc.GetShape();

    std::vector<int64_t> dynamic_shape_dims = {kDynamicDimValue};
    GeShape dynamic_shape(dynamic_shape_dims);
    std::vector<std::pair<int64_t, int64_t>> dynamic_shape_range;

    ge::GeTensor inputTensor;
    ge::GeTensorDesc desc(input_desc);

    bool is_const = false;
    (void)AttrUtils::GetBool(input_desc, CONST_ATTR_NAME_INPUT, is_const);
    if (!is_const) {
      int64_t storage_format = FORMAT_NCHW;
      if (ge::AttrUtils::GetInt(desc, ge::ATTR_NAME_STORAGE_FORMAT, storage_format) &&
          !ge::AttrUtils::SetListInt(desc, ge::ATTR_NAME_STORAGE_SHAPE, dynamic_shape_dims)) {
        REPORT_CALL_ERROR("E19999", "Set attr ATTR_NAME_STORAGE_SHAPE failed to op:%s.", desc.GetName().c_str());
        GELOGE(FAILED, "[Set][Attr] ATTR_NAME_STORAGE_SHAPE fail.");
        return FAILED;
      }
      desc.SetShape(dynamic_shape);
      desc.SetShapeRange(dynamic_shape_range);
    }

    inputTensor.SetTensorDesc(desc);
    inputs_dynamic.push_back(inputTensor);
  }
  return SUCCESS;
}

static Status GetFuzzBuildAttrs(const OpDescPtr &op_desc, const GeRootModelPtr &ge_root_model,
                                GeAttrValue::LIST_NAMED_ATTRS &fuzz_build_attrs) {
  GELOGD("Start get fuzz build attrs of %s.", op_desc->GetName().c_str());
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  for (const auto &node : ge_root_model->GetRootGraph()->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GELOGD("Delete fuzz build attr of %s after build.", node->GetName().c_str());
    node->GetOpDesc()->DelAttr(ATTR_NAME_FUZZ_BUILD);
  }
  (void)AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_FUZZ_BUILD_RES_ATTRS, fuzz_build_attrs);
  if (!fuzz_build_attrs.empty()) {
    GELOGD("%s has split, get ATTR_NAME_FUZZ_BUILD_RES_ATTRS directly.", op_desc->GetName().c_str());
    return SUCCESS;
  } else {
    GELOGW("%s build with fuzz build pattern, but not set ATTR_NAME_FUZZ_BUILD_RES_ATTRS.", op_desc->GetName().c_str());
  }
  return SUCCESS;
}

static bool HasShapeRange(const vector<GeTensor> &inputs) {
  for (const auto &input : inputs) {
    vector<pair<int64_t, int64_t>> shape_range;
    (void)input.GetTensorDesc().GetShapeRange(shape_range);
    if (!shape_range.empty()) {
      GELOGD("Has set shape range.");
      return true;
    }
  }
  return false;
}

class GeGenerator::Impl {
 public:
  Impl(OmgContext &omg_context) : omg_context_(omg_context) {}
  ~Impl() = default;

  Status BuildModel(const Graph &graph, const vector<GeTensor> &inputs, GeRootModelPtr &ge_models);
  Status SaveModel(const string &file_name_prefix, GeModelPtr &models, ModelBufferData &model);

  Status SaveRootModel(const string &file_name_prefix, GeRootModelPtr &model, ModelBufferData &model_buff);

  Status SaveParams(GeModelPtr &ge_model, const string &type, const map<string, GeAttrValue> &attrs,
                    const vector<GeTensor> &inputs, const vector<GeTensor> &outputs);

  Status GenerateInfershapeGraph(const Graph &graph);

  OmgContext &omg_context_;
  GraphManager graph_manager_;
  SaveParam save_param_;
  bool is_offline_ = true;
  bool is_singleop_unregistered_ = false;
  std::string build_mode_;
  std::string build_step_;
  static std::mutex mutex_;

 private:
  static std::string Trim(const std::string &str);
  bool ParseVersion(const std::string &line, std::string &version);
  bool GetVersionFromPath(const std::string &file_path, std::string &version);
  bool SetAtcVersionInfo(AttrHolder &obj);
  bool SetOppVersionInfo(AttrHolder &obj);
  bool SetOmSystemInfo(AttrHolder &obj);
};

Status GeGenerator::Initialize(const map<string, string> &options) {
  return Initialize(options, domi::GetContext());
}

Status GeGenerator::Initialize(const map<string, string> &options, OmgContext &omg_context) {
  impl_ = ge::MakeShared<Impl>(omg_context);
  if (impl_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "create Impl failed.");
    GELOGE(MEMALLOC_FAILED, "[Create][Impl] Make shared failed");
    return MEMALLOC_FAILED;
  }

  ErrorManager::GetInstance().SetStage(error_message::kInitialize, error_message::kOpsProtoInit);
  string opsproto_path;
  GetOpsProtoPath(opsproto_path);
  GELOGI("Get opsproto path is %s", opsproto_path.c_str());
  OpsProtoManager *manager = OpsProtoManager::Instance();
  map<string, string> option_tmp;
  option_tmp.emplace(std::pair<string, string>(string("ge.opsProtoLibPath"), opsproto_path));
  (void)manager->Initialize(option_tmp);

  Status ret = impl_->graph_manager_.Initialize(options);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_INIT_FAILED, "[Call][Initialize] Graph manager initialize failed.");
    return GE_GENERATOR_GRAPH_MANAGER_INIT_FAILED;
  }
  // get ek file
  auto iter = options.find(EK_FILE);
  if (iter != options.end()) {
    impl_->save_param_.ek_file = iter->second;
  }
  // get cert file
  iter = options.find(CERT_FILE);
  if (iter != options.end()) {
    impl_->save_param_.cert_file = iter->second;
  }
  // get hw key file
  iter = options.find(HW_KEY_FILE);
  if (iter != options.end()) {
    impl_->save_param_.hw_key_file = iter->second;
  }
  // get private file
  iter = options.find(PRIVATE_KEY_FILE);
  if (iter != options.end()) {
    impl_->save_param_.pri_key_file = iter->second;
  }

  // get build mode
  iter = options.find(BUILD_MODE);
  if (iter != options.end()) {
    impl_->build_mode_ = iter->second;
  }
  // get build step
  iter = options.find(BUILD_STEP);
  if (iter != options.end()) {
    impl_->build_step_ = iter->second;
  }
  return SUCCESS;
}

Status GeGenerator::Finalize() {
  ErrorManager::GetInstance().SetStage(error_message::kFinalize, error_message::kFinalize);
  if (impl_ == nullptr) {
    return SUCCESS;
  }
  Status ret = impl_->graph_manager_.Finalize();
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED, "[Call][Finalize] Graph manager finalize failed.");
    return GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED;
  }
  return SUCCESS;
}

Status GeGenerator::GenerateOfflineModel(const Graph &graph, const string &file_name_prefix,
                                         const vector<GeTensor> &inputs) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  GELOGI("Start to generate offline model.");
  ModelBufferData model;
  return GenerateModel(graph, file_name_prefix, inputs, model, true);
}

Status GeGenerator::GenerateOnlineModel(const Graph &graph, const vector<GeTensor> &inputs, ModelBufferData &model) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  return GenerateModel(graph, "online", inputs, model, false);
}

Status GeGenerator::GenerateInfershapeGraph(const Graph &graph) {
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);

  Status ret = impl_->GenerateInfershapeGraph(graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][GenerateInfershapeGraph] Dump infershape json failed");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "[Call][Finalize] graph_manager finalize fail.");
    }
    return ret;
  }
  GELOGI("Generate infer shape graph success");
  return SUCCESS;
}

std::mutex GeGenerator::Impl::mutex_;

// Remove the space and tab before and after the string
std::string GeGenerator::Impl::Trim(const std::string &str) {
  if (str.empty()) {
    return str;
  }

  std::string::size_type start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return str;
  }

  std::string::size_type end = str.find_last_not_of(" \t\r\n") + 1;
  return str.substr(start, end);
}

// Parsing the command line
bool GeGenerator::Impl::ParseVersion(const std::string &line, std::string &version) {
  std::string flag = "Version=";
  std::string temp = Trim(line);

  if (temp.empty()) {
    GELOGW("line is empty.");
    return false;
  }

  std::string::size_type pos = temp.find(flag);
  if (pos == std::string::npos) {
    GELOGW("Incorrect line [%s], it must include [%s].", line.c_str(), flag.c_str());
    return false;
  }

  if (temp.size() == flag.size()) {
    GELOGW("version information is empty. %s", line.c_str());
    return false;
  }

  version = temp.substr(pos + flag.size());

  return true;
}

bool GeGenerator::Impl::GetVersionFromPath(const std::string &file_path, std::string &version) {
  // Normalize the path
  string resolved_file_path = RealPath(file_path.c_str());
  if (resolved_file_path.empty()) {
    GELOGW("Invalid input file path [%s], make sure that the file path is correct.", file_path.c_str());
    return false;
  }
  std::ifstream fs(resolved_file_path, std::ifstream::in);
  if (!fs.is_open()) {
    GELOGW("Open %s failed.", file_path.c_str());
    return false;
  }

  std::string line;
  if (getline(fs, line)) {
    if (!ParseVersion(line, version)) {
      GELOGW("Parse version failed. content is [%s].", line.c_str());
      fs.close();
      return false;
    }
  } else {
    GELOGW("No version information found in the file path:%s", file_path.c_str());
    fs.close();
    return false;
  }

  fs.close();  // close the file
  return true;
}

// Set package version information in the model
bool GeGenerator::Impl::SetAtcVersionInfo(AttrHolder &obj) {
  std::string path_base = ge::GELib::GetPath();
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);

  std::string version_path = path_base + "version.info";
  std::string version;
  if (!GetVersionFromPath(version_path, version)) {
    GELOGW("Get atc version information failed!");
    return false;
  }
  // set version info
  if (!ge::AttrUtils::SetStr(obj, ATTR_MODEL_ATC_VERSION, version)) {
    GELOGW("Ge model set atc version failed!");
    return false;
  }
  return true;
}

// Set package version information in the model
bool GeGenerator::Impl::SetOppVersionInfo(AttrHolder &obj) {
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env == nullptr) {
    GELOGW("Get environment variable ASCEND_OPP_PATH failed!");
    return false;
  }
  std::string version_path = path_env;
  version_path += "/version.info";
  std::string version;
  if (!GetVersionFromPath(version_path, version)) {
    GELOGW("Get opp version information failed!");
    return false;
  }
  // set version info
  if (!ge::AttrUtils::SetStr(obj, ATTR_MODEL_OPP_VERSION, version)) {
    GELOGW("Ge model set opp version failed!");
    return false;
  }
  return true;
}

bool GeGenerator::Impl::SetOmSystemInfo(AttrHolder &obj) {
  std::string soc_version;
  (void)ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);
  GELOGI("SetOmSystemInfo soc_version: %s", soc_version.c_str());
  if (!ge::AttrUtils::SetStr(obj, "soc_version", soc_version)) {
    GELOGW("SetStr of soc_version failed.");
    return false;
  }

  std::string framework_type;
  (void)ge::GetContext().GetOption(ge::FRAMEWORK_TYPE, framework_type);
  GELOGI("SetOmSystemInfo framework_type: %s", framework_type.c_str());
  auto iter = ge::kFwkTypeToStr.find(framework_type);
  if (iter == ge::kFwkTypeToStr.end()) {
    GELOGW("Can not find framework_type in the map.");
    return false;
  }
  if (!ge::AttrUtils::SetStr(obj, "framework_type", iter->second)) {
    GELOGW("SetStr of framework_type failed.");
    return false;
  }
  return true;
}

Status GeGenerator::SetModelNameForDump(const GeRootModelPtr &ge_root_model) {
  bool is_unknown_shape = false;
  Status ret = ge_root_model->CheckIsUnknownShape(is_unknown_shape);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Check][IsUnknownShape]Check root model is unknown shape failed, model id:%u",
           ge_root_model->GetModelId());
    REPORT_CALL_ERROR("E19999", "Check root model is unknown shape failed, model id:%u",
                      ge_root_model->GetModelId());
    return FAILED;
  }
  GeModelPtr model_root = nullptr;
  if (is_unknown_shape) {
    model_root = MakeShared<GeModel>();
    GE_CHECK_NOTNULL(model_root);
    model_root->SetGraph(GraphUtils::CreateGraphFromComputeGraph(ge_root_model->GetRootGraph()));
    ge_root_model->SetSubgraphInstanceNameToModel(ge_root_model->GetRootGraph()->GetName(), model_root);
  }

  ModelHelper model_helper;
  string model_name;
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  Status name_ret = model_helper.GetModelNameFromMergedGraphName(ge_root_model->GetRootGraph()->GetName(),
                                                                 model_name);
  if (name_ret != SUCCESS) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10000", {"parameter"}, {"output"});
    GELOGE(FAILED, "[Check][GetModelNameStep]Get model_name failed. Param --output is invalid, root graph name: %s",
           ge_root_model->GetRootGraph()->GetName().c_str());
    return PARAM_INVALID;
  }
  map<string, GeModelPtr> name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GeModelPtr &ge_model = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];
  GE_CHECK_NOTNULL(ge_model);
  ge_model->SetName(model_name);
  return SUCCESS;
}

Status GeGenerator::GenerateModel(const Graph &graph, const string &file_name_prefix, const vector<GeTensor> &inputs,
                                  ModelBufferData &model, bool is_offline) {
  rtContext_t ctx = nullptr;
  auto rt = rtCtxGetCurrent(&ctx);
  if (rt != RT_ERROR_NONE) {
    GELOGD("Current ctx is null.");
    ctx = nullptr;
  }
  std::function<void()> callback = [&]() {
    if (ctx != nullptr) {
      (void)rtCtxSetCurrent(ctx);
    }
  };
  GE_MAKE_GUARD(restore, callback);

  GeRootModelPtr ge_root_model = nullptr;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  impl_->is_offline_ = is_offline;
  Status ret = impl_->BuildModel(graph, inputs, ge_root_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][Model] failed, ret:%d.", ret);
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "[Call][Finalize] graph_manager finalize fail.");
    }
    return ret;
  }

  /// BUILD_MODE_TUNING with BUILD_STEP_BEFORE_UB_MATCH no need save model;
  /// BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER no need save model;
  /// BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER_SUB no need save model.
  if ((impl_->build_mode_ == BUILD_MODE_TUNING) &&
      (impl_->build_step_ == BUILD_STEP_BEFORE_UB_MATCH || impl_->build_step_ == BUILD_STEP_AFTER_BUILDER ||
       impl_->build_step_ == BUILD_STEP_AFTER_BUILDER_SUB)) {
    GELOGI("Build mode:%s with step:%s no need SaveModel.",
           impl_->build_mode_.c_str(),
           impl_->build_step_.c_str());
    return SUCCESS;
  }

  GE_CHECK_NOTNULL(ge_root_model);
  ret = SetModelNameForDump(ge_root_model);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = impl_->SaveRootModel(file_name_prefix, ge_root_model, model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][RootModel] failed, ret:%d, file:%s", ret, file_name_prefix.c_str());
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }
  return SUCCESS;
}

namespace {
  bool IsNeedConnectInputOpForSingleOp(GeTensorDesc &tensor_desc) {
    bool is_need = true;
    // format and dtype is all reserved, stand for Optional input. When singleop scene
    if (tensor_desc.GetFormat() == FORMAT_RESERVED && tensor_desc.GetDataType() == DT_UNDEFINED) {
      is_need = false;
    }
    return is_need;
  }

  Status CheckDynamicSupport(GeModelPtr &ge_model, const ComputeGraphPtr &graph) {
    bool support_dynamic = true;
    bool is_dynamic = false;
    for (const auto &node : graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node);
      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if (op_desc->GetOpEngineName() != kAIcoreEngine) {
        continue;
      }
      if (AttrUtils::HasAttr(op_desc, kAttrSupportDynamicShape)) {
        is_dynamic = true;
        (void) AttrUtils::GetBool(op_desc, kAttrSupportDynamicShape, support_dynamic);
        if (!support_dynamic) {
          GELOGW("Node[%s] doesn't support dynamic shape.", node->GetName().c_str());
          break;
        }
      }
    }
    if (is_dynamic) {
      (void) AttrUtils::SetBool(ge_model, kAttrSupportDynamicShape, support_dynamic);
    }
    return SUCCESS;
  }
}

bool GeGenerator::CheckNoAicore(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    if (op_desc->GetOpEngineName() == kAIcoreEngine) {
      return false;
    }
  }
  return true;
}

void GeGenerator::RemoveConst(const vector<GeTensor> &inputs, vector<GeTensor> &outputs) {
  for (auto &input : inputs) {
    GeTensorDesc input_desc = input.GetTensorDesc();
    bool is_const = false;
    (void)AttrUtils::GetBool(input_desc, CONST_ATTR_NAME_INPUT, is_const);
    bool is_optional = IsOptional(input_desc);
    if (!is_optional && !is_const) {
      outputs.emplace_back(input);
    }
  }
}

Status GeGenerator::CheckForSingleOp(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                     const vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL_EXEC(op_desc, return PARAM_INVALID);
  if (!inputs.empty() && (inputs.size() != op_desc->GetAllInputsSize())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "inputs size" + FmtToStr(op_desc->GetAllInputsSize()),
        "tensor size is " + FmtToStr(inputs.size())});
    GELOGE(PARAM_INVALID, "[Check][Param] Tensor size: %zu, op:%s(%s) Inputs size: %zu, not equal",
           inputs.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetAllInputsSize());
    return PARAM_INVALID;
  }
  if (!outputs.empty() && (outputs.size() != op_desc->GetOutputsSize())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E14001", {"opname", "optype", "value", "reason"},
        {op_desc->GetName(), op_desc->GetType(), "outputs size" + FmtToStr(op_desc->GetOutputsSize()),
        "tensor size is " + FmtToStr(outputs.size())});
    GELOGE(PARAM_INVALID, "[Check][Param] Tensor size: %zu, op:%s(%s) Outputs size: %zu, not equal",
           outputs.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetOutputsSize());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status GeGenerator::InferFormatForSingleOp(OpDescPtr &op_desc, Graph &graph) {
  GE_CHECK_NOTNULL(op_desc);
  if (OperatorFactoryImpl::GetInferFormatFunc(op_desc->GetType()) != nullptr) {
    auto node_op = ge::OperatorFactoryImpl::CreateOperator("node_op", op_desc->GetType());
    if (node_op.IsEmpty()) {
      GELOGW("get op from OperatorFactory fail. op type: %s", op_desc->GetType().c_str());
    } else {
      GELOGD("get op from OperatorFactory success. op type: %s", op_desc->GetType().c_str());
      auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
      if (temp_op_desc == nullptr) {
        REPORT_INNER_ERROR("E19999", "GetOpDescFromOperator failed, as return nullptr, type:%s",
                           op_desc->GetType().c_str());
        GELOGE(FAILED, "[Get][OpDesc] temp op desc is null, type:%s", op_desc->GetType().c_str());
        return FAILED;
      }
      if (!op_desc->UpdateInputName(temp_op_desc->GetAllInputName())) {
        GELOGW("InferFormatForSingleOp UpdateInputName failed");
      }
      if (!op_desc->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
        GELOGW("InferFormatForSingleOp UpdateOutputName failed");
      }
    }
    node_op.BreakConnect();
  }
  auto comp_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(comp_graph);
  auto node = comp_graph->FindNode(op_desc->GetName());
  GE_CHECK_NOTNULL(node);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  auto ret = op_desc->CallInferFormatFunc(op);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERROR("E19999", "call InferFormatFunc for single op:%s fail",
                       op_desc->GetName().c_str());
    GELOGE(FAILED, "[Call][InferFormatFunc] for single op:%s fail.", op_desc->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status GeGenerator::BuildSingleOp(OpDescPtr &op_desc, const vector<GeTensor> &inputs, const vector<GeTensor> &outputs,
                                  const string &model_file_name, OpEngineType engine_type, ModelBufferData &model_buff,
                                  bool is_offline, int32_t compile_flag) {
  GELOGD("Inputs size is %zu, outputs size is %zu.", inputs.size(), outputs.size());
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  impl_->is_offline_ = is_offline;
  (void)AttrUtils::SetBool(op_desc, ATTR_SINGLE_OP_SCENE, true);

  if (CheckForSingleOp(op_desc, inputs, outputs) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][Param] input param is invalid when build single op:%s!",
           op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  OmgContext &omg_context = impl_->omg_context_;
  omg_context.is_dynamic_input = ContainsDynamicInpus(*op_desc);

  if (op_desc->HasAttr(ATTR_NAME_UNREGST_OPPATH)) {
    impl_->is_singleop_unregistered_ = true;
  }

  // 0. Save original attributes.
  OpDescPtr op_desc_tmp = AttrUtils::CloneOpDesc(op_desc);
  GE_CHECK_NOTNULL(op_desc_tmp);

  bool fuzz_compile_flag = false;
  if (!HasShapeRange(inputs) && compile_flag == kFuzzBuildPattern) {
    fuzz_compile_flag = true;
  }
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_FUZZ_BUILD, fuzz_compile_flag);
  impl_->omg_context_.fuzz_compile_flag = fuzz_compile_flag;

  // 1. Create ComputeGraph.
  string name = ge::CurrentTimeInStr() + "_" + model_file_name;
  Graph graph;
  GE_CHK_STATUS(BuildSingleOpGraph(op_desc, inputs, outputs, name, graph),
                "[Build][Graph] for single op:%s fail.", op_desc->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(InferFormatForSingleOp(op_desc, graph));

  // 2. check engine type when compile online
  if (model_file_name == kFileNameSuffix) {
    auto comp_graph = GraphUtils::GetComputeGraph(graph);
    GE_CHECK_NOTNULL(comp_graph);
    auto node = comp_graph->FindNode(op_desc->GetName());
    Status ret = CheckEngineTypeSupport(node, engine_type);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Check][EngineType]not support node:%s with engine of %d.", node->GetName().c_str(), engine_type);
      return ret;
    }
  }

  GELOGI("ATC parser success in single op build.");

  GeRootModelPtr ge_root_model = nullptr;
  vector<GeTensor> data_inputs;
  RemoveConst(inputs, data_inputs);
  GE_CHK_STATUS_RET_NOLOG(impl_->BuildModel(graph, data_inputs, ge_root_model));
  map<string, GeAttrValue> op_attrs = op_desc_tmp->GetAllAttrs();
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  map<string, GeModelPtr> name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  if (name_to_ge_model.empty()) {
    REPORT_CALL_ERROR("E19999", "GetSubgraphInstanceNameToModel failed.");
    GELOGE(PARAM_INVALID, "[Get][Name] GetSubgraphInstanceNameToModel is empty.");
    return PARAM_INVALID;
  }
  const ComputeGraphPtr root_graph = ge_root_model->GetRootGraph();
  GeModelPtr &ge_model = name_to_ge_model.begin()->second;
  GE_CHK_STATUS_RET_NOLOG(CheckDynamicSupport(ge_model, root_graph));
  GELOGI("After build model, The opType in op_desc_tmp is [%s]", op_desc_tmp->GetType().c_str());

  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc, kAicpuAllshape, all_shape);
  GELOGD("Node: %s, all_shape is %d, compile_flag is %d.", op_desc->GetName().c_str(), all_shape, compile_flag);
  (void)AttrUtils::SetInt(ge_model, ATTR_NAME_BUILD_MODE, fuzz_compile_flag);
  if (all_shape) {
    (void)AttrUtils::SetBool(ge_model, kAicpuAllshape, all_shape);
  }
  if (all_shape && CheckNoAicore(root_graph)) {
    GELOGD("Get aicpu all_shape kernel!");
    vector<GeTensor> inputs_dynamic;
    vector<GeTensor> outputs_dynamic;
    GE_CHK_STATUS_RET_NOLOG(ResetTensorVecShape(inputs, inputs_dynamic));
    GE_CHK_STATUS_RET_NOLOG(ResetTensorVecShape(outputs, outputs_dynamic));
    GE_CHK_STATUS_RET_NOLOG(
      impl_->SaveParams(ge_model, op_desc_tmp->GetType(), op_attrs, inputs_dynamic, outputs_dynamic));
  } else if (fuzz_compile_flag) {
    GeAttrValue::LIST_NAMED_ATTRS fuzz_build_attrs;
    if (GetFuzzBuildAttrs(op_desc, ge_root_model, fuzz_build_attrs) != SUCCESS) {
      GELOGE(FAILED, "[Get][FuzzRet]Failed to get fuzz build result of %s.", op_desc->GetName().c_str());
      return FAILED;
    }
    if (!fuzz_build_attrs.empty()) {
      GE_CHK_BOOL_EXEC(AttrUtils::SetListNamedAttrs(ge_model, ATTR_NAME_FUZZ_BUILD_RES_ATTRS, fuzz_build_attrs),
                       REPORT_CALL_ERROR("E19999", "Set model:%s(id:%u) attr:%s failed.",
                                         ge_model->GetName().c_str(), ge_model->GetModelId(),
                                         ATTR_NAME_FUZZ_BUILD_RES_ATTRS.c_str());
                       return FAILED, "Set model:%s(id:%u) attr:%s failed.",
                       ge_model->GetName().c_str(), ge_model->GetModelId(), ATTR_NAME_FUZZ_BUILD_RES_ATTRS.c_str());
    }
    GE_CHK_STATUS_RET_NOLOG(impl_->SaveParams(ge_model, op_desc_tmp->GetType(), op_attrs, inputs, outputs));
  } else {
    GE_CHK_STATUS_RET_NOLOG(impl_->SaveParams(ge_model, op_desc_tmp->GetType(), op_attrs, inputs, outputs));
  }
  GELOGI("Start save GeModel to Model buffer");
  GE_CHK_STATUS_RET_NOLOG(impl_->SaveModel(model_file_name, ge_model, model_buff));
  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Compiling a single operator into an offline model
 * @param [in] OpDescPtr &op_desc: Operator description info that needs to be compiled into an offline model file
 * @param [in] vector<GeTensor> &inputs: Operator input data description information.
 * @param [in] vector<GeTensor> &outputs: Operator output data description information.
 * @param [in] const string &model_file_name: Offline model filename.
 * @param [in] compile_flag: op build flag from atc
 * @return SUCCESS handle successfully / others handle failed
 */
Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                       const vector<GeTensor> &outputs, const string &model_file_name,
                                       int32_t compile_flag) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  GELOGI("Start to build single op offline model, input size: %zu, output size: %zu", inputs.size(), outputs.size());
  ModelBufferData model_buff;
  OpEngineType engine_type = ENGINE_SYS;
  Status status = BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type, model_buff, true, compile_flag);
  GELOGI("Finish build single offline model, status: %u", status);
  return status;
}

/**
 * @ingroup ge
 * @brief Compiling a single operator into online buffer
 * @param [in] OpDescPtr &op_desc: Operator description info that needs to be compiled into an offline model file
 * @param [in] vector<GeTensor> &inputs: Operator input data description information.
 * @param [in] vector<GeTensor> &outputs: Operator output data description information.
 * @param [in] engine_type: specific engine.
 * @param [in] compile_flag: op build flag, compile flag by acl
 * @param [out] ModelBufferData &Model_buff: Model_buff: model buffer of the op.
 * @return SUCCESS handle successfully / others handle failed
 */

Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                       const vector<GeTensor> &outputs, OpEngineType engine_type,
                                       ModelBufferData &model_buff) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  GELOGI("Start to build single op online, input size: %zu, output size: %zu", inputs.size(), outputs.size());
  Status status = BuildSingleOp(op_desc, inputs, outputs, kFileNameSuffix, engine_type, model_buff, false);
  GELOGI("Finish build single online model, status: %u", status);
  return status;
}

Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                       const vector<GeTensor> &outputs, OpEngineType engine_type, int32_t compile_flag,
                                       ModelBufferData &model_buff) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  GELOGI("Start to build single op online, input size: %zu, output size: %zu", inputs.size(), outputs.size());
  Status status = BuildSingleOp(op_desc, inputs, outputs, kFileNameSuffix, engine_type, model_buff, false,
                                compile_flag);
  GELOGI("Finish build single online model, status: %u", status);
  return status;
}

Status GeGenerator::BuildSingleOpGraph(OpDescPtr &op_desc, const vector<GeTensor> &inputs,
                                       const vector<GeTensor> &outputs, std::string graph_name, Graph &graph) {
  ge::ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL_EXEC(compute_graph, return INTERNAL_ERROR);

  // 1. Add Node to ComputeGraph.
  NodePtr op_node = compute_graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(op_node, return INTERNAL_ERROR);

  // 2. Create InputData node.
  int32_t arg_index = 0;
  int32_t data_index = 0;
  if (inputs.empty()) {
    for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
      GE_CHECK_NOTNULL_EXEC(input_desc, return INTERNAL_ERROR);
      if (!IsNeedConnectInputOpForSingleOp(*input_desc)) {
        continue;
      }
      GE_CHK_STATUS_RET_NOLOG(AddInputs(compute_graph, op_node, *input_desc, arg_index, false, data_index));
      arg_index++;
    }
  } else {
    for (const auto &in_desc : inputs) {
      GE_CHK_STATUS_RET_NOLOG(AddInputs(compute_graph, op_node, in_desc.GetTensorDesc(), arg_index, true, data_index));
      arg_index++;
    }
  }

  // 3. Create Output node.
  if (!outputs.empty()) {
    GE_CHK_STATUS_RET_NOLOG(AddOutputs(compute_graph, op_node, outputs));
  }

  // dump ComputeGraph node.
  compute_graph->Dump();
  graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  return SUCCESS;
}

Status GeGenerator::Impl::SaveParams(GeModelPtr &ge_model, const string &type, const map<string, GeAttrValue> &attrs,
                                     const vector<GeTensor> &inputs, const vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL_EXEC(ge_model, return PARAM_INVALID);
  GE_CHK_BOOL_EXEC_NOLOG(graph_manager_.SaveParams(*ge_model, type, attrs, inputs, outputs) == SUCCESS,
                         (void)graph_manager_.Finalize();
                         return FAILED);

  return SUCCESS;
}

Status GeGenerator::Impl::SaveModel(const string &file_name_prefix, GeModelPtr &model, ModelBufferData &model_buff) {
  // set atc version
  if (!SetAtcVersionInfo(*(model.get()))) {
    GELOGW("SetPackageVersionInfo of atc failed!");
  }
  // set opp version
  if (!SetOppVersionInfo(*(model.get()))) {
    GELOGW("SetPackageVersionInfo of ops failed!");
  }
  ModelHelper model_helper;
  model_helper.SetSaveMode(is_offline_);
  Status ret = model_helper.SaveToOmModel(model, save_param_, file_name_prefix, model_buff);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][SaveToOmModel] Save to om model failed");
    return ret;
  }
  return SUCCESS;
}

Status GeGenerator::Impl::SaveRootModel(const string &file_name_prefix, GeRootModelPtr &ge_root_model,
                                        ModelBufferData &model_buff) {
  bool is_unknown_shape = false;
  auto ret = ge_root_model->CheckIsUnknownShape(is_unknown_shape);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "root model(id:%u) CheckIsUnknownShape failed, ret:%d",
                      ge_root_model->GetModelId(), ret);
    GELOGE(FAILED, "[Check][RootModel] is unkonwn shape failed, ret:%d", ret);
    return FAILED;
  }
  GELOGD("begin save root model, cur model is unkonwn shape model ? : %d", is_unknown_shape);
  GE_CHK_BOOL_EXEC(!ge_root_model->GetSubgraphInstanceNameToModel().empty(),
                   REPORT_CALL_ERROR("E19999", "root model(id:%u) has no sub model.", ge_root_model->GetModelId());
                   return FAILED, "[Get][SubModel] ge root model has no sub model")
  GeModelPtr model_root = nullptr;
  if (is_unknown_shape) {
    auto name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
    model_root = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];
  } else {
    model_root = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  }
  GE_CHECK_NOTNULL(model_root);
  // set atc version
  if (!SetAtcVersionInfo(*(model_root.get()))) {
    GELOGW("SetPackageVersionInfo of atc failed!");
  }
  // set opp version
  if (!SetOppVersionInfo(*(model_root.get()))) {
    GELOGW("SetPackageVersionInfo of ops failed!");
  }
  if (!SetOmSystemInfo(*(model_root.get()))) {
    GELOGW("SetOmsystemInfo failed!");
  }
  ModelHelper model_helper;
  model_helper.SetSaveMode(is_offline_);
  ret = model_helper.SaveToOmRootModel(ge_root_model, save_param_, file_name_prefix, model_buff, is_unknown_shape);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "SaveToOmRootModel failed, ret:%d, model id:%u", ret, ge_root_model->GetModelId());
    GELOGE(ret, "[Call][SaveToOmRootModel] failed, ret:%d, model id:%u", ret, ge_root_model->GetModelId());
    return ret;
  }
  return SUCCESS;
}

Status GeGenerator::Impl::BuildModel(const Graph &graph, const vector<GeTensor> &inputs,
                                     GeRootModelPtr &ge_root_model) {
  static std::atomic<GraphId> atomic_graph_id(0);
  auto graph_id = atomic_graph_id.fetch_add(1);
  const std::map<std::string, std::string> options;
  Status ret = graph_manager_.AddGraph(graph_id, graph, options, omg_context_);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "add graph(id:%u) failed, ret:%d", graph_id, ret);
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "[Add][Graph] fail, graph id: %u", graph_id);
    (void)graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  graph_manager_.SetOptionsRunGraphFlag(false);

  static std::atomic<uint64_t> atomic_session_id(0);
  auto session_id = atomic_session_id.fetch_add(1);
  // This is a temporary add for graph with variable
  auto version = static_cast<int32_t>(SessionVersion::ClOUD_VERSION);
  ret = VarManager::Instance(session_id)->Init(version, session_id, kDefaultDeviceId, kDefaultJobId);
  GELOGI("Start init var instance, session_id %lu", session_id);
  if (ret != SUCCESS) {
    GELOGW("Failed init var instance, session_id %lu", session_id);
  }
  if (is_singleop_unregistered_) {
    ret = graph_manager_.BuildGraphForUnregisteredOp(graph_id, inputs, ge_root_model, session_id);
  } else {
    ret = graph_manager_.BuildGraph(graph_id, inputs, ge_root_model, session_id);
  }

  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "build graph failed, graph id:%u, ret:%d", graph_id, ret);
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED, "[Build][Graph] fail, graph id: %u", graph_id);
  }

  RtContextUtil::GetInstance().DestroyRtContexts(session_id);
  Analyzer::GetInstance()->DestroySessionJsonObject(session_id);
  VarManagerPool::Instance().RemoveVarManager(session_id);
  return ret;
}

Status GeGenerator::Impl::GenerateInfershapeGraph(const Graph &graph) {
  static std::atomic<GraphId> atomic_graph_id(0);
  auto graph_id = atomic_graph_id.fetch_add(1);
  const std::map<std::string, std::string> options;
  Status ret = graph_manager_.AddGraph(graph_id, graph, options, omg_context_);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "add graph failed, graph id:%u, ret:%d", graph_id, ret);
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "[Add][Graph] failed, graph id: %u", graph_id);
    (void)graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  ret = graph_manager_.GenerateInfershapeGraph(graph_id);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "GenerateInfershapeGraph failed, graph id:%u, ret:%d", graph_id, ret);
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED,
           "[Generate][Graph] failed, graph id:%u, ret:%d", graph_id, ret);
    return GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED;
  }

  return SUCCESS;
}
}  // namespace ge
