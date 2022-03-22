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

#include <dirent.h>
#include <dlfcn.h>
#include <gflags/gflags.h>
#include <sys/types.h>
#include <unistd.h>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <iostream>
#include "framework/common/gflags_util.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "external/ge/ge_api.h"
#include "framework/generator/ge_generator.h"
#include "graph/anchor.h"
#include "graph/debug/ge_attr_define.h"
#include "external/graph/graph.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "ir_build/option_utils.h"
#include "framework/omg/omg.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "parser/common/register_tbe.h"
#include "register/op_registry.h"
#include "offline/single_op_parser.h"
#include "external/ge/ge_ir_build.h"

using domi::BuildMode;
using domi::OpRegistrationData;
using domi::OpRegistry;
using domi::Status;
using domi::SUCCESS;
using ge::GEN_OM_MODEL;
using ge::GflagsUtils;
using ge::MODEL_TO_JSON;
using ge::ONLY_PRE_CHECK;
using ge::ParseInputShape;
using ge::PBTXT_TO_JSON;
using std::map;
using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {
static bool is_dynamic_input = false;
const char *const kModeSupport = "only support 0(model to framework model), "
                                 "1(framework model to json), 3(only pre-check), "
                                 "5(pbtxt to json), 6(display model info)";
const char *const kModelToJsonSupport = "only support 0(Caffe) 3(TensorFlow) 5(Onnx) when model set 1";
const char *const kCaffeFormatSupport = "only support NCHW, ND in Caffe model";
const char *const kTFFormatSupport = "only support NCHW, NHWC, ND, NCDHW, NDHWC in TF model";
const char *const kONNXFormatSupport = "only support NCHW, ND, NCDHW in ONNX model";
// limit available mem size 2G
const long kMinAvailableMem = 2097152;  // 2 * 1024 * 1024
}  // namespace

DEFINE_string(model, "", "The model file.");
DEFINE_string(output, "", "The output file path&name.");
DEFINE_int32(framework, -1, "Framework type(0:Caffe; 1:LuoJiaNet; 3:Tensorflow; 5:Onnx).");
DEFINE_string(weight, "", "Optional; weight file. Required when framework is Caffe.");

DEFINE_string(input_shape, "",
              "Optional; shape of input data. Required when framework is caffe "
              "or TensorFLow or LuoJiaNet or Onnx. "
              "Format: \"input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2\"");
DEFINE_string(input_shape_range, "",
              "Optional; shape range of input data. Required when framework is caffe "
              "or TensorFLow or Onnx. "
              "Format: \"input_name1:[n1~n2,c1,h1,w1];input_name2:[n2~n3,c2,h2,w2]\"");
DEFINE_bool(h, false, "show this help message");
DEFINE_string(cal_conf, "", "Optional; the calibration config file.");

DEFINE_string(insert_op_conf, "", "Optional; the config file to insert new op, for example AIPP op.");
DEFINE_string(op_name_map, "", "Optional; custom op name mapping file.");

DEFINE_string(target, "", "Optional; mini.");

DEFINE_string(om, "", "The model file to be converted to json.");
DEFINE_string(json, "", "The output json file path&name which is converted from a model.");
DEFINE_int32(mode, 0,
             "Optional; run mode, 0(default): model => framework model; 1: "
             "framework model => json; 3: only pre-check; 5: txt => json.");

DEFINE_string(out_nodes, "",
              "Optional; output nodes designated by users."
              "Format: \"node_name1:0;node_name1:1;node_name2:0\"");

DEFINE_string(op_precision_mode, "", "Optional; operator precision mode configuration file path");

DEFINE_string(precision_mode, "force_fp16",
              "Optional; precision mode."
              "Support force_fp16, force_fp32, allow_mix_precision, allow_fp32_to_fp16, must_keep_origin_dtype.");

DEFINE_string(modify_mixlist, "", "Optional; operator mixed precision configuration file path");

DEFINE_string(keep_dtype, "",
              "Optional; config file to specify the precision used by the operator during compilation.");

DEFINE_string(input_format, "",
              "Optional; input_format, format of input data, NCHW;NHWC."
              "Format:\"NHWC\"");

DEFINE_string(check_report, "check_result.json", "Optional; the pre-checking report file.");

DEFINE_string(input_fp16_nodes, "",
              "Optional; input node datatype is fp16 and format is NC1HWC0."
              "Format:\"node_name1;node_name2\"");

DEFINE_string(is_output_adjust_hw_layout, "",
              "Optional; Net output node's datatype is fp16 and format is "
              "NC1HWC0, or not."
              "Format:\"false,true,false,true\"");

DEFINE_string(is_input_adjust_hw_layout, "",
              "Optional; Intput node's datatype is fp16 and format is "
              "NC1HWC0, or not."
              "Format:\"false,true,false,true\"");

DEFINE_string(output_type, "",
              "Optional; output type! "
              "Support FP32,FP16,INT8,INT16,UINT16,UINT8,INT32,INT64,UINT32,UINT64,DOUBLE.");

DEFINE_string(op_select_implmode, "",
              "Optional; op select implmode! "
              "Support high_precision, high_performance, "
              "high_precision_for_all, high_performance_for_all.");

DEFINE_string(optypelist_for_implmode, "",
              "Optional; Nodes need use implmode selected in op_select_implmode "
              "Format:\"node_name1,node_name2\"");

DEFINE_string(singleop, "", "Optional; If set, generate single op model with the given json file.");

DEFINE_int32(disable_reuse_memory, 0, "Optional; If set to 1, disable reuse memory when generating if.");

DEFINE_string(auto_tune_mode, "", "Optional; Set tune mode.");

DEFINE_string(soc_version, "", "The soc version.");

DEFINE_string(core_type, "AiCore", "Optional; If set to VectorCore, only use vector core.");

DEFINE_string(aicore_num, "", "Optional; Set aicore num");

DEFINE_string(buffer_optimize, "l2_optimize", "Optional; buffer optimize");

DEFINE_string(fusion_switch_file, "", "Optional; Set fusion switch file path");

DEFINE_string(save_original_model, "", "Optional; enable output original offline model. false(default)");

DEFINE_string(dynamic_batch_size, "",
              "Optional; If set, generate dynamic multi batch model. "
              "Different batch sizes are split by ','."
              "dynamic_batch_size, dynamic_image_size and dynamic_dims can only be set one.");

DEFINE_string(dynamic_image_size, "",
              "Optional; If set, generate dynamic multi image size model."
              "Different groups of image size are split by ';',"
              "while different dimensions of each group are split by ','."
              "dynamic_batch_size, dynamic_image_size and dynamic_dims can only be set one.");

DEFINE_string(dynamic_dims, "",
              "Optional; If set, generate dynamic input size model. "
              "Different groups of size are split by ';', while different dimensions of each group are split by ','."
              "dynamic_batch_size, dynamic_image_size and dynamic_dims can only be set one.");

DEFINE_string(enable_small_channel, "0", "Optional; If set to 1, small channel is enabled.");

DEFINE_string(enable_compress_weight, "false",
              "Optional; enable compress weight. true: enable; false(default): disable");

DEFINE_string(compress_weight_conf, "", "Optional; the config file to compress weight");

DEFINE_string(enable_single_stream, "", "Optional; enable single stream. true: enable; false(default): disable");

DEFINE_string(log, "null", "Optional; generate atc log. Support debug, info, warning, error, null");

DEFINE_string(dump_mode, "0", "Optional; generate infershape json,only support 1 , 0.");

DEFINE_int32(op_debug_level, 0, "Optional; configure debug level of compiler. 0(default): close debug; "
             "1: open TBE compiler, export ccec file and TBE instruction mapping file; 2: open ccec compiler; "
             "3: disable debug, and keep generating kernel file (.o and .json); 4: disable debug, "
             "keep generation kernel file (.o and .json) and generate the operator CCE file (.cce) "
             "and the UB fusion computing description file (.json)");
DEFINE_string(enable_scope_fusion_passes, "", "Optional; validate the non-general scope fusion pass,"
              "multiple names can be set and separated by ','.");
DEFINE_string(debug_dir, "", "Optional; the path to save the intermediate files of operator compilation");

DEFINE_string(op_compiler_cache_dir, "", "Optional; the path to cache operator compilation files");

DEFINE_string(op_compiler_cache_mode, "", "Optional; choose the operator compiler cache mode");

DEFINE_string(mdl_bank_path, "", "Optional; model bank path");

DEFINE_string(op_bank_path, "", "Optional; op bank path");

DEFINE_string(display_model_info, "0", "Optional; display model info");

DEFINE_string(device_id, "0", "Optional; user device id");

class GFlagUtils {
 public:
  /**
   * @name   InitGFlag
   * @brief  initialize gflag
   * @return void
   */
  static void InitGFlag(int argc, char *argv[]) {
    // -help
    gflags::SetUsageMessage(
        "usage: ./atc <args>\n"
        "generate offline model example:\n"
        "./atc --model=./alexnet.prototxt --weight=./alexnet.caffemodel \n"
        "--framework=0 --output=./domi \n"
        "generate offline model for single op example:\n"
        "./atc --singleop=./op_list.json --output=./op_model \n"
        "===== Basic Functionality =====\n"
        "[General]\n"
        "  --h/help            Show this help message\n"
        "  --mode              Run mode. 0(default): generate offline model; 1: convert model to JSON format; "
        "3: only pre-check; 5: convert ge dump txt file to JSON format; 6: display model info\n"
        "\n[Input]\n"
        "  --model             Model file\n"
        "  --weight            Weight file. Required when framework is Caffe\n"
        "  --om                The model file to be converted to json\n"
        "  --framework         Framework type. 0:Caffe; 1:LuoJiaNet; 3:Tensorflow; 5:Onnx\n"
        "  --input_format      Format of input data. E.g.: \"NCHW\"\n"
        "  --input_shape       Shape of input data. Separate multiple nodes with semicolons (;). "
        "Use double quotation marks (\") to enclose each argument.\n"
        "                      E.g.: \"input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2\"\n"
        "  --input_shape_range Shape range of input data. Separate multiple nodes with semicolons (;)."
        "Use double quotation marks (\") to enclose each argument.\n"
        "                      E.g.: \"input_name1:[n1~n2,c1,h1,w1];input_name2:[n2,c2~c3,h2,w2]\"\n"
        "  --dynamic_batch_size Set dynamic batch size. E.g.: \"batchsize1,batchsize2,batchsize3\"\n"
        "  --dynamic_image_size Set dynamic image size. Separate multiple nodes with semicolons (;). "
        "Use double quotation marks (\") to enclose each argument.\n"
        "                       E.g.: \"imagesize1_height,imagesize1_width;imagesize2_height,imagesize2_width\"\n"
        "  --dynamic_dims      Set dynamic dims. Separate multiple nodes with semicolons (;). "
        "Use double quotation marks (\") to enclose each argument.\n"
        "                      E.g.: \"dims1_n1,dims1_n2;dims2_n1,dims2_n2\"\n"
        "  --singleop          Single op definition file. atc will generate offline "
        "model(s) for single op if --singleop is set.\n"
        "\n[Output]\n"
        "  --output            Output file path&name(needn't suffix, will add .om automatically). \n"
        "                      If --singleop is set, this arg specifies the directory to "
        "which the single op offline model will be generated\n"
        "  --output_type       Set net output type. Support FP32, FP16, UINT8. "
        "E.g.: FP16, indicates that all out nodes are set to FP16.\n"
        "                      \"node1:0:FP16;node2:1:FP32\", indicates setting the datatype of multiple out nodes.\n"
        "  --check_report      The pre-checking report file. Default value is: \"check_result.json\"\n"
        "  --json              The output json file path&name which is converted from a model\n"
        "\n[Target]\n"
        "  --soc_version       The soc version.\n"
        "  --core_type         Set core type AiCore or VectorCore. VectorCore: use vector core. "
        "Default value is: AiCore\n"
        "  --aicore_num        Set aicore num\n"
        "===== Advanced Functionality =====\n"
        "[Feature]\n"
        "  --out_nodes         Output nodes designated by users. Separate multiple nodes with semicolons (;)."
        "Use double quotation marks (\") to enclose each argument.\n"
        "                      E.g.: \"node_name1:0;node_name1:1;node_name2:0\"\n"
        "  --input_fp16_nodes  Input node datatype is fp16. Separate multiple nodes with semicolons (;). "
        "Use double quotation marks (\") to enclose each argument. "
        "E.g.: \"node_name1;node_name2\"\n"
        "  --insert_op_conf    Config file to insert new op\n"
        "  --op_name_map       Custom op name mapping file\n"
        "                      Note: A semicolon(;) cannot be included in each "
        "path, otherwise the resolved path will not match the expected one.\n"
        "  --is_input_adjust_hw_layout    Intput node datatype is fp16 and format is "
        "NC1HWC0, used with input_fp16_nodes. E.g.: \"true,true,false,true\"\n"
        "  --is_output_adjust_hw_layout   Net output node datatype is fp16 and format is "
        "NC1HWC0, used with out_nodes. E.g.: \"true,true,false,true\"\n"
        "\n[Model Tuning]\n"
        "  --disable_reuse_memory    The switch of reuse memory. Default value is : 0. "
        "0 means reuse memory, 1 means do not reuse memory.\n"
        "  --fusion_switch_file      Set fusion switch file path\n"
        "  --enable_scope_fusion_passes    validate the non-general scope fusion passes, "
        "multiple names can be set and separated by ','. E.g.: ScopePass1,ScopePass2,...\n"
        "  --enable_single_stream    Enable single stream. true: enable; false(default): disable\n"
        "  --enable_small_channel    Set enable small channel. 0(default): disable; 1: enable\n"
        "  --enable_compress_weight  Enable compress weight. true: enable; false(default): disable\n"
        "  --compress_weight_conf    Config file to compress weight\n"
        "  --buffer_optimize         Set buffer optimize. Support \"l2_optimize\" (default), "
        "\"l1_optimize\", \"off_optimize\"\n"
	"  --mdl_bank_path           Set the path of the custom repository generated after model tuning.\n"
        "\n[Operator Tuning]\n"
        "  --op_precision_mode     Set the path of operator precision mode configuration file (.ini)\n"
        "  --precision_mode        precision mode, support force_fp16(default), force_fp32, allow_mix_precision, "
        "allow_fp32_to_fp16, must_keep_origin_dtype.\n"
        "  --modify_mixlist        Set the path of operator mixed precision configuration file.\n"
        "  --keep_dtype            Retains the precision of certain operators in inference "
        "scenarios by using a configuration file.\n"
        "  --auto_tune_mode        Set tune mode. E.g.: \"GA,RL\", support configure multiple, spit by ,\n"
        "  --op_bank_path          Set the path of the custom repository generated after operator tuning with Auto Tune.\n"
	"  --op_select_implmode    Set op select implmode. Support high_precision, high_performance, "
        "high_precision_for_all, high_performance_for_all. default: high_performance\n"
        "  --optypelist_for_implmode    Appoint which op to select implmode, cooperated with op_select_implmode.\n"
        "                               Separate multiple nodes with commas (,). Use double quotation marks (\") "
        "to enclose each argument. E.g.: \"node_name1,node_name2\"\n"
        "  --op_debug_level        Debug enable for TBE operator building.\n"
        "                          0 (default): Disable debug; 1: Enable TBE pipe_all, "
        "and generate the operator CCE file and Python-CCE mapping file (.json);\n"
        "                          2: Enable TBE pipe_all, generate the operator CCE file and Python-CCE mapping file "
        "(.json), and enable the CCE compiler -O0-g.\n"
        "                          3: Disable debug, and keep generating kernel file (.o and .json)\n"
        "                          4: Disable debug, keep generation kernel file (.o and .json) and generate the "
        "operator CCE file (.cce) and the UB fusion computing description file (.json)"
        "\n[Debug]\n"
        "  --save_original_model   Control whether to output original model. E.g.: true: output original model\n"
        "  --log                   Generate log with level. Support debug, info, warning, error, null\n"
        "  --dump_mode             The switch of dump json with shape, to be used with mode 1. "
        "0(default): disable; 1: enable.\n"
        "  --debug_dir                Set the save path of operator compilation intermediate files.\n"
        "Default value: ./kernel_meta\n"
        "  --op_compiler_cache_dir    Set the save path of operator compilation cache files.\n"
        "Default value: $HOME/atc_data\n"
        "  --op_compiler_cache_mode   Set the operator compilation cache mode."
        "Options are disable(default), enable and force(force to refresh the cache)\n"
        "  --display_model_info     enable for display model info; 0(default): close display, 1: open display.");

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    // Using gflags to analyze input parameters
    GflagsUtils::ChangeHelpFlags(FLAGS_h);
    gflags::HandleCommandLineHelpFlags();
  }

  static Status CheckDumpInfershapeJsonFlags() {
    Status ret = CheckFrameWorkValid(FLAGS_framework, FLAGS_weight);
    GE_CHK_BOOL_EXEC(ret == domi::SUCCESS, return domi::FAILED,
                       "[Check][Param:FrameWork]%d value is invalid.", FLAGS_framework);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_weight != "" && !ge::CheckInputPathValid(FLAGS_weight, "--weight"),
        return domi::FAILED, "[Check][Param:weight]value:%s: is invalid, path can not reach.",
        FLAGS_weight.c_str());
    return domi::SUCCESS;
  }

  static Status CheckFlags() {
    Status ret = ge::SUCCESS;
    // No model file information passed in
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_model == "",
        ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {"model"});
        ret = ge::FAILED, "[Check][Param]Input parameter[--model]'s value is empty!");

    // check param disable_reuse_memory
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ge::CheckDisableReuseMemoryParamValid(to_string(FLAGS_disable_reuse_memory)) != ge::SUCCESS,
        ret = ge::FAILED, "[Check][DisableReuseMemory]failed!");

    // check optypelist_for_implmode and op_select_implmode
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ge::CheckImplmodeParamValid(FLAGS_optypelist_for_implmode,
                                    FLAGS_op_select_implmode) != ge::SUCCESS,
        ret = ge::FAILED, "[Check][ImplMode]check optypelist_for_implmode and op_select_implmode failed!");

    if (!FLAGS_op_precision_mode.empty() && !ge::CheckInputPathValid(FLAGS_op_precision_mode, "--op_precision_mode")) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"op_precision_mode", FLAGS_op_precision_mode.c_str(),
                                                      "path is not found"});
      GELOGE(ge::FAILED, "[Check][op_precision_mode] %s not found", FLAGS_op_precision_mode.c_str());
      ret = ge::FAILED;
    }

    if (ge::CheckModifyMixlistParamValid(FLAGS_precision_mode, FLAGS_modify_mixlist) != ge::SUCCESS) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"modify_mixlist", FLAGS_modify_mixlist.c_str(),
                                                      ge::kModifyMixlistError});
      ret = ge::FAILED;
    }

    // No output file information passed in
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_mode == GEN_OM_MODEL && FLAGS_output == "",
        ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {"output"});
        ret = ge::FAILED, "[Check][Param]Input parameter[--output]'s value is empty!");

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        CheckFrameWorkValid(FLAGS_framework, FLAGS_weight) != ge::SUCCESS,
        ret = ge::FAILED,
        "[Check][FrameWork] failed for input --FLAGS_framework and --FLAGS_weight invalid.");

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ge::CheckDynamicInputParamValid(FLAGS_dynamic_batch_size, FLAGS_dynamic_image_size,
                                        FLAGS_dynamic_dims, FLAGS_input_shape, FLAGS_input_shape_range,
                                        FLAGS_input_format, is_dynamic_input) != ge::SUCCESS,
        ret = ge::FAILED, "[Check][DynamicInput]dynamic size(batch size, image size or dims) invalid!");

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        !FLAGS_insert_op_conf.empty() && !FLAGS_dynamic_dims.empty(),
        ErrorManager::GetInstance().ATCReportErrMessage("E10001",
                                                        {"parameter", "value", "reason"},
                                                        {"--insert_op_conf", FLAGS_insert_op_conf,
                                                         "dynamic dims function does not support aipp"});
        ret = ge::FAILED, "[Check][Param]dynamic dims function does not support aipp");

    /**
     * Check the validity of the I / O file path
     */
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_model != "" && !ge::CheckInputPathValid(FLAGS_model, "--model"), ret = ge::FAILED,
        "[Check][InputPath]model file %s not found!!", FLAGS_model.c_str());

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_weight != "" && !ge::CheckInputPathValid(FLAGS_weight, "--weight"),
        ret = ge::FAILED, "[Check][InputPath]weight file %s not found!!",
        FLAGS_weight.c_str());

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_cal_conf != "" && !ge::CheckInputPathValid(FLAGS_cal_conf, "--cal_conf"),
        ret = ge::FAILED, "[Check][InputPath]calibration config file %s not found!!",
        FLAGS_cal_conf.c_str());

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_op_name_map != "" && !ge::CheckInputPathValid(FLAGS_op_name_map, "--op_name_map"),
        ret = ge::FAILED, "[Check][InputPath]op config file %s not found!!",
        FLAGS_op_name_map.c_str());

    GE_CHK_BOOL_EXEC(ge::CheckInsertOpConfParamValid(std::string(FLAGS_insert_op_conf)) == ge::SUCCESS,
                     ret = ge::FAILED, "[Check][InsertOpConf]failed!");

    GE_CHK_BOOL_EXEC(ge::CheckCompressWeightParamValid(
        FLAGS_enable_compress_weight, FLAGS_compress_weight_conf) == ge::SUCCESS,
        ret = ge::FAILED, "[Check][CompressWeight]failed!");

    GE_CHK_BOOL_EXEC(ge::CheckKeepTypeParamValid(FLAGS_keep_dtype) == ge::SUCCESS,
        ret = ge::FAILED, "[Check][KeepType]failed!");

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        !ge::CheckOutputPathValid(FLAGS_check_report, "--check_report"), ret = ge::FAILED,
        "[Check][OutputPath]]check_report file %s not found!!", FLAGS_check_report.c_str());

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_mode == GEN_OM_MODEL && FLAGS_output != "" &&
        (!ge::CheckOutputPathValid(FLAGS_output, "--output") || !CheckPathWithName(FLAGS_output)),
        ret = ge::FAILED, "[Check][OutputPath]output path %s is not valid!!", FLAGS_output.c_str());

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_save_original_model != "" &&
        FLAGS_save_original_model != "true" &&
        FLAGS_save_original_model != "false",
        ErrorManager::GetInstance().ATCReportErrMessage(
            "E10005", {"parameter", "value"}, {"save_original_model", FLAGS_save_original_model});
        ret = ge::FAILED,
        "[Check][Parameter]Input parameter[--save_original_model]'s value[%s] must be true or false.",
        FLAGS_save_original_model.c_str());
    GE_CHK_BOOL_EXEC(ge::CheckBufferOptimizeParamValid(FLAGS_buffer_optimize) == ge::SUCCESS,
        ret = ge::FAILED, "[Check][BufferOptimize]check output type failed!");

    GE_CHK_BOOL_EXEC(
        ge::CheckEnableSingleStreamParamValid(std::string(FLAGS_enable_single_stream)) == ge::SUCCESS,
        ret = ge::FAILED, "[Check][EnableSingleStream]failed!");

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((FLAGS_display_model_info != "0") && (FLAGS_display_model_info != "1"),
      REPORT_INPUT_ERROR("E10006", std::vector<std::string>({"parameter", "value"}),
                         std::vector<std::string>({"display_model_info", FLAGS_display_model_info}));
      ret = ge::FAILED, "[Check][Parameter]Input parameter[--display_model_info]'s value must be 1 or 0.");

    return ret;
  }

  /**
   * Verifying the parameters of converting model to JSON
   * 1. Fmk_model
   * 2. out_json
   **/
  static Status CheckConverJsonParamFlags() {
    Status ret = ge::SUCCESS;

    // No model path passed in
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(FLAGS_om == "",
        ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {"om"});
        ret = ge::FAILED,
        "[Check][Parameter]Input parameter[--om]'s value is empty!!");

    // JSON path not passed in
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(FLAGS_json == "",
        ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {"json"});
        ret = ge::FAILED,
        "[Check][Parameter]Input parameter[--json]'s value is empty!!");

    // Check if the model path is valid
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_om != "" && !ge::CheckInputPathValid(FLAGS_om, "--om"),
        ret = ge::FAILED,
        "[Check][InputPath]model file path is invalid: %s.", FLAGS_om.c_str());

    // Check whether the JSON path is valid
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        FLAGS_json != "" && !ge::CheckOutputPathValid(FLAGS_json, "--json"),
        ret = ge::FAILED,
        "[Check][OutputPath]json file path is invalid: %s.", FLAGS_json.c_str());

    return ret;
  }

  /**
   * Check command line parameters for explicit settings
   * true: Explicit setup
   * false: Not set up
   * */
  static bool CheckFlagSet(string flag) {
    gflags::CommandLineFlagInfo info;
    return !(gflags::GetCommandLineFlagInfo(flag.c_str(), &info) && info.is_default);
  }

 private:
  static bool CheckEncryptModeValid(const int encrypt_mode) {
#if !defined(__ANDROID__) && !defined(ANDROID)
    if (encrypt_mode != 0 && encrypt_mode != -1) {
      DOMI_LOGE("encrypt mode must be 0 or -1");
      return false;
    }
#else
    if (encrypt_mode != -1) {
      DOMI_LOGE("encrypt mode must be -1");
      return false;
    }
#endif

    return true;
  }

  static Status CheckFrameWorkValid(int framework, const std::string weight_file) {
    if (framework != (int32_t)domi::CAFFE && framework != (int32_t)domi::TENSORFLOW &&
        framework != (int32_t)domi::LUOJIANET  && framework != (int32_t)domi::ONNX) {
      // No framework information was passed in or the entered framework is illegal
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E10007", {"parameter", "support"},
          {"framework", "0(Caffe) or 1(LuoJiaNet) or 3(TensorFlow) or 5(Onnx)"});
      DOMI_LOGE("Input parameter[--framework] is mandatory and it's value must be: "
                "0(Caffe) or 1(LuoJiaNet) or 3(TensorFlow) or 5(Onnx).");
      return domi::PARAM_INVALID;
    }

    if ((framework == (int32_t)domi::CAFFE) && (weight_file == "")) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10008", {"parameter"}, {"weight"});
      DOMI_LOGE("Input parameter[--weight]'s value is empty when framework is 0(CAFFE)!");
      return domi::PARAM_INVALID;
    }

    if ((framework == (int32_t)domi::TENSORFLOW) && (weight_file != "")) {
      GELOGW("Parameter weight is ignored for TensorFlow.");
    }

    if ((framework == (int32_t)domi::ONNX) && (weight_file != "")) {
      GELOGW("Parameter weight is ignored for Onnx.");
    }
    return domi::SUCCESS;
  }

  static bool CheckPathWithName(const std::string &fileName) {
    // Determine file path length
    if (fileName.size() > static_cast<int>(PATH_MAX)) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E10021", {"parameter", "size"}, {"output", std::to_string(PATH_MAX)});
      GELOGE(ge::FAILED,
          "[Check][Path]Input parameter[--output]'s path is too long, it must be less than %d", PATH_MAX);
      return false;
    }

    // Find the last separator
    int slashPosition = fileName.size() - 1;
    for (; slashPosition >= 0; slashPosition--) {
      if (fileName[slashPosition] == '\\' || fileName[slashPosition] == '/') {
        break;
      }
    }

    // Failure if no filename follows the path
    if (slashPosition == static_cast<int>(fileName.size() - 1)) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10022", {"parameter", "filename"}, {"output", fileName});
      DOMI_LOGE("Input parameter[--output]'s path[%s] not include file name", fileName.c_str());
      return false;
    }

    return true;
  }
};

void SetDynamicInputSizeOptions() {
  if (!FLAGS_dynamic_batch_size.empty()) {
    domi::GetContext().dynamic_batch_size = FLAGS_dynamic_batch_size;
  }
  if (!FLAGS_dynamic_image_size.empty()) {
    domi::GetContext().dynamic_image_size = FLAGS_dynamic_image_size;
  }
  if (!FLAGS_dynamic_dims.empty()) {
    domi::GetContext().dynamic_dims = FLAGS_dynamic_dims;
  }
}

/// Validate the non-general scope fusion pass.
/// The parameter is set to the name of the fusion rule.
/// Multiple names can be set and separated by ",".
void SetEnableScopeFusionPasses(const std::string pass_names) {
  ge::GetParserContext().enable_scope_fusion_passes = pass_names;
}

static bool CheckInputFormat() {
  if (FLAGS_input_format.empty()) {
    // Set default format
    if (FLAGS_framework == static_cast<int32_t>(domi::TENSORFLOW)) {
      FLAGS_input_format = "NHWC";
    } else {
      FLAGS_input_format = "NCHW";
    }
    return true;
  } else if ((FLAGS_framework == static_cast<int32_t>(domi::CAFFE))) { // caffe
    if (ge::caffe_support_input_format.find(FLAGS_input_format) != ge::caffe_support_input_format.end()) {
      return true;
    }
    // only support NCHW ND
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"}, {"--input_format", FLAGS_input_format, kCaffeFormatSupport});
    GELOGE(ge::FAILED, "[Check][InputFormat]Invalid value for --input_format[%s], %s.",
        FLAGS_input_format.c_str(), kCaffeFormatSupport);
    return false;
  } else if ((FLAGS_framework == static_cast<int32_t>(domi::TENSORFLOW))) { // tf
    if (ge::tf_support_input_format.find(FLAGS_input_format) != ge::tf_support_input_format.end()) {
      return true;
    }
    // only support NCHW NHWC ND NCDHW NDHWC
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"}, {"--input_format", FLAGS_input_format, kTFFormatSupport});
    GELOGE(ge::FAILED, "[Check][InputFormat]Invalid value for --input_format[%s], %s.",
        FLAGS_input_format.c_str(), kTFFormatSupport);
    return false;
  } else if (FLAGS_framework == static_cast<int32_t>(domi::ONNX)) {
    if (ge::onnx_support_input_format.find(FLAGS_input_format) != ge::onnx_support_input_format.end()) {
      return true;
    }
    // only support NCHW ND
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"}, {"--input_format", FLAGS_input_format, kONNXFormatSupport});
    GELOGE(ge::FAILED, "[Check][InputFormat]Invalid value for --input_format[%s], %s.",
        FLAGS_input_format.c_str(), kONNXFormatSupport);
    return false;
  }
  return true;
}

#if !defined(__ANDROID__) && !defined(ANDROID)
static void GetCustomOpPath(std::string &customop_path) {
  GELOGI("Enter get custom op path schedule");
  std::string fmk_type = ge::TypeUtils::FmkTypeToSerialString(static_cast<domi::FrameworkType>(FLAGS_framework));
  GELOGI("Framework type is %s.", fmk_type.c_str());

  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    std::string path = path_env;
    customop_path = (path + "/framework/custom" + "/:") + (path + "/framework/built-in/" + fmk_type);
    GELOGI("Get custom so path from env : %s", path_env);
    return;
  }
  std::string path_base = ge::GELib::GetPath();
  GELOGI("path_base is %s", path_base.c_str());
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  customop_path = (path_base + "ops/framework/custom" + "/:") + (path_base + "ops/framework/built-in/" + fmk_type);
  return;
}

void GetPluginSoFileList(const string &path, vector<string> &fileList, string &caffe_parser_path) {
  // Support to split multiple so directories by ":"
  GELOGI("path is %s", path.c_str());
  vector<string> v_path = ge::StringUtils::Split(path, ':');
  for (size_t i = 0; i < v_path.size(); ++i) {
    ge::FindParserSo(v_path[i], fileList, caffe_parser_path);
    GELOGI("CustomOpLib full name = %s", v_path[i].c_str());
  }
}

void LoadModelParserLib(std::string caffe_parser_path) {
  if (FLAGS_framework == static_cast<int32_t>(domi::TENSORFLOW)) {
    void *tf_handle = dlopen("libfmk_parser.so", RTLD_NOW | RTLD_GLOBAL);
    if (tf_handle == nullptr) {
      GELOGW("dlopen fmk library [libfmk_parser.so] failed.");
      return;
    }
    GELOGI("plugin load libfmk_parser.so success.");
  } else if (FLAGS_framework == static_cast<int32_t>(domi::CAFFE)) {
    // What we are dealing with here is that the user modifies the caffe.proto scenario.
    // If no lib_Caffe_Parser.so is found under the plugin path, use the default lib_Caffe_Parser.so path.
    caffe_parser_path = caffe_parser_path.empty() ? "lib_caffe_parser.so" : caffe_parser_path;

    void *handle = dlopen(caffe_parser_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      GELOGW("dlopen failed, plugin name:%s. Message(%s).", caffe_parser_path.c_str(), dlerror());
      return;
    }
    GELOGI("plugin load %s success.", caffe_parser_path.c_str());
    // According to the dependency, the Caffe parsing module of the framework is loaded here( libfmk_parser.so).
    // (depend on the lib_caffe_parser.so)
    void *fmk_handle = dlopen("libfmk_parser.so", RTLD_NOW | RTLD_GLOBAL);
    if (fmk_handle == nullptr) {
      GELOGW("dlopen fmk library [libfmk_parser.so] failed.");
      if (dlclose(handle) != 0) {
        GELOGW("dlclose lib_caffe_parser.so failed.");
      }
      return;
    }
    GELOGI("plugin load libfmk_parser.so success.");
  } else if (FLAGS_framework == static_cast<int32_t>(domi::ONNX)) {
    void *handle = dlopen("libfmk_onnx_parser.so", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      GELOGW("dlopen fmk library [libfmk_onnx_parser.so] failed.");
      return;
    }
    GELOGI("plugin load libfmk_onnx_parser.so success.");
  } else {
    GELOGW("Framework:%s is not support.",
           ge::TypeUtils::FmkTypeToSerialString(static_cast<domi::FrameworkType>(FLAGS_framework)).c_str());
    return;
  }
  return;
}

void LoadCustomOpLib(bool need_load_ops_plugin) {
  std::string plugin_path;
  GetCustomOpPath(plugin_path);

  vector<string> fileList;
  string caffe_parser_path = "";

  // whether there are files in the plugin so path
  GetPluginSoFileList(plugin_path, fileList, caffe_parser_path);

  // no file
  if (fileList.empty() && caffe_parser_path.empty()) {
    GELOGW("can not find any plugin file in plugin_path: %s", plugin_path.c_str());
  }

  LoadModelParserLib(caffe_parser_path);
  if (!need_load_ops_plugin) {
    GELOGI("No need to load ops plugin so.");
    return;
  }
  OpRegistry::Instance()->registrationDatas.clear();
  // load other so files except lib_caffe_parser.so in the plugin so path
  for (auto elem : fileList) {
    ge::StringUtils::Trim(elem);

    void *handle = dlopen(elem.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      GELOGW("dlopen failed, plugin name:%s. Message(%s).", elem.c_str(), dlerror());
    } else {
      GELOGI("plugin load %s success.", elem.c_str());
    }
  }

  std::vector<OpRegistrationData> registrationDatas = OpRegistry::Instance()->registrationDatas;
  for (OpRegistrationData reg_data : registrationDatas) {
    if (reg_data.GetFrameworkType() == static_cast<domi::FrameworkType>(FLAGS_framework)) {
      (void)ge::OpRegistrationTbe::Instance()->Finalize(reg_data);
      (void)OpRegistry::Instance()->Register(reg_data);
    }
  }
}

void SaveCustomCaffeProtoPath() {
  GELOGI("Enter save custom caffe proto path.");

  std::string path_base = ge::GELib::GetPath();
  GELOGI("path_base is %s", path_base.c_str());
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  ge::GetParserContext().caffe_proto_path = path_base + "include/proto/";

  string customop_path;
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    std::string path = path_env;
    customop_path = path + "/framework/custom/caffe/";
    GELOGI("Get custom proto path from env : %s", path_env);
    ge::GetParserContext().custom_proto_path = customop_path;
    return;
  }
  customop_path = path_base + "ops/framework/custom/caffe/";
  ge::GetParserContext().custom_proto_path = customop_path;
  return;
}

#endif

Status CreateInputsForInference(const ge::Graph &graph, vector<ge::GeTensor> &inputs) {
  auto compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  int64_t index = 0;
  for (ge::NodePtr &input_node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(input_node);
    ge::OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (op->GetType() == ge::DATA) {
      if (!op->HasAttr(ge::ATTR_NAME_INDEX)) {
        (void)ge::AttrUtils::SetInt(op, ge::ATTR_NAME_INDEX, index);
        GELOGD("Set attr index:%ld for data op:%s", index, op->GetName().c_str());
      }
      index++;
      GELOGI("Data op inputDesc size is: %zu", op->GetAllInputsDesc().size());
      ge::GeTensorDesc tensor = op->GetInputDesc(0);
      string data_op_name = op->GetName();
      GELOGI("Data op name is: %s", data_op_name.c_str());
      ge::GeShape data_shape;
      auto iter = domi::GetContext().input_dims.find(data_op_name);
      if (iter != domi::GetContext().input_dims.end()) {
        data_shape = ge::GeShape(iter->second);
        GELOGI("Data op get shape from Context.");
      } else {
        data_shape = tensor.GetShape();
        GELOGI("Data op get shape from InputDesc in geir graph.");
      }

      ge::DataType data_type = tensor.GetDataType();
      string data_type_str = ge::TypeUtils::DataTypeToSerialString(data_type);
      GELOGI("Data op get data type:%s from InputDesc in geir graph.", data_type_str.c_str());

      ge::GeTensor input_tensor;
      ge::GeTensorDesc desc(data_shape, ge::Format(domi::GetContext().format), data_type);
      input_tensor.SetTensorDesc(desc);
      inputs.push_back(input_tensor);
    }
  }
  GELOGI("Build ME model, inputs size is: %zu", inputs.size());
  return ge::SUCCESS;
}

domi::Status GenerateInfershapeJson() {
  if (!CheckInputFormat()) {
    GELOGE(ge::FAILED, "[Check][InputFormat] failed.");
    return domi::FAILED;
  }
  Status ret = GFlagUtils::CheckDumpInfershapeJsonFlags();
  GE_CHK_BOOL_EXEC(ret == domi::SUCCESS, return domi::FAILED, "[Check][DumpInfershapeJsonFlags] failed!");

  ge::GeGenerator ge_generator;
  std::map<string, string> options;
  ge::Status geRet = ge_generator.Initialize(options, domi::GetContext());
  if (geRet != ge::SUCCESS) {
    DOMI_LOGE("GeGenerator initialize failed!");
    return domi::FAILED;
  }

  ge::Graph graph;
  std::map<string, string> atc_params;
  atc_params.insert(std::pair<string, string>("input_format", FLAGS_input_format));
  atc_params.insert(std::pair<string, string>("check_report", FLAGS_check_report));
  ret = ParseGraph(graph, atc_params, FLAGS_om.c_str(), FLAGS_weight.c_str(), (domi::FrameworkType) FLAGS_framework,
                   "", FLAGS_target.c_str(), (ge::RunMode) FLAGS_mode, false);
  if (ret != ge::SUCCESS) {
    DOMI_LOGE("ATC Parse graph domi::FAILED");
    (void)ge_generator.Finalize();
    return domi::FAILED;
  }

  geRet = ge_generator.GenerateInfershapeGraph(graph);
  if (geRet != ge::SUCCESS) {
    DOMI_LOGE("ATC GenerateInfershapeJson failed");
    (void)ge_generator.Finalize();
    return domi::FAILED;
  }
  if (DumpInfershapeJson(graph, FLAGS_json.c_str()) != SUCCESS) {
    DOMI_LOGE("ATC DumpInfershapeJson failed");
    (void)ge_generator.Finalize();
    return domi::FAILED;
  }
  (void)ge_generator.Finalize();
  return ge::SUCCESS;
}

static Status ConvertModelToJson(int fwk_type, const string &model_file, const string &json_file) {
  Status ret = ge::SUCCESS;
  if (fwk_type == -1) {
    ret = ge::ConvertOm(model_file.c_str(), json_file.c_str(), true);
    return ret;
  }

  if ((fwk_type != domi::TENSORFLOW) && (fwk_type != domi::CAFFE) && (fwk_type != domi::ONNX)) {
    ErrorManager::GetInstance().ATCReportErrMessage(
      "E10001", {"parameter", "value", "reason"},
      {"--framework", std::to_string(fwk_type), kModelToJsonSupport});
    GELOGE(ge::FAILED, "[Convert][ModelToJson]Invalid value for --framework[%d], %s.",
        fwk_type, kModelToJsonSupport);
    ret = ge::FAILED;
  }

  if (FLAGS_dump_mode != "0" && FLAGS_dump_mode != "1") {
    REPORT_INPUT_ERROR("E10006", std::vector<std::string>({"parameter", "value"}),
                       std::vector<std::string>({"dump_mode", FLAGS_dump_mode}));
    GELOGE(ge::FAILED, "[Convert][ModelToJson] Input parameter[--dump_mode]'s value must be 1 or 0.");
    ret = ge::FAILED;
  }

  if (ret != ge::SUCCESS) return ret;

  // Need to save caffe.proto path
  SaveCustomCaffeProtoPath();

  if (FLAGS_dump_mode == "0") {
    // Caffe or tf model to json depend on lib_caffe_parser.so or libfmk_parser.so.
    LoadCustomOpLib(false);
    ret = ge::ConvertFwkModelToJson((domi::FrameworkType)fwk_type, model_file.c_str(), json_file.c_str());
  } else if (FLAGS_dump_mode == "1") {
    // Caffe or tf model to json depend on lib_caffe_parser.so or libfmk_parser.so and ops plugin so.
    LoadCustomOpLib(true);
    ret = GenerateInfershapeJson();
  }

  return ret;
}

static Status SetAttrOptions(ge::Graph &graph) {
  if (!FLAGS_keep_dtype.empty()) {
    if (ge::aclgrphSetOpAttr(graph, ge::ATTR_TYPE_KEEP_DTYPE, FLAGS_keep_dtype.c_str()) != ge::GRAPH_SUCCESS) {
      return ge::FAILED;
    }
  }
  if (!FLAGS_compress_weight_conf.empty()) {
    if (ge::aclgrphSetOpAttr(graph, ge::ATTR_TYPE_WEIGHT_COMPRESS, FLAGS_compress_weight_conf.c_str())
        != ge::GRAPH_SUCCESS) {
      return ge::FAILED;
    }
  }

  return ge::SUCCESS;
}

domi::Status GenerateModel(std::map<string, string> &options, std::string output) {
  ge::GeGenerator ge_generator;
  ge::Status geRet = ge::SUCCESS;
  std::shared_ptr<ge::GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    geRet = ge::GELib::Initialize(options);
    if (geRet != ge::SUCCESS) {
      DOMI_LOGE("GE initialize failed!");
      return domi::FAILED;
    }
  }
  geRet = ge_generator.Initialize(options, domi::GetContext());
  if (geRet != ge::SUCCESS) {
    DOMI_LOGE("GeGenerator initialize failed!");
    (void)ge::GELib::GetInstance()->Finalize();
    return domi::FAILED;
  }

  ge::Graph graph;
  std::vector<ge::GeTensor> inputs;
  if (FLAGS_framework == domi::LUOJIANET) {
    ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
    // load model from file
    ge::Model load_model = ge::Model("loadmodel", "version2");
    auto ret1 = load_model.LoadFromFile(FLAGS_model);
    if (ret1 != ge::GRAPH_SUCCESS) {
      REPORT_INPUT_ERROR("E10041", std::vector<std::string>({"parameter"}), std::vector<std::string>({FLAGS_model}));
      DOMI_LOGE("Load model from %s failed, please check model file or "
          "input parameter[--framework] is correct", FLAGS_model.c_str());
      (void)ge_generator.Finalize();
      (void)ge::GELib::GetInstance()->Finalize();
      return domi::FAILED;
    }

    graph = load_model.GetGraph();

    GE_CHK_STATUS_EXEC(ge::InitDomiOmgContext(FLAGS_input_shape, FLAGS_input_format, "", is_dynamic_input),
                       GELOGE(ge::FAILED, "[Init][DomiOmgContext]ATC Generate call InitDomiOmgContext ret fail");
                       (void)ge_generator.Finalize(); (void)ge::GELib::GetInstance()->Finalize(); return domi::FAILED);

    Status ret = CreateInputsForInference(graph, inputs);
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED, "[Create][InputsForInference] failed.");
      REPORT_CALL_ERROR("E19999", "CreateInputsForInference failed for input --graph and --inputs.");
      (void)ge_generator.Finalize();
      (void)ge::GELib::GetInstance()->Finalize();
      return domi::FAILED;
    }

  } else {
    std::map<string, string> atc_params;
    atc_params.insert(std::pair<string, string>("input_shape", FLAGS_input_shape));
    atc_params.insert(std::pair<string, string>(ge::INPUT_SHAPE_RANGE, FLAGS_input_shape_range));
    atc_params.insert(std::pair<string, string>("out_nodes", FLAGS_out_nodes));
    atc_params.insert(std::pair<string, string>("input_format", FLAGS_input_format));
    atc_params.insert(std::pair<string, string>("check_report", FLAGS_check_report));
    atc_params.insert(std::pair<string, string>("input_fp16_nodes", FLAGS_input_fp16_nodes));
    atc_params.insert(std::pair<string, string>("is_input_adjust_hw_layout", FLAGS_is_input_adjust_hw_layout));
    atc_params.insert(std::pair<string, string>("is_output_adjust_hw_layout", FLAGS_is_output_adjust_hw_layout));
    atc_params.insert(std::pair<string, string>(string(ge::OUTPUT_DATATYPE), FLAGS_output_type));
    atc_params.insert(std::pair<string, string>("output", output));

    ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
    Status ret =
        ParseGraph(graph, atc_params, FLAGS_model.c_str(), FLAGS_weight.c_str(), (domi::FrameworkType)FLAGS_framework,
                   FLAGS_op_name_map.c_str(), FLAGS_target.c_str(), (ge::RunMode)FLAGS_mode, is_dynamic_input);

    ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
    // in ONLY_PRE_CHECK mode, pre-checking report has already saved in ParseGraph
    if (FLAGS_mode == ge::ONLY_PRE_CHECK) {
      (void)ge_generator.Finalize();
      (void)ge::GELib::GetInstance()->Finalize();
      if (ret != ge::SUCCESS) {
        DOMI_LOGE("ATC precheck fail.");
        return domi::FAILED;
      }
      return domi::SUCCESS;
    }

    if (ret != ge::SUCCESS) {
      DOMI_LOGE("ATC Parse graph domi::FAILED");
      DOMI_LOGE("ATC Generate execute failed");  // Duplicate log. (for test case
      (void)ge_generator.Finalize();
      (void)ge::GELib::GetInstance()->Finalize();
      return domi::FAILED;
    }
    if (ge::SetOutputNodeInfo(graph, FLAGS_output_type, "") != domi::SUCCESS) {
      DOMI_LOGE("Set output node info fail.");
      (void)ge_generator.Finalize();
      (void)ge::GELib::GetInstance()->Finalize();
      return domi::FAILED;
    }
  }

  if (SetAttrOptions(graph) != ge::SUCCESS) {
    (void)ge_generator.Finalize();
    (void)ge::GELib::GetInstance()->Finalize();
    return domi::FAILED;
  }

  geRet = ge_generator.GenerateOfflineModel(graph, output, inputs);
  if (geRet != ge::SUCCESS) {
    DOMI_LOGE("GE GenerateOfflineModel execute failed");
    DOMI_LOGE("ATC Generate execute failed");  // Duplicate log. (for test case
    // checking error log)
    (void)ge_generator.Finalize();
    (void)ge::GELib::GetInstance()->Finalize();
    return domi::FAILED;
  }
  (void)ge_generator.Finalize();
  (void)ge::GELib::GetInstance()->Finalize();
  return ge::SUCCESS;
}

static void SetEnvForSingleOp(std::map<string, string> &options) {
  string flag_on = "1";
  string flag_off = "0";
  options.emplace(ge::GE_FE_FLAG, flag_on);
  options.emplace(ge::STREAM_NUM, "1");  // single op only use one stream
  options.emplace(ge::RUN_FLAG, flag_off);
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, flag_off);
  options.emplace(ge::SINGLE_OP_FLAG, flag_on);
  options.emplace(ge::OP_PRECISION_MODE, FLAGS_op_precision_mode);
  options.emplace(ge::PRECISION_MODE, FLAGS_precision_mode);
  options.emplace(ge::SOC_VERSION, FLAGS_soc_version);
  options.emplace(ge::CORE_TYPE, FLAGS_core_type);
  options.emplace(ge::AICORE_NUM, FLAGS_aicore_num);
  options.emplace(ge::OP_SELECT_IMPL_MODE, FLAGS_op_select_implmode);
  options.emplace(ge::OPTYPELIST_FOR_IMPLMODE, FLAGS_optypelist_for_implmode);
  options.emplace(ge::AUTO_TUNE_MODE, FLAGS_auto_tune_mode);
  options.emplace(ge::OP_DEBUG_LEVEL, to_string(FLAGS_op_debug_level));
  options.emplace(ge::DEBUG_DIR, FLAGS_debug_dir);
  options.emplace(ge::OP_COMPILER_CACHE_DIR, FLAGS_op_compiler_cache_dir);
  options.emplace(ge::OP_COMPILER_CACHE_MODE, FLAGS_op_compiler_cache_mode);
  options.emplace(ge::MDL_BANK_PATH_FLAG, FLAGS_mdl_bank_path);
  options.emplace(ge::OP_BANK_PATH_FLAG, FLAGS_op_bank_path);
  options.emplace(ge::TUNE_DEVICE_IDS, FLAGS_device_id);
  options.emplace(ge::MODIFY_MIXLIST, FLAGS_modify_mixlist);
}

domi::Status GenerateSingleOp(const std::string& json_file_path) {
  if (!FLAGS_output.empty() && !ge::CheckOutputPathValid(FLAGS_output, "--output")) {
    DOMI_LOGE("output path %s is not valid!", FLAGS_output.c_str());
    return domi::FAILED;
  }
  // check optypelist_for_implmode and op_select_implmode
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      ge::CheckImplmodeParamValid(FLAGS_optypelist_for_implmode, FLAGS_op_select_implmode) != ge::SUCCESS,
      return ge::FAILED, "[Check][ImplmodeParam] fail for input optypelist_for_implmode and op_select_implmode.");

  if (!FLAGS_op_precision_mode.empty() && !ge::CheckInputPathValid(FLAGS_op_precision_mode, "--op_precision_mode")) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {"op_precision_mode", FLAGS_op_precision_mode.c_str(),
                                                    "path is not found"});
    GELOGE(ge::FAILED, "[Check][op_precision_mode] %s not found", FLAGS_op_precision_mode.c_str());
    return ge::FAILED;
  }

  if (ge::CheckModifyMixlistParamValid(FLAGS_precision_mode, FLAGS_modify_mixlist) != ge::SUCCESS) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {"modify_mixlist", FLAGS_modify_mixlist.c_str(),
                                                     ge::kModifyMixlistError});
    return ge::FAILED;
  }

  std::map<string, string> options;
  // need to be changed when ge.ini plan is done
  SetEnvForSingleOp(options);
  // print single op option map
  ge::PrintOptionMap(options, "single op option");

  auto ret = ge::GELib::Initialize(options);
  if (ret != ge::SUCCESS) {
    DOMI_LOGE("GE initialize failed!");
    return domi::FAILED;
  }

  ge::GeGenerator generator;
  ret = generator.Initialize(options, domi::GetContext());
  if (ret != SUCCESS) {
    DOMI_LOGE("GeGenerator initialize failed!");
    (void)ge::GELib::GetInstance()->Finalize();
    return domi::FAILED;
  }

  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  vector<ge::SingleOpBuildParam> build_params;
  if (ge::SingleOpParser::ParseSingleOpList(json_file_path, build_params) != ge::SUCCESS) {
    DOMI_LOGE("parse single op json file failed");
    (void)generator.Finalize();
    (void)ge::GELib::GetInstance()->Finalize();
    return domi::FAILED;
  }

  int index = 0;
  for (auto &param : build_params) {
    string output_path;
    if (!FLAGS_output.empty()) {
      output_path = FLAGS_output + "/";
    }
    output_path += param.file_name;
    ret = generator.BuildSingleOpModel(param.op_desc, param.inputs, param.outputs, output_path, param.compile_flag);
    if (ret != SUCCESS) {
      DOMI_LOGE("Compile op failed. ge ret = %u, op index = %d", ret, index);
      ret = domi::FAILED;
    } else {
      GELOGI("Compile op success. op index = %d, output = %s", index, output_path.c_str());
    }
    index += 1;
  }

  (void)generator.Finalize();
  (void)ge::GELib::GetInstance()->Finalize();
  return ret;
}

domi::Status GenerateOmModel() {
  if (!CheckInputFormat()) {
    GELOGE(ge::FAILED, "[Check][InputFormat]failed.");
    return domi::FAILED;
  }
  Status ret = GFlagUtils::CheckFlags();
  GE_CHK_BOOL_EXEC(ret == domi::SUCCESS, return domi::FAILED,
                   "[Check][Flags] failed! Please check whether some atc params that include semicolons[;] use double "
                   "quotation marks (\") to enclose each argument such as out_nodes, input_shape, dynamic_image_size");
#if !defined(__ANDROID__) && !defined(ANDROID)
  // Load custom operator Library
  LoadCustomOpLib(true);

  SaveCustomCaffeProtoPath();

  GE_CHK_BOOL_EXEC(ret == domi::SUCCESS, return domi::FAILED, "[Check][Flags]check custom aicpu run so failed!");
#endif

  const int f_stream_num = 1;
  std::map<string, string> options;
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(FLAGS_framework)));
  options.insert(std::pair<string, string>(string(ge::STREAM_NUM), to_string(f_stream_num)));
  options.insert(std::pair<string, string>(string(ge::CALIBRATION_CONF_FILE), FLAGS_cal_conf));
  options.insert(std::pair<string, string>(string(ge::OUTPUT_NODE_NAME), FLAGS_out_nodes));
  options.insert(std::pair<string, string>(string(ge::INSERT_OP_FILE), FLAGS_insert_op_conf));
  options.insert(std::pair<string, string>(string(ge::OP_PRECISION_MODE), FLAGS_op_precision_mode));
  options.insert(std::pair<string, string>(string(ge::PRECISION_MODE), FLAGS_precision_mode));
  options.insert(std::pair<string, string>(string(ge::TUNE_DEVICE_IDS), FLAGS_device_id));

  options.insert(std::pair<string, string>(string(ge::RUN_FLAG), to_string(0)));
  options.insert(std::pair<string, string>(string(ge::TRAIN_FLAG), to_string(0)));

  if (!FLAGS_output_type.empty()) {
    options.insert(std::pair<string, string>(string(ge::OUTPUT_DATATYPE), FLAGS_output_type));
  }

  options.insert(std::pair<string, string>(string(ge::OP_SELECT_IMPL_MODE), FLAGS_op_select_implmode));
  options.insert(std::pair<string, string>(string(ge::OPTYPELIST_FOR_IMPLMODE), FLAGS_optypelist_for_implmode));

  if (!FLAGS_input_fp16_nodes.empty()) {
    GELOGI("FLAGS_input_fp16_nodes : %s .", FLAGS_input_fp16_nodes.c_str());
    options.insert(std::pair<string, string>(ge::INPUT_FP16_NODES, FLAGS_input_fp16_nodes));
  }

  options.insert(std::pair<string, string>(string(ge::AUTO_TUNE_MODE), FLAGS_auto_tune_mode));

  options.insert(
      std::pair<string, string>(string(ge::OPTION_EXEC_DISABLE_REUSED_MEMORY), to_string(FLAGS_disable_reuse_memory)));

  options.insert(std::pair<string, string>(string(ge::SOC_VERSION), FLAGS_soc_version));

  options.insert(std::pair<string, string>(string(ge::CORE_TYPE), FLAGS_core_type));

  options.insert(std::pair<string, string>(string(ge::AICORE_NUM), FLAGS_aicore_num));

  options.insert(std::pair<string, string>(string(ge::BUFFER_OPTIMIZE), FLAGS_buffer_optimize));

  options.insert(std::pair<string, string>(string(ge::ENABLE_SMALL_CHANNEL), FLAGS_enable_small_channel));

  options.insert(std::pair<string, string>(string(ge::FUSION_SWITCH_FILE), FLAGS_fusion_switch_file));

  options.insert(std::pair<string, string>(string(ge::ENABLE_COMPRESS_WEIGHT),
                                           (FLAGS_enable_compress_weight == "true") ?
                                           ge::kEnableCompressWeightTrue : ge::kEnableCompressWeightFalse));

  options.insert(std::pair<string, string>(string(ge::ENABLE_SINGLE_STREAM), FLAGS_enable_single_stream));

  options.insert(std::pair<string, string>(string(ge::DEBUG_DIR), FLAGS_debug_dir));

  options.insert(std::pair<string, string>(string(ge::OP_COMPILER_CACHE_DIR), FLAGS_op_compiler_cache_dir));

  options.insert(std::pair<string, string>(string(ge::OP_COMPILER_CACHE_MODE), FLAGS_op_compiler_cache_mode));

  SetDynamicInputSizeOptions();

  if (!FLAGS_save_original_model.empty()) {
    options.insert(std::pair<string, string>(string(ge::SAVE_ORIGINAL_MODEL), FLAGS_save_original_model));
    options.insert(std::pair<string, string>(string(ge::ORIGINAL_MODEL_FILE), FLAGS_output + "_original.om"));
  }

  options.insert(std::pair<string, string>(string(ge::OP_DEBUG_LEVEL), to_string(FLAGS_op_debug_level)));

  options.insert(std::pair<string, string>(string(ge::MDL_BANK_PATH_FLAG), FLAGS_mdl_bank_path));

  options.insert(std::pair<string, string>(string(ge::OP_BANK_PATH_FLAG), FLAGS_op_bank_path));

  options.insert(std::pair<string, string>(string(ge::DISPLAY_MODEL_INFO), FLAGS_display_model_info));

  options.insert(std::pair<string, string>(string(ge::MODIFY_MIXLIST), FLAGS_modify_mixlist));

  // set enable scope fusion passes
  SetEnableScopeFusionPasses(FLAGS_enable_scope_fusion_passes);
  // print atc option map
  ge::PrintOptionMap(options, "atc option");

  // When the ATC module is transferred to a model, the suffix ".om" is automatically added to the model name
  FLAGS_output = FLAGS_output + ".om";
  ret = GenerateModel(options, FLAGS_output);
  if (ret != domi::SUCCESS) {
    return domi::FAILED;
  }

  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  if (FLAGS_display_model_info == "1") {
    GELOGI("need to display model info.");
    return ge::ConvertOm(FLAGS_output.c_str(), "", false);
  }

  return domi::SUCCESS;
}

domi::Status ConvertModelToJson() {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  Status ret = GFlagUtils::CheckConverJsonParamFlags();
  GE_CHK_BOOL_EXEC(ret == domi::SUCCESS, return domi::FAILED, "[CheckConver][JsonParamFlags] failed!");

  ret = ConvertModelToJson(FLAGS_framework, FLAGS_om, FLAGS_json);

  GE_IF_BOOL_EXEC(ret != domi::SUCCESS, return domi::FAILED);
  return domi::SUCCESS;
}

domi::Status DisplayModelInfo() {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  // No model path passed in
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(FLAGS_om == "",
      ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {"om"});
      return ge::FAILED,
      "[Check][Parameter]Input parameter[--om]'s value is empty!!");

  // Check if the model path is valid
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      FLAGS_om != "" && !ge::CheckInputPathValid(FLAGS_om, "--om"),
      return ge::FAILED,
      "[Check][InputPath]model file path is invalid: %s.", FLAGS_om.c_str());

  if (FLAGS_framework == -1) {
    return ge::ConvertOm(FLAGS_om.c_str(), "", false);
  }

  return ge::FAILED;
}

bool CheckRet(domi::Status ret) {
  if (ret != domi::SUCCESS) {
    if (FLAGS_mode == ONLY_PRE_CHECK) {
      GELOGW("ATC precheck failed.");
    } else if (FLAGS_mode == GEN_OM_MODEL) {
      GELOGW("ATC generate offline model failed.");
    } else if (FLAGS_mode == MODEL_TO_JSON) {
      GELOGW("ATC convert model to json file failed.");
    } else if (FLAGS_mode == PBTXT_TO_JSON) {
      GELOGW("ATC convert pbtxt to json file failed.");
    } else {
      return false;
    }
    return false;
  }

  if (FLAGS_mode == ONLY_PRE_CHECK) {
    GELOGI("ATC precheck success.");
  } else if (FLAGS_mode == GEN_OM_MODEL) {
    GELOGI("ATC generate offline model success.");
  } else if (FLAGS_mode == MODEL_TO_JSON) {
    GELOGI("ATC convert model to json file success.");
  } else if (FLAGS_mode == PBTXT_TO_JSON) {
    GELOGI("ATC convert pbtxt to json file success.");
  }
  return true;
}

domi::Status ConvertPbtxtToJson() {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kOther);
  Status ret = GFlagUtils::CheckConverJsonParamFlags();
  if (ret != domi::SUCCESS) {
    GELOGE(ge::FAILED, "[CheckConver][JsonParamFlags] failed!");
    return domi::FAILED;
  }

  ret = ge::ConvertPbtxtToJson(FLAGS_om.c_str(), FLAGS_json.c_str());
  if (ret != domi::SUCCESS) {
    GELOGE(ge::FAILED, "[Convert][PbtxtToJson] fail.");
    REPORT_CALL_ERROR("E19999", "ConvertPbtxtToJson failed, FLAGS_om:%s, FLAGS_json:%s.",
        FLAGS_om.c_str(), FLAGS_json.c_str());
    return domi::FAILED;
  }

  return domi::SUCCESS;
}

int init(int argc, char* argv[]) {
  GFlagUtils::InitGFlag(argc, argv);
  const char *gflag_argv = gflags::GetArgv();
  string cmdline = gflag_argv == nullptr ? "" : gflag_argv;
  domi::GetContext().atc_cmdline = cmdline;
  // set log level
  int ret = -1;
  const std::set<string> log_level = {"null", "debug", "info", "warning", "error"};
  if (log_level.count(FLAGS_log) == 0) {
    std::cout << "E10010: invalid value for --log:" << FLAGS_log
              <<", only support debug, info, warning, error, null"<< std::endl;
    return ret;
  }

  ret = ge::CheckLogParamValidAndSetLogLevel(FLAGS_log);
  if (ret != 0) {
    return ret;
  }

  std::string path_base = ge::GELib::GetPath();
  ret = ErrorManager::GetInstance().Init(path_base);
  if (ret != 0) {
    DOMI_LOGE("ErrorManager init fail !");
    return ret;
  }

  ErrorManager::GetInstance().GenWorkStreamIdDefault();
  return 0;
}

long GetMemInfo(const std::string &key) {
  std::string file_path = "/proc/meminfo";
  std::ifstream fs(file_path, std::ifstream::in);
  if (!fs.is_open()) {
    GELOGW("Can not open %s .", file_path.c_str());
    return 0;
  }
  std::string line;
  while (getline(fs, line)) {  // line not with \n
    if (line.find(key) != std::string::npos) {
      GELOGI("Find mem [%s] info line [%s]", key.c_str(), line.c_str());
      fs.close();
      size_t pos = line.find(":");
      if (pos == std::string::npos) {
        return 0;
      }
      std::string current_mem_info_str = line.substr(pos + 1);
      ge::StringUtils::Trim(current_mem_info_str);
      GELOGI("Find mem [%s] info [%s].", key.c_str(), current_mem_info_str.c_str());
      return stol(current_mem_info_str);
    }
  }
  fs.close();  // close the file
  return 0;
}

bool CheckMemInfo() {
  if (FLAGS_auto_tune_mode.empty()) {
    return true;
  }
  // only check current available mem when auto_tune_mode is set.
  long current_mem_available = GetMemInfo("MemAvailable");
  GELOGI("Get mem available [%lu kB].", current_mem_available);
  std::cout << "Current available mem is " << current_mem_available << "kB." << std::endl;
  if ((current_mem_available > 0) && (current_mem_available < kMinAvailableMem)) {
    GELOGE(ge::PARAM_INVALID, "[Check][MemSize]Current available mem [%lu kB] can not be smaller than [%lu kB] .",
        current_mem_available, kMinAvailableMem);
    ErrorManager::GetInstance().ATCReportErrMessage("E10044", {"value", "min_value"},
                                                    {to_string(current_mem_available), to_string(kMinAvailableMem)});
    return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  ErrorManager::GetInstance().SetStage(error_message::kInitialize, error_message::kOther);
  Status ret = domi::SUCCESS;
  std::cout << "ATC start working now, please wait for a moment." << std::endl;

  // Initialize
  if (init(argc, argv) != 0) {
    std::cout << "ATC run failed, Please check the detail log, Try \'atc --help\' for more information" << std::endl;
    return -1;
  }
  do {
    if (!CheckMemInfo()) {
      GELOGE(ge::PARAM_INVALID, "[Check][MemInfo]Current available mem is too small.");
      ret = domi::FAILED;
      break;
    }
    if (!FLAGS_singleop.empty()) {
      ret = GenerateSingleOp(FLAGS_singleop);
      break;
    }

    // default mode(mode:0), Open source model to model
    if (GEN_OM_MODEL == FLAGS_mode || ONLY_PRE_CHECK == FLAGS_mode) {
      GE_IF_BOOL_EXEC(GenerateOmModel() != domi::SUCCESS, ret = domi::FAILED; break);
    } else if (MODEL_TO_JSON == FLAGS_mode) {  // Mode 1, transfer model to JSON
      GE_CHK_BOOL_EXEC(ConvertModelToJson() == domi::SUCCESS, ret = domi::FAILED;
                       break, "[Convert][ModelToJson]ATC ConvertJson execute failed!!");
    } else if (FLAGS_mode == ge::RunMode::PBTXT_TO_JSON) {
      GE_CHK_BOOL_EXEC(ConvertPbtxtToJson() == domi::SUCCESS, ret = domi::FAILED;
                       break, "[Convert][PbtxtToJson]ATC convert pbtxt to json execute failed!!");
    } else if (FLAGS_mode == ge::RunMode::DISPLAY_OM_INFO) {
      GE_CHK_BOOL_EXEC(DisplayModelInfo() == domi::SUCCESS, ret = domi::FAILED;
        break, "[Display][ModelInfo]ATC DisplayModelInfo failed!!");
    } else {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E10001", {"parameter", "value", "reason"}, {"--mode", std::to_string(FLAGS_mode), kModeSupport});
      GELOGE(ge::PARAM_INVALID, "[Check][Parameter]Invalid value for --mode[%d], %s.", FLAGS_mode, kModeSupport);
      ret = domi::FAILED;
      break;
    }
  } while (0);

  ErrorManager::GetInstance().SetStage(error_message::kFinalize, error_message::kFinalize);
  if (!CheckRet(ret)) {
    std::cout << "ATC run failed, Please check the detail log, Try \'atc --help\' for more information" << std::endl;
    int result = ErrorManager::GetInstance().OutputErrMessage(STDOUT_FILENO);
    if (result != 0) {
      DOMI_LOGE("ErrorManager outputErrMessage fail !");
    }
    GELOGI("Current mem available mem is [%lu kB]", GetMemInfo("MemAvailable"));
    return ret;
  } else {
    std::cout << "ATC run success, welcome to the next use." << std::endl;
    (void)ErrorManager::GetInstance().OutputMessage(STDOUT_FILENO);
    return 0;
  }
} /*lint +e530*/
