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

/*lint -e* */
#ifndef INC_FRAMEWORK_COMMON_GE_INNER_ERROR_CODES_H_
#define INC_FRAMEWORK_COMMON_GE_INNER_ERROR_CODES_H_

#include <map>
#include <string>
#include "ge/ge_api_error_codes.h"

namespace ge {
// System ID
enum SystemIdType { SYSID_GE = 8 };
// Runtime location
enum LogRuntime {
  RT_HOST = 0b01,
  RT_DEVICE = 0b10,
};

// Sub model
enum SubModuleId {
  COMMON_MODULE = 0,
  CLIENT_MODULE = 1,
  INIT_MODULE = 2,
  SESSION_MODULE = 3,
  GRAPH_MODULE = 4,
  ENGINE_MODULE = 5,
  OPS_MODULE = 6,
  PLUGIN_MODULE = 7,
  RUNTIME_MODULE = 8,
  EXECUTOR_MODULE = 9,
  GENERATOR_MODULE = 10,
};

// Error code type
enum ErrorCodeType {
  ERROR_CODE = 0b01,
  EXCEPTION_CODE = 0b10,
};

// Error level
enum ErrorLevel {
  COMMON_LEVEL = 0b000,
  SUGGESTION_LEVEL = 0b001,
  MINOR_LEVEL = 0b010,
  MAJOR_LEVEL = 0b011,
  CRITICAL_LEVEL = 0b100,
};

// Each module defines error codes using the following macros, name can not be modified to (name)
#define GE_ERRORNO_COMMON(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, COMMON_MODULE, name, (value), (desc))
#define GE_ERRORNO_CLIENT(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, CLIENT_MODULE, name, (value), (desc))
#define GE_ERRORNO_INIT(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, INIT_MODULE, name, (value), (desc))
#define GE_ERRORNO_SESSION(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, SESSION_MODULE, name, (value), (desc))
#define GE_ERRORNO_GRAPH(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, GRAPH_MODULE, name, (value), (desc))
#define GE_ERRORNO_ENGINE(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, ENGINE_MODULE, name, (value), (desc))
#define GE_ERRORNO_OPS(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, OPS_MODULE, name, (value), (desc))
#define GE_ERRORNO_PLUGIN(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, PLUGIN_MODULE, name, (value), (desc))
#define GE_ERRORNO_RUNTIME(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, RUNTIME_MODULE, name, (value), (desc))
#define GE_ERRORNO_EXECUTOR(name, value, desc) \
  GE_ERRORNO(RT_DEVICE, ERROR_CODE, COMMON_LEVEL, SYSID_GE, EXECUTOR_MODULE, name, (value), (desc))
#define GE_ERRORNO_GENERATOR(name, value, desc) \
  GE_ERRORNO(RT_HOST, ERROR_CODE, COMMON_LEVEL, SYSID_GE, GENERATOR_MODULE, name, (value), (desc))

// Get error code description
#define GE_GET_ERRORNO_STR(value) ge::StatusFactory::Instance()->GetErrDesc(value)

// Common module error code definition
GE_ERRORNO_COMMON(MEMALLOC_FAILED, 0, "Failed to allocate memory!");  // 1343225856
GE_ERRORNO_COMMON(PARAM_INVALID, 1, "Parameter's invalid!");          // 1343225857
GE_ERRORNO_COMMON(CCE_FAILED, 2, "Failed to call CCE API!");          // 1343225858
GE_ERRORNO_COMMON(RT_FAILED, 3, "Failed to call runtime API!");       // 1343225859
GE_ERRORNO_COMMON(INTERNAL_ERROR, 4, "Internal errors");              // 1343225860
GE_ERRORNO_COMMON(CSEC_ERROR, 5, "Failed to call libc_sec API!");     // 1343225861
GE_ERRORNO_COMMON(TEE_ERROR, 6, "Failed to call tee API!");           // 1343225862
GE_ERRORNO_COMMON(END_OF_SEQUENCE, 7, "End of sequence!");            // 1343225863
GE_ERRORNO_COMMON(PATH_INVALID, 8, "Path is invalid!");               // 1343225864

// Error code for plugin manager
GE_ERRORNO_COMMON(GE_PLGMGR_PATH_INVALID, 30, "Path is invalid!");                   // 1343225886
GE_ERRORNO_COMMON(GE_PLGMGR_SO_NOT_EXIST, 31, "Failed to find any valid so file!");  // 1343225887
GE_ERRORNO_COMMON(GE_PLGMGR_FUNC_NOT_EXIST, 32, "Failed to find any function!");     // 1343225888
GE_ERRORNO_COMMON(GE_PLGMGR_INVOKE_FAILED, 33, "Failed to invoke any function!");    // 1343225889

GE_ERRORNO_COMMON(UNSUPPORTED, 100, "Parameter's unsupported!");

GE_ERRORNO_COMMON(OUT_OF_MEMORY, 101, "Out of memory!");

// Client module error code definition
GE_ERRORNO_CLIENT(GE_CLI_INIT_FAILED, 1, "GEInitialize Failed.");                   // 1343229953
GE_ERRORNO_CLIENT(GE_CLI_FINAL_FAILED, 2, "GEFinalize Failed.");                    // 1343229954
GE_ERRORNO_CLIENT(GE_CLI_SESS_CONSTRUCT_FAILED, 3, "Session constructor Failed.");  // 1343229955
GE_ERRORNO_CLIENT(GE_CLI_SESS_DESTROY_FAILED, 4, "Session destructor Failed.");     // 1343229956
GE_ERRORNO_CLIENT(GE_CLI_SESS_ADD_FAILED, 5, "Session AddGraph Failed.");           // 1343229957
GE_ERRORNO_CLIENT(GE_CLI_SESS_ADD_GRAPH_FAILED, 6,
                  "Session AddGraph Failed converting protobuf GraphProto.");    // 1343229958
GE_ERRORNO_CLIENT(GE_CLI_SESS_REMOVE_FAILED, 7, "Session RemoveGraph Failed.");  // 1343229959
GE_ERRORNO_CLIENT(GE_CLI_SESS_RUN_FAILED, 8, "Session RunGraph Failed.");        // 1343229960
GE_ERRORNO_CLIENT(GE_CLI_SESS_RUN_TENSOR_FAILED, 9,
                  "Session RunGraph Failed converting protobuf TensorProto.");                   // 1343229961
GE_ERRORNO_CLIENT(GE_CLI_GE_ALREADY_INITIALIZED, 10, "GE is already initialized.");              // 1343229962
GE_ERRORNO_CLIENT(GE_CLI_GE_NOT_INITIALIZED, 11, "GE is not yet initialized or is finalized.");  // 1343229963

// Init module error code definition
GE_ERRORNO_INIT(GE_MULTI_INIT, 0, "Multiple initializations are not supported.");                 // 1343234048
GE_ERRORNO_INIT(GE_FINALIZE_NOT_INIT, 1, "Finalize is not allowed before initialization.");       // 1343234049
GE_ERRORNO_INIT(GE_MULTI_FINALIZE, 2, "Multiple finalizations are not supported.");               // 1343234050
GE_ERRORNO_INIT(GE_PROF_MULTI_INIT, 3, "Multiple profiling initializations are not supported.");  // 1343234051
GE_ERRORNO_INIT(GE_PROF_NOT_INIT, 4, "Profing initializations have not been done.");              // 1343234052
GE_ERRORNO_INIT(GE_PROF_MODE_CONFLICT, 5,
                "Profiling command mode which is preferred is running, the api mode will not work.");  // 1343234053

// Session module error code definition
GE_ERRORNO_SESSION(GE_SESS_INIT_FAILED, 0, "Failed to initialize session.");                          // 1343238144
GE_ERRORNO_SESSION(GE_SESS_ALREADY_RUNNING, 1, "Session already running,not support parallel run.");  // 1343238145
GE_ERRORNO_SESSION(GE_SESS_GRAPH_NOT_EXIST, 2, "Graph ID not exist.");                                // 1343238146
GE_ERRORNO_SESSION(GE_SESS_GRAPH_ALREADY_EXIST, 3, "Graph ID already exist.");                        // 1343238147
GE_ERRORNO_SESSION(GE_SESS_GRAPH_IS_RUNNING, 4, "Graph is running.");                                 // 1343238148
GE_ERRORNO_SESSION(GE_SESSION_NOT_EXIST, 5, "Can not find session with specific session id.");        // 1343238149
GE_ERRORNO_SESSION(GE_SESSION_MANAGER_NOT_INIT, 6, "Session manager has not been initialized.");      // 1343238150

// Graph module error code definition
GE_ERRORNO_GRAPH(GE_GRAPH_INIT_FAILED, 0, "Failed to initialize graph.");                                // 1343242240
GE_ERRORNO_GRAPH(GE_GRAPH_ALREADY_RUNNING, 1, "graph already running,not support parallel run.");        // 1343242241
GE_ERRORNO_GRAPH(GE_GRAPH_GRAPH_NOT_EXIST, 2, "graph ID not exist.");                                    // 1343242242
GE_ERRORNO_GRAPH(GE_GRAPH_GRAPH_ALREADY_EXIST, 3, "Graph ID already exist.");                            // 1343242243
GE_ERRORNO_GRAPH(GE_GRAPH_GRAPH_IS_RUNNING, 4, "Graph is running.");                                     // 1343242244
GE_ERRORNO_GRAPH(GE_GRAPH_MALLOC_FAILED, 5, "Graph malloc failed.");                                     // 1343242245
GE_ERRORNO_GRAPH(GE_GRAPH_FREE_FAILED, 6, "Graph FREE failed.");                                         // 1343242246
GE_ERRORNO_GRAPH(GE_GRAPH_NOT_MALLOC_BUFFER, 7, "Graph FREE failed, not malloc buffer.");                // 1343242247
GE_ERRORNO_GRAPH(GE_GRAPH_PARAM_NULLPTR, 8, "Graph param is NULL.");                                     // 1343242248
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, 9, "Get computeGraph by graphNode failed.");      // 1343242249
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_RUN_GRAPH_NODE_NULL, 10, "Run graph node is null.");                  // 1343242250
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_RUN_GRAPH_INVALID, 11, "Get computeGraph by graphNode failed.");      // 1343242251
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_INSERT_DYN_OP_FAILED, 12, "Graph which insert dynamic op failed.");   // 1343242252
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_PREPROCESS_FAILED, 13, "Graph preprocess failed.");                   // 1343242253
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_GRAPH_FUSION_FAILED, 14, "Graph fusion failed.");                     // 1343242254
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_CALIBRATION_FAILED, 16, "Calibration failed.");                       // 1343242256
GE_ERRORNO_GRAPH(GE_GRAPH_SUBGRAPH_NUM_ZERO, 17, "Graph partition success, but subGraph num is 0.");     // 1343242257
GE_ERRORNO_GRAPH(GE_GRAPH_SUBGRAPH_ENGINENAME_REPEATED, 18, "Graph subGraph engine name is repeated.");  // 1343242258
GE_ERRORNO_GRAPH(GE_GRAPH_GET_IN_OUT_FAILED, 19, "OME GetInputOutputDescInfo failed.");                  // 1343242259
GE_ERRORNO_GRAPH(GE_GRAPH_DATA_INPUT_FAILED, 20, "OME DataInput failed.");                               // 1343242260
GE_ERRORNO_GRAPH(GE_GRAPH_EXECUTE_FAILED, 21, "Execute graph failed.");                                  // 1343242261
GE_ERRORNO_GRAPH(GE_GRAPH_DUPLICATE_ENGINE, 22, "Duplicate engine.");                                    // 1343242262
GE_ERRORNO_GRAPH(GE_GRAPH_EMPTY_SUBGRAPH, 23, "Empty sub graph info.");                                  // 1343242263
GE_ERRORNO_GRAPH(GE_GRAPH_EXECUTE_NOT_INIT, 24, "Call SetCondition first.");                             // 1343242264
GE_ERRORNO_GRAPH(GE_GRAPH_PREPARE_FAILED, 25, "Prepare failed.");                                        // 1343242265
GE_ERRORNO_GRAPH(GE_GRAPH_SERIALIZE_FAILED, 26, "OMG SerializeModelDef failed.");                        // 1343242266
GE_ERRORNO_GRAPH(GE_GRAPH_SAVE_FAILED, 27, "OMG SaveModel failed.");                                     // 1343242267
GE_ERRORNO_GRAPH(GE_GRAPH_PRERUN_FAILED, 28, "PreRun failed.");                                          // 1343242268
GE_ERRORNO_GRAPH(GE_GRAPH_SUBGRAPH_ID_INVALID, 29, "Graph subGraph id is invalid.");                     // 1343242269
GE_ERRORNO_GRAPH(GE_GRAPH_INFERSHAPE_FAILED, 30, "Prepare Graph infershape failed");                     // 1343242270
GE_ERRORNO_GRAPH(GE_GRAPH_ISNULL, 31, "RunGraph input compute graph is NULL.");                          // 1343242271
GE_ERRORNO_GRAPH(GE_GRAPH_SYNC_MODEL_FAILED, 32, "Graph SyncExecuteModel failed.");                      // 1343242272
GE_ERRORNO_GRAPH(GE_GRAPH_RUNGRAPH_FAILED, 33, "Graph RunGraph failed.");                                // 1343242273
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_PARSE_DYN_OP_FAILED, 34, "Parse dynamic node config file failed");    // 1343242274
GE_ERRORNO_GRAPH(GE_GRAPH_MULTI_SUBGRAPH_BUILD, 35, "Save model with multiple sub graph");               // 1343242275
GE_ERRORNO_GRAPH(GE_GRAPH_GRAPH_NODE_NULL, 36, "Graph get graph node failed.");                          // 1343242276
GE_ERRORNO_GRAPH(GE_GRAPH_NOT_INIT, 37, "Graph do not init.");                                           // 1343242277
GE_ERRORNO_GRAPH(GE_GRAPH_NULL_INPUT, 38, "input graph is null");                                        // 1343242278
GE_ERRORNO_GRAPH(GE_GRAPH_TOPO_SORT_FAILED, 39, "topological sorting an partition failed");              // 1343242279
GE_ERRORNO_GRAPH(GE_GRAPH_EMPTY_PARTITION, 40, "accessing an empty partition");                          // 1343242280
GE_ERRORNO_GRAPH(GE_GRAPH_UNSUPPORTED, 41, "unsupported feature in partition");                          // 1343242281
GE_ERRORNO_GRAPH(GE_GRAPH_ASSIGN_ENGINE_FAILED, 42, "assign engine failed");                             // 1343242282
GE_ERRORNO_GRAPH(GE_GRAPH_ADD_PLC_END_FAILED, 43, "add placeholder end node failed");                    // 1343242283
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_PARSE_OUT_NODE_FAILED, 44, "Parse out node failed.");                 // 1343242284
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIMIZE_INSERT_OP_PARSE_FAILED, 45,
                 "OMG parse dynamic node config file failed.");                              // 1343242285
GE_ERRORNO_GRAPH(GE_GRAPH_SAVE_WEIGHTS_FAILED, 46, "OMG Save Weights to Model failed.");     // 1343242286
GE_ERRORNO_GRAPH(GE_GRAPH_EMPTY_STRING_NAME, 47, "Empty string name.");                      // 1343242287
GE_ERRORNO_GRAPH(GE_GRAPH_EMPTY_VARIABLE_TENSOR_TABLE, 48, "Empty variable-tensor table.");  // 1343242288
GE_ERRORNO_GRAPH(GE_GRAPH_VARIABLE_ALREADY_EXIST, 49, "Variable already exist.");            // 1343242289
GE_ERRORNO_GRAPH(GE_GRAPH_VARIABLE_DOES_NOT_EXIST, 50, "Variable does not exist.");          // 1343242290
GE_ERRORNO_GRAPH(GE_GRAPH_OPTIONS_INVALID, 51, "Client session options is invalid.");        // 1343242291
GE_ERRORNO_GRAPH(GE_GRAPH_NO_OUTPUT_DESC_INFO, 52, "No output desc info.");                  // 1343242292
GE_ERRORNO_GRAPH(GE_GRAPH_OUTPUT_DESCINFO_TENSOR_NUM_MISMATCH, 53,
                 "Number of output descinfo and tensor mismatch.");                                        // 1343242293
GE_ERRORNO_GRAPH(GE_GRAPH_FILENAMEPREFIX_INVALID, 54, "Graph Save Model fileNamePrefix is invalid.");      // 1343242294
GE_ERRORNO_GRAPH(GE_GRAPH_NOT_BUILT, 55, "Graph is not built before SaveModel.");                          // 1343242295
GE_ERRORNO_GRAPH(GE_GRAPH_SAVEMODEL_FAILED, 56, "Graph SaveModel failed.");                                // 1343242296
GE_ERRORNO_GRAPH(GE_GRAPH_MEMORY_ALLOC_FAILED, 57, "Failed allocating memory for model file header.");     // 1343242297
GE_ERRORNO_GRAPH(GE_GRAPH_NODE_SEARCHER_REMOVE_GRAPH_FAILED, 58, "Failed remove graph in node seacher.");  // 1343242298
GE_ERRORNO_GRAPH(GE_GRAPH_NODE_SEARCHER_ADD_GRAPH_FAILED, 59, "Failed add graph in node seacher.");        // 1343242299
GE_ERRORNO_GRAPH(GE_GRAPH_NODE_SEARCHER_GET_GRAPH_REBUILD_FAILED, 60,
                 "Failed add graph in node seacher.");  // 1343242300
GE_ERRORNO_GRAPH(GE_GRAPH_NODE_SEARCHER_SET_GRAPH_FINISH_REBUILD_GRAPH_FAILED, 61,
                 "Failed set graph finish rebuild in node searcher.");                   // 1343242301
GE_ERRORNO_GRAPH(GE_GRAPH_VARIABLE_OP_PASS_FAILED, 62, "Failed to run variable pass.");  // 1343242302

// Engine_manager module error code definition
GE_ERRORNO_ENGINE(GE_ENG_INIT_FAILED, 0, "Failed to initialize engine.");                             // 1343246336
GE_ERRORNO_ENGINE(GE_ENG_FINALIZE_FAILED, 1, "Engine finalize failed.");                              // 1343246337
GE_ERRORNO_ENGINE(GE_ENG_MEMTYPE_ERROR, 2, "Memory type HBM is necessary when engine is in device");  // 1343246338

// Optimize errocode
GE_ERRORNO_GRAPH(TO_BE_DELETED, 63, "The node of the graph to be deleted.");  // 1343242303
GE_ERRORNO_GRAPH(NOT_CHANGED, 64, "The node of the graph no changed.");       // 1343242304

// Ops module error code definition
GE_ERRORNO_OPS(GE_OPS_KERNEL_STORE_INIT_FAILED, 0, "Failed to initialize OpsKernelInfoStore.");  // 1343250432
GE_ERRORNO_OPS(GE_OPS_GRAPH_OPTIMIZER_INIT_FAILED, 1, "Failed to initialize GraphOptimizer.");   // 1343250433
GE_ERRORNO_OPS(GE_OPS_KERNEL_INFO_NOT_EXIST, 2, "OpsKernelInfo not exist.");                     // 1343250434
GE_ERRORNO_OPS(GE_OPS_KERNEL_STORE_NOT_EXIST, 3, "OpsKernelInfoStore not exist.");               // 1343250435
GE_ERRORNO_OPS(GE_OPS_CALC_RUNNING_PARAM_FAILED, 4, "Failed to CalcOpRunningParam.");            // 1343250436
GE_ERRORNO_OPS(GE_OPS_GENERATE_TASK_FAILED, 5, "Failed to GenerateTask.");                       // 1343250437
GE_ERRORNO_OPS(GE_OPS_OPTIMIZE_ORIGINAL_GRAPH_FAILED, 6, "Failed to OptimizeOriginalGraph.");    // 1343250438
GE_ERRORNO_OPS(GE_OPS_OPTIMIZE_FUSED_GRAPH_FAILED, 7, "Failed to OptimizeFusedGraph.");          // 1343250439
GE_ERRORNO_OPS(GE_OPS_ENGINE_IS_NOT_REGISTERED, 8, "Engine is not registered.");                 // 1343250440
GE_ERRORNO_OPS(GE_OPS_GET_NO_VALID_SO, 9,
               "There is no valid so about OpsKernelInfoStore or GraphOptimizer.");                       // 1343250441
GE_ERRORNO_OPS(GE_OPS_GET_OPTIMIZE_BY_ENGINE_FAILED, 10, "Failed to get graphOptimizer by name.");        // 1343250442
GE_ERRORNO_OPS(GE_OPS_GET_OPTIMIZE_BY_PRIORITY_FAILED, 11, "Failed to get graphOptimizer by priority.");  // 1343250443
GE_ERRORNO_OPS(GE_OPS_LOAD_GE_OPTIMIZER_FAILED, 12, "Failed to load ge graphOptimizer.");                 // 1343250444

// Runtime module error code definition
GE_ERRORNO_RUNTIME(GE_RTI_DEVICE_ID_INVALID, 1, "device id is invalid");
GE_ERRORNO_RUNTIME(GE_RTI_DEVICE_NOT_READY, 2, "set device failed, device not ready");
GE_ERRORNO_RUNTIME(GE_RTI_MEMALLOC_FAILED, 3, "malloc memory failed");
GE_ERRORNO_RUNTIME(GE_RTI_MODEL_NOT_LOADED, 4, "model has not been loaded");
GE_ERRORNO_RUNTIME(GE_RTI_THREAD_POOL_IS_NULL, 5, "model excute failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_CCE_CREATE_HANDLE_FAILED, 6, "cce create handle failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_CCE_SET_STREAM_FAILED, 7, "cce set stream failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_CREATE_RTMODEL_FAILED, 8, "call runtime create rtModel failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_CREATE_STREAM_FAILED, 9, "call runtime create stream failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_BIND_STREAM_FAILED, 10, "call runtime bind stream to model failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_CREATE_LABLE_FAILED, 11, "call runtime create lable failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MODEL_LOAD_COMPLETE_FAILED, 12, "call runtime model load complete failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MODEL_GET_TASK_ID_FAILED, 14, "call runtime get task id failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_KERNEL_LAUNCH_FAILED, 13, "call runtime kernel launch failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_KERNEL_LAUNCHEX_FAILED, 15, "call runtime kernel launchex failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_KERNEL_FUSION_START_FAILED, 16, "call runtime kernel fusion start failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_KERNEL_FUSION_END_FAILED, 17, "call runtime kernel fusion end failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_LABEL_SET_FAILED, 18, "call runtime lable set failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_LABLE_GOTO_FAILED, 19, "call runtime lable goto failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_LABLE_SWITCH_FAILED, 20, "call runtime lable switch failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MEM_ALLOC_MANAGED_FAILED, 21, "call runtime mem alloc managed failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MEM_FREE_MANAGED_FAILED, 22, "call runtime mem free managed failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_FREE_FAILED, 23, "call runtime free failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_STREAM_SYNC_FAILED, 24, "call runtime sync stream failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MODEL_EXCUTE_FAILED, 25, "call runtime model excute failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MEM_ASYNC_FAILED, 26, "call runtime mem async failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MEM_ALLOC_HOST_FAILED, 27, "call runtime alloc host memory failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MEM_FREE_HOST_FAILED, 28, "call runtime free host memory failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MEM_ALLOC_DEVICE_FAILED, 29, "call runtime alloc device memory failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_MEM_FREE_DEVICE_FAILED, 30, "call runtime free device memory failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_FLUSH_CACHE_FAILED, 31, "call runtime flush cache failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_UNBIND_STREAM_FAILED, 32, "unbind rtstream from rtmodel failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_DESTORY_STREAM_FAILED, 33, "destory stream failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_DESTORY_LABEL_FAILED, 34, "destory label failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_DESTORY_MODEL_FAILED, 35, "destory model failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_CCE_TRANS_TENSOR_FAILED, 36, "call cce transfer tensor descriptor failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_CCE_TRANS_FILTER_FAILED, 37, "call cce transfer filter descriptor failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_CCE_UPDATE_KERNEL_ARGS_FAILED, 38, "call cce update kernel args failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_CCE_DESTORY_HANDLE_FAILED, 39, "destory handle failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_CREATE_EVENT_FAILED, 40, "call rutime create event failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_EVENT_RECORD_FAILED, 41, "call rutime event record failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_STREAM_WAIT_EVENT_FAILED, 42, "call rutime stream wait event failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_HCCL_BROADCAST_FAILED, 43, "call hccl hcom broadcast failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_HCCL_ALL_GATHER_FAILED, 44, "call hccl hcom all gather failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_HCCL_ALL_REDUCE_FAILED, 45, "call hccl hcom all reduce failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_RUNTIME_DESTORY_EVENT_FAILED, 46, "destory rt event failed");
GE_ERRORNO_RUNTIME(GE_RTI_CALL_HCCL_REDUCE_SCATTER_FAILED, 47, "call hccl hcom reduce scatter failed");

// Executor module error code definition
GE_ERRORNO_EXECUTOR(GE_EXEC_NOT_INIT, 1, "GE Executor is not yet initialized.");
GE_ERRORNO_EXECUTOR(GE_EXEC_MODEL_PATH_INVALID, 2, "Model file path is invalid.");
GE_ERRORNO_EXECUTOR(GE_EXEC_MODEL_KEY_PATH_INVALID, 3, "Key file path of model is invalid.");
GE_ERRORNO_EXECUTOR(GE_EXEC_MODEL_ID_INVALID, 4, "Model id is invalid.");
GE_ERRORNO_EXECUTOR(GE_EXEC_MODEL_DATA_SIZE_INVALID, 5, "Data size of model is invalid.");
GE_ERRORNO_EXECUTOR(GE_EXEC_MODEL_PARTITION_NUM_INVALID, 6, "Partition number of model is invalid.");
GE_ERRORNO_EXECUTOR(GE_EXEC_MODEL_QUEUE_ID_INVALID, 7, "Queue id of model is invalid.");
GE_ERRORNO_EXECUTOR(GE_EXEC_MODEL_NOT_SUPPORT_ENCRYPTION, 8, "Model does not support encryption.");
GE_ERRORNO_EXECUTOR(GE_EXEC_READ_MODEL_FILE_FAILED, 9, "Failed to read model file.");
GE_ERRORNO_EXECUTOR(GE_EXEC_LOAD_MODEL_REPEATED, 10, "The model is loaded repeatedly.");
GE_ERRORNO_EXECUTOR(GE_EXEC_LOAD_MODEL_PARTITION_FAILED, 11, "Failed to load model partition.");
GE_ERRORNO_EXECUTOR(GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED, 12, "Failed to load weight partition.");
GE_ERRORNO_EXECUTOR(GE_EXEC_LOAD_TASK_PARTITION_FAILED, 13, "Failed to load task partition.");
GE_ERRORNO_EXECUTOR(GE_EXEC_LOAD_KERNEL_PARTITION_FAILED, 14, "Failed to load kernel partition.");
GE_ERRORNO_EXECUTOR(GE_EXEC_ALLOC_FEATURE_MAP_MEM_FAILED, 15, "Failed to allocate feature map memory.");
GE_ERRORNO_EXECUTOR(GE_EXEC_ALLOC_WEIGHT_MEM_FAILED, 16, "Failed to allocate weight memory.");
GE_ERRORNO_EXECUTOR(GE_EXEC_ALLOC_VAR_MEM_FAILED, 17, "Failed to allocate variable memory.");
GE_ERRORNO_EXECUTOR(GE_AIPP_NOT_EXIST, 18, "GE AIPP is not exist.");
GE_ERRORNO_EXECUTOR(GE_DYNAMIC_AIPP_NOT_SUPPORT_QUERY, 19, "GE Dynamic AIPP is not support to query temporarily.");
GE_ERRORNO_EXECUTOR(GE_EXEC_ALLOC_P2P_MEM_FAILED, 20, "Failed to allocate P2P memory");

// Generator module error code definition
GE_ERRORNO_GENERATOR(GE_GENERATOR_GRAPH_MANAGER_INIT_FAILED, 1, "Graph manager initialize failed.");
GE_ERRORNO_GENERATOR(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, 2, "Graph manager add graph failed.");
GE_ERRORNO_GENERATOR(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED, 3, "Graph manager build graph failed.");
GE_ERRORNO_GENERATOR(GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED, 4, "Graph manager finalize failed.");
GE_ERRORNO_GENERATOR(GE_GENERATOR_GRAPH_MANAGER_SAVE_MODEL_FAILED, 5, "Graph manager save model failed.");

static inline Status TransRtErrorCode(const int32_t error_code) {
  return static_cast<Status>(error_code);
}
#define RT_ERROR_TO_GE_STATUS(RT_ERROR) TransRtErrorCode(RT_ERROR)
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_GE_INNER_ERROR_CODES_H_
