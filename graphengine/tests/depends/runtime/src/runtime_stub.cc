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

#include <cce/dnn.h>
#include <securec.h>
#include "runtime_stub.h"
#include "runtime/rt.h"

#define ADD_STUB_RETURN_VALUE(FUNC, TYPE) std::vector<TYPE> g_Stub_##FUNC##_RETURN

#define GET_STUB_RETURN_VALUE(FUNC, TYPE, DEFAULT) ({   \
  TYPE result = DEFAULT;                                \
  if (!g_Stub_##FUNC##_RETURN.empty()) {                \
    result = g_Stub_##FUNC##_RETURN.back();             \
    g_Stub_##FUNC##_RETURN.pop_back();                  \
  }                                                     \
  result;                                               \
})

#define DEL_STUB_RETURN_VALUE(FUNC, TYPE)           \
do {                                                \
  extern std::vector<TYPE> g_Stub_##FUNC##_RETURN;  \
  g_Stub_##FUNC##_RETURN.clear();                   \
} while (0)


#define ADD_STUB_OUTBOUND_VALUE(FUNC, TYPE, NAME) std::vector<TYPE> g_Stub_##FUNC##_OUT_##NAME

#define GET_STUB_OUTBOUND_VALUE(FUNC, TYPE, NAME, DEFAULT) ({ \
  TYPE value;                                                 \
  if (!g_Stub_##FUNC##_OUT_##NAME.empty()) {                  \
    value = g_Stub_##FUNC##_OUT_##NAME.back();                \
    g_Stub_##FUNC##_OUT_##NAME.pop_back();                    \
  } else {                                                    \
    value = DEFAULT;                                          \
  }                                                           \
  value;                                                      \
})

#define DEL_STUB_OUTBOUND_VALUE(FUNC, TYPE, NAME)       \
do {                                                    \
  extern std::vector<TYPE> g_Stub_##FUNC##_OUT_##NAME;  \
  g_Stub_##FUNC##_OUT_##NAME.clear();                   \
} while (0)

#ifdef __cplusplus
extern "C" {
#endif
#define EVENT_LENTH 10

void rtStubTearDown() {
  DEL_STUB_RETURN_VALUE(rtGetDevice, rtError_t);
  DEL_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t);
  DEL_STUB_RETURN_VALUE(rtStreamWaitEvent, rtError_t);
  DEL_STUB_RETURN_VALUE(rtEventReset, rtError_t);
  DEL_STUB_RETURN_VALUE(rtEventCreate, rtError_t);
  DEL_STUB_RETURN_VALUE(rtGetEventID, rtError_t);
}

ADD_STUB_RETURN_VALUE(rtGetDevice, rtError_t);
rtError_t rtGetDevice(int32_t *device) {
  return GET_STUB_RETURN_VALUE(rtGetDevice, rtError_t, RT_ERROR_NONE);
}

ADD_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t);
ADD_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value);
rtError_t rtGetDeviceCapability(int32_t device, int32_t moduleType, int32_t featureType, int32_t *value) {
  *value = GET_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_SUPPORT);
  return GET_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
}

ADD_STUB_RETURN_VALUE(rtStreamWaitEvent, rtError_t);
rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event) {
  return GET_STUB_RETURN_VALUE(rtStreamWaitEvent, rtError_t, RT_ERROR_NONE);
}

ADD_STUB_RETURN_VALUE(rtEventReset, rtError_t);
rtError_t rtEventReset(rtEvent_t event, rtStream_t stream) {
  return GET_STUB_RETURN_VALUE(rtEventReset, rtError_t, RT_ERROR_NONE);
}

ADD_STUB_RETURN_VALUE(rtEventCreate, rtError_t);
rtError_t rtEventCreate(rtEvent_t *event) {
  *event = new int[EVENT_LENTH];
  return GET_STUB_RETURN_VALUE(rtEventCreate, rtError_t, RT_ERROR_NONE);
}

ADD_STUB_RETURN_VALUE(rtGetEventID, rtError_t);
rtError_t rtGetEventID(rtEvent_t event, uint32_t *event_id) {
  *event_id = 0;
  return GET_STUB_RETURN_VALUE(rtEventCreate, rtError_t, RT_ERROR_NONE);
}

rtError_t rtCtxSetCurrent(rtContext_t ctx) { return RT_ERROR_NONE; }

rtError_t rtGetStreamId(rtStream_t stream, int32_t *stream_id) {
  *stream_id = 0;
  return RT_ERROR_NONE;
}

rtError_t rtCtxGetCurrent(rtContext_t *ctx) {
  uintptr_t x = 1;
  *ctx = (rtContext_t *)x;
  return RT_ERROR_NONE;
}

rtError_t rtCtxSetDryRun(rtContext_t ctx, rtDryRunFlag_t enable, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtEventGetTimeStamp(uint64_t *time, rtEvent_t event) {
  *time = 12345;
  return RT_ERROR_NONE;
}

rtError_t rtEventCreateWithFlag(rtEvent_t *event, uint32_t flag) {
  return rtEventCreate(event);
}

rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtEventSynchronize(rtEvent_t event) { return RT_ERROR_NONE; }

rtError_t rtEventDestroy(rtEvent_t event) {
  delete[](int *) event;
  return RT_ERROR_NONE;
}

rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type) {
  *dev_ptr = new uint8_t[size];
  return RT_ERROR_NONE;
}

rtError_t rtMemset(void *dev_ptr, uint64_t dest_max, uint32_t value, uint64_t count) { return RT_ERROR_NONE; }

rtError_t rtFree(void *dev_ptr) {
  delete[](uint8_t *) dev_ptr;
  return RT_ERROR_NONE;
}

rtError_t rtMallocHost(void **host_ptr, uint64_t size) {
  *host_ptr = new uint8_t[size];
  return RT_ERROR_NONE;
}

rtError_t rtFreeHost(void *host_ptr) {
  delete[](uint8_t *) host_ptr;
  return RT_ERROR_NONE;
}

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority) {
  *stream = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtStreamDestroy(rtStream_t stream) {
  if (stream != nullptr) {
    delete (uint32_t *)stream;
  }
  return RT_ERROR_NONE;
}

rtError_t rtSetDevice(int32_t device) { return RT_ERROR_NONE; }

rtError_t rtStreamSynchronize(rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) {
  if (dst != nullptr && src != nullptr) {
    memcpy_s(dst, dest_max, src, count);
  }
  return RT_ERROR_NONE;
}
rtError_t rtMemcpyAsync(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind,
                        rtStream_t stream) {
  if (dst != nullptr && src != nullptr) {
    memcpy_s(dst, dest_max, src, count);
  }
  return RT_ERROR_NONE;
}

rtError_t rtSetTSDevice(uint32_t tsId) {
  return RT_ERROR_NONE;
}

rtError_t rtGetDeviceCount(int32_t *count) {
  *count = 1;
  return RT_ERROR_NONE;
}

rtError_t rtDeviceReset(int32_t device) { return RT_ERROR_NONE; }

rtError_t rtEventElapsedTime(float *time, rtEvent_t start, rtEvent_t end) {
  *time = 10.0f;
  return RT_ERROR_NONE;
}

rtError_t rtFunctionRegister(void *bin_handle, const void *stub_func, const char *stub_name, const void *dev_func,
                             uint32_t func_mode) {
  return RT_ERROR_NONE;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle) { return RT_ERROR_NONE; }

rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle) { return RT_ERROR_NONE; }

rtError_t rtKernelConfigTransArg(const void *ptr, uint64_t size, uint32_t flag, void **arg) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunchWithHandle(void *handle, const void *devFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                   rtSmDesc_t *smDesc, rtStream_t stream, const void *kernelInfo) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelLaunch(const void *stub_func, uint32_t block_dim, void *args, uint32_t args_size, rtSmDesc_t *sm_desc,
                         rtStream_t stream) {
  return RT_ERROR_NONE;
}
rtError_t rtSetupArgument(const void *arg, uint32_t size, uint32_t offset) { return RT_ERROR_NONE; }
rtError_t rtLaunch(const void *stub_func) { return RT_ERROR_NONE; }
rtError_t rtDevBinaryUnRegister(void *handle) { return RT_ERROR_NONE; }
rtError_t rtConfigureCall(uint32_t num_blocks, rtSmDesc_t *sm_desc, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtSetProfDir(char *prof_dir) { return RT_ERROR_NONE; }

rtError_t rtSetProfDirEx(const char *profDir, const char *address, const char *jobCtx) { return RT_ERROR_NONE; }

rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t *aicore_memory_size) { return RT_ERROR_NONE; }

rtError_t rtSetKernelReportCallback(rtKernelReportCallback callback) {
  rtKernelInfo rt_kernel_info = {0};
  rt_kernel_info.arg_size = 12;
  rt_kernel_info.task_offset = 100;
  rt_kernel_info.arg = (void *)100;
  rt_kernel_info.module_addr = (void *)100;
  rt_kernel_info.module_size = 100;

  rtStream_t stream = nullptr;
  callback(stream, &rt_kernel_info);
  return RT_ERROR_NONE;
}

rtError_t rtMemAdvise(void *ptr, uint64_t size, uint32_t advise) { return RT_ERROR_NONE; }

/// @ingroup rt_kernel
/// @brief start fusion kernels.
/// @param [in] stream   stream for fusion kernels
/// @return RT_ERROR_NONE for ok, errno for failed
rtError_t rtKernelFusionStart(rtStream_t stream) { return RT_ERROR_NONE; }

/// @ingroup rt_kernel
/// @brief end fusion kernels.
/// @param [in] stream   stream for fusion kernels
/// @return RT_ERROR_NONE for ok, errno for failed
rtError_t rtKernelFusionEnd(rtStream_t stream) { return RT_ERROR_NONE; }
rtError_t rtMemGetInfo(size_t *free, size_t *total) {
  *free = 512UL * 1024UL * 1024UL;
  *total = 1024UL * 1024UL * 1024UL;
  return RT_ERROR_NONE;
}

rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) {
  *free = 512UL * 1024UL * 1024UL;
  *total = 1024UL * 1024UL * 1024UL;
  return RT_ERROR_NONE;
}

rtError_t rtMemAllocManaged(void **ptr, uint64_t size, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtMemFreeManaged(void *ptr) { return RT_ERROR_NONE; }

rtError_t rtMetadataRegister(void *handle, const char *meta_data) { return RT_ERROR_NONE; }
rtError_t rtSetTaskGenCallback(rtTaskGenCallback callback) { return RT_ERROR_NONE; }

rtError_t rtModelCreate(rtModel_t *model, uint32_t flag) {
  *model = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtModelDestroy(rtModel_t model) {
  uint32_t *stub = static_cast<uint32_t *>(model);
  delete stub;
  return RT_ERROR_NONE;
}

rtError_t rtModelBindStream(rtModel_t model, rtStream_t stream, uint32_t flag) { return RT_ERROR_NONE; }
rtError_t rtModelUnbindStream(rtModel_t model, rtStream_t stream) { return RT_ERROR_NONE; }
rtError_t rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtGetFunctionByName(const char *stub_name, void **stub_func) {
  *(char **)stub_func = "func";
  return RT_ERROR_NONE;
}
rtError_t rtGetAddrByFun(const void *stubFunc, void **addr) {
  *(char **)addr = "dev_func";
  return RT_ERROR_NONE;
}
rtError_t rtQueryFunctionRegistered(const char *stub_name) { return RT_ERROR_NONE; }

rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunchEx(void *args, uint32_t args_size, uint32_t flags, rtStream_t stream_) { return RT_ERROR_NONE; }

rtError_t rtCpuKernelLaunch(const void *so_name, const void *kernel_name, uint32_t block_dim, const void *args,
                            uint32_t args_size, rtSmDesc_t *sm_desc, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtModelGetTaskId(void *handle, uint32_t *task_id, uint32_t *stream_id) {
  *task_id = 0;
  *stream_id = 0;
  return RT_ERROR_NONE;
}
rtError_t rtEndGraph(rtModel_t model, rtStream_t stream) { return RT_ERROR_NONE; }
rtError_t rtEndGraphEx(rtModel_t model, rtStream_t stream, uint32_t flags)
{
  return RT_ERROR_NONE;
}
rtError_t rtProfilerStop(uint64_t profConfig, int32_t numsDev, uint32_t *deviceList) {
  return RT_ERROR_NONE;
}

rtError_t rtSetDvfsProfile(DvfsProfileMode mode) { return RT_ERROR_NONE; }

rtError_t rtUnsetDvfsProfile() { return RT_ERROR_NONE; }

rtError_t rtGetDvfsProfile(DvfsProfileMode *pmode) { return RT_ERROR_NONE; }

rtError_t rtCtxDestroy(rtContext_t ctx) { return RT_ERROR_NONE; }

rtError_t rtProfilerInit(const char *prof_dir, const char *address, const char *job_ctx) { return RT_ERROR_NONE; }

rtError_t rtProfilerStart(uint64_t profConfig, int32_t numsDev, uint32_t *deviceList) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelCreate(rtLabel_t *label) {
  *label = new uint64_t;
  return RT_ERROR_NONE;
}

rtError_t rtLabelCreateEx(rtLabel_t *label, rtStream_t stream) {
  *label = new uint64_t;
  return RT_ERROR_NONE;
}

rtError_t rtLabelCreateV2(rtLabel_t *label, rtModel_t model) {
  *label = new uint64_t;
  return RT_ERROR_NONE;
}

rtError_t rtLabelCreateExV2(rtLabel_t *label, rtModel_t model, rtStream_t stream) {
  *label = new uint64_t;
  return RT_ERROR_NONE;
}

rtError_t rtLabelListCpy(rtLabel_t *label, uint32_t labelNumber, void *dst, uint32_t dstMax) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelDestroy(rtLabel_t label) {
  uint64_t *stub = static_cast<uint64_t *>(label);
  delete stub;
  return RT_ERROR_NONE;
}

rtError_t rtLabelSet(rtLabel_t label, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtLabelSwitch(void *ptr, rtCondition_t condition, uint32_t value, rtLabel_t true_label, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelSwitchByIndex(void *ptr, uint32_t max, void *labelInfoPtr, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelGoto(rtLabel_t label, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtLabelGotoEx(rtLabel_t label, rtStream_t stream) {
  return RT_ERROR_NONE;
}


rtError_t rtInvalidCache(void *base, size_t len) {
  return RT_ERROR_NONE;
}

rtError_t rtModelLoadComplete(rtModel_t model) { return RT_ERROR_NONE; }

rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags) {
  *stream = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtFlushCache(void *base, size_t len) {
  return RT_ERROR_NONE;
}

rtError_t rtProfilerTrace(uint64_t id, bool notify, uint32_t flags, rtStream_t stream_) { return RT_ERROR_NONE; }

rtError_t rtProfilerTraceEx(uint64_t id, uint64_t modelId, uint16_t tagId, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtMemSetRC(const void *dev_ptr, uint64_t size, uint32_t read_count) { return RT_ERROR_NONE; }

rtError_t rtStreamSwitch(void *ptr, rtCondition_t condition, int64_t value, rtStream_t true_stream, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamSwitchEx(void *ptr, rtCondition_t condition, void *value_ptr, rtStream_t true_stream,
                           rtStream_t stream, rtSwitchDataType_t data_type) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamActive(rtStream_t active_stream, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtDatadumpInfoLoad(const void *dump_info, uint32_t length) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunchWithFlag(const void *stub_func, uint32_t block_dim, void *args, uint32_t args_size,
                                 rtSmDesc_t *sm_desc, rtStream_t stream_, uint32_t flags) {
  return RT_ERROR_NONE;
}

rtError_t rtCpuKernelLaunchWithFlag(const void *so_name, const void *kernel_name, uint32_t core_dim, const void *args,
                                    uint32_t args_size, rtL2Ctrl_t *l2ctrl, rtStream_t stream_, uint32_t flags) {
  return RT_ERROR_NONE;
}

rtError_t rtModelGetId(rtModel_t model, uint32_t *modelId)
{
  return RT_ERROR_NONE;
}

rtError_t rtModelBindQueue(rtModel_t model, uint32_t queueId, rtModelQueueFlag_t flag)
{
  return RT_ERROR_NONE;
}

rtError_t rtSetSocVersion(const char *version)
{
  return RT_ERROR_NONE;
}

rtError_t rtGetSocVersion(char *version, const uint32_t maxLen)
{
  return RT_ERROR_NONE;
}

rtError_t rtGetAiCoreCount(uint32_t *aiCoreCnt)
{
  return RT_ERROR_NONE;
}

rtError_t rtSetTaskFailCallback(rtTaskFailCallback callback)
{
  return RT_ERROR_NONE;
}

rtError_t rtMallocHostSharedMemory(rtMallocHostSharedMemoryIn *in,
		                               rtMallocHostSharedMemoryOut *out)
{
  out->ptr = new uint8_t[in->size];
  out->devPtr = new uint8_t[in->size];
  return RT_ERROR_NONE;
}

rtError_t rtFreeHostSharedMemory(rtFreeHostSharedMemoryIn *in)
{
  delete[] (uint8_t*)in->ptr;
  delete[] (uint8_t*)in->devPtr;
  return RT_ERROR_NONE;
}

rtError_t rtGetAicpuDeploy(rtAicpuDeployType_t *deplyType)
{
  return RT_ERROR_NONE;
}

rtError_t rtDebugRegister(rtModel_t model, uint32_t flag, const void *addr, uint32_t *streamId, uint32_t *taskId)
{
  return RT_ERROR_NONE;
}

rtError_t rtDebugUnRegister(rtModel_t model)
{
  return RT_ERROR_NONE;
}

rtError_t rtDumpAddrSet(rtModel_t model, void *addr, uint32_t dumpSize, uint32_t flag)
{
  return RT_ERROR_NONE;
}

rtError_t rtSetCtxINFMode(bool mode)
{
  return RT_ERROR_NONE;
}

rtError_t rtGetRtCapability(rtFeatureType_t featureType, int32_t featureInfo, int64_t *value)
{
  return RT_ERROR_NONE;
}

rtError_t rtGetMaxStreamAndTask(uint32_t streamType, uint32_t *maxStrCount, uint32_t *maxTaskCount)
{
  *maxStrCount = 1024;
  *maxTaskCount = 1024;
  return RT_ERROR_NONE;
}

rtError_t rtModelExit(rtModel_t model, rtStream_t stream)
{
 return RT_ERROR_NONE;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId)
{
 return RT_ERROR_NONE;
}

rtError_t rtDebugRegisterForStream(rtStream_t stream, uint32_t flag, const void *addr, uint32_t *streamId, uint32_t *taskId) {
  return RT_ERROR_NONE;
}

rtError_t rtDebugUnRegisterForStream(rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtFftsTaskLaunch(rtFftsTaskInfo_t *fftsTaskInfo, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchFwk(const char *opName, void *args, uint32_t argSize, uint32_t flags, rtStream_t rtStream) {
  return RT_ERROR_NONE;
}

rtError_t rtAicpuKernelLaunchWithFlag(const rtKernelLaunchNames_t *launchNames, uint32_t blockDim, const void *args,
                                      uint32_t argSize, rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags) {
  return RT_ERROR_NONE;
}

rtError_t rtAicpuKernelLaunch(const rtKernelLaunchNames_t *launchNames, uint32_t blockDim, const void *args,
                              uint32_t argSize, rtSmDesc_t *smDesc, rtStream_t stream) {
  return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif
