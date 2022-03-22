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

#ifndef __INC_LLT_RUNTIME_STUB_H
#define __INC_LLT_RUNTIME_STUB_H

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
void rtStubTearDown();

#define RTS_STUB_SETUP()    \
do {                        \
  rtStubTearDown();         \
} while (0)

#define RTS_STUB_TEARDOWN() \
do {                        \
  rtStubTearDown();         \
} while (0)

#define RTS_STUB_RETURN_VALUE(FUNC, TYPE, VALUE)                          \
do {                                                                      \
  g_Stub_##FUNC##_RETURN.emplace(g_Stub_##FUNC##_RETURN.begin(), VALUE);  \
} while (0)

#define RTS_STUB_OUTBOUND_VALUE(FUNC, TYPE, NAME, VALUE)                          \
do {                                                                              \
  g_Stub_##FUNC##_OUT_##NAME.emplace(g_Stub_##FUNC##_OUT_##NAME.begin(), VALUE);  \
} while (0)


#define RTS_STUB_RETURN_EXTERN(FUNC, TYPE) extern std::vector<TYPE> g_Stub_##FUNC##_RETURN;
#define RTS_STUB_OUTBOUND_EXTERN(FUNC, TYPE, NAME) extern std::vector<TYPE> g_Stub_##FUNC##_OUT_##NAME;

RTS_STUB_RETURN_EXTERN(rtGetDevice, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtGetDevice, int32_t, device)

RTS_STUB_RETURN_EXTERN(rtGetDeviceCapability, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtGetDeviceCapability, int32_t, value);

RTS_STUB_RETURN_EXTERN(rtStreamWaitEvent, rtError_t);

RTS_STUB_RETURN_EXTERN(rtEventReset, rtError_t);

RTS_STUB_RETURN_EXTERN(rtEventCreate, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtEventCreate, rtEvent_t, event);

RTS_STUB_RETURN_EXTERN(rtGetEventID, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtEventCreate, uint32_t, event_id);

#ifdef __cplusplus
}
#endif
#endif // __INC_LLT_RUNTIME_STUB_H
