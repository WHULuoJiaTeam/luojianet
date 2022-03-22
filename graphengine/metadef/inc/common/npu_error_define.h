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

#ifndef INC_COMMON_NPU_ERROR_DEFINE_H_
#define INC_COMMON_NPU_ERROR_DEFINE_H_

typedef enum tagHiAiNpuLocal {
  HIAI_HOST = 1,
  HIAI_DEVICE = 2,
} HiAiNpuLocal;

typedef enum tagHiAiNpuCodeType {
  ERROR_CODE = 1,
  EXCEPTION_CODE = 2,
} HiAiNpuCodeType;

typedef enum tagHiAiNpuErrLevel {
  NONE_LEVEL = 0,
  SUGGESTION_LEVEL = 1,
  NORMAL_LEVEL = 2,
  SERIOUS_LEVEL = 3,
  CRITICAL_ERROR = 4,
} HiAiNpuErrLevel;

typedef enum tagHiAiNpuModuleId {
  HIAI_DRIVER = 1,
  HIAI_CTRLCPU = 2,
  HIAI_TS = 3,
  HIAI_RUNTIME = 4,
  HIAI_AICPU = 5,
  HIAI_CCE = 6,
  HIAI_TVM = 7,
  HIAI_FRAMEWORK = 8,
  HiAI_ENGINE = 9,
  HIAI_DVPP = 10,
  HIAI_AIPP = 11,
  HIAI_LOWPOWER = 12,
  HIAI_MDC = 13,
  HIAI_COMPILE = 14,
  HIAI_TOOLCHIAN = 15,
  HIAI_ALG = 16,
  HIAI_PROFILING = 17,
  HIAI_HCCL = 18,
  HIAI_SIMULATION = 19,
  HIAI_BIOS = 20,
  HIAI_SEC = 21,
  HIAI_TINY = 22,
  HIAI_DP = 23,
} HiAiNpuModuleId;

/* bit 31-bit30 to be hiai local */
#define HIAI_NPULOCAL_MASK 0xC0000000
#define SHIFT_LOCAL_MASK 30
#define HIAI_NPULOCAL_VAL_MASK 0x3
/* bit 29 -bit28 to be hiai aicpu code type */
#define HIAI_CODE_TYPE_MASK 0x30000000
#define SHIFT_CODE_MASK 28
#define HIAI_CODE_TYPE_VAL_MASK 0x3
/* bit 27 -bit25 to be hiai error level */
#define HIAI_ERROR_LEVEL_MASK 0x0E000000
#define SHIFT_ERROR_LVL_MASK 25
#define HIAI_ERROR_LEVEL_VAL_MASK 0x7
/* bit 24 -bit17 to be hiai mod */
#define HIAI_MODE_ID_MASK 0x01FE0000
#define SHIFT_MODE_MASK 17
#define HIAI_MODE_ID_VAL_MASK 0xFF

#define HIAI_NPU_LOC_BIT(a) \
  (HIAI_NPULOCAL_MASK & ((unsigned int)((HiAiNpuLocal)(a)) & HIAI_NPULOCAL_VAL_MASK) << SHIFT_LOCAL_MASK)
#define HIAI_NPU_CODE_TYPE_BIT(a) \
  (HIAI_CODE_TYPE_MASK & ((unsigned int)((HiAiNpuCodeType)(a)) & HIAI_CODE_TYPE_VAL_MASK) << SHIFT_CODE_MASK)
#define HIAI_NPU_ERR_LEV_BIT(a) \
  (HIAI_ERROR_LEVEL_MASK & ((unsigned int)((HiAiNpuErrLevel)(a)) & HIAI_ERROR_LEVEL_VAL_MASK) << SHIFT_ERROR_LVL_MASK)
#define HIAI_NPU_MOD_ID_BIT(a) \
  (HIAI_MODE_ID_MASK & ((unsigned int)((HiAiNpuModuleId)(a)) & HIAI_MODE_ID_VAL_MASK) << SHIFT_MODE_MASK)

#define HIAI_NPU_ERR_CODE_HEAD(npuLocal, codeType, errLevel, moduleId)                              \
  (HIAI_NPU_LOC_BIT(npuLocal) + HIAI_NPU_CODE_TYPE_BIT(codeType) + HIAI_NPU_ERR_LEV_BIT(errLevel) + \
  HIAI_NPU_MOD_ID_BIT(moduleId))

#endif  // INC_COMMON_NPU_ERROR_DEFINE_H_
