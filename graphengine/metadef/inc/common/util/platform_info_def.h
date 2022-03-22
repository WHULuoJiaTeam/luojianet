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

#ifndef PLATFORM_INFO_DEF_H
#define PLATFORM_INFO_DEF_H

#include <map>
#include <string>
#include <vector>

using std::map;
using std::vector;
using std::string;

namespace fe {
enum MemoryType { DDR = 0, HBM };

enum L2Type { Cache = 0, Buff };

typedef struct tag_str_info {
  std::string aic_version;
  std::string ccec_aic_version;
  std::string ccec_aiv_version;
  std::string is_support_ai_cpu_compiler;
} StrInfo;

typedef struct tag_so_c_info {
  uint32_t ai_core_cnt;
  uint32_t vector_core_cnt;
  uint32_t ai_cpu_cnt;
  MemoryType memory_type;
  uint64_t memory_size;
  L2Type l2_type;
  uint64_t l2_size;
  uint32_t l2PageNum;
} SoCInfo;

typedef struct tag_ai_core_spec {
  double cube_freq;
  uint64_t cube_m_size;
  uint64_t cube_n_size;
  uint64_t cube_k_size;
  uint64_t vec_calc_size;
  uint64_t l0_a_size;
  uint64_t l0_b_size;
  uint64_t l0_c_size;
  uint64_t l1_size;
  uint64_t smask_buffer;
  uint64_t ub_size;
  uint64_t ubblock_size;
  uint64_t ubbank_size;
  uint64_t ubbank_num;
  uint64_t ubburst_in_one_block;
  uint64_t ubbank_group_num;
  uint32_t unzip_engines;
  uint32_t unzip_max_ratios;
  uint32_t unzip_channels;
  uint8_t unzip_is_tight;
  uint8_t cube_vector_split;
} AiCoreSpec;

typedef struct tag_ai_core_memory_rates {
  double ddr_rate;
  double ddr_read_rate;
  double ddr_write_rate;
  double l2_rate;
  double l2_read_rate;
  double l2_write_rate;
  double l1_to_l0_a_rate;
  double l1_to_l0_b_rate;
  double l1_to_ub_rate;
  double l0_c_to_ub_rate;
  double ub_to_l2_rate;
  double ub_to_ddr_rate;
  double ub_to_l1_rate;
} AiCoreMemoryRates;

typedef struct tag_vector_core_spec {
  double vec_freq;
  uint64_t vec_calc_size;
  uint64_t smask_buffer;
  uint64_t ub_size;
  uint64_t ubblock_size;
  uint64_t ubbank_size;
  uint64_t ubbank_num;
  uint64_t ubburst_in_one_block;
  uint64_t ubbank_group_num;
  uint64_t vector_reg_size;
  uint64_t predicate_reg_size;
  uint64_t address_reg_size;
  uint64_t alignment_reg_size;
} VectorCoreSpec;

typedef struct tag_vector_core_memory_rates {
  double ddr_rate;
  double ddr_read_rate;
  double ddr_write_rate;
  double l2_rate;
  double l2_read_rate;
  double l2_write_rate;
  double ub_to_l2_rate;
  double ub_to_ddr_rate;
} VectorCoreMemoryRates;

typedef struct tag_cpu_cache {
  uint32_t AICPUSyncBySW;
  uint32_t TSCPUSyncBySW;
} CPUCache;

typedef struct tag_platform_info {
  StrInfo str_info;
  SoCInfo soc_info;
  AiCoreSpec ai_core_spec;
  AiCoreMemoryRates ai_core_memory_rates;
  std::map<std::string, std::vector<std::string>> ai_core_intrinsic_dtype_map;
  VectorCoreSpec vector_core_spec;
  VectorCoreMemoryRates vector_core_memory_rates;
  CPUCache cpucache;
  std::map<std::string, std::vector<std::string>> vector_core_intrinsic_dtype_map;
} PlatformInfo;

typedef struct tag_optional_info {
  std::string soc_version;
  std::string core_type;
  uint32_t ai_core_num;
  std::string l1_fusion_flag;
} OptionalInfo;
}  // namespace fe
#endif
