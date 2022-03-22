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

#ifndef LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_CPU_MPI_MPI_EXPORT_H_
#define LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_CPU_MPI_MPI_EXPORT_H_
#include <vector>
#include <string>
#ifndef FUNC_EXPORT
#define FUNC_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
FUNC_EXPORT int GetMPIRankId();
FUNC_EXPORT int GetMPIRankSize();
FUNC_EXPORT bool MPIReduceScatter(const float *input, float *output, const std::vector<int> &ranks_group,
                                  size_t data_num, const std::string &op_type);
FUNC_EXPORT bool MPIReduceScatterOverwriteInput(float *input, const std::vector<int> &ranks_group, size_t in_data_num,
                                                size_t output_size, const std::string &op_type, float *output);
FUNC_EXPORT bool MPIAllGather(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num);
}
#endif  // LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_CPU_MPI_MPI_EXPORT_H_
