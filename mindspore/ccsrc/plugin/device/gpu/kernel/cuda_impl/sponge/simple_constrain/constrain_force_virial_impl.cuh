/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
/**
 * Note:
 *  ConstrainForce. This is an experimental interface that is subject to change and/or deletion.
 */

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_SIMPLE_CONSTRAIN_CONSTRAIN_FORCE_VIRIAL_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_SIMPLE_CONSTRAIN_CONSTRAIN_FORCE_VIRIAL_IMPL_H_

#include <curand_kernel.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

void constrain_force_cycle_update(int atom_numbers, int constrain_pair_numbers, const unsigned int *uint_crd_f,
                                  const float *scaler_f, float *constrain_pair_f, const float *pair_dr_f,
                                  const int *atom_i_serials, const int *atom_j_serials, const float *constant_rs,
                                  const float *constrain_ks, float *test_frc_f, cudaStream_t stream);

void constrain_force_cycle_with_virial_update(int atom_numbers, int constrain_pair_numbers,
                                              const unsigned int *uint_crd_f, const float *scaler_f,
                                              float *constrain_pair_f, const float *pair_dr_f,
                                              const int *atom_i_serials, const int *atom_j_serials,
                                              const float *constant_rs, const float *constrain_ks, float *test_frc_f,
                                              float *d_atom_virial, cudaStream_t stream);

void refresh_uint_crd_update(int atom_numbers, float half_exp_gamma_plus_half, const float *crd_f,
                             const float *quarter_crd_to_uint_crd_cof_f, float *test_frc_f, const float *mass_inverse,
                             unsigned int *uint_crd_f, cudaStream_t stream);

void set_zero_force_with_virial(int atom_numbers, int constrain_pair_numbers, float *test_frc_f, float *d_atom_virial,
                                cudaStream_t stream);

void set_zero(int numbers, float *x, cudaStream_t stream);

#endif
