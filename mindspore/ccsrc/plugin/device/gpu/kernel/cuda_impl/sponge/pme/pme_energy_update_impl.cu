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
 *  PMEEnergyUpdate. This is an experimental interface that is subject to change and/or deletion.
 */
#include "plugin/device/gpu/kernel/cuda_impl/sponge/pme/pme_energy_update_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/pme/pme_common.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/common_sponge.cuh"

__global__ void PME_Energy_Reciprocal_update(const int element_number, const cufftComplex *FQ, const float *BC,
                                             float *sum) {
  if (threadIdx.x == 0) {
    sum[0] = 0.;
  }
  __syncthreads();
  float lin = 0.0;
  cufftComplex FQ_i;
  for (int i = threadIdx.x; i < element_number; i = i + blockDim.x) {
    FQ_i = FQ[i];
    lin = lin + (FQ_i.x * FQ_i.x + FQ_i.y * FQ_i.y) * BC[i];
  }
  atomicAdd(sum, lin);
}

void PMEEnergyUpdate(int fftx, int ffty, int fftz, int atom_numbers, float beta, float *PME_BC, int *pme_uxyz,
                     float *pme_frxyz, float *PME_Q, float *pme_fq, int *PME_atom_near, int *pme_kxyz,
                     const int *uint_crd_f, const float *charge, int *nl_atom_numbers, int *nl_atom_serial, int *nl,
                     const float *scaler_f, const int *excluded_list_start, const int *excluded_list,
                     const int *excluded_atom_numbers, float *d_reciprocal_ene, float *d_self_ene, float *d_direct_ene,
                     float *d_correction_ene, dim3 thread_PME, int PME_Nin, int PME_Nfft, int PME_Nall,
                     const cufftHandle &PME_plan_r2c, const cufftHandle &PME_plan_c2r, float *neutralizing_factor,
                     float *charge_sum, int max_neighbor_numbers, cudaStream_t stream) {
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));
  // int max_neighbor_numbers = 800;
  NEIGHBOR_LIST *nl_a = reinterpret_cast<NEIGHBOR_LIST *>(nl);
  construct_neighbor_list_kernel<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, max_neighbor_numbers, nl_atom_numbers, nl_atom_serial, nl_a);

  UNSIGNED_INT_VECTOR *PME_uxyz = reinterpret_cast<UNSIGNED_INT_VECTOR *>(pme_uxyz);
  UNSIGNED_INT_VECTOR *PME_kxyz = reinterpret_cast<UNSIGNED_INT_VECTOR *>(pme_kxyz);
  VECTOR *PME_frxyz = reinterpret_cast<VECTOR *>(pme_frxyz);
  cufftComplex *PME_FQ = reinterpret_cast<cufftComplex *>(pme_fq);

  Reset_List<<<3 * atom_numbers / 32 + 1, 32, 0, stream>>>(3 * atom_numbers, reinterpret_cast<int *>(PME_uxyz),
                                                           1 << 30);
  PME_Atom_Near<<<atom_numbers / 32 + 1, 32, 0, stream>>>(
    uint_crd, PME_atom_near, PME_Nin, periodic_factor_inverse * fftx, periodic_factor_inverse * ffty,
    periodic_factor_inverse * fftz, atom_numbers, fftx, ffty, fftz, PME_kxyz, PME_uxyz, PME_frxyz);

  Reset_List<<<PME_Nall / 1024 + 1, 1024, 0, stream>>>(PME_Nall, PME_Q, 0);

  PME_Q_Spread<<<atom_numbers / thread_PME.x + 1, thread_PME, 0, stream>>>(PME_atom_near, charge, PME_frxyz, PME_Q,
                                                                           PME_kxyz, atom_numbers);

  cufftExecR2C(PME_plan_r2c, reinterpret_cast<float *>(PME_Q), reinterpret_cast<cufftComplex *>(PME_FQ));

  PME_Energy_Reciprocal_update<<<1, 1024, 0, stream>>>(PME_Nfft, PME_FQ, PME_BC, d_reciprocal_ene);

  PME_Energy_Product<<<1, 1024, 0, stream>>>(atom_numbers, charge, charge, d_self_ene);
  Scale_List<<<1, 1, 0, stream>>>(1, d_self_ene, -beta / sqrtf(PI));

  Sum_Of_List<<<1, 1024>>>(atom_numbers, charge, charge_sum);
  device_add<<<1, 1>>>(d_self_ene, neutralizing_factor, charge_sum);

  Reset_List<<<1, 1, 0, stream>>>(1, d_direct_ene, 0.0);
  PME_Direct_Energy<<<atom_numbers / thread_PME.x + 1, thread_PME, 0, stream>>>(
    atom_numbers, nl_a, uint_crd, scaler, charge, beta, cutoff * cutoff, d_direct_ene);

  Reset_List<<<1, 1, 0, stream>>>(1, d_correction_ene, 0.0);
  PME_Excluded_Energy_Correction<<<atom_numbers / 32 + 1, 32, 0, stream>>>(
    atom_numbers, uint_crd, scaler, charge, beta, sqrtf(PI), excluded_list_start, excluded_list, excluded_atom_numbers,
    d_correction_ene);
  return;
}
