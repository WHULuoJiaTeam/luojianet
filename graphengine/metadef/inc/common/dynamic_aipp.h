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

#ifndef INC_COMMON_DYNAMIC_AIPP_H_
#define INC_COMMON_DYNAMIC_AIPP_H_

#include <cstdint>

/**
* @ingroup dnn
* @brief struct define of dynamic aipp batch parameter.
*/
typedef struct tagAippDynamicBatchPara {
  int8_t cropSwitch;     // crop switch
  int8_t scfSwitch;      // resize switch
  int8_t paddingSwitch;  // 0: unable padding
                         // 1: padding config value,sfr_filling_hblank_ch0 ~  sfr_filling_hblank_ch2
                         // 2: padding source picture data, single row/collumn copy
                         // 3: padding source picture data, block copy
                         // 4: padding source picture data, mirror copy
  int8_t rotateSwitch;   // rotate switch，0: non-ratate，
                         // 1: ratate 90° clockwise，2: ratate 180° clockwise，3: ratate 270° clockwise
  int8_t reserve[4];
  int32_t cropStartPosW;  // the start horizontal position of cropping
  int32_t cropStartPosH;  // the start vertical position of cropping
  int32_t cropSizeW;      // crop width
  int32_t cropSizeH;      // crop height

  int32_t scfInputSizeW;   // input width of scf
  int32_t scfInputSizeH;   // input height of scf
  int32_t scfOutputSizeW;  // output width of scf
  int32_t scfOutputSizeH;  // output height of scf

  int32_t paddingSizeTop;     // top padding size
  int32_t paddingSizeBottom;  // bottom padding size
  int32_t paddingSizeLeft;    // left padding size
  int32_t paddingSizeRight;   // right padding size

  int16_t dtcPixelMeanChn0;  // mean value of channel 0
  int16_t dtcPixelMeanChn1;  // mean value of channel 1
  int16_t dtcPixelMeanChn2;  // mean value of channel 2
  int16_t dtcPixelMeanChn3;  // mean value of channel 3

  uint16_t dtcPixelMinChn0;      // min value of channel 0
  uint16_t dtcPixelMinChn1;      // min value of channel 1
  uint16_t dtcPixelMinChn2;      // min value of channel 2
  uint16_t dtcPixelMinChn3;      // min value of channel 3
  uint16_t dtcPixelVarReciChn0;  // sfr_dtc_pixel_variance_reci_ch0
  uint16_t dtcPixelVarReciChn1;  // sfr_dtc_pixel_variance_reci_ch1
  uint16_t dtcPixelVarReciChn2;  // sfr_dtc_pixel_variance_reci_ch2
  uint16_t dtcPixelVarReciChn3;  // sfr_dtc_pixel_variance_reci_ch3

  int8_t reserve1[16];  // 32B assign, for ub copy
} kAippDynamicBatchPara;

/**
* @ingroup dnn
* @brief struct define of dynamic aipp parameter. lite:64+96*batchNum byte ; tiny:64+64*batchNum byte
*/
typedef struct tagAippDynamicPara {
  uint8_t inputFormat;    // input format：YUV420SP_U8/XRGB8888_U8/RGB888_U8
  int8_t cscSwitch;       // csc switch
  int8_t rbuvSwapSwitch;  // rb/ub swap switch
  int8_t axSwapSwitch;    // RGBA->ARGB, YUVA->AYUV swap switch
  int8_t batchNum;        // batch parameter number
  int8_t reserve1[3];
  int32_t srcImageSizeW;  // source image width
  int32_t srcImageSizeH;  // source image height
  int16_t cscMatrixR0C0;  // csc_matrix_r0_c0
  int16_t cscMatrixR0C1;  // csc_matrix_r0_c1
  int16_t cscMatrixR0C2;  // csc_matrix_r0_c2
  int16_t cscMatrixR1C0;  // csc_matrix_r1_c0
  int16_t cscMatrixR1C1;  // csc_matrix_r1_c1
  int16_t cscMatrixR1C2;  // csc_matrix_r1_c2
  int16_t cscMatrixR2C0;  // csc_matrix_r2_c0
  int16_t cscMatrixR2C1;  // csc_matrix_r2_c1
  int16_t cscMatrixR2C2;  // csc_matrix_r2_c2
  int16_t reserve2[3];
  uint8_t cscOutputBiasR0;  // output Bias for RGB to YUV, element of row 0, unsigned number
  uint8_t cscOutputBiasR1;  // output Bias for RGB to YUV, element of row 1, unsigned number
  uint8_t cscOutputBiasR2;  // output Bias for RGB to YUV, element of row 2, unsigned number
  uint8_t cscInputBiasR0;   // input Bias for YUV to RGB, element of row 0, unsigned number
  uint8_t cscInputBiasR1;   // input Bias for YUV to RGB, element of row 1, unsigned number
  uint8_t cscInputBiasR2;   // input Bias for YUV to RGB, element of row 2, unsigned number
  uint8_t reserve3[2];
  int8_t reserve4[16];  // 32B assign, for ub copy

  kAippDynamicBatchPara aippBatchPara;  // allow transfer several batch para.
} kAippDynamicPara;

#endif  // INC_COMMON_DYNAMIC_AIPP_H_
