/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/winograd_transform.h"
#include "nnacl/op_base.h"

void PrepareTransInput(const float *src_data, float *dst_data, int interval_x_s, int interval_x_e, int interval_y_s,
                       int interval_y_e, int real_c, const ConvParameter *conv_param) {
  int input_unit = conv_param->input_unit_;
  int in_channel = conv_param->input_channel_;
  int input_w = conv_param->input_w_;
#ifdef ENABLE_AVX
  int channel_tile = C8NUM;
#else
  int channel_tile = C4NUM;
#endif
  // clear tmp buffer
  if (interval_x_e - interval_x_s != input_unit || interval_y_e - interval_y_s != input_unit) {
    memset(dst_data, 0, input_unit * input_unit * channel_tile * (int)(sizeof(float)));
  }

  // get real input block with padding
  if (real_c == channel_tile) {
    for (int interval = interval_y_s; interval < interval_y_e; interval++) {
      int src_y_offset = (interval * input_w + interval_x_s) * in_channel;
      int dst_y_offset = interval * input_unit * channel_tile + interval_x_s * channel_tile;
      for (int j = 0; j < (interval_x_e - interval_x_s); j++) {
        int src_x_offset = src_y_offset + j * in_channel;
        int dst_x_offset = dst_y_offset + j * channel_tile;
        const float *src_addr = src_data + src_x_offset;
        float *dst_addr = dst_data + dst_x_offset;
#ifdef ENABLE_AVX
        MS_ST256_F32(dst_addr, MS_LD256_F32(src_addr));
#elif defined(ENABLE_ARM) || defined(ENABLE_SSE)
        MS_STQ_F32(dst_addr, MS_LDQ_F32(src_addr));
#else
        for (int k = 0; k < channel_tile; k++) {
          dst_addr[k] = src_addr[k];
        }
#endif
      }  // interval x loop
    }    // interval y loop
  } else {
    for (int interval = interval_y_s; interval < interval_y_e; interval++) {
      int src_y_offset = (interval * input_w + interval_x_s) * in_channel;
      int dst_y_offset = interval * input_unit * channel_tile + interval_x_s * channel_tile;
      for (int j = 0; j < (interval_x_e - interval_x_s); j++) {
        int src_x_offset = src_y_offset + j * in_channel;
        int dst_x_offset = dst_y_offset + j * channel_tile;
        const float *src_addr = src_data + src_x_offset;
        float *dst_addr = dst_data + dst_x_offset;
        for (int k = 0; k < real_c; k++) {
          dst_addr[k] = src_addr[k];
        }
      }  // interval x loop
    }    // interval y loop
  }
}

// fp32 conv winograd
void WinogradInputTransform(const float *input_data, float *trans_input, float *tmp_data, int cal_num,
                            int out_tile_index, int out_w_block_num, const ConvParameter *conv_param,
                            InputTransFunc func) {
  int input_unit = conv_param->input_unit_;
  int output_unit = conv_param->output_unit_;
  int in_channel = conv_param->input_channel_;
#ifdef ENABLE_AVX
  int channel_tile = C8NUM;
#else
  int channel_tile = C4NUM;
#endif
  int ic4 = UP_DIV(in_channel, channel_tile);
  int pad_h = conv_param->pad_u_;
  int pad_w = conv_param->pad_l_;
  int input_h = conv_param->input_h_;
  int input_w = conv_param->input_w_;
  NNACL_CHECK_ZERO_RETURN(out_w_block_num);

  for (int c = 0; c < cal_num; c++) {  // actual tiled number
    int src_x_s = (out_tile_index % out_w_block_num) * output_unit - pad_w;
    int src_y_s = (out_tile_index / out_w_block_num) * output_unit - pad_h;
    int interval_x_s = src_x_s > 0 ? 0 : -src_x_s;
    int interval_y_s = src_y_s > 0 ? 0 : -src_y_s;
    int src_x_e = src_x_s + input_unit;
    int src_y_e = src_y_s + input_unit;
    int interval_x_e = src_x_e < input_w ? input_unit : (input_w - src_x_s);
    int interval_y_e = src_y_e < input_h ? input_unit : (input_h - src_y_s);

    int src_plane_offset = in_channel * (src_y_s * input_w + src_x_s);
    int dst_plane_offset = c * in_channel;
    for (int ic = 0; ic < ic4; ic++) {
      int real_c = in_channel - ic * channel_tile;
      real_c = real_c > channel_tile ? channel_tile : real_c;
      const float *src_data = input_data + src_plane_offset + ic * channel_tile;
      PrepareTransInput(src_data, tmp_data, interval_x_s, interval_x_e, interval_y_s, interval_y_e, real_c, conv_param);

      // input transform
      const int tile_num = C12NUM;
      int dst_ic4_offset = dst_plane_offset + ic * channel_tile;
      int dst_step = tile_num * in_channel;
      float *trans_input_ptr = trans_input + dst_ic4_offset;
      func(tmp_data, trans_input_ptr, channel_tile, dst_step, real_c);
    }
    out_tile_index++;
  }  // cal_tile_num loop
}

// Only support arm64
void WinogradInputTransformOptStep(const float *input_data, float *trans_input, float *tmp_data, int cal_num,
                                   int out_tile_index, int out_w_block_num, const ConvParameter *conv_param,
                                   InputTransStepFunc func) {
  int input_unit = conv_param->input_unit_;
  int output_unit = conv_param->output_unit_;
  int in_channel = conv_param->input_channel_;
  int channel_tile = C4NUM;
  int ic4 = UP_DIV(in_channel, channel_tile);
  int pad_h = conv_param->pad_u_;
  int pad_w = conv_param->pad_l_;
  int input_h = conv_param->input_h_;
  int input_w = conv_param->input_w_;
  NNACL_CHECK_ZERO_RETURN(out_w_block_num);

  for (int c = 0; c < cal_num; c++) {  // actual tiled number
    int src_x_s = (out_tile_index % out_w_block_num) * output_unit - pad_w;
    int src_y_s = (out_tile_index / out_w_block_num) * output_unit - pad_h;
    int interval_x_s = src_x_s > 0 ? 0 : -src_x_s;
    int interval_y_s = src_y_s > 0 ? 0 : -src_y_s;
    int src_x_e = src_x_s + input_unit;
    int src_y_e = src_y_s + input_unit;
    int interval_x_e = src_x_e < input_w ? input_unit : (input_w - src_x_s);
    int interval_y_e = src_y_e < input_h ? input_unit : (input_h - src_y_s);

    int src_plane_offset = in_channel * (src_y_s * input_w + src_x_s);
    int dst_plane_offset = c * channel_tile;
    for (int ic = 0; ic < ic4; ic++) {
      int real_c = in_channel - ic * channel_tile;
      real_c = real_c > channel_tile ? channel_tile : real_c;
      const float *src_data = input_data + src_plane_offset + ic * channel_tile;
      PrepareTransInput(src_data, tmp_data, interval_x_s, interval_x_e, interval_y_s, interval_y_e, real_c, conv_param);

      // input transform
      const int block_tile = C12NUM;
      int dst_ic8_offset = dst_plane_offset + ic * block_tile * input_unit * input_unit * channel_tile;
      size_t dst_step = input_unit * block_tile * channel_tile;
      float *trans_input_ptr = trans_input + dst_ic8_offset;
      func(tmp_data, trans_input_ptr, channel_tile, dst_step, block_tile * channel_tile);
    }
    out_tile_index++;
  }  // cal_tile_num loop
}

void WinogradOutputNHWCTransform(const float *gemm_out, float *out_data, const float *bias_data, int cal_num,
                                 int out_tile_index, int output_unit_num, const ConvParameter *conv_param,
                                 OutputTransFunc func) {
  int output_unit = conv_param->output_unit_;
  int output_w = conv_param->output_w_;
  int output_h = conv_param->output_h_;
  int output_channel = conv_param->output_channel_;
#ifndef ENABLE_AVX
  int oc4 = UP_DIV(output_channel, C4NUM);
#endif
  int oc8 = UP_DIV(output_channel, C8NUM);
  int input_unit = conv_param->input_unit_;
  NNACL_CHECK_ZERO_RETURN(output_unit_num);

  for (int i = 0; i < cal_num; i++) {
    int dst_x_s = out_tile_index % output_unit_num;
    int dst_y_s = out_tile_index / output_unit_num;
    int r_w = output_w - dst_x_s * output_unit;
    r_w = r_w > output_unit ? output_unit : r_w;
    int r_h = output_h - dst_y_s * output_unit;
    r_h = r_h > output_unit ? output_unit : r_h;
    int tmp_ix = dst_x_s * output_unit;
    dst_x_s = tmp_ix > output_w ? output_w : tmp_ix;
    int tmp_iy = dst_y_s * output_unit;
    dst_y_s = tmp_iy > output_h ? output_h : tmp_iy;

    int src_tile_offset = i * oc8 * C8NUM * input_unit * input_unit;
    int dst_tile_offset = output_channel * (dst_x_s + dst_y_s * output_w);
#ifndef ENABLE_AVX
    // avx is write to nc4hw4
    for (int j = 0; j < oc4; j++) {
      int c8_block = j / 2;
      int c8_res = j % 2;
      int r_c = output_channel - j * C4NUM;
      r_c = r_c > C4NUM ? C4NUM : r_c;
      int src_oc4_offset = src_tile_offset + c8_block * input_unit * input_unit * C8NUM + c8_res * C4NUM;
      int dst_oc4_offset = dst_tile_offset + j * C4NUM;
      const float *src_ptr = gemm_out + src_oc4_offset;
      const float *bias_ptr = bias_data + j * C4NUM;
      float *dst_ptr = out_data + dst_oc4_offset;
      func(src_ptr, dst_ptr, bias_ptr, C8NUM, output_w, output_channel, r_w, r_h, r_c);
    }
#else
    // avx is write to nc8hw8
    for (int j = 0; j < oc8; j++) {
      int r_c = output_channel - j * C8NUM;
      r_c = r_c > C8NUM ? C8NUM : r_c;
      int src_oc8_offset = src_tile_offset + j * input_unit * input_unit * C8NUM;
      int dst_oc8_offset = dst_tile_offset + j * C8NUM;
      const float *src_ptr = gemm_out + src_oc8_offset;
      const float *bias_ptr = bias_data + j * C8NUM;
      float *dst_ptr = out_data + dst_oc8_offset;
      func(src_ptr, dst_ptr, bias_ptr, C8NUM, output_w, output_channel, r_w, r_h, r_c);
    }
#endif
    out_tile_index++;
  }
}

void WinogradOutputNC4HW4Transform(const float *gemm_out, float *out_data, const float *bias_data, int cal_num,
                                   int out_tile_index, int output_unit_num, const ConvParameter *conv_param,
                                   OutputTransFunc func) {
  int output_unit = conv_param->output_unit_;
  int output_w = conv_param->output_w_;
  int output_h = conv_param->output_h_;
  int output_plane = output_w * output_h;
  int output_channel = conv_param->output_channel_;
#ifndef ENABLE_AVX
  int oc4 = UP_DIV(output_channel, C4NUM);
#endif
  int oc8 = UP_DIV(output_channel, C8NUM);
  int input_unit = conv_param->input_unit_;
  NNACL_CHECK_ZERO_RETURN(output_unit_num);

  for (int i = 0; i < cal_num; i++) {
    int dst_x_s = out_tile_index % output_unit_num;
    int dst_y_s = out_tile_index / output_unit_num;
    int r_w = output_w - dst_x_s * output_unit;
    r_w = r_w > output_unit ? output_unit : r_w;
    int r_h = output_h - dst_y_s * output_unit;
    r_h = r_h > output_unit ? output_unit : r_h;
    int tmp_ix = dst_x_s * output_unit;
    dst_x_s = tmp_ix > output_w ? output_w : tmp_ix;
    int tmp_iy = dst_y_s * output_unit;
    dst_y_s = tmp_iy > output_h ? output_h : tmp_iy;

    int src_tile_offset = i * oc8 * C8NUM * input_unit * input_unit;
    int dst_tile_offset = dst_x_s + dst_y_s * output_w;
#ifdef ENABLE_AVX
    for (int j = 0; j < oc8; j++) {
      int r_c = output_channel - j * C8NUM;
      r_c = r_c > C8NUM ? C8NUM : r_c;
      int src_oc8_offset = src_tile_offset + j * input_unit * input_unit * C8NUM;
      int dst_oc8_offset = (dst_tile_offset + output_plane * j) * C8NUM;
      const float *src_ptr = gemm_out + src_oc8_offset;
      const float *bias_ptr = bias_data + j * C8NUM;
      float *dst_ptr = out_data + dst_oc8_offset;
      func(src_ptr, dst_ptr, bias_ptr, C8NUM, output_w, r_c, r_w, r_h, r_c);
    }
#else
    for (int j = 0; j < oc4; j++) {
      int c8_block = j / 2;
      int c8_res = j % 2;
      int r_c = output_channel - j * C4NUM;
      r_c = r_c > C4NUM ? C4NUM : r_c;
      int src_oc4_offset = src_tile_offset + c8_block * input_unit * input_unit * C8NUM + c8_res * C4NUM;
      int dst_oc4_offset = (dst_tile_offset + output_plane * j) * C4NUM;
      const float *src_ptr = gemm_out + src_oc4_offset;
      const float *bias_ptr = bias_data + j * C4NUM;
      float *dst_ptr = out_data + dst_oc4_offset;
      func(src_ptr, dst_ptr, bias_ptr, C8NUM, output_w, r_c, r_w, r_h, r_c);
    }
#endif
    out_tile_index++;
  }
}
