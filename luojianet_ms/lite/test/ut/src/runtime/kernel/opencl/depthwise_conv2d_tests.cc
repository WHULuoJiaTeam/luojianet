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
#include "ut/src/runtime/kernel/opencl/common.h"
#include "nnacl/conv_parameter.h"

namespace luojianet_ms::lite::opencl::test {

class TestOpenCL_DepthwiseConv2d : public CommonTest {};

namespace {
// Check and optimize
// PrimitiveType_DepthwiseConv2D: src/ops/populate/depthwise_conv2d_populate.cc
OpParameter *CreateParameter(int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_u, int pad_d, int pad_l,
                             int pad_r, int dilation_h, int dilation_w, ActType act_type, int input_channel) {
  auto *param = test::CreateParameter<ConvParameter>(schema::PrimitiveType_Conv2DFusion);
  param->kernel_h_ = kernel_h;
  param->kernel_w_ = kernel_w;
  param->stride_h_ = stride_h;
  param->stride_w_ = stride_w;
  param->pad_u_ = pad_u;
  param->pad_d_ = pad_d;
  param->pad_l_ = pad_l;
  param->pad_r_ = pad_r;
  param->input_channel_ = input_channel;
  param->output_channel_ = input_channel;
  param->group_ = input_channel;
  param->dilation_h_ = dilation_h;
  param->dilation_w_ = dilation_w;
  param->act_type_ = act_type;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_DepthwiseConv2d, NoPad) {
  int kernel_h = 3;
  int kernel_w = 3;
  int stride_h = 1;
  int stride_w = 1;
  int pad_u = 0;
  int pad_d = 0;
  int pad_l = 0;
  int pad_r = 0;
  int dilation_h = 1;
  int dilation_w = 1;
  ActType act_type = ActType_No;

  std::vector<int> input_shape = {1, 4, 4, 4};
  std::vector<int> output_shape = {1, 2, 2, 4};
  std::vector<int> weight_shape = {1, kernel_h, kernel_w, output_shape.back()};
  std::vector<int> bias_shape = {output_shape.back()};
  float input_data[] = {
    0.5488135,  0.71518934, 0.60276335, 0.5448832,  0.4236548,  0.6458941,  0.4375872,  0.891773,
    0.96366274, 0.3834415,  0.79172504, 0.5288949,  0.56804454, 0.92559665, 0.07103606, 0.0871293,
    0.0202184,  0.83261985, 0.77815676, 0.87001216, 0.9786183,  0.7991586,  0.46147937, 0.7805292,
    0.11827443, 0.639921,   0.14335328, 0.9446689,  0.5218483,  0.41466194, 0.2645556,  0.7742337,
    0.45615032, 0.56843394, 0.0187898,  0.6176355,  0.6120957,  0.616934,   0.94374806, 0.6818203,
    0.3595079,  0.43703195, 0.6976312,  0.06022547, 0.6667667,  0.67063785, 0.21038257, 0.12892629,
    0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,  0.10204481, 0.20887676, 0.16130951,
    0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958, 0.11037514, 0.6563296,  0.13818295,
  };
  float bias_data[] = {0, 0, 0, 0};
  float weight_data[] = {0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,  0.09609841, 0.97645944, 0.4686512,
                         0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772,
                         0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146, 0.2653895,  0.5232481,  0.09394051,
                         0.5759465,  0.9292962,  0.31856894, 0.6674104,  0.13179787, 0.7163272,  0.2894061,  0.18319136,
                         0.5865129,  0.02010755, 0.82894003, 0.00469548};
  float output_data[] = {2.9720426, 1.890834,  2.3618119, 2.3867798, 2.5666943, 1.6261611, 2.0977764, 1.6445805,
                         2.462798,  1.6643658, 1.6861027, 1.8428761, 2.5156446, 1.5366757, 1.6767557, 1.6905226};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(kernel_h, kernel_w, stride_h, stride_w, pad_u, pad_d, pad_l, pad_r, dilation_h,
                                  dilation_w, act_type, input_shape.back());
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-2 : 1e-5, 1e-1, true);
  }
}

TEST_F(TestOpenCL_DepthwiseConv2d, Pad) {
  int kernel_h = 3;
  int kernel_w = 3;
  int stride_h = 1;
  int stride_w = 1;
  int pad_u = 1;
  int pad_d = 1;
  int pad_l = 1;
  int pad_r = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  ActType act_type = ActType_No;

  std::vector<int> input_shape = {1, 3, 3, 5};
  std::vector<int> output_shape = {1, 3, 3, 5};
  std::vector<int> weight_shape = {1, kernel_h, kernel_w, output_shape.back()};
  std::vector<int> bias_shape = {output_shape.back()};
  float input_data[] = {0.5488135,  0.3834415,  0.77815676, 0.9446689, 0.6120957,  0.71518934, 0.79172504, 0.87001216,
                        0.5218483,  0.616934,   0.60276335, 0.5288949, 0.9786183,  0.41466194, 0.94374806, 0.5448832,
                        0.56804454, 0.7991586,  0.2645556,  0.6818203, 0.4236548,  0.92559665, 0.46147937, 0.7742337,
                        0.3595079,  0.6458941,  0.07103606, 0.7805292, 0.45615032, 0.43703195, 0.4375872,  0.0871293,
                        0.11827443, 0.56843394, 0.6976312,  0.891773,  0.0202184,  0.639921,   0.0187898,  0.06022547,
                        0.96366274, 0.83261985, 0.14335328, 0.6176355, 0.6667667};
  float weight_data[] = {0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,
                         0.10204481, 0.20887676, 0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958,
                         0.11037514, 0.6563296,  0.13818295, 0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,
                         0.09609841, 0.97645944, 0.4686512,  0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696,
                         0.12019656, 0.2961402,  0.11872772, 0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146,
                         0.2653895,  0.5232481,  0.09394051, 0.5759465,  0.9292962};
  float bias_data[] = {0, 0, 0, 0, 0};
  float output_data[] = {1.189188,   1.0425153,  1.8012011,  0.6074867,  1.2120346,  1.5005531,  0.8346756, 2.4365785,
                         0.54975945, 1.6815965,  1.2690231,  0.60214907, 1.6158017,  0.42115876, 0.8854959, 1.1709145,
                         1.0929465,  1.3534508,  1.1985044,  1.2932993,  2.4621446,  1.7086457,  2.6977584, 2.1960166,
                         2.3769147,  2.3185873,  0.6133741,  0.9687358,  0.9987654,  1.0254729,  0.8368954, 0.74171704,
                         0.8749627,  0.8953936,  0.5093431,  1.5496738,  0.54936385, 0.7683113,  1.165742,  1.3682933,
                         1.0517888,  0.59817517, 0.75649744, 1.2075498,  0.38804203};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(kernel_h, kernel_w, stride_h, stride_w, pad_u, pad_d, pad_l, pad_r, dilation_h,
                                  dilation_w, act_type, input_shape.back());
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-2 : 1e-5, 1e-1, true);
  }
}

TEST_F(TestOpenCL_DepthwiseConv2d, NoPad1) {
  int kernel_h = 2;
  int kernel_w = 2;
  int stride_h = 1;
  int stride_w = 1;
  int pad_u = 0;
  int pad_d = 0;
  int pad_l = 0;
  int pad_r = 0;
  int dilation_h = 1;
  int dilation_w = 1;
  ActType act_type = ActType_No;

  std::vector<int> input_shape = {1, 4, 4, 4};
  std::vector<int> output_shape = {1, 3, 3, 4};
  std::vector<int> weight_shape = {1, kernel_h, kernel_w, output_shape.back()};
  std::vector<int> bias_shape = {output_shape.back()};
  float input_data[] = {0.5488135,  0.71518934, 0.60276335, 0.5448832,  0.4236548,  0.6458941,  0.4375872,  0.891773,
                        0.96366274, 0.3834415,  0.79172504, 0.5288949,  0.56804454, 0.92559665, 0.07103606, 0.0871293,
                        0.0202184,  0.83261985, 0.77815676, 0.87001216, 0.9786183,  0.7991586,  0.46147937, 0.7805292,
                        0.11827443, 0.639921,   0.14335328, 0.9446689,  0.5218483,  0.41466194, 0.2645556,  0.7742337,
                        0.45615032, 0.56843394, 0.0187898,  0.6176355,  0.6120957,  0.616934,   0.94374806, 0.6818203,
                        0.3595079,  0.43703195, 0.6976312,  0.06022547, 0.6667667,  0.67063785, 0.21038257, 0.12892629,
                        0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,  0.10204481, 0.20887676, 0.16130951,
                        0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958, 0.11037514, 0.6563296,  0.13818295};
  float bias_data[] = {0, 0, 0, 0};
  float weight_data[] = {0.19658236, 0.36872517, 0.82099323, 0.09710128, 0.83794491, 0.09609841,
                         0.97645947, 0.4686512,  0.97676109, 0.60484552, 0.73926358, 0.03918779,
                         0.28280696, 0.12019656, 0.2961402,  0.11872772};
  float output_data[] = {0.3757235,  1.8489048,  1.4467758,  0.6116009,  1.2535334, 1.6583176, 1.2530621,  0.6590755,
                         0.5466661,  1.22944,    0.93263525, 0.5317252,  0.7987474, 1.618667,  1.090071,   0.60372007,
                         0.773425,   1.5383728,  1.262479,   0.54334986, 0.5755667, 1.3171062, 0.82401496, 0.39336145,
                         0.6703031,  0.9385749,  1.018886,   0.40566355, 1.1277528, 0.7773028, 1.5164642,  0.27685273,
                         0.86816025, 0.72971237, 1.1791146,  0.12131907};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(kernel_h, kernel_w, stride_h, stride_w, pad_u, pad_d, pad_l, pad_r, dilation_h,
                                  dilation_w, act_type, input_shape.back());
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-2 : 1e-5, 1e-1, true);
  }
}
TEST_F(TestOpenCL_DepthwiseConv2d, Pad1) {
  int kernel_h = 3;
  int kernel_w = 3;
  int stride_h = 1;
  int stride_w = 1;
  int pad_u = 1;
  int pad_d = 1;
  int pad_l = 1;
  int pad_r = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  ActType act_type = ActType_No;

  std::vector<int> input_shape = {1, 5, 5, 6};
  std::vector<int> output_shape = {1, 5, 5, 6};
  std::vector<int> weight_shape = {1, kernel_h, kernel_w, output_shape.back()};
  std::vector<int> bias_shape = {output_shape.back()};
  float input_data[] = {
    0.5488135,  0.71518934, 0.60276335, 0.5448832,  0.4236548,  0.6458941,  0.4375872,  0.891773,   0.96366274,
    0.3834415,  0.79172504, 0.5288949,  0.56804454, 0.92559665, 0.07103606, 0.0871293,  0.0202184,  0.83261985,
    0.77815676, 0.87001216, 0.9786183,  0.7991586,  0.46147937, 0.7805292,  0.11827443, 0.639921,   0.14335328,
    0.9446689,  0.5218483,  0.41466194, 0.2645556,  0.7742337,  0.45615032, 0.56843394, 0.0187898,  0.6176355,
    0.6120957,  0.616934,   0.94374806, 0.6818203,  0.3595079,  0.43703195, 0.6976312,  0.06022547, 0.6667667,
    0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,  0.10204481,
    0.20887676, 0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958, 0.11037514, 0.6563296,
    0.13818295, 0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,  0.09609841, 0.97645944, 0.4686512,
    0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772, 0.31798318,
    0.41426298, 0.06414749, 0.6924721,  0.56660146, 0.2653895,  0.5232481,  0.09394051, 0.5759465,  0.9292962,
    0.31856894, 0.6674104,  0.13179787, 0.7163272,  0.2894061,  0.18319136, 0.5865129,  0.02010755, 0.82894003,
    0.00469548, 0.6778165,  0.27000797, 0.735194,   0.96218854, 0.24875315, 0.57615733, 0.5920419,  0.5722519,
    0.22308163, 0.952749,   0.44712538, 0.84640867, 0.6994793,  0.29743695, 0.81379783, 0.39650574, 0.8811032,
    0.5812729,  0.8817354,  0.6925316,  0.7252543,  0.50132436, 0.95608366, 0.6439902,  0.42385504, 0.6063932,
    0.0191932,  0.30157483, 0.66017354, 0.2900776,  0.6180154,  0.4287687,  0.13547407, 0.29828233, 0.5699649,
    0.59087276, 0.57432526, 0.6532008,  0.65210325, 0.43141845, 0.8965466,  0.36756188, 0.43586493, 0.89192337,
    0.806194,   0.7038886,  0.10022689, 0.9194826,  0.7142413,  0.998847};
  float weight_data[] = {0.1494483,  0.86812606, 0.16249293, 0.61555956, 0.12381998, 0.84800823, 0.80731896, 0.56910074,
                         0.4071833,  0.069167,   0.69742877, 0.45354268, 0.7220556,  0.86638233, 0.97552151, 0.85580334,
                         0.01171408, 0.35997806, 0.72999056, 0.17162968, 0.52103661, 0.05433799, 0.19999652, 0.01852179,
                         0.7936977,  0.22392469, 0.34535168, 0.92808129, 0.7044144,  0.03183893, 0.16469416, 0.6214784,
                         0.57722859, 0.23789282, 0.934214,   0.61396596, 0.5356328,  0.58990998, 0.73012203, 0.311945,
                         0.39822106, 0.20984375, 0.18619301, 0.94437239, 0.7395508,  0.49045881, 0.22741463, 0.25435648,
                         0.05802916, 0.43441663, 0.31179588, 0.69634349, 0.37775184, 0.17960368};
  float bias_data[] = {0, 0, 0, 0, 0, 0};
  float output_data[] = {
    0.8388255,  1.7207233,  0.56646764, 1.50962,   0.6184657,  0.7572999, 1.7197044,  2.8834608, 1.0304408,  1.5622743,
    0.95027775, 1.1451806,  2.0191956,  2.9541533, 1.1799709,  1.6366025, 1.3484346,  1.0071151, 1.3740869,  2.1602216,
    1.0846798,  1.7810996,  1.6170096,  0.6889053, 0.8671698,  1.4957678, 0.68065727, 1.0596768, 0.9761665,  0.38881996,
    1.524128,   2.2121127,  1.1506181,  1.330961,  1.8186853,  0.9094476, 2.3777275,  2.5568333, 1.8321692,  1.8297466,
    2.069798,   1.3701197,  2.7548862,  2.0871775, 2.3611763,  1.5387508, 1.6725919,  1.2565864, 2.6130712,  2.0915375,
    1.2955335,  1.6571269,  1.7603228,  1.3315495, 1.0005323,  1.0135669, 1.2701392,  1.8230836, 1.6048919,  1.4224635,
    1.4651375,  1.0251865,  1.0325887,  1.2355556, 1.3313429,  0.6756204, 2.602416,   2.1827717, 1.4354478,  1.6628273,
    2.0171032,  1.0299077,  2.6085434,  1.3310422, 2.1677747,  2.457499,  2.6715999,  1.0225507, 2.5822947,  2.1068158,
    1.6401942,  2.5422354,  2.6937182,  1.3813802, 1.1241511,  1.273326,  1.2024405,  1.4564767, 2.016776,   1.0182433,
    1.228782,   0.83329916, 1.033041,   1.3280122, 1.9437144,  0.6729013, 2.438968,   2.3275855, 2.289177,   1.4376242,
    2.4595368,  1.325891,   2.018128,   2.676854,  1.9685578,  1.8240746, 2.3104675,  1.4958379, 2.474168,   2.6657124,
    1.6738743,  2.336092,   2.3048637,  1.802324,  1.7594845,  1.6022205, 1.2564734,  1.8977238, 1.6991055,  1.8674731,
    0.47793916, 1.2031221,  0.6579696,  1.0724078, 0.96408695, 0.5074543, 1.2399375,  1.410824,  0.56263226, 1.3138686,
    1.4859737,  0.7219256,  1.3437214,  2.0015993, 1.0472497,  1.064316,  1.7359762,  0.9249617, 1.2835678,  2.1866667,
    0.92954785, 2.005947,   1.8761289,  1.2612648, 1.2410495,  1.263778,  0.54638237, 1.8269669, 1.3152003,  0.7890457};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(kernel_h, kernel_w, stride_h, stride_w, pad_u, pad_d, pad_l, pad_r, dilation_h,
                                  dilation_w, act_type, input_shape.back());
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-2 : 1e-5, 1e-1, true);
  }
}
}  // namespace luojianet_ms::lite::opencl::test
