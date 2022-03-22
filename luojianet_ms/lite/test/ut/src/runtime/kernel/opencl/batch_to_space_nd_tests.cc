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
#include <array>
#include "ut/src/runtime/kernel/opencl/common.h"
#include "nnacl/batch_to_space.h"

namespace luojianet_ms::lite::opencl::test {

class TestOpenCL_BatchToSpaceND : public CommonTest {};

namespace {
// PrimitiveType_BatchToSpaceND: src/ops/populate/batch_to_space_populate.cc
OpParameter *CreateParameter(int block_shape[], int crops[], const std::vector<int> &input_shape,
                             std::vector<int> *output_shape) {
  auto *param = test::CreateParameter<BatchToSpaceParameter>(schema::PrimitiveType_BatchToSpace);
  memcpy(param->block_shape_, block_shape, sizeof(param->block_shape_));
  memcpy(param->crops_, crops, sizeof(param->crops_));
  *output_shape = {input_shape[0] / param->block_shape_[0] / param->block_shape_[1],
                   input_shape[1] * param->block_shape_[0] - param->crops_[0] - param->crops_[1],
                   input_shape[2] * param->block_shape_[1] - param->crops_[2] - param->crops_[3], input_shape[3]};
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_BatchToSpaceND, H2W2Pad2020) {
  std::vector<int> input_shape = {4, 5, 5, 4};
  int block_shape[] = {2, 2};
  int crops[] = {2, 0, 2, 0};
  std::vector<int> output_shape;
  float input_data[] = {
    172, 47,  117, 192, 67,  251, 195, 103, 9,   211, 21,  242, 36,  87,  70,  216, 88,  140, 58,  193, 230, 39,  87,
    174, 88,  81,  165, 25,  77,  72,  9,   148, 115, 208, 243, 197, 254, 79,  175, 192, 82,  99,  216, 177, 243, 29,
    147, 147, 142, 167, 32,  193, 9,   185, 127, 32,  31,  202, 244, 151, 163, 254, 203, 114, 183, 28,  34,  128, 128,
    164, 53,  133, 38,  232, 244, 17,  79,  132, 105, 42,  186, 31,  120, 1,   65,  231, 169, 57,  35,  102, 119, 11,
    174, 82,  91,  128, 142, 99,  53,  140, 121, 170, 84,  203, 68,  6,   196, 47,  127, 244, 131, 204, 100, 180, 232,
    78,  143, 148, 227, 186, 23,  207, 141, 117, 85,  48,  49,  69,  169, 163, 192, 95,  197, 94,  0,   113, 178, 36,
    162, 48,  93,  131, 98,  42,  205, 112, 231, 149, 201, 127, 0,   138, 114, 43,  186, 127, 23,  187, 130, 121, 98,
    62,  163, 222, 123, 195, 82,  174, 227, 148, 209, 50,  155, 14,  41,  58,  193, 36,  10,  86,  43,  104, 11,  2,
    51,  80,  32,  182, 128, 38,  19,  174, 42,  115, 184, 188, 232, 77,  30,  24,  125, 2,   3,   94,  226, 107, 13,
    112, 40,  72,  19,  95,  72,  154, 194, 248, 180, 67,  236, 61,  14,  96,  4,   195, 237, 139, 252, 86,  205, 121,
    109, 75,  184, 16,  152, 157, 149, 110, 25,  208, 188, 121, 118, 117, 189, 83,  161, 104, 160, 228, 251, 251, 121,
    70,  213, 31,  13,  71,  184, 152, 79,  41,  18,  40,  182, 207, 11,  166, 111, 93,  249, 129, 223, 118, 44,  216,
    125, 24,  67,  210, 239, 3,   234, 204, 230, 35,  214, 254, 189, 197, 215, 43,  32,  11,  104, 212, 138, 182, 235,
    165, 125, 156, 111, 232, 2,   27,  211, 217, 151, 53,  51,  174, 148, 181, 29,  67,  35,  39,  137, 73,  41,  151,
    131, 46,  218, 178, 108, 3,   31,  9,   138, 27,  173, 199, 167, 61,  85,  97,  44,  34,  162, 88,  33,  133, 232,
    36,  0,   203, 34,  197, 126, 181, 254, 80,  190, 136, 189, 129, 209, 112, 35,  120, 91,  168, 116, 36,  176, 25,
    67,  103, 252, 35,  114, 30,  29,  241, 33,  146, 17,  221, 84,  253, 2,   69,  101, 140, 44,  117, 253, 66,  111,
    91,  85,  167, 39,  203, 150, 158, 145, 198};
  float output_data[] = {
    88,  81,  165, 25,  85,  48,  49,  69,  77,  72,  9,   148, 169, 163, 192, 95,  115, 208, 243, 197, 197, 94,
    0,   113, 254, 79,  175, 192, 178, 36,  162, 48,  237, 139, 252, 86,  218, 178, 108, 3,   205, 121, 109, 75,
    31,  9,   138, 27,  184, 16,  152, 157, 173, 199, 167, 61,  149, 110, 25,  208, 85,  97,  44,  34,  243, 29,
    147, 147, 205, 112, 231, 149, 142, 167, 32,  193, 201, 127, 0,   138, 9,   185, 127, 32,  114, 43,  186, 127,
    31,  202, 244, 151, 23,  187, 130, 121, 189, 83,  161, 104, 232, 36,  0,   203, 160, 228, 251, 251, 34,  197,
    126, 181, 121, 70,  213, 31,  254, 80,  190, 136, 13,  71,  184, 152, 189, 129, 209, 112, 183, 28,  34,  128,
    123, 195, 82,  174, 128, 164, 53,  133, 227, 148, 209, 50,  38,  232, 244, 17,  155, 14,  41,  58,  79,  132,
    105, 42,  193, 36,  10,  86,  182, 207, 11,  166, 116, 36,  176, 25,  111, 93,  249, 129, 67,  103, 252, 35,
    223, 118, 44,  216, 114, 30,  29,  241, 125, 24,  67,  210, 33,  146, 17,  221, 65,  231, 169, 57,  51,  80,
    32,  182, 35,  102, 119, 11,  128, 38,  19,  174, 174, 82,  91,  128, 42,  115, 184, 188, 142, 99,  53,  140,
    232, 77,  30,  24,  230, 35,  214, 254, 101, 140, 44,  117, 189, 197, 215, 43,  253, 66,  111, 91,  32,  11,
    104, 212, 85,  167, 39,  203, 138, 182, 235, 165, 150, 158, 145, 198};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_shape, crops, input_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_BatchToSpaceND, H3W3Pad0101) {
  std::vector<int> input_shape = {9, 3, 3, 4};
  int block_shape[] = {3, 3};
  int crops[] = {0, 1, 0, 1};
  std::vector<int> output_shape;
  float input_data[] = {
    172, 47,  117, 192, 67,  251, 195, 103, 9,   211, 21,  242, 36,  87,  70,  216, 88,  140, 58,  193, 230, 39,
    87,  174, 88,  81,  165, 25,  77,  72,  9,   148, 115, 208, 243, 197, 254, 79,  175, 192, 82,  99,  216, 177,
    243, 29,  147, 147, 142, 167, 32,  193, 9,   185, 127, 32,  31,  202, 244, 151, 163, 254, 203, 114, 183, 28,
    34,  128, 128, 164, 53,  133, 38,  232, 244, 17,  79,  132, 105, 42,  186, 31,  120, 1,   65,  231, 169, 57,
    35,  102, 119, 11,  174, 82,  91,  128, 142, 99,  53,  140, 121, 170, 84,  203, 68,  6,   196, 47,  127, 244,
    131, 204, 100, 180, 232, 78,  143, 148, 227, 186, 23,  207, 141, 117, 85,  48,  49,  69,  169, 163, 192, 95,
    197, 94,  0,   113, 178, 36,  162, 48,  93,  131, 98,  42,  205, 112, 231, 149, 201, 127, 0,   138, 114, 43,
    186, 127, 23,  187, 130, 121, 98,  62,  163, 222, 123, 195, 82,  174, 227, 148, 209, 50,  155, 14,  41,  58,
    193, 36,  10,  86,  43,  104, 11,  2,   51,  80,  32,  182, 128, 38,  19,  174, 42,  115, 184, 188, 232, 77,
    30,  24,  125, 2,   3,   94,  226, 107, 13,  112, 40,  72,  19,  95,  72,  154, 194, 248, 180, 67,  236, 61,
    14,  96,  4,   195, 237, 139, 252, 86,  205, 121, 109, 75,  184, 16,  152, 157, 149, 110, 25,  208, 188, 121,
    118, 117, 189, 83,  161, 104, 160, 228, 251, 251, 121, 70,  213, 31,  13,  71,  184, 152, 79,  41,  18,  40,
    182, 207, 11,  166, 111, 93,  249, 129, 223, 118, 44,  216, 125, 24,  67,  210, 239, 3,   234, 204, 230, 35,
    214, 254, 189, 197, 215, 43,  32,  11,  104, 212, 138, 182, 235, 165, 125, 156, 111, 232, 2,   27,  211, 217,
    151, 53,  51,  174, 148, 181, 29,  67,  35,  39,  137, 73,  41,  151, 131, 46};
  float output_data[] = {
    172, 47,  117, 192, 254, 79,  175, 192, 38,  232, 244, 17,  67,  251, 195, 103, 82,  99,  216, 177, 79,  132,
    105, 42,  9,   211, 21,  242, 243, 29,  147, 147, 127, 244, 131, 204, 205, 112, 231, 149, 43,  104, 11,  2,
    100, 180, 232, 78,  201, 127, 0,   138, 51,  80,  32,  182, 143, 148, 227, 186, 114, 43,  186, 127, 180, 67,
    236, 61,  121, 70,  213, 31,  189, 197, 215, 43,  14,  96,  4,   195, 13,  71,  184, 152, 32,  11,  104, 212,
    237, 139, 252, 86,  79,  41,  18,  40,  36,  87,  70,  216, 142, 167, 32,  193, 65,  231, 169, 57,  88,  140,
    58,  193, 9,   185, 127, 32,  35,  102, 119, 11,  230, 39,  87,  174, 31,  202, 244, 151, 23,  207, 141, 117,
    23,  187, 130, 121, 42,  115, 184, 188, 85,  48,  49,  69,  98,  62,  163, 222, 232, 77,  30,  24,  169, 163,
    192, 95,  123, 195, 82,  174, 205, 121, 109, 75,  182, 207, 11,  166, 125, 156, 111, 232, 184, 16,  152, 157,
    111, 93,  249, 129, 2,   27,  211, 217, 149, 110, 25,  208, 223, 118, 44,  216, 88,  81,  165, 25,  163, 254,
    203, 114, 142, 99,  53,  140, 77,  72,  9,   148, 183, 28,  34,  128, 121, 170, 84,  203, 115, 208, 243, 197,
    128, 164, 53,  133, 197, 94,  0,   113, 227, 148, 209, 50,  226, 107, 13,  112, 178, 36,  162, 48,  155, 14,
    41,  58,  40,  72,  19,  95,  93,  131, 98,  42,  193, 36,  10,  86};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_shape, crops, input_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

}  // namespace luojianet_ms::lite::opencl::test
