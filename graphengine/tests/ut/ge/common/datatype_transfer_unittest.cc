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

#include <gtest/gtest.h>

#include "common/formats/format_transfers/datatype_transfer.h"

//#include "common/formats/format_transfers/format_transfer.h"
#include "common/formats/formats.h"
#include "common/fp16_t.h"

namespace ge {
namespace formats {
class UtestDataTypeTransfer : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestDataTypeTransfer, fp16_fp32) {
  fp16_t data[1 * 4 * 4 * 2] = {
      15272, 12501, 13940, 10024, 13356, 13068, 12088, 13733, 15257, 14104, 11089, 15298, 10597, 14359, 14402, 14748,
      14596, 14063, 14674, 13393, 12937, 13466, 14313, 13295, 15000, 15167, 15311, 13122, 10691, 15165, 14621, 14000,
  };

  float ret[1 * 4 * 4 * 2] = {
      0.957031, 0.151001,  0.40332,  0.0279541, 0.260742, 0.220215, 0.112793,  0.352783, 0.949707, 0.443359, 0.0571594,
      0.969727, 0.0421448, 0.51123,  0.532227,  0.701172, 0.626953, 0.43335,   0.665039, 0.269775, 0.204224, 0.287598,
      0.494385, 0.247925,  0.824219, 0.905762,  0.976074, 0.226807, 0.0450134, 0.904785, 0.63916,  0.417969,
  };
  TransResult result;
  DataTypeTransfer transfer;
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_FLOAT16, DT_FLOAT};
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  bool is_equal = true;
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    if (abs((reinterpret_cast<float *>(result.data.get()))[i] - ret[i]) > 1.0e-6) {
      is_equal = false;
      break;
    }
  }
  EXPECT_FLOAT_EQ(is_equal, true);

  TransResult result2;
  DataTypeTransfer transfer2;
  CastArgs args2{reinterpret_cast<uint8_t *>(ret), sizeof(ret) / sizeof(ret[0]), DT_FLOAT, DT_FLOAT16};
  EXPECT_EQ(transfer2.TransDataType(args2, result2), SUCCESS);
  EXPECT_EQ(result2.length, sizeof(data));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<fp16_t *>(result2.data.get()))[i].val, data[i].val);
  }
  EXPECT_EQ(TransDataType(args2, result2), SUCCESS);
}

TEST_F(UtestDataTypeTransfer, int32_fp16) {
  int32_t data[1 * 4 * 4 * 2] = {
      15272, 12501, 13940, 10024, 13356, 13068, 12088, 13733, 15257, 14104, 11089, 15298, 10597, 14359, 14402, 14748,
      14596, 14063, 14674, 13393, 12937, 13466, 14313, 13295, 15000, 15167, 15311, 13122, 10691, 15165, 14621, 14000,
  };

  fp16_t ret[1 * 4 * 4 * 2] = {
      29557, 29211, 29390, 28901, 29318, 29282, 29159, 29365, 29555, 29411, 29034, 29560, 28973, 29443, 29448, 29492,
      29472, 29406, 29482, 29322, 29265, 29331, 29437, 29310, 29523, 29544, 29562, 29288, 28984, 29544, 29476, 29398,
  };
  TransResult result;
  DataTypeTransfer transfer;
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_INT32, DT_FLOAT16};
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<fp16_t *>(result.data.get()))[i].val, ret[i].val);
  }

  TransResult result2;
  DataTypeTransfer transfer2;
  CastArgs args2{reinterpret_cast<uint8_t *>(ret), sizeof(data) / sizeof(data[0]), DT_FLOAT16, DT_INT32};
  EXPECT_EQ(transfer2.TransDataType(args2, result2), SUCCESS);
  EXPECT_EQ(result2.length, sizeof(data));
  bool is_equal = true;
  for (size_t i = 0; i < sizeof(data) / sizeof(data[0]); ++i) {
    if (abs((reinterpret_cast<int32_t *>(result2.data.get()))[i] - data[i]) / abs(data[i]) > 0.05) {
      is_equal = false;
      break;
    }
  }
  EXPECT_EQ(is_equal, true);
}

TEST_F(UtestDataTypeTransfer, fp32_fp16) {
  fp16_t data[1 * 4 * 4 * 16] = {
      15272, 12501, 13940, 10024, 13356, 13068, 12088, 13733, 15257, 14104, 11089, 15298, 10597, 14359, 14402, 14748,
      14596, 14063, 14674, 13393, 12937, 13466, 14313, 13295, 15000, 15167, 15311, 13122, 10691, 15165, 14621, 14000,
      13584, 14715, 15105, 14479, 14007, 9846,  14325, 12765, 13343, 13988, 10021, 14598, 14623, 15077, 15204, 12528,
      12024, 14236, 14857, 13009, 15216, 12916, 12754, 14807, 15174, 15075, 12998, 13834, 15174, 13674, 15251, 12683,
      13116, 14819, 11956, 14416, 14717, 14954, 15267, 15143, 15292, 9704,  14781, 14965, 14808, 15008, 11416, 15074,
      14168, 14417, 13441, 10673, 14945, 15114, 15358, 15116, 11950, 12057, 15321, 14973, 14950, 13984, 14900, 11361,
      10161, 14742, 13366, 13683, 13439, 12865, 10623, 14720, 14545, 13063, 10190, 12474, 9850,  15088, 15228, 14195,
      13428, 12443, 14719, 14816, 13231, 12818, 13667, 9680,  14814, 13924, 12757, 15178, 13444, 13673, 14405, 12711,
      15279, 14207, 9089,  13774, 13008, 14685, 13887, 15293, 13983, 14590, 15232, 15285, 15071, 14974, 15257, 13900,
      14907, 15269, 10955, 13635, 15132, 15026, 14218, 14498, 15235, 11243, 14704, 11563, 14394, 6840,  13619, 14655,
      12830, 14094, 12487, 13016, 13128, 15082, 6517,  14170, 14713, 14208, 13583, 12831, 15064, 13157, 13761, 14456,
      14905, 14798, 11391, 14668, 13906, 11053, 12381, 15210, 13567, 15159, 15270, 15073, 13887, 11861, 14615, 12627,
      15209, 14630, 13394, 14228, 14184, 13719, 14805, 13748, 14215, 13234, 13053, 14651, 14753, 14560, 12289, 14957,
      12826, 14788, 15236, 14249, 15211, 14329, 14830, 14793, 13202, 14635, 14489, 14664, 10751, 10992, 13459, 13658,
      14947, 14484, 15045, 14431, 14644, 13939, 14088, 14092, 14765, 14096, 14696, 13201, 15162, 14751, 14119, 13506,
      14659, 15355, 14904, 13374, 15048, 15188, 14733, 14307, 12518, 12511, 15187, 11018, 13072, 15023, 11355, 14216,
  };

  float ret[1 * 4 * 4 * 16] = {
      0.957031,  0.151001,  0.40332,   0.0279541, 0.260742,  0.220215,  0.112793,   0.352783,   0.949707,  0.443359,
      0.0571594, 0.969727,  0.0421448, 0.51123,   0.532227,  0.701172,  0.626953,   0.43335,    0.665039,  0.269775,
      0.204224,  0.287598,  0.494385,  0.247925,  0.824219,  0.905762,  0.976074,   0.226807,   0.0450134, 0.904785,
      0.63916,   0.417969,  0.316406,  0.685059,  0.875488,  0.569824,  0.419678,   0.025238,   0.497314,  0.183228,
      0.257568,  0.415039,  0.0279083, 0.62793,   0.640137,  0.861816,  0.923828,   0.154297,   0.108887,  0.475586,
      0.754395,  0.213013,  0.929688,  0.20166,   0.181885,  0.72998,   0.90918,    0.86084,    0.21167,   0.377441,
      0.90918,   0.338379,  0.946777,  0.173218,  0.226074,  0.73584,   0.104736,   0.539062,   0.686035,  0.801758,
      0.95459,   0.894043,  0.966797,  0.0230713, 0.717285,  0.807129,  0.730469,   0.828125,   0.0717773, 0.860352,
      0.458984,  0.539551,  0.281494,  0.0444641, 0.797363,  0.879883,  0.999023,   0.880859,   0.10437,   0.110901,
      0.980957,  0.811035,  0.799805,  0.414062,  0.775391,  0.0684204, 0.0300446,  0.698242,   0.263184,  0.340576,
      0.281006,  0.195435,  0.0429382, 0.6875,    0.602051,  0.219604,  0.0304871,  0.147705,   0.0252991, 0.867188,
      0.935547,  0.465576,  0.27832,   0.143921,  0.687012,  0.734375,  0.240112,   0.189697,   0.33667,   0.0227051,
      0.733398,  0.399414,  0.182251,  0.911133,  0.282227,  0.338135,  0.533691,   0.176636,   0.960449,  0.468506,
      0.0146561, 0.362793,  0.212891,  0.67041,   0.390381,  0.967285,  0.413818,   0.624023,   0.9375,    0.963379,
      0.858887,  0.811523,  0.949707,  0.393555,  0.778809,  0.955566,  0.0530701,  0.328857,   0.888672,  0.836914,
      0.471191,  0.579102,  0.938965,  0.0618591, 0.679688,  0.0807495, 0.52832,    0.00328064, 0.324951,  0.655762,
      0.191162,  0.440918,  0.149292,  0.213867,  0.227539,  0.864258,  0.00266457, 0.459473,   0.684082,  0.46875,
      0.316162,  0.191284,  0.855469,  0.231079,  0.359619,  0.558594,  0.777832,   0.725586,   0.0702515, 0.662109,
      0.39502,   0.0560608, 0.136353,  0.926758,  0.312256,  0.901855,  0.956055,   0.859863,   0.390381,  0.098938,
      0.63623,   0.166382,  0.92627,   0.643555,  0.27002,   0.473633,  0.462891,   0.349365,   0.729004,  0.356445,
      0.470459,  0.240479,  0.218384,  0.653809,  0.703613,  0.609375,  0.125122,   0.803223,   0.190674,  0.720703,
      0.939453,  0.47876,   0.927246,  0.498291,  0.741211,  0.723145,  0.236572,   0.645996,   0.574707,  0.660156,
      0.0468445, 0.0541992, 0.285889,  0.334473,  0.79834,   0.572266,  0.846191,   0.546387,   0.650391,  0.403076,
      0.439453,  0.44043,   0.709473,  0.441406,  0.675781,  0.23645,   0.90332,    0.702637,   0.447021,  0.297363,
      0.657715,  0.997559,  0.777344,  0.265137,  0.847656,  0.916016,  0.693848,   0.49292,    0.153076,  0.152222,
      0.915527,  0.0549927, 0.220703,  0.835449,  0.0680542, 0.470703,
  };
  TransResult result;
  DataTypeTransfer transfer;
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_FLOAT16, DT_FLOAT};
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  bool is_equal = true;
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    if (abs((reinterpret_cast<float *>(result.data.get()))[i] - ret[i]) > 1.0e-6) {
      is_equal = false;
      break;
    }
  }
  EXPECT_FLOAT_EQ(is_equal, true);

  TransResult result2;
  DataTypeTransfer transfer2;
  CastArgs args2{reinterpret_cast<uint8_t *>(ret), sizeof(data) / sizeof(data[0]), DT_FLOAT, DT_FLOAT16};
  EXPECT_EQ(transfer2.TransDataType(args2, result2), SUCCESS);
  EXPECT_EQ(result2.length, sizeof(data));
  for (size_t i = 0; i < sizeof(data) / sizeof(data[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<fp16_t *>(result2.data.get()))[i].val, data[i].val);
  }
}

TEST_F(UtestDataTypeTransfer, int8_fp32) {
  int8_t data[2 * 4 * 5 * 3] = {
      0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
      2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9,
      1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
      3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10,
  };

  float ret[2 * 4 * 5 * 3] = {
      0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
      2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9,
      1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
      3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10,
  };

  TransResult result;
  DataTypeTransfer transfer;
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_INT8, DT_FLOAT};
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<float *>(result.data.get()))[i], ret[i]);
  }
}

TEST_F(UtestDataTypeTransfer, int8_int32) {
  int8_t data[2 * 4 * 5 * 3] = {
      0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
      2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9,
      1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
      3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10,
  };

  int32_t ret[2 * 4 * 5 * 3] = {
      0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
      2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9,
      1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
      3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10,
  };

  TransResult result;
  DataTypeTransfer transfer;
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_INT8, DT_INT32};
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<int32_t *>(result.data.get()))[i], ret[i]);
  }
}

TEST_F(UtestDataTypeTransfer, uint8_fp32) {
  uint8_t data[1 * 4 * 4 * 3] = {
      2,  6,  1, 6, 11, 12, 30, 24, 4,  28, 22, 25, 20, 5,  18, 15, 23, 27, 1,  25, 26, 24, 11, 8,
      21, 15, 6, 5, 23, 17, 11, 18, 21, 24, 14, 20, 19, 12, 23, 16, 3,  9,  10, 3,  15, 31, 18, 9,
  };
  float ret[1 * 4 * 4 * 3] = {
      2,  6,  1, 6, 11, 12, 30, 24, 4,  28, 22, 25, 20, 5,  18, 15, 23, 27, 1,  25, 26, 24, 11, 8,
      21, 15, 6, 5, 23, 17, 11, 18, 21, 24, 14, 20, 19, 12, 23, 16, 3,  9,  10, 3,  15, 31, 18, 9,
  };

  CastArgs args{data, sizeof(ret) / sizeof(ret[0]), DT_UINT8, DT_FLOAT};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_EQ((reinterpret_cast<float *>(result.data.get()))[i], ret[i]);
  }
}

TEST_F(UtestDataTypeTransfer, uint8_int32) {
  uint8_t data[1 * 4 * 4 * 3] = {
      2,  6,  1, 6, 11, 12, 30, 24, 4,  28, 22, 25, 20, 5,  18, 15, 23, 27, 1,  25, 26, 24, 11, 8,
      21, 15, 6, 5, 23, 17, 11, 18, 21, 24, 14, 20, 19, 12, 23, 16, 3,  9,  10, 3,  15, 31, 18, 9,
  };
  float ret[1 * 4 * 4 * 3] = {
      2,  6,  1, 6, 11, 12, 30, 24, 4,  28, 22, 25, 20, 5,  18, 15, 23, 27, 1,  25, 26, 24, 11, 8,
      21, 15, 6, 5, 23, 17, 11, 18, 21, 24, 14, 20, 19, 12, 23, 16, 3,  9,  10, 3,  15, 31, 18, 9,
  };

  CastArgs args{data, sizeof(ret) / sizeof(ret[0]), DT_UINT8, DT_INT32};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_EQ((reinterpret_cast<int32_t *>(result.data.get()))[i], ret[i]);
  }
}

TEST_F(UtestDataTypeTransfer, fp32_int32) {
  float data[2 * 3 * 2 * 2] = {
      12069.558428388797, 15153.257385100667, 14984.26436591075,  14609.400052018833, 14685.809894536767,
      15086.047829821913, 14608.516342177387, 15045.212419446521, 14990.208555280951, 15160.085043556863,
      12590.929413431828, 13329.114266064971, 14156.37125633003,  13280.133356778959, 12436.203018490567,
      15326.305606200556, 14378.984205090384, 14309.322926704968, 15127.534200544495, 14504.079809440058,
      14404.89917121715,  10767.05264755489,  13679.223916928482, 14460.12063510443,
  };
  float ret[2 * 3 * 2 * 2] = {
      12069, 15153, 14984, 14609, 14685, 15086, 14608, 15045, 14990, 15160, 12590, 13329,
      14156, 13280, 12436, 15326, 14378, 14309, 15127, 14504, 14404, 10767, 13679, 14460,
  };
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_FLOAT, DT_INT32};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<int32_t *>(result.data.get()))[i], ret[i]);
  }
}

TEST_F(UtestDataTypeTransfer, int32_fp32) {
  int32_t data[1 * 3 * 4 * 4] = {
      15322, 14946, 12601, 14058, 12641, 14470, 14686, 15052, 11964, 14846, 13154, 13571, 14947, 12467, 12786, 14238,
      15334, 14814, 13608, 12163, 14169, 15274, 14780, 15303, 14986, 14841, 14290, 13785, 13878, 11576, 14411, 14522,
      14394, 13508, 13021, 14691, 13263, 15145, 14724, 15167, 14523, 13334, 14834, 13844, 9902,  14984, 15051, 14511,
  };
  float ret[1 * 3 * 4 * 4] = {
      15322, 14946, 12601, 14058, 12641, 14470, 14686, 15052, 11964, 14846, 13154, 13571, 14947, 12467, 12786, 14238,
      15334, 14814, 13608, 12163, 14169, 15274, 14780, 15303, 14986, 14841, 14290, 13785, 13878, 11576, 14411, 14522,
      14394, 13508, 13021, 14691, 13263, 15145, 14724, 15167, 14523, 13334, 14834, 13844, 9902,  14984, 15051, 14511,
  };
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_INT32, DT_FLOAT};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<float *>(result.data.get()))[i], ret[i]);
  }
}

TEST_F(UtestDataTypeTransfer, int32_uint8) {
  int32_t data[2 * 4 * 5 * 3] = {
      0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
      2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9,
      1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
      3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10,
  };

  uint8_t ret[2 * 4 * 5 * 3] = {
      0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
      2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9,
      1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
      3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10,
  };
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_INT32, DT_UINT8};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (size_t i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<uint8_t *>(result.data.get()))[i], ret[i]);
  }
}

TEST_F(UtestDataTypeTransfer, int32_int8) {
  int32_t data[2 * 4 * 5 * 3] = {
      0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
      2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9,
      1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
      3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10,
  };

  int8_t ret[2 * 4 * 5 * 3] = {
      0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7,
      2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9,
      1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
      3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10,
  };
  CastArgs args{reinterpret_cast<uint8_t *>(data), sizeof(ret) / sizeof(ret[0]), DT_INT32, DT_INT8};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), SUCCESS);
  EXPECT_EQ(result.length, sizeof(ret));
  for (int i = 0; i < sizeof(ret) / sizeof(ret[0]); ++i) {
    EXPECT_FLOAT_EQ((reinterpret_cast<int8_t *>(result.data.get()))[i], ret[i]);
  }
}

TEST_F(UtestDataTypeTransfer, invalid_src_data_type) {
  uint8_t data[1 * 4 * 4 * 1] = {0};
  CastArgs args{reinterpret_cast<uint8_t *>(data), 16, DT_UNDEFINED, DT_FLOAT};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), ACL_ERROR_GE_DATATYPE_INVALID);
}

/*
TEST_F(UtestDataTypeTransfer, src_shape_empry) {
  uint8_t data[1 * 4 * 4 * 1] = {0};

  DataTypeTransfer transfer;
  CastArgs args{reinterpret_cast<uint8_t *>(data), 0, DT_UINT8, DT_INT32};

  TransResult result;
  EXPECT_EQ(transfer.TransDataType(args, result), PARAM_INVALID);
}
*/

TEST_F(UtestDataTypeTransfer, unsupprot_trans) {
  bool data[1 * 4 * 4 * 1] = {0};
  CastArgs args{reinterpret_cast<uint8_t *>(data), 16, DT_BOOL, DT_INT8};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), ACL_ERROR_GE_DATATYPE_INVALID);
  EXPECT_EQ(TransDataType(args, result), ACL_ERROR_GE_DATATYPE_INVALID);
}

TEST_F(UtestDataTypeTransfer, unsupprot_trans2) {
  bool data[1 * 4 * 4 * 1] = {0};
  CastArgs args{reinterpret_cast<uint8_t *>(data), 16, DT_BOOL, DT_INT32};
  TransResult result;

  DataTypeTransfer transfer;
  EXPECT_EQ(transfer.TransDataType(args, result), ACL_ERROR_GE_DATATYPE_INVALID);
  EXPECT_EQ(TransDataType(args, result), ACL_ERROR_GE_DATATYPE_INVALID);
}
}  // namespace formats
}  // namespace ge
