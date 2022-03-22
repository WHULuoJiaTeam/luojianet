#include <gtest/gtest.h>
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_log.h"
#include <iostream>
using namespace std;

namespace ge {

class ge_test_tensor_utils : public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(ge_test_tensor_utils, shape) {
  GeTensorDesc tensorDesc;

  int64_t s1 = 1;
  int64_t s2 = 0;
  TensorUtils::SetSize(tensorDesc, s1);
  EXPECT_EQ(TensorUtils::GetSize(tensorDesc, s2), GRAPH_SUCCESS);
  EXPECT_EQ(s2, 1);
  TensorUtils::SetSize(tensorDesc, 2);
  EXPECT_EQ(TensorUtils::GetSize(tensorDesc, s2), GRAPH_SUCCESS);
  EXPECT_EQ(s2, 2);

  bool f1(true);
  bool f2(false);
  TensorUtils::SetReuseInput(tensorDesc, f1);
  EXPECT_EQ(TensorUtils::GetReuseInput(tensorDesc, f2), GRAPH_SUCCESS);
  EXPECT_EQ(f2, true);

  f1 = true;
  f2 = false;
  TensorUtils::SetOutputTensor(tensorDesc, f1);
  EXPECT_EQ(TensorUtils::GetOutputTensor(tensorDesc, f2), GRAPH_SUCCESS);
  EXPECT_EQ(f2, true);

  DeviceType d1(DeviceType::CPU);
  DeviceType d2(DeviceType::NPU);
  TensorUtils::SetDeviceType(tensorDesc, d1);
  EXPECT_EQ(TensorUtils::GetDeviceType(tensorDesc, d2), GRAPH_SUCCESS);
  EXPECT_EQ(d2, true);

  f1 = true;
  f2 = false;
  TensorUtils::SetInputTensor(tensorDesc, f1);
  EXPECT_EQ(TensorUtils::GetInputTensor(tensorDesc, f2), GRAPH_SUCCESS);
  EXPECT_EQ(f2, true);

  uint32_t s5 = 1;
  uint32_t s6 = 0;
  TensorUtils::SetRealDimCnt(tensorDesc, s5);
  EXPECT_EQ(TensorUtils::GetRealDimCnt(tensorDesc, s6), GRAPH_SUCCESS);
  EXPECT_EQ(s6, 1);

  s5 = 1;
  s6 = 0;
  TensorUtils::SetReuseInputIndex(tensorDesc, s5);
  EXPECT_EQ(TensorUtils::GetReuseInputIndex(tensorDesc, s6), GRAPH_SUCCESS);
  EXPECT_EQ(s6, 1);

  int64_t s3(1);
  int64_t s4(0);
  TensorUtils::SetDataOffset(tensorDesc, s3);
  EXPECT_EQ(TensorUtils::GetDataOffset(tensorDesc, s4), GRAPH_SUCCESS);
  EXPECT_EQ(s4, 1);

  s5 = 1;
  s6 = 0;
  TensorUtils::SetRC(tensorDesc, s5);
  EXPECT_EQ(TensorUtils::GetRC(tensorDesc, s6), GRAPH_SUCCESS);
  EXPECT_EQ(s6, 1);
  TensorUtils::SetRC(tensorDesc, 2);
  EXPECT_EQ(TensorUtils::GetRC(tensorDesc, s6), GRAPH_SUCCESS);
  EXPECT_EQ(s6, 2);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_failed_datatype_notsupport) {
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_NCHW;
  DataType data_type = DT_UNDEFINED;

  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_failed_format_notsupport) {
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_RESERVED;
  DataType data_type = DT_FLOAT16;

  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

// not 4 calc by nd
TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NCHW_shape_not_4) {
  vector<int64_t> dims({2, 3, 4, 5, 6});
  GeShape ge_shape(dims);
  Format format = FORMAT_NCHW;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = sizeof(uint16_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NCHW_SUCCESS) {
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_NCHW;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = sizeof(uint16_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

// not 4 calc by nd
TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NHWC_shape_not_4) {
  vector<int64_t> dims({2, 3, 4});
  GeShape ge_shape(dims);
  Format format = FORMAT_NHWC;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = sizeof(uint16_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NHWC_SUCCESS) {
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_NHWC;
  DataType data_type = DT_FLOAT;
  int64_t expect_mem_size = sizeof(float);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_ND_FAILED_overflow_with_type) {
  vector<int64_t> dims({1024 * 1024, 1024 * 1024, 1024 * 1024});
  GeShape ge_shape(dims);
  Format format = FORMAT_ND;
  DataType data_type = DT_UINT64;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_ND_FAILED_overflow) {
  vector<int64_t> dims({1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024});
  GeShape ge_shape(dims);
  Format format = FORMAT_ND;
  DataType data_type = DT_UINT64;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_ND_SUCCESS) {
  vector<int64_t> dims({10, 2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_ND;
  DataType data_type = DT_UINT64;
  int64_t expect_mem_size = sizeof(uint64_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_MD_SUCCESS) {
  vector<int64_t> dims({10, 20, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_MD;
  DataType data_type = DT_UINT32;
  int64_t expect_mem_size = sizeof(uint32_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NC1HWC0_shape_error_less) {
  vector<int64_t> dims({2, 3, 4});
  GeShape ge_shape(dims);
  Format format = FORMAT_NC1HWC0;
  DataType data_type = DT_INT8;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NC1HWC0_shape_error_more) {
  vector<int64_t> dims({2, 3, 4, 5, 6, 7});
  GeShape ge_shape(dims);
  Format format = FORMAT_NC1HWC0;
  DataType data_type = DT_UINT8;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NC1HWC0_SUCCESS_NONEEDPAD) {
  vector<int64_t> dims({10, 32, 3, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_NC1HWC0;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = sizeof(uint16_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NC1HWC0_SUCCESS_NEEDPAD) {
  vector<int64_t> dims({10, 20, 3, 5});
  vector<int64_t> pad_dims({10, 32, 3, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_NC1HWC0;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = sizeof(uint16_t);
  for (int64_t dim:pad_dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_NC1HWC0_SUCCESS_5D) {
  vector<int64_t> dims({10, 2, 3, 5, 16});
  GeShape ge_shape(dims);
  Format format = FORMAT_NC1HWC0;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = sizeof(uint16_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_ND_SUCCESS_size_0) {
  vector<int64_t> dims({10, 0, 3, 5, 16});
  GeShape ge_shape(dims);
  Format format = FORMAT_NC1HWC0;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = 0;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_ND_SUCCESS_size_unknown) {
  vector<int64_t> dims({10, -1, 3, 5, 16});
  GeShape ge_shape(dims);
  Format format = FORMAT_NC1HWC0;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = -1;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_C1HWNCoC0_SUCCESS) {
  vector<int64_t> dims({10, 2, 3, 5, 8, 16});
  GeShape ge_shape(dims);
  Format format = FORMAT_C1HWNCoC0;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = sizeof(uint16_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(mem_size, expect_mem_size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_CCE_FractalZ_shape_error) {
  setenv("PARSER_PRIORITY", "cce", 0);
  vector<int64_t> dims({2, 3, 4, 5, 6, 7});
  GeShape ge_shape(dims);
  Format format = FORMAT_FRACTAL_Z;
  DataType data_type = DT_UINT8;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  unsetenv("PARSER_PRIORITY");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_CCE_FractalZ_SUCCESS) {
  setenv("PARSER_PRIORITY", "cce", 0);
  vector<int64_t> dims({16, 16, 6, 7});
  GeShape ge_shape(dims);
  Format format = FORMAT_FRACTAL_Z;
  DataType data_type = DT_FLOAT16;
  int64_t expect_mem_size = sizeof(uint16_t);
  for (int64_t dim:dims) {
    expect_mem_size *= dim;
  }
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  unsetenv("PARSER_PRIORITY");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSize_TBE_FractalZ_OverFlow_FAIL) {
  vector<int64_t> dims({1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024});
  GeShape ge_shape(dims);
  Format format = FORMAT_FRACTAL_Z;
  DataType data_type = DT_FLOAT16;

  int64_t mem_size = 0;
  graphStatus ret =
    TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  //cout<<"mem_size:"<<mem_size<<endl;
  //cout<<" aa:" << aa <<endl;
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, GetTensorMemorySizeInBytes_SUCCESS) {
  GeTensorDesc tensorDesc;
  int64_t size;
//  MOCKER(TensorUtils::GetTensorSizeInBytes).stubs().will(returnValue(GRAPH_SUCCESS));
  graphStatus ret = TensorUtils::GetTensorMemorySizeInBytes(tensorDesc, size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, GetTensorMemorySizeInBytes_FAILED) {
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format;
  DataType data_type;
  GeTensorDesc tensorDesc(ge_shape, format, data_type);
  int64_t size;
//  MOCKER(TensorUtils::GetTensorSizeInBytes).stubs().will(returnValue(GRAPH_FAILED));
  graphStatus ret = TensorUtils::GetTensorMemorySizeInBytes(tensorDesc, size);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(ge_test_tensor_utils, GetTensorSizeInBytes_SUCCESS) {
  GeTensorDesc tensorDesc;
  int64_t size;
  int64_t memSize = 2;
//  MOCKER(TensorUtils::CalcTensorMemSize).stubs().with(any(),any(),any(),outBound(memSize)).will(returnValue(GRAPH_SUCCESS));
  graphStatus ret = TensorUtils::GetTensorSizeInBytes(tensorDesc, size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, GetTensorSizeInBytes_FAILED) {
  GeTensorDesc tensorDesc;
  int64_t size;
//  MOCKER(TensorUtils::CalcTensorMemSize).stubs().will(returnValue(GRAPH_FAILED));
  graphStatus ret = TensorUtils::GetTensorSizeInBytes(tensorDesc, size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, GetTensorSizeInBytes_NoTiling_SUCCESS) {
  GeTensorDesc tensorDesc(GeShape({1, -1}));
  tensorDesc.SetShapeRange({{1, 1}, {1, 10}});
  int64_t size;
  (void)AttrUtils::SetBool(&tensorDesc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  graphStatus ret = TensorUtils::GetTensorSizeInBytes(tensorDesc, size);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSizeFilterHwckTest) {
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_FILTER_HWCK;
  DataType data_type = DT_STRING;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSizeFractalZnRnn) {
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_FRACTAL_ZN_RNN;
  DataType data_type = DT_STRING;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSize(ge_shape, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CheckShapeByShapeRangeShapeRangeIsNull) {
  vector<int64_t> dims({2, 3, 4, 5, 6, 7, 8});
  GeShape ge_shape(dims);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  graphStatus ret =
      TensorUtils::CheckShapeByShapeRange(ge_shape, shape_range);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CheckShapeByShapeRangeFailTest) {
  vector<int64_t> dims({2, 3, 4, 5, 6, 7, 8});
  GeShape ge_shape(dims);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.push_back(std::make_pair<int64_t, int64_t>(1, 1));
  graphStatus ret =
      TensorUtils::CheckShapeByShapeRange(ge_shape, shape_range);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(ge_test_tensor_utils, CheckShapeByShapeRangeLeftRangeLessThan0) {
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.push_back(std::make_pair<int64_t, int64_t>(-1, 1));
  shape_range.push_back(std::make_pair<int64_t, int64_t>(2, 2));
  shape_range.push_back(std::make_pair<int64_t, int64_t>(3, 3));
  shape_range.push_back(std::make_pair<int64_t, int64_t>(4, 4));
  graphStatus ret =
      TensorUtils::CheckShapeByShapeRange(ge_shape, shape_range);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(ge_test_tensor_utils, CheckShapeByShapeRangeCurDimIsUnknownDim) {
  vector<int64_t> dims({-1, -1});
  GeShape ge_shape(dims);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.push_back(std::make_pair<int64_t, int64_t>(1, 1));
  shape_range.push_back(std::make_pair<int64_t, int64_t>(2, 2));
  graphStatus ret =
      TensorUtils::CheckShapeByShapeRange(ge_shape, shape_range);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CheckShapeByShapeRangeCurDimLessThanLeftRange) {
  vector<int64_t> dims({1, 2});
  GeShape ge_shape(dims);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.push_back(std::make_pair<int64_t, int64_t>(3, 3));
  shape_range.push_back(std::make_pair<int64_t, int64_t>(4, 4));
  graphStatus ret =
      TensorUtils::CheckShapeByShapeRange(ge_shape, shape_range);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(ge_test_tensor_utils, CheckShapeByShapeRangeRightRangeLessThan0) {
  vector<int64_t> dims({3, 4});
  GeShape ge_shape(dims);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.push_back(std::make_pair<int64_t, int64_t>(3, -3));
  shape_range.push_back(std::make_pair<int64_t, int64_t>(4, -4));
  graphStatus ret =
      TensorUtils::CheckShapeByShapeRange(ge_shape, shape_range);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(ge_test_tensor_utils, CheckShapeByShapeRangeCurDimGreaterThanRightRange) {
  vector<int64_t> dims({5, 6});
  GeShape ge_shape(dims);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.push_back(std::make_pair<int64_t, int64_t>(3, 3));
  shape_range.push_back(std::make_pair<int64_t, int64_t>(4, 4));
  graphStatus ret =
      TensorUtils::CheckShapeByShapeRange(ge_shape, shape_range);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSizeForNoTilingSuccess) {
  GeTensorDesc tensor;
  Format format = FORMAT_NCHW;
  DataType data_type = DT_STRING;
  int64_t mem_size = 0;
  graphStatus ret =
      TensorUtils::CalcTensorMemSizeForNoTiling(tensor, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ge_test_tensor_utils, CalcTensorMemSizeForNoTilingDimsSizeIs0) {
  vector<int64_t> dims({});
  GeShape ge_shape(dims);
  Format format = FORMAT_FRACTAL_Z;
  DataType data_type;
  int64_t mem_size = 0;
  GeTensorDesc tensor(ge_shape, format, data_type);
  graphStatus ret =
      TensorUtils::CalcTensorMemSizeForNoTiling(tensor, format, data_type, mem_size);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}
}
