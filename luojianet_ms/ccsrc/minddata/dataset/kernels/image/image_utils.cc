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
#include "minddata/dataset/kernels/image/image_utils.h"
#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include "utils/ms_utils.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/image/math_utils.h"
#include "minddata/dataset/kernels/image/resize_cubic_op.h"

#include "gdal_priv.h"
#include "gdal.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include "minddata/dataset/core/GDALOpenCV.h"
#include "minddata/dataset/core/GLCM_utils.h"


const int32_t MAX_INT_PRECISION = 16777216;  // float int precision is 16777216
const int32_t DEFAULT_NUM_HEIGHT = 1;
const int32_t DEFAULT_NUM_WIDTH = 1;

auto max_(uchar a,uchar b,uchar c)
{
    uchar temp=a;
	if(b>a) temp=b;
	if(temp<c) temp=c;
	return temp;
}

cv::Mat RGB2GRAY(cv::Mat src) {
    CV_Assert(src.channels() == 3);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Vec3b rgb;
    int r = src.rows;
    int c = src.cols;

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            rgb = src.at<cv::Vec3b>(i, j);
            uchar B = rgb[0]; uchar G = rgb[1]; uchar R = rgb[2];
            dst.at<uchar>(i, j) = max_(R, G ,B);
        }
    }
    return dst;
}

cv::Mat OLBP(cv::Mat img)
{
	cv::Mat result;
	result.create(img.rows - 2, img.cols - 2, img.type());

	result.setTo(0);

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			uchar center = img.at<uchar>(i, j);
			uchar code = 0;
			code |= (img.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (img.at<uchar>(i - 1, j) >= center) << 6;
			code |= (img.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (img.at<uchar>(i, j + 1) >= center) << 4;
			code |= (img.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (img.at<uchar>(i + 1, j) >= center) << 2;
			code |= (img.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (img.at<uchar>(i, j - 1) >= center) << 0;
			result.at<uchar>(i - 1, j - 1) = code;
		}
	}
	return result;
}

cv::Mat ELBP(cv::Mat img, int radius, int neighbors)
{
	cv::Mat result;
	result.create(img.rows - radius * 2, img.cols - radius * 2, img.type());
	result.setTo(0);

	for (int n = 0; n < neighbors; n++)
	{
		// sample points
		float x = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx * ty;
		// iterate through your data
		for (int i = radius; i < img.rows - radius; i++)
		{
			for (int j = radius; j < img.cols - radius; j++)
			{
				// calculate interpolated value
				float t = static_cast<float>(w1*img.at<uchar>(i + fy, j + fx) + w2 * img.at<uchar>(i + fy, j + cx) + w3 * img.at<uchar>(i + cy, j + fx) + w4 * img.at<uchar>(i + cy, j + cx));
				// floating point precision, so check some machine-dependent epsilon
				result.at<uchar>(i - radius, j - radius) += ((t > img.at<uchar>(i, j)) || (std::abs(t - img.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	return result;
}

int getHopCount(uchar i)
{
	uchar a[8] = { 0 };
	int cnt = 0;
	int k = 7;

	while (k)
	{
		a[k] = i & 1;
		i = i >> 1;
		--k;
	}

	for (int k = 0; k < 7; k++)
	{
		if (a[k] != a[k + 1])
			++cnt;
	}

	if (a[0] != a[7])
		++cnt;

	return cnt;
}

cv::Mat RILBP(cv::Mat img)
{
	uchar RITable[256];
	int temp;
	int val;
	cv::Mat result;
	result.create(img.rows - 2, img.cols - 2, img.type());
	result.setTo(0);

	for (int i = 0; i < 256; i++)
	{
		val = i;
		for (int j = 0; j < 7; j++)
		{
			temp = i >> 1;
			if (val > temp)
			{
				val = temp;
			}
		}
		RITable[i] = val;
	}

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			uchar center = img.at<uchar>(i, j);
			uchar code = 0;
			code |= (img.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (img.at<uchar>(i - 1, j) >= center) << 6;
			code |= (img.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (img.at<uchar>(i, j + 1) >= center) << 4;
			code |= (img.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (img.at<uchar>(i + 1, j) >= center) << 2;
			code |= (img.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (img.at<uchar>(i, j - 1) >= center) << 0;
			result.at<uchar>(i - 1, j - 1) = RITable[code];
		}
	}
	return result;
}

cv::Mat UniformLBP(cv::Mat img)
{
	uchar UTable[256];
	memset(UTable, 0, 256 * sizeof(uchar));
	uchar temp = 1;
	for (int i = 0; i < 256; i++)
	{
		if (getHopCount(i) <= 2)
		{
			UTable[i] = temp;
			++temp;
		}
	}
	cv::Mat result;
	result.create(img.rows - 2, img.cols - 2, img.type());

	result.setTo(0);

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			uchar center = img.at<uchar>(i, j);
			uchar code = 0;
			code |= (img.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (img.at<uchar>(i - 1, j) >= center) << 6;
			code |= (img.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (img.at<uchar>(i, j + 1) >= center) << 4;
			code |= (img.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (img.at<uchar>(i + 1, j) >= center) << 2;
			code |= (img.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (img.at<uchar>(i, j - 1) >= center) << 0;
			result.at<uchar>(i - 1, j - 1) = UTable[code];
		}
	}
	return result;
}

cv::Mat gabor_kernal_wiki(cv::Size ksize, double theta, double sigma,
	double lambd, double gamma, double psi, int ktype){

	double sigma_x = sigma;
	double sigma_y = sigma / gamma;

	int xmin, xmax, ymin, ymax;

	xmax = ksize.width / 2;
	ymax = ksize.height / 2;
	xmin = -xmax;
	ymin = -ymax;

	CV_Assert(ktype == CV_32F || ktype == CV_64F);

	cv::Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);

	for (int x = xmin; x <= xmax; x++)
	{
		for (int y = ymin; y <= ymax; y++)
		{
			double x_alpha = x * cos(theta) + y * sin(theta);
			double y_alpha = -x * sin(theta) + y * cos(theta);
			double exponent = exp(-0.5*(x_alpha*x_alpha / pow(sigma_x, 2) +
				y_alpha * y_alpha / pow(sigma_y, 2)
				)
			);
			double v = exponent * cos(2 * CV_PI / lambd * x_alpha + psi);
			if (ktype == CV_32F)
			{  
				kernel.at<float>(y + ymax, x + xmax) = (float)v;
			}
			else
				kernel.at<double>(y + ymax, x + xmax) = v;
		}
	}
	return kernel;
}

void  gabor_filter(bool if_opencv_kernal, cv::Mat gray_img, cv::Mat& gabor_img, cv::Mat gabor_tmp, 
                  int k, float sigma, float gamma, float lambda, float psi){
	int ddepth = CV_8U;
	double theta[4] = {
		0.0,
		CV_PI / 4,
		CV_PI / 2,
		CV_PI / 4 * 3,
	};

	cv::Size ksize = cv::Size(k, k); 
	cv::Mat gabor_kernel;
	gabor_img = cv::Mat::zeros(gray_img.size(), CV_8UC1);
	for (int i = 0; i < 4; i++)
	{
		if (if_opencv_kernal)
			gabor_kernel = cv::getGaborKernel(ksize, sigma, theta[i], lambda, gamma, psi, CV_32F);
		else
			gabor_kernel = gabor_kernal_wiki(ksize, theta[i], sigma, lambda, gamma, psi, CV_32F);

		cv::filter2D(gray_img, gabor_tmp, ddepth, gabor_kernel);

		cv::max(gabor_tmp, gabor_img, gabor_img);
	}
}

namespace luojianet_ms {
namespace dataset {
int GetCVInterpolationMode(InterpolationMode mode) {
  switch (mode) {
    case InterpolationMode::kLinear:
      return static_cast<int>(cv::InterpolationFlags::INTER_LINEAR);
    case InterpolationMode::kCubic:
      return static_cast<int>(cv::InterpolationFlags::INTER_CUBIC);
    case InterpolationMode::kArea:
      return static_cast<int>(cv::InterpolationFlags::INTER_AREA);
    case InterpolationMode::kNearestNeighbour:
      return static_cast<int>(cv::InterpolationFlags::INTER_NEAREST);
    default:
      return static_cast<int>(cv::InterpolationFlags::INTER_LINEAR);
  }
}

int GetCVBorderType(BorderType type) {
  switch (type) {
    case BorderType::kConstant:
      return static_cast<int>(cv::BorderTypes::BORDER_CONSTANT);
    case BorderType::kEdge:
      return static_cast<int>(cv::BorderTypes::BORDER_REPLICATE);
    case BorderType::kReflect:
      return static_cast<int>(cv::BorderTypes::BORDER_REFLECT101);
    case BorderType::kSymmetric:
      return static_cast<int>(cv::BorderTypes::BORDER_REFLECT);
    default:
      return static_cast<int>(cv::BorderTypes::BORDER_CONSTANT);
  }
}

Status GetConvertShape(ConvertMode convert_mode, const std::shared_ptr<CVTensor> &input_cv,
                       std::vector<dsize_t> *node) {
  std::vector<ConvertMode> one_channels = {ConvertMode::COLOR_BGR2GRAY, ConvertMode::COLOR_RGB2GRAY,
                                           ConvertMode::COLOR_BGRA2GRAY, ConvertMode::COLOR_RGBA2GRAY};
  std::vector<ConvertMode> three_channels = {
    ConvertMode::COLOR_BGRA2BGR, ConvertMode::COLOR_RGBA2RGB, ConvertMode::COLOR_RGBA2BGR, ConvertMode::COLOR_BGRA2RGB,
    ConvertMode::COLOR_BGR2RGB,  ConvertMode::COLOR_RGB2BGR,  ConvertMode::COLOR_GRAY2BGR, ConvertMode::COLOR_GRAY2RGB};
  std::vector<ConvertMode> four_channels = {ConvertMode::COLOR_BGR2BGRA,  ConvertMode::COLOR_RGB2RGBA,
                                            ConvertMode::COLOR_BGR2RGBA,  ConvertMode::COLOR_RGB2BGRA,
                                            ConvertMode::COLOR_BGRA2RGBA, ConvertMode::COLOR_RGBA2BGRA,
                                            ConvertMode::COLOR_GRAY2BGRA, ConvertMode::COLOR_GRAY2RGBA};
  if (std::find(three_channels.begin(), three_channels.end(), convert_mode) != three_channels.end()) {
    *node = {input_cv->shape()[0], input_cv->shape()[1], 3};
  } else if (std::find(four_channels.begin(), four_channels.end(), convert_mode) != four_channels.end()) {
    *node = {input_cv->shape()[0], input_cv->shape()[1], 4};
  } else if (std::find(one_channels.begin(), one_channels.end(), convert_mode) != one_channels.end()) {
    *node = {input_cv->shape()[0], input_cv->shape()[1]};
  } else {
    RETURN_STATUS_UNEXPECTED(
      "The mode of image channel conversion must be in ConvertMode, which mainly includes "
      "conversion between RGB, BGR, GRAY, RGBA etc.");
  }
  return Status::OK();
}

bool CheckTensorShape(const std::shared_ptr<Tensor> &tensor, const int &channel) {
  if (tensor == nullptr) {
    return false;
  }
  bool rc = false;
  if (tensor->shape().Size() <= channel) {
    return false;
  }
  if (tensor->Rank() != DEFAULT_IMAGE_RANK ||
      (tensor->shape()[channel] != 1 && tensor->shape()[channel] != DEFAULT_IMAGE_CHANNELS)) {
    rc = true;
  }
  return rc;
}

Status Flip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, int flip_code) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));

  if (input_cv->Rank() == 1 || input_cv->mat().dims > 2) {
    std::string err_msg =
      "Flip: shape of input is not <H,W,C> or <H,W>, but got rank:" + std::to_string(input_cv->Rank());
    if (input_cv->Rank() == 1) {
      err_msg = err_msg + ", may need to do Decode first.";
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::shared_ptr<CVTensor> output_cv;
  RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));

  if (input_cv->mat().data) {
    try {
      cv::flip(input_cv->mat(), output_cv->mat(), flip_code);
      *output = std::static_pointer_cast<Tensor>(output_cv);
      return Status::OK();
    } catch (const cv::Exception &e) {
      RETURN_STATUS_UNEXPECTED("Flip: " + std::string(e.what()));
    }
  } else {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Flip: allocate memory failed.");
  }
}

Status HorizontalFlip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  return Flip(std::move(input), output, 1);
}

Status VerticalFlip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  return Flip(std::move(input), output, 0);
}

Status Resize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t output_height,
              int32_t output_width, double fx, double fy, InterpolationMode mode) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Resize: load image failed.");
  }
  RETURN_IF_NOT_OK(ValidateImageRank("Resize", input_cv->Rank()));

  cv::Mat in_image = input_cv->mat();
  const uint32_t kResizeShapeLimits = 1000;
  // resize image too large or too small, 1000 is arbitrarily chosen here to prevent open cv from segmentation fault
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kResizeShapeLimits) > in_image.rows,
                               "Resize: in_image rows out of bounds.");
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kResizeShapeLimits) > in_image.cols,
                               "Resize: in_image cols out of bounds.");
  if (output_height > in_image.rows * kResizeShapeLimits || output_width > in_image.cols * kResizeShapeLimits) {
    std::string err_msg =
      "Resize: the resizing width or height is too big, it's 1000 times bigger than the original image, got output "
      "height: " +
      std::to_string(output_height) + ", width: " + std::to_string(output_width) +
      ", and original image size:" + std::to_string(in_image.rows) + ", " + std::to_string(in_image.cols);
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }
  if (output_height == 0 || output_width == 0) {
    std::string err_msg = "Resize: the input value of 'resize' is invalid, width or height is zero.";
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }

  if (mode == InterpolationMode::kCubicPil) {
    if (input_cv->shape().Size() != DEFAULT_IMAGE_CHANNELS ||
        input_cv->shape()[CHANNEL_INDEX] != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("Resize: Interpolation mode PILCUBIC only supports image with 3 channels, but got: " +
                               input_cv->shape().ToString());
    }

    LiteMat imIn, imOut;
    std::shared_ptr<Tensor> output_tensor;
    TensorShape new_shape = TensorShape({output_height, output_width, 3});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input_cv->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    imOut.Init(output_width, output_height, input_cv->shape()[2], reinterpret_cast<void *>(buffer), LDataType::UINT8);
    imIn.Init(input_cv->shape()[1], input_cv->shape()[0], input_cv->shape()[2], input_cv->mat().data, LDataType::UINT8);
    if (ResizeCubic(imIn, imOut, output_width, output_height) == false) {
      RETURN_STATUS_UNEXPECTED("Resize: failed to do resize, please check the error msg.");
    }
    *output = output_tensor;
    return Status::OK();
  }
  try {
    TensorShape shape{output_height, output_width};
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->Rank() == DEFAULT_IMAGE_RANK) shape = shape.AppendDim(num_channels);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(shape, input_cv->type(), &output_cv));

    auto cv_mode = GetCVInterpolationMode(mode);
    cv::resize(in_image, output_cv->mat(), cv::Size(output_width, output_height), fx, fy, cv_mode);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Resize: " + std::string(e.what()));
  }
}

bool IsNonEmptyJPEG(const std::shared_ptr<Tensor> &input) {
  const unsigned char *kJpegMagic = (unsigned char *)"\xFF\xD8\xFF";
  constexpr dsize_t kJpegMagicLen = 3;
  return input->SizeInBytes() > kJpegMagicLen && memcmp(input->GetBuffer(), kJpegMagic, kJpegMagicLen) == 0;
}

bool IsNonEmptyPNG(const std::shared_ptr<Tensor> &input) {
  const unsigned char *kPngMagic = (unsigned char *)"\x89\x50\x4E\x47";
  constexpr dsize_t kPngMagicLen = 4;
  return input->SizeInBytes() > kPngMagicLen && memcmp(input->GetBuffer(), kPngMagic, kPngMagicLen) == 0;
}

Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  if (IsNonEmptyJPEG(input)) {
    return JpegCropAndDecode(input, output);
  } else {
    return DecodeCv(input, output);
  }
}

Status DecodeCv(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Decode: load image failed.");
  }
  try {
    cv::Mat img_mat = cv::imdecode(input_cv->mat(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (img_mat.data == nullptr) {
      std::string err = "Decode: image decode failed.";
      RETURN_STATUS_UNEXPECTED(err);
    }
    cv::cvtColor(img_mat, img_mat, static_cast<int>(cv::COLOR_BGR2RGB));
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(img_mat, 3, &output_cv));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Decode: " + std::string(e.what()));
  }
}

static void JpegInitSource(j_decompress_ptr cinfo) {}

static boolean JpegFillInputBuffer(j_decompress_ptr cinfo) {
  if (cinfo->src->bytes_in_buffer == 0) {
    // Under ARM platform raise runtime_error may cause core problem,
    // so we catch runtime_error and just return FALSE.
    try {
      ERREXIT(cinfo, JERR_INPUT_EMPTY);
    } catch (std::runtime_error &e) {
      return FALSE;
    }
    return FALSE;
  }
  return TRUE;
}

static void JpegTermSource(j_decompress_ptr cinfo) {}

static void JpegSkipInputData(j_decompress_ptr cinfo, int64_t jump) {
  if (jump < 0) {
    return;
  }
  if (static_cast<size_t>(jump) > cinfo->src->bytes_in_buffer) {
    cinfo->src->bytes_in_buffer = 0;
    return;
  } else {
    cinfo->src->bytes_in_buffer -= jump;
    cinfo->src->next_input_byte += jump;
  }
}

void JpegSetSource(j_decompress_ptr cinfo, const void *data, int64_t datasize) {
  cinfo->src = static_cast<struct jpeg_source_mgr *>(
    (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo), JPOOL_PERMANENT, sizeof(struct jpeg_source_mgr)));
  cinfo->src->init_source = JpegInitSource;
  cinfo->src->fill_input_buffer = JpegFillInputBuffer;
#if defined(_WIN32) || defined(_WIN64) || defined(ENABLE_ARM32) || defined(__APPLE__)
  cinfo->src->skip_input_data = reinterpret_cast<void (*)(j_decompress_ptr, long)>(JpegSkipInputData);
#else
  cinfo->src->skip_input_data = JpegSkipInputData;
#endif
  cinfo->src->resync_to_restart = jpeg_resync_to_restart;
  cinfo->src->term_source = JpegTermSource;
  cinfo->src->bytes_in_buffer = datasize;
  cinfo->src->next_input_byte = static_cast<const JOCTET *>(data);
}

static Status JpegReadScanlines(jpeg_decompress_struct *const cinfo, int max_scanlines_to_read, JSAMPLE *buffer,
                                int buffer_size, int crop_w, int crop_w_aligned, int offset, int stride) {
  // scanlines will be read to this buffer first, must have the number
  // of components equal to the number of components in the image
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int64_t>::max() / cinfo->output_components) > crop_w_aligned,
                               "JpegReadScanlines: multiplication out of bounds.");
  int64_t scanline_size = crop_w_aligned * cinfo->output_components;
  std::vector<JSAMPLE> scanline(scanline_size);
  JSAMPLE *scanline_ptr = &scanline[0];
  while (cinfo->output_scanline < static_cast<unsigned int>(max_scanlines_to_read)) {
    int num_lines_read = 0;
    try {
      num_lines_read = jpeg_read_scanlines(cinfo, &scanline_ptr, 1);
    } catch (std::runtime_error &e) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Decode: image decode failed.");
    }
    if (cinfo->out_color_space == JCS_CMYK && num_lines_read > 0) {
      for (int i = 0; i < crop_w; ++i) {
        const int cmyk_pixel = 4 * i + offset;
        const int c = scanline_ptr[cmyk_pixel];
        const int m = scanline_ptr[cmyk_pixel + 1];
        const int y = scanline_ptr[cmyk_pixel + 2];
        const int k = scanline_ptr[cmyk_pixel + 3];
        int r, g, b;
        if (cinfo->saw_Adobe_marker) {
          r = (k * c) / 255;
          g = (k * m) / 255;
          b = (k * y) / 255;
        } else {
          r = (255 - c) * (255 - k) / 255;
          g = (255 - m) * (255 - k) / 255;
          b = (255 - y) * (255 - k) / 255;
        }
        buffer[3 * i + 0] = r;
        buffer[3 * i + 1] = g;
        buffer[3 * i + 2] = b;
      }
    } else if (num_lines_read > 0) {
      int copy_status = memcpy_s(buffer, buffer_size, scanline_ptr + offset, stride);
      if (copy_status != 0) {
        jpeg_destroy_decompress(cinfo);
        RETURN_STATUS_UNEXPECTED("[Internal ERROR] Decode: memcpy failed.");
      }
    } else {
      jpeg_destroy_decompress(cinfo);
      std::string err_msg = "[Internal ERROR] Decode: image decode failed.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    buffer += stride;
    buffer_size = buffer_size - stride;
  }
  return Status::OK();
}

static Status JpegSetColorSpace(jpeg_decompress_struct *cinfo) {
  switch (cinfo->num_components) {
    case 1:
      // we want to output 3 components if it's grayscale
      cinfo->out_color_space = JCS_RGB;
      return Status::OK();
    case 3:
      cinfo->out_color_space = JCS_RGB;
      return Status::OK();
    case 4:
      // Need to manually convert to RGB
      cinfo->out_color_space = JCS_CMYK;
      return Status::OK();
    default:
      jpeg_destroy_decompress(cinfo);
      std::string err_msg = "[Internal ERROR] Decode: image decode failed.";
      RETURN_STATUS_UNEXPECTED(err_msg);
  }
}

void JpegErrorExitCustom(j_common_ptr cinfo) {
  char jpeg_last_error_msg[JMSG_LENGTH_MAX];
  (*(cinfo->err->format_message))(cinfo, jpeg_last_error_msg);
  throw std::runtime_error(jpeg_last_error_msg);
}

Status JpegCropAndDecode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int crop_x, int crop_y,
                         int crop_w, int crop_h) {
  struct jpeg_decompress_struct cinfo;
  auto DestroyDecompressAndReturnError = [&cinfo](const std::string &err) {
    jpeg_destroy_decompress(&cinfo);
    RETURN_STATUS_UNEXPECTED(err);
  };
  struct JpegErrorManagerCustom jerr;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = JpegErrorExitCustom;
  try {
    jpeg_create_decompress(&cinfo);
    JpegSetSource(&cinfo, input->GetBuffer(), input->SizeInBytes());
    (void)jpeg_read_header(&cinfo, TRUE);
    RETURN_IF_NOT_OK(JpegSetColorSpace(&cinfo));
    jpeg_calc_output_dimensions(&cinfo);
  } catch (std::runtime_error &e) {
    return DestroyDecompressAndReturnError(e.what());
  }
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - crop_w) > crop_x,
                               "JpegCropAndDecode: addition(crop x and crop width) out of bounds, got crop x:" +
                                 std::to_string(crop_x) + ", and crop width:" + std::to_string(crop_w));
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - crop_h) > crop_y,
                               "JpegCropAndDecode: addition(crop y and crop height) out of bounds, got crop y:" +
                                 std::to_string(crop_y) + ", and crop height:" + std::to_string(crop_h));
  if (crop_x == 0 && crop_y == 0 && crop_w == 0 && crop_h == 0) {
    crop_w = cinfo.output_width;
    crop_h = cinfo.output_height;
  } else if (crop_w == 0 || static_cast<unsigned int>(crop_w + crop_x) > cinfo.output_width || crop_h == 0 ||
             static_cast<unsigned int>(crop_h + crop_y) > cinfo.output_height) {
    return DestroyDecompressAndReturnError(
      "Crop: invalid crop size, corresponding crop value equal to 0 or too big, got crop width: " +
      std::to_string(crop_w) + ", crop height:" + std::to_string(crop_h) +
      ", and crop x coordinate:" + std::to_string(crop_x) + ", crop y coordinate:" + std::to_string(crop_y));
  }
  const int mcu_size = cinfo.min_DCT_scaled_size;
  CHECK_FAIL_RETURN_UNEXPECTED(mcu_size != 0, "JpegCropAndDecode: divisor mcu_size is zero.");
  unsigned int crop_x_aligned = (crop_x / mcu_size) * mcu_size;
  unsigned int crop_w_aligned = crop_w + crop_x - crop_x_aligned;
  try {
    (void)jpeg_start_decompress(&cinfo);
    jpeg_crop_scanline(&cinfo, &crop_x_aligned, &crop_w_aligned);
  } catch (std::runtime_error &e) {
    return DestroyDecompressAndReturnError(e.what());
  }
  JDIMENSION skipped_scanlines = jpeg_skip_scanlines(&cinfo, crop_y);
  // three number of output components, always convert to RGB and output
  constexpr int kOutNumComponents = 3;
  TensorShape ts = TensorShape({crop_h, crop_w, kOutNumComponents});
  std::shared_ptr<Tensor> output_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(ts, DataType(DataType::DE_UINT8), &output_tensor));
  const int buffer_size = output_tensor->SizeInBytes();
  JSAMPLE *buffer = reinterpret_cast<JSAMPLE *>(&(*output_tensor->begin<uint8_t>()));
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() - skipped_scanlines) > crop_h,
                               "JpegCropAndDecode: addition out of bounds.");
  const int max_scanlines_to_read = skipped_scanlines + crop_h;
  // stride refers to output tensor, which has 3 components at most
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() / crop_w) > kOutNumComponents,
                               "JpegCropAndDecode: multiplication out of bounds.");
  const int stride = crop_w * kOutNumComponents;
  // offset is calculated for scanlines read from the image, therefore
  // has the same number of components as the image
  int minius_value = crop_x - crop_x_aligned;
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / minius_value) > cinfo.output_components,
                               "JpegCropAndDecode: multiplication out of bounds.");
  const int offset = minius_value * cinfo.output_components;
  RETURN_IF_NOT_OK(
    JpegReadScanlines(&cinfo, max_scanlines_to_read, buffer, buffer_size, crop_w, crop_w_aligned, offset, stride));
  *output = output_tensor;
  jpeg_destroy_decompress(&cinfo);
  return Status::OK();
}

Status Rescale(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rescale, float shift) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Rescale: load image failed.");
  }
  cv::Mat input_image = input_cv->mat();
  std::shared_ptr<CVTensor> output_cv;
  RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), DataType(DataType::DE_FLOAT32), &output_cv));
  try {
    input_image.convertTo(output_cv->mat(), CV_32F, rescale, shift);
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Rescale: " + std::string(e.what()));
  }
  return Status::OK();
}

Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Crop: load image failed.");
  }
  RETURN_IF_NOT_OK(ValidateImageRank("Crop", input_cv->Rank()));
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - y) > h,
                               "Crop: addition(x and height) out of bounds, got height:" + std::to_string(h) +
                                 ", and coordinate y:" + std::to_string(y));
  // account for integer overflow
  if (y < 0 || (y + h) > input_cv->shape()[0] || (y + h) < 0) {
    RETURN_STATUS_UNEXPECTED(
      "Crop: invalid y coordinate value for crop, y coordinate value exceeds the boundary of the image, got y: " +
      std::to_string(y));
  }
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - x) > w, "Crop: addition out of bounds.");
  // account for integer overflow
  if (x < 0 || (x + w) > input_cv->shape()[1] || (x + w) < 0) {
    RETURN_STATUS_UNEXPECTED(
      "Crop: invalid x coordinate value for crop, "
      "x coordinate value exceeds the boundary of the image, got x: " +
      std::to_string(x));
  }
  try {
    TensorShape shape{h, w};
    if (input_cv->Rank() == DEFAULT_IMAGE_RANK) {
      int num_channels = input_cv->shape()[CHANNEL_INDEX];
      shape = shape.AppendDim(num_channels);
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(shape, input_cv->type(), &output_cv));
    cv::Rect roi(x, y, w, h);
    (input_cv->mat())(roi).copyTo(output_cv->mat());
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Crop: " + std::string(e.what()));
  }
}

Status ConvertColor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, ConvertMode convert_mode) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    RETURN_IF_NOT_OK(ValidateImageRank("ConvertColor", input_cv->Rank()));
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] ConvertColor: load image failed.");
    }
    if (input_cv->Rank() == DEFAULT_IMAGE_RANK) {
      int num_channels = input_cv->shape()[CHANNEL_INDEX];
      if (num_channels != DEFAULT_IMAGE_CHANNELS && num_channels != MAX_IMAGE_CHANNELS) {
        RETURN_STATUS_UNEXPECTED("ConvertColor: number of channels of image should be 3 or 4, but got:" +
                                 std::to_string(num_channels));
      }
    }
    std::vector<dsize_t> node;
    RETURN_IF_NOT_OK(GetConvertShape(convert_mode, input_cv, &node));
    if (node.empty()) {
      RETURN_STATUS_UNEXPECTED(
        "ConvertColor: convert mode must be in ConvertMode, which mainly includes conversion "
        "between RGB, BGR, GRAY, RGBA etc.");
    }
    TensorShape out_shape = TensorShape(node);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(out_shape, input_cv->type(), &output_cv));
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(convert_mode));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("ConvertColor: " + std::string(e.what()));
  }
}

Status HwcToChw(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] HWC2CHW: load image failed.");
    }
    if (input_cv->Rank() == 2) {
      // If input tensor is 2D, we assume we have hw dimensions
      *output = input;
      return Status::OK();
    }
    CHECK_FAIL_RETURN_UNEXPECTED(input_cv->shape().Size() > CHANNEL_INDEX,
                                 "HWC2CHW: rank of input data should be greater than:" + std::to_string(CHANNEL_INDEX) +
                                   ", but got:" + std::to_string(input_cv->shape().Size()));
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->shape().Size() != DEFAULT_IMAGE_RANK) {
      RETURN_STATUS_UNEXPECTED("HWC2CHW: image shape should be <H,W,C>, but got rank: " +
                               std::to_string(input_cv->shape().Size()));
    }
    cv::Mat output_img;

    int height = input_cv->shape()[0];
    int width = input_cv->shape()[1];

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(TensorShape{num_channels, height, width}, input_cv->type(), &output_cv));
    for (int i = 0; i < num_channels; ++i) {
      cv::Mat mat;
      RETURN_IF_NOT_OK(output_cv->MatAtIndex({i}, &mat));
      cv::extractChannel(input_cv->mat(), mat, i);
    }
    *output = std::move(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("HWC2CHW: " + std::string(e.what()));
  }
}

Status MaskWithTensor(const std::shared_ptr<Tensor> &sub_mat, std::shared_ptr<Tensor> *input, int x, int y,
                      int crop_width, int crop_height, ImageFormat image_format) {
  if (image_format == ImageFormat::HWC) {
    if (CheckTensorShape(*input, 2)) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "input shape doesn't match <H,W,C> format, got shape:" +
        (*input)->shape().ToString());
    }
    if (CheckTensorShape(sub_mat, 2)) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "sub_mat shape doesn't match <H,W,C> format, got shape:" +
        (*input)->shape().ToString());
    }
    int number_of_channels = (*input)->shape()[CHANNEL_INDEX];
    for (int i = 0; i < crop_width; i++) {
      for (int j = 0; j < crop_height; j++) {
        for (int c = 0; c < number_of_channels; c++) {
          RETURN_IF_NOT_OK(CopyTensorValue(sub_mat, input, {j, i, c}, {y + j, x + i, c}));
        }
      }
    }
  } else if (image_format == ImageFormat::CHW) {
    if (CheckTensorShape(*input, 0)) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "input shape doesn't match <C,H,W> format, got shape:" +
        (*input)->shape().ToString());
    }
    if (CheckTensorShape(sub_mat, 0)) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "sub_mat shape doesn't match <C,H,W> format, got shape:" +
        (*input)->shape().ToString());
    }
    int number_of_channels = (*input)->shape()[0];
    for (int i = 0; i < crop_width; i++) {
      for (int j = 0; j < crop_height; j++) {
        for (int c = 0; c < number_of_channels; c++) {
          RETURN_IF_NOT_OK(CopyTensorValue(sub_mat, input, {c, j, i}, {c, y + j, x + i}));
        }
      }
    }
  } else if (image_format == ImageFormat::HW) {
    if ((*input)->Rank() != MIN_IMAGE_DIMENSION) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "input shape doesn't match <H,W> format, got shape:" +
        (*input)->shape().ToString());
    }
    if (sub_mat->Rank() != MIN_IMAGE_DIMENSION) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "sub_mat shape doesn't match <H,W> format, got shape:" +
        (*input)->shape().ToString());
    }
    for (int i = 0; i < crop_width; i++) {
      for (int j = 0; j < crop_height; j++) {
        RETURN_IF_NOT_OK(CopyTensorValue(sub_mat, input, {j, i}, {y + j, x + i}));
      }
    }
  } else {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: MaskWithTensor failed: "
      "image format must be <C,H,W>, <H,W,C>, or <H,W>, got shape:" +
      (*input)->shape().ToString());
  }
  return Status::OK();
}

Status CopyTensorValue(const std::shared_ptr<Tensor> &source_tensor, std::shared_ptr<Tensor> *dest_tensor,
                       const std::vector<int64_t> &source_indx, const std::vector<int64_t> &dest_indx) {
  if (source_tensor->type() != (*dest_tensor)->type())
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: CopyTensorValue failed: "
      "source and destination tensor must have the same type.");
  if (source_tensor->type() == DataType::DE_UINT8) {
    uint8_t pixel_value = 0;
    RETURN_IF_NOT_OK(source_tensor->GetItemAt(&pixel_value, source_indx));
    RETURN_IF_NOT_OK((*dest_tensor)->SetItemAt(dest_indx, pixel_value));
  } else if (source_tensor->type() == DataType::DE_FLOAT32) {
    float pixel_value = 0;
    RETURN_IF_NOT_OK(source_tensor->GetItemAt(&pixel_value, source_indx));
    RETURN_IF_NOT_OK((*dest_tensor)->SetItemAt(dest_indx, pixel_value));
  } else {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: CopyTensorValue failed: "
      "Tensor type is not supported. Tensor type must be float32 or uint8.");
  }
  return Status::OK();
}

Status SwapRedAndBlue(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    CHECK_FAIL_RETURN_UNEXPECTED(
      input_cv->shape().Size() > CHANNEL_INDEX,
      "SwapRedAndBlue: rank of input is should greater than:" + std::to_string(CHANNEL_INDEX) +
        ", but got:" + std::to_string(input_cv->shape().Size()));
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->shape().Size() != 3 || num_channels != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("SwapRedBlue: image shape should be in <H,W,C> format, but got:" +
                               input_cv->shape().ToString());
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));

    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_BGR2RGB));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("SwapRedBlue: " + std::string(e.what()));
  }
}

Status CropAndResize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y,
                     int crop_height, int crop_width, int target_height, int target_width, InterpolationMode mode) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] CropAndResize: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("CropAndResize", input_cv->Rank()));
    // image too large or too small, 1000 is arbitrary here to prevent opencv from segmentation fault
    const uint32_t kCropShapeLimits = 1000;
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kCropShapeLimits) > crop_height,
                                 "CropAndResize: crop_height out of bounds.");
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kCropShapeLimits) > crop_width,
                                 "CropAndResize: crop_width out of bounds.");
    if (crop_height == 0 || crop_width == 0 || target_height == 0 || target_height > crop_height * kCropShapeLimits ||
        target_width == 0 || target_width > crop_width * kCropShapeLimits) {
      std::string err_msg =
        "CropAndResize: the resizing width or height 1) is too big, it's up to " + std::to_string(kCropShapeLimits) +
        " times the original image; 2) can not be 0. Detail info is: crop_height: " + std::to_string(crop_height) +
        ", crop_width: " + std::to_string(crop_width) + ", target_height: " + std::to_string(target_height) +
        ", target_width: " + std::to_string(target_width);
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    cv::Rect roi(x, y, crop_width, crop_height);
    auto cv_mode = GetCVInterpolationMode(mode);
    cv::Mat cv_in = input_cv->mat();

    if (mode == InterpolationMode::kCubicPil) {
      if (input_cv->shape().Size() != DEFAULT_IMAGE_CHANNELS ||
          input_cv->shape()[CHANNEL_INDEX] != DEFAULT_IMAGE_CHANNELS) {
        RETURN_STATUS_UNEXPECTED(
          "CropAndResize: Interpolation mode PILCUBIC only supports image with 3 channels, but got: " +
          input_cv->shape().ToString());
      }

      cv::Mat input_roi = cv_in(roi);
      std::shared_ptr<CVTensor> input_image;
      RETURN_IF_NOT_OK(CVTensor::CreateFromMat(input_roi, input_cv->Rank(), &input_image));
      LiteMat imIn, imOut;
      std::shared_ptr<Tensor> output_tensor;
      TensorShape new_shape = TensorShape({target_height, target_width, 3});
      RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input_cv->type(), &output_tensor));
      uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
      imOut.Init(target_width, target_height, input_cv->shape()[2], reinterpret_cast<void *>(buffer), LDataType::UINT8);
      imIn.Init(input_image->shape()[1], input_image->shape()[0], input_image->shape()[2], input_image->mat().data,
                LDataType::UINT8);
      if (ResizeCubic(imIn, imOut, target_width, target_height) == false) {
        RETURN_STATUS_UNEXPECTED("Resize: failed to do resize, please check the error msg.");
      }
      *output = output_tensor;
      return Status::OK();
    }

    TensorShape shape{target_height, target_width};
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->Rank() == DEFAULT_IMAGE_RANK) {
      shape = shape.AppendDim(num_channels);
    }
    std::shared_ptr<CVTensor> cvt_out;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(shape, input_cv->type(), &cvt_out));
    cv::resize(cv_in(roi), cvt_out->mat(), cv::Size(target_width, target_height), 0, 0, cv_mode);
    *output = std::static_pointer_cast<Tensor>(cvt_out);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("CropAndResize: " + std::string(e.what()));
  }
}

Status Rotate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> center,
              float degree, InterpolationMode interpolation, bool expand, uint8_t fill_r, uint8_t fill_g,
              uint8_t fill_b) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Rotate: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("Rotate", input_cv->Rank()));

    cv::Mat input_img = input_cv->mat();
    if (input_img.cols > (MAX_INT_PRECISION * 2) || input_img.rows > (MAX_INT_PRECISION * 2)) {
      RETURN_STATUS_UNEXPECTED("Rotate: image is too large and center is not precise, got image width:" +
                               std::to_string(input_img.cols) + ", and image height:" + std::to_string(input_img.rows) +
                               ", both should be small than:" + std::to_string(MAX_INT_PRECISION * 2));
    }
    float fx = 0, fy = 0;
    if (center.empty()) {
      // default to center of image
      fx = (input_img.cols - 1) / 2.0;
      fy = (input_img.rows - 1) / 2.0;
    } else {
      fx = center[0];
      fy = center[1];
    }
    cv::Mat output_img;
    cv::Scalar fill_color = cv::Scalar(fill_b, fill_g, fill_r);
    // maybe don't use uint32 for image dimension here
    cv::Point2f pc(fx, fy);
    cv::Mat rot = cv::getRotationMatrix2D(pc, degree, 1.0);
    std::shared_ptr<CVTensor> output_cv;
    if (!expand) {
      // this case means that the shape doesn't change, size stays the same
      // We may not need this memcpy if it is in place.
      RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
      // using inter_nearest to comply with python default
      cv::warpAffine(input_img, output_cv->mat(), rot, input_img.size(), GetCVInterpolationMode(interpolation),
                     cv::BORDER_CONSTANT, fill_color);
    } else {
      // we resize here since the shape changes
      // create a new bounding box with the rotate
      cv::Rect2f bbox = cv::RotatedRect(pc, input_img.size(), degree).boundingRect2f();
      rot.at<double>(0, 2) += bbox.width / 2.0 - input_img.cols / 2.0;
      rot.at<double>(1, 2) += bbox.height / 2.0 - input_img.rows / 2.0;
      // use memcpy and don't compute the new shape since openCV has a rounding problem
      cv::warpAffine(input_img, output_img, rot, bbox.size(), GetCVInterpolationMode(interpolation),
                     cv::BORDER_CONSTANT, fill_color);
      RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
      RETURN_UNEXPECTED_IF_NULL(output_cv);
    }
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Rotate: " + std::string(e.what()));
  }
  return Status::OK();
}

template <typename T>
void Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> mean,
               std::vector<float> std) {
  auto itr_out = (*output)->begin<float>();
  auto itr = input->begin<T>();
  auto end = input->end<T>();
  int64_t num_channels = (*output)->shape()[CHANNEL_INDEX];

  while (itr != end) {
    for (int64_t i = 0; i < num_channels; i++) {
      *itr_out = static_cast<float>(*itr) / std[i] - mean[i];
      ++itr_out;
      ++itr;
    }
  }
}

Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> mean,
                 std::vector<float> std) {
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), DataType(DataType::DE_FLOAT32), output));
  if (input->Rank() == MIN_IMAGE_DIMENSION) {
    RETURN_IF_NOT_OK((*output)->ExpandDim(MIN_IMAGE_DIMENSION));
  }

  CHECK_FAIL_RETURN_UNEXPECTED((*output)->Rank() == DEFAULT_IMAGE_RANK,
                               "Normalize: output image rank should be:" + std::to_string(DEFAULT_IMAGE_RANK) +
                                 ", but got:" + std::to_string((*output)->Rank()));
  CHECK_FAIL_RETURN_UNEXPECTED(std.size() == mean.size(),
                               "Normalize: mean and std vectors are not of same size, got size of std:" +
                                 std::to_string(std.size()) + ", and mean size:" + std::to_string(mean.size()));

  // caller provided 1 mean/std value and there are more than one channel --> duplicate mean/std value
  if (mean.size() == 1 && (*output)->shape()[CHANNEL_INDEX] != 1) {
    for (int64_t i = 0; i < (*output)->shape()[CHANNEL_INDEX] - 1; i++) {
      mean.push_back(mean[0]);
      std.push_back(std[0]);
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED((*output)->shape()[CHANNEL_INDEX] == mean.size(),
                               "Normalize: number of channels does not match the size of mean and std vectors, got "
                               "channels: " +
                                 std::to_string((*output)->shape()[CHANNEL_INDEX]) +
                                 ", size of mean:" + std::to_string(mean.size()));

  switch (input->type().value()) {
    case DataType::DE_BOOL:
      Normalize<bool>(input, output, mean, std);
      break;
    case DataType::DE_INT8:
      Normalize<int8_t>(input, output, mean, std);
      break;
    case DataType::DE_UINT8:
      Normalize<uint8_t>(input, output, mean, std);
      break;
    case DataType::DE_INT16:
      Normalize<int16_t>(input, output, mean, std);
      break;
    case DataType::DE_UINT16:
      Normalize<uint16_t>(input, output, mean, std);
      break;
    case DataType::DE_INT32:
      Normalize<int32_t>(input, output, mean, std);
      break;
    case DataType::DE_UINT32:
      Normalize<uint32_t>(input, output, mean, std);
      break;
    case DataType::DE_INT64:
      Normalize<int64_t>(input, output, mean, std);
      break;
    case DataType::DE_UINT64:
      Normalize<uint64_t>(input, output, mean, std);
      break;
    case DataType::DE_FLOAT16:
      Normalize<float16>(input, output, mean, std);
      break;
    case DataType::DE_FLOAT32:
      Normalize<float>(input, output, mean, std);
      break;
    case DataType::DE_FLOAT64:
      Normalize<double>(input, output, mean, std);
      break;
    default:
      RETURN_STATUS_UNEXPECTED(
        "Normalize: unsupported type, currently supported types include "
        "[bool,int8_t,uint8_t,int16_t,uint16_t,int32_t,uint32_t,int64_t,uint64_t,float16,float,double].");
  }

  if (input->Rank() == MIN_IMAGE_DIMENSION) {
    (*output)->Squeeze();
  }
  return Status::OK();
}

Status NormalizePad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                    const std::shared_ptr<Tensor> &mean, const std::shared_ptr<Tensor> &std, const std::string &dtype) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!(input_cv->mat().data && input_cv->Rank() == DEFAULT_IMAGE_CHANNELS)) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] NormalizePad: load image failed.");
  }
  DataType tensor_type = DataType(DataType::DE_FLOAT32);
  int compute_type = CV_32F;
  int channel_type = CV_32FC1;
  if (dtype == "float16") {
    compute_type = CV_16F;
    channel_type = CV_16FC1;
    tensor_type = DataType(DataType::DE_FLOAT16);
  }
  cv::Mat in_image = input_cv->mat();
  std::shared_ptr<CVTensor> output_cv;
  TensorShape new_shape({input_cv->shape()[0], input_cv->shape()[1], input_cv->shape()[2] + 1});
  RETURN_IF_NOT_OK(CVTensor::CreateEmpty(new_shape, tensor_type, &output_cv));
  mean->Squeeze();
  if (mean->type() != DataType::DE_FLOAT32 || mean->Rank() != 1 || mean->shape()[0] != DEFAULT_IMAGE_CHANNELS) {
    std::string err_msg =
      "NormalizePad: mean tensor should be of size 3 and type float, but got rank: " + std::to_string(mean->Rank()) +
      ", and type: " + mean->type().ToString();
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }
  std->Squeeze();
  if (std->type() != DataType::DE_FLOAT32 || std->Rank() != 1 || std->shape()[0] != DEFAULT_IMAGE_CHANNELS) {
    std::string err_msg =
      "NormalizePad: std tensor should be of size 3 and type float, but got rank: " + std::to_string(std->Rank()) +
      ", and type: " + std->type().ToString();
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }
  try {
    // NOTE: We are assuming the input image is in RGB and the mean
    // and std are in RGB
    std::vector<cv::Mat> rgb;
    cv::split(in_image, rgb);
    if (rgb.size() != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("NormalizePad: input image is not in RGB, got rank: " + std::to_string(in_image.dims));
    }
    for (int8_t i = 0; i < DEFAULT_IMAGE_CHANNELS; i++) {
      float mean_c, std_c;
      RETURN_IF_NOT_OK(mean->GetItemAt<float>(&mean_c, {i}));
      RETURN_IF_NOT_OK(std->GetItemAt<float>(&std_c, {i}));
      rgb[i].convertTo(rgb[i], compute_type, 1.0 / std_c, (-mean_c / std_c));
    }
    rgb.push_back(cv::Mat::zeros(in_image.rows, in_image.cols, channel_type));
    cv::merge(rgb, output_cv->mat());
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("NormalizePad: " + std::string(e.what()));
  }
}

Status AdjustBrightness(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustBrightness: load image failed.");
    }
    CHECK_FAIL_RETURN_UNEXPECTED(
      input_cv->shape().Size() > CHANNEL_INDEX,
      "AdjustBrightness: image rank should not bigger than:" + std::to_string(CHANNEL_INDEX) +
        ", but got: " + std::to_string(input_cv->shape().Size()));
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    // Rank of the image represents how many dimensions, image is expected to be HWC
    if (input_cv->Rank() != DEFAULT_IMAGE_RANK || num_channels != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("AdjustBrightness: image shape is not <H,W,C> or channel is not 3, got image rank: " +
                               std::to_string(input_cv->Rank()) + ", and channel:" + std::to_string(num_channels));
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    output_cv->mat() = input_img * alpha;
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustBrightness: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AdjustContrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustContrast: load image failed.");
    }
    CHECK_FAIL_RETURN_UNEXPECTED(input_cv->shape().Size() > CHANNEL_INDEX,
                                 "AdjustContrast: image rank should bigger than:" + std::to_string(CHANNEL_INDEX) +
                                   ", but got: " + std::to_string(input_cv->shape().Size()));
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->Rank() != DEFAULT_IMAGE_CHANNELS || num_channels != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("AdjustContrast: image shape is not <H,W,C> or channel is not 3, got image rank: " +
                               std::to_string(input_cv->Rank()) + ", and channel:" + std::to_string(num_channels));
    }
    cv::Mat gray, output_img;
    cv::cvtColor(input_img, gray, CV_RGB2GRAY);
    int mean_img = static_cast<int>(cv::mean(gray).val[0] + 0.5);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    output_img = cv::Mat::zeros(input_img.rows, input_img.cols, CV_8UC1);
    output_img = output_img + mean_img;
    cv::cvtColor(output_img, output_img, CV_GRAY2RGB);
    output_cv->mat() = output_img * (1.0 - alpha) + input_img * alpha;
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustContrast: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AdjustGamma(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &gamma,
                   const float &gain) {
  try {
    int num_channels = 1;
    if (input->Rank() < MIN_IMAGE_DIMENSION) {
      RETURN_STATUS_UNEXPECTED("AdjustGamma: input tensor is not in shape of <...,H,W,C> or <H,W>, got shape:" +
                               input->shape().ToString());
    }
    if (input->Rank() > 2) {
      num_channels = input->shape()[-1];
    }
    if (num_channels != 1 && num_channels != 3) {
      RETURN_STATUS_UNEXPECTED("AdjustGamma: channel of input image should be 1 or 3, but got: " +
                               std::to_string(num_channels));
    }
    if (input->type().IsFloat()) {
      for (auto itr = input->begin<float>(); itr != input->end<float>(); itr++) {
        *itr = pow((*itr) * gain, gamma);
        *itr = std::min(std::max((*itr), 0.0f), 1.0f);
      }
      *output = input;
    } else {
      std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
      if (!input_cv->mat().data) {
        RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustGamma: load image failed.");
      }
      cv::Mat input_img = input_cv->mat();
      std::shared_ptr<CVTensor> output_cv;
      RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
      uchar LUT[256] = {};
      for (int i = 0; i < 256; i++) {
        float f = i / 255.0;
        f = pow(f, gamma);
        LUT[i] = static_cast<uchar>(floor(std::min(f * (255.0 + 1 - 1e-3) * gain, 255.0)));
      }
      if (input_img.channels() == 1) {
        cv::MatIterator_<uchar> it = input_img.begin<uchar>();
        cv::MatIterator_<uchar> it_end = input_img.end<uchar>();
        for (; it != it_end; ++it) {
          *it = LUT[(*it)];
        }
      } else {
        cv::MatIterator_<cv::Vec3b> it = input_img.begin<cv::Vec3b>();
        cv::MatIterator_<cv::Vec3b> it_end = input_img.end<cv::Vec3b>();
        for (; it != it_end; ++it) {
          (*it)[0] = LUT[(*it)[0]];
          (*it)[1] = LUT[(*it)[1]];
          (*it)[2] = LUT[(*it)[2]];
        }
      }
      output_cv->mat() = input_img * 1;
      *output = std::static_pointer_cast<Tensor>(output_cv);
    }
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustGamma: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AutoContrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &cutoff,
                    const std::vector<uint32_t> &ignore) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AutoContrast: load image failed.");
    }
    if (input_cv->Rank() != DEFAULT_IMAGE_RANK && input_cv->Rank() != MIN_IMAGE_DIMENSION) {
      std::string err_msg = "AutoContrast: image rank should be 2 or 3,  but got: " + std::to_string(input_cv->Rank());
      if (input_cv->Rank() == 1) {
        err_msg = err_msg + ", may need to do Decode operation first.";
      }
      RETURN_STATUS_UNEXPECTED("AutoContrast: image rank should be 2 or 3,  but got: " +
                               std::to_string(input_cv->Rank()));
    }
    // Reshape to extend dimension if rank is 2 for algorithm to work. then reshape output to be of rank 2 like input
    if (input_cv->Rank() == MIN_IMAGE_DIMENSION) {
      RETURN_IF_NOT_OK(input_cv->ExpandDim(MIN_IMAGE_DIMENSION));
    }
    // Get number of channels and image matrix
    std::size_t num_of_channels = input_cv->shape()[CHANNEL_INDEX];
    if (num_of_channels != MIN_IMAGE_CHANNELS && num_of_channels != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("AutoContrast: channel of input image should be 1 or 3, but got: " +
                               std::to_string(num_of_channels));
    }
    cv::Mat image = input_cv->mat();
    // Separate the image to channels
    std::vector<cv::Mat> planes(num_of_channels);
    cv::split(image, planes);
    cv::Mat b_hist, g_hist, r_hist;
    // Establish the number of bins and set variables for histogram
    int32_t hist_size = 256;
    int32_t channels = 0;
    float range[] = {0, 256};
    const float *hist_range[] = {range};
    bool uniform = true, accumulate = false;
    // Set up lookup table for LUT(Look up table algorithm)
    std::vector<int32_t> table;
    std::vector<cv::Mat> image_result;
    for (std::size_t layer = 0; layer < planes.size(); layer++) {
      // Reset lookup table
      table = std::vector<int32_t>{};
      // Calculate Histogram for channel
      cv::Mat hist;
      cv::calcHist(&planes[layer], 1, &channels, cv::Mat(), hist, 1, &hist_size, hist_range, uniform, accumulate);
      hist.convertTo(hist, CV_32SC1);
      std::vector<int32_t> hist_vec;
      hist.col(0).copyTo(hist_vec);
      // Ignore values in ignore
      for (const auto &item : ignore) hist_vec[item] = 0;
      int32_t hi = 255;
      int32_t lo = 0;
      RETURN_IF_NOT_OK(ComputeUpperAndLowerPercentiles(&hist_vec, cutoff, cutoff, &hi, &lo));
      if (hi <= lo) {
        for (int32_t i = 0; i < 256; i++) {
          table.push_back(i);
        }
      } else {
        const float scale = 255.0 / (hi - lo);
        const float offset = -1 * lo * scale;
        for (int32_t i = 0; i < 256; i++) {
          int32_t ix = static_cast<int32_t>(i * scale + offset);
          ix = std::max(ix, 0);
          ix = std::min(ix, MAX_BIT_VALUE);
          table.push_back(ix);
        }
      }
      cv::Mat result_layer;
      cv::LUT(planes[layer], table, result_layer);
      image_result.push_back(result_layer);
    }
    cv::Mat result;
    cv::merge(image_result, result);
    result.convertTo(result, input_cv->mat().type());
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(result, input_cv->Rank(), &output_cv));
    (*output) = std::static_pointer_cast<Tensor>(output_cv);
    RETURN_IF_NOT_OK((*output)->Reshape(input_cv->shape()));
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AutoContrast: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AdjustSaturation(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustSaturation: load image failed.");
    }
    CHECK_FAIL_RETURN_UNEXPECTED(
      input_cv->shape().Size() > CHANNEL_INDEX,
      "AdjustSaturation: image rank should not bigger than: " + std::to_string(CHANNEL_INDEX) +
        ", but got: " + std::to_string(input_cv->shape().Size()));
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->Rank() != DEFAULT_IMAGE_RANK || num_channels != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("AdjustSaturation: image shape is not <H,W,C> or channel is not 3, but got rank: " +
                               std::to_string(input_cv->Rank()) + ", and channel: " + std::to_string(num_channels));
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    cv::Mat output_img = output_cv->mat();
    cv::Mat gray;
    cv::cvtColor(input_img, gray, CV_RGB2GRAY);
    cv::cvtColor(gray, output_img, CV_GRAY2RGB);
    output_cv->mat() = output_img * (1.0 - alpha) + input_img * alpha;
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustSaturation: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AdjustHue(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &hue) {
  if (hue > 0.5 || hue < -0.5) {
    RETURN_STATUS_UNEXPECTED("AdjustHue: invalid parameter, hue should within [-0.5, 0.5], but got: " +
                             std::to_string(hue));
  }
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustHue: load image failed.");
    }
    CHECK_FAIL_RETURN_UNEXPECTED(input_cv->shape().Size() > 2,
                                 "AdjustHue: image rank should not bigger than:" + std::to_string(2) +
                                   ", but got: " + std::to_string(input_cv->shape().Size()));
    int num_channels = input_cv->shape()[2];
    if (input_cv->Rank() != DEFAULT_IMAGE_RANK || num_channels != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("AdjustHue: image shape is not <H,W,C> or channel is not 3, but got rank: " +
                               std::to_string(input_cv->Rank()) + ", and channel: " + std::to_string(num_channels));
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    cv::Mat output_img;
    cv::cvtColor(input_img, output_img, CV_RGB2HSV_FULL);
    for (int y = 0; y < output_img.cols; y++) {
      for (int x = 0; x < output_img.rows; x++) {
        uint8_t cur1 = output_img.at<cv::Vec3b>(cv::Point(y, x))[0];
        uint8_t h_hue = 0;
        h_hue = static_cast<uint8_t>(hue * MAX_BIT_VALUE);
        cur1 += h_hue;
        output_img.at<cv::Vec3b>(cv::Point(y, x))[0] = cur1;
      }
    }
    cv::cvtColor(output_img, output_cv->mat(), CV_HSV2RGB_FULL);
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustHue: " + std::string(e.what()));
  }
  return Status::OK();
}

Status Equalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Equalize: load image failed.");
    }
    if (input_cv->Rank() != DEFAULT_IMAGE_RANK && input_cv->Rank() != MIN_IMAGE_DIMENSION) {
      RETURN_STATUS_UNEXPECTED("Equalize: image rank should be 2 or 3,  but got: " + std::to_string(input_cv->Rank()));
    }
    // For greyscale images, extend dimension if rank is 2 and reshape output to be of rank 2.
    if (input_cv->Rank() == MIN_IMAGE_DIMENSION) {
      RETURN_IF_NOT_OK(input_cv->ExpandDim(MIN_IMAGE_DIMENSION));
    }
    // Get number of channels and image matrix
    std::size_t num_of_channels = input_cv->shape()[CHANNEL_INDEX];
    if (num_of_channels != MIN_IMAGE_CHANNELS && num_of_channels != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("Equalize: channel of input image should be 1 or 3, but got: " +
                               std::to_string(num_of_channels));
    }
    cv::Mat image = input_cv->mat();
    // Separate the image to channels
    std::vector<cv::Mat> planes(num_of_channels);
    cv::split(image, planes);
    // Equalize each channel separately
    std::vector<cv::Mat> image_result;
    for (std::size_t layer = 0; layer < planes.size(); layer++) {
      cv::Mat channel_result;
      cv::equalizeHist(planes[layer], channel_result);
      image_result.push_back(channel_result);
    }
    cv::Mat result;
    cv::merge(image_result, result);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(result, input_cv->Rank(), &output_cv));
    (*output) = std::static_pointer_cast<Tensor>(output_cv);
    RETURN_IF_NOT_OK((*output)->Reshape(input_cv->shape()));
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Equalize: " + std::string(e.what()));
  }
  return Status::OK();
}

Status Erase(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t box_height,
             int32_t box_width, int32_t num_patches, bool bounded, bool random_color, std::mt19937 *rnd, uint8_t fill_r,
             uint8_t fill_g, uint8_t fill_b) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    CHECK_FAIL_RETURN_UNEXPECTED(input_cv->shape().Size() > CHANNEL_INDEX, "Erase: shape is invalid.");
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->mat().data == nullptr) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] CutOut: load image failed.");
    }
    if (input_cv->Rank() != DEFAULT_IMAGE_RANK || num_channels != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("CutOut: image shape is not <H,W,C> or channel is not 3, but got rank: " +
                               std::to_string(input_cv->Rank()) + ", and channel: " + std::to_string(num_channels));
    }
    cv::Mat input_img = input_cv->mat();
    int32_t image_h = input_cv->shape()[0];
    int32_t image_w = input_cv->shape()[1];
    // check if erase size is bigger than image itself
    if (box_height > image_h || box_width > image_w) {
      RETURN_STATUS_UNEXPECTED(
        "CutOut: box size is too large for image erase, got box height: " + std::to_string(box_height) +
        "box weight: " + std::to_string(box_width) + ", and image height: " + std::to_string(image_h) +
        ", image width: " + std::to_string(image_w));
    }

    // for random color
    std::normal_distribution<double> normal_distribution(0, 1);
    std::uniform_int_distribution<int> height_distribution_bound(0, image_h - box_height);
    std::uniform_int_distribution<int> width_distribution_bound(0, image_w - box_width);
    std::uniform_int_distribution<int> height_distribution_unbound(0, image_h + box_height);
    std::uniform_int_distribution<int> width_distribution_unbound(0, image_w + box_width);
    // core logic
    // update values based on random erasing or cutout

    for (int32_t i = 0; i < num_patches; i++) {
      // rows in cv mat refers to the height of the cropped box
      // we determine h_start and w_start using two different distributions as erasing is used by two different
      // image augmentations. The bounds are also different in each case.
      int32_t h_start = (bounded) ? height_distribution_bound(*rnd) : (height_distribution_unbound(*rnd) - box_height);
      int32_t w_start = (bounded) ? width_distribution_bound(*rnd) : (width_distribution_unbound(*rnd) - box_width);

      int32_t max_width = (w_start + box_width > image_w) ? image_w : w_start + box_width;
      int32_t max_height = (h_start + box_height > image_h) ? image_h : h_start + box_height;
      // check for starting range >= 0, here the start range is checked after for cut out, for random erasing
      // w_start and h_start will never be less than 0.
      h_start = (h_start < 0) ? 0 : h_start;
      w_start = (w_start < 0) ? 0 : w_start;
      for (int y = w_start; y < max_width; y++) {
        for (int x = h_start; x < max_height; x++) {
          if (random_color) {
            // fill each box with a random value
            input_img.at<cv::Vec3b>(cv::Point(y, x))[0] = static_cast<int32_t>(normal_distribution(*rnd));
            input_img.at<cv::Vec3b>(cv::Point(y, x))[1] = static_cast<int32_t>(normal_distribution(*rnd));
            input_img.at<cv::Vec3b>(cv::Point(y, x))[2] = static_cast<int32_t>(normal_distribution(*rnd));
          } else {
            input_img.at<cv::Vec3b>(cv::Point(y, x))[0] = fill_r;
            input_img.at<cv::Vec3b>(cv::Point(y, x))[1] = fill_g;
            input_img.at<cv::Vec3b>(cv::Point(y, x))[2] = fill_b;
          }
        }
      }
    }
    *output = std::static_pointer_cast<Tensor>(input);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("CutOut: " + std::string(e.what()));
  }
}

Status Pad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int32_t &pad_top,
           const int32_t &pad_bottom, const int32_t &pad_left, const int32_t &pad_right, const BorderType &border_types,
           uint8_t fill_r, uint8_t fill_g, uint8_t fill_b) {
  try {
    // input image
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);

    // validate rank
    if (input_cv->Rank() == 1 || input_cv->mat().dims > MIN_IMAGE_DIMENSION) {
      std::string err_msg = "Pad: input shape is not <H,W,C> or <H, W>, got rank: " + std::to_string(input_cv->Rank());
      if (input_cv->Rank() == 1) {
        err_msg = err_msg + ", may need to do Decode operation first.";
      }
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    // get the border type in openCV
    auto b_type = GetCVBorderType(border_types);
    // output image
    cv::Mat out_image;
    if (b_type == cv::BORDER_CONSTANT) {
      cv::Scalar fill_color = cv::Scalar(fill_b, fill_g, fill_r);
      cv::copyMakeBorder(input_cv->mat(), out_image, pad_top, pad_bottom, pad_left, pad_right, b_type, fill_color);
    } else {
      cv::copyMakeBorder(input_cv->mat(), out_image, pad_top, pad_bottom, pad_left, pad_right, b_type);
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(out_image, input_cv->Rank(), &output_cv));
    // pad the dimension if shape information is only 2 dimensional, this is grayscale
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->Rank() == DEFAULT_IMAGE_RANK && num_channels == MIN_IMAGE_CHANNELS &&
        output_cv->Rank() == MIN_IMAGE_DIMENSION)
      RETURN_IF_NOT_OK(output_cv->ExpandDim(CHANNEL_INDEX));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Pad: " + std::string(e.what()));
  }
}

Status RandomLighting(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rnd_r, float rnd_g,
                      float rnd_b) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();

    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] RandomLighting: load image failed.");
    }

    if (input_cv->Rank() != DEFAULT_IMAGE_RANK || input_cv->shape()[CHANNEL_INDEX] != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED(
        "RandomLighting: input tensor is not in shape of <H,W,C> or channel is not 3, got rank: " +
        std::to_string(input_cv->Rank()) + ", and channel: " + std::to_string(input_cv->shape()[CHANNEL_INDEX]));
    }
    auto input_type = input->type();
    CHECK_FAIL_RETURN_UNEXPECTED(input_type != DataType::DE_UINT32 && input_type != DataType::DE_UINT64 &&
                                   input_type != DataType::DE_INT64 && input_type != DataType::DE_STRING,
                                 "RandomLighting: invalid tensor type of uint32, int64, uint64 or string.");

    std::vector<std::vector<float>> eig = {{55.46 * -0.5675, 4.794 * 0.7192, 1.148 * 0.4009},
                                           {55.46 * -0.5808, 4.794 * -0.0045, 1.148 * -0.8140},
                                           {55.46 * -0.5836, 4.794 * -0.6948, 1.148 * 0.4203}};

    float pca_r = eig[0][0] * rnd_r + eig[0][1] * rnd_g + eig[0][2] * rnd_b;
    float pca_g = eig[1][0] * rnd_r + eig[1][1] * rnd_g + eig[1][2] * rnd_b;
    float pca_b = eig[2][0] * rnd_r + eig[2][1] * rnd_g + eig[2][2] * rnd_b;
    for (int row = 0; row < input_img.rows; row++) {
      for (int col = 0; col < input_img.cols; col++) {
        float r = static_cast<float>(input_img.at<cv::Vec3b>(row, col)[0]);
        float g = static_cast<float>(input_img.at<cv::Vec3b>(row, col)[1]);
        float b = static_cast<float>(input_img.at<cv::Vec3b>(row, col)[2]);
        input_img.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(r + pca_r);
        input_img.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g + pca_g);
        input_img.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(b + pca_b);
      }
    }

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(input_img, input_cv->Rank(), &output_cv));

    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RandomLighting: " + std::string(e.what()));
  }
}

Status RgbaToRgb(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->shape().Size() != DEFAULT_IMAGE_CHANNELS || num_channels != 4) {
      std::string err_msg = "RgbaToRgb: rank of image is not: " + std::to_string(DEFAULT_IMAGE_CHANNELS) +
                            ", but got: " + std::to_string(input_cv->shape().Size()) +
                            ", or channels of image should be 4, but got: " + std::to_string(num_channels);
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    TensorShape out_shape = TensorShape({input_cv->shape()[0], input_cv->shape()[1], 3});
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(out_shape, input_cv->type(), &output_cv));
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_RGBA2RGB));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbaToRgb: " + std::string(e.what()));
  }
}

Status RgbaToBgr(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    int num_channels = input_cv->shape()[CHANNEL_INDEX];
    if (input_cv->shape().Size() != DEFAULT_IMAGE_CHANNELS || num_channels != MAX_IMAGE_CHANNELS) {
      std::string err_msg = "RgbaToBgr: rank of image is not: " + std::to_string(DEFAULT_IMAGE_CHANNELS) +
                            ", but got: " + std::to_string(input_cv->shape().Size()) +
                            ", or channels of image should be 4, but got: " + std::to_string(num_channels);
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    TensorShape out_shape = TensorShape({input_cv->shape()[0], input_cv->shape()[1], 3});
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(out_shape, input_cv->type(), &output_cv));
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_RGBA2BGR));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbaToBgr: " + std::string(e.what()));
  }
}

Status RgbToBgr(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    auto input_type = input->type();
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] RgbToBgr: load image failed.");
    }
    if (input_cv->Rank() != DEFAULT_IMAGE_RANK || input_cv->shape()[2] != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED("RgbToBgr: input tensor is not in shape of <H,W,C> or channel is not 3, got rank: " +
                               std::to_string(input_cv->Rank()) +
                               ", and channel: " + std::to_string(input_cv->shape()[2]));
    }

    cv::Mat image = input_cv->mat().clone();
    if (input_type == DataType::DE_FLOAT16 || input_type == DataType::DE_INT16 || input_type == DataType::DE_UINT16) {
      for (int i = 0; i < input_cv->mat().rows; ++i) {
        cv::Vec3s *p1 = input_cv->mat().ptr<cv::Vec3s>(i);
        cv::Vec3s *p2 = image.ptr<cv::Vec3s>(i);
        for (int j = 0; j < input_cv->mat().cols; ++j) {
          p2[j][2] = p1[j][0];
          p2[j][1] = p1[j][1];
          p2[j][0] = p1[j][2];
        }
      }
    } else if (input_type == DataType::DE_FLOAT32 || input_type == DataType::DE_INT32) {
      for (int i = 0; i < input_cv->mat().rows; ++i) {
        cv::Vec3f *p1 = input_cv->mat().ptr<cv::Vec3f>(i);
        cv::Vec3f *p2 = image.ptr<cv::Vec3f>(i);
        for (int j = 0; j < input_cv->mat().cols; ++j) {
          p2[j][2] = p1[j][0];
          p2[j][1] = p1[j][1];
          p2[j][0] = p1[j][2];
        }
      }
    } else if (input_type == DataType::DE_FLOAT64) {
      for (int i = 0; i < input_cv->mat().rows; ++i) {
        cv::Vec3d *p1 = input_cv->mat().ptr<cv::Vec3d>(i);
        cv::Vec3d *p2 = image.ptr<cv::Vec3d>(i);
        for (int j = 0; j < input_cv->mat().cols; ++j) {
          p2[j][2] = p1[j][0];
          p2[j][1] = p1[j][1];
          p2[j][0] = p1[j][2];
        }
      }
    } else {
      cv::cvtColor(input_cv->mat(), image, cv::COLOR_RGB2BGR);
    }

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(image, input_cv->Rank(), &output_cv));

    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbToBgr: " + std::string(e.what()));
  }
}

Status RgbToGray(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    if (input_cv->Rank() != DEFAULT_IMAGE_RANK || input_cv->shape()[CHANNEL_INDEX] != DEFAULT_IMAGE_CHANNELS) {
      RETURN_STATUS_UNEXPECTED(
        "RgbToGray: image shape is not <H,W,C> or channel is not 3, got rank: " + std::to_string(input_cv->Rank()) +
        ", and channel: " + std::to_string(input_cv->shape()[2]));
    }
    TensorShape out_shape = TensorShape({input_cv->shape()[0], input_cv->shape()[1]});
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(out_shape, input_cv->type(), &output_cv));
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_RGB2GRAY));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbToGray: " + std::string(e.what()));
  }
}

Status GetJpegImageInfo(const std::shared_ptr<Tensor> &input, int *img_width, int *img_height) {
  struct jpeg_decompress_struct cinfo {};
  struct JpegErrorManagerCustom jerr {};
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = JpegErrorExitCustom;
  try {
    jpeg_create_decompress(&cinfo);
    JpegSetSource(&cinfo, input->GetBuffer(), input->SizeInBytes());
    (void)jpeg_read_header(&cinfo, TRUE);
    jpeg_calc_output_dimensions(&cinfo);
  } catch (std::runtime_error &e) {
    jpeg_destroy_decompress(&cinfo);
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  *img_height = cinfo.output_height;
  *img_width = cinfo.output_width;
  jpeg_destroy_decompress(&cinfo);
  return Status::OK();
}

Status Affine(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::vector<float_t> &mat,
              InterpolationMode interpolation, uint8_t fill_r, uint8_t fill_g, uint8_t fill_b) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    RETURN_IF_NOT_OK(ValidateImageRank("Affine", input_cv->Rank()));

    cv::Mat affine_mat(mat);
    affine_mat = affine_mat.reshape(1, {2, 3});

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    cv::warpAffine(input_cv->mat(), output_cv->mat(), affine_mat, input_cv->mat().size(),
                   GetCVInterpolationMode(interpolation), cv::BORDER_CONSTANT, cv::Scalar(fill_r, fill_g, fill_b));
    (*output) = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Affine: " + std::string(e.what()));
  }
}

Status GaussianBlur(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t kernel_x,
                    int32_t kernel_y, float sigma_x, float sigma_y) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (input_cv->mat().data == nullptr) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] GaussianBlur: load image failed.");
    }
    cv::Mat output_cv_mat;
    cv::GaussianBlur(input_cv->mat(), output_cv_mat, cv::Size(kernel_x, kernel_y), static_cast<double>(sigma_x),
                     static_cast<double>(sigma_y));
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_cv_mat, input_cv->Rank(), &output_cv));
    (*output) = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("GaussianBlur: " + std::string(e.what()));
  }
}

Status ComputePatchSize(const std::shared_ptr<CVTensor> &input_cv,
                        std::shared_ptr<std::pair<int32_t, int32_t>> *patch_size, int32_t num_height, int32_t num_width,
                        SliceMode slice_mode) {
  if (input_cv->mat().data == nullptr) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] SlicePatches: Tensor could not convert to CV Tensor.");
  }
  RETURN_IF_NOT_OK(ValidateImageRank("Affine", input_cv->Rank()));

  cv::Mat in_img = input_cv->mat();
  cv::Size s = in_img.size();
  if (num_height == 0 || num_height > s.height) {
    RETURN_STATUS_UNEXPECTED(
      "SlicePatches: The number of patches on height axis equals 0 or is greater than height, got number of patches:" +
      std::to_string(num_height));
  }
  if (num_width == 0 || num_width > s.width) {
    RETURN_STATUS_UNEXPECTED(
      "SlicePatches: The number of patches on width axis equals 0 or is greater than width, got number of patches:" +
      std::to_string(num_width));
  }
  int32_t patch_h = s.height / num_height;
  if (s.height % num_height != 0) {
    if (slice_mode == SliceMode::kPad) {
      patch_h += 1;  // patch_h * num_height - s.height
    }
  }
  int32_t patch_w = s.width / num_width;
  if (s.width % num_width != 0) {
    if (slice_mode == SliceMode::kPad) {
      patch_w += 1;  // patch_w * num_width - s.width
    }
  }
  (*patch_size)->first = patch_h;
  (*patch_size)->second = patch_w;
  return Status::OK();
}

Status SlicePatches(const std::shared_ptr<Tensor> &input, std::vector<std::shared_ptr<Tensor>> *output,
                    int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value) {
  if (num_height == DEFAULT_NUM_HEIGHT && num_width == DEFAULT_NUM_WIDTH) {
    (*output).push_back(input);
    return Status::OK();
  }

  auto patch_size = std::make_shared<std::pair<int32_t, int32_t>>(0, 0);
  int32_t patch_h = 0;
  int32_t patch_w = 0;

  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  RETURN_IF_NOT_OK(ComputePatchSize(input_cv, &patch_size, num_height, num_width, slice_mode));
  std::tie(patch_h, patch_w) = *patch_size;

  cv::Mat in_img = input_cv->mat();
  cv::Size s = in_img.size();
  try {
    cv::Mat out_img;
    if (slice_mode == SliceMode::kPad) {  // padding on right and bottom directions
      auto padding_h = patch_h * num_height - s.height;
      auto padding_w = patch_w * num_width - s.width;
      out_img = cv::Mat(s.height + padding_h, s.width + padding_w, in_img.type(), cv::Scalar::all(fill_value));
      in_img.copyTo(out_img(cv::Rect(0, 0, s.width, s.height)));
    } else {
      out_img = in_img;
    }
    for (int i = 0; i < num_height; ++i) {
      for (int j = 0; j < num_width; ++j) {
        std::shared_ptr<CVTensor> patch_cv;
        cv::Rect rect(j * patch_w, i * patch_h, patch_w, patch_h);
        cv::Mat patch(out_img(rect));
        RETURN_IF_NOT_OK(CVTensor::CreateFromMat(patch, input_cv->Rank(), &patch_cv));
        (*output).push_back(std::static_pointer_cast<Tensor>(patch_cv));
      }
    }
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("SlicePatches: " + std::string(e.what()));
  }
}

Status ValidateImageRank(const std::string &op_name, int32_t rank) {
  if (rank != 2 && rank != 3) {
    std::string err_msg = op_name + ": image shape is not <H,W,C> or <H, W>, but got rank:" + std::to_string(rank);
    if (rank == 1) {
      err_msg = err_msg + ", may need to do Decode operation first.";
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

//RS index
//ANDWI
Status ANDWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] ANDWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("ANDWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hBlue =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hRed =
        GDALGetRasterBand(m_outPoDataSet, 3);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 5);
    GDALRasterBandH hMir2 =
        GDALGetRasterBand(m_outPoDataSet, 6);

    float *blue = new float[nImgSizeX * nImgSizeY];
    float *green = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *mir2 = new float[nImgSizeX * nImgSizeY];
    float *andwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hBlue, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)blue, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)red, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r4 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r5 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r6 = GDALRasterIO(hMir2, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir2, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure || r4 == CE_Failure || r5 == CE_Failure || r6 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          andwi[i] = (blue[i] + green[i] + red[i] - nir[i] - mir1[i] - mir2[i]) / (blue[i] + green[i] + red[i] + nir[i] + mir1[i] + mir2[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, andwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("ANDWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//AWEI
Status AWEI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AWEI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("AWEI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    //  循环写入文件
    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    float *awei = new float[nImgSizeX * nImgSizeY];

    if (nBandCount == 4)
    {
        GDALRasterBandH hGreen =
            GDALGetRasterBand(m_outPoDataSet, 1);
        GDALRasterBandH hNir =
            GDALGetRasterBand(m_outPoDataSet, 2);
        GDALRasterBandH hMir1 =
            GDALGetRasterBand(m_outPoDataSet, 3);
        GDALRasterBandH hMir2 = 
            GDALGetRasterBand(m_outPoDataSet, 4);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *mir2 = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r4 = GDALRasterIO(hMir2, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir2, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure || r4 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          awei[i] = 4 * (green[i] - mir1[i]) - (0.25 * nir[i] + 2.75 * mir2[i]);
    }
    }
    else if (nBandCount == 5)
    {
        GDALRasterBandH hBlue =
            GDALGetRasterBand(m_outPoDataSet, 1);
        GDALRasterBandH hGreen =
            GDALGetRasterBand(m_outPoDataSet, 2);
        GDALRasterBandH hNir =
            GDALGetRasterBand(m_outPoDataSet, 3);
        GDALRasterBandH hMir1 =
            GDALGetRasterBand(m_outPoDataSet, 4);
        GDALRasterBandH hMir2 = 
            GDALGetRasterBand(m_outPoDataSet, 5);

    float *blue = new float[nImgSizeX * nImgSizeY];
    float *green = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *mir2 = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hBlue, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)blue, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r4 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r5 = GDALRasterIO(hMir2, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir2, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure || r4 == CE_Failure || r5 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          awei[i] = blue[i] + 2.5 * green[i] - 1.5 * (nir[i] + mir1[i]) - 0.25 * mir2[i];
    }
    }
    else
        return Status::OK();

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, awei);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AWEI: " + std::string(e.what()));
  }
  return Status::OK();
}

//BMI_SAR
Status BMI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] BMI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("BMI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister(); 
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH HH =
        GDALGetRasterBand(m_outPoDataSet, 1); 
    GDALRasterBandH VV =
        GDALGetRasterBand(m_outPoDataSet, 2); 

    float *bufferhh = new float[nImgSizeX * nImgSizeY];
    float *buffervv = new float[nImgSizeX * nImgSizeY];
    float *bmi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(HH, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhh, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(VV, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)buffervv, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      bmi[i] = (bufferhh[i] + buffervv[i]) / 2;
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, bmi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("BMI: " + std::string(e.what()));
  }
  return Status::OK();
}

//CIWI
Status CIWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &digital_C) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] CIWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("CIWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet; 
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4); 
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3); 

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *ciwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(nir[i] + red[i]) < 0.1)
            ciwi[i] = -1;
        else
            ciwi[i] = (nir[i] - red[i]) / (nir[i] + red[i]) + nir[i] + digital_C;
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, ciwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("CIWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//CSI_SAR
Status CSI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] CSI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("CSI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH HH =
        GDALGetRasterBand(m_outPoDataSet, 1); 
    GDALRasterBandH VV =
        GDALGetRasterBand(m_outPoDataSet, 2); 

    float *bufferhh = new float[nImgSizeX * nImgSizeY];
    float *buffervv = new float[nImgSizeX * nImgSizeY];
    float *csi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(HH, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhh, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(VV, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)buffervv, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      csi[i] = buffervv[i] / (bufferhh[i] + buffervv[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, csi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("CSI: " + std::string(e.what()));
  }
  return Status::OK();
}

//EWI_W
Status EWI_W(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &m, const float &n) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] EWI_W: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("EWI_W", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hRed =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 3);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 4);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *ewi_w = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)red, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r4 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure || r4 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          ewi_w[i] = (green[i] - mir1[i] + m) / ((green[i] + mir1[i])*((nir[i] - red[i]) / (nir[i] + red[i]) + n));
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, ewi_w);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("EWI_W: " + std::string(e.what()));
  }
  return Status::OK();
}

//EWI_Y
Status EWI_Y(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] EWI_Y: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("EWI_Y", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister(); 
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 3);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *ewi_y = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          ewi_y[i] = (green[i] - nir[i] - mir1[i]) / (green[i] + nir[i] + mir1[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, ewi_y);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("EWI_Y: " + std::string(e.what()));
  }
  return Status::OK();
}

//FNDWI
Status FNDWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int &S, const int &CNIR) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] FNDWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("FNDWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 4);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *fg = new float[nImgSizeX * nImgSizeY];
    float *fndwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          fg[i] = green[i] + S * (CNIR - nir[i]);
          fndwi[i] = (fg[i] - nir[i]) / (fg[i] + nir[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, fndwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("FNDWI: " + std::string(e.what()));
  }
  return Status::OK();
}

int rotateImage(const cv::Mat &src, cv::Mat &dst, const double angle, const int mode)
{
    //mode = 0 ,Keep the original image size unchanged
    //mode = 1, Change the original image size to fit the rotated scale, padding with zero

    if (src.empty())
    {
        std::cout << "The input image is empty!\n";
        return -1;
    }

    if (mode == 0)
    {
        cv::Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(src, dst, rot, src.size());//the original size
    }
    else {

        double alpha = -angle * CV_PI / 180.0;//convert angle to radian format 

        cv::Point2f srcP[3];
        cv::Point2f dstP[3];
        srcP[0] = cv::Point2f(0, src.rows);
        srcP[1] = cv::Point2f(src.cols, 0);
        srcP[2] = cv::Point2f(src.cols, src.rows);

        //rotate the pixels
        for (int i = 0; i < 3; i++)
            dstP[i] = cv::Point2f(srcP[i].x*cos(alpha) - srcP[i].y*sin(alpha), srcP[i].y*cos(alpha) + srcP[i].x*sin(alpha));
        double minx, miny, maxx, maxy;
        minx = std::min(std::min(std::min(dstP[0].x, dstP[1].x), dstP[2].x), float(0.0));
        miny = std::min(std::min(std::min(dstP[0].y, dstP[1].y), dstP[2].y), float(0.0));
        maxx = std::max(std::max(std::max(dstP[0].x, dstP[1].x), dstP[2].x), float(0.0));
        maxy = std::max(std::max(std::max(dstP[0].y, dstP[1].y), dstP[2].y), float(0.0));

        int w = maxx - minx;
        int h = maxy - miny;

        //translation
        for (int i = 0; i < 3; i++)
        {
            if (minx < 0)
                dstP[i].x -= minx;
            if (miny < 0)
                dstP[i].y -= miny;
        }

        cv::Mat warpMat = cv::getAffineTransform(srcP, dstP);
        cv::warpAffine(src, dst, warpMat, cv::Size(w, h));//extend size

    }//end else

    return 0;
}

//Gabor
Status Gabor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, bool if_opencv_kernal) {
  try {
    auto input_type = input->type();
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Gabor: load image failed.");
    }
    if (input_cv->Rank() != 3 || input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("Gabor: input tensor is not in shape of <H,W,C> or channel is not 3.");
    }

    cv::Mat input_img = input_cv->mat();
    cv::Mat src_gray, output_img, gabor_tmp;
    cv::cvtColor(input_img, src_gray, cv::COLOR_BGR2GRAY);
    int k = 9;
    float sigma = 1.0;
    float gamma = 0.5;
    float lambda = 5.0;
    float psi = -CV_PI / 2;
    gabor_filter(if_opencv_kernal, src_gray, output_img, gabor_tmp, k, sigma, gamma, lambda, psi);

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Gabor: " + std::string(e.what()));
  }
}

//GLCM
Status GLCM(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int &N) {
  try {
    auto input_type = input->type();
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("GLCM: load image failed.");
    }
    if (input_cv->Rank() != 3 || input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("GLCM: input tensor is not in shape of <H,W,C> or channel is not 3.");
    }

    cv::Mat input_img = input_cv->mat();

    CALGLCM glcm;
    cv::Mat imgEnergy, imgContrast, imgHomogenity, imgEntropy;
    cv::Mat dstChannel;
    glcm.getOneChannel(input_img, dstChannel, CHANNEL_B);
    glcm.GrayMagnitude(dstChannel, dstChannel, GRAY_8);
    glcm.CalcuTextureImages(dstChannel, imgEnergy, imgContrast, imgHomogenity, imgEntropy, 5, GRAY_8, true);

    cv::Mat output_img;
    switch (N){
    case 0:
        output_img = imgEnergy;
        break;
    case 1:
        output_img = imgContrast;
        break;
    case 2:
        output_img = imgHomogenity;
        break;
    case 3:
        output_img = imgEntropy;
        break;
    default:
        return Status::OK();
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("GLCM: " + std::string(e.what()));
  }
}

//GNDWI
Status GNDWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] GNDWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("GNDWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 4);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *ndwi = new float[nImgSizeX * nImgSizeY];
    float *gndwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    float sum = 0.0, mean, variance = 0.0, delta;
    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
        ndwi[i] = (green[i] - nir[i]) / (green[i] + nir[i]);
        sum += ndwi[i];
    }
    mean = sum / (nImgSizeX * nImgSizeY);
    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
        ndwi[i] = (green[i] - nir[i]) / (green[i] + nir[i]);
        variance += pow(ndwi[i] - mean, 2);
    }
    variance = variance / (nImgSizeX * nImgSizeY);
    delta = sqrt(variance);
    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
        ndwi[i] = (green[i] - nir[i]) / (green[i] + nir[i]);
        gndwi[i] = (ndwi[i] - mean) / delta;
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, gndwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("GNDWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//LBP
Status LBP(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int &N) {
  try {
    auto input_type = input->type();
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("LBP: load image failed.");
    }
    if (input_cv->Rank() != 3 || input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("LBP: input tensor is not in shape of <H,W,C> or channel is not 3.");
    }

    cv::Mat color_img = input_cv->mat();
    cv::Mat input_img;
    cv::cvtColor(color_img, input_img, cv::COLOR_BGR2GRAY);
    cv::Mat output_img;
    
    switch (N){
    case 0:
        output_img = OLBP(input_img);
        break;
    case 1:
        output_img = ELBP(input_img, 1 , 8);
        break;
    case 2:
        output_img = RILBP(input_img);
        break;
    case 3:
        output_img = UniformLBP(input_img);
        break;
    default:
        return Status::OK();
    }

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("LBP: " + std::string(e.what()));
  }
}

//MBI
Status MBI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t s_min,
           int32_t s_max,  int32_t delta_s) {
  try {
    auto input_type = input->type();
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("MBI: load image failed.");
    }
    if (input_cv->Rank() != 3 || input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("MBI: input tensor is not in shape of <H,W,C> or channel is not 3.");
    }

    cv::Mat input_img = input_cv->mat();

    cv::Mat gray = RGB2GRAY(input_img);

    cv::Mat dst_gray;
    cv::copyMakeBorder(gray, dst_gray, s_min, s_min, s_min, s_min, cv::BORDER_REFLECT_101);

    std::vector<cv::Mat> MP_MBI_list;
    std::vector<cv::Mat> DMP_MBI_list;

    for (int i = s_min; i < s_max + 1; i = i + 2 * delta_s)
    {
        cv::Mat in_0 = cv::Mat::zeros(i, i, CV_8U);
        cv::Mat in_1 = cv::Mat::ones(i, i, CV_8U);

        for (int row = (i - 1) / 2, col = 0; col < in_0.cols; ++col) {
            in_0.at<char>(row, col) = 1;
        }

        cv::Mat out_1, out_2, out_3, out_4;
        rotateImage(in_0, out_1, 0, 0);
        rotateImage(in_0, out_2, 45, 0);
        rotateImage(in_0, out_3, 90, 0);
        rotateImage(in_0, out_4, 135, 0);

        cv::Mat MP_MBI_1, MP_MBI_2, MP_MBI_3, MP_MBI_4;

        morphologyEx(dst_gray, MP_MBI_1, cv::MORPH_TOPHAT, out_1);
        morphologyEx(dst_gray, MP_MBI_2, cv::MORPH_TOPHAT, out_2);
        morphologyEx(dst_gray, MP_MBI_3, cv::MORPH_TOPHAT, out_3);
        morphologyEx(dst_gray, MP_MBI_4, cv::MORPH_TOPHAT, out_4);


        MP_MBI_list.push_back(MP_MBI_1);
        MP_MBI_list.push_back(MP_MBI_2);
        MP_MBI_list.push_back(MP_MBI_3);
        MP_MBI_list.push_back(MP_MBI_4);
    }

    for (int j = 4; j < MP_MBI_list.size(); j++)
    {
        auto DMP_MBI = cv::abs(MP_MBI_list[j] - MP_MBI_list[j-4]);
        DMP_MBI_list.push_back(DMP_MBI);
    }

    cv::Mat input_;

    for (int i = 0; i < DMP_MBI_list.size(); ++i) {
        if (i == 0)
            input_ = DMP_MBI_list[i];
        else
            input_ += DMP_MBI_list[i];
    }

    cv::Mat output_MBI = input_ / (4 * (((s_max - s_min) / delta_s) + 1));
    output_MBI(cv::Rect(s_min, s_min, output_MBI.size[0]-s_min, output_MBI.size[1] - s_min));
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_MBI, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("MBI: " + std::string(e.what()));
  }
}

//MCIWI
Status MCIWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] MCIWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("MCIWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hRed =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 3);

    float *red = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *mciwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)red, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
        mciwi[i] = (nir[i] - red[i]) / (nir[i] + red[i]) + (mir1[i] - nir[i]) / (mir1[i] + nir[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, mciwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("MCIWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//MNDWI
Status MNDWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] MNDWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("MNDWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 2);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *mndwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      if (fabs(green[i] + mir1[i]) < 0.1)
          mndwi[i] = -1;
      else
          mndwi[i] = (green[i] - mir1[i]) / (green[i] + mir1[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, mndwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("MNDWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//NDPI
Status NDPI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] NDPI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("NDPI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 2);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *ndpi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      if (fabs(green[i] + mir1[i]) < 0.1)
          ndpi[i] = -1;
      else
          ndpi[i] = (mir1[i] - green[i]) / (green[i] + mir1[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, ndpi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("NDPI: " + std::string(e.what()));
  }
  return Status::OK();
}

//NDVI
Status NDVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] NDVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("NDVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);
    
    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *ndvi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(nir[i] + red[i]) < 0.1)
            ndvi[i] = -1;
        else
            ndvi[i] = (nir[i] - red[i]) / (nir[i] + red[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, ndvi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("NDVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//NDWI
Status NDWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] NDWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("NDWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hGreen = GDALGetRasterBand(m_outPoDataSet, 2);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *green = new float[nImgSizeX * nImgSizeY];
    float *ndwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)green, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(green[i] + nir[i]) < 0.1)
            ndwi[i] = -1;
        else
            ndwi[i] = (green[i] - nir[i]) / (green[i] + nir[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, ndwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("NDWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//NWI
Status NWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] NWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("NWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hBlue =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 3);
    GDALRasterBandH hMir2 = 
        GDALGetRasterBand(m_outPoDataSet, 4);

    float *blue = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *mir2 = new float[nImgSizeX * nImgSizeY];
    float *nwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hBlue, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)blue, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r4 = GDALRasterIO(hMir2, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir2, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure || r4 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      nwi[i] = (blue[i] -(nir[i] + mir1[i] + mir2[i])) /(blue[i] + (nir[i] + mir1[i] + mir2[i]));
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, nwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("NWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//PSI_SAR
Status PSI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] PSI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("PSI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH HH =
        GDALGetRasterBand(m_outPoDataSet, 1); 
    GDALRasterBandH HV =
        GDALGetRasterBand(m_outPoDataSet, 2); 

    float *bufferhh = new float[nImgSizeX * nImgSizeY];
    float *bufferhv = new float[nImgSizeX * nImgSizeY];
    float *psi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(HH, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhh, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(HV, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhv, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      psi[i] = (pow(bufferhv[i],2) - pow(bufferhh[i], 2)) / (pow(bufferhv[i], 2) + pow(bufferhh[i], 2));
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, psi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("PSI: " + std::string(e.what()));
  }
  return Status::OK();
}

//RFDI_SAR
Status RFDI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] RFDI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("RFDI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH HH =
        GDALGetRasterBand(m_outPoDataSet, 1); 
    GDALRasterBandH HV =
        GDALGetRasterBand(m_outPoDataSet, 2); 

    float *bufferhh = new float[nImgSizeX * nImgSizeY];
    float *bufferhv = new float[nImgSizeX * nImgSizeY];
    float *rfdi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(HH, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhh, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(HV, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhv, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      rfdi[i] = (bufferhh[i] - bufferhv[i]) / (bufferhh[i] + bufferhv[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, rfdi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RFDI: " + std::string(e.what()));
  }
  return Status::OK();
}

//RVI
Status RVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] RVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("RVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *rvi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (red[i] < 0.1)
            rvi[i] = -1;
        else
            rvi[i] = nir[i] / red[i];
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, rvi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//SRWI
Status SRWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] SRWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("SRWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 2);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *srwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          srwi[i] = green[i] / mir1[i];
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, srwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("SRWI: " + std::string(e.what()));
  }
  return Status::OK();
}


//DVI
Status DVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] DVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("DVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *dvi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        dvi[i] = nir[i] - red[i];
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, dvi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("DVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//EVI
Status EVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] EVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("EVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);
    GDALRasterBandH hBlue = GDALGetRasterBand(m_outPoDataSet, 1);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *blue = new float[nImgSizeX * nImgSizeY];
    float *evi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hBlue, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)blue, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

  if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(nir[i] + 6.0 * red[i] - 7.5 * blue[i] + 1) < 0.1)
            evi[i] = -1;
        else
            evi[i] = 2.5 * (nir[i] - red[i]) / (nir[i] + 6.0 * red[i] - 7.5 * blue[i] + 1);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, evi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("EVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//MBWI
Status MBWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] MBWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("MBWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hRed =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 3);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hMir2 =
        GDALGetRasterBand(m_outPoDataSet, 5);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *mir2 = new float[nImgSizeX * nImgSizeY];
    float *mbwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)red, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r4 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r5 = GDALRasterIO(hMir2, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir2, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure || r4 == CE_Failure || r5 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          mbwi[i] = 2 * green[i] - red[i] - nir[i] - mir1[i] - mir2[i];
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, mbwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("MBWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//MSAVI
Status MSAVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] MSAVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("MSAVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *msavi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        msavi[i] = (2 * nir[i] + 1 - sqrt( pow((2 * nir[i] +1),2) - 8 * (nir[i] - red[i]))) / 2;
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, msavi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("MSAVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//OSAVI
Status OSAVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float theta) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] OSAVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("OSAVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *osavi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(nir[i] + red[i] + theta) < 0.1)
            osavi[i] = -1;
        else
            osavi[i] = (nir[i] - red[i]) / (nir[i] + red[i] + theta);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, osavi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("OSAVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//VSI_SAR
Status VSI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] VSI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("VSI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH HH =
        GDALGetRasterBand(m_outPoDataSet, 1); 
    GDALRasterBandH HV =
        GDALGetRasterBand(m_outPoDataSet, 2); 
    GDALRasterBandH VV =
        GDALGetRasterBand(m_outPoDataSet, 3); 

    float *bufferhh = new float[nImgSizeX * nImgSizeY];
    float *bufferhv = new float[nImgSizeX * nImgSizeY];
    float *buffervv = new float[nImgSizeX * nImgSizeY];
    float *vsi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(HH, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhh, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(HV, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhv, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(VV, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)buffervv, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      vsi[i] = bufferhv[i] / (bufferhv[i] + (bufferhh[i] + buffervv[i]) / 2);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, vsi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("VSI: " + std::string(e.what()));
  }
  return Status::OK();
}

//WDRVI
Status WDRVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float alpha) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] WDRVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("WDRVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *wdrvi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(alpha * nir[i] + red[i]) < 0.1)
            wdrvi[i] = -1;
        else
            wdrvi[i] = (alpha * nir[i] - red[i]) / (alpha * nir[i] + red[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, wdrvi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("WDRVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//WI_F
Status WI_F(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] WI_F: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("WI_F", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hRed =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 3);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hMir2 =
        GDALGetRasterBand(m_outPoDataSet, 5);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *mir2 = new float[nImgSizeX * nImgSizeY];
    float *wi_f = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)red, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r4 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r5 = GDALRasterIO(hMir2, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir2, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure || r4 == CE_Failure || r5 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          wi_f[i] = 1.7204 + 171 * green[i] + 3 * red[i] - 70 * nir[i] - 45 * mir1[i] - 71 * mir2[i];
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, wi_f);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("WI_F: " + std::string(e.what()));
  }
  return Status::OK();
}

//WI_H
Status WI_H(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] WI_H: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("WI_H", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hRed =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 3);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *wi_h = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)red, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          wi_h[i] = (1.75 * green[i] - red[i] - 1.08 * mir1[i]) / (green[i] + mir1[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, wi_h);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("WI_H: " + std::string(e.what()));
  }
  return Status::OK();
}

//WNDWI
Status WNDWI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] WNDWI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("WNDWI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hGreen =
        GDALGetRasterBand(m_outPoDataSet, 1);
    GDALRasterBandH hNir =
        GDALGetRasterBand(m_outPoDataSet, 2);
    GDALRasterBandH hMir1 =
        GDALGetRasterBand(m_outPoDataSet, 3);

    float *green = new float[nImgSizeX * nImgSizeY];
    float *nir = new float[nImgSizeX * nImgSizeY];
    float *mir1 = new float[nImgSizeX * nImgSizeY];
    float *wndwi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hGreen, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)green, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)nir, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(hMir1, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)mir1, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
          wndwi[i] = (green[i] - alpha * nir[i] - (1-alpha)* mir1[i]) / (green[i] + alpha * nir[i] + (1 - alpha)* mir1[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, wndwi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("WNDWI: " + std::string(e.what()));
  }
  return Status::OK();
}

//RDVI
Status RDVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] RDVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("RDVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *rdvi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(nir[i] + red[i]) < 0.1)
            rdvi[i] = -1;
        else
            rdvi[i] = (nir[i] - red[i]) / sqrt(nir[i] + red[i]);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, rdvi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RDVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//RVI_SAR
Status RVI_SAR(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] RVI_SAR: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("RVI_SAR", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH HH =
        GDALGetRasterBand(m_outPoDataSet, 1); 
    GDALRasterBandH HV =
        GDALGetRasterBand(m_outPoDataSet, 2); 
    GDALRasterBandH VV =
        GDALGetRasterBand(m_outPoDataSet, 3); 

    float *bufferhh = new float[nImgSizeX * nImgSizeY];
    float *bufferhv = new float[nImgSizeX * nImgSizeY];
    float *buffervv = new float[nImgSizeX * nImgSizeY];
    float *rvi_sar = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(HH, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhh, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(HV, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)bufferhv, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);
    CPLErr r3 = GDALRasterIO(VV, GDALRWFlag::GF_Read, 0, 0, nImgSizeX,
                           nImgSizeY, (void *)buffervv, nImgSizeX, nImgSizeY,
                           GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure || r3 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX * nImgSizeY; i++) {
      rvi_sar[i] = bufferhv[i] / (bufferhv[i] + (bufferhh[i] + buffervv[i]) / 2);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, rvi_sar);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);

    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RVI_SAR: " + std::string(e.what()));
  }
  return Status::OK();
}

//SAVI
Status SAVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &L) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] SAVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("SAVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4);
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3);

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *savi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(nir[i] + red[i] + L) < 0.1)
            savi[i] = -1;
        else
            savi[i] = (nir[i] - red[i]) / (nir[i] + red[i] + L) * (1 + L);
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, savi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("SAVI: " + std::string(e.what()));
  }
  return Status::OK();
}

//TVI
Status TVI(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] TVI: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("TVI", input_cv->Rank()));
    
    cv::Mat input_img = input_cv->mat();
    const int nBandCount = input_img.channels();
    const int nImgSizeX = input_img.cols;
    const int nImgSizeY = input_img.rows;
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(input_img, *imgMat);
    
    GDALAllRegister();
    GDALDataset *m_outPoDataSet;
    GDALDriver *poDriver;

    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = static_cast<GDALOpenCV *>(nullptr)->OPenCVType2GCType(OPenCVty);

    poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL){
      RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to create a new file.");
    }

    m_outPoDataSet = poDriver->Create("temp_luojianet_gdal.tif", nImgSizeX, nImgSizeY, nBandCount,
        static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), NULL);
    m_outPoDataSet->SetProjection(m_outPoDataSet->GetProjectionRef());
    double dGeotransform[6];
    m_outPoDataSet->GetGeoTransform(dGeotransform);
    m_outPoDataSet->SetGeoTransform(dGeotransform);

    GDALRasterBand *pBand = NULL;
    void *ppafScan = static_cast<GDALOpenCV *>(nullptr)->AllocateMemory(GCty, nImgSizeX * nImgSizeY);
    cv::Mat tmpMat;
    for (int i = 1; i <= nBandCount; i++) {
      pBand = m_outPoDataSet->GetRasterBand(i);
      tmpMat = imgMat->at(i - 1);
      static_cast<GDALOpenCV *>(nullptr)->SetMemCopy(ppafScan, (void *)tmpMat.ptr(0), GCty, nImgSizeX * nImgSizeY);
      CPLErr err = pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan, nImgSizeX, nImgSizeY,
                                    static_cast<GDALOpenCV *>(nullptr)->GCType2GDALType(GCty), 0, 0);
      if (err == CE_Failure){
            RETURN_STATUS_UNEXPECTED("[ERROR]: Failed from CV to GDAL.");
      }
    }

    GDALRasterBandH hNir = GDALGetRasterBand(m_outPoDataSet, 4); 
    GDALRasterBandH hRed = GDALGetRasterBand(m_outPoDataSet, 3); 

    float *nir = new float[nImgSizeX * nImgSizeY];
    float *red = new float[nImgSizeX * nImgSizeY];
    float *tvi = new float[nImgSizeX * nImgSizeY];

    CPLErr r1 = GDALRasterIO(hNir, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)nir, nImgSizeX, nImgSizeY,GDALDataType::GDT_Float32, 0, 0);
    CPLErr r2 = GDALRasterIO(hRed, GDALRWFlag::GF_Read, 0, 0, nImgSizeX, nImgSizeY,
                    (void *)red, nImgSizeX, nImgSizeY, GDALDataType::GDT_Float32, 0, 0);

    if (r1 == CE_Failure || r2 == CE_Failure){
          RETURN_STATUS_UNEXPECTED("[ERROR]: Failed to read bands.");
    }

    for (int i = 0; i < nImgSizeX*nImgSizeY; i++)
    {
        if (fabs(nir[i] + red[i]) < 0.1)
            tvi[i] = -1;
        else
            tvi[i] = sqrt((nir[i] - red[i]) / (nir[i] + red[i])) + 0.5;
    }

    cv::Mat output_img = cv::Mat(nImgSizeY, nImgSizeX, CV_32FC1, tvi);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    
    imgMat->clear();
    delete imgMat;
    imgMat = NULL;
    GDALClose(m_outPoDataSet);
    GDALDestroyDriverManager();
    remove("temp_luojianet_gdal.tif");
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("TVI: " + std::string(e.what()));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace luojianet_ms
