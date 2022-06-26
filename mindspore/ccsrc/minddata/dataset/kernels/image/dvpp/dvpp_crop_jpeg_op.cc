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

#include <string>
#include <vector>
#include <iostream>
#include "include/api/context.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/kernels/image/dvpp/utils/MDAclProcess.h"
#include "minddata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_crop_jpeg_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
Status DvppCropJpegOp::Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) {
  IO_CHECK(input, output);
  try {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetDeviceBuffer() != nullptr, "The input image buffer is empty.");
    std::string last_step = "Resize";
    std::shared_ptr<DvppDataInfo> imageinfo(processor_->Get_Resized_DeviceData());
    if (!imageinfo->data) {
      last_step = "Decode";
    }
    APP_ERROR ret = processor_->JPEG_C(last_step);
    if (ret != APP_ERR_OK) {
      ret = processor_->Release();
      CHECK_FAIL_RETURN_UNEXPECTED(ret == APP_ERR_OK, "Release memory failed.");
      std::string error = "Error in dvpp crop processing:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    std::shared_ptr<DvppDataInfo> CropOut(processor_->Get_Croped_DeviceData());
    const TensorShape dvpp_shape({1, 1, 1});
    const DataType dvpp_data_type(DataType::DE_UINT8);
    RETURN_IF_NOT_OK(mindspore::dataset::DeviceTensor::CreateEmpty(dvpp_shape, dvpp_data_type, output));
    RETURN_IF_NOT_OK((*output)->SetAttributes(CropOut->data, CropOut->dataSize, CropOut->width, CropOut->widthStride,
                                              CropOut->height, CropOut->heightStride));
    if (!((*output)->HasDeviceData())) {
      std::string error = "[ERROR] Fail to get the Output result from device memory!";
      RETURN_STATUS_UNEXPECTED(error);
    }
  } catch (const std::exception &e) {
    std::string error = "[ERROR] Fail in DvppCropJpegOp:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppCropJpegOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!IsNonEmptyJPEG(input)) {
    RETURN_STATUS_UNEXPECTED("DvppCropJpegOp only support process jpeg image.");
  }
  try {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetBuffer() != nullptr, "The input image buffer is empty.");
    unsigned char *buffer = const_cast<unsigned char *>(input->GetBuffer());
    DvppDataInfo imageinfo;
    imageinfo.dataSize = input->SizeInBytes();
    imageinfo.data = static_cast<uint8_t *>(buffer);
    std::vector<uint32_t> yuv_shape_ = input->GetYuvShape();
    const size_t yuv_shape_size = 4;
    CHECK_FAIL_RETURN_UNEXPECTED(yuv_shape_.size() == yuv_shape_size, "yuv_shape requires 4 elements.");
    imageinfo.width = yuv_shape_[0];
    imageinfo.widthStride = yuv_shape_[1];
    imageinfo.height = yuv_shape_[2];
    imageinfo.heightStride = yuv_shape_[3];
    imageinfo.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ResourceInfo resource;
    resource.deviceIds.insert(0);
    std::shared_ptr<ResourceManager> instance = ResourceManager::GetInstance();
    APP_ERROR ret = instance->InitResource(resource);
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in Init D-chip:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    int deviceId = *(resource.deviceIds.begin());
    aclrtContext context = instance->GetContext(deviceId);
    // Second part end where we initialize the resource of D-chip and set up all configures
    MDAclProcess process(crop_width_, crop_height_, context, true);
    ret = process.InitResource();
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in Init resource:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }

    ret = process.JPEG_C(imageinfo);
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in dvpp crop processing:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }

    // Third part end where we execute the core function of dvpp
    auto data = std::static_pointer_cast<unsigned char>(process.Get_Memory_Data());
    unsigned char *ret_ptr = data.get();
    std::shared_ptr<DvppDataInfo> CropOut(process.Get_Croped_DeviceData());
    dsize_t dvpp_length = CropOut->dataSize;
    const TensorShape dvpp_shape({dvpp_length, 1, 1});
    uint32_t crop_height = CropOut->height;
    uint32_t crop_heightStride = CropOut->heightStride;
    uint32_t crop_width = CropOut->width;
    uint32_t crop_widthStride = CropOut->widthStride;
    const DataType dvpp_data_type(DataType::DE_UINT8);
    RETURN_IF_NOT_OK(mindspore::dataset::Tensor::CreateFromMemory(dvpp_shape, dvpp_data_type, ret_ptr, output));
    RETURN_IF_NOT_OK((*output)->SetYuvShape(crop_width, crop_widthStride, crop_height, crop_heightStride));
    if (!((*output)->HasData())) {
      std::string error = "[ERROR] Fail to get the Output result from memory!";
      RETURN_STATUS_UNEXPECTED(error);
    }
    ret = process.device_memory_release();
    CHECK_FAIL_RETURN_UNEXPECTED(ret == APP_ERR_OK, "Release device memory failed.");
    ret = process.Release();
    CHECK_FAIL_RETURN_UNEXPECTED(ret == APP_ERR_OK, "Release host memory failed.");
    // Last part end where we transform the processed data into a tensor which can be applied in later units.
  } catch (const std::exception &e) {
    std::string error = "[ERROR] Fail in DvppCropJpegOp:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppCropJpegOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({-1, 1, 1});  // we don't know what is output image size, but we know it should be 1 channels
  if (inputs.size() < 1) {
    RETURN_STATUS_UNEXPECTED("DvppCropJpegOp::OutputShape inputs is null");
  }
  if (inputs[0].Rank() == 1) outputs.emplace_back(out);
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "Input has a wrong shape");
}

Status DvppCropJpegOp::SetAscendResource(const std::shared_ptr<DeviceResource> &resource) {
  processor_ = std::static_pointer_cast<MDAclProcess>(resource->GetInstance());
  if (!processor_) {
    RETURN_STATUS_UNEXPECTED("Resource initialize fail, please check your env");
  }
  APP_ERROR ret = processor_->SetCropParas(crop_width_, crop_height_);
  CHECK_FAIL_RETURN_UNEXPECTED(ret == APP_ERR_OK, "SetCropParas failed.");
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
