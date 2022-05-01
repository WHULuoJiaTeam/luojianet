# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import numpy as np
from PIL import Image
import luojianet_ms.dataset.vision.c_transforms as vision
from luojianet_ms import log as logger

def test_eager_CIWI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape
    img = vision.CIWI(digital_C = 100.0)(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_CIWI.tif", img)

def test_eager_DVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = vision.DVI()(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_dvi.tif", img)
    
def test_eager_EVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = vision.EVI()(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_evi.tif", img)
    
def test_eager_LBP():
    img = cv2.imread("../data/dataset/test_rs_index/RGB/building.tiff")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape
    img = vision.LBP(N = 3)(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_LBP.tif", img)

def test_eager_MBI():
    img = cv2.imread("../data/dataset/test_rs_index/RGB/building.tiff")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape
    img = vision.MBI(s_min = 3, s_max = 20, delta_s = 1)(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_MBI.tif", img)

def test_eager_MSAVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = vision.MSAVI()(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_msavi.tif", img)

def test_eager_NDVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = vision.NDVI()(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_ndvi.tif", img)
    
def test_eager_NDWI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = vision.NDWI()(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_ndwi.tif", img)

def test_eager_OSAVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape
    img = vision.OSAVI(theta = 0.16)(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_osavi.tif", img)

def test_eager_RDVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = vision.RDVI()(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_rdvi.tif", img)
    
def test_eager_RVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = vision.RVI()(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_rvi.tif", img)
    
def test_eager_SAVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape
    img = vision.SAVI(L = 0.5)(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_savi.tif", img)

def test_eager_TVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape
    img = vision.TVI()(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_tvi.tif", img)
    
def test_eager_WDRVI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif", cv2.IMREAD_UNCHANGED)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape
    img = vision.WDRVI(alpha = 0.1)(img)
    
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_wdrvi.tif", img)
    
def test_eager_ANDWI():
    band2 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B2.TIF",-1)
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band4 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B4.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    band7 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B7.TIF",-1)
    img = cv2.merge([band2, band3, band4, band5, band6, band7])
    img = vision.ANDWI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_andwi.tif", img)

def test_eager_AWEI():
    #band2 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B2.TIF",-1)
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    band7 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B7.TIF",-1)
    #img = cv2.merge([band2, band3, band5, band6, band7])
    img = cv2.merge([band3, band5, band6, band7])
    img = vision.NWI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_aweiF.tif", img)
'''
def test_eager_BMI():
    bandhh = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hh-20220117t144920-20220117t145019-041502-04ef79-001.tiff",-1) ##
    bandvv = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hv-20220117t144920-20220117t145019-041502-04ef79-002.tiff",-1) ##
    img = cv2.merge([bandhh, bandvv])
    img = vision.BMI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_bmi.tif", img)
'''
'''
def test_eager_CSI():
    bandhh = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hh-20220117t144920-20220117t145019-041502-04ef79-001.tiff",-1) ##
    bandvv = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hv-20220117t144920-20220117t145019-041502-04ef79-002.tiff",-1) ##
    img = cv2.merge([bandhh, bandvv])
    img = vision.CSI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_csi.tif", img)
'''
def test_eager_EWI_W():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band4 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B4.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    img = cv2.merge([band3, band4, band5, band6])
    img = vision.EWI_W(m = 0.1, n = 0.5)(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_ewi_w.tif", img)

def test_eager_EWI_Y():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)

    img = cv2.merge([band3, band5, band6])
    img = vision.EWI_Y()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_ewi_y.tif", img)

def test_eager_FNDWI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif",-1)
    img = vision.FNDWI(S = 1, CNIR = 40)(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_fndwi.tif", img)

def test_eager_LBP():
    img = cv2.imread("../data/dataset/test_rs_index/RGB/building.tiff")
    img = vision.LBP(N = 2)(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/building_lbp.tif", img)
    
def test_eager_Gabor():
    img = cv2.imread("../data/dataset/test_rs_index/RGB/building.tiff")
    img = vision.Gabor(if_opencv_kernal = True)(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/building_gabor_true.tif", img)
    
def test_eager_GLCM():
    img = cv2.imread("../data/dataset/test_rs_index/RGB/building.tiff")
    img = vision.GLCM(N = 2)(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/building_glcm_2.tif", img)
    
def test_eager_GNDWI():
    img = cv2.imread("../data/dataset/test_rs_index/MULTIBAND/temp.tif",-1)
    img = vision.GNDWI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/temp_gndwi.tif", img)

def test_eager_MBWI():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band4 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B4.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    band7 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B7.TIF",-1)
    img = cv2.merge([band3, band4, band5, band6, band7])
    img = vision.MBWI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_mbwi.tif", img)

def test_eager_MCIWI():
    band4 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B4.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    img = cv2.merge([band4, band5, band6])
    img = vision.MCIWI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_mciwi.tif", img)
    
def test_eager_MNDWI():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    img = cv2.merge([band3, band6])
    img = vision.MNDWI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_mndwi.tif", img)
    
def test_eager_NDPI():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    img = cv2.merge([band3, band6])
    img = vision.NDPI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_ndpi.tif", img)
    
def test_eager_NWI():
    band2 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B2.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    band7 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B7.TIF",-1)
    img = cv2.merge([band2, band5, band6, band7])
    img = vision.NWI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_nwi.tif", img)
    
def test_eager_PSI():
    bandhh = cv2.imread("../data/dataset/test_rs_index/SAR/HHHV/s1a-ew-grd-hh-20220117t144920-20220117t145019-041502-04ef79-001.tiff",-1)
    bandhv = cv2.imread("../data/dataset/test_rs_index/SAR/HHHV/s1a-ew-grd-hv-20220117t144920-20220117t145019-041502-04ef79-002.tiff",-1)
    img = cv2.merge([bandhh, bandhv])
    img = vision.PSI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/sar_PSI.tif", img)
    
def test_eager_RFDI():
    bandhh = cv2.imread("../data/dataset/test_rs_index/SAR/HHHV/s1a-ew-grd-hh-20220117t144920-20220117t145019-041502-04ef79-001.tiff",-1)
    bandhv = cv2.imread("../data/dataset/test_rs_index/SAR/HHHV/s1a-ew-grd-hv-20220117t144920-20220117t145019-041502-04ef79-002.tiff",-1)
    img = cv2.merge([bandhh, bandhv])
    img = vision.RFDI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/sar_RFDI.tif", img)
'''
def test_eager_RVI_SAR():
    bandhh = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hh-20220117t144920-20220117t145019-041502-04ef79-001.tiff",-1) ##
    bandhv = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hh-20220117t144920-20220117t145019-041502-04ef79-001.tiff",-1) ##
    bandvv = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hv-20220117t144920-20220117t145019-041502-04ef79-002.tiff",-1) ##
    img = cv2.merge([bandhh, bandhv, bandvv])
    img = vision.RVI_SAR()(img)
    img = cv2.imwrite("/test_luojianet/luojianet1/luojianet/landsat8_RVI_SAR.tif", img)
'''    
def test_eager_SRWI():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    img = cv2.merge([band3, band6])
    img = vision.SRWI()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_srwi.tif", img)
'''    
def test_eager_VSI():
    bandhh = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hh-20220117t144920-20220117t145019-041502-04ef79-001.tiff",-1) ##
    bandhv = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hh-20220117t144920-20220117t145019-041502-04ef79-001.tiff",-1) ##
    bandvv = cv2.imread("./test_data/SAR/HHHV/s1a-ew-grd-hv-20220117t144920-20220117t145019-041502-04ef79-002.tiff",-1) ##
    img = cv2.merge([bandhh, bandhv, bandvv])
    img = vision.VSI()(img)
    img = cv2.imwrite("/test_luojianet/luojianet1/luojianet/landsat8_VSI.tif", img)
'''
def test_eager_WI_H():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band4 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B4.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    img = cv2.merge([band3, band4, band6])
    img = vision.WI_H()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_wi_h.tif", img)
    
def test_eager_WI_F():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band4 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B4.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    band7 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B7.TIF",-1)
    img = cv2.merge([band3, band4, band5, band6, band7])
    img = vision.WI_F()(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_wi_f.tif", img)
    
def test_eager_WNDWI():
    band3 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B3.TIF",-1)
    band5 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B5.TIF",-1)
    band6 = cv2.imread("../data/dataset/test_rs_index/LANDSAT8/LC08_L1TP_122039_20131012_20170429_01_T1_B6.TIF",-1)
    img = cv2.merge([band3, band5, band6])
    img = vision.WNDWI(alpha = 0.45)(img)
    img = cv2.imwrite("../data/dataset/test_rs_index/test_results/land8_wndwi.tif", img)

if __name__ == '__main__':
    test_eager_NDVI()
    test_eager_CIWI()
    test_eager_SAVI()
    test_eager_DVI()
    test_eager_EVI()
    test_eager_MSAVI()
    test_eager_RVI()
    test_eager_TVI()
