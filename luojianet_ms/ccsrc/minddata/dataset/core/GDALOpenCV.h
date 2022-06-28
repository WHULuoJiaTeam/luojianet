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

#ifdef ENABLE_RS
#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <gdal_priv.h>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>

/***********************************************************/
// GCDataType:GDAL和OpenCV数据类型转换的中间格式
// GC_Byte   =======  GDT_Byte   =======  CV_8U  =======  unsigned char
// GC_UInt16 =======  GDT_UInt16 =======  CV_16U =======  unsigned short
// GC_Int16  =======  GDT_Int16  =======  CV_16S =======  short int
// GC_UInt32 =======  GDT_UInt32 =======  缺失   =======  unsigned long
// GC_Int32  =======  GDT_Int32  =======  CV_32S =======  long
// GC_Float32=======  GDT_Float32=======  CV_32F =======  float
// GC_Float64=======  GDT_Float64=======  CV_64F =======  double
/***********************************************************/
typedef enum{
    GC_Byte = 0,
    GC_UInt16 = 1,
    GC_Int16 = 2,
    GC_UInt32 = 3,
    GC_Int32 = 4,
    GC_Float32 = 5,
    GC_Float64 = 6,
    GC_ERRType = 7
} GCDataType;

class GDALOpenCV
{
    /***********************************************************/
    // PatchIndex:存储具体影像分块信息
    // iPatch 分类表：1----两个边界重合    2----三个边界重合   3----四个边界重合
    //  11 12 13 14     21 22 23 24    31
    // =======================
    //   11 =    21    =  12
    // =======================
    //   24 =    31    =  22
    // =======================
    //   14 =    23    =  13
    // =======================
    /***********************************************************/
    typedef struct {
        int iPatch;      //  第几类
        int row_begin;   // 影像中起始行
        int col_begin;   // 影像中起始列
        int width;       // 影像块的宽度
        int heigth;     // 影像块的高度
    } PatchIndex;


public:
    GDALOpenCV(const std::string fileName);  // 唯一构造函数
    ~GDALOpenCV(void);   // 析构函数

public:
    void Initialization();   // 初始化  实际上为获取影像格式;
    cv::Size SetPatchSize(const int r,const int c)  // 设置分块大小
    {
        m_patchSize.width = c;  m_patchSize.height = r;
        return m_patchSize;
    };

    void SetOverlappedPixel(const int num)
    {
        m_overlappedPixel = num;
    };


    bool GDAL2Mat(cv::Mat &img);  // 影像读取为Mat格式  不分块
    bool Mat2File(const std::string outFileName,cv::Mat &img,const int flag = 1);  // Mat文件输出为影像
    //  flag = 默认为1  输出TIFF   另外还支持ENVI 和 ARDAS数据格式

    int GetImgToPatchNum();  // 返回影像分块数  和  获取影像分块信息
    void GetROIFromPatchIndex(const int,cv::Mat &);  //  获取对应块编号的影像
    bool SetROIMatToFileByIndex(const std::string outFile,cv::Mat &img,
        const int index,const int flag = 1); // 影像分块写入  有待改进  具体细节有点商榷

    GCDataType GDALType2GCType(const GDALDataType ty); // GDAL Type ==========> GDALOpenCV Type
    GDALDataType GCType2GDALType(const GCDataType ty); //  GDALOpenCV Type ==========> GDAL Type
    GCDataType OPenCVType2GCType(const int ty); // OPenCV Type ==========> GDALOpenCV Type
    int GCType2OPenCVType(const GCDataType ty); // GDALOpenCV Type ==========> OPenCV Type

    void* SetMemCopy(void *dst,const void *src,const GCDataType lDataType,const long long lSize);
    void* AllocateMemory(const GCDataType lDataType,const long long lSize);   // 智能分配内存


public:
    GCDataType m_dataType;  // 数据类型
    int m_imgWidth; // 影像宽度   列数
    int m_imgHeigth; // 影像高度   行数
    int m_bandNum; // 影像波段数
private:
    //GDALDataType m_gdalType;
    GDALDataset *m_poDataSet;  // 数据驱动集
    GDALDataset *m_outPoDataSet;
    cv::Size m_patchSize;// 分块图像大小
    //std::string m_fileName; // 文件名  打开
    std::vector<PatchIndex> *m_patchIndex;//分块标识
    int m_overlappedPixel;
};


//////////////////////////////////////////////////////////
//    使用说明：
/////////////////////////////////////////////////////////
//   GDALOpenCV gdalOpenCV(fileName);
//   gdalOpenCV.Initialization();

#endif
