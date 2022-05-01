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

#include "GDALOpenCV.h"
 
GDALOpenCV::GDALOpenCV(const std::string fileName)
{
    m_poDataSet = NULL;
    GDALAllRegister();
    m_poDataSet = (GDALDataset*)GDALOpen(fileName.c_str(),GA_ReadOnly);
    m_outPoDataSet = NULL;
}
 
GDALOpenCV::~GDALOpenCV(void)
{
    if(m_poDataSet!=NULL)
        GDALClose((GDALDatasetH)m_poDataSet);
    if(m_outPoDataSet!=NULL)
        GDALClose((GDALDatasetH)m_outPoDataSet);
    m_patchIndex->clear();
    delete m_patchIndex;
    m_patchIndex = NULL;
 
}
 
void GDALOpenCV::Initialization()
{
    if(!m_poDataSet)
        return;
    m_imgHeigth= m_poDataSet->GetRasterYSize();  // 影像行
    m_imgWidth = m_poDataSet->GetRasterXSize();  // 影像列
    m_bandNum = m_poDataSet->GetRasterCount(); // 影像波段数
    m_overlappedPixel = -1;     //  重复像素个数
 
    GDALRasterBand *pBand = m_poDataSet->GetRasterBand(1);
    GDALDataType gdalTy = pBand->GetRasterDataType();
    m_dataType = GDALType2GCType(gdalTy);
 
    m_patchSize.width = m_imgWidth;
    m_patchSize.height = m_imgHeigth;
 
    m_patchIndex = new std::vector<PatchIndex>(1);
    PatchIndex tmp = {1,0,0,m_imgWidth,m_imgHeigth};
    m_patchIndex->at(0) = tmp;
}
 
bool GDALOpenCV::GDAL2Mat(cv::Mat &img)
{
    if(!m_poDataSet)
        return false;
 
    GDALRasterBand *pBand = NULL;  // 波段
    void *pafBuffer = AllocateMemory(m_dataType,m_imgHeigth*m_imgWidth);  // 开辟内存
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(m_bandNum);   // 存储各波段
    cv::Mat *tmpMat = NULL;  // 临时存储一个波段
 
    int iBand = 0; // 波段标记
    while(iBand<m_bandNum)       
    {
        pBand = m_poDataSet->GetRasterBand(++iBand);
        pBand->RasterIO(GF_Read,0,0,m_imgWidth,m_imgHeigth,pafBuffer,m_imgWidth,
            m_imgHeigth,GCType2GDALType(m_dataType),0,0);
        tmpMat = new cv::Mat(m_imgHeigth,m_imgWidth,GCType2OPenCVType(m_dataType),pafBuffer);
        imgMat->at(iBand-1) = (*tmpMat).clone();
        delete tmpMat;
        tmpMat = NULL;
    }
    cv::merge(*imgMat,img);
 
    // 内存管理
    delete pafBuffer;   pafBuffer = NULL;
    imgMat->clear(); delete imgMat;  imgMat = NULL;
 
    return true;
}
 
 
bool GDALOpenCV::Mat2File( const std::string outFileName,cv::Mat &img,
                          const int flag /*= 1*/ )
{
    if(img.empty())
        return false;
 
    const int nBandCount=img.channels();
    const int nImgSizeX=img.cols;
    const int nImgSizeY=img.rows;
 
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(nBandCount);
    cv::split(img,*imgMat);
 
    GDALAllRegister();
    //GDALDataset *poDataset;   //GDAL数据集
    GDALDriver *poDriver;     //驱动，用于创建新的文件
    ////////////////////////////////
    ///flag：1 ====》TIFF
    ///      2 ====》HFA
    ///      3 ====》ENVI
    std::string pszFormat; //存储数据类型
    switch (flag) {
    case 1:
        pszFormat = "GTiff";
        break;
    case 2:
        pszFormat = "HFA";
        break;
    case 3:
        pszFormat = "ENVI";
        break;
    default:
        return 0;
    }
 
    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = OPenCVType2GCType(OPenCVty);
 
    poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat.c_str());
    if(poDriver == NULL)
        return 0;
    if(m_outPoDataSet == NULL)
    {
        m_outPoDataSet=poDriver->Create(outFileName.c_str(),nImgSizeX,nImgSizeY,nBandCount,
            GCType2GDALType(GCty),NULL);
        m_outPoDataSet->SetProjection(m_poDataSet->GetProjectionRef());
        double dGeotransform[6];
        m_poDataSet->GetGeoTransform(dGeotransform);
        m_outPoDataSet->SetGeoTransform(dGeotransform);
    }
 
    //  循环写入文件
    GDALRasterBand *pBand = NULL;
    void *ppafScan = AllocateMemory(GCty,nImgSizeX*nImgSizeY);
    int n1 = nImgSizeY;
    int nc = nImgSizeX;
    cv::Mat tmpMat;
    for(int i = 1;i<=nBandCount;i++)
    {
        pBand = m_outPoDataSet->GetRasterBand(i);
        tmpMat = imgMat->at(i-1);
        if(tmpMat.isContinuous())
            SetMemCopy(ppafScan,(void*)tmpMat.ptr(0),GCty,nImgSizeX*nImgSizeY);
        else
            return false;
        CPLErr err = pBand->RasterIO(GF_Write,0,0,nImgSizeX,nImgSizeY,ppafScan,
            nImgSizeX,nImgSizeY,GCType2GDALType(GCty),0,0);
    }
 
    delete ppafScan;    ppafScan = NULL;
    imgMat->clear();delete imgMat;imgMat = NULL;
    return 1;
}
 
 
int GDALOpenCV::GetImgToPatchNum()
{
    if(m_patchSize.width >= m_imgWidth || m_patchSize.height >= m_imgHeigth)
        return 1;
    if(m_overlappedPixel == -1)
        return 1;
    ////////////////分块核心代码/////////////////////
    //////////////分块数确定////////////////////////
    int rPatchNum = cvCeil((m_imgHeigth*1.0 -m_patchSize.height)/(m_patchSize.height - m_overlappedPixel)) + 1;
    int cPatchNum = cvCeil((m_imgWidth*1.0 - m_patchSize.width)/(m_patchSize.width - m_overlappedPixel)) +1;
 
    PatchIndex tmpPatchIndex;
    int rowBegin = 0;
    int colBegin = 0;
 
    m_patchIndex->clear();
    for(int i = 0;i != rPatchNum; i++)
    {
        for(int j = 0;j != cPatchNum; j++)
        {
            if(0x00 == i && 0x00 == j)
                tmpPatchIndex.iPatch = 11;
            else if(0x00 == i && cPatchNum-1 == j)
                tmpPatchIndex.iPatch = 12;
            else if(rPatchNum-1 == i && cPatchNum-1 == j)
                tmpPatchIndex.iPatch = 13;
            else if(rPatchNum-1 == i && 0x00 == j)
                tmpPatchIndex.iPatch = 14;
            else if(0x00 == i && j>0 && j< cPatchNum-1)
                tmpPatchIndex.iPatch = 21;
            else if(j == cPatchNum -1 && i>0 && i<rPatchNum -1)
                tmpPatchIndex.iPatch = 22;
            else if(i == rPatchNum-1 && j>0 && j<cPatchNum -1)
                tmpPatchIndex.iPatch = 23;
            else if(0x00 == j && i > 0 && i<rPatchNum -1)
                tmpPatchIndex.iPatch = 24;
            else
                tmpPatchIndex.iPatch = 31;
 
            tmpPatchIndex.row_begin = rowBegin;
            tmpPatchIndex.col_begin = colBegin;
            if(rowBegin+m_patchSize.height > m_imgHeigth)
                tmpPatchIndex.heigth = m_imgHeigth - rowBegin;
            else
                tmpPatchIndex.heigth = m_patchSize.height;
            if(colBegin+m_patchSize.width > m_imgWidth)
                tmpPatchIndex.width = m_imgWidth - colBegin;
            else
                tmpPatchIndex.width = m_patchSize.width;
            m_patchIndex->push_back(tmpPatchIndex);
            colBegin = colBegin + m_patchSize.width - m_overlappedPixel;
        }
        rowBegin = rowBegin + m_patchSize.height - m_overlappedPixel;
        colBegin = 0;
    }
    return (int)m_patchIndex->size();
}
 
 
void GDALOpenCV::GetROIFromPatchIndex(const int index,cv::Mat &img)
{
    int patchNum = (int)m_patchIndex->size();
    if(index > patchNum || index < 1)
        return;
    PatchIndex curPatchIndex = m_patchIndex->at(index-1);
    int patchRowBegin = curPatchIndex.row_begin;
    int patchColBegin = curPatchIndex.col_begin;
    int patchWidth = curPatchIndex.width;
    int patchHeight = curPatchIndex.heigth;
 
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(m_bandNum);// 存储读取的每个波段数据
    void *pafBuffer = AllocateMemory(m_dataType,patchWidth*patchHeight); // 内存分配
    GDALRasterBand *pBand = NULL;
    cv::Mat *tmpMat = NULL;
 
    int iBand = 0; // 波段标记
    while(iBand<m_bandNum)
    {
        pBand = m_poDataSet->GetRasterBand(++iBand);
        pBand->RasterIO(GF_Read,patchColBegin,patchRowBegin,patchWidth,patchHeight,pafBuffer,
            patchWidth,patchHeight,GCType2GDALType(m_dataType),0,0);
        tmpMat =new cv::Mat(patchHeight,patchWidth,GCType2OPenCVType(m_dataType),pafBuffer);
        imgMat->at(iBand-1) = (*tmpMat).clone();
        delete tmpMat;
        tmpMat = NULL;
    }
    cv::merge(*imgMat,img);
 
    delete pafBuffer;pafBuffer = NULL;
    //delete pBand;pBand = NULL;
    imgMat->clear();delete imgMat;imgMat = NULL;
}
 
 
bool GDALOpenCV::SetROIMatToFileByIndex( const std::string outFileName,cv::Mat &img,
                                        const int index,const int flag /*= 1*/ )
{
    if(!outFileName.c_str() || img.empty())
        return false;
 
    const int nBandCount=img.channels();
    const int nImgSizeX=img.cols;
    const int nImgSizeY=img.rows;
 
    PatchIndex tmpPatchIndex = m_patchIndex->at(index-1);
    if(tmpPatchIndex.heigth != nImgSizeY || tmpPatchIndex.width != nImgSizeX)
        return false;
 
    std::vector<cv::Mat> *imgMat = new std::vector<cv::Mat>(m_bandNum);
    cv::split(img,*imgMat);
 
    const int ty = (*imgMat).at(0).type();
 
    GDALAllRegister();
    //GDALDataset *poDataset = NULL;   //GDAL数据集
    GDALDriver *poDriver = NULL;      //驱动，用于创建新的文件
 
    ////////////////////////////////
    ///flag：1 ====》TIFF
    ///      2 ====》HFA
    ///      3 ====》ENVI
    std::string pszFormat; //存储数据类型
    switch (flag) {
    case 1:
        pszFormat = "GTiff";
        break;
    case 2:
        pszFormat = "HFA";
        break;
    case 3:
        pszFormat = "ENVI";
        break;
    default:
        return 0;
    }
 
    poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat.c_str());
    int OPenCVty = imgMat->at(0).type();
    GCDataType GCty = OPenCVType2GCType(OPenCVty);
    if(poDriver == NULL)
        return 0;
    if(m_outPoDataSet == NULL)
    {
        m_outPoDataSet=poDriver->Create(outFileName.c_str(),m_imgWidth,m_imgHeigth,nBandCount,
            GCType2GDALType(GCty),NULL);
        m_outPoDataSet->SetProjection(m_poDataSet->GetProjectionRef());
        double dGeotransform[6];
        m_poDataSet->GetGeoTransform(dGeotransform);
        m_outPoDataSet->SetGeoTransform(dGeotransform);
    }
    //  循环写入文件
    GDALRasterBand *pBand = NULL;  
    int n1 = nImgSizeY;
    int nc = nImgSizeX;
    int overPix = int(m_overlappedPixel/2);
 
    void *ppafScan = NULL;
    int curCol = 0;
    int curRow = 0;
     
    for(int i = 1;i<=nBandCount;i++)
    {
        pBand = m_outPoDataSet->GetRasterBand(i);
        cv::Mat tmpMat;
        tmpMat = imgMat->at(i-1);
        // 文件的写入
        if(tmpPatchIndex.iPatch == 11)     
        {  
            curCol = nImgSizeX-overPix;
            curRow = nImgSizeY - overPix;
            cv::Rect r1(0,0,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin,tmpPatchIndex.row_begin,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
        if(tmpPatchIndex.iPatch == 12)
        {  
            curCol = nImgSizeX-overPix;
            curRow = nImgSizeY - overPix;
            cv::Rect r1(overPix,0,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin+overPix,tmpPatchIndex.row_begin,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
        if(tmpPatchIndex.iPatch == 13)
        {
            curCol = nImgSizeX-overPix;
            curRow = nImgSizeY - overPix;
            cv::Rect r1(overPix,overPix,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin+overPix,tmpPatchIndex.row_begin+overPix,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
        if(tmpPatchIndex.iPatch == 14)
        {
            curCol = nImgSizeX-overPix;
            curRow = nImgSizeY - overPix;
            cv::Rect r1(0,overPix,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin,tmpPatchIndex.row_begin+overPix,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
        if(tmpPatchIndex.iPatch == 21)
        {
            curCol = nImgSizeX-2*overPix;
            curRow = nImgSizeY - overPix;
            cv::Rect r1(overPix,0,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin+overPix,tmpPatchIndex.row_begin,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
        if(tmpPatchIndex.iPatch == 22)
        {
            curCol = nImgSizeX- overPix;
            curRow = nImgSizeY - 2*overPix;
            cv::Rect r1(overPix,overPix,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin+overPix,tmpPatchIndex.row_begin+overPix,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
        if(tmpPatchIndex.iPatch == 23)
        {
            curCol = nImgSizeX- 2*overPix;
            curRow = nImgSizeY - overPix;
            cv::Rect r1(overPix,overPix,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin+overPix,tmpPatchIndex.row_begin+overPix,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
        if(tmpPatchIndex.iPatch == 24)
        {
            curCol = nImgSizeX- overPix;
            curRow = nImgSizeY - 2*overPix;
            cv::Rect r1(0,overPix,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin,tmpPatchIndex.row_begin+overPix,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
        if(tmpPatchIndex.iPatch == 31)
        {
            curCol = nImgSizeX- 2*overPix;
            curRow = nImgSizeY - 2*overPix;
            cv::Rect r1(overPix,overPix,curCol,curRow);
            tmpMat = tmpMat(r1);
            cv::Mat *newMat = new cv::Mat(curRow,curCol,ty);
            *newMat = tmpMat.clone();
            ppafScan = AllocateMemory(GCty,curCol*curRow);
            if((*newMat).isContinuous())
                SetMemCopy(ppafScan,(void*)(*newMat).ptr(0),GCty,curCol*curRow);
            else
                return false;
            pBand->RasterIO(GF_Write,tmpPatchIndex.col_begin+overPix,tmpPatchIndex.row_begin+overPix,
                curCol,curRow,ppafScan,curCol,curRow,GCType2GDALType(GCty),0,0);
            delete newMat;newMat = NULL;
        }
    }
    delete ppafScan;ppafScan = NULL;
    imgMat->clear();delete imgMat;imgMat = NULL;
    return 1;
}
 
GCDataType GDALOpenCV::GDALType2GCType( const GDALDataType ty )
{
    switch(ty)
    {
    case GDT_Byte:
        return GC_Byte;
    case GDT_UInt16:
        return GC_UInt16;
    case GDT_Int16:
        return GC_Int16;
    case GDT_UInt32:
        return GC_UInt32;
    case GDT_Int32:
        return GC_Int32;
    case GDT_Float32:
        return GC_Float32;
    case GDT_Float64:
        return GC_Float64;
    default:
        assert(false);
        return GC_ERRType;
    }
}
 
GCDataType GDALOpenCV::OPenCVType2GCType( const int ty )
{
    switch(ty)
    {
    case 0:
        return GC_Byte;
    case 2:
        return GC_UInt16;
    case 3:
        return GC_Int16;
    case 4:
        return GC_Int32;
    case 5:
        return GC_Float32;
    case 6:
        return GC_Float64;
    default:
        assert(false);
        return GC_ERRType;
    }
}
 
GDALDataType GDALOpenCV::GCType2GDALType( const GCDataType ty )
{
    switch(ty)
    {
    case GC_Byte:
        return GDT_Byte;
    case GC_UInt16:
        return GDT_UInt16;
    case GC_Int16:
        return GDT_Int16;
    case GC_UInt32:
        return GDT_UInt32;
    case GC_Int32:
        return GDT_Int32;
    case GC_Float32:
        return GDT_Float32;
    case GC_Float64:
        return GDT_Float64;
    default:
        assert(false);
        return GDT_TypeCount;
    }
}
 
int GDALOpenCV::GCType2OPenCVType( const GCDataType ty )
{
    switch(ty)
    {
    case GC_Byte:
        return 0;
    case GC_UInt16:
        return 2;
    case GC_Int16:
        return 3;
    case GC_Int32:
        return 4;
    case GC_Float32:
        return 5;
    case GC_Float64:
        return 6;
    default:
        assert(false);
        return -1;
    }
}
 
void* GDALOpenCV::AllocateMemory(const GCDataType lDataType,const long long lSize )
{
    assert(0!=lSize);
    void* pvData = NULL;
    switch (lDataType)
    {
    case GC_Byte:
        pvData = new(std::nothrow) unsigned char[lSize];
        break;
    case GC_UInt16:
        pvData = new(std::nothrow) unsigned short int[lSize];
        break;
    case GC_Int16:
        pvData = new(std::nothrow) short int[lSize];
        break;
    case GC_UInt32:
        pvData = new(std::nothrow) unsigned long[lSize];
        break;
    case GC_Int32:
        pvData = new(std::nothrow) long[lSize];
        break;
    case GC_Float32:
        pvData = new(std::nothrow) float[lSize];
        break;
    case GC_Float64:
        pvData = new(std::nothrow) double[lSize];
        break;
    default:
        assert(false);
        break;
    }
    return pvData;
}
 
void* GDALOpenCV::SetMemCopy( void *dst,const void *src,
                             const GCDataType lDataType,
                             const long long lSize )
{
    assert(0!=lSize);
    switch (lDataType)
    {
    case GC_Byte:
        return memmove(dst,src,sizeof(unsigned char)*lSize);
    case GC_UInt16:
        return memmove(dst,src,sizeof(unsigned short)*lSize);
    case GC_Int16:
        return memmove(dst,src,sizeof(short int)*lSize);
    case GC_UInt32:
        return memmove(dst,src,sizeof(unsigned long)*lSize);
    case GC_Int32:
        return memmove(dst,src,sizeof(long)*lSize);
    case GC_Float32:
        return memmove(dst,src,sizeof(float)*lSize);
    case GC_Float64:
        return memmove(dst,src,sizeof(double)*lSize);
    default:
        return NULL;
    }
}