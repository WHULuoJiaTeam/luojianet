#include "gdal_priv.h"

/**
 大幅影像进行分块读取处理

 @param pszSrcFile 指定的影像文件名
 @param pszDstFile 指定的输出影像文件名
 @param pszFormat 输出影像格式
 @param nBlockSize 分块处理的大小
 @return 成功返回 0，否则返回 -1
 */
int BlockProcess(const char* pszSrcFile, const char* pszDstFile, const char* pszFormat, int nBlockSize)
{
	// 注册所有驱动
	GDALAllRegister();

	/* --------------获取输入影像信息----------------- */
	// 打开输入影像
	GDALDataset *poSrcDS = (GDALDataset*)GDALOpen(pszSrcFile, GA_ReadOnly);
	if (poSrcDS == NULL)
	{
		return -1;
	}

	// 获取影像宽高和波段数据
	int initXSize = poSrcDS->GetRasterXSize();
	int initYSize = poSrcDS->GetRasterYSize();
	int initBands = poSrcDS->GetRasterCount();
	GDALDataType initDataType = poSrcDS->GetRasterBand(1)->GetRasterDataType();

	// 获取输入影像仿射变换参数
	double adfGeotransform[6] = { 0 };
	poSrcDS->GetGeoTransform(adfGeotransform);
	//adfGeoTransform[0] /* top left x */
	//adfGeoTransform[1] /* w-e pixel resolution */
	//adfGeoTransform[2] /* rotation, 0 if image is "north up" */
	//adfGeoTransform[3] /* top left y */
	//adfGeoTransform[4] /* rotation, 0 if image is "north up" */
	//adfGeoTransform[5] /* n-s pixel resolution */

	// 获取输入影像空间参考(投影信息)
	const char* pszProj = poSrcDS->GetProjectionRef();

	/* ------------------------对输出影像进行操作---------------- ------*/
	// 获取输出影像驱动
	GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
	if (poDriver == NULL)
	{
		return -1;
	}

	// 创建输出影像，输出影像波段和原始影像波段一样，大小为nBlockSize
	GDALDataset *poDstDS = poDriver->Create(pszDstFile, nBlockSize, nBlockSize, initBands, initDataType, NULL);
	if (poDstDS == NULL) // 如果创建失败则返回
	{
		GDALClose((GDALDatasetH)poSrcDS);
		return -1;
	}

	// 设置输出影像仿射变换参数，与原图一致
	poDstDS->SetGeoTransform(adfGeotransform);
	// 设置输出影像空间参考，与原图一致
	poDstDS->SetProjection(pszProj);

	// 分配输入影像分块缓存
	GByte *pSrcData = new GByte[nBlockSize*nBlockSize*initBands];
	// 分配输出影像分块缓存
	GByte *pDstData = new GByte[nBlockSize*nBlockSize*initBands];

	// 定义读取输入影像波段顺序
	int *pBandMaps = new int[initBands];
	for (int b = 0; b < initBands; b++)
	{
		pBandMaps[b] = b + 1;
	}

	// 循环分块并进行处理
	for (int i = 0; i < initYSize; i += nBlockSize) // 循环影像高
	{
		for (int j = 0; j < initXSize; j += nBlockSize) // 循环影像宽
		{
			// 定义两个变量来保存分块大小
			int nXBK = nBlockSize;
			int nYBk = nBlockSize;

			// 如果最下面和最右边的块不够，剩下多少读取多少
			if (i + nBlockSize > initYSize)
			{
				nYBk = initYSize - i;
			}

			if (j + nBlockSize > initXSize)
			{
				nXBK = initXSize - j;
			}

			// 读取原始影像块, pSrcData读取到的分块数据
			poSrcDS->RasterIO(GF_Read, j, i, nXBK, nYBk, pSrcData, nXBK, nYBk, GDT_Byte, initBands, pBandMaps, 0, 0, 0, NULL);

			/*....*/
			// 填写相应的影像处理算法
			// pSrcData 就是读取到的分块数据
			// pDstData 就是输出得到的数据

			// 将原始影像所有波段数据都复制到输出的影像中
			memcpy(pDstData, pSrcData, sizeof(GByte)*nXBK*nYBk*initBands);

			// 写到结果影像
			poDstDS->RasterIO(GF_Write, j, i, nXBK, nYBk, pSrcData, nXBK, nYBk, GDT_Byte, initBands, pBandMaps, 0, 0, 0, NULL);
		}
	}

	// 释放申请的内存
	delete[]pSrcData;
	//delete[]pDstData;
	delete[]pBandMaps;

	// 关闭原始影像和结果影像
	GDALClose((GDALDatasetH)poSrcDS);
	GDALClose((GDALDatasetH)poDstDS);

	return 0;
}

// int main() {
	// const char* pszSrcFile = "D:/GdalTest/zy3_yuenan/ZY301918220141215F.tif";
	// const char* pszDstFile = "D:/GdalTest/process.tif";
	// const char* pszFormat = "GTiff";
	// int nBlockSize = 8196;
	// int FLAG = BlockProcess(pszSrcFile, pszDstFile, pszFormat, nBlockSize);
// }