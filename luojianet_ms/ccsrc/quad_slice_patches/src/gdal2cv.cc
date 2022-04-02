#include "gdal2cv.h"


GDAL2CV::GDAL2CV() {

}


GDAL2CV::~GDAL2CV() {
	
}


double range_cast(const GDALDataType& gdalType, const int& cvDepth, const double& value) {
	// uint8 -> uint8
	if (gdalType == GDT_Byte && cvDepth == CV_8U) {
		return value;
	}
	// uint8 -> uint16
	if (gdalType == GDT_Byte && (cvDepth == CV_16U || cvDepth == CV_16S)) {
		return (value * 256);
	}

	// uint8 -> uint32
	if (gdalType == GDT_Byte && (cvDepth == CV_32F || cvDepth == CV_32S)) {
		return (value * 16777216);
	}

	// int16 -> uint8
	if ((gdalType == GDT_UInt16 || gdalType == GDT_Int16) && cvDepth == CV_8U) {
		return std::floor(value / 256.0);
	}

	// int16 -> int16
	if ((gdalType == GDT_UInt16 || gdalType == GDT_Int16) &&
		(cvDepth == CV_16U || cvDepth == CV_16S)) {
		return value;
	}

	// float32 -> float32
	// float64 -> float64
	if ((gdalType == GDT_Float32 || gdalType == GDT_Float64) &&
		(cvDepth == CV_32F || cvDepth == CV_64F)) {
		return value;
	}

	std::cout << GDALGetDataTypeName(gdalType) << std::endl;
	std::cout << "warning: unknown range cast requested." << std::endl;
	return (value);
}


void write_pixel(const double& pixelValue, const GDALDataType& gdalType, const int& gdalChannels,
				 cv::Mat& image, const int& row, const int& col, const int& channel) {
	// convert the pixel
	double newValue = range_cast(gdalType, image.depth(), pixelValue);

	// input: 1 channel, output: 1 channel
	if (gdalChannels == 1 && image.channels() == 1) {
		if (image.depth() == CV_8U) { image.ptr<uchar>(row)[col] = static_cast<uchar>(newValue); }
		else if (image.depth() == CV_16U) { image.ptr<unsigned short>(row)[col] = static_cast<unsigned short>(newValue); }
		else if (image.depth() == CV_16S) { image.ptr<short>(row)[col] = static_cast<short>(newValue); }
		else if (image.depth() == CV_32S) { image.ptr<int>(row)[col] = static_cast<int>(newValue); }
		else if (image.depth() == CV_32F) { image.ptr<float>(row)[col] = static_cast<float>(newValue); }
		else if (image.depth() == CV_64F) { image.ptr<double>(row)[col] = newValue; }
		else { throw std::runtime_error("Unknown image depth, gdal: 1, img: 1"); }
	}

	// input: 1 channel, output: 3 channel
	else if (gdalChannels == 1 && image.channels() == 3) {
		if (image.depth() == CV_8U) { image.ptr<cv::Vec3b>(row)[col] = cv::Vec3b(newValue, newValue, newValue); }
		else if (image.depth() == CV_16U) { image.ptr<cv::Vec3w>(row)[col] = cv::Vec3w(newValue, newValue, newValue); }
		else if (image.depth() == CV_16S) { image.ptr<cv::Vec3s>(row)[col] = cv::Vec3s(newValue, newValue, newValue); }
		else if (image.depth() == CV_32S) { image.ptr<cv::Vec3i>(row)[col] = cv::Vec3i(newValue, newValue, newValue); }
		else if (image.depth() == CV_32F) { image.ptr<cv::Vec3f>(row)[col] = cv::Vec3f(newValue, newValue, newValue); }
		else if (image.depth() == CV_64F) { image.ptr<cv::Vec3d>(row)[col] = cv::Vec3d(newValue, newValue, newValue); }
		else { throw std::runtime_error("Unknown image depth, gdal:1, img: 3"); }
	}

	// input: 3 channel, output: 1 channel
	else if (gdalChannels == 3 && image.channels() == 1) {
		if (image.depth() == CV_8U) { image.ptr<uchar>(row)[col] += (newValue / 3.0); }
		else { throw std::runtime_error("Unknown image depth, gdal:3, img: 1"); }
	}

	// input: 4 channel, output: 1 channel
	else if (gdalChannels == 4 && image.channels() == 1) {
		if (image.depth() == CV_8U) { image.ptr<uchar>(row)[col] = newValue; }
		else { throw std::runtime_error("Unknown image depth, gdal: 4, image: 1"); }
	}

	// input: 3 channel, output: 3 channel
	else if (gdalChannels == 3 && image.channels() == 3) {
		if (image.depth() == CV_8U) { image.at<cv::Vec3b>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_16U) { image.at<cv::Vec3w>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_16S) { image.at<cv::Vec3s>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_32S) { image.at<cv::Vec3i>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_32F) { image.at<cv::Vec3f>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_64F) { image.at<cv::Vec3d>(row, col)[channel] = newValue; }
		else { throw std::runtime_error("Unknown image depth, gdal: 3, image: 3"); }
	}

	// input: 4 channel, output: 3 channel
	else if (gdalChannels == 4 && image.channels() == 3) {
		if (channel >= 4) { return; }
		else if (image.depth() == CV_8U && channel < 4) { image.at<cv::Vec3b>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_16U && channel < 4) { image.at<cv::Vec3w>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_16S && channel < 4) { image.at<cv::Vec3s>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_32S && channel < 4) { image.at<cv::Vec3i>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_32F && channel < 4) { image.at<cv::Vec3f>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_64F && channel < 4) { image.at<cv::Vec3d>(row, col)[channel] = newValue; }
		else { throw std::runtime_error("Unknown image depth, gdal: 4, image: 3"); }
	}

	// input: 4 channel, output: 4 channel
	else if (gdalChannels == 4 && image.channels() == 4) {
		if (image.depth() == CV_8U) { image.at<cv::Vec4b>(row, col)[channel] = newValue; }
		//if (image.depth() == CV_8U){ image.ptr<cv::Vec4b>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_16U) { image.at<cv::Vec4w>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_16S) { image.at<cv::Vec4s>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_32S) { image.at<cv::Vec4i>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_32F) { image.at<cv::Vec4f>(row, col)[channel] = newValue; }
		else if (image.depth() == CV_64F) { image.at<cv::Vec4d>(row, col)[channel] = newValue; }
		else { throw std::runtime_error("Unknown image depth, gdal: 4, image: 4"); }
	}

	// input: > 4 channels, output: > 4 channels
	else if (gdalChannels > 4 && image.channels() > 4) {
		if (image.depth() == CV_8U) {
			uchar * data = image.ptr<uchar>(row);
			data[col*image.channels() + channel] = newValue;
			//image.ptr<uchar>(row, col)[channel] = newValue;
		}
		else if (image.depth() == CV_16U) {
			ushort * data = image.ptr<ushort>(row);
			data[col*image.channels() + channel] = newValue;
			//image.ptr<unsigned short>(row, col)[channel] = newValue;
		}
		else if (image.depth() == CV_16S) {
			short * data = image.ptr<short>(row);
			data[col*image.channels() + channel] = newValue;
			//image.ptr<short>(row, col)[channel] = newValue;
		}
		else if (image.depth() == CV_32S) {
			int * data = image.ptr<int>(row);
			data[col*image.channels() + channel] = newValue;
			//image.ptr<int>(row, col)[channel] = newValue;
		}
		else if (image.depth() == CV_32F) {
			float * data = image.ptr<float>(row);
			data[col*image.channels() + channel] = newValue;
			//image.ptr<float>(row, col)[channel] = newValue;
		}
		else if (image.depth() == CV_64F) {
			double * data = image.ptr<double>(row);
			data[col*image.channels() + channel] = newValue;
			//image.ptr<double>(row, col)[channel] = newValue;
		}
		else { throw std::runtime_error("Unknown image depth, gdal: N, img: N"); }
	}
	// otherwise, throw an error
	else {
		throw std::runtime_error("error: can't convert types.");
	}
}


void write_ctable_pixel(const double& pixelValue, const GDALDataType& gdalType, GDALColorTable const* gdalColorTable,
						cv::Mat& image, const int& y, const int& x, const int& c) {
	if (gdalColorTable == NULL) {
		write_pixel(pixelValue, gdalType, 1, image, y, x, c);
	}

	// if we are Grayscale, then do a straight conversion
	if (gdalColorTable->GetPaletteInterpretation() == GPI_Gray) {
		write_pixel(pixelValue, gdalType, 1, image, y, x, c);
	}

	// if we are rgb, then convert here
	else if (gdalColorTable->GetPaletteInterpretation() == GPI_RGB) {

		// get the pixel
		short r = gdalColorTable->GetColorEntry((int)pixelValue)->c1;
		short g = gdalColorTable->GetColorEntry((int)pixelValue)->c2;
		short b = gdalColorTable->GetColorEntry((int)pixelValue)->c3;
		short a = gdalColorTable->GetColorEntry((int)pixelValue)->c4;

		write_pixel(r, gdalType, 4, image, y, x, 2);
		write_pixel(g, gdalType, 4, image, y, x, 1);
		write_pixel(b, gdalType, 4, image, y, x, 0);
		if (image.channels() > 3) {
			write_pixel(a, gdalType, 4, image, y, x, 1);
		}
	}

	// otherwise, set zeros
	else {
		write_pixel(pixelValue, gdalType, 1, image, y, x, c);
	}
}


int gdal2opencv(const GDALDataType& gdalType, const int& channels) {

	switch (gdalType) {

		/// UInt8
	case GDT_Byte:
		if (channels == 1) { return CV_8UC1; }
		if (channels == 3) { return CV_8UC3; }
		if (channels == 4) { return CV_8UC4; }
		else { return CV_8UC(channels); }
		return -1;

		/// UInt16
	case GDT_UInt16:
		if (channels == 1) { return CV_16UC1; }
		if (channels == 3) { return CV_16UC3; }
		if (channels == 4) { return CV_16UC4; }
		else { return CV_16UC(channels); }
		return -1;

		/// Int16
	case GDT_Int16:
		if (channels == 1) { return CV_16SC1; }
		if (channels == 3) { return CV_16SC3; }
		if (channels == 4) { return CV_16SC4; }
		else { return CV_16SC(channels); }
		return -1;

		/// UInt32
	case GDT_UInt32:
	case GDT_Int32:
		if (channels == 1) { return CV_32SC1; }
		if (channels == 3) { return CV_32SC3; }
		if (channels == 4) { return CV_32SC4; }
		else { return CV_32SC(channels); }
		return -1;

	case GDT_Float32:
		if (channels == 1) { return CV_32FC1; }
		if (channels == 3) { return CV_32FC3; }
		if (channels == 4) { return CV_32FC4; }
		else { return CV_32FC(channels); }
		return -1;

	case GDT_Float64:
		if (channels == 1) { return CV_64FC1; }
		if (channels == 3) { return CV_64FC3; }
		if (channels == 4) { return CV_64FC4; }
		else { return CV_64FC(channels); }
		return -1;

	default:
		std::cout << "Unknown GDAL Data Type" << std::endl;
		std::cout << "Type: " << GDALGetDataTypeName(gdalType) << std::endl;
		return -1;
	}

	return -1;
}


int gdalPaletteInterpretation2OpenCV(GDALPaletteInterp const& paletteInterp, GDALDataType const& gdalType) {
	
	switch (paletteInterp) {

	/// GRAYSCALE
	case GPI_Gray:
		if (gdalType == GDT_Byte) { return CV_8UC1; }
		if (gdalType == GDT_UInt16) { return CV_16UC1; }
		if (gdalType == GDT_Int16) { return CV_16SC1; }
		if (gdalType == GDT_UInt32) { return CV_32SC1; }
		if (gdalType == GDT_Int32) { return CV_32SC1; }
		if (gdalType == GDT_Float32) { return CV_32FC1; }
		if (gdalType == GDT_Float64) { return CV_64FC1; }
		return -1;

	/// RGB
	case GPI_RGB:
		if (gdalType == GDT_Byte) { return CV_8UC1; }
		if (gdalType == GDT_UInt16) { return CV_16UC3; }
		if (gdalType == GDT_Int16) { return CV_16SC3; }
		if (gdalType == GDT_UInt32) { return CV_32SC3; }
		if (gdalType == GDT_Int32) { return CV_32SC3; }
		if (gdalType == GDT_Float32) { return CV_32FC3; }
		if (gdalType == GDT_Float64) { return CV_64FC3; }
		return -1;

	/// otherwise
	default:
		return -1;
	}
}


cv::Mat GDAL2CV::gdal_read(string filename, int xStart, int yStart, int xWidth, int yWidth) {
	const char* filepath = filename.data();
	GDALAllRegister();
	GDALDataset *poSrc = (GDALDataset*)GDALOpen(filepath, GA_ReadOnly);
	if (poSrc == NULL) {
		cout << "GDAL failed to open " << filepath;
	}

	int m_width = poSrc->GetRasterXSize();
	int m_height = poSrc->GetRasterYSize();
	GDALDataType gdalType = poSrc->GetRasterBand(1)->GetRasterDataType();

	bool hasColorTable;
	int dataType;
	if (poSrc->GetRasterBand(1)->GetColorInterpretation() == GCI_PaletteIndex) {
		hasColorTable = true;
		if (poSrc->GetRasterBand(1)->GetColorTable() == NULL) {
			return cv::Mat();
		}

		dataType = gdalPaletteInterpretation2OpenCV(poSrc->GetRasterBand(1)->GetColorTable()->GetPaletteInterpretation(),
													poSrc->GetRasterBand(1)->GetRasterDataType());
		if (dataType == -1) {
			return cv::Mat();
		}
	}
	else {
		hasColorTable = false;
		dataType = gdal2opencv(poSrc->GetRasterBand(1)->GetRasterDataType(), poSrc->GetRasterCount());
		if (dataType == -1) {
			return cv::Mat();
		}
	}

	if (xStart < 0 || yStart < 0 || xWidth < 1 || yWidth < 1 || xStart > m_width - 1 || yStart > m_height - 1) {
		return cv::Mat();
	}
	if (xStart + xWidth > m_width) {
		std::cout << "The specified width is invalid, Automatic optimization is executed!" << std::endl;
		xWidth = m_width - xStart;
	}
	if (yStart + yWidth > m_height) {
		std::cout << "The specified height is invalid, Automatic optimization is executed!" << std::endl;
		yWidth = m_height - yStart;
	}

	cv::Mat img(yWidth, xWidth, dataType, cv::Scalar::all(0.f));
	int nChannels = poSrc->GetRasterCount();

	GDALColorTable* gdalColorTable = NULL;
	if (poSrc->GetRasterBand(1)->GetColorTable() != NULL) {
		gdalColorTable = poSrc->GetRasterBand(1)->GetColorTable();
	}

	for (int c = 0; c < img.channels(); c++) {

		int realBandIndex = c;

		// get the GDAL Band
		GDALRasterBand* band = poSrc->GetRasterBand(c + 1);

		if (GCI_RedBand == band->GetColorInterpretation()) {
			realBandIndex = 2;
		}
		if (GCI_GreenBand == band->GetColorInterpretation()) {
			realBandIndex = 1;
		}
		if (GCI_BlueBand == band->GetColorInterpretation()) {
			realBandIndex = 0;
		}

		if (hasColorTable && gdalColorTable->GetPaletteInterpretation() == GPI_RGB) {
			c = img.channels() - 1;
		}
		// make sure the image band has the same dimensions as the image
		if (band->GetXSize() != m_width || band->GetYSize() != m_height) { 
			return cv::Mat();
		}

		// create a temporary scanline pointer to store data
		double* scanline = new double[xWidth];

		// iterate over each row and column
		for (int y = 0; y < yWidth; y++) {

			// get the entire row
			band->RasterIO(GF_Read, xStart, y + yStart, xWidth, 1, scanline, xWidth, 1, GDT_Float64, 0, 0);
			// set inside the image
			for (int x = 0; x < xWidth; x++) {
				if (hasColorTable == false) {
					write_pixel(scanline[x], gdalType, nChannels, img, y, x, realBandIndex);
				}
				else {
					write_ctable_pixel(scanline[x], gdalType, gdalColorTable, img, y, x, c);
				}
			}
		}
		delete[] scanline;
	}
	GDALClose((GDALDatasetH)poSrc);
	return img;
}