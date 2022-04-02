#ifndef GDAL2CV_H
#define GDAL2CV_H

#include <gdal_priv.h>

#include <opencv2/opencv.hpp>

using namespace std;


class GDAL2CV {

public:
	GDAL2CV();
	~GDAL2CV();

	cv::Mat gdal_read(string filename, int xStart, int yStart, int xWidth, int yWidth);
};


#endif