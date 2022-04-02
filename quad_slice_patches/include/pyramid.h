/*
* Created by: ZhangZhan
* Wuhan University
* zhangzhanstep@whu.edu.cn
* Copyright (c) 2021
*/

#ifndef PYRAMID_H
#define PYRAMID_H

#include <opencv2/opencv.hpp>

using namespace std;

// Pyramid settings
#define NLEVELS 6


class Pyramid {
public:
	vector<cv::Mat> image_pyramid;
	vector<cv::Mat> label_pyramid;
public:
	Pyramid();
	~Pyramid();

	void create_pyramid(cv::Mat &image, cv::Mat &label);
	void make_border(cv::Mat &image, cv::Mat &label);
};


#endif