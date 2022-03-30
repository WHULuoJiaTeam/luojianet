#include "pyramid.h"


Pyramid::Pyramid() {

}


Pyramid::~Pyramid() {

}


void Pyramid::create_pyramid(cv::Mat &image, cv::Mat &label) {
	make_border(image, label);

	image_pyramid.push_back(image);
	label_pyramid.push_back(label);

	const float ratio = 0.5f;
	for (int level = 1; level < NLEVELS; level++) {
		if (image.cols <= 512) {
			break;
		}
		cv::resize(image, image, cv::Size(image.cols*ratio, image.rows*ratio));

		/* Mode = 'inter_nearest' when resize the label. */
		cv::resize(label, label, cv::Size(label.cols*ratio, label.rows*ratio), 0, 0, cv::INTER_NEAREST);
		image_pyramid.push_back(image);
		label_pyramid.push_back(label);
	}
}


void Pyramid::make_border(cv::Mat &image, cv::Mat &label) {
	int w = image.cols;
	int h = image.rows;
	int exponent = log(min(w, h)) / log(2);
	int size = pow(2.0, (float)exponent);

	while (size < w || size < h) {
		exponent += 1;
		size = pow(2.0, (float)exponent);
	}

	int pad_h = size - h;
	int pad_w = size - w;
	if (pad_h > 0 || pad_w > 0) {
		cv::Scalar zero_value = cv::Scalar(0, 0, 0);
		cv::Scalar ignore_value = cv::Scalar(255, 255, 255);
		copyMakeBorder(image, image, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, zero_value);
		copyMakeBorder(label, label, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, ignore_value);
	}
}