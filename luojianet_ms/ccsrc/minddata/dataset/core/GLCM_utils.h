#ifndef CALGLCM_H
#define CALGLCM_H

#include "opencv2/highgui/highgui.hpp"
#include <math.h>

using namespace cv;
using namespace std;

// Gray Level (Choose in 4/8/16)
enum GrayLevel
{
	GRAY_4,
	GRAY_8,
	GRAY_16
};

// Gray Value Statistical Direction
// (Choose in 0°, 45°, 90°, 135°)
enum GrayDirection
{
	DIR_0,
	DIR_45,
	DIR_90,
	DIR_135
};

// 彩色图中的指定通道
// Point out R, G, B Channel of a Image
enum RGBChannel
{
	CHANNEL_R,
	CHANNEL_G,
	CHANNEL_B
};

// struct including Texture Eigenvalues
struct TextureEValues
{
	float energy;
	float contrast;
	float homogenity;
	float entropy;
};

class CALGLCM
{
public:
	// Extract a channel from RGB Image
	void getOneChannel(Mat src, Mat& dstChannel, RGBChannel channel = CHANNEL_R);

	// Magnitude all pixels of Gray Image, and Magnitude Level can be chosen in 4/8/16;
	void GrayMagnitude(Mat src, Mat& dst, GrayLevel level = GRAY_8);

	// Calculate the GLCM of one Mat Window according to one Statistical Direction.
	void CalcuOneGLCM(Mat src, Mat &dst, int src_i, int src_j, int size, GrayLevel level = GRAY_8, GrayDirection direct = DIR_0);

	//   Normalize the Martix, make all pixels of Mat divided by the sum of all pixels of Mat, then get Probability Matrix.
	void NormalizeMat(Mat src, Mat& dst);

	// Calculate Texture Eigenvalues of One Window Mat, which is including Energy, Contrast, Homogenity, Entropy.
	void CalcuOneTextureEValue(Mat src, TextureEValues& EValue, bool ToCheckMat = false);

	// Calculate Texture Eigenvalues of One Window Mat, which is including Energy, Contrast, Homogenity, Entropy.
	void CalcuTextureEValue(Mat src, TextureEValues& EValue,
		int size = 5, GrayLevel level = GRAY_8);

	void CalcuTextureImages(Mat src, Mat& imgEnergy, Mat& imgContrast, Mat& imgHomogenity, Mat& imgEntropy,
		int size = 5, GrayLevel level = GRAY_8, bool ToAdjustImg = false);
};

#endif // CALGLCM_H