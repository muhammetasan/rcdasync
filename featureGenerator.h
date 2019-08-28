#ifndef FEATUREGENERATOR_H
#define FEATUREGENERATOR_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

#define D_X 0
#define D_Y 1
#define R_CHANNEL 2
#define G_CHANNEL 3
#define B_CHANNEL 4
#define D_XX 5
#define D_YY 6

#define FEATURE_COUNT 5
#define SCALE_COUNT 8
#define ELEM_PER_BLOCK 1024

__device__ const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__device__ const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

__device__ const int sobel_xx[5][5] = {
	{1, 0, -2, 0, 1},
	{4, 0, -8, 0, 4},
	{6, 0, -12, 0, 6},
	{4, 0, -8, 0, 4},
	{1, 0, -2, 0, 1},
};

__device__ const int sobel_yy[5][5] =
	{
		{1, 4, 6, 4, 1},
		{0, 0, 0, 0, 0},
		{-2, -8, -12, -8, -2},
		{0, 0, 0, 0, 0},
		{1, 4, 6, 4, 1}};

#if FEATURE_COUNT == 5
__device__ const int xIndex[15] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
__device__ const int yIndex[15] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4};
#elif FEATURE_COUNT == 7
__device__ const int xIndex[28] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6};
__device__ const int yIndex[28] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6};
#endif



class featureGenerator
{
  public:
	featureGenerator();
	~featureGenerator(); // device reset, cudafree �a��r�lmal�
	void generateFeatureImages();
	void generateSquareFeatures();
	void calculateIntegralImages();
	void calculateObjectIntegral();
	void calculateCovarianceMatrix();
	void findMinimumOnGPU(int & indexFound);
	void setInputImage(const cv::Mat &image); // real time da setFrame Yapabiliriz. cols/rows de�i�meyece�i i�in
	void setWinSize(const unsigned int &winSize, const float &tolerance);
	void updateImage(const Mat &image);
	void resetDetector();
	static void syncAnderrorCheck();

	std::pair<std::vector<Mat>, std::vector<std::vector<Mat>>> getIntegralImages();

	void copyObjectDescriptor2Device(float *objectDescriptor, float objectLogDeterminant);
	void copyDataToDevice();
	void allocateCovarianceMatrixSpace(vector<KeyPoint> &vec);

  public:
	uchar *inputImage_UCP_d;
	uchar *greyImage_UCP_d;
	int imageRows;
	int imageCols;
	int pixelNumber;
	int byteCount_float;
	Mat greyImage_Mat;
	Mat image_Mat;
	int totalKeypoints;
	int blockNumber;

	float **features_fp;
	float **sqFEatures_fp;
	float **_features_fp_d;
	float *_covMatDistances_d;
	float *__covMat;
	float *__objDescriptor_d;
	float *__objLogDeterminant_d;
	int * __resultsIndex_d;
	float * _resultsVals_d;
	float * _resultsVals_h;
	int * __resultsIndex_h;
	unsigned int winSize;
	float tolerance;

	std::vector<Mat> features;
	std::vector<std::vector<Mat>> sqFeatures;
	void allocateDevice();
	vector<double> __vScaleList;
	vector<double> __vCurWinSize;
	vector<int> __vWinStep;
	vector<int> __vKpNumberX;
	vector<int> __vKpNumberY;
	cudaStream_t  computeStream;
	cudaStream_t  copyStream;
	cudaEvent_t waitevent;
};

#endif
