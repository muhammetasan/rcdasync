#include "featureGenerator.h"
#include <opencv2/opencv.hpp>
#include<vector>
#include"CovarFeature.h"
#include<time.h>

using namespace cv;
int main(int argc,char * argv[])
{
	featureGenerator featGen;
	Mat img;
	Mat objImage = imread("Untitled.png");
	const int winSize = objImage.rows < objImage.cols ? objImage.rows : objImage.cols;

	cudaEventCreate(&featGen.waitevent);
	cudaStreamCreate(&featGen.computeStream);
	cudaStreamCreate(&featGen.copyStream);

	auto copyStream = featGen.copyStream;
	auto computeStream = featGen.computeStream;

	featGen.copyStream = 0;
	featGen.computeStream = 0;

	const float tolerance = 0.1;
	// object part
	cv::Mat objDescriptor;
	featGen.setInputImage(objImage);
	featGen.copyDataToDevice();
	featGen.generateFeatureImages();
    featGen.generateSquareFeatures();
	featGen.calculateObjectIntegral();
	static const cv::Point origin(0, 0);
	const cv::KeyPoint objKeyPoint(origin, winSize);
	std::vector<cv::KeyPoint> objKeyPoints;
	objKeyPoints.push_back(objKeyPoint);

	objDescriptor.create(objKeyPoints.size(), FEATURE_COUNT*FEATURE_COUNT, CV_32FC1);
	const std::pair<std::vector<Mat>, std::vector< std::vector<Mat> > > ObjintImgs = featGen.getIntegralImages();


	for (std::size_t i = 0; i < objKeyPoints.size(); ++i) {
		const KeyPoint curKeyPoint = objKeyPoints.at(i);


		//calculate covariance matrix and store as a 1d vector
		const Rect ROI(curKeyPoint.pt.x, curKeyPoint.pt.y, static_cast<int>(curKeyPoint.size), static_cast<int>(curKeyPoint.size));

		Mat covarMat = CovarFeature::generateMainCovarMatrix(ObjintImgs, ROI);

		covarMat = covarMat.reshape(0, 1);

		//!!!: Need to add 0 in current opencv implementation
		objDescriptor.row(i) = covarMat.row(0) + Mat::zeros(1, FEATURE_COUNT*FEATURE_COUNT, CV_32FC1);
	}

	// end object part

	featGen.computeStream = computeStream;
	featGen.copyStream = copyStream;

	featGen.setWinSize(winSize, tolerance);
    VideoCapture capture("a.mp4");
	int videoWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int videoHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	
	
	std::vector<uchar*> frames;

	while (true)
	{
		uchar * memAdres;
		cudaMallocHost(&memAdres, videoHeight*videoWidth * 3);
		frames.push_back(memAdres);

		capture >> img;
		if (img.empty()) break;
		memcpy(memAdres, img.data, videoHeight*videoWidth * 3);
	}

	

	img = Mat(Size(videoWidth, videoHeight), CV_8UC3, frames[0], Mat::AUTO_STEP);
	featGen.setInputImage(img);
	featGen.copyObjectDescriptor2Device((float*)objDescriptor.data, log(cv::determinant(objDescriptor.reshape(1, FEATURE_COUNT))));

	vector<KeyPoint> cudaKeypts;
	featGen.allocateCovarianceMatrixSpace(cudaKeypts);
	featGen.syncAnderrorCheck();
	int counter = 0;
	featGen.updateImage(img);
	featGen.syncAnderrorCheck();
	int frameCounter = 1;

	

	while (true)
	{
	
		static int counter = 0;
		
		// featGen.syncAnderrorCheck();
		clock_t start = clock();
		featGen.generateFeatureImages();
		// featGen.syncAnderrorCheck();
		// std::cout<<(1000*(clock()-start))/CLOCKS_PER_SEC<<endl; // 1ms

		//img = frames[counter++];
		if (counter == frames.size() - 1)
			break;

		//cudaStreamWaitEvent(featGen.computeStream, featGen.waitevent);
		cudaMemcpyAsync(featGen.inputImage_UCP_d, frames[frameCounter++], videoHeight*videoWidth * 3, cudaMemcpyDeviceToHost, featGen.copyStream);

		// start async copy. 

		// start = clock();
	 	featGen.generateSquareFeatures();
		// featGen.syncAnderrorCheck();
		// std::cout<<(1000*(clock()-start))/CLOCKS_PER_SEC<<endl; // 5ms

		// start = clock();
		featGen.calculateIntegralImages();
		// featGen.syncAnderrorCheck();
		// std::cout<<"calculateIntegralImages "<<(1000*(clock()-start))/CLOCKS_PER_SEC<<endl; // 15ms

		// start=clock();
		featGen.calculateCovarianceMatrix();
		// featGen.syncAnderrorCheck();
		// std::cout<<(1000*(clock()-start))/CLOCKS_PER_SEC<<endl; // 15ms
		int matchIndexBF = 0;
		// featGen.syncAnderrorCheck();
		// start = clock();
		
		featGen.findMinimumOnGPU(matchIndexBF);
		 featGen.syncAnderrorCheck();
		 std::cout << (double)(clock() - start) / CLOCKS_PER_SEC<<std::endl;
		Rect2d r(cudaKeypts[matchIndexBF].pt, Size(cudaKeypts[matchIndexBF].size, cudaKeypts[matchIndexBF].size));
		cv::Mat outputImg=Mat(Size(videoWidth,videoHeight),CV_8UC3,frames[frameCounter-1],Mat::AUTO_STEP);
		rectangle(outputImg, r, Scalar(255, 0, 0), 10);
		resize(outputImg, outputImg, cv::Size(outputImg.cols / 2, outputImg.rows / 2));
		// static int xctrrr = 0;
		// xctrrr++;
		// cout<<"*******"<<endl;
		// if( xctrrr==5)
		// return 0;
		imshow("Output", outputImg);
		waitKey(1);
		counter++;
		if (counter == 5);

	}


	/*for (int x = 0; x < frames.size(); ++x)
	{
		delete[] frames[x];
	}*/
}
