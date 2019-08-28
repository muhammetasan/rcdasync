/*
 *  CovarFeature.cpp
 *  regionCovariance
 *
 *  Created by Stephen McKeague
 *
 */


#include "CovarFeature.h"		//!!!: must be included before the buggy f2c.h

#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cassert>


//check if image is well formed
#define IMG_INIT(img) ( !img.empty() && (img.rows > 0) && (img.cols > 0) )

//check if integral images are well formed
#define INT_IMGS_INIT(intImgs) ( (intImgs.first.size() > 0) && (intImgs.second.size() == intImgs.first.size()) && (intImgs.second.front().size() == 1) && (intImgs.second.back().size() == intImgs.first.size()) && (intImgs.first.front().type() == CV_32FC1) )

//check if match is well formed
#define MATCH_INIT(match) ( (match.first.area() > 0) && (match.second >= 0) )

//check if covariance matrix is well formed
#define COVAR_INIT(covarMat) ( !covarMat.empty() && (covarMat.rows > 0) && (covarMat.rows == covarMat.cols) && (covarMat.type() == CV_32FC1) )

//check if stored scale arguments are valid
#define SCALE_PARAMS ( (this->lowerScale > 0) && (this->upperScale > 0) && (this->lowerScale <= this->upperScale) && (this->tolerance > 0) )


//check if ROI is within image
#define RECT_IN_MAT(ROI, img) ( (ROI.area() > 0) && (ROI.br().y -1 < img.rows) && (ROI.br().x - 1 < img.cols) )

//check if point lies within image
#define PT_IN_MAT(PT, img) ( (PT.x < img.cols) && (PT.y < img.rows) )

//check if window size can lie within image
#define SMALLER_SIZE(win, img) ( (win.width > 0) && (win.width <= img.width) && (win.height > 0) && (win.height <= img.height) )

//check if ROI can lie within window size
#define RECT_IN_SIZE(ROI, img) ( (ROI.area() > 0) && (ROI.br().y - 1< img.height) && (ROI.br().x - 1 < img.width) )

//check if two images have a size difference of 1 row and 1 column
#define SIZE_INC(img1, img2) ( (img1.rows == img2.rows + 1) && ( img1.cols == img2.cols + 1 ) )


//check if object has been previously stored
#define COVAR_OBJ_STORED ( (this->objSize.width > 0) && (this->objSize.height > 0) && (this->objCovarMatrices.size() > 0) )

//check if image has been previously stored
#define COVAR_IMG_STORED ( (this->imgSize.width > 0) && (this->imgSize.height > 0) && INT_IMGS_INIT(this->imgIntImgs) )

//check that the object size at maximum scale is not bigger than the image
#define MAX_SCALE_IMGS(img, obj) ( (img.rows >= (this->upperScale * obj.rows)) && (img.cols >= (this->upperScale * obj.cols)) )

//check that the object size at minimum scale is greater than 1
#define MIN_SCALE_OBJ(obj) ( ((this->lowerScale * obj.rows) > 1) && ((this->lowerScale * obj.cols) > 1) )

//check that stored object size at maximum scale is not bigger than the image
#define MAX_SCALE_STORED(img)  ( (img.rows >= (this->upperScale * this->objSize.height)) && (img.cols >= (this->upperScale * this->objSize.width)) )


#undef max 		//!!!: fixes bug when including f2c.h which means that std::numeric_limits::max() can't be used


using namespace cv;


CovarFeature::CovarFeature() : tolerance(0.15f), lowerScale(0.6f), upperScale(1.75f), preFilterSize(1000)
{
	//POST:
	if (!(SCALE_PARAMS && (this->preFilterSize > 0)))
		throw std::invalid_argument("Invalid CovarFeature Constructor");
}


CovarFeature::CovarFeature(const float tolerance, const float lowerScale, const float upperScale, const std::size_t preFilterSize) : tolerance(tolerance), lowerScale(lowerScale), upperScale(upperScale), preFilterSize(preFilterSize)
{
	//POST:
	if (!(SCALE_PARAMS && (this->preFilterSize > 0)))
		throw std::invalid_argument("Invalid CovarFeature Constructor");
}


std::vector<Mat> CovarFeature::generateFeatureImgs(const Mat& img)
{
	//PRE:
	assert(IMG_INIT(img) && (img.type() == CV_8UC3));

	std::vector<Mat> featureImgs;

	//!!!: Including X Y features increase processing time without much performance benefit 
	//!!!: They have thus been removed by default.  Uncomment the following sections to include them

	//set x to normalized col values
	Mat x(img.size(), CV_32FC1);
	const float xDiff = 1.0f / (img.cols - 1);
	float xTotal = 0;

	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			x.at<float>(i, j) = xTotal;
			xTotal += xDiff;
		}
		xTotal = 0;
	}
	featureImgs.push_back(x);


	//set y to normalized row values
	Mat y(img.size(), CV_32FC1);
	const float yDiff = 1.0f / (img.rows - 1);
	float yTotal = 0;

	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
			y.at<float>(i, j) = yTotal;
		yTotal += yDiff;
	}
	featureImgs.push_back(y);


	//split rgb components
	Mat b, g, r;
	std::vector<Mat> bgrImgs;
	split(img, bgrImgs);

	//add to feature image in reverse order i.e. R G B order while changing type to 32F
	for (int i = bgrImgs.size() - 1; i >= 0; --i)
	{
		Mat floatImg;
		bgrImgs.at(i).convertTo(floatImg, CV_32FC1);
		featureImgs.push_back(floatImg);
	}

	//calculate image gradients
	Mat greyImg;
	cvtColor(img, greyImg, CV_BGR2GRAY);

	Mat dx, dy, d2x, d2y;
	Sobel(greyImg, dx, CV_32FC1, 1, 0, 1);
	Sobel(greyImg, dy, CV_32F, 0, 1, 1);
	Sobel(greyImg, d2x, CV_32F, 2, 0, 1);
	Sobel(greyImg, d2y, CV_32F, 0, 2, 1);

	featureImgs.push_back(dx);
	featureImgs.push_back(dy);
	featureImgs.push_back(d2x);
	featureImgs.push_back(d2y);

	//POST:
	assert((featureImgs.size() > 0) && (featureImgs.at(0).type() == CV_32FC1));

	return featureImgs;
}


std::pair<std::vector<Mat>, std::vector<std::vector<Mat>>> CovarFeature::generateIntImgs(const std::vector<Mat>& featureImgs)
{
	//PRE:
	assert((featureImgs.size() > 0) && (featureImgs.at(0).type() == CV_32FC1));

	//generate integral images of features
	std::vector<Mat> regIntImgs;

	for (std::vector<Mat>::const_iterator it = featureImgs.begin(); it != featureImgs.end(); ++it)
	{
		Mat intImg;
		integral(*it, intImg, CV_64FC1);
		regIntImgs.push_back(intImg);
	}

	//generate integral images of each permutation of two multiplied features
	std::vector<std::vector<Mat>> sqIntImgs;

	for (std::vector<Mat>::const_iterator it = featureImgs.begin(); it != featureImgs.end(); ++it)
	{
		std::vector<Mat> sqIntRow;

		//since sqIntImgs is symmetric matrix, only process unique elements
		const std::size_t rowCount = it - featureImgs.begin();
		for (std::vector<Mat>::const_iterator jt = featureImgs.begin(); jt <= featureImgs.begin() + rowCount; ++jt)
		{
			const Mat sqImg = it->mul(*jt); //per element multiplication operation

			Mat sqIntImg;
			integral(sqImg, sqIntImg, CV_64FC1);
			sqIntRow.push_back(sqIntImg);
		}

		sqIntImgs.push_back(sqIntRow);
	}

	const std::pair<std::vector<Mat>, std::vector<std::vector<Mat>>> intImgs = std::make_pair(regIntImgs, sqIntImgs);

	//POST:
	assert((intImgs.first.size() == featureImgs.size()) && INT_IMGS_INIT(intImgs) && SIZE_INC(intImgs.first.front(), featureImgs.front()));

	return intImgs;
}


Mat CovarFeature::generateMainCovarMatrix(const std::pair<std::vector<Mat>, std::vector<std::vector<Mat>>>& intImgs, const Rect& ROI)
{
	//Integral images have a row and column of 0s on the x and y axis not accounted for in the ROIs of the original image
	const Rect adjustedROI(ROI.x, ROI.y, ROI.width, ROI.height);

    std::cout<<adjustedROI<<std::endl;
	//PRE:
	assert(INT_IMGS_INIT(intImgs));
	assert(RECT_IN_MAT(adjustedROI, intImgs.first.front()));

	Mat covarMatrix;
	////find the average covariance matrix over all the pixels feature vectors
	//if ((adjustedROI.x == 1) && (adjustedROI.y == 1))
	//	covarMatrix = calcCovarMatrix(intImgs, adjustedROI.br() - Point(1, 1));
	//else
	covarMatrix = calcCovarMatrix(intImgs, adjustedROI);

	//POST:
	assert(COVAR_INIT(covarMatrix));

	return covarMatrix;
}




Mat CovarFeature::calcCovarMatrix(const std::pair<std::vector<Mat>, std::vector<std::vector<Mat>>>& intImgs, const Rect& ROI)
{
	//PRE:
	assert(INT_IMGS_INIT(intImgs) && RECT_IN_MAT(ROI, intImgs.first.front()));

	Mat covarMatrix(intImgs.first.size(), intImgs.first.size(), CV_32FC1);

	const unsigned int nPixels = ROI.area();

	//for every unique element in the covar matrix
	for (int i = 0; i < covarMatrix.rows; ++i)
	{
		//calculate first P terms
		const Mat individualIntImg1 = intImgs.first.at(i);


		Point a, b, c, d;
		a = ROI.br() - Point(1, 1);
		b = ROI.tl();
		c.y = ROI.y;
		c.x = ROI.br().x - 1;
		d.y = ROI.br().y - 1;
		d.x = ROI.x;

		/*	if (i == 0 )
			{
			printf("TL TR BL BR (%d,%d)  (%d,%d)  (%d,%d)  (%d,%d)\n",b.x,b.y,c.x,c.y,d.x,d.y,a.x,a.y);
			}*/
		const float P1X__Y__ = individualIntImg1.at<float>(ROI.br() - Point(1, 1));
		const float P1X_Y_ = individualIntImg1.at<float>(ROI.tl());
		const float P1X__Y_ = individualIntImg1.at<float>(ROI.y, ROI.br().x - 1); //!!!: Matrix index is the opposite
		const float P1X_Y__ = individualIntImg1.at<float>(ROI.br().y - 1, ROI.x); //to the index of a graph
		const float sumIndividualFeatureVals1 = P1X__Y__ + P1X_Y_ - P1X__Y_ - P1X_Y__;

		//populate lower triangle
		for (int j = 0; j <= i; ++j)
		{
			//calculate Q terms
			const Mat combinedIntImg = intImgs.second.at(i).at(j);

			const float QX__Y__ = combinedIntImg.at<float>(ROI.br() - Point(1, 1));
			const float QX_Y_ = combinedIntImg.at<float>(ROI.tl());
			const float QX__Y_ = combinedIntImg.at<float>(ROI.y, ROI.br().x - 1);
			const float QX_Y__ = combinedIntImg.at<float>(ROI.br().y - 1, ROI.x);
			const float sumCombinedFeatureVals = QX__Y__ + QX_Y_ - QX__Y_ - QX_Y__;

			float summedTerms;
			//P1 = P2 on diagonal entries of covariance matrix
			//if (j==i)
			//		//Sum and average
			//	summedTerms = sumCombinedFeatureVals - sumIndividualFeatureVals1 * sumIndividualFeatureVals1 / nPixels;
			//else {			
			//		//calculate second P terms
			const Mat individualIntImg2 = intImgs.first.at(j);

			const float P2X__Y__ = individualIntImg2.at<float>(ROI.br() - Point(1, 1));
			const float P2X_Y_ = individualIntImg2.at<float>(ROI.tl());
			const float P2X__Y_ = individualIntImg2.at<float>(ROI.y, ROI.br().x - 1);
			const float P2X_Y__ = individualIntImg2.at<float>(ROI.br().y - 1, ROI.x);
			const float sumIndividualFeatureVals2 = P2X__Y__ + P2X_Y_ - P2X__Y_ - P2X_Y__;

			//Sum and average
			summedTerms = sumCombinedFeatureVals - sumIndividualFeatureVals1 * sumIndividualFeatureVals2 / nPixels;
			//}

			covarMatrix.at<float>(i, j) = summedTerms / (nPixels - 1);
		}
	}

	//duplicate symmetric elements in upper triangle
	for (int i = 0; i < covarMatrix.rows; ++i)
	for (int j = covarMatrix.cols - 1; j > i; --j)
		covarMatrix.at<float>(i, j) = covarMatrix.at<float>(j, i);

	//POST:
	assert(COVAR_INIT(covarMatrix));

	return covarMatrix;
}


bool CovarFeature::FLE(const float val1, const float val2)
{
	//if val1 is less than val2
	if (val1 <= val2)
		return true;

	//if val1 is equal to val2
	if (std::fabs(val1 - val2) <= (std::numeric_limits<float>::epsilon() * val2))
		return true;
	else
		return false;
}


//float CovarFeature::compareMatrices(Mat &mat1, const Mat &mat2) {
//        //PRE:
//    assert( COVAR_INIT(mat1) && COVAR_INIT(mat2) && (mat1.rows == mat2.rows) );

//    int info;
//    Mat eigenvals;

//    dsygv(info, eigenvals, mat1, mat2);

//    if (info != 0) {


//            //print non-positive definite object covariance matrix message
//        std::cerr << "Problem calculating covariance eigenvalues for a matrix.  Continuing execution with approximately 20% slower eigenvalue calculation, assuming that object covariance matrix is not full rank and is thus not positive-definite" << '\n' << "info = " << info << '\n' << std::endl;


//        dggev(info, eigenvals, mat1, mat2);

//        if (info != 0)
//            throw std::runtime_error ("Fatal problem calculating covariance eigenvalues");
//    }

//    log(eigenvals, eigenvals);
//    pow(eigenvals, 2.0, eigenvals);

//    const Scalar distSq = sum(eigenvals);

//    const float dist = std::sqrt(distSq[0]);

//        //POST:
//    assert(dist >= 0);

//    return dist;
//}


//void CovarFeature::dsygv(int &info, cv::Mat &w, cv::Mat &mat1, const cv::Mat &mat2) {
//		//PRE:
//	assert( COVAR_INIT(mat1) && COVAR_INIT(mat2) && (mat1.rows == mat2.rows) );

//		//lapack functions can't be consted
//		//function arguments
//	static integer itype = 1;		//problem is Ax = (lambda)Bx
//	static char jobz = 'N';			//compute eivenvalues only
//	static char uplo = 'U';			//Upper/Lower triangles of A and B are stored, makes no difference to result

//		//intput eigenvalue matrices
//	integer n = mat1.cols;			//assume mat1 and mat2 are square matrices of same size
//	integer lda = mat1.rows;		//leading dimension of array A
//	Mat b = mat2.clone();			//input array that needs to be overwritten
//	integer ldb = b.rows;			//leading dimension of array B

//		//output eigenvalue matrix
//	w.create(n, 1, CV_64FC1);		//Output matrix can be row or column vector.
//									//Contains eigenvalues in ascending order
//									//working variables
//	Mat work(42*n, 1, CV_64FC1);	//working array
//	integer lwork = 42*n;			//length of the work array
//	integer infoT = 0;						//result of operation

//	dsygv_(&itype, &jobz, &uplo, &n, (float*)mat1.data, &lda, (float*)b.data, &ldb, (float*)w.data, (float*)work.data, &lwork, &infoT);

//	info = static_cast<int>(infoT);

//		//POST:
//	assert( !w.empty() && (w.rows == mat1.rows) && (w.cols == 1) && (w.type() == CV_64FC1) );

//}


//void CovarFeature::dggev(int &info, Mat &w, Mat &mat1, const Mat &mat2) {
//		//PRE:
//	assert( COVAR_INIT(mat1) && COVAR_INIT(mat2) && (mat1.rows == mat2.rows) );

//		//function arguments
//	static char jobvl = 'N';		//do not compute the left generalised eigenvectors
//	static char jobvr = 'N';		//do not compute the right generalised eigenvectors

//		//intput eigenvalue matrices
//	integer n = mat1.cols;			//order of the matrices
//	integer lda = mat1.rows;		//leading dimension of a
//	Mat b = mat2.clone();
//	integer ldb = mat2.rows;		//leading dimension of b

//		//output eigenvalue matrices
//	Mat alphar(n, 1, CV_64FC1);
//	Mat alphai(n, 1, CV_64FC1);		//seems to be empty, but not used anyway since we want the right eigenvalues
//	Mat beta(n, 1, CV_64FC1);

//		//unused engenvector variables
//    static float vl = 0;
//	static integer ldvl = 1;
//    static float vr = 0;
//	static integer ldvr = 1;

//		//working variables
//	Mat work(47*n, 1, CV_64FC1);	//working array
//	integer lwork = 47*n;			//length of the work array
//	integer infoT = 0;				//result of operation

//	dggev_(&jobvl, &jobvr, &n, (float*)mat1.data, &lda, (float*)b.data, &ldb, (float*)alphar.data, (float*)alphai.data, (float*)beta.data, &vl, &ldvl, &vr, &ldvr, (float*)work.data, &lwork, &infoT);

//		//actual eigenvalues are represented as a ratio: alpha / beta
//	w.create(n, 1, CV_64FC1);

//	for (int i=0; i<w.rows; ++i)
//		for (int j=0; j<w.cols; ++j) {
//			if (beta.at<float>(i, j) == 0)
//				w.at<float>(i, j) = 0.0;
//			else
//				w.at<float>(i, j) = alphar.at<float>(i, j) / beta.at<float>(i, j);
//		}

//	info = static_cast<int>(infoT);

//		//POST:
//	assert( !w.empty() && (w.rows == mat1.rows) && (w.cols == 1) && (w.type() == CV_64FC1) );

//}
