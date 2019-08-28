/*
 *  CovarFeature.h
 *  regionCovariance
 *
 *  Created by Stephen McKeague
 *
 */


#ifndef CovarFeature_H
#define CovarFeature_H


#include <vector>
#include <utility>
#include <functional>
#include <cstddef>

#include <opencv2/opencv.hpp>


/**
 * Covariance Feature Class as proposed in Tuzel et. al.
 * Matches an object in an image by comparing covariance matrices generated from a number of defined image features.
 * Integral images are generated for all object and image feature images to speed up covariance matrix generation.
 * The object covariance matrix is compared with the covariance matrix of a scaled sliding region within the image.
 * The best matches from this process are retained and further processed with 4 extra covariance matrix comparisons.
 * The best match after filtering is returned as the result of the obejct detection process.
 *
 * @author Stephen McKeague
 */
class CovarFeature
{
public:

	/** 
	 * Default Constructor.
	 * Searches 9 different scales, 4 smaller, 4 larger, with a 15% scaling factor between two consecutive scales.
	 * Defines 1000 pre-filter matches.
	 *
	 * @post stored lowerScale > 0.
	 *		 stored upperScale > 0.
	 *		 stored lowerScale <= stored upperScale.
	 *		 stored tolerance > 0.
	 *		 stored preFilterSize > 0
	 */
	CovarFeature();

	/** 
	 * Constructor for initialising all Covariance parameters.
	 *
	 * @param tolerance classifier tolerance specified as a fraction
	 * @param lowerScale lower fraction of object to scale from
	 * @param upperScale upper fraction of object to scale to
	 * @param preFilterSize the number of matches to retain for further filtering after initial image scanning
	 *
	 * @post stored lowerScale > 0.
	 *		 stored upperScale > 0.
	 *		 stored lowerScale <= stored upperScale.
	 *		 stored tolerance > 0.
	 *		 stored preFilterSize > 0
	 */
	CovarFeature(const float tolerance, const float lowerScale, const float upperScale, const std::size_t preFilterSize);


public:

	/** Classifier tolerance given as a fraction. Used during scaling. */
	const float tolerance;

	/** Lower fraction of object to scale from. */
	const float lowerScale;

	/** Upper fraction of object to scale to. */
	const float upperScale;

	/** The number of results to retain for further filtering.
	 *	Only the first covariance matrix is compared to generate these results. */
	const std::size_t preFilterSize;

	/** Image size of original object. */
	cv::Size objSize;

	/** Object covariance matrices generated from extracted image features.
	 *	Target bounds of covariance matrices detailed in Tuzel et. al. */
	std::vector<cv::Mat> objCovarMatrices;

	/** Size of image in which to find object. */
	cv::Size imgSize;

	/** A pair of 1d and 2d vectors containing the integral feature images of the entire target image.
	 *	The first vector contains the integral images of each feature image.
	 *	The second 2d matrix contains the integral of each permutation of feature images multiplied together. */
	std::pair<std::vector<cv::Mat>, std::vector<std::vector<cv::Mat>>> imgIntImgs;


	/**
	 * Generate a vector of feature images from a variety of hard coded features.
	 *
	 * @param img BGR image from which to generate the feature images
	 * @return vector of feature images
	 *
	 * @pre img argument is well formed and has 3 channels whose elements are of type uchar
	 * @post returned feature image vector must be non-empty.
	 *		 feature image elements must have 1 channel whose elements are of type float
	 */
	static std::vector<cv::Mat> generateFeatureImgs(const cv::Mat& img);

	/**
	 * Generate a pair of hard coded integral images from a vector of supplied feature images.  
	 *
	 * @param featureImgs vector of feature images from target image
	 * @return A pair of 1d and 2d vectors containing the integral feature images of the entire object or image.  
	 *		   The first vector contains the integral images of each feature image.  
	 *		   The second 2d matrix contains the integral of each permutation of multiplied feature images.
	 *		   As this is a symmetric matrix, the upper triangle in not computed for performance reasons
	 *
	 * @post featureImgs argument must be non-empty.
	 *		 featureImgs elements must have 1 channel whose elements are of type float
	 * @post The first returned vector of integral images is the same length as the input feature images (n).
	 *		 The second returned matrix must have the size of the lower triangle of an n*n matrix.  
	 *		 Every integral image must have 1 channel with elements of type double.  
	 *		 Every integral image must have an added row and column of 0s added at the 0th index.
	 */
	static std::pair<std::vector<cv::Mat>, std::vector<std::vector<cv::Mat>>> generateIntImgs(const std::vector<cv::Mat>& featureImgs);


	/**
	 * Generate a covariance matrix of image features using integral images over an complete ROI.
	 *
	 * @param intImgs a pair of 1d and 2d vectors containing the integral feature images of the entire object or image.  
	 *				  The first vector contains the integral images of each feature image.  
	 *				  The second 2d matrix contains the integral of each permutation of multiplied feature images.
	 *				  As this is a symmetric matrix, the upper triangle in not computed for performance reasons
	 * @param ROI ROI of the provided integral images under which to generate the covariance matrix
	 * @return generated covariance matrix
	 *
	 * @pre intImgs argument is well formed.
	 *		ROI argument is well formed and must fit inside the first integral image
	 * @post returned covarMatrix is well formed
	 */
	static cv::Mat generateMainCovarMatrix(const std::pair<std::vector<cv::Mat>, std::vector<std::vector<cv::Mat>>>& intImgs, const cv::Rect& ROI);

	/**
	 * Generate a covariance matrix of image features from the origin to a specified point using integral images.
	 *
	 * @param intImgs a pair of 1d and 2d vectors containing the integral feature images of the entire object or image.  
	 *				  The first vector contains the integral images of each feature image.  
	 *				  The second 2d matrix contains the integral of each permutation of multiplied feature images.
	 *				  As this is a symmetric matrix, the upper triangle in not computed for performance reasons
	 * @param br the bottom right boundary of the image, over which the covariance maxrix will be calculated
	 * @return covariance matrix of the image features from the origin to the specified point
	 *
	 * @pre intImgs argument is well formed.
	 *		br argument must fit inside the first integral image
 	 * @post returned covariance matrix is a non-empty square matrix with size > 0.
	 *		 returned covariance matrix has 1 channel whose elements are of type double
	 */
	

	/**
	 * Generate a covariance matrix from a general region of an image using integral images.
	 *
	 * @param intImgs a pair of 1d and 2d vectors containing the integral feature images of the entire object or image.  
	 *				  The first vector contains the integral images of each feature image.  
	 *				  The second 2d matrix contains the integral of each permutation of multiplied feature images.
	 *				  As this is a symmetric matrix, the upper triangle in not computed for performance reasons
	 * @param ROI the area within the image, over which the covariance maxrix will be calculated
	 * @return covariance matrix of the feature images over the specified region
	 *
	 * @pre intImgs argument is well formed.
	 *		ROI argument is well formed and must fit inside the first integral image
 	 * @post returned covariance matrix is a non-empty square matrix with size > 0.
	 *		 returned covariance matrix has 1 channel whose elements are of type double
	 */
	static cv::Mat calcCovarMatrix(const std::pair<std::vector<cv::Mat>, std::vector<std::vector<cv::Mat>>>& intImgs, const cv::Rect& ROI);

	/**
	 * Generate a set of 4 covariance matrices from a provided feature image and ROI for use during match filtering.
	 * Target bounds of each covariance matrix detailed in Tuzel et. al.
	 *
	 * @param covarMatrices RETURN vector containing generated covariance matrices
	 * @param intImgs a pair of 1d and 2d vectors containing the integral feature images of the entire object or image.  
	 *				  The first vector contains the integral images of each feature image.  
	 *				  The second 2d matrix contains the integral of each permutation of multiplied feature images.
	 *				  As this is a symmetric matrix, the upper triangle in not computed for performance reasons
	 * @param overallROI maximum ROI of the provided feature images to consider
	 *
	 * @pre intImgs argument is well formed.  
	 *		overallROI argument is well formed and must fit inside the first integral image
	 */


	/**
	 * Returns whether val1 is less than or equal to val2.
	 * Takes into account errors that may arrise from floating point arithmetic.
	 *
	 * @param val1 value to be compared
	 * @param val2 value to compare against
	 * @return whether val1 is less than or equal to val2
	 */
	static bool FLE(const float val1, const float val2);

	/**
	 * Compare two covariance matrices using a covariance matrix distance measure proposed by Forstner et. al.
	 * The function will be much faster if both matrices are symmetric and mat2 is positive definite.
	 * In this case only dsygv is called.  Otherwise both dsygv and dggev are called to return the correct eigenvalues.
	 * A covariance matrix is only positive semi-definite if it is not full rank.
	 *
	 * @param mat1 matrix of arbitrary dimensions.
	 *			   this matrix will be overwritten
	 * @param mat2 matrix to compare against
	 * @return the similarity distance of the compared matrices.
	 *		   better feature distance are minimal values
	 *
	 * @pre mat1 argument is well formed.
	 *		mat2 argument is well formed and is the same size as mat1 argument
	 * @post returned distance is >= 0
	 */
	//    static double compareMatrices(cv::Mat &mat1, const cv::Mat &mat2);

	/**
	 * Calculates the generalised eigenvalues of 2 symmetric matrices using the lapack function dsygv_.
	 * If mat2 is not positive definite the function will quit with no eigenvalues produced. 
	 * If mat2 is positive definite mat 1 will be overwritten and correct eigenvalues will be produced.	 	 
	 *
	 * @param info RETURN output information.
	 *			   will be 0 on successful eigenvalue generation and will contain error code otherwise
	 * @param w RETURN output vector containing eigenvalues on successful function exit
	 * @param mat1 one of two symmetric input matrices for which to find generalised eigenvalues.  
	 *			   on successful exit this will be overwritten
	 * @param mat2 other positive definite symmetric input matrix for which to find generalised eigenvalues.
	 *			   contents is cloned within function to preserve contents
	 *
	 * @pre mat1 argument is well formed.
	 *		mat2 argument is well formed and is the same size as mat1 argument
	 * @post returned column vector is non-empty with size == the order of the input matrices.  
	 *		 returned column vector has 1 channel whose elements are of type double
	 */
	//	static void dsygv(int &info, cv::Mat &w, cv::Mat &mat1, const cv::Mat &mat2);

	/**
	 * Calculates the generalised eigenvalues of 2 general matrices using the lapack function dggev_.
	 *
	 * @param info RETURN output information.
	 *			   will be 0 on successful eigenvalue generation and will contain error code otherwise
	 * @param w RETURN output vector containing eigenvalues on successful function exit
	 * @param mat1 one of two input matrices for which to find generalised eigenvalues.  
	 *			   on successful exit this will be overwritten
	 * @param mat2 other input matrix for which to find generalised eigenvalues.
	 *			   contents is cloned within function to preserve contents
	 *
	 * @pre mat1 argument is well formed.
	 *		mat2 argument is well formed and is the same size as mat1 argument
	 * @post returned column vector is non-empty with size == the order of the input matrices.  
	 *		 returned column vector has 1 channel whose elements are of type double
	 */
	//	static void dggev(int &info, cv::Mat &w, cv::Mat &mat1, const cv::Mat &mat2);


	/**
	   * Expose the FLE method to the Covariance Feature Detector.
	   */
	friend class CovarDetector;

	/**
	 * Allow the Covariance Descriptor Extractor to generate feature image, integral images and covariance matrices.
	 */
	friend class CovarExtractor;

	/**
	 * Allow the Covariance Matrix distance metrix to calculate the similarity of two matrices
	 */
	friend class CovarDist;
};


#endif
