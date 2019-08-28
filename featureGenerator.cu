#include "featureGenerator.h"
#include <iostream>

#include<cuda.h>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<ctime>
const int sqFeatureSize = (FEATURE_COUNT)*(FEATURE_COUNT + 1) / 2;
__global__ void integralRow(float ** i_data_array, int width, int height)
{
	int r = blockDim.x * blockIdx.x + threadIdx.x;
	if (r >= height)
		return;

	float * i_data = i_data_array[blockIdx.y];
	float  rs = 0.0;
	for (int c = 0; c < width; c++)
	{
		rs += i_data[r * width + c];
		i_data[r * width + c] = rs;
	}
}


__global__ void scan(float **i_data_array, int width, int height)
{
	int imageNumber = blockIdx.x;
	int rowNumber = blockIdx.y; // kontrol e gerek yok .height kadar olacak

	__shared__ float temp[2048];
	int thid = threadIdx.x;
	int offset = 1;

	temp[2 * thid] = i_data_array[imageNumber][rowNumber*width + 2 * thid];
	temp[2 * thid + 1] = i_data_array[imageNumber][rowNumber*width + 2 * thid + 1];

	for (int d = width >> 1; d > 0; d >>= 1)                    // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) { temp[width - 1] = 0; } // clear the last element
	for (int d = 1; d < width; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
		i_data_array[imageNumber][rowNumber*width + 2047] = temp[2047] + i_data_array[imageNumber][rowNumber*width + 2047];
		return;
	}
	i_data_array[imageNumber][rowNumber*width + 2 * thid - 1] = temp[2 * thid];
	i_data_array[imageNumber][rowNumber*width + 2 * thid] = temp[2 * thid + 1];
}


__global__ void integralCol(float **i_data_array, int width, int height)
{
	unsigned int imageIndex = blockIdx.y;
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= width)
		return;
	float * i_data = i_data_array[imageIndex];
	float rs = i_data[c];
	for (int r = 1; r < height; r++)
	{
		rs += i_data[r * width + c];
		i_data[r * width + c] = rs;
	}
}


__device__ float getDeterminant(float* m)
{
	float  tmp, det = 1;
	int k = 1, t;

	for (int i = 0; i < FEATURE_COUNT - 1; i++)
	{
		tmp = m[i*FEATURE_COUNT + i];
		if (tmp == 0){
			t = i;
			do{
				t++;
				if (t == FEATURE_COUNT) return 0;
				tmp = m[t*FEATURE_COUNT + i];
			} while (tmp == 0);
			float temp;
			for (int xx = 0; xx < FEATURE_COUNT; xx++){
				temp = m[i*FEATURE_COUNT + xx];
				m[i*FEATURE_COUNT + xx] = m[t*FEATURE_COUNT + xx];
				m[t*FEATURE_COUNT + xx] = temp;
			}
			k *= -1;
		}

		det *= tmp;
		for (int j = FEATURE_COUNT - 1; j >= i; j--)
			m[i*FEATURE_COUNT + j] /= tmp;
		for (int j = i + 1; j < FEATURE_COUNT; j++)
		{
			tmp = m[j*FEATURE_COUNT + i];
			for (int k = FEATURE_COUNT - 1; k >= i; k--)
				m[j*FEATURE_COUNT + k] -= tmp*m[i*FEATURE_COUNT + k];
		}
	}
	tmp = m[(FEATURE_COUNT - 1) * FEATURE_COUNT + FEATURE_COUNT - 1];
	det *= tmp*k;
	return det;
}

__global__  void kernel_calCovMatrix(float * distancesPtr, float * objectDescriptor,float * objectLogDet, int KpNumX, int kpNumY, int winStep, int winSize, int imgCols, int imgRows, int roiArea, float ** intImgs, float ** sqIntImgs)
{
	if (threadIdx.x >= (FEATURE_COUNT*(FEATURE_COUNT + 1)) / 2) return;
	const unsigned int covMatIndex = yIndex[threadIdx.x] * FEATURE_COUNT + xIndex[threadIdx.x];
	unsigned int integralImage1; // row
	unsigned int integralImage2; // col
	const unsigned int keyptIndex = blockIdx.x + blockIdx.y * KpNumX;

	integralImage1 = yIndex[threadIdx.x];
	integralImage2 = xIndex[threadIdx.x];

	const int sqIndex = (integralImage1*(integralImage1 + 1) / 2) + integralImage2;
	float * individiualIntImg1 = intImgs[integralImage1];
	float * individualIntImg2 = intImgs[integralImage2];
	float * combinedIntImg = sqIntImgs[sqIndex];

	const unsigned int ROITopLeftIndex = (blockIdx.y*winStep)*imgCols + (blockIdx.x*winStep);
	const unsigned int ROITopRightIndex = ROITopLeftIndex + winSize - 1;
	const unsigned int ROIBottomLeftIndex = ROITopLeftIndex + (winSize - 1)*imgCols;
	const unsigned int ROIBottomRightIndex = ROIBottomLeftIndex + winSize - 1;

	const float sumIndFeatureVals1 = (individiualIntImg1[ROITopLeftIndex] - individiualIntImg1[ROIBottomLeftIndex] - individiualIntImg1[ROITopRightIndex]) + individiualIntImg1[ROIBottomRightIndex];
	const float sumCombinedFeatureVals = (combinedIntImg[ROITopLeftIndex] - combinedIntImg[ROIBottomLeftIndex] - combinedIntImg[ROITopRightIndex]) + combinedIntImg[ROIBottomRightIndex];
	const float sumIndFeatureVals2 = (individualIntImg2[ROITopLeftIndex] - individualIntImg2[ROIBottomLeftIndex] - individualIntImg2[ROITopRightIndex]) + individualIntImg2[ROIBottomRightIndex];
	const float summedTerms = sumCombinedFeatureVals - sumIndFeatureVals1*sumIndFeatureVals2 / roiArea;

	__shared__ float descrptr[FEATURE_COUNT*FEATURE_COUNT];
	descrptr[covMatIndex] = summedTerms / (roiArea - 1);
	descrptr[xIndex[threadIdx.x] * FEATURE_COUNT + yIndex[threadIdx.x]] = descrptr[covMatIndex];
	__shared__ float objAndWindow[FEATURE_COUNT*FEATURE_COUNT];
	objAndWindow[covMatIndex] = (descrptr[covMatIndex] + objectDescriptor[covMatIndex])/2;
	objAndWindow[xIndex[threadIdx.x] * FEATURE_COUNT + yIndex[threadIdx.x]] = objAndWindow[covMatIndex];
	__syncthreads();

	float left = log(getDeterminant(objAndWindow));
	float right = 0.5*(objectLogDet[0] + log(getDeterminant(descrptr)));
	distancesPtr[keyptIndex] = left - right;

}

__global__ void kernel_calculateGreyImage(uchar * inputImageColor,uchar * greyInput,int row,int col)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (y >= row || x >= col) return;
	unsigned int index = y*col + x;

	//Y = 0.299 R + 0.587 G + 0.114 B
	//float r,g,b;
	greyInput[index] = 0.114*(float)inputImageColor[3 * index]+0.587*(float)inputImageColor[3 * index + 1]+0.299*(float)inputImageColor[3 * index + 2];

}


__global__ void kernel_generateFeatureImages(uchar * inputImageColor, uchar * greyInput, float ** features, int col, int row)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (y >= row || x >= col) return;
	unsigned int index = y*col + x;

	features[B_CHANNEL][index] = (float)inputImageColor[3 * index];
	features[G_CHANNEL][index] = (float)inputImageColor[3 * index + 1];
	features[R_CHANNEL][index] = (float)inputImageColor[3 * index + 2];

	float magnitude_xx = 0;
	float magnitude_yy = 0;
#pragma unroll
	for (uint j = 0; j < 3; ++j)
	{
#pragma unroll
		for (uint i = 0; i < 3; ++i)
		{
			const int x_focus = i + x - 1;
			const int y_focus = j + y - 1;
			const int filterIndex = x_focus + y_focus * col;
			if (filterIndex < 0 || filterIndex >= row*col) continue;
			magnitude_xx += greyInput[filterIndex] * sobel_x[i][j];
			magnitude_yy += greyInput[filterIndex] * sobel_y[i][j];
		}
	}
	features[D_X][index] = fminf(255, fmaxf(0, magnitude_xx/9));
	features[D_Y][index] = fminf(255, fmaxf(0, magnitude_yy/9));
#if FEATURE_COUNT == 7
	magnitude_xx = 0;
	magnitude_yy = 0;
#pragma unroll
	for (uint j = 0; j < 5; ++j)
	{
#pragma unroll
		for (uint i = 0; i < 5; ++i)
		{
			const int x_focus = i + x - 1;
			const int y_focus = j + y - 1;
			const int filterIndex = x_focus + y_focus * col;
			if (filterIndex < 0 || filterIndex >= row*col) continue;
			magnitude_xx += greyInput[filterIndex] * sobel_xx[i][j];
			magnitude_yy += greyInput[filterIndex] * sobel_yy[i][j];
		}
	}

	features[D_XX][index] = fminf(255, fmaxf(0, magnitude_xx));
	features[D_YY][index] = fminf(255, fmaxf(0, magnitude_yy));

#endif

}

void featureGenerator::generateFeatureImages()
{
	// bunu integralcol gibi cagirabiliriz
	kernel_calculateGreyImage<<< {(uint)(imageCols+31)/32,(uint)(imageRows + 31) / 32},{32,32}>>>(inputImage_UCP_d,greyImage_UCP_d,
		imageRows,imageCols);
	kernel_generateFeatureImages << < { (uint)(imageCols + 31) / 32, (uint)(imageRows + 31) / 32}, { 32, 32 } >> >(inputImage_UCP_d,
		 greyImage_UCP_d, _features_fp_d, imageCols, imageRows);
}


__global__ void kernel_pointwiseMultiply(float ** features, float ** sqFeatures, int size)
{
	// bura 4 erli 4 erli yapilabilir
	int pixelPos = blockIdx.x*blockDim.x + threadIdx.x;
	if (pixelPos >= size / 2) return;
	sqFeatures[blockIdx.y][2 * pixelPos] = features[xIndex[blockIdx.y]][2 * pixelPos] * features[xIndex[blockIdx.y]][2 * pixelPos];
	sqFeatures[blockIdx.y][2 * pixelPos + 1] = features[xIndex[blockIdx.y]][2 * pixelPos + 1] * features[xIndex[blockIdx.y]][2 * pixelPos + 1];
}

void featureGenerator::generateSquareFeatures()
{
	kernel_pointwiseMultiply << <{(unsigned int)imageCols / 2, ((FEATURE_COUNT + 1)*FEATURE_COUNT) / 2}, { 1024 } >> >(_features_fp_d, _features_fp_d + FEATURE_COUNT, imageCols*imageRows);
}


void featureGenerator::syncAnderrorCheck()
{
	cudaDeviceSynchronize();

	cudaError error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		std::cout << cudaGetErrorString(error) << std::endl;
	}
}

void featureGenerator::allocateDevice()
{
	cudaDeviceReset();
	cudaMalloc(&inputImage_UCP_d, pixelNumber * image_Mat.channels());
	cudaMalloc(&greyImage_UCP_d, pixelNumber);
	cudaMalloc(static_cast<float***>(&_features_fp_d), (sqFeatureSize + FEATURE_COUNT)*sizeof(float*));
	for (int featureIndex = 0; featureIndex < FEATURE_COUNT; featureIndex++)
	{
		cudaMalloc(&features_fp[featureIndex], pixelNumber * sizeof(float));
		for (int j = 0; j <= featureIndex; j++)
		{
			const int indexInLowerTriangle = ((featureIndex)*(featureIndex + 1) / 2) + j;
			cudaMalloc(&sqFEatures_fp[indexInLowerTriangle], byteCount_float);
		}
	}
	cudaMemcpy(_features_fp_d, features_fp, FEATURE_COUNT*sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(_features_fp_d + FEATURE_COUNT, sqFEatures_fp, sqFeatureSize*sizeof(float*), cudaMemcpyHostToDevice);

}


void featureGenerator::allocateCovarianceMatrixSpace(vector<KeyPoint> & keyptsCuda)
{
	int totalKp = 0;
	for (int sc = 0; sc < 8; sc++)
	{
		const double objectScale = __vScaleList[sc];
		__vCurWinSize[sc] = std::floor(winSize*objectScale);
		__vWinStep[sc] = std::ceil(__vCurWinSize[sc] * tolerance);
		const int maxXCord = imageCols - __vCurWinSize[sc];
		const int maxYCord = imageRows - __vCurWinSize[sc];

		const unsigned int kpNumberX = (maxXCord + __vWinStep[sc] - 1) / __vWinStep[sc];
		const unsigned int kpNumberY = (maxYCord + __vWinStep[sc] - 1) / __vWinStep[sc];
		__vKpNumberX[sc] = kpNumberX;
		__vKpNumberY[sc] = kpNumberY;
		totalKp+=kpNumberX*kpNumberY;

		for (int j = 0; j < kpNumberY; ++j)
		{
			for (int i = 0; i < kpNumberX; ++i)
			{
				cv::KeyPoint kp;
				kp.pt.x = i*__vWinStep[sc];
				kp.pt.y = j*__vWinStep[sc];
				kp.size = winSize*objectScale;
				keyptsCuda.push_back(kp);
			}
		}
	}

	totalKeypoints = totalKp;
	blockNumber = (totalKeypoints+ELEM_PER_BLOCK-1)/ELEM_PER_BLOCK;

	cudaMallocHost(&__covMat,totalKp*sizeof(float));
	//__covMat = new float[totalKp];
	cudaMalloc(static_cast<float**>(&_covMatDistances_d),totalKp * sizeof(float));
	cudaMalloc(static_cast<int**>(&__resultsIndex_d),blockNumber*sizeof(int));
	cudaMalloc(static_cast<float**>(&_resultsVals_d),blockNumber*sizeof(float));

	cudaMallocHost( &_resultsVals_h,blockNumber*sizeof(float));
	//_resultsVals_h = new float[blockNumber];
	cudaMallocHost( &__resultsIndex_h,blockNumber*sizeof(int));
	//	__resultsIndex_h = new int[blockNumber];

}



void featureGenerator::copyObjectDescriptor2Device(float * objectDescriptor, float objectLogDeterminant)
{
	cudaMalloc((void**)&__objDescriptor_d, FEATURE_COUNT*FEATURE_COUNT*sizeof(float));
	cudaMalloc((void**)&__objLogDeterminant_d, sizeof(float));
	cudaMemcpy(__objDescriptor_d, objectDescriptor, FEATURE_COUNT*FEATURE_COUNT*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(__objLogDeterminant_d, &objectLogDeterminant, sizeof(float), cudaMemcpyHostToDevice);
}

void featureGenerator::copyDataToDevice()
{
	cudaMemcpy(inputImage_UCP_d, image_Mat.data, pixelNumber * image_Mat.channels(), cudaMemcpyHostToDevice);
}

void featureGenerator::setWinSize(const unsigned & winSize, const float & tolerance)
{
	this->winSize = winSize;
	this->tolerance = tolerance;
}

void featureGenerator::resetDetector()
{
	cudaDeviceReset();
}



void featureGenerator::calculateCovarianceMatrix()
{
	int indexOfKpInFeatures = 0;
	for (int scale_ = 0; scale_ < 8; scale_++)
	{
		const float curWinSize = __vCurWinSize[scale_];
		const int winStep = __vWinStep[scale_];
		const unsigned int kpNumberX = __vKpNumberX[scale_];
		const unsigned int kpNumberY = __vKpNumberY[scale_];
		kernel_calCovMatrix << < {kpNumberX, kpNumberY}, { 32 } >> >(_covMatDistances_d+indexOfKpInFeatures, 
		__objDescriptor_d,__objLogDeterminant_d, kpNumberX, kpNumberY, winStep, curWinSize, imageCols, imageRows, curWinSize*curWinSize,
			_features_fp_d, _features_fp_d + FEATURE_COUNT);
		indexOfKpInFeatures +=kpNumberX*kpNumberY;
	}
}


std::pair< std::vector<Mat>, std::vector< std::vector<Mat> > >featureGenerator::getIntegralImages()

{
	std::vector<Mat> IntImgs;
	std::vector< std::vector<Mat> > sq;
	int ctr = 0;
	for (int x = 0; x < FEATURE_COUNT; ++x)
	{
		float * temp = new float[imageRows*imageCols];
		cudaMemcpy(temp, features_fp[x], byteCount_float, cudaMemcpyDeviceToHost);
		IntImgs.push_back(Mat(imageRows, imageCols, CV_32F, temp));
		std::vector<Mat> row;
		for (int y = 0; y <= x; y++)
		{
			float * temp2 = new float[imageRows*imageCols];
			cudaMemcpy(temp2, sqFEatures_fp[ctr++], byteCount_float, cudaMemcpyDeviceToHost);
			Mat sqat = Mat(imageRows, imageCols, CV_32F, temp2);
			row.push_back(sqat);
		}
		sq.push_back(row);
	}
	std::pair< std::vector<Mat>, std::vector< std::vector<Mat> > > retTemp = std::make_pair(IntImgs, sq);
	return retTemp;
}

void featureGenerator::calculateObjectIntegral()
{
	dim3 grid, block;

	block.x = 1024;
	block.y = 1;
	block.z = 1;

	grid.x = (imageRows + 1023) / 1024;
	grid.y = FEATURE_COUNT + sqFeatureSize;
	grid.z = 1;

	dim3 gridForCol;
	gridForCol.x = (imageCols + 1023) / 1024;
	gridForCol.y = FEATURE_COUNT + sqFeatureSize;
	gridForCol.z = 1;

	integralCol << <gridForCol, block >> >(_features_fp_d, imageCols, imageRows);
	integralRow << <grid, block >> >(_features_fp_d, imageCols, imageRows);
}

void featureGenerator::calculateIntegralImages()
{

	dim3  block;

	block.x = 1024;
	block.y = 1;
	block.z = 1;

	dim3 gridForCol;
	gridForCol.x = (imageCols + 1023) / 1024;
	gridForCol.y = FEATURE_COUNT + sqFeatureSize;
	gridForCol.z = 1;



	integralCol << <gridForCol, block >> >(_features_fp_d, imageCols, imageRows);
	scan << < {FEATURE_COUNT + sqFeatureSize, (unsigned int)imageRows}, { 1024 } >> >(_features_fp_d, imageCols, imageRows);
}


featureGenerator::featureGenerator()
{
	__vScaleList = vector<double>(8);
	__vCurWinSize = vector<double>(8);
	__vWinStep = vector<int>(8);
	__vKpNumberX = vector<int>(8);
	__vKpNumberY = vector<int>(8);

	__vScaleList[0] = 0.5;
	__vScaleList[1] = 0.75;
	__vScaleList[2] = 0.85;
	__vScaleList[3] = 1.0;
	__vScaleList[4] = 1.2;
	__vScaleList[5] = 1.5;
	__vScaleList[6] = 1.75;
	__vScaleList[7] = 2;

	features_fp = new float*[FEATURE_COUNT];
	sqFEatures_fp = new float*[sqFeatureSize];
	totalKeypoints=0;
}


featureGenerator::~featureGenerator()
{
}
void featureGenerator::setInputImage(const Mat& image)
{
	this->image_Mat = image;
	this->imageRows = image_Mat.rows;
	this->imageCols = image_Mat.cols;
	this->pixelNumber = imageCols * imageRows;
	this->byteCount_float = sizeof(float)*pixelNumber;
	allocateDevice(); // must include covariance space
}


void featureGenerator::updateImage(const Mat & imag)
{
	this->image_Mat = imag;
	copyDataToDevice();
}



__global__ void ReductionMin(float *input,int n, float *resultVals,int * resultIndex)    //take thread divergence into account
{	
    __shared__ float sdata[ELEM_PER_BLOCK];
    __shared__ float indexes[ELEM_PER_BLOCK];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    indexes[threadIdx.x]= i;
	 //load input into __shared__ memory 
     sdata[threadIdx.x] = FLT_MAX;  
	if(i < n) 
	sdata[threadIdx.x] = input[i]; 
	__syncthreads();
    
	// block-wide reduction
	for(unsigned int offset = blockDim.x>>1; offset > 0; offset >>= 1)
	{
		__syncthreads();
		if(threadIdx.x < offset)
	    {
            if(sdata[threadIdx.x + offset] < sdata[threadIdx.x] && sdata[threadIdx.x + offset]>=0 )
            {
                sdata[threadIdx.x] = sdata[threadIdx.x + offset];
                indexes[threadIdx.x]= indexes[threadIdx.x + offset];
            }
		}

	}

		// finally, thread 0 writes the result 
	if(threadIdx.x == 0) 
	{ 
		// the result is per-block 
        resultVals[blockIdx.x] = sdata[0]; 
        resultIndex[blockIdx.x]= indexes[0];
	} 
}



void featureGenerator::findMinimumOnGPU(int & IndexFound)
{
	float* covmatdistances_h = new float[totalKeypoints];
//	ReductionMin<<<blockNumber,1024>>>(_covMatDistances_d,totalKeypoints,_resultsVals_d,__resultsIndex_d);
//	syncAnderrorCheck();
////	clock_t start = clock();
//	cudaMemcpy(__resultsIndex_h,__resultsIndex_d,blockNumber*sizeof(int),cudaMemcpyDeviceToHost);
//	//std::cout<<blockNumber<<" time "<<(1000*(clock()-start))/CLOCKS_PER_SEC<<endl;
//	cudaMemcpy(_resultsVals_h,_resultsVals_d,blockNumber*sizeof(float),cudaMemcpyDeviceToHost);
 
	cudaMemcpy(covmatdistances_h, _covMatDistances_d, totalKeypoints*sizeof(float), cudaMemcpyDeviceToHost);
	syncAnderrorCheck();
	float minVal = FLT_MAX;
	int indexMin = 0;
	for (int m = 0; m < totalKeypoints;m++ )
	{
		if (minVal >= covmatdistances_h[m])
		{
			minVal = covmatdistances_h[m];
			indexMin = m;
		}
	}
	delete covmatdistances_h;
	/*float minVal = FLT_MAX;
	int indexMin =0;
	for( int x = 0;x<blockNumber;x++)
	{
		if( minVal>_resultsVals_h[x])
		{
			minVal = _resultsVals_h[x];
			indexMin = __resultsIndex_h[x];
		}
	}*/
	IndexFound = indexMin;

}
