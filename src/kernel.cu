
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <windows.h>
#include <cuda.h>


using namespace std;
using namespace cv;


extern "C" void Convolutional_layer1(double* bicubic_Y, double* (&conv1_result), double* weight_conv1, double* bias_conv1, int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num, double& TestTime_total_CUDA);
extern "C" void Convolutional_layer_2_to_19(int Layer_num, double* Prior_conv_result, double* total_dummy, double* (&conv_result), double* weight_conv, double* bias_conv, int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num, double& TestTime_total_CUDA);
extern "C" void Convolutional_layer20(double* conv19_result, double* (&conv20_data), double* weight_conv20, double* bias_conv20, int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num, int Layer_20_CH_Num, double& TestTime_total_CUDA);

#define blocksize 32
#define KERNEL_RADIUS1	4
#define KERNEL_RADIUS2	2
#define KERNEL_RADIUS3	2

extern "C" void gpu_bicubic(double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda, double* bicuY_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* bicuYtem_cuda, double* bicuCbtem_cuda, double* bicuCrtem_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void gpu_ycbcr2rgb(double* conv3_data_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* output_R_cuda, double* output_G_cuda, double* output_B_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void gpu_rgb2ycbcr(double* oriR_cuda, double* oriG_cuda, double* oriB_cuda, double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void gpu_srcnn(double* bicuY_cuda, double* conv1_data_cuda, double* conv2_data_cuda, double* conv3_data_cuda, float* w1, float* w2, float* w3, double* b1, double* b2, double* b3, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void all(float* weight_conv1_float_cuda, float* weight_conv1_float, float* weight_conv2_float_cuda, float* weight_conv2_float, float* weight_conv3_float_cuda, float* weight_conv3_float, double* biases_conv1_cuda, double* biases_conv1, double* biases_conv2_cuda, double* biases_conv2, double* biases_conv3_cuda, double* biases_conv3,
	double* oriR, double* oriG, double* oriB, double* oriR_cuda, double* oriG_cuda, double* oriB_cuda, double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda,
	double* bicuY_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* bicuYtem_cuda, double* bicuCbtem_cuda, double* bicuCrtem_cuda, double* conv1_data_cuda, double* conv2_data_cuda, double* conv3_data_cuda,
	double* output_R, double* output_R_cuda, double* output_G, double* output_G_cuda, double* output_B, double* output_B_cuda, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void unbindTexture();

texture<float, 1, cudaReadModeElementType> texture_weight1;
texture<float, 1, cudaReadModeElementType> texture_weight2;
texture<float, 1, cudaReadModeElementType> texture_weight3;


__constant__ double biases_conv1_constant[64];
__constant__ double biases_conv2_constant[32];
__constant__ double biases_conv3_constant[1];

__global__ void convtest(double*bicubic_Y, double*conv1_data, double*weight_conv1, double* bias_conv1, double* total_result, int layer_idx, int HIgh_w, int HIgh_h, int Filter_Size) {//int HIgh_w, int HIgh_h, int Filter_Size)
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k;
	int l;
	int half_filter_size = (int)(Filter_Size / 2);

	if ((half_filter_size <= i) && (i < HIgh_h - half_filter_size)) {
		if ((half_filter_size <= j) && (j < HIgh_w - half_filter_size)) {
			conv1_data[i*HIgh_w + j] = 0;

			// 3x3 convolution
			for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
				for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
					conv1_data[i*HIgh_w + j] = conv1_data[i*HIgh_w + j]
						+ (bicubic_Y[(i + k)*HIgh_w + (j + l)] / 255)
						* weight_conv1[layer_idx*Filter_Size*Filter_Size + (l + half_filter_size)*Filter_Size + (k + half_filter_size)];
				}
			}
			conv1_data[i*HIgh_w + j] = (conv1_data[i*HIgh_w + j] + bias_conv1[layer_idx]);

			// ReLU activation function
			if (conv1_data[i*HIgh_w + j] < 0)
				conv1_data[i*HIgh_w + j] = 0;

			total_result[layer_idx*HIgh_h*HIgh_w + i*HIgh_w + j] = conv1_data[i*HIgh_w + j];
		}
	}
}

__global__ void conv(double*bicubic_Y, double*conv1_data, double*weight_conv1, int layer1_idx, int layer2_idx, int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num) {//int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num)
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k;
	int l;
	int half_filter_size = (int)(Filter_Size / 2);

	if ((half_filter_size <= i) && (i < HIgh_h - half_filter_size)) {
		if ((half_filter_size <= j) && (j < HIgh_w - half_filter_size)) {
			conv1_data[i*HIgh_w + j] = 0;
			// 3x3 convolution
			for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
				for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
					conv1_data[i*HIgh_w + j] = conv1_data[i*HIgh_w + j]
											  + bicubic_Y[layer1_idx*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
											  * weight_conv1[layer2_idx*Filter_Size*Filter_Size*CH_Num + layer1_idx*Filter_Size*Filter_Size + (l + half_filter_size)*Filter_Size + (k + half_filter_size)];
				}
			}
		}
	}
}

__global__ void adding(double*conv1_data, double*conv2_data, int layer2_idx, int HIgh_w, int HIgh_h) {//int HIgh_w, int HIgh_h)
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	conv2_data[layer2_idx*HIgh_h*HIgh_w + i*HIgh_w + j] = conv1_data[i*HIgh_w + j] + conv2_data[layer2_idx *HIgh_h*HIgh_w + i*HIgh_w + j];
}

__global__ void add_bias(double*conv2_data, double*bias_conv, int layer2_idx, int HIgh_w, int HIgh_h) {//, int HIgh_w, int HIgh_h
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	conv2_data[layer2_idx*HIgh_h*HIgh_w + i*HIgh_w + j] = conv2_data[layer2_idx *HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv[layer2_idx];

	// ReLU activation function
	if (conv2_data[layer2_idx*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
		conv2_data[layer2_idx*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
}

void Convolutional_layer1(double* bicubic_Y, double* (&conv1_result), double* weight_conv1, double* bias_conv1, int HIgh_w, int HIgh_h, int Filter_Size,int CH_Num, double& TestTime_total_CUDA) {
	// ================================================= layer 1 =================================================
	// feature channel : 64
	printf("Layer 1 Foward Pass : ");

	weight_conv1 = (double *)malloc(sizeof(double)*Filter_Size*Filter_Size*CH_Num);
	FILE *in = fopen("Input/filter/weights_conv1.raw", "rb");
	fread(weight_conv1, sizeof(double), Filter_Size*Filter_Size*CH_Num, in);
	fclose(in);

	bias_conv1 = (double *)malloc(sizeof(double)*CH_Num);
	in = fopen("Input/filter/bias_conv1.raw", "rb");
	fread(bias_conv1, sizeof(double), CH_Num, in);
	fclose(in);


	// allocate data for Cuda input
	double *input1; cudaMalloc((void**)&input1, HIgh_h * HIgh_w * sizeof(double));
	double *result_temp; cudaMalloc((void**)&result_temp, HIgh_h*HIgh_w * sizeof(double));
	double *filter1; cudaMalloc((void**)&filter1, CH_Num*Filter_Size*Filter_Size * sizeof(double));
	double *filter_bias1; cudaMalloc((void**)&filter_bias1, CH_Num * sizeof(double));
	cudaMalloc((void**)&conv1_result, CH_Num * HIgh_h*HIgh_w * sizeof(double));

	// RGB assert into Cuda
	cudaMemcpy(input1, bicubic_Y, HIgh_h * HIgh_w * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(filter1, weight_conv1, CH_Num*Filter_Size*Filter_Size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(filter_bias1, bias_conv1, CH_Num * sizeof(double), cudaMemcpyHostToDevice);

	// Thread & Block assign
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(HIgh_w / 32, HIgh_h / 32, 1);

	int64 tStart = cvGetTickCount();

	for (int layer_idx = 0; layer_idx < CH_Num; layer_idx++) {
		convtest << <dimGrid, dimBlock >> >(input1, result_temp, filter1, filter_bias1, conv1_result, layer_idx, HIgh_w, HIgh_h, Filter_Size);
		cudaThreadSynchronize();
	}

	int64 tEnd = cvGetTickCount();
	double TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//printf("\nBefore TestTime_total_CUDA: %f3\n", TestTime_total_CUDA);
	TestTime_total_CUDA = TestTime_total_CUDA + TestTime;
	//printf("After TestTime_total_CUDA: %f3\n", TestTime_total_CUDA);

	printf("Done\n");
	printf("%f msec\n", TestTime);

}



void Convolutional_layer_2_to_19(int Layer_num,double* Prior_conv_result, double* total_dummy, double* (&conv_result), double* weight_conv, double* bias_conv, int HIgh_w, int HIgh_h, int Filter_Size,int CH_Num, double& TestTime_total_CUDA) {
	// ================================================= layer2 =================================================
	// feature channel of layer (N-1) : 64
	//printf("Layer 2 Forward Pass : ");
	printf("Layer %d Forward Pass : ", Layer_num);

	string weight_path = "Input/filter/weights_conv" + to_string(Layer_num) + ".raw";
	weight_conv = (double *)malloc(sizeof(double)*CH_Num*Filter_Size*Filter_Size*CH_Num);
	FILE *in = fopen(weight_path.c_str(), "rb");//FILE *in = fopen("Input/filter/weights_conv2.raw", "rb");
	fread(weight_conv, sizeof(double), CH_Num*Filter_Size*Filter_Size*CH_Num, in);
	fclose(in);

	string bias_path = "Input/filter/bias_conv" + to_string(Layer_num) + ".raw";
	bias_conv = (double *)malloc(sizeof(double)*CH_Num);
	in = fopen(bias_path.c_str(), "rb");//in = fopen("Input/filter/bias_conv2.raw", "rb");
	fread(bias_conv, sizeof(double), CH_Num, in);
	fclose(in);


	// allocate data for Cuda input

	double *result_temp; cudaMalloc((void**)&result_temp, HIgh_h*HIgh_w * sizeof(double));
	double *filter; cudaMalloc((void**)&filter, CH_Num*CH_Num*Filter_Size*Filter_Size * sizeof(double));
	double *filter_bias; cudaMalloc((void**)&filter_bias, CH_Num * sizeof(double));
	cudaMalloc((void**)&conv_result, CH_Num * HIgh_h*HIgh_w * sizeof(double));//double *conv2_result; 

	// RGB assert into Cuda
	cudaMemcpy(conv_result, total_dummy, CH_Num*HIgh_w*HIgh_h * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(filter, weight_conv, CH_Num*CH_Num*Filter_Size*Filter_Size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(filter_bias, bias_conv, CH_Num * sizeof(double), cudaMemcpyHostToDevice);

	// Thread & Block assign
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(HIgh_w / 32, HIgh_h / 32, 1);

	int64 tStart = cvGetTickCount();

	for (int layer2_idx = 0; layer2_idx < CH_Num; layer2_idx++)	{
		for (int layer1_idx = 0; layer1_idx < CH_Num; layer1_idx++)	{
			conv << <dimGrid, dimBlock >> > (Prior_conv_result, result_temp, filter, layer1_idx, layer2_idx, HIgh_w, HIgh_h, Filter_Size, CH_Num);
			cudaThreadSynchronize();
			adding << <dimGrid, dimBlock >> > (result_temp, conv_result, layer2_idx, HIgh_w, HIgh_h);
			cudaThreadSynchronize();
		}
		add_bias << <dimGrid, dimBlock >> > (conv_result, filter_bias, layer2_idx, HIgh_w, HIgh_h);
		cudaThreadSynchronize();
	}


	int64 tEnd = cvGetTickCount();
	double TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//printf("\nBefore TestTime_total_CUDA: %f3\n", TestTime_total_CUDA);
	TestTime_total_CUDA = TestTime_total_CUDA + TestTime;
	//printf("After TestTime_total_CUDA: %f3\n", TestTime_total_CUDA);

	printf("%f msec\n", TestTime);
}


void Convolutional_layer20(double* conv19_result, double* (&conv20_data), double* weight_conv20, double* bias_conv20, int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num,int Layer_20_CH_Num, double& TestTime_total_CUDA) {
	//================================================= layer 20 =================================================
	printf("Layer 20 Forward Pass : ");
	double* conv19_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*CH_Num);
	cudaMemcpy(conv19_data, conv19_result, CH_Num*HIgh_w*HIgh_h * sizeof(double), cudaMemcpyDeviceToHost);

	weight_conv20 = (double *)malloc(sizeof(double)*CH_Num*Filter_Size*Filter_Size);
	FILE *in = fopen("Input/filter/weights_conv20.raw", "rb");
	fread(weight_conv20, sizeof(double), CH_Num*Filter_Size*Filter_Size, in);
	fclose(in);

	bias_conv20 = (double *)malloc(sizeof(double)*Layer_20_CH_Num);
	in = fopen("Input/filter/bias_conv20.raw", "rb");
	fread(bias_conv20, sizeof(double), Layer_20_CH_Num, in);
	fclose(in);

	double* temp_data20 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*CH_Num);
	conv20_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_20_CH_Num);

	int64 tStart = cvGetTickCount();
	int i, j, k, l;
	int half_filter_size = (int)(Filter_Size / 2);

	for (i = 0; i < HIgh_h; i++){
		for (j = 0; j < HIgh_w; j++){
			conv20_data[i*HIgh_w + j] = 0;
		}
	}

	for (int layer19_index = 0; layer19_index < CH_Num; layer19_index++){
		for (i = 0; i < HIgh_h; i++){
			for (j = 0; j < HIgh_w; j++){
				temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// Convolution without padding
		for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
			for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
				// 3x3 convolution
				for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
					for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
						temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] 
																				   + conv19_data[layer19_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)] 
																				   * weight_conv20[layer19_index*Filter_Size*Filter_Size + (l + half_filter_size)*Filter_Size + (k + half_filter_size)];
					}
				}
			}
		}

		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv20_data[i*HIgh_w + j] = conv20_data[i*HIgh_w + j] + temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j];
			}
		}
	}
	for (i = 0; i < HIgh_h; i++) {
		for (j = 0; j < HIgh_w; j++) {
			conv20_data[i*HIgh_w + j] = conv20_data[i*HIgh_w + j] + bias_conv20[0];
			// ReLU activation function
			/*if (conv20_data[i*HIgh_w + j] <0)
				conv20_data[i*HIgh_w + j] = 0;*/
		}
	}

	int64 tEnd = cvGetTickCount();
	double TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	TestTime_total_CUDA = TestTime_total_CUDA + TestTime;

	printf("Done\n");
	printf("%f msec\n", TestTime);

}



///////////////////////////////////////////////////////////////////////////////////

//SRCNN

///////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_rgb2ycbcr(double *R, double *G, double *B, double *pcuY, double *pcuCb, double *pcuCr, int w, int h)
{
	__shared__ double data1[blocksize][blocksize];
	__shared__ double data2[blocksize][blocksize];
	__shared__ double data3[blocksize][blocksize];


	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y*w + x;

	data1[threadIdx.y][threadIdx.x] = R[index];
	data2[threadIdx.y][threadIdx.x] = G[index];
	data3[threadIdx.y][threadIdx.x] = B[index];

	__syncthreads();

	pcuY[index] = 0.256788235294118*data1[threadIdx.y][threadIdx.x] + 0.504129411764706*data2[threadIdx.y][threadIdx.x] + 0.0979058823529412*data3[threadIdx.y][threadIdx.x] + 16;
	pcuCb[index] = -0.148223529411765*data1[threadIdx.y][threadIdx.x] - 0.290992156862745*data2[threadIdx.y][threadIdx.x] + 0.4392156862745100*data3[threadIdx.y][threadIdx.x] + 128;
	pcuCr[index] = 0.439215686274510*data1[threadIdx.y][threadIdx.x] - 0.367788235294118*data2[threadIdx.y][threadIdx.x] - 0.0714274509803922*data3[threadIdx.y][threadIdx.x] + 128;


}


__global__ void cuda_ycbcr2rgb(double *pcuY, double *pcuCb, double *pcuCr, double *R, double *G, double *B, int w, int h)

{
	__shared__ double data1[blocksize][blocksize];
	__shared__ double data2[blocksize][blocksize];
	__shared__ double data3[blocksize][blocksize];


	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y*w + x;

	data1[threadIdx.y][threadIdx.x] = pcuY[index];
	data2[threadIdx.y][threadIdx.x] = pcuCb[index];
	data3[threadIdx.y][threadIdx.x] = pcuCr[index];

	__syncthreads();

	R[index] = floor(1.16438356164384000000*data1[threadIdx.y][threadIdx.x] + 0.00000030112439741101*data2[threadIdx.y][threadIdx.x] + 1.59602688733570000000*data3[threadIdx.y][threadIdx.x] - 222.921617109194 + 0.5);
	G[index] = floor(1.16438356164384000000*data1[threadIdx.y][threadIdx.x] - 0.39176253994145000000*data2[threadIdx.y][threadIdx.x] - 0.81296829216220500000*data3[threadIdx.y][threadIdx.x] + 135.575409522967 + 0.5);
	B[index] = floor(1.16438356164384000000*data1[threadIdx.y][threadIdx.x] + 2.01723263955646000000*data2[threadIdx.y][threadIdx.x] + 0.00000305426174524847*data3[threadIdx.y][threadIdx.x] - 276.836305795032 + 0.5);

}




__global__ void cuda_bicubic_temp(double *R, double *G, double *B, double *pcuDst, double *cuCb, double *cuCr, double* pcuDsttem, double *cuCbtem, double *cuCrtem, int w, int h, int Low_h)
{
	__shared__ double data1[blocksize + 4][blocksize];
	__shared__ double data2[blocksize + 4][blocksize];
	__shared__ double data3[blocksize + 4][blocksize];



	double bicubic2[8] = { -0.0234375, 0.2265625, 0.8671875, -0.0703125 ,
		-0.0703125, 0.8671875, 0.2265625, -0.0234375 };
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y*w + x;


	data1[threadIdx.y][threadIdx.x] = R[index - 2 * Low_h];
	data1[threadIdx.y + 2 * 2][threadIdx.x] = R[index + 2 * Low_h];
	data2[threadIdx.y][threadIdx.x] = G[index - 2 * Low_h];
	data2[threadIdx.y + 2 * 2][threadIdx.x] = G[index + 2 * Low_h];
	data3[threadIdx.y][threadIdx.x] = B[index - 2 * Low_h];
	data3[threadIdx.y + 2 * 2][threadIdx.x] = B[index + 2 * Low_h];




	__syncthreads();

	if (y > 2 && y < w - 2)
	{
		pcuDsttem[2 * index - x - Low_h] = floor(bicubic2[4] * data1[threadIdx.y][threadIdx.x] + bicubic2[5] * data1[threadIdx.y + 1][threadIdx.x] + bicubic2[6] * data1[threadIdx.y + 2][threadIdx.x] + bicubic2[7] * data1[threadIdx.y + 3][threadIdx.x] + 0.5);
		pcuDsttem[2 * index - x] = floor(bicubic2[0] * data1[threadIdx.y][threadIdx.x] + bicubic2[1] * data1[threadIdx.y + 1][threadIdx.x] + bicubic2[2] * data1[threadIdx.y + 2][threadIdx.x] + bicubic2[3] * data1[threadIdx.y + 3][threadIdx.x] + 0.5);

		cuCbtem[2 * index - x - Low_h] = floor(bicubic2[4] * data2[threadIdx.y][threadIdx.x] + bicubic2[5] * data2[threadIdx.y + 1][threadIdx.x] + bicubic2[6] * data2[threadIdx.y + 2][threadIdx.x] + bicubic2[7] * data2[threadIdx.y + 3][threadIdx.x] + 0.5);
		cuCbtem[2 * index - x] = floor(bicubic2[0] * data2[threadIdx.y][threadIdx.x] + bicubic2[1] * data2[threadIdx.y + 1][threadIdx.x] + bicubic2[2] * data2[threadIdx.y + 2][threadIdx.x] + bicubic2[3] * data2[threadIdx.y + 3][threadIdx.x] + 0.5);


		cuCrtem[2 * index - x - Low_h] = floor(bicubic2[4] * data3[threadIdx.y][threadIdx.x] + bicubic2[5] * data3[threadIdx.y + 1][threadIdx.x] + bicubic2[6] * data3[threadIdx.y + 2][threadIdx.x] + bicubic2[7] * data3[threadIdx.y + 3][threadIdx.x] + 0.5);
		cuCrtem[2 * index - x] = floor(bicubic2[0] * data3[threadIdx.y][threadIdx.x] + bicubic2[1] * data3[threadIdx.y + 1][threadIdx.x] + bicubic2[2] * data3[threadIdx.y + 2][threadIdx.x] + bicubic2[3] * data3[threadIdx.y + 3][threadIdx.x] + 0.5);
	}
	if (x>2 && x < h - 2 && y < Low_h)
	{
		pcuDst[2 * index - 1] = floor(bicubic2[4] * pcuDsttem[index - 2] + bicubic2[5] * pcuDsttem[index - 1] + bicubic2[6] * pcuDsttem[index] + bicubic2[7] * pcuDsttem[index + 1]) / 255;
		pcuDst[2 * index] = floor(bicubic2[0] * pcuDsttem[index - 2] + bicubic2[1] * pcuDsttem[index - 1] + bicubic2[2] * pcuDsttem[index] + bicubic2[3] * pcuDsttem[index + 1]) / 255;

		cuCb[2 * index - 1] = floor(bicubic2[4] * cuCbtem[index - 2] + bicubic2[5] * cuCbtem[index - 1] + bicubic2[6] * cuCbtem[index] + bicubic2[7] * cuCbtem[index + 1]);
		cuCb[2 * index] = floor(bicubic2[0] * cuCbtem[index - 2] + bicubic2[1] * cuCbtem[index - 1] + bicubic2[2] * cuCbtem[index] + bicubic2[3] * cuCbtem[index + 1]);

		cuCr[2 * index - 1] = floor(bicubic2[4] * cuCrtem[index - 2] + bicubic2[5] * cuCrtem[index - 1] + bicubic2[6] * cuCrtem[index] + bicubic2[7] * cuCrtem[index + 1]);
		cuCr[2 * index] = floor(bicubic2[0] * cuCrtem[index - 2] + bicubic2[1] * cuCrtem[index - 1] + bicubic2[2] * cuCrtem[index] + bicubic2[3] * cuCrtem[index + 1]);

	}
}




__global__ void cuda_bicubic_last(double *R, double *G, double *B, double *pcuDst, double *cuCb, double *cuCr, double* pcuDsttem, double *cuCbtem, double *cuCrtem, int w, int h, int Low_w, int Low_h, int HIgh_h)
{

	double bicubic2[8] = { -0.0234375, 0.2265625, 0.8671875, -0.0703125 ,
		-0.0703125, 0.8671875, 0.2265625, -0.0234375 };
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y*w + x;

	if (x>2 && x < h - 2)
	{
		pcuDst[2 * index + Low_w*HIgh_h - 1] = (floor(bicubic2[4] * pcuDsttem[index + Low_w*Low_h - 2] + bicubic2[5] * pcuDsttem[index + Low_w*Low_h - 1] + bicubic2[6] * pcuDsttem[index + Low_w*Low_h] + bicubic2[7] * pcuDsttem[index + Low_w*Low_h + 1])) / 255;
		pcuDst[2 * index + Low_w*HIgh_h] = (floor(bicubic2[0] * pcuDsttem[index + Low_w*Low_h - 2] + bicubic2[1] * pcuDsttem[index + Low_w*Low_h - 1] + bicubic2[2] * pcuDsttem[index + Low_w*Low_h] + bicubic2[3] * pcuDsttem[index + Low_w*Low_h + 1])) / 255;

		cuCb[2 * index + Low_w*HIgh_h - 1] = (floor(bicubic2[4] * cuCbtem[index + Low_w*Low_h - 2] + bicubic2[5] * cuCbtem[index + Low_w*Low_h - 1] + bicubic2[6] * cuCbtem[index + Low_w*Low_h] + bicubic2[7] * cuCbtem[index + Low_w*Low_h + 1]));
		cuCb[2 * index + Low_w*HIgh_h] = (floor(bicubic2[0] * cuCbtem[index + Low_w*Low_h - 2] + bicubic2[1] * cuCbtem[index + Low_w*Low_h - 1] + bicubic2[2] * cuCbtem[index + Low_w*Low_h] + bicubic2[3] * cuCbtem[index + Low_w*Low_h + 1]));

		cuCr[2 * index + Low_w*HIgh_h - 1] = (floor(bicubic2[4] * cuCrtem[index + Low_w*Low_h - 2] + bicubic2[5] * cuCrtem[index + Low_w*Low_h - 1] + bicubic2[6] * cuCrtem[index + Low_w*Low_h] + bicubic2[7] * cuCrtem[index + Low_w*Low_h + 1]));
		cuCr[2 * index + Low_w*HIgh_h] = (floor(bicubic2[0] * cuCrtem[index + Low_w*Low_h - 2] + bicubic2[1] * cuCrtem[index + Low_w*Low_h - 1] + bicubic2[2] * cuCrtem[index + Low_w*Low_h] + bicubic2[3] * cuCrtem[index + Low_w*Low_h + 1]));

	}
}


__global__ void cuda_Filter2D_texture(double * pSrcImage, double *pDstImage, int SrcWidth, int SrcHeight, int SrcDepth, int KWidth, int KHeight, int numKernel, double* bias)
{


	__shared__ double data[blocksize + 2 * KERNEL_RADIUS1][blocksize + 2 * KERNEL_RADIUS1];

	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y*SrcWidth + x;
	int x0, y0;
	int kCenter = KWidth*KHeight / 2;
	int kernelSize = KWidth * KHeight;
	float kernel_coeff;

	x0 = x - KERNEL_RADIUS1;
	y0 = y - KERNEL_RADIUS1;
	if (x0 < 0 || y0 < 0) {
		data[threadIdx.y][threadIdx.x] = 0;
	}
	else { data[threadIdx.y][threadIdx.x] = pSrcImage[index - KERNEL_RADIUS1 - KERNEL_RADIUS1*SrcWidth]; }
	x0 = x + KERNEL_RADIUS1;
	y0 = y - KERNEL_RADIUS1;
	if (y0 < 0 || x0 > SrcWidth) {
		data[threadIdx.y][threadIdx.x + 2 * KERNEL_RADIUS1] = 0;
	}
	else { data[threadIdx.y][threadIdx.x + 2 * KERNEL_RADIUS1] = pSrcImage[index + KERNEL_RADIUS1 - KERNEL_RADIUS1*SrcWidth]; }
	x0 = x - KERNEL_RADIUS1;
	y0 = y + KERNEL_RADIUS1;
	if (y0 > SrcHeight - 1 || x0 < 0) {
		data[threadIdx.y + 2 * KERNEL_RADIUS1][threadIdx.x] = 0;
	}
	else { data[threadIdx.y + 2 * KERNEL_RADIUS1][threadIdx.x] = pSrcImage[index - KERNEL_RADIUS1 + KERNEL_RADIUS1*SrcWidth]; }
	x0 = x + KERNEL_RADIUS1;
	y0 = y + KERNEL_RADIUS1;
	if (y0 > SrcHeight - 1 || x0 > SrcWidth - 1) {
		data[threadIdx.y + 2 * KERNEL_RADIUS1][threadIdx.x + 2 * KERNEL_RADIUS1] = 0;
	}
	else { data[threadIdx.y + 2 * KERNEL_RADIUS1][threadIdx.x + 2 * KERNEL_RADIUS1] = pSrcImage[index + KERNEL_RADIUS1 + KERNEL_RADIUS1*SrcWidth]; }
	__syncthreads();

	x0 = KERNEL_RADIUS1 + threadIdx.x;
	y0 = KERNEL_RADIUS1 + threadIdx.y;
	if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		for (int k = 0; k < numKernel; k++)
		{
			for (int l = 0; l < SrcDepth; l++)
			{
				for (int j = -KHeight / 2; j <= KHeight / 2; j++)
				{
					for (int i = -KWidth / 2; i <= KWidth / 2; i++)
					{

						kernel_coeff = tex1Dfetch(texture_weight1, k * kernelSize * SrcDepth + l * kernelSize + kCenter + KWidth*j + i);
						pDstImage[index + k*SrcWidth*SrcHeight] += data[y0 + j][x0 + i] * kernel_coeff;
					}
				}
			}
			//pDstImage[k*SrcWidth*SrcHeight + index] += biases_conv1_constant[k];
			pDstImage[k*SrcWidth*SrcHeight + index] += bias[k];
			if (pDstImage[k*SrcWidth*SrcHeight + index] < 0) pDstImage[k*SrcWidth*SrcHeight + index] = 0;
		}
	}
	else
	{
		for (int k = 0; k < numKernel; k++) {
			pDstImage[k*SrcWidth*SrcHeight + index] = 0;
		}
	}


}

__global__ void cuda_Filter2D2_texture(double * pSrcImage, double *pDstImage, int SrcWidth, int SrcHeight, int SrcDepth, int KWidth, int KHeight, int numKernel, double* bias)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y*SrcWidth + x;
	float kernel_coeff;

	int SrcArea = SrcWidth * SrcHeight;
	int kCenter = KWidth*KHeight / 2;
	int kernelSize = KWidth * KHeight;
	if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		for (int k = 0; k < numKernel; k++)
		{
			for (int l = 0; l < SrcDepth; l++)
			{
				for (int j = -KHeight / 2; j <= KHeight / 2; j++)
				{
					for (int i = -KWidth / 2; i <= KWidth / 2; i++)
					{
						kernel_coeff = tex1Dfetch(texture_weight2, k * kernelSize * SrcDepth + l * kernelSize + kCenter + KWidth*j + i);
						pDstImage[index + k*SrcWidth*SrcHeight] += pSrcImage[index + l * SrcArea + SrcWidth*j + i] * kernel_coeff;
					}
				}
			}
			//pDstImage[k*SrcWidth*SrcHeight + index] += biases_conv1_constant[k];
			pDstImage[k*SrcWidth*SrcHeight + index] += bias[k];
			if (pDstImage[k*SrcWidth*SrcHeight + index] < 0) pDstImage[k*SrcWidth*SrcHeight + index] = 0;
		}
	}
	else
	{
		for (int k = 0; k < numKernel; k++)
		{
			pDstImage[k*SrcWidth*SrcHeight + index] = 0;
		}
	}
}

__global__ void cuda_Filter2D_Last_texture(double * pSrcImage, double *pDstImage, int SrcWidth, int SrcHeight, int SrcDepth, int KWidth, int KHeight, int numKernel, double* bias)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y*SrcWidth + x;
	float kernel_coeff;

	int SrcArea = SrcWidth * SrcHeight;
	int kCenter = KWidth*KHeight / 2;
	int kernelSize = KWidth * KHeight;
	if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		for (int k = 0; k < numKernel; k++)
		{
			for (int l = 0; l < SrcDepth; l++)
			{
				for (int j = -KHeight / 2; j <= KHeight / 2; j++)
				{
					for (int i = -KWidth / 2; i <= KWidth / 2; i++)
					{
						kernel_coeff = tex1Dfetch(texture_weight3, k * kernelSize * SrcDepth + l * kernelSize + kCenter + KWidth*j + i);
						pDstImage[index + k*SrcWidth*SrcHeight] += pSrcImage[index + l * SrcArea + SrcWidth*j + i] * kernel_coeff;
					}
				}
			}
			//pDstImage[k*SrcWidth*SrcHeight + index] += biases_conv1_constant[k];
			pDstImage[k*SrcWidth*SrcHeight + index] += bias[k];
			if (pDstImage[k*SrcWidth*SrcHeight + index] < 0) pDstImage[k*SrcWidth*SrcHeight + index] = 0;
			pDstImage[k*SrcWidth*SrcHeight + index] *= 255;
		}
	}
	else
	{
		for (int k = 0; k < numKernel; k++)
		{
			pDstImage[k*SrcWidth*SrcHeight + index] = 0;
		}
	}
}


void gpu_srcnn(double* bicuY_cuda, double* conv1_data_cuda, double* conv2_data_cuda, double* conv3_data_cuda, float* w1, float* w2, float* w3, double* b1, double* b2, double* b3, int Low_w, int Low_h, int HIgh_w, int HIgh_h) {
	dim3 block1;
	dim3 grid1;
	dim3 block;
	dim3 grid;

	block1.x = blocksize;
	block1.y = blocksize;

	grid1.x = Low_h / block1.x;
	grid1.y = Low_w / block1.y;


	block.x = blocksize;
	block.y = blocksize;

	grid.x = HIgh_h / block.x;
	grid.y = HIgh_w / block.y;

	cudaBindTexture(0, texture_weight1, w1, 81 * 64 * sizeof(float));
	cudaBindTexture(0, texture_weight2, w2, 64 * 25 * 32 * sizeof(float));
	cudaBindTexture(0, texture_weight3, w3, 32 * 25 * sizeof(float));

	cudaMemcpyToSymbol(biases_conv1_constant, b1, sizeof(double) * 64);
	cudaMemcpyToSymbol(biases_conv2_constant, b2, sizeof(double) * 32);
	cudaMemcpyToSymbol(biases_conv3_constant, b3, sizeof(double) * 1);

	cuda_Filter2D_texture << < grid, block >> >(bicuY_cuda, conv1_data_cuda, HIgh_h, HIgh_w, 1, 9, 9, 64, b1);
	cuda_Filter2D2_texture << < grid, block >> >(conv1_data_cuda, conv2_data_cuda, HIgh_h, HIgh_w, 64, 5, 5, 32, b2);
	cuda_Filter2D_Last_texture << < grid, block >> >(conv2_data_cuda, conv3_data_cuda, HIgh_h, HIgh_w, 32, 5, 5, 1, b3);
}

void gpu_rgb2ycbcr(double* oriR_cuda, double* oriG_cuda, double* oriB_cuda, double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h) {
	dim3 block1;
	dim3 grid1;
	dim3 block;
	dim3 grid;

	block1.x = blocksize;
	block1.y = blocksize;

	grid1.x = Low_h / block1.x;
	grid1.y = Low_w / block1.y;


	block.x = blocksize;
	block.y = blocksize;

	grid.x = HIgh_h / block.x;
	grid.y = HIgh_w / block.y;

	cuda_rgb2ycbcr << < grid1, block1 >> >(oriR_cuda, oriG_cuda, oriB_cuda, oriY_cuda, oriCb_cuda, oriCr_cuda, h, w);
}

void gpu_ycbcr2rgb(double* conv3_data_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* output_R_cuda, double* output_G_cuda, double* output_B_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h) {
	dim3 block1;
	dim3 grid1;
	dim3 block;
	dim3 grid;

	block1.x = blocksize;
	block1.y = blocksize;

	grid1.x = Low_h / block1.x;
	grid1.y = Low_w / block1.y;


	block.x = blocksize;
	block.y = blocksize;

	grid.x = HIgh_h / block.x;
	grid.y = HIgh_w / block.y;

	cuda_ycbcr2rgb << < grid, block >> >(conv3_data_cuda, bicuCb_cuda, bicuCr_cuda, output_R_cuda, output_G_cuda, output_B_cuda, h, w);
}

void gpu_bicubic(double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda, double* bicuY_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* bicuYtem_cuda, double* bicuCbtem_cuda, double* bicuCrtem_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h) {
	dim3 block1;
	dim3 grid1;
	dim3 block;
	dim3 grid;

	block1.x = blocksize;
	block1.y = blocksize;

	grid1.x = Low_h / block1.x;
	grid1.y = Low_w / block1.y;


	block.x = blocksize;
	block.y = blocksize;

	grid.x = HIgh_h / block.x;
	grid.y = HIgh_w / block.y;

	cuda_bicubic_temp << < grid1, block1 >> >(oriY_cuda, oriCb_cuda, oriCr_cuda, bicuY_cuda, bicuCb_cuda, bicuCr_cuda, bicuYtem_cuda, bicuCbtem_cuda, bicuCrtem_cuda, Low_h, Low_w, Low_h);
	cuda_bicubic_last << < grid1, block1 >> >(oriY_cuda, oriCb_cuda, oriCr_cuda, bicuY_cuda, bicuCb_cuda, bicuCr_cuda, bicuYtem_cuda, bicuCbtem_cuda, bicuCrtem_cuda, Low_h, Low_w, Low_w, Low_h, HIgh_h);
}

void unbindTexture() {
	cudaUnbindTexture(texture_weight1);
	cudaUnbindTexture(texture_weight2);
	cudaUnbindTexture(texture_weight3);
}

void all(float* weight_conv1_float_cuda, float* weight_conv1_float, float* weight_conv2_float_cuda, float* weight_conv2_float, float* weight_conv3_float_cuda, float* weight_conv3_float, double* biases_conv1_cuda, double* biases_conv1, double* biases_conv2_cuda, double* biases_conv2, double* biases_conv3_cuda, double* biases_conv3,
	double* oriR, double* oriG, double* oriB, double* oriR_cuda, double* oriG_cuda, double* oriB_cuda, double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda,
	double* bicuY_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* bicuYtem_cuda, double* bicuCbtem_cuda, double* bicuCrtem_cuda, double* conv1_data_cuda, double* conv2_data_cuda, double* conv3_data_cuda,
	double* output_R, double* output_R_cuda, double* output_G, double* output_G_cuda, double* output_B, double* output_B_cuda, int Low_w, int Low_h, int HIgh_w, int HIgh_h) {

	dim3 block1;
	dim3 grid1;
	dim3 block;
	dim3 grid;

	block1.x = blocksize;
	block1.y = blocksize;

	grid1.x = Low_h / block1.x;
	grid1.y = Low_w / block1.y;


	block.x = blocksize;
	block.y = blocksize;

	grid.x = HIgh_h / block.x;
	grid.y = HIgh_w / block.y;

	(cudaMemcpy(weight_conv1_float_cuda, weight_conv1_float, 81 * 64 * sizeof(float), cudaMemcpyHostToDevice));
	cudaBindTexture(0, texture_weight1, weight_conv1_float_cuda, 81 * 64 * sizeof(float));

	(cudaMemcpy(weight_conv2_float_cuda, weight_conv2_float, 64 * 25 * 32 * sizeof(float), cudaMemcpyHostToDevice));
	cudaBindTexture(0, texture_weight2, weight_conv2_float_cuda, 64 * 25 * 32 * sizeof(float));

	(cudaMemcpy(weight_conv3_float_cuda, weight_conv3_float, 32 * 25 * sizeof(float), cudaMemcpyHostToDevice));
	cudaBindTexture(0, texture_weight3, weight_conv3_float_cuda, 32 * 25 * sizeof(float));



	(cudaMemcpy(biases_conv1_cuda, biases_conv1, 64 * sizeof(double), cudaMemcpyHostToDevice));
	(cudaMemcpy(biases_conv2_cuda, biases_conv2, 32 * sizeof(double), cudaMemcpyHostToDevice));
	(cudaMemcpy(biases_conv3_cuda, biases_conv3, 1 * sizeof(double), cudaMemcpyHostToDevice));

	cudaMemcpyToSymbol(biases_conv1_constant, biases_conv1, sizeof(double) * 64);
	cudaMemcpyToSymbol(biases_conv2_constant, biases_conv2, sizeof(double) * 32);
	cudaMemcpyToSymbol(biases_conv3_constant, biases_conv3, sizeof(double) * 1);


	(cudaMemcpy(oriR_cuda, oriR, Low_h * Low_w * sizeof(double), cudaMemcpyHostToDevice));
	(cudaMemcpy(oriG_cuda, oriG, Low_h * Low_w * sizeof(double), cudaMemcpyHostToDevice));
	(cudaMemcpy(oriB_cuda, oriB, Low_h * Low_w * sizeof(double), cudaMemcpyHostToDevice));

	cuda_rgb2ycbcr << < grid1, block1 >> >(oriR_cuda, oriG_cuda, oriB_cuda, oriY_cuda, oriCb_cuda, oriCr_cuda, Low_h, Low_w);


	cuda_bicubic_temp << < grid1, block1 >> >(oriY_cuda, oriCb_cuda, oriCr_cuda, bicuY_cuda, bicuCb_cuda, bicuCr_cuda, bicuYtem_cuda, bicuCbtem_cuda, bicuCrtem_cuda, Low_h, Low_w, Low_h);
	cuda_bicubic_last << < grid1, block1 >> >(oriY_cuda, oriCb_cuda, oriCr_cuda, bicuY_cuda, bicuCb_cuda, bicuCr_cuda, bicuYtem_cuda, bicuCbtem_cuda, bicuCrtem_cuda, Low_h, Low_w, Low_w, Low_h, HIgh_h);




	cuda_Filter2D_texture << < grid, block >> >(bicuY_cuda, conv1_data_cuda, HIgh_h, HIgh_w, 1, 9, 9, 64, biases_conv1_cuda);
	cuda_Filter2D2_texture << < grid, block >> >(conv1_data_cuda, conv2_data_cuda, HIgh_h, HIgh_w, 64, 5, 5, 32, biases_conv2_cuda);
	cuda_Filter2D_Last_texture << < grid, block >> >(conv2_data_cuda, conv3_data_cuda, HIgh_h, HIgh_w, 32, 5, 5, 1, biases_conv3_cuda);



	cuda_ycbcr2rgb << < grid, block >> >(conv3_data_cuda, bicuCb_cuda, bicuCr_cuda, output_R_cuda, output_G_cuda, output_B_cuda, HIgh_w, HIgh_h);


	(cudaMemcpy(output_R, output_R_cuda, HIgh_h*HIgh_w * sizeof(double), cudaMemcpyDeviceToHost));
	(cudaMemcpy(output_G, output_G_cuda, HIgh_h*HIgh_w * sizeof(double), cudaMemcpyDeviceToHost));
	(cudaMemcpy(output_B, output_B_cuda, HIgh_h*HIgh_w * sizeof(double), cudaMemcpyDeviceToHost));
}