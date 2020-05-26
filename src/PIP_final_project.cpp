#include "PIP_final_project.h"
#include <QtWidgets/QApplication>
#include <QPushButton>
#include <iostream>
#include "atlstr.h"
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qspinbox.h>
#include <io.h>
#include <string>
#include <stdio.h>
#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <qlistwidget.h>
#include <qtextstream.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#pragma warning(disable:4996)

using namespace std;
using namespace cv;

#define Layer_CH_Num 64
#define Layer20_CH_Num 1
#define Layer_Filter_Size 3
string input_path = "";
string Input_File_Name = "";

int Low_h;
int Low_w;

int HIgh_h;
int HIgh_w;

int half_filter_size = (int)(Layer_Filter_Size / 2);

#define __first 1
#define __first1 0

extern "C" void Convolutional_layer1(double* bicubic_Y, double* (&conv1_result), double* weight_conv1, double* bias_conv1, int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num, double& TestTime_total_CUDA);
extern "C" void Convolutional_layer_2_to_19(int Layer_num, double* Prior_conv_result, double* total_dummy, double* (&conv_result), double* weight_conv, double* bias_conv, int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num, double& TestTime_total_CUDA);
extern "C" void Convolutional_layer20(double* conv19_result, double* (&conv20_data), double* weight_conv20, double* bias_conv20, int HIgh_w, int HIgh_h, int Filter_Size, int CH_Num, int Layer_20_CH_Num, double& TestTime_total_CUDA);

double TestTime_total_CUDA = 0;

extern "C" void gpu_bicubic(double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda, double* bicuY_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* bicuYtem_cuda, double* bicuCbtem_cuda, double* bicuCrtem_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void gpu_ycbcr2rgb(double* conv3_data_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* output_R_cuda, double* output_G_cuda, double* output_B_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void gpu_rgb2ycbcr(double* oriR_cuda, double* oriG_cuda, double* oriB_cuda, double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda, int h, int w, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void gpu_srcnn(double* bicuY_cuda, double* conv1_data_cuda, double* conv2_data_cuda, double* conv3_data_cuda, float* w1, float* w2, float* w3, double* b1, double* b2, double* b3, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void all(float* weight_conv1_float_cuda, float* weight_conv1_float, float* weight_conv2_float_cuda, float* weight_conv2_float, float* weight_conv3_float_cuda, float* weight_conv3_float, double* biases_conv1_cuda, double* biases_conv1, double* biases_conv2_cuda, double* biases_conv2, double* biases_conv3_cuda, double* biases_conv3,
	double* oriR, double* oriG, double* oriB, double* oriR_cuda, double* oriG_cuda, double* oriB_cuda, double* oriY_cuda, double* oriCb_cuda, double* oriCr_cuda,
	double* bicuY_cuda, double* bicuCb_cuda, double* bicuCr_cuda, double* bicuYtem_cuda, double* bicuCbtem_cuda, double* bicuCrtem_cuda, double* conv1_data_cuda, double* conv2_data_cuda, double* conv3_data_cuda,
	double* output_R, double* output_R_cuda, double* output_G, double* output_G_cuda, double* output_B, double* output_B_cuda, int Low_w, int Low_h, int HIgh_w, int HIgh_h);
extern "C" void unbindTexture();


QString input_file_path = "C:";	// input file path
QString output_dir_path = "C:";	// output directory path



//(const Mat& Image1, const Mat& Image2) {
//
//	const Mat I1 = Image1(cv::Rect(10, 10, Image1.cols - 10, Image1.rows - 10));
//	const Mat I2 = Image2(cv::Rect(10, 10, Image2.cols - 10, Image2.rows - 10));

double getPSNR(const Mat& Image1, const Mat& Image2){
	Mat RGB2 = Image2.clone();
	Mat RGB1 = Image1.clone();
	cvtColor(RGB2, RGB2, cv::COLOR_BGR2GRAY);
	cvtColor(RGB1, RGB1, cv::COLOR_RGB2GRAY);
	int skip = 20;
	const Mat img_ref = RGB1(cv::Rect(skip, skip, RGB1.cols - skip, RGB1.rows - skip));
	const Mat img_interp = RGB2(cv::Rect(skip, skip, RGB2.cols - skip, RGB2.rows - skip));

	Mat img_ref_bgr[3];
	Mat img_interp_bgr[3];

	/*img_ref.convertTo(img_ref, CV_32F);
	img_interp.convertTo(img_interp, CV_32F);*/

	//split(img_ref, img_ref_bgr);
	//split(img_interp, img_interp_bgr);

	

	Mat diff;
	absdiff(img_ref, img_interp, diff);       // |I1 - I2|
	

	diff = diff.mul(diff);

	double mse = sum(diff)[0] / (diff.cols * diff.rows);

	//Mat diff[3];
	//absdiff(img_ref_bgr[0], img_interp_bgr[0], diff[0]);       // |I1 - I2|
	//absdiff(img_ref_bgr[1], img_interp_bgr[1], diff[1]);
	//absdiff(img_ref_bgr[2], img_interp_bgr[2], diff[2]);

	//diff[0] = diff[0].mul(diff[0]);
	//diff[1] = diff[1].mul(diff[1]);
	//diff[2] = diff[2].mul(diff[2]);

	//double mseB = sum(diff[0])[0] / (diff[0].cols * diff[0].rows);
	//double mseG = sum(diff[1])[0] / (diff[0].cols * diff[0].rows);
	//double mseR = sum(diff[2])[0] / (diff[0].cols * diff[0].rows);

	//double mse = (mseR + mseG + mseB) / 3;


	/*
	double minVal, maxVal;
	minMaxLoc(diff, &minVal, &maxVal);

	diff.convertTo(diff, CV_32F);  // cannot make a square on 8 bits
	diff = diff.mul(diff);           // |I1 - I2|^2

	Scalar s = sum(diff);         // sum elements per channel

	double sse = (s.val[0] + s.val[1] + s.val[2]) / 3; // sum channels

	double psnr;
	if (sse <= 1e-10) // for small values return zero
		psnr = 0;
	else{
		//double  mse = sse / (double)(img_ref.channels() * img_ref.total());*/
	double psnr = 10.0*log10((255 * 255) / mse);
	return psnr;
	//}
}

PIP_final_project::PIP_final_project(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}


// ConHigh_ht OpenCV::Mat -> Qt:QImage
QImage PIP_final_project::convertOpenCVMatToQtQImage(cv::Mat mat) {
	if (mat.channels() == 1) {                  // if grayscale image
		return QImage((uchar*)mat.data, (int)mat.cols, (int)mat.rows, (int)mat.step, QImage::Format_Indexed8);     // declare and return a QImage
	}
	else if (mat.channels() == 3) {             // if 3 channel color image
		cv::cvtColor(mat, mat, CV_BGR2RGB);     // inHigh_ht BGR to RGB
		return QImage((uchar*)mat.data, (int)mat.cols, (int)mat.rows, (int)mat.step, QImage::Format_RGB888);       // declare and return a QImage
	}

	return QImage();  // return a blank QImage if the above did not work
}

// Select Input File to Qfiledialog && Display Original Image
void PIP_final_project::selectInputFile() {
	input_file_path = QFileDialog::getOpenFileName(
		this,
		tr("Select Src Image File"),
		"C://",
		"All files(*.*)"
	);
	if (input_file_path > 0) {
		QMessageBox::information(this, tr("Input File Name"), input_file_path);
	}
	else {
		QMessageBox::information(this, tr("Input File Name"), "Cancled");
	}

	// Display Input Box
	ui.label_2->setText(input_file_path);

	// Display Original Image
	input_path = input_file_path.toLocal8Bit().constData();
	string img_path = input_path;

	//Regist Input_File_Name
	Input_File_Name = input_path;

	char *Input_path_char_pointer = new char[Input_File_Name.size() + 1];
	std::copy(Input_File_Name.begin(), Input_File_Name.end(), Input_path_char_pointer);
	Input_path_char_pointer[Input_File_Name.size()] = '\0';

	char* result = strtok(Input_path_char_pointer, "/");
	while (result != NULL) {
		Input_File_Name = result;
		result = strtok(NULL, "/");
	}

	// Raw Image
	Mat Origianl_image = imread(img_path, 1);

	// Display output
	QImage qimgSerial = convertOpenCVMatToQtQImage(Origianl_image);
	ui.label->setPixmap(QPixmap::fromImage(qimgSerial.scaled(280, 280, Qt::KeepAspectRatio)));

	//Regist input_path of half Image 
	//ui.listWidget->addItem(QString::fromStdString(input_path));
	input_path = input_path.substr(0, input_path.size() - Input_File_Name.size()) + "Half_" + Input_File_Name;

	Mat Origianl_image_half = Origianl_image.clone();
	cv::resize(Origianl_image_half, Origianl_image_half, cv::Size((int)(Origianl_image_half.cols / 2), (int)(Origianl_image_half.rows / 2)), 0, 0);//, CV_INTER_CUBIC
	cv::cvtColor(Origianl_image_half, Origianl_image_half, CV_RGB2BGR);
	imwrite(input_path, Origianl_image_half);
	//ui.listWidget->addItem(QString::fromStdString(input_path));
}


// Select Output Directory to Qfiledialog
void PIP_final_project::selectOutputDir() {
	output_dir_path = QFileDialog::getExistingDirectory(
		this,
		tr("Select Directory"),
		"C://"
	);
	if (output_dir_path > 0) {
		QMessageBox::information(this, tr("Output Directory Name"), output_dir_path);
	}
	else {
		QMessageBox::information(this, tr("Output Directory Name"), "Cancled");
	}

	// Display Onput Directory
	ui.label_3->setText(output_dir_path);
}

// ============================ Processing & Display ============================
// Bicubic_Interpolation
void PIP_final_project::slotDisplay_Bicubic_Interpolation() {
	int64 tStart, tEnd, tStart_total, tEnd_total;
	double TestTime, TestTime_total;

	ui.listWidget->addItem("====== Bicubic Interpolation ======");

	//load image using opencv
	IplImage* Input = cvLoadImage(input_path.c_str(), CV_LOAD_IMAGE_COLOR);

	Low_w = Input->width;
	Low_h = Input->height;

	HIgh_w = Low_w * 2;
	HIgh_h = Low_h * 2;


	// R, G, B channel
	IplImage* r = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //R channel
	IplImage* g = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //G channel
	IplImage* b = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //B channel

	int i, j, k, l;

	double bicubic2[2][4] = { { -0.0234375, 0.2265625, 0.8671875, -0.0703125 },
	{ -0.0703125, 0.8671875, 0.2265625, -0.0234375 } };

	// Input RGB
	double* input_R; double* input_G; double* input_B;

	// Output RGB
	double* output_R; double* output_G; double* output_B;

	// Input YCbCr
	double* input_Y; double* input_Cb; double* input_Cr;

	// Bicubic YCbCr
	double* bicubic_Y;		double* bicubic_Cb;		 double* bicubic_Cr;
	double* bicubic_Y_temp; double* bicubic_Cb_temp; double* bicubic_Cr_temp;

	// Allocation
	/// Low Image
	input_R = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_G = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_B = (double *)malloc(sizeof(double)*Low_h*Low_w);

	input_Y = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_Cb = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_Cr = (double *)malloc(sizeof(double)*Low_h*Low_w);

	/// High Image
	bicubic_Y = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicubic_Cb = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicubic_Cr = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	bicubic_Y_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicubic_Cb_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicubic_Cr_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);


	ui.listWidget->addItem("Initialization : DONE");


	// Input Split to (IplImage*) r, g, b
	cvSplit(Input, b, g, r, NULL);
	tStart = cvGetTickCount();
	// (IplImage*)r, g, b -> (double*)input_R, G, B
	for (i = 0; i < Low_h; i++) {
		for (j = 0; j < Low_w; j++) {//
			input_R[i*Low_w + j] = cvGetReal2D(r, i, j);//unsigned char(r->imageData[i*Low_w + j]);
			input_G[i*Low_w + j] = cvGetReal2D(g, i, j);//unsigned char(g->imageData[i*Low_w + j]);
			input_B[i*Low_w + j] = cvGetReal2D(b, i, j);//unsigned char(b->imageData[i*Low_w + j]);
		}
	}

	// ConHigh_ht RGB to YCbCr 
	// (double*)input_R, G, B -> (double*)input_Y, Cb, Cr
	for (j = 0; j < Low_w; j++) {
		for (i = 0; i < Low_h; i++) {
			input_Y[i*Low_w + j] = 0.256788235294118*input_R[i*Low_w + j] + 0.504129411764706*input_G[i*Low_w + j] + 0.0979058823529412*input_B[i*Low_w + j] + 16;
			input_Cb[i*Low_w + j] = -0.148223529411765*input_R[i*Low_w + j] - 0.290992156862745*input_G[i*Low_w + j] + 0.4392156862745100*input_B[i*Low_w + j] + 128;
			input_Cr[i*Low_w + j] = 0.439215686274510*input_R[i*Low_w + j] - 0.367788235294118*input_G[i*Low_w + j] - 0.0714274509803922*input_B[i*Low_w + j] + 128;
		}
	}

	ui.listWidget->addItem("RGB to YCbCr : DONE");


	// Bicubic Interpolation
	for (i = 2; i <= Low_h - 2; i++) { // High_wizontal
		for (j = 0; j < Low_w; j++) {
			bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
			bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

			bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
			bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

			bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
		}
	}
	for (i = 0; i < HIgh_h; i++) { // High_htical
		for (j = 2; j <= Low_w - 2; j++) {
			bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
			bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

			bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
			bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

			bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////

	// Interpolation output
	double* interpolation_R;
	double* interpolation_G;
	double* interpolation_B;


	// Initialization interpolation R, G, B
	interpolation_R = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	interpolation_G = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	interpolation_B = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	// (double*)Bicubic_Y, Cb, Cr -> (double*)Interpolation_R, G, B
	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			interpolation_R[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] + 0.00000030112439741101*bicubic_Cb[i*HIgh_w + j] + 1.59602688733570000000*bicubic_Cr[i*HIgh_w + j] - 222.921617109194 + 0.5);
			interpolation_G[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] - 0.39176253994145000000*bicubic_Cb[i*HIgh_w + j] - 0.81296829216220500000*bicubic_Cr[i*HIgh_w + j] + 135.575409522967 + 0.5);
			interpolation_B[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] + 2.01723263955646000000*bicubic_Cb[i*HIgh_w + j] + 0.00000305426174524847*bicubic_Cr[i*HIgh_w + j] - 276.836305795032 + 0.5);
		}
	}

	// (double*)Interpolation_R, G, B -> (IplImage*)interpolationImage_R, G, B
	CvSize cvsize2 = { HIgh_w, HIgh_h };
	IplImage* interpolationImage1 = cvCreateImage(cvsize2, IPL_DEPTH_8U, 3);
	IplImage* interpolationImage_R = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);
	IplImage* interpolationImage_G = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);
	IplImage* interpolationImage_B = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);

	for (int y = 0; y < cvsize2.height; y++) {
		for (int x = 0; x < cvsize2.width; x++) {
			//cvSetReal2D(TempImage1, y, x, conv3_data[ y*cvsize1.width+x]);
			cvSetReal2D(interpolationImage_R, y, x, interpolation_R[y*cvsize2.width + x]);
			cvSetReal2D(interpolationImage_G, y, x, interpolation_G[y*cvsize2.width + x]);
			cvSetReal2D(interpolationImage_B, y, x, interpolation_B[y*cvsize2.width + x]);
		}
	}

	// Merge: (IplImage*)interpolationImage_R, G, B -> (IplImage*)interpolationImage1(RGB)
	cvMerge(interpolationImage_B, interpolationImage_G, interpolationImage_R, NULL, interpolationImage1);

	// Output Path
	string out_path = output_dir_path.toLocal8Bit().constData();
	out_path.append("/output_Bicubic_Interpolation_");
	out_path.append(Input_File_Name);

	// Create char * output path
	char * Output_Path_char_pointer;
	Output_Path_char_pointer = new char[out_path.size() + 1];
	std::copy(out_path.begin(), out_path.end(), Output_Path_char_pointer);
	Output_Path_char_pointer[out_path.size()] = '\0';

	// Store Image
	cvSaveImage(Output_Path_char_pointer, interpolationImage1);
	delete[] Output_Path_char_pointer;

	// Display Image
	// ConHigh_ht IplImage* to Mat Image
	Mat bicubic_interpolation = cvarrToMat(interpolationImage1);//Input

																// Display output
	QImage qimgSerial = convertOpenCVMatToQtQImage(bicubic_interpolation);
	ui.label_9->setPixmap(QPixmap::fromImage(qimgSerial.scaled(280, 280, Qt::KeepAspectRatio)));

	free(interpolation_R);
	free(interpolation_G);
	free(interpolation_B);

	cvReleaseImage(&interpolationImage_R);
	cvReleaseImage(&interpolationImage_G);
	cvReleaseImage(&interpolationImage_B);

	ui.listWidget->addItem("Bicubic Interpolation : Done");

	free(input_R); free(input_G); free(input_B);
	free(input_Y); free(input_Cb); free(input_Cr);
	free(bicubic_Y_temp); free(bicubic_Cb_temp); free(bicubic_Cr_temp);

	cvReleaseImage(&r); cvReleaseImage(&g); cvReleaseImage(&b);

	tEnd = cvGetTickCount();// for check processing
	TestTime_total = 0.001 * (tEnd - tStart) / cvGetTickFrequency(); // for msec

																	 // Display Input Box
	ui.label_20->setText(QString::number(TestTime_total) + " ms  ");

	ui.listWidget->addItem("TestTime_total : " + QString::number(TestTime_total));

	Mat output_Image = cvarrToMat(interpolationImage1);
	Mat Origianl_image = imread(input_file_path.toLocal8Bit().constData(), 1);
	double PSNR = getPSNR(Origianl_image, output_Image);
	ui.label_17->setText(QString::number(PSNR) + " dB  ");
}

// Processing & Display 
void PIP_final_project::slotDisplay_Srcnn_CUDA() {
	double TestTime_total_SRCNN = 0;
	ui.listWidget->addItem("========= SRCNN CUDA =========");
	float tStart, tEnd;
	double ProTime;


	IplImage* pInput = cvLoadImage(input_path.c_str(), CV_LOAD_IMAGE_COLOR); //load image using opencv
	IplImage* r = cvCreateImage(cvGetSize(pInput), IPL_DEPTH_8U, 1);    //R channel
	IplImage* g = cvCreateImage(cvGetSize(pInput), IPL_DEPTH_8U, 1);    //G channel
	IplImage* b = cvCreateImage(cvGetSize(pInput), IPL_DEPTH_8U, 1);    //B channel

	Low_h = pInput->width;
	Low_w = pInput->height;

	HIgh_w = Low_w * 2;
	HIgh_h = Low_h * 2;

	int i, j, k, l;
	double* weight_conv1; double* weight_conv2; double* weight_conv3;

	float* weight_conv1_float; float* weight_conv2_float; float* weight_conv3_float;

	double* weight_conv1_cuda; double* weight_conv2_cuda; double* weight_conv3_cuda;

	float* weight_conv1_float_cuda; float* weight_conv2_float_cuda; float* weight_conv3_float_cuda;

	double* biase_conv1; double* biase_conv2; double* biase_conv3;

	double* biase_conv1_cuda; double* biase_conv2_cuda; double* biase_conv3_cuda;

	double* input;
	double* R; double* G; double* B;
	double* conv1_data; double* conv2_data; double* conv3_data;

	double* temp_data;
	double* temp_data2;

	double* input_cuda;
	double* R_cuda; double* G_cuda; double* B_cuda;
	double* conv1_data_cuda; double* conv2_data_cuda; double* conv3_data_cuda;

	double* temp_data_cuda;
	double* temp_data2_cuda;


	double *oriR; double *oriG; double *oriB;
	double *oriY; double *oriCb; double *oriCr;
	double *bicuY; double *bicuCb; double *bicuCr;
	double *bicuYtem; double *bicuCbtem; double *bicuCrtem;

	double *oriInput_cuda;
	double *oriR_cuda; double *oriG_cuda; double *oriB_cuda;
	double *oriY_cuda; double *oriCb_cuda; double *oriCr_cuda;
	double *bicuY_cuda; double *bicuCb_cuda; double *bicuCr_cuda;
	double *bicuYtem_cuda; double *bicuCbtem_cuda; double *bicuCrtem_cuda;

	double* output_R; double* output_G; double* output_B;

	double* output_R_cuda; double* output_G_cuda; double* output_B_cuda;

	double temp_filter[25];
	double temp_block[25];
	double temp_sum, tempsum2;

	FILE *in;

	weight_conv1 = (double *)malloc(sizeof(double) * 81 * 64);
	weight_conv2 = (double *)malloc(sizeof(double) * 64 * 25 * 32);
	weight_conv3 = (double *)malloc(sizeof(double) * 32 * 25);

	weight_conv1_float = (float *)malloc(sizeof(float) * 81 * 64);
	weight_conv2_float = (float *)malloc(sizeof(float) * 64 * 25 * 32);
	weight_conv3_float = (float *)malloc(sizeof(float) * 32 * 25);

	biase_conv1 = (double *)malloc(sizeof(double) * 64);
	biase_conv2 = (double *)malloc(sizeof(double) * 32);
	biase_conv3 = (double *)malloc(sizeof(double) * 1);

	//(cudaMalloc((void**)&pcuCr, w * h * sizeof(double)));


	cudaMalloc((void**)&weight_conv1_cuda, sizeof(double) * 81 * 64);
	cudaMalloc((void**)&weight_conv2_cuda, sizeof(double) * 64 * 25 * 32);
	cudaMalloc((void**)&weight_conv3_cuda, sizeof(double) * 32 * 25);

	cudaMalloc((void**)&weight_conv1_float_cuda, sizeof(float) * 81 * 64);
	cudaMalloc((void**)&weight_conv2_float_cuda, sizeof(float) * 64 * 25 * 32);
	cudaMalloc((void**)&weight_conv3_float_cuda, sizeof(float) * 32 * 25);

	cudaMalloc((void**)&biase_conv1_cuda, sizeof(double) * 64);
	cudaMalloc((void**)&biase_conv2_cuda, sizeof(double) * 32);
	cudaMalloc((void**)&biase_conv3_cuda, sizeof(double) * 1);



	conv1_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w * 64);
	conv2_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w * 32);
	conv3_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w * 1);
	temp_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w * 64);
	temp_data2 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w * 32);

	cudaMalloc((void**)&conv1_data_cuda, sizeof(double)*HIgh_h*HIgh_w * 64);
	cudaMalloc((void**)&conv2_data_cuda, sizeof(double)*HIgh_h*HIgh_w * 32);
	cudaMalloc((void**)&conv3_data_cuda, sizeof(double)*HIgh_h*HIgh_w * 1);
	cudaMalloc((void**)&temp_data_cuda, sizeof(double)*HIgh_h*HIgh_w * 64);
	cudaMalloc((void**)&temp_data2_cuda, sizeof(double)*HIgh_h*HIgh_w * 32);


	input = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	R = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	G = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	B = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	output_R = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	output_G = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	output_B = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);


	cudaMalloc((void**)&input_cuda, sizeof(double)*HIgh_h*HIgh_w);
	cudaMalloc((void**)&R_cuda, sizeof(double)*HIgh_h*HIgh_w);
	cudaMalloc((void**)&G_cuda, sizeof(double)*HIgh_h*HIgh_w);
	cudaMalloc((void**)&B_cuda, sizeof(double)*HIgh_h*HIgh_w);

	cudaMalloc((void**)&output_R_cuda, sizeof(double)*HIgh_h*HIgh_w);
	cudaMalloc((void**)&output_G_cuda, sizeof(double)*HIgh_h*HIgh_w);
	cudaMalloc((void**)&output_B_cuda, sizeof(double)*HIgh_h*HIgh_w);

	//original color image split to three channel 
	oriR = (double *)malloc(sizeof(double)*Low_h*Low_w);
	oriG = (double *)malloc(sizeof(double)*Low_h*Low_w);
	oriB = (double *)malloc(sizeof(double)*Low_h*Low_w);

	cudaMalloc((void**)&oriR_cuda, sizeof(double)*Low_h*Low_w);
	cudaMalloc((void**)&oriG_cuda, sizeof(double)*Low_h*Low_w);
	cudaMalloc((void**)&oriB_cuda, sizeof(double)*Low_h*Low_w);
	//original rgb channel change to YCbCr channel
	oriY = (double *)malloc(sizeof(double)*Low_h*Low_w);
	oriCb = (double *)malloc(sizeof(double)*Low_h*Low_w);
	oriCr = (double *)malloc(sizeof(double)*Low_h*Low_w);

	cudaMalloc((void**)&oriY_cuda, sizeof(double)*Low_h*Low_w);
	cudaMalloc((void**)&oriCb_cuda, sizeof(double)*Low_h*Low_w);
	cudaMalloc((void**)&oriCr_cuda, sizeof(double)*Low_h*Low_w);
	//y CB Cr channel for inteoploation 
	bicuY = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicuCb = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicuCr = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	cudaMalloc((void**)&bicuY_cuda, sizeof(double)*HIgh_h*HIgh_w);
	cudaMalloc((void**)&bicuCb_cuda, sizeof(double)*HIgh_h*HIgh_w);
	cudaMalloc((void**)&bicuCr_cuda, sizeof(double)*HIgh_h*HIgh_w);

	bicuYtem = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicuCbtem = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicuCrtem = (double *)malloc(sizeof(double)*HIgh_h*Low_w);

	cudaMalloc((void**)&bicuYtem_cuda, sizeof(double)*HIgh_h*Low_w);
	cudaMalloc((void**)&bicuCbtem_cuda, sizeof(double)*HIgh_h*Low_w);
	cudaMalloc((void**)&bicuCrtem_cuda, sizeof(double)*HIgh_h*Low_w);


	cvSplit(pInput, b, g, r, NULL);
	tStart = cvGetTickCount();
	for (j = 0; j < Low_w; j++) {
		for (i = 0; i < Low_h; i++) {
			oriR[i*Low_w + j] = cvGetReal2D(r, i, j);//unsigned char(r->imageData[i*Low_w + j]);unsigned char(r->imageData[i*Low_w + j]);
			oriG[i*Low_w + j] = cvGetReal2D(g, i, j);//unsigned char(g->imageData[i*Low_w + j]);unsigned char(g->imageData[i*Low_w + j]);
			oriB[i*Low_w + j] = cvGetReal2D(b, i, j);//unsigned char(b->imageData[i*Low_w + j]);unsigned char(b->imageData[i*Low_w + j]); 
		}
	}

	in = fopen("Input/filter/SRCNN_weight_conv1.raw", "rb");
	fread(weight_conv1, sizeof(double), 1 * 81 * 64, in);
	fclose(in);

	in = fopen("Input/filter/SRCNN_weight_conv2.raw", "rb");
	fread(weight_conv2, sizeof(double), 64 * 25 * 32, in);
	fclose(in);

	in = fopen("Input/filter/SRCNN_weight_conv3.raw", "rb");
	fread(weight_conv3, sizeof(double), 32 * 25 * 1, in);
	fclose(in);

	in = fopen("Input/filter/SRCNN_bias_conv1.raw", "rb");
	fread(biase_conv1, sizeof(double), 64, in);
	fclose(in);

	in = fopen("Input/filter/SRCNN_bias_conv2.raw", "rb");
	fread(biase_conv2, sizeof(double), 32, in);
	fclose(in);

	in = fopen("Input/filter/SRCNN_bias_conv3.raw", "rb");
	fread(biase_conv3, sizeof(double), 1, in);
	fclose(in);

	for (i = 0; i<1 * 81 * 64; i++) {
		weight_conv1_float[i] = (float)weight_conv1[i];
	}
	for (i = 0; i<64 * 25 * 32; i++) {
		weight_conv2_float[i] = (float)weight_conv2[i];
	}
	for (i = 0; i<32 * 25 * 1; i++) {
		weight_conv3_float[i] = (float)weight_conv3[i];
	}

	all(weight_conv1_float_cuda, weight_conv1_float, weight_conv2_float_cuda, weight_conv2_float, weight_conv3_float_cuda, weight_conv3_float, biase_conv1_cuda, biase_conv1, biase_conv2_cuda, biase_conv2, biase_conv3_cuda, biase_conv3,
		oriR, oriG, oriB, oriR_cuda, oriG_cuda, oriB_cuda, oriY_cuda, oriCb_cuda, oriCr_cuda,
		bicuY_cuda, bicuCb_cuda, bicuCr_cuda, bicuYtem_cuda, bicuCbtem_cuda, bicuCrtem_cuda, conv1_data_cuda, conv2_data_cuda, conv3_data_cuda,
		output_R, output_R_cuda, output_G, output_G_cuda, output_B, output_B_cuda, Low_w, Low_h, HIgh_w, HIgh_h);


	CvSize cvsize1 = { HIgh_w, HIgh_h };
	//CvSize cvsize2 = { Low_w, Low_h };


	IplImage* TempImage4 = cvCreateImage(cvsize1, IPL_DEPTH_8U, 3);
	IplImage* TempImage_R = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);
	IplImage* TempImage_G = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);
	IplImage* TempImage_B = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);

	for (int y = 0; y < cvsize1.height; y++) {
		for (int x = 0; x < cvsize1.width; x++) {

			cvSetReal2D(TempImage_R, y, x, output_R[y*cvsize1.width + x]);
			cvSetReal2D(TempImage_G, y, x, output_G[y*cvsize1.width + x]);
			cvSetReal2D(TempImage_B, y, x, output_B[y*cvsize1.width + x]);
		}
	}
	cvMerge(TempImage_B, TempImage_G, TempImage_R, NULL, TempImage4);


	// Output Path
	string out_path = output_dir_path.toLocal8Bit().constData();
	out_path.append("/output_SRCNN_CUDA_");
	out_path.append(Input_File_Name);

	// Create char * output path
	char * Output_Path_char_pointer;
	Output_Path_char_pointer = new char[out_path.size() + 1];
	std::copy(out_path.begin(), out_path.end(), Output_Path_char_pointer);
	Output_Path_char_pointer[out_path.size()] = '\0';

	// Store Image
	cvSaveImage(Output_Path_char_pointer, TempImage4);

	// Display Image
	// ConHigh_ht IplImage* to Mat Image
	Mat SRCNN_CUDA = cvarrToMat(TempImage4);

	// Display output
	QImage qimgSerial = convertOpenCVMatToQtQImage(SRCNN_CUDA);
	ui.label_4->setPixmap(QPixmap::fromImage(qimgSerial.scaled(280, 280, Qt::KeepAspectRatio)));

	tEnd = cvGetTickCount();// for check processing
	ProTime = 0.001 * (tEnd - tStart) / cvGetTickFrequency(); // for msec

															  // Display Input Box
	ui.label_22->setText(QString::number(ProTime) + " ms  ");

	Mat output_Image = cvarrToMat(TempImage4);
	Mat Origianl_image = imread(input_file_path.toLocal8Bit().constData(), 1);
	double PSNR = getPSNR(Origianl_image, output_Image);
	ui.label_24->setText(QString::number(PSNR) + " dB  ");
}


// Processing & Display 
void PIP_final_project::slotDisplay_VDSR_Serial() {
	double TestTime_total_Serial = 0;
	ui.listWidget->addItem("========= VDSR Serial =========");
	//load image using opencv
	IplImage* Input = cvLoadImage(input_path.c_str(), CV_LOAD_IMAGE_COLOR);

	Low_w = Input->width;
	Low_h = Input->height;

	HIgh_w = Low_w * 2;
	HIgh_h = Low_h * 2;



	// R, G, B channel
	IplImage* r = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //R channel
	IplImage* g = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //G channel
	IplImage* b = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //B channel

	int i, j, k, l;
	int layer1_index, layer2_index, layer3_index, layer4_index, layer5_index, layer6_index, layer7_index, layer8_index, layer9_index, layer10_index;
	int layer11_index, layer12_index, layer13_index, layer14_index, layer15_index, layer16_index, layer17_index, layer18_index, layer19_index;


	double bicubic2[2][4] = { { -0.0234375, 0.2265625, 0.8671875, -0.0703125 },
	{ -0.0703125, 0.8671875, 0.2265625, -0.0234375 } };

	// Weight
	double* weight_conv1;  double* weight_conv2;  double* weight_conv3;  double* weight_conv4;  double* weight_conv5;
	double* weight_conv6;  double* weight_conv7;  double* weight_conv8;  double* weight_conv9;  double* weight_conv10;
	double* weight_conv11; double* weight_conv12; double* weight_conv13; double* weight_conv14; double* weight_conv15;
	double* weight_conv16; double* weight_conv17; double* weight_conv18; double* weight_conv19; double* weight_conv20;

	// Bias
	double* bias_conv1;  double* bias_conv2;  double* bias_conv3;  double* bias_conv4;  double* bias_conv5;
	double* bias_conv6;  double* bias_conv7;  double* bias_conv8;  double* bias_conv9;  double* bias_conv10;
	double* bias_conv11; double* bias_conv12; double* bias_conv13; double* bias_conv14; double* bias_conv15;
	double* bias_conv16; double* bias_conv17; double* bias_conv18; double* bias_conv19; double* bias_conv20;

	// Input RGB
	double* input_R; double* input_G; double* input_B;

	// Output RGB
	double* output_R; double* output_G; double* output_B;

	// Input YCbCr
	double* input_Y; double* input_Cb; double* input_Cr;

	// Bicubic YCbCr
	double* bicubic_Y;		double* bicubic_Cb;		 double* bicubic_Cr;
	double* bicubic_Y_temp; double* bicubic_Cb_temp; double* bicubic_Cr_temp;

	// Convolution data
	double* conv1_data;  double* conv2_data;  double* conv3_data;  double* conv4_data;  double* conv5_data;
	double* conv6_data;  double* conv7_data;  double* conv8_data;  double* conv9_data;  double* conv10_data;
	double* conv11_data; double* conv12_data; double* conv13_data; double* conv14_data; double* conv15_data;
	double* conv16_data; double* conv17_data; double* conv18_data; double* conv19_data; double* conv20_data;

	double* summation_data;

	// Temp data
	double* temp_data2;  double* temp_data3;  double* temp_data4;  double* temp_data5;
	double* temp_data6;  double* temp_data7;  double* temp_data8;  double* temp_data9;  double* temp_data10;
	double* temp_data11; double* temp_data12; double* temp_data13; double* temp_data14; double* temp_data15;
	double* temp_data16; double* temp_data17; double* temp_data18; double* temp_data19; double* temp_data20;

	FILE *in;

	// Allocation
	/// Low Image
	input_R = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_G = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_B = (double *)malloc(sizeof(double)*Low_h*Low_w);

	input_Y = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_Cb = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_Cr = (double *)malloc(sizeof(double)*Low_h*Low_w);

	/// High Image
	bicubic_Y = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicubic_Cb = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicubic_Cr = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	bicubic_Y_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicubic_Cb_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicubic_Cr_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);

	int64 tStart, tEnd, tStart_total, tEnd_total;
	double TestTime, TestTime_total;

	ui.listWidget->addItem("Initialization : DONE");


	// Input Split to (IplImage*) r, g, b
	cvSplit(Input, b, g, r, NULL);

	// (IplImage*)r, g, b -> (double*)input_R, G, B
	for (j = 0; j < Low_w; j++) {
		for (i = 0; i < Low_h; i++) {
			input_R[i*Low_w + j] = cvGetReal2D(r, i, j);//unsigned char(r->imageData[i*Low_w + j]);
			input_G[i*Low_w + j] = cvGetReal2D(g, i, j);//unsigned char(g->imageData[i*Low_w + j]);
			input_B[i*Low_w + j] = cvGetReal2D(b, i, j);//unsigned char(b->imageData[i*Low_w + j]);
		}
	}

	// ConHigh_ht RGB to YCbCr 
	// (double*)input_R, G, B -> (double*)input_Y, Cb, Cr
	for (j = 0; j < Low_w; j++) {
		for (i = 0; i < Low_h; i++) {
			input_Y[i*Low_w + j] = 0.256788235294118*input_R[i*Low_w + j] + 0.504129411764706*input_G[i*Low_w + j] + 0.0979058823529412*input_B[i*Low_w + j] + 16;
			input_Cb[i*Low_w + j] = -0.148223529411765*input_R[i*Low_w + j] - 0.290992156862745*input_G[i*Low_w + j] + 0.4392156862745100*input_B[i*Low_w + j] + 128;
			input_Cr[i*Low_w + j] = 0.439215686274510*input_R[i*Low_w + j] - 0.367788235294118*input_G[i*Low_w + j] - 0.0714274509803922*input_B[i*Low_w + j] + 128;
		}
	}

	ui.listWidget->addItem("RGB to YCbCr : DONE");

	// Bicubic Interpolation
	for (i = 2; i <= Low_h - 2; i++) { // High_wizontal
		for (j = 0; j < Low_w; j++) {
			bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
			bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

			bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
			bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

			bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
		}
	}
	for (i = 0; i < HIgh_h; i++) { // High_htical
		for (j = 2; j <= Low_w - 2; j++) {
			bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
			bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

			bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
			bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

			bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
		}
	}

	// Interpolation output
	double* interpolation_R;
	double* interpolation_G;
	double* interpolation_B;

	// Initialization interpolation R, G, B
	interpolation_R = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	interpolation_G = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	interpolation_B = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	// (double*)Bicubic_Y, Cb, Cr -> (double*)Interpolation_R, G, B
	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			interpolation_R[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] + 0.00000030112439741101*bicubic_Cb[i*HIgh_w + j] + 1.59602688733570000000*bicubic_Cr[i*HIgh_w + j] - 222.921617109194 + 0.5);
			interpolation_G[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] - 0.39176253994145000000*bicubic_Cb[i*HIgh_w + j] - 0.81296829216220500000*bicubic_Cr[i*HIgh_w + j] + 135.575409522967 + 0.5);
			interpolation_B[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] + 2.01723263955646000000*bicubic_Cb[i*HIgh_w + j] + 0.00000305426174524847*bicubic_Cr[i*HIgh_w + j] - 276.836305795032 + 0.5);
		}
	}

	// (double*)Interpolation_R, G, B -> (IplImage*)interpolationImage_R, G, B
	CvSize cvsize2 = { HIgh_w, HIgh_h };
	IplImage* interpolationImage_R = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);
	IplImage* interpolationImage_G = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);
	IplImage* interpolationImage_B = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);

	for (int y = 0; y < cvsize2.height; y++) {
		for (int x = 0; x < cvsize2.width; x++) {
			//cvSetReal2D(TempImage1, y, x, conv3_data[ y*cvsize1.width+x]);
			cvSetReal2D(interpolationImage_R, y, x, interpolation_R[y*cvsize2.width + x]);
			cvSetReal2D(interpolationImage_G, y, x, interpolation_G[y*cvsize2.width + x]);
			cvSetReal2D(interpolationImage_B, y, x, interpolation_B[y*cvsize2.width + x]);
		}
	}

	//free(interpolation_R);
	//free(interpolation_G);
	//free(interpolation_B);

	//cvReleaseImage(&interpolationImage_R);
	//cvReleaseImage(&interpolationImage_G);
	//cvReleaseImage(&interpolationImage_B);

	ui.listWidget->addItem("Bicubic Interpolation : Done");

	//free(input_R); free(input_G); free(input_B);
	//free(input_Y); free(input_Cb); free(input_Cr);
	//free(bicubic_Y_temp); free(bicubic_Cb_temp); free(bicubic_Cr_temp);

	//cvReleaseImage(&r); cvReleaseImage(&g); cvReleaseImage(&b);
	tStart_total = cvGetTickCount();
	// layer 1
	// feature channel of layer : 64

	// Read weight1(3x3x64)
	weight_conv1 = (double *)malloc(sizeof(double)*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv1.raw", "rb");
	fread(weight_conv1, sizeof(double), Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	// Read bias1(64)
	bias_conv1 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv1.raw", "rb");
	fread(bias_conv1, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	// Total conv1 data : (double*) conv1_data;
	conv1_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// Channel:64
	for (layer1_index = 0; layer1_index < Layer_CH_Num; layer1_index++) {
		// Initialization to Zero
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
		// Wx + b
		// Convolution without padding
		for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
			for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
				conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				// 3x3 convolution
				for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
					for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
						conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j]
							+ (bicubic_Y[+(i + k)*HIgh_w + (j + l)] / 255)
							* weight_conv1[layer1_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
					}
				}
				conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv1[layer1_index]);

				// ReLU activation function
				if (conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv1);
	//free(bias_conv1);

	ui.listWidget->addItem("Layer 1 Forward Pass : " + QString::number(TestTime));

	// layer2
	// feature channel of layer : 64

	weight_conv2 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv2.raw", "rb");       //Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num
	fread(weight_conv2, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv2 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv2.raw", "rb");
	fread(bias_conv2, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data2 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv2_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	for (layer2_index = 0; layer2_index < Layer_CH_Num; layer2_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer1_index = 0; layer1_index < Layer_CH_Num; layer1_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data2[layer1_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {

					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data2[layer1_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data2[layer1_index *HIgh_h*HIgh_w + i*HIgh_w + j]
								+ conv1_data[layer1_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv2[layer2_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer1_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data2[layer1_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv2[layer2_index]);

				// ReLU activation function
				if (conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv2);
	//free(bias_conv2);
	//free(temp_data2);
	//free(conv1_data);

	ui.listWidget->addItem("Layer 2 Forward Pass : " + QString::number(TestTime));

	// layer3
	//Convolutional_layer(3, weight_conv3, bias_conv3, temp_data3, conv3_data, conv2_data);//3, weight_conv3, bias_conv3, temp_data3, conv3_data, conv2_data


	// layer3
	// feature channel of layer : 64

	weight_conv3 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv3.raw", "rb");
	fread(weight_conv3, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv3 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv3.raw", "rb");
	fread(bias_conv3, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data3 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv3_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	for (layer3_index = 0; layer3_index < Layer_CH_Num; layer3_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer2_index = 0; layer2_index < Layer_CH_Num; layer2_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data3[layer2_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data3[layer2_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data3[layer2_index *HIgh_h*HIgh_w + i*HIgh_w + j]
								+ conv2_data[layer2_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv3[layer3_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer2_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data3[layer2_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv3[layer3_index]);

				// ReLU activation function
				if (conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv3);
	//free(bias_conv3);
	//free(temp_data3);
	//free(conv2_data);

	ui.listWidget->addItem("Layer 3 Forward Pass : " + QString::number(TestTime));



	// layer 4
	// feature channel of layer : 64

	weight_conv4 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv4.raw", "rb");
	fread(weight_conv4, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv4 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv4.raw", "rb");
	fread(bias_conv4, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data4 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv4_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();
	for (layer4_index = 0; layer4_index < Layer_CH_Num; layer4_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer3_index = 0; layer3_index < Layer_CH_Num; layer3_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data4[layer3_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data4[layer3_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data4[layer3_index *HIgh_h*HIgh_w + i*HIgh_w + j]
								+ conv3_data[layer3_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv4[layer4_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer3_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data4[layer3_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv4[layer4_index]);

				// ReLU activation function
				if (conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv4);
	//free(bias_conv4);
	//free(temp_data4);
	//free(conv3_data);

	ui.listWidget->addItem("Layer 5 Forward Pass : " + QString::number(TestTime));




	// layer5
	// feature channel of layer : 64

	weight_conv5 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv5.raw", "rb");
	fread(weight_conv5, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv5 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv5.raw", "rb");
	fread(bias_conv5, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data5 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv5_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();
	for (layer5_index = 0; layer5_index < Layer_CH_Num; layer5_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer4_index = 0; layer4_index < Layer_CH_Num; layer4_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data5[layer4_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data5[layer4_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data5[layer4_index *HIgh_h*HIgh_w + i*HIgh_w + j]
								+ conv4_data[layer4_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv5[layer5_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer4_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data5[layer4_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv5[layer5_index]);

				// ReLU activation function
				if (conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv5);
	//free(bias_conv5);
	//free(temp_data5);
	//free(conv4_data);

	ui.listWidget->addItem("Layer 5 Forward Pass : " + QString::number(TestTime));



	// layer6
	// feature channel of layer : 64

	weight_conv6 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv6.raw", "rb");
	fread(weight_conv6, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv6 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv6.raw", "rb");
	fread(bias_conv6, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data6 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv6_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer6_index = 0; layer6_index < Layer_CH_Num; layer6_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer5_index = 0; layer5_index < Layer_CH_Num; layer5_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data6[layer5_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data6[layer5_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data6[layer5_index *HIgh_h*HIgh_w
								+ i*HIgh_w + j] + conv5_data[layer5_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv6[layer6_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer5_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data6[layer5_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv6[layer6_index]);

				// ReLU activation function
				if (conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv6);
	//free(bias_conv6);
	//free(temp_data6);
	//free(conv5_data);

	ui.listWidget->addItem("Layer 6 Forward Pass : " + QString::number(TestTime));



	// layer7
	// feature channel of layer : 64

	weight_conv7 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv7.raw", "rb");
	fread(weight_conv7, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv7 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv7.raw", "rb");
	fread(bias_conv7, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data7 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv7_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer7_index = 0; layer7_index < Layer_CH_Num; layer7_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer6_index = 0; layer6_index < Layer_CH_Num; layer6_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data7[layer6_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data7[layer6_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data7[layer6_index *HIgh_h*HIgh_w + i*HIgh_w + j]
								+ conv6_data[layer6_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv7[layer7_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer6_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data7[layer6_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv7[layer7_index]);

				// ReLU activation function
				if (conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv7);
	//free(bias_conv7);
	//free(temp_data7);
	//free(conv6_data);

	ui.listWidget->addItem("Layer 7 Forward Pass : " + QString::number(TestTime));




	// layer8
	// feature channel of layer : 64

	weight_conv8 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv8.raw", "rb");
	fread(weight_conv8, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv8 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv8.raw", "rb");
	fread(bias_conv8, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data8 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv8_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer8_index = 0; layer8_index < Layer_CH_Num; layer8_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer7_index = 0; layer7_index < Layer_CH_Num; layer7_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data8[layer7_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data8[layer7_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data8[layer7_index *HIgh_h*HIgh_w + i*HIgh_w + j]
								+ conv7_data[layer7_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv8[layer8_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer7_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data8[layer7_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv8[layer8_index]);

				// ReLU activation function
				if (conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv8);
	//free(bias_conv8);
	//free(temp_data8);
	//free(conv7_data);

	ui.listWidget->addItem("Layer 8 Forward Pass : " + QString::number(TestTime));



	// layer9
	// feature channel of layer : 64

	weight_conv9 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv9.raw", "rb");
	fread(weight_conv9, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv9 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv9.raw", "rb");
	fread(bias_conv9, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data9 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv9_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	for (layer9_index = 0; layer9_index < Layer_CH_Num; layer9_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer8_index = 0; layer8_index < Layer_CH_Num; layer8_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data9[layer8_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data9[layer8_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data9[layer8_index *HIgh_h*HIgh_w + i*HIgh_w + j]
								+ conv8_data[layer8_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv9[layer9_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer8_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data9[layer8_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv9[layer9_index]);

				// ReLU activation function
				if (conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv9);
	//free(bias_conv9);
	//free(temp_data9);
	//free(conv8_data);

	ui.listWidget->addItem("Layer 9 Forward Pass : " + QString::number(TestTime));




	// layer10
	// feature channel of layer : 64

	weight_conv10 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv10.raw", "rb");
	fread(weight_conv10, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv10 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv10.raw", "rb");
	fread(bias_conv10, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data10 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv10_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	for (layer10_index = 0; layer10_index < Layer_CH_Num; layer10_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer9_index = 0; layer9_index < Layer_CH_Num; layer9_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data10[layer9_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data10[layer9_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data10[layer9_index *HIgh_h*HIgh_w + i*HIgh_w + j]
								+ conv9_data[layer9_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv10[layer10_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer9_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data10[layer9_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv10[layer10_index]);

				// ReLU activation function
				if (conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv10);
	//free(bias_conv10);
	//free(temp_data10);
	//free(conv9_data);

	ui.listWidget->addItem("Layer 10 Forward Pass : " + QString::number(TestTime));

	// layer11
	// feature channel of layer (N-1) : 64
	weight_conv11 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv11.raw", "rb");
	fread(weight_conv11, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv11 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv11.raw", "rb");
	fread(bias_conv11, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data11 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv11_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer11_index = 0; layer11_index < Layer_CH_Num; layer11_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer10_index = 0; layer10_index < Layer_CH_Num; layer10_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data11[layer10_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data11[layer10_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data11[layer10_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv10_data[layer10_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv11[layer11_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer10_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data11[layer10_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv11[layer11_index]);

				// ReLU activation function
				if (conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv11);
	//free(bias_conv11);
	//free(temp_data11);
	//free(conv10_data);

	ui.listWidget->addItem("Layer 11 Forward Pass : " + QString::number(TestTime));

	// layer12
	// feature channel of layer (N-1) : 64
	printf("Layer 12 Forward Pass : ");

	weight_conv12 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv12.raw", "rb");
	fread(weight_conv12, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv12 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv12.raw", "rb");
	fread(bias_conv12, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data12 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv12_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer12_index = 0; layer12_index < Layer_CH_Num; layer12_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer11_index = 0; layer11_index < Layer_CH_Num; layer11_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data12[layer11_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data12[layer11_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data12[layer11_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv11_data[layer11_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv12[layer12_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer11_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data12[layer11_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv12[layer12_index]);

				// ReLU activation function
				if (conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv12);
	//free(bias_conv12);
	//free(temp_data12);
	//free(conv11_data);

	ui.listWidget->addItem("Layer 12 Forward Pass : " + QString::number(TestTime));




	// layer13
	// feature channel of layer (N-1) : 64
	printf("Layer 13 Forward Pass : ");

	weight_conv13 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv13.raw", "rb");
	fread(weight_conv13, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv13 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv13.raw", "rb");
	fread(bias_conv13, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data13 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv13_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer13_index = 0; layer13_index < Layer_CH_Num; layer13_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer12_index = 0; layer12_index < Layer_CH_Num; layer12_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data13[layer12_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data13[layer12_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data13[layer12_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv12_data[layer12_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv13[layer13_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer12_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data13[layer12_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv13[layer13_index]);

				// ReLU activation function
				if (conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv13);
	//free(bias_conv13);
	//free(temp_data13);
	//free(conv12_data);

	ui.listWidget->addItem("Layer 13 Forward Pass : " + QString::number(TestTime));

	// layer14
	// feature channel of layer (N-1) : 64
	printf("Layer 14 Forward Pass : ");

	weight_conv14 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv14.raw", "rb");
	fread(weight_conv14, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv14 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv14.raw", "rb");
	fread(bias_conv14, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data14 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv14_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer14_index = 0; layer14_index < Layer_CH_Num; layer14_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer13_index = 0; layer13_index < Layer_CH_Num; layer13_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data14[layer13_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data14[layer13_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data14[layer13_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv13_data[layer13_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv14[layer14_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer13_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data14[layer13_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv14[layer14_index]);

				// ReLU activation function
				if (conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv14);
	//free(bias_conv14);
	//free(temp_data14);
	//free(conv13_data);

	ui.listWidget->addItem("Layer 14 Forward Pass : " + QString::number(TestTime));

	// layer15
	// feature channel of layer (N-1) : 64
	printf("Layer 15 Forward Pass : ");

	weight_conv15 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv15.raw", "rb");
	fread(weight_conv15, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv15 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv15.raw", "rb");
	fread(bias_conv15, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data15 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv15_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer15_index = 0; layer15_index < Layer_CH_Num; layer15_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer14_index = 0; layer14_index < Layer_CH_Num; layer14_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data15[layer14_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data15[layer14_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data15[layer14_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv14_data[layer14_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv15[layer15_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer14_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data15[layer14_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv15[layer15_index]);

				// ReLU activation function
				if (conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv15);
	//free(bias_conv15);
	//free(temp_data15);
	//free(conv14_data);

	ui.listWidget->addItem("Layer 15 Forward Pass : " + QString::number(TestTime));

	// layer16
	// feature channel of layer (N-1) : 64
	printf("Layer 16 Forward Pass : ");

	weight_conv16 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv16.raw", "rb");
	fread(weight_conv16, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv16 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv16.raw", "rb");
	fread(bias_conv16, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data16 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv16_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer16_index = 0; layer16_index < Layer_CH_Num; layer16_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer15_index = 0; layer15_index < Layer_CH_Num; layer15_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data16[layer15_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data16[layer15_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data16[layer15_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv15_data[layer15_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv16[layer16_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer15_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data16[layer15_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv16[layer16_index]);

				// ReLU activation function
				if (conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv16);
	//free(bias_conv16);
	//free(temp_data16);
	//free(conv15_data);

	ui.listWidget->addItem("Layer 16 Forward Pass : " + QString::number(TestTime));

	// layer17
	// feature channel of layer (N-1) : 64
	printf("Layer 17 Forward Pass : ");

	weight_conv17 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv17.raw", "rb");
	fread(weight_conv17, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv17 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv17.raw", "rb");
	fread(bias_conv17, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data17 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv17_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer17_index = 0; layer17_index < Layer_CH_Num; layer17_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer16_index = 0; layer16_index < Layer_CH_Num; layer16_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data17[layer16_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data17[layer16_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data17[layer16_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv16_data[layer16_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv17[layer17_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer16_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data17[layer16_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv17[layer17_index]);

				// ReLU activation function
				if (conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv17);
	//free(bias_conv17);
	//free(temp_data17);
	//free(conv16_data);

	ui.listWidget->addItem("Layer 17 Forward Pass : " + QString::number(TestTime));


	// layer18
	// feature channel of layer (N-1) : 64
	printf("Layer 18 Forward Pass : ");

	weight_conv18 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv18.raw", "rb");
	fread(weight_conv18, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv18 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv18.raw", "rb");
	fread(bias_conv18, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data18 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv18_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	for (layer18_index = 0; layer18_index < Layer_CH_Num; layer18_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer17_index = 0; layer17_index < Layer_CH_Num; layer17_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data18[layer17_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data18[layer17_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data18[layer17_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv17_data[layer17_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv18[layer18_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer17_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data18[layer17_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv18[layer18_index]);

				// ReLU activation function
				if (conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv18);
	//free(bias_conv18);
	//free(temp_data18);
	//free(conv17_data);

	ui.listWidget->addItem("Layer 18 Forward Pass : " + QString::number(TestTime));

	// layer19
	// feature channel of layer (N-1) : 64
	printf("Layer 19 Forward Pass : ");

	weight_conv19 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv19.raw", "rb");
	fread(weight_conv19, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv19 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv19.raw", "rb");
	fread(bias_conv19, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data19 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv19_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();


	for (layer19_index = 0; layer19_index < Layer_CH_Num; layer19_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// feature channel of layer N : 64
		for (layer18_index = 0; layer18_index < Layer_CH_Num; layer18_index++) {
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					temp_data19[layer18_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
							temp_data19[layer18_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data19[layer18_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv18_data[layer18_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv19[layer19_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer18_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}

			for (i = 0; i < HIgh_h; i++) {
				for (j = 0; j < HIgh_w; j++) {
					conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data19[layer18_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}
		}
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv19[layer19_index]);

				// ReLU activation function
				if (conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv19);
	//free(bias_conv19);
	//free(temp_data19);
	//free(conv18_data);

	ui.listWidget->addItem("Layer 19 Forward Pass : " + QString::number(TestTime));

	// layer 20
	printf("Layer 20 Forward Pass : ");

	weight_conv20 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size);
	in = fopen("Input/filter/weights_conv20.raw", "rb");
	fread(weight_conv20, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size, in);
	fclose(in);

	bias_conv20 = (double *)malloc(sizeof(double)*Layer20_CH_Num);
	in = fopen("Input/filter/bias_conv20.raw", "rb");
	fread(bias_conv20, sizeof(double), Layer20_CH_Num, in);
	fclose(in);

	temp_data20 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv20_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer20_CH_Num);

	tStart = cvGetTickCount();


	for (i = 0; i < HIgh_h; i++) {
		for (j = 0; j < HIgh_w; j++) {
			conv20_data[i*HIgh_w + j] = 0;
		}
	}
	for (layer19_index = 0; layer19_index < Layer_CH_Num; layer19_index++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// Convolution without padding
		for (i = half_filter_size; i < HIgh_h - half_filter_size; i++) {
			for (j = half_filter_size; j < HIgh_w - half_filter_size; j++) {
				// 3x3 convolution
				for (k = -1 * half_filter_size; k < half_filter_size + 1; k++) {
					for (l = -1 * half_filter_size; l < half_filter_size + 1; l++) {
						temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv19_data[layer19_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)] * weight_conv20[layer19_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
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
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	//free(weight_conv20);
	//free(bias_conv20);
	//free(temp_data20);
	//free(conv19_data);

	ui.listWidget->addItem("Layer 20 Forward Pass : " + QString::number(TestTime));


	summation_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			summation_data[i*HIgh_w + j] = (conv20_data[i*HIgh_w + j] + (bicubic_Y[i*HIgh_w + j] / 255)) * 255;
		}
	}

	ui.listWidget->addItem("Summation : Done");

	output_R = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	output_G = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	output_B = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			output_R[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] + 0.00000030112439741101*bicubic_Cb[i*HIgh_w + j] + 1.59602688733570000000*bicubic_Cr[i*HIgh_w + j] - 222.921617109194 + 0.5);
			output_G[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] - 0.39176253994145000000*bicubic_Cb[i*HIgh_w + j] - 0.81296829216220500000*bicubic_Cr[i*HIgh_w + j] + 135.575409522967 + 0.5);
			output_B[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] + 2.01723263955646000000*bicubic_Cb[i*HIgh_w + j] + 0.00000305426174524847*bicubic_Cr[i*HIgh_w + j] - 276.836305795032 + 0.5);

		}
	}
	ui.listWidget->addItem("YCbCr to RGB : Done");

	CvSize cvsize1 = { HIgh_w, HIgh_h };
	IplImage* TempImage1 = cvCreateImage(cvsize1, IPL_DEPTH_8U, 3);
	IplImage* TempImage_R = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);
	IplImage* TempImage_G = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);
	IplImage* TempImage_B = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);

	for (int y = 0; y < cvsize1.height; y++) {
		for (int x = 0; x < cvsize1.width; x++) {
			//cvSetReal2D(TempImage1, y, x, conv3_data[ y*cvsize1.width+x]);
			cvSetReal2D(TempImage_R, y, x, output_R[y*cvsize1.width + x]);
			cvSetReal2D(TempImage_G, y, x, output_G[y*cvsize1.width + x]);
			cvSetReal2D(TempImage_B, y, x, output_B[y*cvsize1.width + x]);
		}
	}
	cvMerge(TempImage_B, TempImage_G, TempImage_R, NULL, TempImage1);

	//free(output_R);
	//free(output_G);
	//free(output_B);

	//free(bicubic_Y);
	//free(bicubic_Cb);
	//free(bicubic_Cr);

	//free(conv20_data);
	//free(summation_data);

	//cvReleaseImage(&TempImage_R);
	//cvReleaseImage(&TempImage_G);
	//cvReleaseImage(&TempImage_B);


	// Output Path
	string out_path = output_dir_path.toLocal8Bit().constData();
	out_path.append("/output_VDSR_Serial_");
	out_path.append(Input_File_Name);

	// Create char * output path
	char * Output_Path_char_pointer;
	Output_Path_char_pointer = new char[out_path.size() + 1];
	std::copy(out_path.begin(), out_path.end(), Output_Path_char_pointer);
	Output_Path_char_pointer[out_path.size()] = '\0';

	// Store Image
	cvSaveImage(Output_Path_char_pointer, TempImage1);

	// Display Image
	// ConHigh_ht IplImage* to Mat Image
	Mat VDSR_Serial = cvarrToMat(TempImage1);//interpolationImage1

											 // Display output
	QImage qimgSerial = convertOpenCVMatToQtQImage(VDSR_Serial);
	ui.label_12->setPixmap(QPixmap::fromImage(qimgSerial.scaled(280, 280, Qt::KeepAspectRatio)));


	//free(Input);
	//cvReleaseImage(&TempImage1);

	//ui.listWidget->addItem("TestTime_total_Serial : " + QString::number(TestTime_total_Serial));

	tEnd = cvGetTickCount();// for check processing
	tEnd_total = 0.001 * (tEnd - tStart_total) / cvGetTickFrequency(); // for msec

																	   // Display Input Box
	ui.label_28->setText(QString::number(tEnd_total) + " ms  ");

	Mat output_Image = cvarrToMat(TempImage1);
	Mat Origianl_image = imread(input_file_path.toLocal8Bit().constData(), 1);
	double PSNR = getPSNR(Origianl_image, output_Image);
	ui.label_25->setText(QString::number(PSNR) + " dB  ");
}

// VDSR_OpenMP Processing & Display 
void PIP_final_project::slotDisplay_VDSR_OpenMP() {
	double TestTime_total_OpenMP = 0;
	ui.listWidget->addItem("========= VDSR OpenMP =========");
	IplImage* pInput = cvLoadImage(input_path.c_str(), CV_LOAD_IMAGE_COLOR); //load image using opencv
	IplImage* r = cvCreateImage(cvGetSize(pInput), IPL_DEPTH_8U, 1);    //R channel
	IplImage* g = cvCreateImage(cvGetSize(pInput), IPL_DEPTH_8U, 1);    //G channel
	IplImage* b = cvCreateImage(cvGetSize(pInput), IPL_DEPTH_8U, 1);    //B channel

	Low_w = pInput->width;
	Low_h = pInput->height;

	HIgh_w = Low_w * 2;
	HIgh_h = Low_h * 2;

	//Mat VDSR_OpenMP1 = cvarrToMat(pInput);//

	int i, j, k, l;
	int layer1_index, layer2_index, layer3_index, layer4_index, layer5_index, layer6_index, layer7_index, layer8_index, layer9_index, layer10_index;
	int layer11_index, layer12_index, layer13_index, layer14_index, layer15_index, layer16_index, layer17_index, layer18_index, layer19_index;
	int half_filter_size = 1;

	double bicubic2[2][4] = { { -0.0234375, 0.2265625, 0.8671875, -0.0703125 },
	{ -0.0703125, 0.8671875, 0.2265625, -0.0234375 } };

	// Weight
	double* weight_conv1;  double* weight_conv2;  double* weight_conv3;  double* weight_conv4;  double* weight_conv5;
	double* weight_conv6;  double* weight_conv7;  double* weight_conv8;  double* weight_conv9;  double* weight_conv10;
	double* weight_conv11; double* weight_conv12; double* weight_conv13; double* weight_conv14; double* weight_conv15;
	double* weight_conv16; double* weight_conv17; double* weight_conv18; double* weight_conv19; double* weight_conv20;

	// Bias
	double* bias_conv1;  double* bias_conv2;  double* bias_conv3;  double* bias_conv4;  double* bias_conv5;
	double* bias_conv6;  double* bias_conv7;  double* bias_conv8;  double* bias_conv9;  double* bias_conv10;
	double* bias_conv11; double* bias_conv12; double* bias_conv13; double* bias_conv14; double* bias_conv15;
	double* bias_conv16; double* bias_conv17; double* bias_conv18; double* bias_conv19; double* bias_conv20;

	// Input RGB
	double* input_R; double* input_G; double* input_B;

	// Output RGB
	double* output_R; double* output_G; double* output_B;

	// Input YCbCr
	double* input_Y; double* input_Cb; double* input_Cr;

	// Bicubic YCbCr
	double* bicubic_Y;		double* bicubic_Cb;		 double* bicubic_Cr;
	double* bicubic_Y_temp; double* bicubic_Cb_temp; double* bicubic_Cr_temp;

	// Convolution data
	double* conv1_data;  double* conv2_data;  double* conv3_data;  double* conv4_data;  double* conv5_data;
	double* conv6_data;  double* conv7_data;  double* conv8_data;  double* conv9_data;  double* conv10_data;
	double* conv11_data; double* conv12_data; double* conv13_data; double* conv14_data; double* conv15_data;
	double* conv16_data; double* conv17_data; double* conv18_data; double* conv19_data; double* conv20_data;

	double* summation_data;

	// Temp data
	double* temp_data2;  double* temp_data3;  double* temp_data4;  double* temp_data5;
	double* temp_data6;  double* temp_data7;  double* temp_data8;  double* temp_data9;  double* temp_data10;
	double* temp_data11; double* temp_data12; double* temp_data13; double* temp_data14; double* temp_data15;
	double* temp_data16; double* temp_data17; double* temp_data18; double* temp_data19; double* temp_data20;

	FILE *in;

	input_R = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_G = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_B = (double *)malloc(sizeof(double)*Low_h*Low_w);

	input_Y = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_Cb = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_Cr = (double *)malloc(sizeof(double)*Low_h*Low_w);

	bicubic_Y = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicubic_Cb = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicubic_Cr = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	bicubic_Y_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicubic_Cb_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicubic_Cr_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);

	int64 tStart, tEnd, tStart_total, tEnd_total;
	double TestTime, TestTime_total;

	ui.listWidget->addItem("Initialization : DONE");

	cvSplit(pInput, b, g, r, NULL);

	omp_set_num_threads(8);
	tStart_total = cvGetTickCount();

#pragma omp parallel for /*private(j,i)*/
	for (j = 0; j < Low_w; j++) {
		for (i = 0; i < Low_h; i++){
			input_R[i*Low_w + j] = unsigned char(r->imageData[i*Low_w + j]);//cvGetReal2D(r, i, j);
			input_G[i*Low_w + j] = unsigned char(g->imageData[i*Low_w + j]);//cvGetReal2D(g, i, j);
			input_B[i*Low_w + j] = unsigned char(b->imageData[i*Low_w + j]);//cvGetReal2D(b, i, j);
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#pragma omp parallel for private(j,i)
	for (j = 0; j < Low_w; j++) {
		for (i = 0; i < Low_h; i++)
		{
			input_Y[i*Low_w + j] = 0.256788235294118*input_R[i*Low_w + j] + 0.504129411764706*input_G[i*Low_w + j] + 0.0979058823529412*input_B[i*Low_w + j] + 16;
			input_Cb[i*Low_w + j] = -0.148223529411765*input_R[i*Low_w + j] - 0.290992156862745*input_G[i*Low_w + j] + 0.4392156862745100*input_B[i*Low_w + j] + 128;
			input_Cr[i*Low_w + j] = 0.439215686274510*input_R[i*Low_w + j] - 0.367788235294118*input_G[i*Low_w + j] - 0.0714274509803922*input_B[i*Low_w + j] + 128;
		}
	}

	ui.listWidget->addItem("RGB to YCbCr : DONE");

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#pragma omp parallel sections private(j,i)
	{
#pragma omp section
		for (i = 2; i < Low_h / 2; i++) // High_wizontal
		{
			for (j = 0; j < Low_w / 4; j++)
			{
				bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
				bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			}
		}
#pragma omp section
		for (i = 2; i < Low_h / 2; i++) // High_wizontal
		{
			for (j = Low_w / 4; j < Low_w / 2; j++)
			{
				bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
				bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			}
		}
#pragma omp section
		for (i = 2; i < Low_h / 2; i++) // High_wizontal
		{
			for (j = Low_w / 2; j < Low_w / 4 * 3; j++)
			{
				bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
				bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			}
		}
#pragma omp section
		for (i = 2; i < Low_h / 2; i++) // High_wizontal
		{
			for (j = Low_w / 4 * 3; j < Low_w; j++)
			{
				bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
				bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			}
		}

#pragma omp section
		for (i = Low_h / 2; i < Low_h - 2; i++) // High_wizontal
		{
			for (j = 0; j < Low_w / 4; j++)
			{
				bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
				bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			}
		}
#pragma omp section
		for (i = Low_h / 2; i < Low_h - 2; i++) // High_wizontal
		{
			for (j = Low_w / 4; j < Low_w / 2; j++)
			{
				bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
				bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			}
		}
#pragma omp section
		for (i = Low_h / 2; i < Low_h - 2; i++) // High_wizontal
		{
			for (j = Low_w / 2; j < Low_w / 4 * 3; j++)
			{
				bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
				bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			}
		}
#pragma omp section
		for (i = Low_h / 2; i < Low_h - 2; i++) // High_wizontal
		{
			for (j = Low_w / 4 * 3; j < Low_w; j++)
			{
				bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
				bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

				bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
				bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel sections private(j,i)
	{
#pragma omp section
		for (i = 0; i < HIgh_h / 2; i++) // High_htical
		{
			for (j = 2; j < Low_w / 4; j++)
			{
				bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
				bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

				bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
				bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

				bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
				bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			}
		}
#pragma omp section
		for (i = 0; i < HIgh_h / 2; i++) // High_htical
		{
			for (j = Low_w / 4; j < Low_w / 2; j++)
			{
				bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
				bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

				bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
				bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

				bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
				bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			}
		}
#pragma omp section
		for (i = 0; i < HIgh_h / 2; i++) // High_htical
		{
			for (j = Low_w / 2; j < Low_w / 4 * 3; j++)
			{
				bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
				bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

				bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
				bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

				bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
				bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			}
		}
#pragma omp section
		for (i = 0; i < HIgh_h / 2; i++) // High_htical
		{
			for (j = Low_w / 4 * 3; j < Low_w - 2; j++)
			{
				bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
				bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

				bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
				bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

				bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
				bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			}
		}
#pragma omp section
		for (i = HIgh_h / 2; i < HIgh_h; i++) // High_htical
		{
			for (j = 2; j < Low_w / 4; j++)
			{
				bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
				bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

				bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
				bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

				bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
				bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			}
		}
#pragma omp section
		for (i = HIgh_h / 2; i < HIgh_h; i++) // High_htical
		{
			for (j = Low_w / 4; j < Low_w / 2; j++)
			{
				bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
				bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

				bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
				bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

				bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
				bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			}
		}
#pragma omp section
		for (i = HIgh_h / 2; i < HIgh_h; i++) // High_htical
		{
			for (j = Low_w / 2; j < Low_w / 4 * 3; j++)
			{
				bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
				bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

				bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
				bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

				bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
				bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			}
		}
#pragma omp section
		for (i = HIgh_h / 2; i < HIgh_h; i++) // High_htical
		{
			for (j = Low_w / 4 * 3; j < Low_w - 2; j++)
			{
				bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
				bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

				bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
				bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

				bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
				bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			}
		}
	}

	//int64 check_e = cvGetTickCount();
	//double check_t = 0.001 * (check_e - check_s) / (double)cvGetTickFrequency();
	//printf("%f msec\n", check_t);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Mat VDSR_OpenMPY = cvarrToMat(bicubic_Y);//
	//Mat VDSR_OpenMPCb = cvarrToMat(bicubic_Cb);//
	//Mat VDSR_OpenMPCr = cvarrToMat(bicubic_Cr);//

	ui.listWidget->addItem("Bicubic Interpolation : Done");


	// layer 1
	// feature channel : 64
	printf("Layer 1 Foward Pass : ");

	weight_conv1 = (double *)malloc(sizeof(double)*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv1.raw", "rb");
	fread(weight_conv1, sizeof(double), Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv1 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv1.raw", "rb");
	fread(bias_conv1, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	conv1_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

#pragma omp parallel for private(j, i, k, l)
	for (layer1_index = 0; layer1_index < Layer_CH_Num; layer1_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}

		// Convolution without padding
		for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
		{
			for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
			{
				conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				// 9x9 convolution
				for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
				{
					for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
					{
						conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] + (bicubic_Y[+(i + k)*HIgh_w + (j + l)] / 255)* weight_conv1[layer1_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
					}
				}
				conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv1[layer1_index]);

				// ReLU activation function
				if (conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv1_data[layer1_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();
	printf("%f msec\n", TestTime);

	//Mat VDSR_OpenMP_1 = cvarrToMat(conv1_data);//


	// layer2
	// feature channel of layer (N-1) : 64

	printf("Layer 2 Forward Pass : ");

	weight_conv2 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv2.raw", "rb");
	fread(weight_conv2, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv2 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv2.raw", "rb");
	fread(bias_conv2, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data2 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv2_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer2_index = 0; layer2_index < Layer_CH_Num; layer2_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}

#pragma omp parallel for private(j, i, k, l)
		// feature channel of layer N : 64
		for (layer1_index = 0; layer1_index < Layer_CH_Num; layer1_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data2[layer1_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
				}
			}
			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data2[layer1_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data2[layer1_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv1_data[layer1_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv2[layer2_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer1_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}
			}
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data2[layer1_index *HIgh_h*HIgh_w + i*HIgh_w + j];
				}
			}

		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv2[layer2_index]);
				// ReLU activation function
				if (conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv2_data[layer2_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 3 Forward Pass : ");

	weight_conv3 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv3.raw", "rb");
	fread(weight_conv3, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv3 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv3.raw", "rb");
	fread(bias_conv3, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data3 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv3_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer3
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer3_index = 0; layer3_index < Layer_CH_Num; layer3_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer2_index = 0; layer2_index < Layer_CH_Num; layer2_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data3[layer2_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data3[layer2_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data3[layer2_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv2_data[layer2_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv3[layer3_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer2_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data3[layer2_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv3[layer3_index]);

				// ReLU activation function
				if (conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv3_data[layer3_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);




	printf("Layer 4 Forward Pass : ");

	weight_conv4 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv4.raw", "rb");
	fread(weight_conv4, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv4 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv4.raw", "rb");
	fread(bias_conv4, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data4 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv4_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer4
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer4_index = 0; layer4_index < Layer_CH_Num; layer4_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer3_index = 0; layer3_index < Layer_CH_Num; layer3_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data4[layer3_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data4[layer3_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data4[layer3_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv3_data[layer3_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv4[layer4_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer3_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data4[layer3_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv4[layer4_index]);

				// ReLU activation function
				if (conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv4_data[layer4_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 5 Forward Pass : ");

	weight_conv5 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv5.raw", "rb");
	fread(weight_conv5, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv5 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv5.raw", "rb");
	fread(bias_conv5, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data5 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv5_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer5
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer5_index = 0; layer5_index < Layer_CH_Num; layer5_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer4_index = 0; layer4_index < Layer_CH_Num; layer4_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data5[layer4_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data5[layer4_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data5[layer4_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv4_data[layer4_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv5[layer5_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer4_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data5[layer4_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv5[layer5_index]);

				// ReLU activation function
				if (conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv5_data[layer5_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 6 Forward Pass : ");

	weight_conv6 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv6.raw", "rb");
	fread(weight_conv6, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv6 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv6.raw", "rb");
	fread(bias_conv6, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data6 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv6_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer6
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer6_index = 0; layer6_index < Layer_CH_Num; layer6_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer5_index = 0; layer5_index < Layer_CH_Num; layer5_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data6[layer5_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data6[layer5_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data6[layer5_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv5_data[layer5_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv6[layer6_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer5_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data6[layer5_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv6[layer6_index]);

				// ReLU activation function
				if (conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv6_data[layer6_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 7 Forward Pass : ");

	weight_conv7 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv7.raw", "rb");
	fread(weight_conv7, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv7 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv7.raw", "rb");
	fread(bias_conv7, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data7 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv7_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer7
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer7_index = 0; layer7_index < Layer_CH_Num; layer7_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer6_index = 0; layer6_index < Layer_CH_Num; layer6_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data7[layer6_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data7[layer6_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data7[layer6_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv6_data[layer6_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv7[layer7_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer6_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data7[layer6_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv7[layer7_index]);

				// ReLU activation function
				if (conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv7_data[layer7_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 8 Forward Pass : ");

	weight_conv8 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv8.raw", "rb");
	fread(weight_conv8, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv8 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv8.raw", "rb");
	fread(bias_conv8, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data8 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv8_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer8
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer8_index = 0; layer8_index < Layer_CH_Num; layer8_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer7_index = 0; layer7_index < Layer_CH_Num; layer7_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data8[layer7_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data8[layer7_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data8[layer7_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv7_data[layer7_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv8[layer8_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer7_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data8[layer7_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv8[layer8_index]);

				// ReLU activation function
				if (conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv8_data[layer8_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 9 Forward Pass : ");

	weight_conv9 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv9.raw", "rb");
	fread(weight_conv9, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv9 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv9.raw", "rb");
	fread(bias_conv9, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data9 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv9_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer9
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer9_index = 0; layer9_index < Layer_CH_Num; layer9_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer8_index = 0; layer8_index < Layer_CH_Num; layer8_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data9[layer8_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data9[layer8_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data9[layer8_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv8_data[layer8_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv9[layer9_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer8_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data9[layer8_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv9[layer9_index]);

				// ReLU activation function
				if (conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv9_data[layer9_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 10 Forward Pass : ");

	weight_conv10 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv10.raw", "rb");
	fread(weight_conv10, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv10 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv10.raw", "rb");
	fread(bias_conv10, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data10 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv10_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer10
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer10_index = 0; layer10_index < Layer_CH_Num; layer10_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer9_index = 0; layer9_index < Layer_CH_Num; layer9_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data10[layer9_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data10[layer9_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data10[layer9_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv9_data[layer9_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv10[layer10_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer9_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data10[layer9_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv10[layer10_index]);

				// ReLU activation function
				if (conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv10_data[layer10_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 11 Forward Pass : ");

	weight_conv11 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv11.raw", "rb");
	fread(weight_conv11, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv11 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv11.raw", "rb");
	fread(bias_conv11, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data11 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv11_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer11
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer11_index = 0; layer11_index < Layer_CH_Num; layer11_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer10_index = 0; layer10_index < Layer_CH_Num; layer10_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data11[layer10_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data11[layer10_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data11[layer10_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv10_data[layer10_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv11[layer11_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer10_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data11[layer10_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv11[layer11_index]);

				// ReLU activation function
				if (conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv11_data[layer11_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 12 Forward Pass : ");

	weight_conv12 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv12.raw", "rb");
	fread(weight_conv12, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv12 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv12.raw", "rb");
	fread(bias_conv12, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data12 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv12_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer12
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer12_index = 0; layer12_index < Layer_CH_Num; layer12_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer11_index = 0; layer11_index < Layer_CH_Num; layer11_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data12[layer11_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data12[layer11_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data12[layer11_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv11_data[layer11_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv12[layer12_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer11_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data12[layer11_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv12[layer12_index]);

				// ReLU activation function
				if (conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv12_data[layer12_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 13 Forward Pass : ");

	weight_conv13 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv13.raw", "rb");
	fread(weight_conv13, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv13 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv13.raw", "rb");
	fread(bias_conv13, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data13 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv13_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer13
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer13_index = 0; layer13_index < Layer_CH_Num; layer13_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer12_index = 0; layer12_index < Layer_CH_Num; layer12_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data13[layer12_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data13[layer12_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data13[layer12_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv12_data[layer12_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv13[layer13_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer12_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data13[layer12_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv13[layer13_index]);

				// ReLU activation function
				if (conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv13_data[layer13_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 14 Forward Pass : ");

	weight_conv14 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv14.raw", "rb");
	fread(weight_conv14, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv14 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv14.raw", "rb");
	fread(bias_conv14, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data14 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv14_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer14
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer14_index = 0; layer14_index < Layer_CH_Num; layer14_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer13_index = 0; layer13_index < Layer_CH_Num; layer13_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data14[layer13_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data14[layer13_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data14[layer13_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv13_data[layer13_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv14[layer14_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer13_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data14[layer13_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv14[layer14_index]);

				// ReLU activation function
				if (conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv14_data[layer14_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 15 Forward Pass : ");

	weight_conv15 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv15.raw", "rb");
	fread(weight_conv15, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv15 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv15.raw", "rb");
	fread(bias_conv15, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data15 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv15_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer15
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer15_index = 0; layer15_index < Layer_CH_Num; layer15_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer14_index = 0; layer14_index < Layer_CH_Num; layer14_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data15[layer14_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data15[layer14_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data15[layer14_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv14_data[layer14_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv15[layer15_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer14_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data15[layer14_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv15[layer15_index]);

				// ReLU activation function
				if (conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv15_data[layer15_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 16 Forward Pass : ");

	weight_conv16 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv16.raw", "rb");
	fread(weight_conv16, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv16 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv16.raw", "rb");
	fread(bias_conv16, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data16 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv16_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer16
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer16_index = 0; layer16_index < Layer_CH_Num; layer16_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer15_index = 0; layer15_index < Layer_CH_Num; layer15_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data16[layer15_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data16[layer15_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data16[layer15_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv15_data[layer15_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv16[layer16_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer15_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data16[layer15_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv16[layer16_index]);

				// ReLU activation function
				if (conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv16_data[layer16_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 17 Forward Pass : ");

	weight_conv17 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv17.raw", "rb");
	fread(weight_conv17, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv17 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv17.raw", "rb");
	fread(bias_conv17, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data17 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv17_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer17
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer17_index = 0; layer17_index < Layer_CH_Num; layer17_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif		// feature channel of layer N : 64
		for (layer16_index = 0; layer16_index < Layer_CH_Num; layer16_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data17[layer16_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data17[layer16_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data17[layer16_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv16_data[layer16_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv17[layer17_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer16_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data17[layer16_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv17[layer17_index]);

				// ReLU activation function
				if (conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv17_data[layer17_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 18 Forward Pass : ");

	weight_conv18 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv18.raw", "rb");
	fread(weight_conv18, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv18 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv18.raw", "rb");
	fread(bias_conv18, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data18 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv18_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer18
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer18_index = 0; layer18_index < Layer_CH_Num; layer18_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif
		// feature channel of layer N : 64
		for (layer17_index = 0; layer17_index < Layer_CH_Num; layer17_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data18[layer17_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data18[layer17_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data18[layer17_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv17_data[layer17_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv18[layer18_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer17_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data18[layer17_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}


		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv18[layer18_index]);

				// ReLU activation function
				if (conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv18_data[layer18_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 19 Forward Pass : ");

	weight_conv19 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num);
	in = fopen("Input/filter/weights_conv19.raw", "rb");
	fread(weight_conv19, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num, in);
	fclose(in);

	bias_conv19 = (double *)malloc(sizeof(double)*Layer_CH_Num);
	in = fopen("Input/filter/bias_conv19.raw", "rb");
	fread(bias_conv19, sizeof(double), Layer_CH_Num, in);
	fclose(in);

	temp_data19 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv19_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	tStart = cvGetTickCount();

	// layer19
	// feature channel of layer (N-1) : 64
#if __first1
#pragma omp parallel for private(j, i, k, l)
#endif
	for (layer19_index = 0; layer19_index < Layer_CH_Num; layer19_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}
#if __first
#pragma omp parallel for private(j, i, k, l)
#endif
		// feature channel of layer N : 64
		for (layer18_index = 0; layer18_index < Layer_CH_Num; layer18_index++)
		{
			//temp_filter[3][3];
			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					temp_data19[layer18_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

				}
			}

			// Convolution without padding
			for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
			{
				for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
				{
					// 3x3 convolution
					for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
					{
						for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
						{
							temp_data19[layer18_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data19[layer18_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv18_data[layer18_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)]
								* weight_conv19[layer19_index*Layer_Filter_Size*Layer_Filter_Size*Layer_CH_Num + layer18_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
						}
					}
				}

			}

			for (i = 0; i < HIgh_h; i++)
			{
				for (j = 0; j < HIgh_w; j++)
				{
					conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] = conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] + temp_data19[layer18_index *HIgh_h*HIgh_w + i*HIgh_w + j];

				}
			}
		}
#pragma omp parallel for private(j, i)
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] = (conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] + bias_conv19[layer19_index]);

				// ReLU activation function
				if (conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] < 0)
					conv19_data[layer19_index*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);





	printf("Layer 20 Forward Pass : ");

	weight_conv20 = (double *)malloc(sizeof(double)*Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size);
	in = fopen("Input/filter/weights_conv20.raw", "rb");
	fread(weight_conv20, sizeof(double), Layer_CH_Num*Layer_Filter_Size*Layer_Filter_Size, in);
	fclose(in);

	bias_conv20 = (double *)malloc(sizeof(double)*Layer20_CH_Num);
	in = fopen("Input/filter/bias_conv20.raw", "rb");
	fread(bias_conv20, sizeof(double), Layer20_CH_Num, in);
	fclose(in);

	temp_data20 = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);
	conv20_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer20_CH_Num);

	tStart = cvGetTickCount();

	// layer 20
	for (i = 0; i < HIgh_h; i++)
	{
		for (j = 0; j < HIgh_w; j++)
		{
			conv20_data[i*HIgh_w + j] = 0;
		}
	}
#pragma omp parallel for private(j, i, k, l)
	for (layer19_index = 0; layer19_index < Layer_CH_Num; layer19_index++)
	{
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] = 0;

			}
		}

		// Convolution without padding
		for (i = half_filter_size; i < HIgh_h - half_filter_size; i++)
		{
			for (j = half_filter_size; j < HIgh_w - half_filter_size; j++)
			{
				// 3x3 convolution
				for (k = -1 * half_filter_size; k < half_filter_size + 1; k++)
				{
					for (l = -1 * half_filter_size; l < half_filter_size + 1; l++)
					{
						temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] = temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j] + conv19_data[layer19_index*HIgh_h*HIgh_w + (i + k)*HIgh_w + (j + l)] * weight_conv20[layer19_index*Layer_Filter_Size*Layer_Filter_Size + (l + half_filter_size)*Layer_Filter_Size + (k + half_filter_size)];
					}
				}
			}

		}
		for (i = 0; i < HIgh_h; i++)
		{
			for (j = 0; j < HIgh_w; j++)
			{
				conv20_data[i*HIgh_w + j] = conv20_data[i*HIgh_w + j] + temp_data20[layer19_index *HIgh_h*HIgh_w + i*HIgh_w + j];

			}
		}
	}
#pragma omp parallel for private(j, i)
	for (i = 0; i < HIgh_h; i++)
	{
		for (j = 0; j < HIgh_w; j++)
		{
			conv20_data[i*HIgh_w + j] = conv20_data[i*HIgh_w + j] + bias_conv20[0];
			//// ReLU activation function
			//if (conv20_data[i*HIgh_w + j] <0)
			//	conv20_data[i*HIgh_w + j] = 0;
		}
	}

	tEnd = cvGetTickCount();
	TestTime = 0.001 * (tEnd - tStart) / (double)cvGetTickFrequency();

	printf("%f msec\n", TestTime);

	printf("Summation : ");

	summation_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			summation_data[i*HIgh_w + j] = (conv20_data[i*HIgh_w + j] + (bicubic_Y[i*HIgh_w + j] / 255)) * 255;
		}
	}

	printf("DONE\n");

	printf("YCbCr to RGB : ");

	output_R = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	output_G = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	output_B = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			output_R[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] + 0.00000030112439741101*bicubic_Cb[i*HIgh_w + j] + 1.59602688733570000000*bicubic_Cr[i*HIgh_w + j] - 222.921617109194 + 0.5);
			output_G[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] - 0.39176253994145000000*bicubic_Cb[i*HIgh_w + j] - 0.81296829216220500000*bicubic_Cr[i*HIgh_w + j] + 135.575409522967 + 0.5);
			output_B[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] + 2.01723263955646000000*bicubic_Cb[i*HIgh_w + j] + 0.00000305426174524847*bicubic_Cr[i*HIgh_w + j] - 276.836305795032 + 0.5);

		}
	}

	printf("DONE\n");


	CvSize cvsize1 = { HIgh_w, HIgh_h };
	IplImage* TempImage1 = cvCreateImage(cvsize1, IPL_DEPTH_8U, 3);
	IplImage* TempImage_R = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);
	IplImage* TempImage_G = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);
	IplImage* TempImage_B = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);

	for (int y = 0; y < cvsize1.height; y++) {
		for (int x = 0; x < cvsize1.width; x++) {
			//cvSetReal2D(TempImage1, y, x, conv3_data[ y*cvsize1.width+x]);
			cvSetReal2D(TempImage_R, y, x, output_R[y*cvsize1.width + x]);
			cvSetReal2D(TempImage_G, y, x, output_G[y*cvsize1.width + x]);
			cvSetReal2D(TempImage_B, y, x, output_B[y*cvsize1.width + x]);


		}
	}

	cvMerge(TempImage_B, TempImage_G, TempImage_R, NULL, TempImage1);


	// Output Path
	string out_path = output_dir_path.toLocal8Bit().constData();
	out_path.append("/output_VDSR_OpenMP_");
	out_path.append(Input_File_Name);

	// Create char * output path
	char * Output_Path_char_pointer;
	Output_Path_char_pointer = new char[out_path.size() + 1];
	std::copy(out_path.begin(), out_path.end(), Output_Path_char_pointer);
	Output_Path_char_pointer[out_path.size()] = '\0';

	// Store Image
	cvSaveImage(Output_Path_char_pointer, TempImage1);

	// Display Image
	// ConHigh_ht IplImage* to Mat Image
	Mat VDSR_OpenMP = cvarrToMat(TempImage1);//

											 // Display output
	QImage qimgSerial = convertOpenCVMatToQtQImage(VDSR_OpenMP);//
	ui.label_13->setPixmap(QPixmap::fromImage(qimgSerial.scaled(280, 280, Qt::KeepAspectRatio)));

	//ui.listWidget->addItem("TestTime_total_OpenMP : " + QString::number(TestTime_total_OpenMP));

	tEnd = cvGetTickCount();// for check processing
	tEnd_total = 0.001 * (tEnd - tStart_total) / cvGetTickFrequency(); // for msec

																	   // Display Input Box
	ui.label_30->setText(QString::number(tEnd_total) + " ms  ");

	Mat output_Image = cvarrToMat(TempImage1);
	Mat Origianl_image = imread(input_file_path.toLocal8Bit().constData(), 1);
	double PSNR = getPSNR(Origianl_image, output_Image);
	ui.label_29->setText(QString::number(PSNR) + " dB  ");
}




// VDSR_CUDA Processing & Display 
void PIP_final_project::slotDisplay_VDSR_CUDA() {
	TestTime_total_CUDA = 0;
	ui.listWidget->addItem("========= VDSR CUDA =========");
	//load image using opencv
	IplImage* Input = cvLoadImage(input_path.c_str(), CV_LOAD_IMAGE_COLOR);

	Low_w = Input->width;
	Low_h = Input->height;

	HIgh_w = Low_w * 2;
	HIgh_h = Low_h * 2;


	// R, G, B channel
	IplImage* r = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //R channel
	IplImage* g = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //G channel
	IplImage* b = cvCreateImage(cvGetSize(Input), IPL_DEPTH_8U, 1);    //B channel

																	   //cv::Mat test = cv::Mat(Input);
	Mat test = cvarrToMat(Input);

	int i, j, k, l;
	int layer1_index, layer2_index, layer3_index, layer4_index, layer5_index, layer6_index, layer7_index, layer8_index, layer9_index, layer10_index;
	int layer11_index, layer12_index, layer13_index, layer14_index, layer15_index, layer16_index, layer17_index, layer18_index, layer19_index;


	double bicubic2[2][4] = { { -0.0234375, 0.2265625, 0.8671875, -0.0703125 },
	{ -0.0703125, 0.8671875, 0.2265625, -0.0234375 } };

	// Weight
	double* weight_conv1 = 0;  double* weight_conv2 = 0;  double* weight_conv3 = 0;  double* weight_conv4 = 0;  double* weight_conv5 = 0;
	double* weight_conv6 = 0;  double* weight_conv7 = 0;  double* weight_conv8 = 0;  double* weight_conv9 = 0;  double* weight_conv10 = 0;
	double* weight_conv11 = 0; double* weight_conv12 = 0; double* weight_conv13 = 0; double* weight_conv14 = 0; double* weight_conv15 = 0;
	double* weight_conv16 = 0; double* weight_conv17 = 0; double* weight_conv18 = 0; double* weight_conv19 = 0; double* weight_conv20 = 0;

	// Bias
	double* bias_conv1 = 0;  double* bias_conv2 = 0;  double* bias_conv3 = 0;  double* bias_conv4 = 0;  double* bias_conv5 = 0;
	double* bias_conv6 = 0;  double* bias_conv7 = 0;  double* bias_conv8 = 0;  double* bias_conv9 = 0;  double* bias_conv10 = 0;
	double* bias_conv11 = 0; double* bias_conv12 = 0; double* bias_conv13 = 0; double* bias_conv14 = 0; double* bias_conv15 = 0;
	double* bias_conv16 = 0; double* bias_conv17 = 0; double* bias_conv18 = 0; double* bias_conv19 = 0; double* bias_conv20 = 0;

	// Input RGB
	double* input_R; double* input_G; double* input_B;

	// Output RGB
	double* output_R; double* output_G; double* output_B;

	// Input YCbCr
	double* input_Y; double* input_Cb; double* input_Cr;

	// Bicubic YCbCr
	double* bicubic_Y;		double* bicubic_Cb;		 double* bicubic_Cr;
	double* bicubic_Y_temp; double* bicubic_Cb_temp; double* bicubic_Cr_temp;

	double* summation_data;

	// Conv1_result
	double *conv1_result = 0; double *conv2_result = 0; double *conv3_result = 0; double *conv4_result = 0; double *conv5_result = 0;
	double *conv6_result = 0; double *conv7_result = 0; double *conv8_result = 0; double *conv9_result = 0; double *conv10_result = 0;
	double *conv11_result = 0; double *conv12_result = 0; double *conv13_result = 0; double *conv14_result = 0; double *conv15_result = 0;
	double *conv16_result = 0; double *conv17_result = 0; double *conv18_result = 0; double *conv19_result = 0; double *conv20_result = 0;
	FILE *in;

	// Allocation
	/// Low Image
	input_R = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_G = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_B = (double *)malloc(sizeof(double)*Low_h*Low_w);

	input_Y = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_Cb = (double *)malloc(sizeof(double)*Low_h*Low_w);
	input_Cr = (double *)malloc(sizeof(double)*Low_h*Low_w);

	/// High Image
	bicubic_Y = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicubic_Cb = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	bicubic_Cr = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	bicubic_Y_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicubic_Cb_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);
	bicubic_Cr_temp = (double *)malloc(sizeof(double)*HIgh_h*Low_w);

	int64 tStart, tEnd, tStart_total, tEnd_total;
	double TestTime, TestTime_total;

	ui.listWidget->addItem("Initialization : DONE");


	// Input Split to (IplImage*) r, g, b
	cvSplit(Input, b, g, r, NULL);

	// (IplImage*)r, g, b -> (double*)input_R, G, B
	for (j = 0; j < Low_w; j++) {
		for (i = 0; i < Low_h; i++) {
			input_R[i*Low_w + j] = cvGetReal2D(r, i, j);//unsigned char(r->imageData[i*Low_w + j]);
			input_G[i*Low_w + j] = cvGetReal2D(g, i, j);//unsigned char(g->imageData[i*Low_w + j]);
			input_B[i*Low_w + j] = cvGetReal2D(b, i, j);//unsigned char(b->imageData[i*Low_w + j]);
		}
	}

	// ConHigh_ht RGB to YCbCr 


	// (double*)input_R, G, B -> (double*)input_Y, Cb, Cr
	for (j = 0; j < Low_w; j++) {
		for (i = 0; i < Low_h; i++) {
			input_Y[i*Low_w + j] = 0.256788235294118*input_R[i*Low_w + j] + 0.504129411764706*input_G[i*Low_w + j] + 0.0979058823529412*input_B[i*Low_w + j] + 16;
			input_Cb[i*Low_w + j] = -0.148223529411765*input_R[i*Low_w + j] - 0.290992156862745*input_G[i*Low_w + j] + 0.4392156862745100*input_B[i*Low_w + j] + 128;
			input_Cr[i*Low_w + j] = 0.439215686274510*input_R[i*Low_w + j] - 0.367788235294118*input_G[i*Low_w + j] - 0.0714274509803922*input_B[i*Low_w + j] + 128;
		}
	}

	ui.listWidget->addItem("RGB to YCbCr : DONE");

	// Bicubic Interpolation

	for (i = 2; i <= Low_h - 2; i++) { // High_wizontal
		for (j = 0; j < Low_w; j++) {
			bicubic_Y_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Y[i*Low_w + j] + bicubic2[1][3] * input_Y[(i + 1)*Low_w + j] + 0.5);
			bicubic_Y_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Y[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Y[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Y[i*Low_w + j] + bicubic2[0][3] * input_Y[(i + 1)*Low_w + j] + 0.5);

			bicubic_Cb_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cb[i*Low_w + j] + bicubic2[1][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);
			bicubic_Cb_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cb[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cb[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cb[i*Low_w + j] + bicubic2[0][3] * input_Cb[(i + 1)*Low_w + j] + 0.5);

			bicubic_Cr_temp[(2 * i - 1)*Low_w + j] = floor(bicubic2[1][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[1][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[1][2] * input_Cr[i*Low_w + j] + bicubic2[1][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
			bicubic_Cr_temp[(2 * i)*Low_w + j] = floor(bicubic2[0][0] * input_Cr[(i - 2)*Low_w + j] + bicubic2[0][1] * input_Cr[(i - 1)*Low_w + j] + bicubic2[0][2] * input_Cr[i*Low_w + j] + bicubic2[0][3] * input_Cr[(i + 1)*Low_w + j] + 0.5);
		}
	}
	for (i = 0; i < HIgh_h; i++) { // High_htical
		for (j = 2; j <= Low_w - 2; j++) {
			bicubic_Y[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Y_temp[i*Low_w + j + 1]);
			bicubic_Y[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Y_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Y_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Y_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Y_temp[i*Low_w + j + 1]);

			bicubic_Cb[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cb_temp[i*Low_w + j + 1]);
			bicubic_Cb[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cb_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cb_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cb_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cb_temp[i*Low_w + j + 1]);

			bicubic_Cr[i*HIgh_w + (2 * j - 1)] = floor(bicubic2[1][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[1][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[1][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[1][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
			bicubic_Cr[i*HIgh_w + 2 * j] = floor(bicubic2[0][0] * bicubic_Cr_temp[i*Low_w + j - 2] + bicubic2[0][1] * bicubic_Cr_temp[i*Low_w + j - 1] + bicubic2[0][2] * bicubic_Cr_temp[i*Low_w + j] + bicubic2[0][3] * bicubic_Cr_temp[i*Low_w + j + 1]);
		}
	}

	// Interpolation output
	double* interpolation_R;
	double* interpolation_G;
	double* interpolation_B;

	// Initialization interpolation R, G, B
	interpolation_R = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	interpolation_G = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	interpolation_B = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	// (double*)Bicubic_Y, Cb, Cr -> (double*)Interpolation_R, G, B
	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			interpolation_R[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] + 0.00000030112439741101*bicubic_Cb[i*HIgh_w + j] + 1.59602688733570000000*bicubic_Cr[i*HIgh_w + j] - 222.921617109194 + 0.5);
			interpolation_G[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] - 0.39176253994145000000*bicubic_Cb[i*HIgh_w + j] - 0.81296829216220500000*bicubic_Cr[i*HIgh_w + j] + 135.575409522967 + 0.5);
			interpolation_B[i*HIgh_w + j] = floor(1.16438356164384000000*bicubic_Y[i*HIgh_w + j] + 2.01723263955646000000*bicubic_Cb[i*HIgh_w + j] + 0.00000305426174524847*bicubic_Cr[i*HIgh_w + j] - 276.836305795032 + 0.5);
		}
	}

	// (double*)Interpolation_R, G, B -> (IplImage*)interpolationImage_R, G, B
	CvSize cvsize2 = { HIgh_w, HIgh_h };
	IplImage* interpolationImage_R = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);
	IplImage* interpolationImage_G = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);
	IplImage* interpolationImage_B = cvCreateImage(cvsize2, IPL_DEPTH_8U, 1);

	for (int y = 0; y < cvsize2.height; y++) {
		for (int x = 0; x < cvsize2.width; x++) {
			//cvSetReal2D(TempImage1, y, x, conv3_data[ y*cvsize1.width+x]);
			cvSetReal2D(interpolationImage_R, y, x, interpolation_R[y*cvsize2.width + x]);
			cvSetReal2D(interpolationImage_G, y, x, interpolation_G[y*cvsize2.width + x]);
			cvSetReal2D(interpolationImage_B, y, x, interpolation_B[y*cvsize2.width + x]);
		}
	}


	ui.listWidget->addItem("Bicubic Interpolation : Done");
	tStart_total = cvGetTickCount();

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	double* total_dummy;
	total_dummy = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w*Layer_CH_Num);

	for (int layer_idx = 0; layer_idx < Layer_CH_Num; layer_idx++) {
		for (i = 0; i < HIgh_h; i++) {
			for (j = 0; j < HIgh_w; j++) {
				total_dummy[layer_idx*HIgh_h*HIgh_w + i*HIgh_w + j] = 0;
			}
		}
	}

	Convolutional_layer1(bicubic_Y, conv1_result, weight_conv1, bias_conv1, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);

	Convolutional_layer_2_to_19(2, conv1_result, total_dummy, conv2_result, weight_conv2, bias_conv2, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(3, conv2_result, total_dummy, conv3_result, weight_conv3, bias_conv3, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(4, conv3_result, total_dummy, conv4_result, weight_conv4, bias_conv4, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(5, conv4_result, total_dummy, conv5_result, weight_conv5, bias_conv5, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(6, conv5_result, total_dummy, conv6_result, weight_conv6, bias_conv6, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(7, conv6_result, total_dummy, conv7_result, weight_conv7, bias_conv7, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(8, conv7_result, total_dummy, conv8_result, weight_conv8, bias_conv8, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(9, conv8_result, total_dummy, conv9_result, weight_conv9, bias_conv9, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(10, conv9_result, total_dummy, conv10_result, weight_conv10, bias_conv10, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(11, conv10_result, total_dummy, conv11_result, weight_conv11, bias_conv11, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(12, conv11_result, total_dummy, conv12_result, weight_conv12, bias_conv12, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(13, conv12_result, total_dummy, conv13_result, weight_conv13, bias_conv13, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(14, conv13_result, total_dummy, conv14_result, weight_conv14, bias_conv14, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(15, conv14_result, total_dummy, conv15_result, weight_conv15, bias_conv15, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(16, conv15_result, total_dummy, conv16_result, weight_conv16, bias_conv16, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(17, conv16_result, total_dummy, conv17_result, weight_conv17, bias_conv17, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(18, conv17_result, total_dummy, conv18_result, weight_conv18, bias_conv18, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);
	Convolutional_layer_2_to_19(19, conv18_result, total_dummy, conv19_result, weight_conv19, bias_conv19, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, TestTime_total_CUDA);

	Convolutional_layer20(conv19_result, conv20_result, weight_conv20, bias_conv20, HIgh_w, HIgh_h, Layer_Filter_Size, Layer_CH_Num, Layer20_CH_Num, TestTime_total_CUDA);

	summation_data = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			summation_data[i*HIgh_w + j] = (conv20_result[i*HIgh_w + j] + (bicubic_Y[i*HIgh_w + j] / 255)) * 255;
		}
	}

	ui.listWidget->addItem("Summation : : Done");

	output_R = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	output_G = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);
	output_B = (double *)malloc(sizeof(double)*HIgh_h*HIgh_w);

	for (j = 0; j < HIgh_w; j++) {
		for (i = 0; i < HIgh_h; i++) {
			output_R[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] + 0.00000030112439741101*bicubic_Cb[i*HIgh_w + j] + 1.59602688733570000000*bicubic_Cr[i*HIgh_w + j] - 222.921617109194 + 0.5);
			output_G[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] - 0.39176253994145000000*bicubic_Cb[i*HIgh_w + j] - 0.81296829216220500000*bicubic_Cr[i*HIgh_w + j] + 135.575409522967 + 0.5);
			output_B[i*HIgh_w + j] = floor(1.16438356164384000000*summation_data[i*HIgh_w + j] + 2.01723263955646000000*bicubic_Cb[i*HIgh_w + j] + 0.00000305426174524847*bicubic_Cr[i*HIgh_w + j] - 276.836305795032 + 0.5);
		}
	}
	ui.listWidget->addItem("YCbCr to RGB : Done");

	CvSize cvsize1 = { HIgh_w, HIgh_h };
	IplImage* TempImage1 = cvCreateImage(cvsize1, IPL_DEPTH_8U, 3);
	IplImage* TempImage_R = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);
	IplImage* TempImage_G = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);
	IplImage* TempImage_B = cvCreateImage(cvsize1, IPL_DEPTH_8U, 1);

	for (int y = 0; y < cvsize1.height; y++) {
		for (int x = 0; x < cvsize1.width; x++) {
			//cvSetReal2D(TempImage1, y, x, conv3_data[ y*cvsize1.width+x]);
			cvSetReal2D(TempImage_R, y, x, output_R[y*cvsize1.width + x]);
			cvSetReal2D(TempImage_G, y, x, output_G[y*cvsize1.width + x]);
			cvSetReal2D(TempImage_B, y, x, output_B[y*cvsize1.width + x]);
		}
	}
	cvMerge(TempImage_B, TempImage_G, TempImage_R, NULL, TempImage1);

	cvReleaseImage(&TempImage_R);
	cvReleaseImage(&TempImage_G);
	cvReleaseImage(&TempImage_B);

	// Output Path
	string out_path = output_dir_path.toLocal8Bit().constData();
	out_path.append("/output_VDSR_CUDA_");
	out_path.append(Input_File_Name);

	// Create char * output path
	char * Output_Path_char_pointer;
	Output_Path_char_pointer = new char[out_path.size() + 1];
	std::copy(out_path.begin(), out_path.end(), Output_Path_char_pointer);
	Output_Path_char_pointer[out_path.size()] = '\0';

	// Store Image
	cvSaveImage(Output_Path_char_pointer, TempImage1);

	// Display Image
	// ConHigh_ht IplImage* to Mat Image
	Mat VDSR_Serial = cvarrToMat(TempImage1);//interpolationImage1

											 // Display output
	QImage qimgSerial = convertOpenCVMatToQtQImage(VDSR_Serial);
	ui.label_11->setPixmap(QPixmap::fromImage(qimgSerial.scaled(280, 280, Qt::KeepAspectRatio)));

	ui.listWidget->addItem("TestTime_total_CUDA : " + QString::number(TestTime_total_CUDA));

	tEnd = cvGetTickCount();// for check processing
	tEnd_total = 0.001 * (tEnd - tStart_total) / cvGetTickFrequency(); // for msec

																	   // Display Input Box
	ui.label_33->setText(QString::number(tEnd_total) + " ms  ");

	Mat output_Image = cvarrToMat(TempImage1);
	Mat Origianl_image = imread(input_file_path.toLocal8Bit().constData(), 1);
	double PSNR = getPSNR(Origianl_image, output_Image);
	ui.label_35->setText(QString::number(PSNR) + " dB  ");
}


