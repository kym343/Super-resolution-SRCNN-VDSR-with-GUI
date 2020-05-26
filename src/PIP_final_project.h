#pragma once

#include <QtWidgets/QApplication>
#include <QPushButton>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QtWidgets/QMainWindow>
#include "ui_PIP_final_project.h"
#include "atlstr.h"
#include <Windows.h>
#include <qlistwidget.h>

class PIP_final_project : public QMainWindow
{
	Q_OBJECT

public:
	PIP_final_project(QWidget *parent = Q_NULLPTR);
	QImage PIP_final_project::convertOpenCVMatToQtQImage(cv::Mat mat);
	

private:
	Ui::PIP_final_projectClass ui;
	

private slots:
	void selectInputFile();
	void selectOutputDir();
	void slotDisplay_Bicubic_Interpolation();
	void slotDisplay_Srcnn_CUDA();
	void slotDisplay_VDSR_Serial();
	void slotDisplay_VDSR_OpenMP();
	void slotDisplay_VDSR_CUDA();
};

	