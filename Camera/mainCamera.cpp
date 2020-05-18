#include <iostream>
#include "mainCamera.hpp"

/* ########################
 * Name: mainCamera.cpp
 * Made by: Thor Christiansen - s173949 
 * Data: 18.05.2020
 * Objective: The source file mainCamera.cpp contains the functions used
 * by main.cpp to treat the images - find features in the images using 
 * Harris corner etc.
 * Project: Bachelor project 2020
 * ########################
*/

using namespace std;

/*
void draw::circles() {
	// Draw circles
	std::cout << "Start of main File!\n";
	//piCamera.set(cv::CAP_PROP_FRAME_WIDTH, (float)640 );
	
	Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	imshow("Display Window", image);
	waitKey(0);
}
*/


void Harris::corner(Mat src, Mat src_gray) {
	
	// Define variables
	const char* corners_window = "Corners detected";
	
	// Define variables related to Harris corner
	int blockSize = 2; 
	int apertureSize = 3;
	double k = 0.04;		// Magic parameter 
	int thres = 200;
	
	Mat dst = Mat::zeros( src.size(), CV_32FC1 );
	cornerHarris (src_gray, dst, blockSize, apertureSize, k);

	
	Mat dst_norm, dst_norm_scaled;
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	for (int i = 0; i < dst_norm.rows; i++) {
		for (int j = 0; j < dst_norm.cols; j++) {
			if ( (int) dst_norm.at<float>(i,j) > thres) {
				//circle (dst_norm_scaled, Point(j,i), 5, Scalar(0), 2,8,0);
				circle (src, Point(j,i), 5, Scalar(200), 2,8,0);
			}

		}

	}
	
	namedWindow( corners_window) ; 
	imshow( corners_window, src);
	
	return; 
}





