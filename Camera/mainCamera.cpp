#include <iostream>
#include "mainCamera.hpp"
//#include "Matrix.h"

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
//using namespace Numeric_lib;

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


void Harris::corner(Mat src, Mat src_gray, bool display) {
	
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

	
	
	// Find corners and return 
	if (display == false) {
		int nr_corners = 0;
		int index_i[dst_norm.rows]; // In theory there might be more keypoints?
		int index_j[dst_norm.cols];
		
		cout << "Number of rows: " << dst_norm.rows << endl;
		cout << "Number of columns: " << dst_norm.cols<< endl;
		for (int i = 0; i < dst_norm.rows; i++) {
			for (int j = 0; j < dst_norm.cols; j++) {
				if ( (int) dst_norm.at<float>(i,j) > thres) {
					cout << "intensity: " << (int) dst_norm.at<float>(i,j) << endl;
					index_i[nr_corners] = i;
					index_j[nr_corners] = j;
					nr_corners ++;
				}
			}
		}
		nr_corners--;
		//int interest_points[2][nr_corners];
		Matrix<int,2>keypoints(2,nr_corners);
	
		
		//cout << "Number of corners before loop: " << nr_corners << endl;
		cout << "Number of matrix rows: " << keypoints.dim1() << endl;
		cout << "Number of matrix columns: " << keypoints.dim2() << endl;
		for (int k = 0; k < nr_corners; k++) {
			//interest_points[0][k] = index_i[k]; // This should maybe be changed to a zero
			//interest_points[1][k] = index_j[k];
			//keypoints(0,k) = 5;
			cout << "k = " << k << endl;
			keypoints(0,k) = index_i[k];
			keypoints(1,k) = index_j[k];
			cout << "Keypoint updated: " << keypoints(1,k) << endl;
		}
		cout << "Number of corners: " << nr_corners << endl;
		//cout << "Corner end: (" << interest_points[1][nr_corners] << "," << interest_points[2][nr_corners] << ")" << endl;
		cout << "Corner end matrix: (" << keypoints(0,nr_corners-1) << "," << keypoints(1,nr_corners-1) << ")" << endl;
		
		//return interest_points;
	}
	
	
	if (display) {
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
		
		int emptyArray[1][1];
		
		//return emptyArray;
	}
	
	//return; 
}





