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

// Heap operations
Matrix insertNodeHeap(Matrix Corners, int value, int y, int x, int n) {
	Corners(n,0) = value;
	Corners(n,1) = y;
	Corners(n,2) = x;
	// Bubble up 
	int index = floor(n/2);
	while(Corners(index,0) < Corners(n,0) && index >= 1) {
		Corners.swap_rows(index,n);
		n = index;
		index = floor(n/2);
	}
	
	return Corners;
	
}

void printfunction(Matrix Corners, int n) {
	cout << "Start printing" << endl;
	for (int i = 0; i <= n; i ++) {
		cout << "Corners intensity: " << Corners(i,0) << " at (" << Corners(i,1) << "," << Corners(i,2) << ")" << endl;
	}
	
}

Matrix extractMaxHeap(Matrix Corners, int n) {
	//cout << "Inside max-extraction function" << endl;
	
	/*
	Matrix firstRow(1,3);
	int value = Corners(1,0);
	int y = Corners(1,1);
	int x = Corners(1,2);
	firstRow(0,0) = value;
	firstRow(0,1) = y;
	firstRow(0,2) = x;
	Corners[1] = Corners[n];
	*/
	Corners.swap_rows(1,n);
	
	
	//cout << "First row updated " << endl;
	//Corners[1] = Corners[n];
	n--;
	// Bubble down
	int index = 1;
	cout << "Begin bubble down" << endl;
	while ((2*index+1) <= n && (Corners(index,0) < Corners(2*index,0) || Corners(index,0) < Corners(2*index+1,0))) {
		cout << "Bubble down continued" << endl;
		if (Corners(2*index,0) > Corners(2*index+1,0)) {
			cout << "bubble left" << endl;
			Corners.swap_rows(index,2*index);
			index = 2*index;
		}
		else {
			cout << "bubble right" << endl;
			Corners.swap_rows(index,2*index+1);
			index = 2*index+1;
		}
	}
	//cout << "Done with while loop extractMaxHeap " << endl;
	//cout << "Max returned as = (" << Corners(n+1,1) << "," << Corners(n+1,2) << ")" << endl;
	//return firstRow;
	cout << "Print inside extract" << endl;
	printfunction(Corners,n+1);
	return Corners;
}


Matrix Harris::corner(Mat src, Mat src_gray, bool display) {
	
	// Define variables
	const char* corners_window = "Corners detected";
	
	// Define variables related to Harris corner
	int blockSize = 2; 
	int apertureSize = 3;
	double k = 0.04;		// Magic parameter 
	int thres = 200;
	
	// Variables related to Non Maximum suppression 
	int NMSBox = 5;
	
	Mat dst = Mat::zeros( src.size(), CV_32FC1 );
	cornerHarris (src_gray, dst, blockSize, apertureSize, k);

	
	Mat dst_norm, dst_norm_scaled;
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	
	
	// Find corners and return 
	if (display == false) {
		int nr_corners = 0;
		
		// We only focus on the 200 biggest keypoints
		int keypoints_limit = 200; // Change to 200
		Matrix Corners(keypoints_limit,3); // Column1: Corner responses, Column2: Pixel i, Column3: Pixel j
		
		//int responses[keypoints_limit];
		//int index_i[keypoints_limit]; // In theory there might be more keypoints?
		//int index_j[keypoints_limit];
		
		cout << "Number of rows: " << dst_norm.rows << endl;
		cout << "Number of columns: " << dst_norm.cols<< endl;
		int CornerResponse = 0;
		int maxCornerResponse = 0;
		Corners(0,0) = 0;
		Corners(1,0) = 0;
		
		for (int i = 0; i < dst_norm.rows; i++) {
			for (int j = 0; j < dst_norm.cols; j++) {
				
				CornerResponse = (int) dst_norm.at<float>(i,j);
				if ( CornerResponse > thres && nr_corners < keypoints_limit-1) {
					cout << "intensity for corner " << nr_corners << " at (" << i << "," << j << "): " << (int) dst_norm.at<float>(i,j) << endl;
					// Insert node in heap
					Corners = insertNodeHeap(Corners, CornerResponse,i,j,nr_corners+1);
					cout << "Heap Corners intensity " << Corners(nr_corners+1,0)  << " at (" << Corners(nr_corners+1,1) << "," << Corners(nr_corners+1,2) << ")" << endl;
					nr_corners++;
				}
					
				/*	
					
					int index = 0;
					discardPoint = false;
					while (CornerResponse < Corners(i,0)) {
						index++;
						pixel_i = Corners(i,1);
						pixel_j = Corners(i,2);
						if ((i >= pixel_i-NMSBox && i <= pixel_i+NMSBox) && (j >= pixel_j-NMSBox && j <= pixel_j+NMSBox)) {
							discardPoint = true;
							break;
						}
					}
					if (discardPoint == false) {
						Matrix temp = Corners[index];
						Corners(index,0) = CornerResponse;
						Corners(index,1) = i;
						Corners(index,2) = j;
						
						nr_corners ++;
						Corners(nr_corners,0) = 0;
						pixel_i = temp(0,1);
						pixel_j = temp(0,2);
						if ((i >= pixel_i-NMSBox && i <= pixel_i+NMSBox) && (j >= pixel_j-NMSBox && j <= pixel_j+NMSBox)) {
							// Check the rest  
						}
						else {
							// Switch and check the rest. 
							Matrix temp2 = Corners[index + 1];
							Corners[index+1] = temp;
							temp = temp2;
							for (int m = index + 1; m <= nr_corners; m++) {
								if ()
								
							}
						}
						for (int m = index+1; m <= nr_corners; m++) {
							Matrix temp2 = Corners[m];
							
							int temp_response = responses[m];
							int temp
							PrevCornerResponse;
							 
						} 
					}
				}
				*/
			}
		}
		printfunction(Corners,nr_corners);
		
		//waitKey(0);
		cout << "Max in heap: " << Corners(1,0) << " at (" << Corners(1,1) << "," << Corners(1,2) << ")" << endl;
		//nr_corners--;
		// Maybe reduce size of corners Corners = Corners.slice(0,nr_corners);
		cout << "Number of corners: " << nr_corners << endl;
		Matrix keypoints(nr_corners,3); // Maybe you don't need to store the intensity value too?
		keypoints(1,1) = 0;
		keypoints(1,2) = 0;
		int n = nr_corners;
		bool Discard_Point = false;
		int corners_left = 0;
		// Extract maximum and automatically enforce non-maximum suppression 
		cout << "Going in to for loop" << endl;
		for (int m = 0; m < nr_corners; m++) {
			//cout << "Begin deleting corners" << endl;
			keypoints(m,1) = 0;
			keypoints(m,2) = 0;
			cout << "Before max extraction. Parameter n is: " << n << endl;
			cout << "Parameter m is = " << m << endl;
			//Matrix temp = extractMaxHeap(Corners, n);
			Corners = extractMaxHeap(Corners, n);
			//cout << "Max extracted" << endl;
			cout << "n is here: " << n << endl;
			int intensity = Corners(n,0);
			int y_temp = Corners(n,1);
			int x_temp = Corners(n,2);
			n--;
			cout << "Max intensity: " << intensity << " at (x_temp,y_temp) = " << x_temp << "," << y_temp << endl;
			//waitKey(0);
			Discard_Point = false;
			for (int l = 0; l < m; l++) {
				int y = keypoints(l,1);
				int x = keypoints(l,2);
				cout << "(x,y) = " << x << "," << y << endl;
				if ((x_temp >= x - NMSBox && x_temp <= x + NMSBox) && (y_temp >= y-NMSBox && y_temp <= y+NMSBox)) {
					cout << "Point discarted" << endl;
					cout << "(x_temp,y_temp) = (" << x_temp << "," << y_temp << ")" << endl;
					cout << "(x,y) = (" << x << "," << y << ")" << endl;
					//waitKey(0);
					Discard_Point = true;
				}
			}
			if (Discard_Point == false) {
				cout << "Point being inserted" << endl;
				//keypoints[m] = temp;
				keypoints(m,0) = intensity;
				keypoints(m,1) = y_temp;
				keypoints(m,2) = x_temp;
				corners_left++;
				cout << "Keypoints printed" << endl;
				printfunction(keypoints,corners_left);
			}
		}
		cout << "Number of corners left: " <<  corners_left << endl;
		
		/*
		//cout << "Number of corners before loop: " << nr_corners << endl;
		cout << "Number of matrix rows: " << keypoints.dim1() << endl;
		cout << "Number of matrix columns: " << keypoints.dim2() << endl;
		for (int k = 0; k < nr_corners; k++) {
			cout << "k = " << k << endl;
			keypoints(0,k) = index_i[k];
			keypoints(1,k) = index_j[k];
			cout << "Keypoint updated: " << keypoints(1,k) << endl;
		}
		cout << "Number of corners: " << nr_corners << endl;
		//cout << "Corner end: (" << interest_points[1][nr_corners] << "," << interest_points[2][nr_corners] << ")" << endl;
		cout << "Corner end matrix: (" << keypoints(0,nr_corners-1) << "," << keypoints(1,nr_corners-1) << ")" << endl;
		cout << "mainCamera. Dimensioner: (" << keypoints.dim1() << "," << keypoints.dim2() << ")" << endl;
		*/
		
		return keypoints.slice(0,corners_left);
	}
	
	
	if (display == true) {
		for (int i = 0; i < dst_norm.rows; i++) {
			for (int j = 0; j < dst_norm.cols; j++) {
				if ( (int) dst_norm.at<float>(i,j) > thres) {
					//circle (dst_norm_scaled, Point(j,i), 5, Scalar(0), 2,8,0);
					circle (src, Point(j,i), 5, Scalar(200), 2,8,0);
				}
			}
		}
		cout << "Display function" << endl;
		namedWindow( corners_window) ; 
		imshow( corners_window, src);
		waitKey(0);
		Matrix emptyArray(1,1);
		
		
		return emptyArray;
	}
	Matrix emptyArray(1,1);
	cout << "End of Harris " << endl;	
		
	return emptyArray;
	
}





