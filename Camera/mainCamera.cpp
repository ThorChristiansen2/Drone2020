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

void MatType( Mat inputMat ) {
	
    int inttype = inputMat.type();

    string r, a;
    uchar depth = inttype & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (inttype >> CV_CN_SHIFT);
    switch ( depth ) {
        case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;  
        case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;  
        case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break; 
        case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break; 
        case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break; 
        case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break; 
        case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break; 
        default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break; 
    }   
    r += "C";
    r += (chans+'0');
    cout << "Mat is of type " << r << " and should be accessed with " << a << endl;
	
}

// Select region of interest from image 
Mat selectRegionOfInterest(Mat img, int y1, int x1, int y2, int x2) {
	Mat ROI;
	if (x1 < 0) {
		x1 = 0;
	}
	if (x1 > img.cols) {
		x1 = img.cols;
	}
	if (y1 < 0) {
		y1 = 0;
	}
	if (y1 > img.rows) {
		y1 = img.rows;
	}
	if (y2 > img.rows) {
		y2 = img.rows;
	}
	if (x2 > img.cols) {
		x2 = img.cols;
	}
	cout << "Rectangle : (" << x1 << "," << y1 << "," << y2-y1 << "," << x2-x1 << ")" << endl;
	//Rect region(y1, x1, x2-x1, y2-y1);
	//Rect region(x1,y1,y2-y1,x2-x1);
	Rect region(x1,y1,y2-y1,x2-x1);
	ROI = img(region);
	cout << "ROI is: " << endl;
	MatType(ROI);
	waitKey(0);
	cout << "Draw ROI" << endl;
	for (int hh = 0; hh <= ROI.rows; hh++) {
		for (int jj = 0; jj <= ROI.cols; jj++) {
			cout << (int) ROI.at<uchar>(hh,jj) << ", ";
		}
		cout << "" << endl;
	} 
	cout << "Region extracted" << endl;
	return ROI;
}





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
	Corners.swap_rows(1,n); // Exchange max and last row
	n--;
	// Bubble down
	int index = 1;
	//cout << "Begin bubble down" << endl;
	while ((2*index+1) <= n && (Corners(index,0) < Corners(2*index,0) || Corners(index,0) < Corners(2*index+1,0))) {
		//cout << "Bubble down continued" << endl;
		if (Corners(2*index,0) > Corners(2*index+1,0)) {
			//cout << "bubble left" << endl;
			Corners.swap_rows(index,2*index);
			index = 2*index;
		}
		else {
			//cout << "bubble right" << endl;
			Corners.swap_rows(index,2*index+1);
			index = 2*index+1;
		}
	}
	return Corners;
}


Matrix Harris::corner(Mat src, Mat src_gray, bool display) {
	
	// Define variables
	const char* corners_window = "Corners detected";
	
	// Define variables related to Harris corner
	int blockSize = 4; 
	int apertureSize = 5;
	double k = 0.04;		// Magic parameter 
	int thres = 200;	
	// Parameters before: blocksize = 2, aperturesize = 3, thres = 200, k = 0.04
	
	// Variables related to Non Maximum suppression 
	int NMSBox = 5;
	int boundaries = 15; 
	
	Mat dst = Mat::zeros( src.size(), CV_32FC1 );
	cornerHarris (src_gray, dst, blockSize, apertureSize, k);

	Mat dst_norm, dst_norm_scaled;
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	// Find corners and return 
	if (display == false) {
		int nr_corners = 0;
		
		int keypoints_limit = 200; // Change to 200
		Matrix Corners(keypoints_limit,3); // Column1: Corner responses, Column2: Pixel i, Column3: Pixel j
		
		int CornerResponse = 0;
		int maxCornerResponse = 0;
		Corners(0,0) = 0;
		Corners(1,0) = 0;
		
		for (int i = 0+boundaries; i < dst_norm.rows-boundaries; i++) {
			for (int j = 0+boundaries; j < dst_norm.cols-boundaries; j++) {
				
				CornerResponse = (int) dst_norm.at<float>(i,j);
				if ( CornerResponse > thres && nr_corners < keypoints_limit-1) {
					//cout << "intensity for corner " << nr_corners << " at (" << i << "," << j << "): " << (int) dst_norm.at<float>(i,j) << endl;
					// Insert node in heap
					Corners = insertNodeHeap(Corners, CornerResponse,i,j,nr_corners+1);
					//cout << "Heap Corners intensity " << Corners(nr_corners+1,0)  << " at (" << Corners(nr_corners+1,1) << "," << Corners(nr_corners+1,2) << ")" << endl;
					nr_corners++;
				}
			}
		}
		printfunction(Corners,nr_corners);
		
		//waitKey(0);
		//cout << "Max in heap: " << Corners(1,0) << " at (" << Corners(1,1) << "," << Corners(1,2) << ")" << endl;
		//nr_corners--;
		// Maybe reduce size of corners Corners = Corners.slice(0,nr_corners);
		//cout << "Number of corners: " << nr_corners << endl;
		//waitKey(0);
		Matrix keypoints(nr_corners,3); // Maybe you don't need to store the intensity value too?
		keypoints(1,1) = 0;
		keypoints(1,2) = 0;
		int n = nr_corners;
		bool Discard_Point = false;
		int corners_left = 0;
		// Extract maximum and automatically enforce non-maximum suppression 
		//cout << "Going in to for loop" << endl;
		for (int m = 0; m < nr_corners; m++) {
			//cout << "Begin deleting corners" << endl;
			keypoints(m,1) = 0;
			keypoints(m,2) = 0;
			//cout << "Before max extraction. Parameter n is: " << n << endl;
			//cout << "Parameter m is = " << m << endl;
			Corners = extractMaxHeap(Corners, n);
			//cout << "Max extracted" << endl;
			//cout << "n is here: " << n << endl;
			int intensity = Corners(n,0);
			int y_temp = Corners(n,1);
			int x_temp = Corners(n,2);
			n--;
			//cout << "Max intensity: " << intensity << " at (x_temp,y_temp) = " << x_temp << "," << y_temp << endl;
			Discard_Point = false;
			for (int l = 0; l <= m; l++) { // Maybe you should change m to corners_left
			//for (int l = 0; l < m; l++) {
				int y = keypoints(l,1);
				int x = keypoints(l,2);
				//cout << "(x,y) = " << x << "," << y << endl;
				if ((x_temp >= x - NMSBox && x_temp <= x + NMSBox) && (y_temp >= y-NMSBox && y_temp <= y+NMSBox)) {
					// Discard point if it is within another maximum
					Discard_Point = true;
					break;
				}
			}
			if (Discard_Point == false) {
				//cout << "Point being inserted" << endl;
				keypoints(corners_left,0) = intensity;
				keypoints(corners_left,1) = y_temp;
				keypoints(corners_left,2) = x_temp;
				corners_left++;
				//cout << "Keypoints printed" << endl;
				//printfunction(keypoints,corners_left);
			}
		}
		cout << "Number of corners left: " <<  corners_left << endl;
		return keypoints.slice(0,corners_left);
	}
	cout << "End of Harris " << endl;	
	Matrix emptyArray(1,3);	
	return emptyArray;
}

// Find SIFT Desriptors 
Matrix SIFT::FindDescriptors(Mat src, Matrix keypoints) {
	int n = keypoints.dim1();
	
	// Initialize matrix containing keypoints descriptors
	Matrix Descriptors(n,128);
	
	cout << "Dimensions of image (rows,cols) = (" << src.rows << "," << src.cols << ")" << endl;
	waitKey(0);
	// Find Image gradients
	
	
	cout << "Inside SIFT" << endl;
	for (int i = 0; i < 1; i++) {
		int y = keypoints(i,1);
		int x = keypoints(i,2); 
		circle (src, Point(x,y), 5, Scalar(7), 2,8,0);
		cout << "Keypoint at (y = " << y << ",x = " << x << ")" << endl;
		cout << "Intensity at keypoint :" << (int) src.at<uchar>(y,x) << endl;
		cout << "Intensity at pixel left of keypoint :" << (int) src.at<uchar>(y,x-1) << endl;
		cout << "Intensity at pixel right of keypoint :" << (int) src.at<uchar>(y,x+1) << endl;
		
		cout << "Draw image" << endl;
		for (int k = y-7; k<= y+8; k++) {
				for (int j = x-7; j<= x+8; j++) {
					cout << (int) src.at<uchar>(k,j) << ", ";
				}
				cout << "" << endl;
		}
		
		waitKey(0);
		// Patch of size 16,16
		Mat Patch;
		Patch = selectRegionOfInterest(src, y-7, x-7, y+8, x+8);
		cout << "Number of rows: " << Patch.rows << endl;
		cout << "Number of columns: " << Patch.cols << endl;
		
		
		cout << "Draw patch " << endl;
		for (int k = 0; k<= Patch.rows; k++) {
			for (int j = 0; j<= Patch.cols; j++) {
				cout << (int) Patch.at<uchar>(k,j) << ", ";
			}
			cout << "" << endl;
		}
		waitKey(0);
		Rect region(x-7,y-7,15,15);
		cout << "Draw square " << endl;
		rectangle(src, region, Scalar(255), 1, 8, 0);
		imshow ("image with square", src);
		waitKey(0);
	
	// Scale the norms of the gradients by multiplying a the graidents with a gaussian
	// centered in the keypoint and with Sigma_w = 1.5*16. 
	
	
	
	}
	
	return Descriptors;
}



