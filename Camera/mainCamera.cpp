#include <iostream>
#include "mainCamera.hpp"
#include <limits> 
//#include "Matrix.h"

/* ########################
 * Name: mainCamera.cpp
 * Made by: Thor Christiansen - s173949 
 * Date: 18.05.2020
 * Objective: The source file mainCamera.cpp contains the functions used
 * by main.cpp to treat the images - find features in the images using 
 * Harris corner etc.
 * Project: Bachelor project 2020
 * ########################
*/

using namespace std;
//using namespace Numeric_lib;

Matrix SIFT::matchDescriptors(Matrix descriptor1, Matrix descriptor2) {

	int n1 = descriptor1.dim1();	// Matrix containing descriptors for keypoints in image 0
	int n2 = descriptor2.dim1();	// Matrix containing descriptors for keypoints in image 1
	
	// Threshold on the euclidean distance between keypoints 
	double threshold = 400;
	
	// Match keypoints in frame 2 to keypoints in frame 1 
	Matrix matches(n2,5);
	Matrix multiple_matches(n1,2);
	
	int count = 0;
	for (int i = 0; i < n2; i++) {
		double SSD;
		double min = std::numeric_limits<double>::infinity();
		double match;
		// Just to initialize the values
		matches(i,0) = 0;
		matches(i,1) = min;
		matches(i,2) = 0;
		matches(i,3) = min;
		matches(i,4) = 0;
		for (int j = 0; j < n1; j++) {
			SSD = 0;
			for (int k = 0; k < 128; k++) {
				SSD = SSD + pow((descriptor2(i,k)-descriptor1(j,k)),2);
			}
			// If closest neighbour is detected 
			if (SSD < matches(i,1)) {
				matches(i,2) = matches(i,0);
				matches(i,3) = matches(i,1);
				matches(i,1) = SSD;
				matches(i,0) = j;
				if (multiple_matches(j,1) == 0) {
					multiple_matches(j,0) = i;
					multiple_matches(j,1) = SSD;
					matches(i,4) = 0;
				}
				else if (SSD < multiple_matches(j,1)) {
					int temp_index = multiple_matches(j,0);
					multiple_matches(j,0) = i;
					multiple_matches(j,1) = SSD;
					if (matches(temp_index,0) == j) {
						matches(temp_index,4) = 2;
					}
					matches(i,4) = 0;
				}
				else {
					//matches(i,4) = 2; --> Should maybe be enabled
				}
			}
			// The second closest neighbour 
			else if (SSD < matches(i,3)) {
				matches(i,2) = j;
				matches(i,3) = SSD;
			}
		}
	}
	for (int i = 0; i < n2; i++) {
		double distance_ratio = matches(i,1)/matches(i,3);
		if (distance_ratio < 0.8 && matches(i,1) < threshold) {
			//matches(i,4) = 1;
			if (matches(i,4) != 2) {
				matches(i,4) = 1;
				count++;
			}
			
		}
		cout << "Keypoint " << i << " : " << matches(i,0)  << ", " << matches(i,1) << ", " << matches(i,2) << ", " << matches(i,3) << " Match = " << matches(i,4) <<  endl;
	}
	
	// Create matrix with the keypoints that are valid and which are returned.
	Matrix valid_matches(count,2);
	cout << "Count = " << count << endl;
	int index = 0;
	for (int i = 0; i < n2; i++) {
		if (matches(i,4) == 1) {
			valid_matches(index,0) = i;
			valid_matches(index,1) = matches(i,0);
			index++;
		}
	}
	
	// Print the keypoints
	for (int i = 0; i < valid_matches.dim1(); i++) {
		cout << "Keypoint " << valid_matches(i,0) << " match with " << valid_matches(i,1) << endl;
	}
	
	return valid_matches;
}

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
	/* Make a region of the proper size. 
	 * Start point in point (x1,y1) width = y2-y1 and height x2-x1. 
	 */ 
	Rect region(x1,y1,y2-y1,x2-x1);
	
	// Extract the rectangle from the image
	ROI = img(region);
	
	return ROI;
}

// Function to initialize GaussWindow. 
Mat gaussWindow(int filter_size, float sigma) {
	
	float size = filter_size;
	float norm_const = 246.7938;
	
	Mat g = Mat::zeros(Size(filter_size,filter_size),CV_64FC1);

	float x0 = (size+1)/2;
	float y0 = (size+1)/2;
	for (float i = -(size-1)/2; i <= (size-1)/2; i++) {
		for (float j = -(size-1)/2; j <= (size-1)/2; j++) {
			float x = i + x0;
			float y = j + y0;
			int x_coor = x-1;
			int y_coor = y-1;
			g.at<double>(y_coor,x_coor) = (exp(-( (x-x0)*(x-x0) + (y-y0)*(y-y0))/(2*sigma*sigma)))/norm_const;
		}
	}
	
	return g;
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
	while ((2*index+1) <= n && (Corners(index,0) < Corners(2*index,0) || Corners(index,0) < Corners(2*index+1,0))) {
		if (Corners(2*index,0) > Corners(2*index+1,0)) {
			Corners.swap_rows(index,2*index);
			index = 2*index;
		}
		else {
			Corners.swap_rows(index,2*index+1);
			index = 2*index+1;
		}
	}
	return Corners;
}


Matrix Harris::corner(Mat src, Mat src_gray) {
	
	// Define variables
	const char* corners_window = "Corners detected";
	
	// Maybe use minMaxLoc(img, &minVal, &maxVal); for at finde max og minimum, som gemmes i værdierne minVal og maxVal. 	
	
	// Define variables related to Harris corner
	int blockSize = 9; 
	int apertureSize = 3;
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
	bool display = false;
	if (display == false) {
		int nr_corners = 0;
		
		int keypoints_limit = 1000; // Change to 200
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
		cout << "Number of corners: " << nr_corners << endl;
		//printfunction(Corners,nr_corners);
		
		
		// Maybe reduce size of corners Corners = Corners.slice(0,nr_corners);
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
			Corners = extractMaxHeap(Corners, n);
			int intensity = Corners(n,0);
			int y_temp = Corners(n,1);
			int x_temp = Corners(n,2);
			n--;
			Discard_Point = false;
			for (int l = 0; l <= m; l++) { // Maybe you should change m to corners_left
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

// Part of the SIFT function - Rearrange the histogram, so the max representation is the first element. 
Matrix circularShift(Matrix histogram) {
	float max = 0;
	int index = 0;
	for (int ii = 0; ii < histogram.dim2(); ii++) {
		double value = histogram(0,ii);
		if (value > max) {
			max = histogram(0,ii);
			index = ii;
		}
	}

	Matrix temp(1,index);
	// This could maybe be done faster
	// Save first part of histogram  
	for (int ii = 0; ii < index; ii++) {
		double v = histogram(0,ii);
		//cout << "Value given" << endl;
		temp(0,ii) = v;
	}

	for (int ii = index; ii < histogram.dim2(); ii++) {
		histogram(0,ii-index) = histogram(0,ii);
	}
	
	for (int ii = histogram.dim2()-index; ii < histogram.dim2(); ii++) {
		//cout << "ii = " << ii << endl;
		//cout << "histogram.dim2()-index = " << histogram.dim2()-index << endl;
		double v = temp(0,ii-(histogram.dim2()-index));
		//cout << "v = " << v << endl;
		histogram(0,ii) = v;
	}
	/*
	for (int mm = 0; mm < histogram.dim2(); mm++) {
		//cout << "m = " << mm << " and v = ";
		double v = histogram(0,mm);
		//cout << v << endl;
	}
	*/
	return histogram;
}

// Find SIFT Desriptors 
Matrix SIFT::FindDescriptors(Mat src_gray, Matrix keypoints) {
	
	// Simplification of SIFT
	
	// Maybe the image should be smoothed first with a Gaussian Kernel
	int n = keypoints.dim1();
	
	// Initialize matrix containing keypoints descriptors
	Matrix Descriptors(n,128);
	
	// Find Image gradients
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int ksize = 1;
	int scale = 1; 
	int delta = 0; 
	int ddepth = CV_16S;
	
	// Find the gradients in the Sobel operator by using the OPENCV function
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	
	// Converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	
	
	//cout << "Calculating gauss window" << endl;
	int filter_size = 16;
	float sigma = 1.5*16;
	Mat GaussWindow;
	GaussWindow = gaussWindow(filter_size, sigma);
	
	// For each keypoint
	for (int i = 0; i < n; i++) {
		int y = keypoints(i,1);
		int x = keypoints(i,2); 
		
		//waitKey(0);
		
		// Extract a patch of size 16,16 from the image with x-gradients and y-gradients
		Mat Patch_Ix, Patch_Iy;
		Patch_Ix = selectRegionOfInterest(grad_x, y-7, x-7, y+8, x+8);
		Patch_Iy = selectRegionOfInterest(grad_y, y-7, x-7, y+8, x+8);

		// This is the scaled gradients 
		Mat Gradients = Mat::zeros(Size(16,16),CV_64FC1);
		// This is the orientations (angles of the gradients in radians)
		Mat Orientations = Mat::zeros(Size(16,16),CV_64FC1);

		for (int coor_y = 0; coor_y < 16; coor_y++) {
			for (int coor_x = 0; coor_x < 16; coor_x++) {
				float norm = sqrt( pow(Patch_Ix.at<short>(coor_y,coor_x),2) + pow(Patch_Iy.at<short>(coor_y,coor_x),2));
				Gradients.at<double>(coor_y,coor_x) = norm*GaussWindow.at<double>(coor_y,coor_x);
				Orientations.at<double>(coor_y,coor_x) = atan2(Patch_Iy.at<short>(coor_y,coor_x),Patch_Ix.at<short>(coor_y,coor_x));
			}
		}		
		// Maybe you should rotate the patch, so it coincides with the orientation in the strongest direction
		
		// Divde the 16x16 patch into subpatches of 4x4 
		Matrix descrip(1,128);
		Mat subPatchGradients, subPatchOrientations;
		int nindex = 0;
		for (int k1 = 0; k1 <= 12; k1 = k1+4) {
			for (int k2 = 0; k2 <= 12; k2 = k2 + 4) {
				// Extract sub patches
				subPatchGradients = selectRegionOfInterest(Gradients, k1, k2, k1+4, k2+4);
				subPatchOrientations = selectRegionOfInterest(Orientations, k1, k2, k1+4, k2+4);
				//cout << "Orientations extracted " << endl;
				Matrix Histogram(1,8);
				for (int l1 = 0; l1 < 4; l1++) {
					for (int l2 = 0; l2 < 4; l2++) {
						//cout << "Size subPatchOrientations = (" << subPatchOrientations.rows << ", " << subPatchOrientations.cols << ") " << endl;
						double angle = subPatchOrientations.at<double>(l1,l2);
						//cout << "Mistake here" << endl;
						if (angle >= -M_PI && angle < -(3*M_PI)/4) {
							Histogram(0,0) = Histogram(0,0) + subPatchGradients.at<double>(l1,l2);
						}
						else if (angle >= -(3*M_PI)/4 && angle < -M_PI/2) {
							Histogram(0,1) = Histogram(0,1) + subPatchGradients.at<double>(l1,l2);
						}
						else if (angle >= -M_PI/2 && angle < -M_PI/4) {
							Histogram(0,2) = Histogram(0,2) + subPatchGradients.at<double>(l1,l2);
						}
						else if (angle >= -M_PI/4 && angle < 0) {
							Histogram(0,3) = Histogram(0,3) + subPatchGradients.at<double>(l1,l2);
						}
						else if (angle >= 0 && angle < M_PI/4) {
							Histogram(0,4) = Histogram(0,4) + subPatchGradients.at<double>(l1,l2);
						}
						else if (angle >= M_PI/4 && angle < M_PI/2) {
							Histogram(0,5) = Histogram(0,5) + subPatchGradients.at<double>(l1,l2);
						}
						else if (angle >= M_PI/2 && angle < 3*M_PI/4) {
							Histogram(0,6) = Histogram(0,6) + subPatchGradients.at<double>(l1,l2);
						}
						else if (angle >= 3*M_PI/4 && angle < M_PI) {
							Histogram(0,7) = Histogram(0,7) + subPatchGradients.at<double>(l1,l2);
						}
					}
				}
				// Rotate it so it becomes rotation invariant
				//cout << "Print histogram" << endl;
				//cout << "Second dimension: " << Histogram.dim2() << endl;
				/*
				for (int mm = 0; mm < Histogram.dim2(); mm++) {
					//cout << "m = " << mm << " and v = ";
					double v = Histogram(0,mm);
					//cout << v << endl;
				}
				*/
				//waitKey(0);
				Histogram = circularShift(Histogram);
				//cout << "Histogram updated " << endl;
				for (int ii = 0; ii < Histogram.dim2(); ii++) {
					//Descriptors[i].slice(nindex*8,nindex*8+ii) = Histogram(0,ii);
					Descriptors(i,nindex*8+ii) = Histogram(0,ii);
				}
				//cout << "Descriptor Done" << endl;
				nindex++;
			}
		}
		
		// Normalizing the vector 
		double SumOfSquares = 0;
		for (int ii = 0; ii < Descriptors.dim2(); ii++) {
			SumOfSquares = SumOfSquares + Descriptors(i,ii)*Descriptors(i,ii);
		}

	// Scale the norms of the gradients by multiplying a the graidents with a gaussian
	// centered in the keypoint and with Sigma_w = 1.5*16. 

	}
	cout << "SIFT Done " << endl;
	return Descriptors;
}










