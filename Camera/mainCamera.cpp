#include <iostream>
#include "mainCamera.hpp"
#include <limits> 
#include <assert.h> 
//#include <complex.h>
#include <complex>
#include <iomanip>
#include <algorithm>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <cmath> 
#include <math.h>
//#include <complex>

//#include "Matrix.h"

/* ########################
 * Name: mainCamera.cpp
 * Made by: Thor Christiansen - s173949 
 * Date: 18.05.2020
 * Objective: The source file mainCamera.cpp contains the functions used
 * by main.cpp to treat the images - find features in the images using 
 * Harris corner etc.
 * Project: Bachelor project 2020
 * Notes: Maybe you should restrict yourself to only use one type of matrix
 * element - Instead of using both Matrix and Mat, maybe just use one of them
 * When you are at it: Use Pointers instead of indexing. 
 * ########################
*/

using namespace std;
using namespace std::complex_literals;
//using namespace Numeric_lib;

// Match SIFT Descriptors 
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
		//cout << "Keypoint " << i << " : " << matches(i,0)  << ", " << matches(i,1) << ", " << matches(i,2) << ", " << matches(i,3) << " Match = " << matches(i,4) <<  endl;
	}
	
	// Create matrix with the keypoints that are valid and which are returned.
	Matrix valid_matches(count,2);
	cout << "Count of valid matches  = " << count << endl;
	int index = 0;
	for (int i = 0; i < n2; i++) {
		if (matches(i,4) == 1) {
			valid_matches(index,0) = i;
			valid_matches(index,1) = matches(i,0);
			index++;
		}
	}
	
	// Print the keypoints
	/*
	for (int i = 0; i < valid_matches.dim1(); i++) {
		cout << "Keypoint " << valid_matches(i,0) << " match with " << valid_matches(i,1) << endl;
	}
	*/
	
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
		if (y2-y1 > img.rows) {
			y2 = img.rows;
		}
	}
	if (x2 > img.cols) {
		if (x2-x1 > img.cols) {
			x2 = img.cols;
		}
	}
	/* Make a region of the proper size. 
	 * Start point in point (x1,y1) width = y2-y1 and height x2-x1. 
	 */ 
	Rect region(x1,y1,y2-y1,x2-x1);
	
	// Extract the rectangle from the image
	ROI = img(region);
	return ROI;
}


void nonMaximumSuppression(Mat img, int y1, int x1, int y2, int x2) {
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
	img(region) = 0;
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

// ############################# Harris ############################# 
/* Objective: To find keypoints in the image using Harris Corner detector
 * Inputs: 
 * Matrix: Image src 
 * Matrix: Image src_gray
 * int maxinum_keypoint - The number of keypoints that Harris Corner detector should find
 * Matrix keypoints of size 3 x maxinum_keypoint, where the keypoints are organized
 * as [keypoint_value, y, x].
 */
Mat Harris::corner(Mat src, Mat src_gray, int maxinum_keypoint, Mat suppression) {
	
	// Define variables
	const char* corners_window = "Corners detected";
	
	// Maybe use minMaxLoc(img, &minVal, &maxVal); for at finde max og minimum, som gemmes i værdierne minVal og maxVal. 	
	
	// Define variables related to Harris corner
	int blockSize = 9; 
	int apertureSize = 3;
	double k = 0.08;		// Magic parameter 
	//int thres = 200;	
	// Parameters before: blocksize = 2, aperturesize = 3, thres = 200, k = 0.04
	
	// Variables related to Non Maximum suppression 
	int NMSBox = 5;
	int boundaries = 10; // Boundaries in the image 
	
	Mat dst = Mat::zeros( src.size(), CV_32FC1 );
	cornerHarris (src_gray, dst, blockSize, apertureSize, k);

	Mat dst_norm, dst_norm_scaled;
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	// Find corners and return 
	int nr_corners = 0;
		
		
	int keypoints_limit = maxinum_keypoint; // Change to 200
		/*
		Matrix Corners(keypoints_limit,3); // Column1: Corner responses, Column2: Pixel i, Column3: Pixel j
		
		int CornerResponse = 0;
		int maxCornerResponse = 0;
		Corners(0,0) = 0;
		Corners(1,0) = 0;
		*/
	double row_nr, col_nr;
	if (suppression.cols != 0) {
		for (int c = 0; c < suppression.cols; c++) {
			row_nr = suppression.at<double>(0,c);
			col_nr = suppression.at<double>(1,c);
			for (int i = -3; i < 4; i++) {
				for (int j = -3; j < 4; j++) {
					dst_norm.at<float>(row_nr + i, col_nr + j) = 0;
				}
			}
		}
	}
		
		
		//Mat keypoints = Mat::zeros(keypoints_limit, 3, CV_64FC1);
	Mat keypoints = Mat::zeros(3, keypoints_limit, CV_64FC1);
	for (int count = 0; count < keypoints_limit; count++) {
		double max = 0; 
		int x = 0; 
		int y = 0; 
		for (int i = 0+boundaries; i < dst_norm.rows-boundaries; i++) {
			for (int j = 0+boundaries; j < dst_norm.cols-boundaries; j++) {
				if ((double) dst_norm.at<float>(i,j) > max) {
					max = (double) dst_norm.at<float>(i,j) ;
					y = i;
					x = j;
						
				}
					/*
					CornerResponse = (int) dst_norm.at<float>(i,j);
					if ( CornerResponse > thres && nr_corners < keypoints_limit-1) {
						//cout << "intensity for corner " << nr_corners << " at (" << i << "," << j << "): " << (int) dst_norm.at<float>(i,j) << endl;
						// Insert node in heap
						Corners = insertNodeHeap(Corners, CornerResponse,i,j,nr_corners+1);
						//cout << "Heap Corners intensity " << Corners(nr_corners+1,0)  << " at (" << Corners(nr_corners+1,1) << "," << Corners(nr_corners+1,2) << ")" << endl;
						nr_corners++;
					}
					*/
				}
			}
			//cout << "Keypoints extracted" << endl;
			//keypoints.at<double>(count, 0) = max;
			//keypoints.at<double>(count, 1) = y;
			//keypoints.at<double>(count, 2) = x;
		keypoints.at<double>(0, count) = max;
		keypoints.at<double>(1, count) = y;
		keypoints.at<double>(2, count) = x;
			/* 
			cout << "Keypoint at (y,x) = (" << y << "," << x << ") with intensity = " << max << endl;
			waitKey(0);
			for (int nn = y-NMSBox; nn <= y + NMSBox; nn++) {
				for (int mm = x - NMSBox; mm <= x + NMSBox; mm++) {
					cout << (double) dst_norm.at<float>(nn,mm) << ", ";
				}
				cout << "" << endl;
			}
			waitKey(0);
			*/
		nonMaximumSuppression(dst_norm, y-NMSBox, x-NMSBox, y+NMSBox+1, x+NMSBox+1);
			/*
			for (int nn = y-NMSBox; nn <= y + NMSBox; nn++) {
				for (int mm = x - NMSBox; mm <= x + NMSBox; mm++) {
					cout << (double) dst_norm.at<float>(nn,mm) << ", ";
				}
				cout << "" << endl;
			}
			waitKey(0);
			*/
	}
		
		/*
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
		*/
	return keypoints;
	
	//cout << "End of Harris " << endl;	
	//Mat emptyArray;	
	//return emptyArray;
}

// ############################# SIFT ############################# 
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
Matrix SIFT::FindDescriptors(Mat src_gray, Mat keypoints) {
	
	// Simplification of SIFT
	//cout << "Error here" << endl;
	// Maybe the image should be smoothed first with a Gaussian Kernel
	int n = keypoints.cols;
	
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
	
	//cout << "Size grad_x = (" << grad_x.rows << "," << grad_x.cols << ")" << endl;
	//cout << "Size grad_y = (" << grad_y.rows << "," << grad_y.cols << ")" << endl;
	
	// Converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	
	//cout << "Error 2 here" << endl;
	//cout << "Calculating gauss window" << endl;
	int filter_size = 16;
	float sigma = 1.5*16;
	Mat GaussWindow;
	GaussWindow = gaussWindow(filter_size, sigma);
	
	// For each keypoint
	for (int i = 0; i < n; i++) {
		//int y = keypoints.at<double>(i, 1);
		//int x = keypoints.at<double>(i, 2); 
		int y = keypoints.at<double>(1, i);
		int x = keypoints.at<double>(2, i); 
		//cout << "(y,x) = (" << y << "," << x << ")" << endl;
		
		//waitKey(0);
		
		// Extract a patch of size 16,16 from the image with x-gradients and y-gradients
		Mat Patch_Ix, Patch_Iy;
		//cout << "Error to here" << endl;
		Patch_Ix = selectRegionOfInterest(grad_x, y-7, x-7, y+8, x+8);
		Patch_Iy = selectRegionOfInterest(grad_y, y-7, x-7, y+8, x+8);
		//cout << "Error 2 here" << endl;
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

 




// Estimate pose of camera
Mat cross2Matrix(Mat x) {
	Mat M = Mat::zeros(3, 3, CV_64FC1);
	
	M.at<double>(0,1) = -x.at<double>(0,2);
	M.at<double>(0,2) = x.at<double>(0,1);
	M.at<double>(1,0) = x.at<double>(0,2);
	M.at<double>(1,2) = -x.at<double>(0,0);
	M.at<double>(2,0) = -x.at<double>(0,1); 
	M.at<double>(2,1) = x.at<double>(0,0);
	
	return M;
}

// Multiplicaiton of a matrix with a vector 
Vector MatVecMul(const Matrix& m, const Vector& u) {
	int n = m.dim1();
	Vector v(n);
	for (int i = 0; i <n; ++i) {
		v(i) = dot_product(m[i],u);
	}
	return v;
}

Matrix MatMatMul(const Matrix& m1, const Matrix& m2) {
	//cout << "MatMatMul" << endl;
	// Checks for proper dimensions
	assert (m1.dim2() == m2.dim1());
	Matrix m(m1.dim1(),m2.dim2());
	for (int i = 0; i < m1.dim2(); i++) {
		for (int j = 0; j < m2.dim2(); j++) {
			Vector u(m2.dim1());
			for (int k = 0; k < m2.dim1(); k++) {
				u(k) = m2(k,j);
			}
			m(i,j) = dot_product(m1[i],u);
			//cout << m(i,j) << ", ";
		}
		//cout << "" << endl;
	}
	return m;
}

// Should be changed to work with Mat instead 
Mat linearTriangulation(Mat p1, Mat p2, Mat M1, Mat M2) {
	
	
	assert(p1.rows == p2.rows);
	assert(p1.cols == p2.cols);
	int NumPoints = p1.cols;
	
	Mat P = Mat::zeros(4, NumPoints, CV_64FC1);
	
	for (int j = 0; j < NumPoints; j++) {
		Mat temp_point = Mat::zeros(3, 1, CV_64FC1);
		temp_point.at<double>(0,0) = p1.at<double>(0,j);
		temp_point.at<double>(1,0) = p1.at<double>(1,j);
		if (p1.rows == 2) {
			temp_point.at<double>(2,0) = 1;
		}
		else {
			temp_point.at<double>(2,0) = p1.at<double>(2,j);
		}
		
		Mat A1 = cross2Matrix(temp_point) * M1;
	
	
		temp_point.at<double>(0,0) = p2.at<double>(0,j);
		temp_point.at<double>(1,0) = p2.at<double>(1,j);
		if (p2.rows == 2) {
			temp_point.at<double>(2,0) = 1;
		}
		else {
			temp_point.at<double>(2,0) = p2.at<double>(2,j);
		}
		
		Mat A2 = cross2Matrix(temp_point) * M2;
		
		Mat A = Mat::zeros((A1.rows+A2.rows), A1.cols, CV_64FC1);
		
		for (int h1 = 0; h1 < A.rows; h1++) {
			for (int h2 = 0; h2 < A.cols; h2++) {
				if (h1 < A.rows/2) {
					A.at<double>(h1,h2) = A1.at<double>(h1,h2);
				}
				else {
					A.at<double>(h1,h2) = A2.at<double>(h1-A2.rows,h2);
				} 
			}
		}
		
		Mat S, U, VT;
		SVDecomp(A, S, U, VT, SVD::FULL_UV);
		
		VT = VT.t();
		for (int i = 0; i < P.rows; i++) {
			if (VT.at<double>(2,3)/VT.at<double>(3,3) < 0) {
				P.at<double>(i,j) = (-1)*VT.at<double>(i,4)/(VT.at<double>(3,3));
			}
			else {
				P.at<double>(i,j) = VT.at<double>(i,3)/(VT.at<double>(3,3));
			}
		}
		//cout << "Point P" << endl;
		for (int nn = 0; nn < 4; nn++) {
			//cout << P.at<double>(nn,j) << endl;
		}
	}
	return P;
}


// ############################# Eight Point Algorithm ############################# 
Mat estimateEssentialMatrix(Mat fundamental_matrix, Mat K) {
		
	Mat M = K.t();
	
	Mat essential_matrix = M * fundamental_matrix * K; // Check if the transpose is the right one?
	
	return essential_matrix;
}


Mat findRotationAndTranslation(Mat essential_matrix, Mat K, Mat points1Mat, Mat points2Mat) {
	Mat transformation_matrix;
	
	Mat S, U, VT;  // VT = V transposed
	// SV Decomposition of essential matrix
	SVDecomp(essential_matrix, S, U, VT, SVD::FULL_UV);
	
	// Translation vector t 
	

	
	
	
	// Find the two possible rotations 
	Vector t(3);
	for (int i = 0; i < 3; i++) {
		t(i) = U.at<double>(i,2);
	}
	
	Mat W = (Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	
	/*
	cout << "U" << endl;
	for (int i = 0; i < U.rows; i++) {
		for (int j = 0; j < U.cols; j++) {
			cout << U.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	cout << "S" << endl;
	for (int i = 0; i < S.rows; i++) {
		for (int j = 0; j < S.cols; j++) {
			cout << S.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	VT = VT.t();
	cout << "VT" << endl;
	for (int i = 0; i < VT.rows; i++) {
		for (int j = 0; j < VT.cols; j++) {
			cout << VT.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	Mat R1 = U * W * VT;
	
	
	Mat R2 = U * W.t() * VT;
	
	
	
	if (determinant(R1) < 0) {
		R1 = -1*R1;
	}
	
	if (determinant(R2) < 0) {
		R2 = -1*R2;
	}
	
	double length = sqrt(t(0)*t(0) + t(1)*t(1) + t(2)*t(2));
	if (length != 0) {
		for (int i = 0; i < 3; i++) {
			t(i) = t(i)/length;
		}
	}
	
	/*
	cout << "u3" << endl;
	for (int i = 0; i<3; i++) {
		cout << t(i) << endl;
	}
	
	cout << "R1" << endl;
	for (int i = 0; i < R1.rows; i++) {
		for (int j = 0; j < R1.cols; j++) {
			cout << R1.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	cout << "R2" << endl;
	for (int i = 0; i < R2.rows; i++) {
		for (int j = 0; j < R2.cols; j++) {
			cout << R2.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	*/
	// Disambiguate pose 
	Mat Trans_I0 = (Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
	Mat M0 = K * Trans_I0;
	
	Mat R;
	int total_points_in_front_best = 0;
	for(int iRot = 0; iRot <= 1; iRot++) {
		if (iRot == 0) {
			R = R1;
		}
		else {
			R = R2;
		}
		for (int iSignT = 1; iSignT <= 2; iSignT++) {
			Vector T = t * pow((-1),iSignT);
			
			Mat Trans_I1 = (Mat_<double>(3,4) << R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),T(0), R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),T(1), R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),T(2));
			Mat M1 = K * Trans_I1;
			Mat P_C0 = linearTriangulation(points1Mat, points2Mat, M0, M1);
			
			// Project in both cameras
			// Only the 3rd dimension is needed, so the matrix is reduced
			Mat P_C1 = Mat::zeros(1, P_C0.cols, CV_64FC1);
			int num_points_in_front1 = 0;
			for (int i = 0; i < P_C1.cols; i++) {
				for (int j = 0; j < P_C0.rows; j++) {
					P_C1.at<double>(0,i) = P_C1.at<double>(0,i) + Trans_I1.at<double>(2,j)*P_C0.at<double>(j,i);
				}
				if (P_C1.at<double>(0,i) > 0) {
					num_points_in_front1++;
				}
			}
			int num_points_in_front0 = 0;
			for (int i = 0; i < P_C0.cols; i++) {
				if (P_C0.at<double>(3,i) > 0) {
					num_points_in_front0++;
				}
			}
			int total_points_in_front = num_points_in_front1 + num_points_in_front0;
			if (total_points_in_front > total_points_in_front_best) {
				
				
				transformation_matrix = Trans_I1;
				
				/*
				transformation_matrix.at<double>(0,0) = R.at<double>(0,0);
				transformation_matrix.at<double>(0,1) = R.at<double>(0,1);
				transformation_matrix.at<double>(0,2) = R.at<double>(0,2);
				transformation_matrix.at<double>(1,0) = R.at<double>(1,0);
				transformation_matrix.at<double>(1,1) = R.at<double>(1,1);
				transformation_matrix.at<double>(1,2) = R.at<double>(1,2);
				transformation_matrix.at<double>(2,0) = R.at<double>(2,0);
				transformation_matrix.at<double>(2,0) = R.at<double>(2,1);
				transformation_matrix.at<double>(2,0) = R.at<double>(2,2);
				
				transformation_matrix.at<double>(0,3) = T(0);
				transformation_matrix.at<double>(1,3) = T(1);
				transformation_matrix.at<double>(2,3) = T(2);
				*/
				
				
				total_points_in_front_best = total_points_in_front;
			}
		}
	}
	return transformation_matrix;
}



// ############################# KLT ############################# 
Mat warpImage(Mat I_R, Mat W) {
	Mat I_warped = Mat::zeros(I_R.rows, I_R.cols, CV_64FC1);
	
	for (int x = 0; x < I_R.cols; x++) {
		for (int y = 0; y < I_R.rows; y++) {
			Mat vector = Mat::ones(3, 1, CV_64FC1);
			vector.at<double>(0,0) = x;
			vector.at<double>(1,0) = y; 
			Mat warped = W * vector;
			warped = warped.t();

			if (warped.at<double>(0,0) < I_R.cols && warped.at<double>(0,1) < I_R.rows) {
				if (warped.at<double>(0,0) > 1 && warped.at<double>(0,1) > 1) {
					
					uchar m = I_R.at<double>(floor(warped.at<double>(0,1)),floor(warped.at<double>(0,0)));
					
					I_warped.at<double>(y,x) = I_R.at<double>(floor(warped.at<double>(0,1)),floor(warped.at<double>(0,0)));
				}
			}
		}
	} 
	return I_warped;
}



// Get the warping matrix
Mat getSimWarp(double dx, double dy, double alpha_deg, double lambda) {
	
	Mat W = Mat::zeros(2, 3, CV_64FC1);
	double alpha_rad = (alpha_deg * M_PI) / 180;
	W.at<double>(0,0) = lambda * cos( alpha_rad );
	W.at<double>(1,0) = lambda * sin( alpha_rad );
	W.at<double>(0,1) = lambda * -sin( alpha_rad );
	W.at<double>(1,1) = lambda * cos( alpha_rad );
	W.at<double>(0,2 ) = lambda * dx;
	W.at<double>(1,2) = lambda * dy;
	
	return W;
}

// Get the patch
Mat getWarpedPatch(Mat I_new, Mat W, Mat x_T, int r_T) {
	
	//imshow("I_new", I_new);
	//waitKey(0);
	
	//cout << "Type of image" << endl;
	//MatType(I_new);
	
	// Initialize patch
	Mat patch = Mat::zeros(2*r_T + 1, 2*r_T + 1, CV_64FC1);
	
	// Get dimensions of image 
	int max_coords_rows = I_new.rows;
	int max_coords_cols = I_new.cols;
	
	//cout << "Dimensions of image = (" << max_coords_rows << "," << max_coords_cols << ")" << endl;
	
	// Find the transpose
	Mat WT = W.t();
	
	/*
	int hej = 0;
	if (WT.at<double>(0,0) > 1) {
		hej = 1;
	}
	if (hej == 1) {
		cout << "WT " << endl;
		for (int r = 0; r < WT.rows; r++) {
				for (int c = 0; c < WT.cols; c++) {
					cout << WT.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
			}
			waitKey(0);
	}
	//cout << "Inside getwarpedPatch" << endl;
	//waitKey(10000);
	*/
	
	Mat pre_warp = Mat::zeros(1, 3, CV_64FC1);
	for (int x = -r_T; x <= r_T; x++) {
		for (int y = -r_T; y <= r_T; y++) {
			pre_warp.at<double>(0,0) = x;
			pre_warp.at<double>(0,1) = y;
			pre_warp.at<double>(0,2) = 1;

			/*
			if (hej == 1) {
				cout << "pre_warp" << endl;
				for (int r = 0; r < pre_warp.rows; r++) {
					for (int c = 0; c < pre_warp.cols; c++) {
						cout << pre_warp.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
				}
				waitKey(0);
			}
			*/
			
			Mat warped = x_T + pre_warp * WT;
			
			/*
			if (hej == 1) {
				cout << "warped" << endl;
				for (int r = 0; r < warped.rows; r++) {
					for (int c = 0; c < warped.cols; c++) {
						cout << warped.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
				}
				waitKey(0);
			}
			*/
			
			if (warped.at<double>(0,0) < max_coords_cols && warped.at<double>(0,1) < max_coords_rows) {
				if (warped.at<double>(0,0) > 0 && warped.at<double>(0,1) > 0) { // It should be greater than 0 (C++ 0-indexing)
					
					//cout << "Inside if-statements" << endl;

					Mat floors = Mat::zeros(warped.rows, warped.cols, CV_64FC1);
					
					
					for (int r = 0; r < floors.rows; r++) {
						for (int c = 0; c < floors.cols; c++) {
							floors.at<double>(r, c) = floor(warped.at<double>(r, c));
						}
					}
					
					/*
					if (hej == 1) {
						cout << "floors " << endl;
						for (int r = 0; r < floors.rows; r++) {
							for (int c = 0; c < floors.cols; c++) {
								cout << floors.at<double>(r, c) << ", ";
							}
						}
					}
					*/
					
					/*
					cout << "floors " << endl;
					for (int r = 0; r < floors.rows; r++) {
						for (int c = 0; c < floors.cols; c++) {
							cout << floors.at<double>(r,c) << ", ";
						}
						cout << "" << endl;
					}
					//waitKey(2000);
					*/
					
					Mat weights = warped - floors;
					
					/*
					if (hej == 1) {
						cout << "weights " << endl;
						for (int r = 0; r < weights.rows; r++) {
							for (int c = 0; c < weights.cols; c++) {
								cout << weights.at<double>(r, c) << ", ";
							}
						}
					}
					*/
					
					/*
					cout << "weights " << endl;
					for (int r = 0; r < weights.rows; r++) {
						for (int c = 0; c < weights.cols; c++) {
							cout << weights.at<double>(r,c) << ", ";
						}
						cout << "" << endl;
					}
					//waitKey(2000);
					*/
					
					double a = weights.at<double>(0,0);
					double b = weights.at<double>(0,1);
					
					/*
					if (hej == 1) {
						cout << "a and b " << endl;
						cout << "a = " << a;
						cout << "b = " << b;
						cout << "" << endl;
					}
					*/
					
					//cout << "a = " << a << " and b = " << b << endl;
					
					//cout << "floors.at<double>(0,1)-1 = " << floors.at<double>(0,1)-1 << endl;
					
					//cout << "floors.at<double>(0,0)-1) = " << floors.at<double>(0,0)-1 << endl;
					
					//cout << "Image intensity 1 = " <<  I_new.at<uchar>((int) floors.at<double>(0,1)-1,(int) floors.at<double>(0,0)-1) << endl;
					
					//cout << "With switched coordinates = " << I_new.at<double>(floors.at<double>(0,0)-1,floors.at<double>(0,1)-1) << endl;
					
					//cout << "Image intensity 1 = " << I_new.at<double>(floors.at<double>(0,1)-1,floors.at<double>(0,0)) << endl;
					
					/*
					if (hej == 1) {
						cout << "First image index = " << I_new.at<uchar>(floors.at<double>(0,1)-1,floors.at<double>(0,0)-1) << endl;
						cout << "Second image index = " << I_new.at<uchar>(floors.at<double>(0,1)-1,floors.at<double>(0,0)) << endl;
						cout << "index 1 = " << floors.at<double>(0,1)-1 << endl; // Should be one less
						cout << "index 2 = " << floors.at<double>(0,0)-1 << endl;
						cout << "index 3 = " << floors.at<double>(0,1)-1 << endl;
						cout << "index 4 = " << floors.at<double>(0,0) << endl;
					}
					*/
					
					double intensity = (1-b) * ((1-a) * I_new.at<uchar>(floors.at<double>(0,1)-1,floors.at<double>(0,0)-1) + a * I_new.at<uchar>(floors.at<double>(0,1)-1,floors.at<double>(0,0)));
					
					/*
					if (hej == 1) {
						cout << "temp_intensity = " << intensity << endl;
					}
					*/
					
					//cout << "temp-intensity = " << intensity << endl;
					
					intensity = intensity + b * ((1-a) * I_new.at<uchar>(floors.at<double>(0,1),floors.at<double>(0,0)-1) + a * I_new.at<uchar>(floors.at<double>(0,1),floors.at<double>(0,0)));
					
					/*
					if (intensity == 0) {
						waitKey(0);
					}
					*/
					/*
					if (hej == 1) {
						cout << "intensity = " << intensity << endl; 
					}
					*/
					
					
					
					//cout << "Intensity = " << intensity << endl;;
					
					patch.at<double>(y + r_T, x + r_T) = intensity;
					
					/*
					if (hej == 1) {
						cout << "patch value = " << patch.at<double>(y + r_T, x + r_T) << endl;
					}
					*/
					
					//cout << "y + r_T, x+r_T = (" << y + r_T << "," << x + r_T << ")" << endl;
					
					//cout << "patch = " << patch.at<double>(y + r_T, x + r_T) << endl;
				}	
			}
		}
	}
	return patch;
}

Mat Kroneckerproduct(Mat A, Mat B) {
	
	int rowa = A.rows;
	int cola = A.cols;
	int rowb = B.rows;
	int colb = B.cols;
	
	Mat C = Mat::zeros(rowa * rowb, cola * colb, CV_64FC1);
	for (int i = 0; i < rowa; i++) {
		
		for (int k = 0; k < rowb; k++) {
			
			for (int j = 0; j < cola; j++) {
				
				for (int l = 0; l < colb; l++) {
					
					C.at<double>(i*rowb + k, j*colb + l) = A.at<double>(i, j) * B.at<double>(k, l);
				}
			}
		}
	}
	return C;
}

// Track KLT
Mat trackKLT(Mat I_R, Mat I_new, Mat x_T, int r_T, int num_iters) {
	Mat p_hist = Mat::zeros(6, num_iters+1, CV_64FC1);
	Mat W = getSimWarp(0, 0, 0, 1);
	
	int temp_index = 0;
	for (int c = 0; c < W.cols; c++) {
		for (int r = 0; r < W.rows; r++) {
			p_hist.at<double>(temp_index, 0) = W.at<double>(r,c);
			temp_index++;
		}
	}
	
	// Get the warped patch
	Mat I_RT = getWarpedPatch(I_R, W, x_T, r_T);
	
	/*
	// Output for debug 
	cout << "r_T = " << r_T << endl;
	cout << "W" << endl;
	for (int r = 0; r < W.rows; r++) {
		for (int c = 0; c < W.cols; c++) {
			cout << W.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "x_T" << endl;
	for (int r = 0; r < x_T.rows; r++) {
		for (int c = 0; c < x_T.cols; c++) {
			cout << x_T.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	MatType(I_RT);
	cout << "I_RT" << endl;
	for (int r = 0; r < I_RT.rows; r++) {
		for (int c = 0; c < I_RT.cols; c++) {
			cout << I_RT.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	waitKey(0);
	*/
	
	
	I_RT = I_RT.t();
	Mat i_R = I_RT.reshape(0,I_RT.rows * I_RT.cols);
	
	int n = 2*r_T + 1;
	Mat xy1 = Mat::zeros(n * n, 3, CV_64FC1);
	temp_index = 0; 
	for (int i = -r_T; i <= r_T; i++) {
		for (int j = -r_T; j <= r_T; j++) {
			xy1.at<double>(temp_index,0) = i;
			xy1.at<double>(temp_index,1) = j;
			xy1.at<double>(temp_index,2) = 1;
			temp_index++;
		} 
	}
	
	// Find the Kroeneckerproduct 
	Mat dwdx = Kroneckerproduct(xy1, Mat::eye(2, 2, CV_64FC1));
	
	// 2D filters for convolution
	Mat kernelx, kernely; 
	Point anchorx, anchory;
	double delta;
	int ddepth; 
	int kernel_size;
	
	anchorx = Point(-1,0);
	anchory = Point(0,-1);
	delta = 0; 
	ddepth = -1;
	
	kernelx = Mat::zeros(1, 3, CV_64FC1);
	kernelx.at<double>(0,0) = -1;
	kernelx.at<double>(0,2) = 1;
	kernely = Mat::zeros(3, 1, CV_64FC1);
	kernely.at<double>(0,0) = -1;
	kernely.at<double>(2,0) = 1;
	
	// About to begin iteration 
	for (int iter = 0; iter < num_iters; iter++) {
		Mat big_IWT = getWarpedPatch(I_new, W, x_T, r_T + 1); // We are here 
		
		/*
		cout << "big_IWT" << endl;
		for (int r = 0; r < big_IWT.rows; r++) {
			for (int c = 0; c < big_IWT.cols; c++) {
				cout << big_IWT.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		//waitKey(0);
		*/
		
		
		
		Mat IWT_temp, IWT;
		IWT_temp = selectRegionOfInterest(big_IWT, 1, 1, big_IWT.rows-1, big_IWT.cols-1);
		IWT_temp.copyTo(IWT);
	
		IWT = IWT.t();
		Mat i = IWT.reshape(0, IWT.rows * IWT.cols);
		
		// Getting di/dp 
		//cout << "Getting di/dp" << endl;
		Mat IWTx, IWTy, temp_IWTx, temp_IWTy, temp_IWTx2, temp_IWTy2;
		temp_IWTx2 = selectRegionOfInterest(big_IWT, 1, 0, big_IWT.cols+1, big_IWT.rows-2); // Maybe check for values
		temp_IWTy2 = selectRegionOfInterest(big_IWT, 0, 1, big_IWT.cols-2, big_IWT.rows+1);
		
		//cout << "Not here yet " << endl;
		temp_IWTx2.copyTo(temp_IWTx);
		temp_IWTy2.copyTo(temp_IWTy);
		
		// Convolve x
		filter2D(temp_IWTx, IWTx, ddepth, kernelx, anchorx, delta, BORDER_DEFAULT);
		IWTx = selectRegionOfInterest(IWTx, 0, 1, IWTx.cols-1, IWTx.rows+1);
		
		// Convolve y 
		filter2D(temp_IWTy, IWTy, ddepth, kernely, anchory, delta, BORDER_DEFAULT);
		IWTy = selectRegionOfInterest(IWTy, 1, 0, IWTy.cols+1, IWTy.rows-1);
		
		// Concatenate vectors 
		Mat IWTx_new, IWTy_new;
		IWTx.copyTo(IWTx_new);
		IWTy.copyTo(IWTy_new);
		IWTx_new = IWTx_new.t();
		IWTy_new = IWTy_new.t();
		temp_IWTx = IWTx_new.reshape(0, IWTx.rows * IWTx.cols);
		temp_IWTy = IWTy_new.reshape(0, IWTy.rows * IWTy.cols);
		Mat didw;
		hconcat(temp_IWTx, temp_IWTy, didw);
				
		Mat didp = Mat::zeros(n * n, 6, CV_64FC1);
		double vdidw1, vdidw2, vdwdx1, vdwdx2;
		for (int pixel_i = 0; pixel_i < didp.rows; pixel_i++) {
			vdidw1 = didw.at<double>(pixel_i,0);
			vdidw2 = didw.at<double>(pixel_i,1);
			for (int j = 0; j < 6; j++) {
				vdwdx1 = dwdx.at<double>(pixel_i * 2, j);
				vdwdx2 = dwdx.at<double>(pixel_i * 2 + 1, j);
				didp.at<double>(pixel_i, j) = vdidw1 * vdwdx1 + vdidw2 * vdwdx2;
			}
		} 
		
		// Hessian matrix 
		Mat H = didp.t() * didp;
		
		// Hessian matrix check
		
		Mat temp_delta_p = didp.t() * (i_R - i);
		
		// Calculate delta_p 
		Mat delta_p = H.inv() * (didp.t() * (i_R - i)); // Maybe problem with 
		
		// Reshape delta_p 
		Mat delta_p_temp = delta_p.reshape(0, 3);
		
		delta_p_temp = delta_p_temp.t();
		
		W = W + delta_p_temp; // 2 = W.rows
		 
		W = W.t();
		Mat temp_W = W.reshape(0, W.rows * W.cols);
		
		// Create Matrix of p_hist
		for (int hh = 0; hh < 6; hh++) {
			p_hist.at<double>(hh, iter+1) =  temp_W.at<double>(hh,0);
		}
		
		// Transpose W to get the right shape for the next iteration - C++ is different from Matlab
		W = W.t();
		
		/*
		cout << "W" << endl;
		for (int r = 0; r < W.rows; r++) {
			for (int c = 0; c < W.cols; c++) {
				cout << W.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		//waitKey(0);
		*/
		
	}
	return W;
}

Mat KLT::trackKLTrobustly(Mat I_R, Mat I_new, Mat keypoint, int r_T, int num_iters, double lambda) {
	
	Mat W = trackKLT(I_R, I_new, keypoint, r_T, num_iters);
	
	// delta_keypoint contains the y- and x-coordinate of the keypoint as the first and second coordiate
	// and the third coordiate is a boolean-value, which is either 1 or 0 depending on whether the value is smaller 
	// than lambda  
	Mat delta_keypoint = Mat::zeros(3, 1, CV_64FC1);
	delta_keypoint.at<double>(0,0) = W.at<double>(0,2);
	delta_keypoint.at<double>(1,0) = W.at<double>(1,2);
	
	// The reverse keypoint that is used to find the backwards warp
	Mat reverse_keypoint = Mat::zeros(1, 2, CV_64FC1);
	reverse_keypoint.at<double>(0,0) = keypoint.at<double>(0,0) + delta_keypoint.at<double>(0,0);
	reverse_keypoint.at<double>(0,1) = keypoint.at<double>(0,1) + delta_keypoint.at<double>(1,0);
	
	Mat Winv = trackKLT(I_new, I_R, reverse_keypoint, r_T, num_iters);
	
	Mat dkpinv = Mat::zeros(2, 1, CV_64FC1);
	dkpinv.at<double>(0,0) = Winv.at<double>(0,2);
	dkpinv.at<double>(1,0) = Winv.at<double>(1,2);
	
	if (sqrt(pow(delta_keypoint.at<double>(0,0) + dkpinv.at<double>(0,0),2.0) + pow(delta_keypoint.at<double>(1,0) + dkpinv.at<double>(1,0),2.0)) < lambda) {
		delta_keypoint.at<double>(2,0) = 1;
	}
	
	return delta_keypoint;
}

// ############################# ransacLocalizaiton #############################
Mat crossProduct(Mat vector1, Mat vector2) {
	Mat product = Mat::zeros(3, 1, CV_64FC1);
	double a1, a2, a3, b1, b2, b3;
	a1 = vector1.at<double>(0,0);
	a2 = vector1.at<double>(1,0);
	a3 = vector1.at<double>(2,0);
	b1 = vector2.at<double>(0,0);
	b2 = vector2.at<double>(1,0);
	b3 = vector2.at<double>(2,0);
	
	product.at<double>(0,0) = a2 * b3 - a3 * b2;
	product.at<double>(1,0) = a3 * b1 - a1 * b3;
	product.at<double>(2,0) = a1 * b2 - a2 * b1;
	
	return product;
}

Mat projectPoints(Mat points_3d, Mat K) {
	Mat projected_points;
	
	Mat D = Mat::zeros(4, 1, CV_64FC1);
	
	int num_points = points_3d.cols;
	
	Mat x_y_points = Mat::ones(3, num_points, CV_64FC1);
	for (int i = 0; i < num_points; i++) {
		x_y_points.at<double>(0,i) = points_3d.at<double>(0,i) / points_3d.at<double>(2,i); 
		x_y_points.at<double>(1,i) = points_3d.at<double>(1,i) / points_3d.at<double>(2,i); 
	}
	
	// Not necessary to apply distortion 
	
	// Convert to pixel coordinates 
	projected_points = K * x_y_points;
	vconcat(projected_points.row(0), projected_points.row(1), projected_points);
	
	return projected_points;
}


Mat solveQuartic(Mat factors) {
	cout << "Inside solveQuartic " << endl;
	Mat roots = Mat::zeros(1, 4, CV_64FC1);
	
	double A = factors.at<double>(0,0);
	double B = factors.at<double>(0,1);
	double C = factors.at<double>(0,2);
	double D = factors.at<double>(0,3);
	double E = factors.at<double>(0,4);
	
	cout << "Factors = " << A << "," << B << "," << C << "," << D << "," << E << endl;
	
	double A_pw2 = A*A;
	double B_pw2 = B*B; 
	double A_pw3 = A_pw2*A;
	double B_pw3 = B_pw2*B;
	double A_pw4 = A_pw3*A;
	double B_pw4 = B_pw3*B;
	
	cout << "Values = " << A_pw2 << "," << B_pw2 << "," << A_pw3 << "," << B_pw3 << "," << A_pw4 << "," << B_pw4 << endl;
	
	double alpha = -3*B_pw2/(8*A_pw2) + C/A;
	double beta = B_pw3/(8*A_pw3) - B*C/(2*A_pw2) + D/A;
	double gamma = -3*B_pw4/(256*A_pw4) + B_pw2*C/(16*A_pw3) - B*D/(4*A_pw2) + E/A;
	
	cout << "Values 2 = " << alpha << "," << beta << "," << gamma << endl;
	
	double alpha_pw2 = alpha * alpha; 
	double alpha_pw3 = alpha_pw2 * alpha;
	
	cout << "Values 3 = " << alpha_pw2 << "," << alpha_pw3 << endl;
	
	double P = -alpha_pw2/12 - gamma;
	double Q = -alpha_pw3/108 + alpha*gamma/3 - pow(beta,2.0)/8;
	
	
	std::complex<double> i_value;
	i_value = 1i;
	/*
	std::complex<double> H, HH, i_value;
	double v = 0.7;
	HH = 1i;
	i_value = 1i;
	H = 10/5. + v*HH;
	
	cout << "H = " << H << endl;
	cout << "H*H = " << pow(H,2.0) << endl;
	cout << "H*1/3 = " << pow(H,1.0/3.0) << endl;
	double qq = 10.0/5.0 + 11.0;
	H = qq + 1i;
	cout << "H = " << H + qq << endl;
	
	H = 2. + 3i;
	//H = 4./(3.*H);
	double qqq = 17;
	H = qqq/(H);
	//H = H*3.;
	cout << "New Value = " << H << endl;
	
	H = 2. + 3i;
	H = 0.5 * H;
	cout << " New H = " << H << endl;
	*/
	
	std::complex<double> R, U, y, w, null_v;
	double real_value, imaginary_value;
	if (pow(Q,2.0)/4 + pow(P,3.0)/27 < 0) {
		imaginary_value = sqrt(-(pow(Q,2.0)/4 + pow(P,3.0)/27));
		real_value = (-Q/2.0);
		R = real_value + imaginary_value*i_value;
	}
	else {
		real_value = -Q/2.0 + sqrt(pow(Q,2.0)/4 + pow(P,3.0)/27);
		R = real_value + 0i; 
	}
	U = pow(R, 1.0/3.0);
	
	//cout << "P, Q, R, U  = " << P << ", " << Q << ", " << R  << ", " << U << endl;
	
	null_v = 0. + 0i;
	real_value = -5.0*alpha/6.0;
	if (U == null_v) {
		y = real_value - pow(Q, 1.0/3.0);
	}
	else {
		y = real_value - P/(3.*U) + U;
	}
	
	//cout << "y  = " << y << endl;
	
	w = pow(alpha+2.*y, 1.0/2.0);
	
	//cout << "w = " << w << endl;
	
	std::complex<double> temp0, temp1, temp2, temp3;
	real_value = -B/(4*A);
	
	//cout << "Complex value = " << pow( -(3*alpha+2.*y+2.*beta/w), 1.0/2.0) << endl;
	
	temp0 = real_value + 0.5*(w + pow( -(3*alpha+2.*y+2.*beta/w), 1.0/2.0));
	temp1 = real_value + 0.5*(w - pow( -(3*alpha+2.*y+2.*beta/w), 1.0/2.0));
	temp2 = real_value + 0.5*(-w + pow( -(3*alpha+2.*y-2.*beta/w), 1.0/2.0));
	temp3 = real_value + 0.5*(-w - pow( -(3*alpha+2.*y-2.*beta/w), 1.0/2.0));
	
	//cout << "temp0, temp1, temp2, temp3  = " << temp0 << ", " << temp1 << ", " << temp2  << ", " << temp3 << endl;
	
	//roots.at<double>(0,0) = real(temp0);
	//roots.at<double>(0,1) = real(temp1);
	//roots.at<double>(0,0) = real(temp2);
	//roots.at<double>(0,1) = real(temp3);
	
	
	//cout << "Real values " << endl;
	//cout << "real temp values = " << real(temp0) << ", " << real(temp1) << ", " << real(temp2) << ", " << real(temp3) << endl;
	
	temp0 = real(temp0);
	temp1 = real(temp1);
	temp2 = real(temp2);
	temp3 = real(temp3);
	
	roots.at<double>(0,0) = real(temp0);
	roots.at<double>(0,1) = real(temp1);
	roots.at<double>(0,2) = real(temp2);
	roots.at<double>(0,3) = real(temp3);
	
	
	/*
	std::complex<double> U, R;
	if (pow(Q,2.0)/4 + pow(P,3.0)/27 < 0) {
		R = (-Q/2, sqrt(pow(Q,2.0)/4 + pow(P,3.0)/27)); 
		U = pow(R, 1.0/3.0);
	}
	else {
		R = (-Q/2 + sqrt(pow(Q,2.0)/4 + pow(P,3.0)/27), 0); 
		U = pow(R, 1.0/3.0);
	}
	
	//double R = -Q/2 + sqrt(pow(Q,2.0)/4 + pow(P,3.0)/27);
	//double U = pow(R,(1.0/3.0));
	
	//cout << "Values 4 = " << P << "," << Q <<  "," << R << "," << U << endl;

	
	//double y;
	std::complex<double> y;
	if (U == 0) {
		if (Q < 0) {
			y (-5*alpha/6, - pow(Q,(1.0/3.0)));
		}
		else {
			y (-5*alpha/6 - pow(Q,(1.0/3.0)), 0);
		}
		//y = -5*alpha/6 - pow(Q,(1.0/3.0));
	} 
	else {
		//std::complex<double> y = -5*alpha/6 - P/(3*U) + U
		
		if (Q < 0) {
			y (-5*alpha/6, - pow(Q,(1.0/3.0)));
			y = y + U;
		}
		else {
			y (-5*alpha/6 - pow(Q,(1.0/3.0)), 0);
			y = y + U;
		}
	}
	//cout << "Value y = " << y << endl;
	
	//double w = sqrt(alpha+2*y);
	std::complex<double> w = pow(alpha+2*y, 1.0/2.0);
	
	
	
	//cout << "w = " << w << endl;
	
	//std::complex<double> a (-B/(4.0*A) + 0.5*w, 0.5*sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
	//std::complex<double> b (-B/(4.0*A) - 0.5*w, -0.5*sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
	
	if ( -(3.0*alpha+2.0*y+2.0*beta/w) < 0) {
		std::complex<double> a (-B/(4.0*A) + 0.5*w, 0.5*sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
		std::complex<double> b (-B/(4.0*A) + 0.5*w, -0.5*sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
		roots.at<double>(0,0) = real(a);
		roots.at<double>(0,1) = real(b);
	}
	else {
		roots.at<double>(0,0) = -B/(4.0*A) + 0.5*(w + sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
		roots.at<double>(0,1) = -B/(4.0*A) + 0.5*(w - sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
		
	}
	if ( -(3.0*alpha+2.0*y-2.0*beta/w) < 0) {
		std::complex<double> c (-B/(4.0*A) - 0.5*w, 0.5*sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
		std::complex<double> d (-B/(4.0*A) - 0.5*w, -0.5*sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
		roots.at<double>(0,2) = real(c);
		roots.at<double>(0,3) = real(d);
	}
	else {
		roots.at<double>(0,2) = -B/(4.0*A) + 0.5*(-w + sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
		roots.at<double>(0,3) = -B/(4.0*A) + 0.5*(-w - sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
	}
	*/
	if (isnan(roots.at<double>(0,0)) || isnan(roots.at<double>(0,1)) || isnan(roots.at<double>(0,2)) || isnan(roots.at<double>(0,3))) {
		//cout << "Roots = " << roots.at<double>(0,0) << ", " << roots.at<double>(0,1) << ", " << roots.at<double>(0,2) << ", " << roots.at<double>(0,3) << endl;
		//waitKey(0);
		//cout << "Factors = " << A << "," << B << "," << C << "," << D << "," << E << endl;
		//waitKey(0);
	}
	
	

	
	
	//Mat roots = Mat::zeros(1, 4, CV_64FC1);
	//cout << "Roots = " << roots.at<double>(0,0) << ", " << roots.at<double>(0,1) << ", " << roots.at<double>(0,2) << ", " << roots.at<double>(0,3) << endl;
	//cout << "a = " << real(a) << endl;
	//cout << "b = " << real(b) << endl;
	return roots;
}

 
Mat p3p(Mat worldPoints, Mat imageVectors) {
	
	/* Copyright (c) 2011, Laurent Kneip, ETH Zürich 
	 * 
	 * Insert list of conditions 
	 * 
	 */
	
	Mat poses = Mat::zeros(3, 16, CV_64FC1);
	
	Mat P1 = Mat::zeros(3, 1, CV_64FC1);
	Mat P1_complex = Mat_<std::complex<double>>(3,1);
	Mat P2 = Mat::zeros(3, 1, CV_64FC1); 
	Mat P3 = Mat::zeros(3, 1, CV_64FC1);
	// Assign values to P1, P2 and P3 
	for (int i = 0; i < 3; i++) {
		P1.at<double>(i,0) = worldPoints.at<double>(i,0);
		P1_complex.at<std::complex<double>>(i,0) = std::complex<double> (worldPoints.at<double>(i,0),0);
		P2.at<double>(i,0) = worldPoints.at<double>(i,1);
		P3.at<double>(i,0) = worldPoints.at<double>(i,2);
	}
	
	Mat vector1 = P2 - P1;
	Mat vector2 = P3 - P1;
	
	Mat crossV = crossProduct(vector1, vector2);
	if (sqrt(pow(crossV.at<double>(0,0),2.0) + pow(crossV.at<double>(1,0),2.0) + pow(crossV.at<double>(2,0),2.0)) == 0) {
		Mat emptyMatrix;
		return emptyMatrix;
	}
	
	Mat f1 = Mat::zeros(3, 1, CV_64FC1);
	Mat f2 = Mat::zeros(3, 1, CV_64FC1); 
	Mat f3 = Mat::zeros(3, 1, CV_64FC1);
	// Assign values to f1, f2 and f3
	for (int i = 0; i < 3; i++) {
		f1.at<double>(i,0) = imageVectors.at<double>(i,0);
		f2.at<double>(i,0) = imageVectors.at<double>(i,1);
		f3.at<double>(i,0) = imageVectors.at<double>(i,2);
	}
	
	Mat e1 = f1; 
	Mat e3 = crossProduct(f1,f2);
	double norm_e3 = sqrt(pow(e3.at<double>(0,0),2.0) + pow(e3.at<double>(1,0),2.0) + pow(e3.at<double>(2,0),2.0));
	e3.at<double>(0,0) = e3.at<double>(0,0) / norm_e3;
	e3.at<double>(1,0) = e3.at<double>(1,0) / norm_e3;
	e3.at<double>(2,0) = e3.at<double>(2,0) / norm_e3;
	Mat e2 = crossProduct(e3, e1);
	
	Mat T = Mat::zeros(3, 3, CV_64FC1);
	Mat T_complex = Mat_<std::complex<double>>(3,3);
	// Assign values to matrix T
	T.at<double>(0,0) = e1.at<double>(0,0);	
	T.at<double>(0,1) = e1.at<double>(1,0);	
	T.at<double>(0,2) = e1.at<double>(2,0);	
	T.at<double>(1,0) = e2.at<double>(0,0);	
	T.at<double>(1,1) = e2.at<double>(1,0);	
	T.at<double>(1,2) = e2.at<double>(2,0);	
	T.at<double>(2,0) = e3.at<double>(0,0);	
	T.at<double>(2,1) = e3.at<double>(1,0);	
	T.at<double>(2,2) = e3.at<double>(2,0);	
	T_complex.at<std::complex<double>>(0,0) = std::complex<double> (e1.at<double>(0,0),0);
	T_complex.at<std::complex<double>>(0,1) = std::complex<double> (e1.at<double>(1,0),0);
	T_complex.at<std::complex<double>>(0,2) = std::complex<double> (e1.at<double>(2,0),0);
	T_complex.at<std::complex<double>>(1,0) = std::complex<double> (e2.at<double>(0,0),0);
	T_complex.at<std::complex<double>>(1,1) = std::complex<double> (e2.at<double>(1,0),0);
	T_complex.at<std::complex<double>>(1,2) = std::complex<double> (e2.at<double>(2,0),0);
	T_complex.at<std::complex<double>>(2,0) = std::complex<double> (e3.at<double>(0,0),0);
	T_complex.at<std::complex<double>>(2,1) = std::complex<double> (e3.at<double>(1,0),0);
	T_complex.at<std::complex<double>>(2,2) = std::complex<double> (e3.at<double>(2,0),0);
	
	f3 = T * f3;
	
	// To reinforce that f3[2] > 0 for having theta in [0;pi]
	if (f3.at<double>(2,0) > 0) {

		// Assign values to f1, f2 and f3  
		for (int i = 0; i < 3; i++) {
			f1.at<double>(i,0) = imageVectors.at<double>(i,1);
			f2.at<double>(i,0) = imageVectors.at<double>(i,0);
			f3.at<double>(i,0) = imageVectors.at<double>(i,2);
		}
		
		Mat e1 = f1; 
		Mat e3 = crossProduct(f1,f2);
		double norm_e3 = sqrt(pow(e3.at<double>(0,0),2.0) + pow(e3.at<double>(1,0),2.0) + pow(e3.at<double>(2,0),2.0));
		e3.at<double>(0,0) = e3.at<double>(0,0) / norm_e3;
		e3.at<double>(1,0) = e3.at<double>(1,0) / norm_e3;
		e3.at<double>(2,0) = e3.at<double>(2,0) / norm_e3;
		Mat e2 = crossProduct(e3, e1);
		
		// Assign values to matrix T
		T.at<double>(0,0) = e1.at<double>(0,0);	
		T.at<double>(0,1) = e1.at<double>(1,0);	
		T.at<double>(0,2) = e1.at<double>(2,0);	
		T.at<double>(1,0) = e2.at<double>(0,0);	
		T.at<double>(1,1) = e2.at<double>(1,0);	
		T.at<double>(1,2) = e2.at<double>(2,0);	
		T.at<double>(2,0) = e3.at<double>(0,0);	
		T.at<double>(2,1) = e3.at<double>(1,0);	
		T.at<double>(2,2) = e3.at<double>(2,0);	
		T_complex.at<std::complex<double>>(0,0) = std::complex<double> (e1.at<double>(0,0),0);
		T_complex.at<std::complex<double>>(0,1) = std::complex<double> (e1.at<double>(1,0),0);
		T_complex.at<std::complex<double>>(0,2) = std::complex<double> (e1.at<double>(2,0),0);
		T_complex.at<std::complex<double>>(1,0) = std::complex<double> (e2.at<double>(0,0),0);
		T_complex.at<std::complex<double>>(1,1) = std::complex<double> (e2.at<double>(1,0),0);
		T_complex.at<std::complex<double>>(1,2) = std::complex<double> (e2.at<double>(2,0),0);
		T_complex.at<std::complex<double>>(2,0) = std::complex<double> (e3.at<double>(0,0),0);
		T_complex.at<std::complex<double>>(2,1) = std::complex<double> (e3.at<double>(1,0),0);
		T_complex.at<std::complex<double>>(2,2) = std::complex<double> (e3.at<double>(2,0),0);
		
		f3 = T * f3;
		
		// Reassign values to P1, P2 and P3 
		for (int i = 0; i < 3; i++) {
			P1.at<double>(i,0) = worldPoints.at<double>(i,1);
			P1_complex.at<std::complex<double>>(i,0) = std::complex<double> (worldPoints.at<double>(i,1),0);
			P2.at<double>(i,0) = worldPoints.at<double>(i,0);
			P3.at<double>(i,0) = worldPoints.at<double>(i,2);
		}
		
	}
	
	Mat n1 = P2 - P1; 
	double norm_n1 = sqrt(pow(n1.at<double>(0,0),2.0) + pow(n1.at<double>(1,0),2.0) + pow(n1.at<double>(2,0),2.0)); 
	n1.at<double>(0,0) = n1.at<double>(0,0) / norm_n1;
	n1.at<double>(1,0) = n1.at<double>(1,0) / norm_n1;
	n1.at<double>(2,0) = n1.at<double>(2,0) / norm_n1;
	Mat n3 = crossProduct(n1, (P3-P1));
	double norm_n3 = sqrt(pow(n3.at<double>(0,0),2.0) + pow(n3.at<double>(1,0),2.0) + pow(n3.at<double>(2,0),2.0)); 
	n3.at<double>(0,0) = n3.at<double>(0,0) / norm_n3;
	n3.at<double>(1,0) = n3.at<double>(1,0) / norm_n3;
	n3.at<double>(2,0) = n3.at<double>(2,0) / norm_n3; 
	Mat n2 = crossProduct(n3, n1);
	
	// Matrix N 
	Mat N = Mat::zeros(3, 3, CV_64FC1);
	N.at<double>(0,0) = n1.at<double>(0,0);	
	N.at<double>(0,1) = n1.at<double>(1,0);	
	N.at<double>(0,2) = n1.at<double>(2,0);	
	N.at<double>(1,0) = n2.at<double>(0,0);	
	N.at<double>(1,1) = n2.at<double>(1,0);	
	N.at<double>(1,2) = n2.at<double>(2,0);	
	N.at<double>(2,0) = n3.at<double>(0,0);	
	N.at<double>(2,1) = n3.at<double>(1,0);	
	N.at<double>(2,2) = n3.at<double>(2,0);	
	
	Mat N_complex = Mat_<std::complex<double>>(3, 3);
	N_complex.at<std::complex<double>>(0,0) = std::complex<double> (n1.at<double>(0,0),0);
	N_complex.at<std::complex<double>>(0,1) = std::complex<double> (n1.at<double>(1,0),0);
	N_complex.at<std::complex<double>>(0,2) = std::complex<double> (n1.at<double>(2,0),0);
	N_complex.at<std::complex<double>>(1,0) = std::complex<double> (n2.at<double>(0,0),0);
	N_complex.at<std::complex<double>>(1,1) = std::complex<double> (n2.at<double>(1,0),0);
	N_complex.at<std::complex<double>>(1,2) = std::complex<double> (n2.at<double>(2,0),0);
	N_complex.at<std::complex<double>>(2,0) = std::complex<double> (n3.at<double>(0,0),0);
	N_complex.at<std::complex<double>>(2,1) = std::complex<double> (n3.at<double>(1,0),0);
	N_complex.at<std::complex<double>>(2,2) = std::complex<double> (n3.at<double>(2,0),0);
	
	
	// Extraction of known parameters 
	P3 = N * (P3 - P1);
	
	Mat v = P2 - P1;
	double d_12 = sqrt(pow(v.at<double>(0,0),2.0) + pow(v.at<double>(1,0),2.0) + pow(v.at<double>(2,0),2.0));
	double f_1 = f3.at<double>(0,0) / f3.at<double>(2,0);
	double f_2 = f3.at<double>(1,0) / f3.at<double>(2.0);
	double p_1 = P3.at<double>(0,0);
	double p_2 = P3.at<double>(1,0);
	
	//double cos_beta = f1.t() * f2; // Check this calculation. There might be a mistake. 
	double cos_beta = f1.at<double>(0,0) * f2.at<double>(0,0) + f1.at<double>(1,0) * f2.at<double>(1,0) + f1.at<double>(2,0) * f2.at<double>(2,0);
	double b = (1/(1 - pow(cos_beta,2.0))) - 1;
	
	if (cos_beta < 0) {
		b = -sqrt(b);
	}
	else {
		b = sqrt(b);
	}
	
	double f_1_pw2 = pow(f_1, 2.0);
	double f_2_pw2 = pow(f_2, 2.0);
	double p_1_pw2 = pow(p_1, 2.0);
	double p_1_pw3 = p_1_pw2 * p_1;
	double p_1_pw4 = p_1_pw3 * p_1;
	double p_2_pw2 = pow(p_2, 2.0);
	double p_2_pw3 = p_2_pw2 * p_2;
	double p_2_pw4 = p_2_pw3 * p_2;
	double d_12_pw2 = pow(d_12, 2.0);
	double b_pw2 =  pow(b, 2.0);
	
	// Factors of the 4th degree polynomial
	Mat factors = Mat::zeros(1, 5, CV_64FC1);
	factors.at<double>(0,0) = -f_2_pw2 * p_2_pw4 - p_2_pw4 * f_1_pw2 - p_2_pw4;
	
	factors.at<double>(0,1)= 2*p_2_pw3*d_12*b + 2*f_2_pw2*p_2_pw3*d_12*b - 2*f_2*p_2_pw3*f_1*d_12;
	
	factors.at<double>(0,2) = -f_2_pw2*p_2_pw2*p_1_pw2 - f_2_pw2*p_2_pw2*d_12_pw2*b_pw2 - f_2_pw2*p_2_pw2*d_12_pw2;
	factors.at<double>(0,2) = factors.at<double>(0,2) + f_2_pw2*p_2_pw4 + p_2_pw4*f_1_pw2 + 2*p_1*p_2_pw2*d_12;
	factors.at<double>(0,2) = factors.at<double>(0,2) + 2*f_1*f_2*p_1*p_2_pw2*d_12*b - p_2_pw2*p_1_pw2*f_1_pw2;
	factors.at<double>(0,2) = factors.at<double>(0,2) + 2*p_1*p_2_pw2*f_2_pw2*d_12 - p_2_pw2*d_12_pw2*b_pw2 - 2*p_1_pw2*p_2_pw2;
	
	factors.at<double>(0,3) = 2*p_1_pw2*p_2*d_12*b + 2*f_2*p_2_pw3*f_1*d_12 - 2*f_2_pw2*p_2_pw3*d_12*b - 2*p_1*p_2*d_12_pw2*b;
	
	factors.at<double>(0,4) = -2*f_2*p_2_pw2*f_1*p_1*d_12*b + f_2_pw2*p_2_pw2*d_12_pw2 + 2*p_1_pw3*d_12 - p_1_pw2*d_12_pw2 + f_2_pw2*p_2_pw2*p_1_pw2 - p_1_pw4;
	factors.at<double>(0,4) = factors.at<double>(0,4) - 2*f_2_pw2*p_2_pw2*p_1*d_12 + p_2_pw2*f_1_pw2*p_1_pw2 + f_2_pw2*p_2_pw2*d_12_pw2*b_pw2;
	
	// Computation of roots 
	Mat x = solveQuartic( factors );
	
	// Variables 
	Mat C = Mat_<std::complex<double>>(3, 1);
	Mat R = Mat_<std::complex<double>> (3,3);
	//std::complex<double> null_v (0,0);
	std::complex<double> real_value;
	
	for (int i = 0; i < 4; i++) {
		std::complex<double> cot_alpha ((-f_1*p_1/f_2 - x.at<double>(0,i)*p_2+d_12*b)/(-f_1*x.at<double>(0,i)*p_2/f_2 + p_1-d_12),0);
		
		//cout << "cot_alpha = " << cot_alpha << endl;
		
		//double cos_theta = x.at<double>(0, i);
		//double sin_theta = sqrt( 1 - pow(x.at<double>(0, i),2.0) );
		//double sin_alpha = sqrt( 1/(pow(cot_alpha,2.0) +1) ); 
		//double cos_alpha = sqrt( 1 - pow(sin_alpha,2.0) );	
		
		std::complex<double> cos_theta = x.at<double>(0,i) + 0i;
		//cout << "cos_theta = " << cos_theta << endl;
		real_value = pow(x.at<double>(0,i), 2.0);
		std::complex<double> sin_theta = pow(1. - real_value, 1.0/2.0);
		//cout << "sin_theta = " << sin_theta << endl;
		real_value = pow(cot_alpha, 2.0);
		std::complex<double> sin_alpha = pow(1./(real_value + 1.), 1.0/2.0);
		//cout << "sin_alpha = " << sin_alpha << endl;
		real_value = pow(sin_alpha, 2.0);
		std::complex<double> cos_alpha = pow(1. - real_value, 1.0/2.0);
		//cout << "cos_alpha = " << cos_alpha << endl;
		
		
		if (real(cot_alpha) < 0) {
			cos_alpha = -cos_alpha;  // Check this for potential mistake 
		}
		//cout << "cos_alpha after if = " << cos_alpha << endl;
		
		//Mat C = Mat_<std::complex<double>(3, 1);
		C.at<std::complex<double>>(0,0) = d_12*cos_alpha*(sin_alpha*b+cos_alpha);
		C.at<std::complex<double>>(1,0) = cos_theta*d_12*sin_alpha*(sin_alpha*b + cos_alpha);
		C.at<std::complex<double>>(2,0) = sin_theta*d_12*sin_alpha*(sin_alpha*b + cos_alpha);
		
		//cout << "C Matrix " << endl;
		for (int r = 0; r < C.rows; r++) {
			for (int c = 0; c < C.cols; c++) {
			//	cout << C.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
		//cout << "N_complex" << endl;
		for (int r = 0; r < N_complex.rows; r++) {
			for (int c = 0; c < N_complex.cols; c++) {
			//	cout << N_complex.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
		Mat Nt = N_complex.t();
		
		//cout << "Nt " << endl;
		for (int r = 0; r < Nt.rows; r++) {
			for (int c = 0; c < Nt.cols; c++) {
			//	cout << Nt.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
		Mat temp = Nt*C;
		//cout << "Product Nt and C " << endl;
		for (int r = 0; r < temp.rows; r++) {
			for (int c = 0; c < temp.cols; c++) {
				cout << temp.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
	//	cout << "P1_complex " << endl;
		for (int r = 0; r < P1_complex.rows; r++) {
			for (int c = 0; c < P1_complex.cols; c++) {
			//	cout << P1_complex.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
		C = P1_complex + temp;
		
		//cout << "C Matrix after product" << endl;
		for (int r = 0; r < C.rows; r++) {
			for (int c = 0; c < C.cols; c++) {
			//	cout << C.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
		//Mat R = Mat_<std::complex<double>> (3,3);
		R.at<std::complex<double>>(0,0) = -cos_alpha;
		R.at<std::complex<double>>(0,1) = -sin_alpha*cos_theta;
		R.at<std::complex<double>>(0,2) = -sin_alpha*sin_theta;
		R.at<std::complex<double>>(1,0) = sin_alpha;
		R.at<std::complex<double>>(1,1) = -cos_alpha*cos_theta;
		R.at<std::complex<double>>(1,2) = -cos_alpha*sin_theta;
		R.at<std::complex<double>>(2,0) = 0;
		R.at<std::complex<double>>(2,1) = -sin_theta; 
		R.at<std::complex<double>>(2,2) = cos_theta;
		
		//cout << "R" << endl;
		for (int r = 0; r < R.rows; r++) {
			for (int c = 0; c < R.cols; c++) {
				//cout << R.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
		//cout << "Matrix T" << endl;
		for (int r = 0; r < T_complex.rows; r++) {
			for (int c = 0; c < T_complex.cols; c++) {
			//	cout << T_complex.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
		R = N_complex.t() * R.t() * T_complex;
		
		//cout << "R after product" << endl;
		for (int r = 0; r < R.rows; r++) {
			for (int c = 0; c < R.cols; c++) {
			//	cout << R.at<std::complex<double>>(r,c) << ", ";
			}
			//cout << "" << endl;
		}
		
		/*
		Mat C = Mat::zeros(3, 1, CV_64FC1);
		C.at<double>(0,0) = d_12*cos_alpha*(sin_alpha*b+cos_alpha);
		C.at<double>(1,0) = cos_theta*d_12*sin_alpha*(sin_alpha*b + cos_alpha);
		C.at<double>(2,0) = sin_theta*d_12*sin_alpha*(sin_alpha*b + cos_alpha);
		
		C  = P1 + N.t() * C; // Be aware of transpose of N here 
		
		Mat R = Mat::zeros(3, 3, CV_64FC1);
		R.at<double>(0,0) = -cos_alpha;
		R.at<double>(0,1) = -sin_alpha*cos_theta;
		R.at<double>(0,2) = -sin_alpha*sin_theta;
		R.at<double>(1,0) = sin_alpha;
		R.at<double>(1,1) = -cos_alpha*cos_theta;
		R.at<double>(1,2) = -cos_alpha*sin_theta;
		R.at<double>(2,1) = -sin_theta; 
		R.at<double>(2,2) = cos_theta;
		
		R = N.t() * R.t() * T; 	// Be aware of transpose here
		*/

		// Update poses 
		poses.at<double>(0,i*4) = C.at<std::complex<double>>(0,0).real();
		poses.at<double>(1,i*4) = C.at<std::complex<double>>(1,0).real();
		poses.at<double>(2,i*4) = C.at<std::complex<double>>(2,0).real();
		
		// Insert values from matrix R
		poses.at<double>(0,i*4+1) = R.at<std::complex<double>>(0,0).real();
		poses.at<double>(1,i*4+1) = R.at<std::complex<double>>(1,0).real();
		poses.at<double>(2,i*4+1) = R.at<std::complex<double>>(2,0).real();
		
		poses.at<double>(0,i*4+2) = R.at<std::complex<double>>(0,1).real();
		poses.at<double>(1,i*4+2) = R.at<std::complex<double>>(1,1).real();
		poses.at<double>(2,i*4+2) = R.at<std::complex<double>>(2,1).real();
		
		poses.at<double>(0,i*4+3) = R.at<std::complex<double>>(0,2).real();
		poses.at<double>(1,i*4+3) = R.at<std::complex<double>>(1,2).real();
		poses.at<double>(2,i*4+3) = R.at<std::complex<double>>(2,2).real();
	}
	return poses;
}



tuple<Mat, Mat> Localize::ransacLocalization(Mat keypoints_i, Mat corresponding_landmarks, Mat K) {
	// Transformation matrix 
	Mat transformation_matrix = Mat::zeros(3, 4, CV_64FC1);
	
	// Method parameter
	bool adaptive_ransac = true;
	
	// Other parameters 
	double num_iterations;
	int pixel_tolerance = 10; 
	double k = 3.0;
	int min_inlier_count = 15; // This parameter should be tuned for the implementation
	double record_inlier = 0;
		
	if (adaptive_ransac) {
		num_iterations = 1000;
	}
	else {
		num_iterations = INFINITY;
	}
	
	// Initialize RANSAC
	Mat best_inlier_mask = Mat::zeros(1, keypoints_i.cols, CV_64FC1);
	// Switch from (row, col) to (u, v)
	Mat matched_query_keypoints = Mat::zeros(2, keypoints_i.cols, CV_64FC1);
	for (int i = 0; i < keypoints_i.cols; i++) {
		matched_query_keypoints.at<double>(0,i) = keypoints_i.at<double>(1,i); // x-coordinate in iamge (u)
		matched_query_keypoints.at<double>(1,i) = keypoints_i.at<double>(0,i); // y-coordinate in image (v)
	}
	Mat max_num_inliers_history = Mat::zeros(50, 1, CV_64FC1); // Should probably be changed
	Mat num_iteration_history = Mat::zeros(50, 1, CV_64FC1); // Should probably be changed to 
	int max_num_inliers = 0;
	
	// RANSAC
	int i = 0;
	
	// Needed variables 
	Mat landmark_sample = Mat::zeros(3, k, CV_64FC1);
	Mat keypoint_sample = Mat::ones(3, k, CV_64FC1); // To avoid having to set last row to 1's afterwards
	
	// Initialization of matrices 
	Mat normalized_bearings, poses, points, projected_points, difference, errors, is_inlier, alternative_is_inlier;
	Mat best_R_C_W, best_t_C_W;
	Mat R_C_W_guess = Mat::zeros(3, 3, CV_64FC1);
	Mat t_C_W_guess = Mat::zeros(3, 1, CV_64FC1);
	Mat R_W_C = Mat::zeros(3, 3, CV_64FC1);
	Mat t_W_C = Mat::zeros(3, 1, CV_64FC1);
	
	
	Mat random_test = Mat::zeros(3,3,CV_64FC1);
	random_test.at<double>(0,0) = 187-1; 
	random_test.at<double>(0,1) = 203-1;
	random_test.at<double>(0,2) = 122-1;
	random_test.at<double>(1,0) = 23-1;
	random_test.at<double>(1,1) = 62-1;
	random_test.at<double>(1,2) = 246-1;
	random_test.at<double>(2,0) = 42-1;
	random_test.at<double>(2,1) = 223-1;
	random_test.at<double>(2,2) = 145-1;
	
	

	
	while ( num_iterations-1 > i ) {
	//while (1 > i) {
		
		int random_nums[corresponding_landmarks.cols];
		for (int mm = 0; mm < corresponding_landmarks.cols; mm++) {
			random_nums[mm] = mm;
		}
		random_shuffle(random_nums, random_nums + corresponding_landmarks.cols);
		for (int mm = 0; mm < k;  mm++) {
			cout << "Random number = " << random_nums[mm] << endl;
			// Landmark sample 
			landmark_sample.at<double>(0,mm) = corresponding_landmarks.at<double>(0, random_nums[mm]);
			landmark_sample.at<double>(1,mm) = corresponding_landmarks.at<double>(1, random_nums[mm]);
			landmark_sample.at<double>(2,mm) = corresponding_landmarks.at<double>(2, random_nums[mm]);
			
			// Keypoint sample 
			keypoint_sample.at<double>(0,mm) = matched_query_keypoints.at<double>(0, random_nums[mm]);
			keypoint_sample.at<double>(1,mm) = matched_query_keypoints.at<double>(1, random_nums[mm]);
		}
		
		cout << "Print of landmark-sample" << endl;
		for (int r = 0; r < landmark_sample.rows; r++) {
			for (int c = 0; c < landmark_sample.cols; c++) {
				cout << landmark_sample.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		cout << "Print of keypoint-sample" << endl;
		for (int r = 0; r < keypoint_sample.rows; r++) {
			for (int c = 0; c < keypoint_sample.cols; c++) {
				cout << keypoint_sample.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		//waitKey(5000);
		
		
		
		
		
		/*
		for (int qq = 0; qq < 3; qq++) {
			cout << "random_test point = " << random_test.at<double>(i,qq) << endl;
			landmark_sample.at<double>(0,qq) = corresponding_landmarks.at<double>(0, random_test.at<double>(i,qq));
			landmark_sample.at<double>(1,qq) = corresponding_landmarks.at<double>(1, random_test.at<double>(i,qq)); 
			landmark_sample.at<double>(2,qq) = corresponding_landmarks.at<double>(2, random_test.at<double>(i,qq));
			keypoint_sample.at<double>(0,qq) = matched_query_keypoints.at<double>(0, random_test.at<double>(i,qq));
			keypoint_sample.at<double>(1,qq) = matched_query_keypoints.at<double>(1, random_test.at<double>(i,qq));
		}
		cout << "Print of landmark-sample" << endl;
		for (int r = 0; r < landmark_sample.rows; r++) {
			for (int c = 0; c < landmark_sample.cols; c++) {
				cout << landmark_sample.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		cout << "Print of keypoint-sample" << endl;
		for (int r = 0; r < keypoint_sample.rows; r++) {
			for (int c = 0; c < keypoint_sample.cols; c++) {
				cout << keypoint_sample.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		//waitKey(10000);
		*/
		
		
		
		
		normalized_bearings = K.inv() * keypoint_sample;
		
		for (int ii = 0; ii < 3; ii++) {
			double vector_norm = sqrt(pow(normalized_bearings.at<double>(0,ii),2.0) + pow(normalized_bearings.at<double>(1,ii),2.0) + pow(normalized_bearings.at<double>(2,ii),2.0));
			normalized_bearings.at<double>(0,ii) = normalized_bearings.at<double>(0,ii)/vector_norm;
			normalized_bearings.at<double>(1,ii) = normalized_bearings.at<double>(1,ii)/vector_norm;
			normalized_bearings.at<double>(2,ii) = normalized_bearings.at<double>(2,ii)/vector_norm;
		
		}
		
		
		poses = p3p(landmark_sample, normalized_bearings);
		/*
		cout << "Poses " << endl;
		for (int r = 0; r < poses.rows; r++) {
			for (int c = 0; c < poses.cols; c++) {
				cout << poses.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		*/
		for (int r = 0; r < poses.rows; r++) {
			for (int c = 0; c < poses.cols; c++) {
				if (isnan(poses.at<double>(r,c))) {
					cout << "Nan Value detected in ransac localization" << endl;
					cout << "landmark_sample" << endl;
					for (int r = 0; r < landmark_sample.rows; r++) {
						for (int c = 0; c < landmark_sample.cols; c++) {
							cout << landmark_sample.at<double>(r,c) << ", ";
						}
						cout << "" << endl;
					}
					cout << "normalized_bearings" << endl;
					for (int r = 0; r < normalized_bearings.rows; r++) {
						for (int c = 0; c < normalized_bearings.cols; c++) {
							cout << normalized_bearings.at<double>(r,c) << ", ";
						}
						cout << "" << endl;
					}
					
					
					Mat empty = Mat::zeros(3, 3, CV_64FC1);
					return make_tuple(empty, empty);
				}
			}
		}
		
		/*
		Mat poses = Mat::zeros(3, 16, CV_64FC1);
		
		
		ifstream MyReadFile2("new_poses.txt");
		
		if (MyReadFile2.is_open()) {
			for (int i = 0; i < 16; i++) {
					MyReadFile2 >> poses.at<double>(0,i);
					MyReadFile2 >> poses.at<double>(1,i);
					MyReadFile2 >> poses.at<double>(2,i);
				
			}
		}
		cout << "Poses from file" << endl;
		for (int r = 0; r < poses.rows; r++) {
			for (int c = 0; c < poses.cols; c++) {
				cout << poses.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
			cout << "" << endl;
		}

		MyReadFile2.close();
		*/
		
			
		// Decode the p3p output 
		// Four possible rotations and four possible translations 
		//Mat R_C_W_guess = Mat::zeros(3, 3, CV_64FC1);
		//Mat t_C_W_guess = Mat::zeros(3, 1, CV_64FC1);

		
		// First guess 
		//Mat R_W_C = Mat::zeros(3, 3, CV_64FC1);
		//Mat t_W_C = Mat::zeros(3, 1, CV_64FC1);
		for (int k = 0; k < 3; k++) {
			R_W_C.at<double>(0,k) = poses.at<double>(0,k+1);
			R_W_C.at<double>(1,k) = poses.at<double>(1,k+1);
			R_W_C.at<double>(2,k) = poses.at<double>(2,k+1);
			t_W_C.at<double>(k,0) = poses.at<double>(k,0);
		}
		// From frame W_C til C_W
		R_C_W_guess = R_W_C.t(); // Be aware of tranpose 
		t_C_W_guess = -R_W_C.t()*t_W_C;
		
		/*
		cout << "R_C_W_guess" << endl;
		for (int r = 0; r < R_C_W_guess.rows; r++) {
			for (int c = 0; c < R_C_W_guess.cols; c++) {
				cout << R_C_W_guess.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		cout << "t_C_W_guess" << endl;
		for (int r = 0; r < t_C_W_guess.rows; r++) {
			for (int c = 0; c < t_C_W_guess.cols; c++) {
				cout << t_C_W_guess.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		*/
		
		
		points = R_C_W_guess * corresponding_landmarks + repeat(t_C_W_guess, 1, corresponding_landmarks.cols);
		
		projected_points = projectPoints(points, K);
		
		/*
		cout << "projected_points" << endl;
		for (int r = 0; r < projected_points.rows; r++) {
			for (int c = 0; c < projected_points.cols; c++) {
				cout << projected_points.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		cout << "keypoints_i" << endl;
		for (int r = 0; r < keypoints_i.rows; r++) {
			for (int c = 0; c < keypoints_i.cols; c++) {
				cout << keypoints_i.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		*/
		
		
		difference = matched_query_keypoints - projected_points;
		
		/*
		cout << "difference" << endl;
		for (int r = 0; r < difference.rows; r++) {
			for (int c = 0; c < difference.cols; c++) {
				cout << difference.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		cout << "" << endl;
		*/
		errors = difference.mul(difference);
		errors = errors.row(0) + errors.row(1);
		
		/*
		cout << "errors" << endl;
		for (int r = 0; r < errors.rows; r++) {
			for (int c = 0; c < errors.cols; c++) {
				cout << errors.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		cout << "" << endl;
		*/
		is_inlier = errors < pow(pixel_tolerance,2.0); // Remember this matrix is of type uchar 
		
		/*
		cout << "is_inlier" << endl;
		for (int r = 0; r < is_inlier.rows; r++) {
			for (int c = 0; c < is_inlier.cols; c++) {
				double v = is_inlier.at<uchar>(r,c);
				cout << v << ", ";
			}
			cout << "" << endl;
		}
		cout << "" << endl;
		*/
		//cout << "First iter" << endl;
		//cout << "countNonZero(is_inlier) = " << countNonZero(is_inlier) << endl;
		//waitKey(5000);
		if (countNonZero(is_inlier) > record_inlier && countNonZero(is_inlier) >= min_inlier_count) {
			//waitKey(5000);
			record_inlier = countNonZero(is_inlier);
			R_C_W_guess.copyTo(best_R_C_W);
			t_C_W_guess.copyTo(best_t_C_W);
			
			/*
			cout << "best_R_C_W" << endl;
			for (int r = 0; r < best_R_C_W.rows; r++) {
				for (int c = 0; c < best_R_C_W.cols; c++) {
					cout << best_R_C_W.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
			}
			cout << "" << endl;
			cout << "best_t_C_W" << endl;
			for (int r = 0; r < best_t_C_W.rows; r++) {
				for (int c = 0; c < best_t_C_W.cols; c++) {
					cout << best_t_C_W.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
			}
			cout << "" << endl;
			*/
		
		}
		
		for (int alt_idx = 1; alt_idx <= 3; alt_idx++) {
			for (int k = 0; k < 3; k++) {
				R_W_C.at<double>(0,k) = poses.at<double>(0,k+1 + alt_idx*4);
				R_W_C.at<double>(1,k) = poses.at<double>(1,k+1 + alt_idx*4);
				R_W_C.at<double>(2,k) = poses.at<double>(2,k+1 + alt_idx*4);
				t_W_C.at<double>(k,0) = poses.at<double>(k, alt_idx*4);
			}
			
			/*
			cout << "R_W_C" << endl;
			for (int r = 0; r < R_W_C.rows; r++) {
				for (int c = 0; c < R_W_C.cols; c++) {
					cout << R_W_C.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
			}
			*/
			R_C_W_guess = R_W_C.t();
			/*
			cout << "R_C_W_guess" << endl;
			for (int r = 0; r < R_C_W_guess.rows; r++) {
				for (int c = 0; c < R_C_W_guess.cols; c++) {
					cout << R_C_W_guess.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
			}
			*/
			t_C_W_guess = -R_W_C.t()*t_W_C;
			
			/*
			cout << "t_C_W_guess" << endl;
			for (int r = 0; r < t_C_W_guess.rows; r++) {
				for (int c = 0; c < t_C_W_guess.cols; c++) {
					cout << t_C_W_guess.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
			}
			*/
			
			points = R_C_W_guess * corresponding_landmarks + repeat(t_C_W_guess, 1, corresponding_landmarks.cols);
			projected_points = projectPoints(points, K);
			
			/*
			cout << "projected_points" << endl;
			for (int r = 0; r < projected_points.rows; r++) {
				for (int c = 0; c < projected_points.cols; c++) {
					//cout << projected_points.at<double>(r,c) << ", ";
				}
				//cout << "" << endl;
			}
			*/
			difference = matched_query_keypoints - projected_points;
			errors = difference.mul(difference);
			errors = errors.row(0) + errors.row(1);
			alternative_is_inlier = errors < pow(pixel_tolerance,2.0);
			
			//cout << "Inside the three other rotations" << endl;
			//cout << " countNonZero(alternative_is_inlier) = " << countNonZero(alternative_is_inlier) << endl;
			//cout << "Before if countNonZero(is_inlier) = " << countNonZero(is_inlier) << endl;
			if (countNonZero(alternative_is_inlier) > countNonZero(is_inlier) ) {
				//is_inlier = alternative_is_inlier;
				alternative_is_inlier.copyTo(is_inlier);
			} 
			//cout << "countNonZero(is_inlier) = " << countNonZero(is_inlier) << endl;
			//cout << "Test if it is a pointer" << endl;
			//alternative_is_inlier = Mat::zeros(1, alternative_is_inlier.cols, CV_64FC1);
			//alternative_is_inlier = errors < pow(2,2.0);
			//cout << " countNonZero(alternative_is_inlier) = " << countNonZero(alternative_is_inlier) << endl;
			//cout << "countNonZero(is_inlier) = " << countNonZero(is_inlier) << endl;
			
			//waitKey(5000);
			if (countNonZero(is_inlier) > record_inlier && countNonZero(is_inlier) >= min_inlier_count) {
				//waitKey(5000);
				record_inlier = countNonZero(is_inlier);
				R_C_W_guess.copyTo(best_R_C_W);
				t_C_W_guess.copyTo(best_t_C_W);
				
				/*
				cout << "best_R_C_W" << endl;
				for (int r = 0; r < best_R_C_W.rows; r++) {
					for (int c = 0; c < best_R_C_W.cols; c++) {
						cout << best_R_C_W.at<double>(r,c) << ", ";
					}
					cout << "" << endl;
				}
				cout << "" << endl;
				cout << "best_t_C_W" << endl;
				for (int r = 0; r < best_t_C_W.rows; r++) {
					for (int c = 0; c < best_t_C_W.cols; c++) {
						cout << best_t_C_W.at<double>(r,c) << ", ";
					}
					cout << "" << endl;
				}
				cout << "" << endl;
				*/
			}
			
		}
		
		//cout << "Update max_num_inliers " << endl;
		//cout << "countNonZero(is_inlier) = " << countNonZero(is_inlier) << endl;
		//waitKey(5000);
	
		
		if (countNonZero(is_inlier) > max_num_inliers && countNonZero(is_inlier) >= min_inlier_count) {
			max_num_inliers = countNonZero(is_inlier);
			best_inlier_mask = is_inlier;
		}
		
		//cout << "Before adaptive_ransac " << endl;
		//cout << "max_num_inliers = " << max_num_inliers << endl;
		//cout << "is_inlier.cols = " << is_inlier.cols << endl;
		if (adaptive_ransac) {
			float division = (float) max_num_inliers/ (float) is_inlier.cols;
			//cout << "division = " << division << endl;
			float outlier_ratio = 1. - division;
			//cout << "max_num_inliers " << max_num_inliers << endl;
			//cout << "oulier ratio = "  << outlier_ratio << endl;
			
			float confidence = 0.95; 
			float upper_bound_on_outlier_ratio = 0.90;
			outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio);
			num_iterations = log( 1 - confidence)/log(1-pow((1-outlier_ratio),k)); 
			//cout << "num_iterations = " << num_iterations << endl;
			
			double v = 15000;
			num_iterations = min(v, num_iterations);
			//cout << "num iterations after min-operation = " << num_iterations << endl;
		}
		
		i++;
	}	
	cout << "best_R_C_W" << endl;
				for (int r = 0; r < best_R_C_W.rows; r++) {
					for (int c = 0; c < best_R_C_W.cols; c++) {
						cout << best_R_C_W.at<double>(r,c) << ", ";
					}
					cout << "" << endl;
				}
				cout << "" << endl;
				cout << "best_t_C_W" << endl;
				for (int r = 0; r < best_t_C_W.rows; r++) {
					for (int c = 0; c < best_t_C_W.cols; c++) {
						cout << best_t_C_W.at<double>(r,c) << ", ";
					}
					cout << "" << endl;
				}
				cout << "" << endl;
	
	if (max_num_inliers != 0) {
		hconcat(best_R_C_W, best_t_C_W, transformation_matrix);
	}
	
	return make_tuple(transformation_matrix, best_inlier_mask);
}


// ############################# triangulate New Landmarks #############################
/* Objective: Locate new keypoints in the image 
 * Inputs: 
 * Ii - C++ Matrix - current image
 * Si - state (a struct) with multiple attributes
 * T_wc - 3x4 Matrix 
 * Output 
 * state Si
 */
state newCandidateKeypoints(Mat Ii, state Si, Mat T_wc) {
	
	// Convert the image to gray scale 
	Mat Ii_gray;
	cvtColor(Ii, Ii_gray, COLOR_BGR2GRAY);
	
	/*
	// Make sure you only find new keypoints and not keypoints you have already tracked
	for (int i = 0; i < Si.k; i++) {
		double y = Si.Pi.at<double>(0,i); // y-coordinate of image
		double x = Si.Pi.at<double>(1,i); // x-coordinate of image
		
		// Area of 5x5 is set to zero in the image
		for (int r = -2; r < 3; r++) {
			for (int c = -2; c < 3; c++) {
				Ii.at<uchar>(y+r,x+c) = 0;
				Ii_gray.at<uchar>(y+r,x+c) = 0;
			}
		}
	}
	*/
	
	
	// Find new keypoints 
	int keypoint_max = 100;
	Mat candidate_keypoints = Harris::corner(Ii, Ii_gray, keypoint_max, Si.Pi);
	
	vconcat(candidate_keypoints.row(1), candidate_keypoints.row(2), candidate_keypoints); // y-coordinates in row 1. x-coordinates in row 2.
	
	// Update state 
	Si.num_candidates = keypoint_max;
	
	// Update keypoints
	candidate_keypoints.copyTo(Si.Ci); 
	
	// Update first observation of keypoints
	candidate_keypoints.copyTo(Si.Fi);
	
	// Update the camera poses at the first observations.
	Mat t_C_W_vector = T_wc.reshape(0,T_wc.rows * T_wc.cols);
	Mat poses = repeat(t_C_W_vector, 1, candidate_keypoints.cols);
	poses.copyTo(Si.Ti);
	
	return Si;
}

state continuousCandidateKeypoints(Mat Ii_1, Mat Ii, state Si, Mat T_wc, Mat extracted_keypoints) {
	
	int r_T = 15; 
	int num_iters = 50;
	double lambda = 0.1;
	
	//Mat kpold = Mat::zeros(3, Si.num_candidates, CV_64FC1);
	Mat failed_candidates = Mat::zeros(1, Si.num_candidates, CV_64FC1);
	
	int nr_keep = 0;
	
	Mat Ii_1_gray, Ii_gray;
	cvtColor(Ii_1, Ii_1_gray, COLOR_BGR2GRAY);
	cvtColor(Ii, Ii_gray, COLOR_BGR2GRAY);
	
	cout << "Mistake 1" << endl;
	
	for (int r = 0; r < extracted_keypoints.rows; r++) {
		for (int c = 0; c < extracted_keypoints.cols; c++) {
			cout << extracted_keypoints.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	cout << "Si.num_candidates = " << Si.num_candidates << endl;
	
	// Track candidate keypoints 
	Mat x_T = Mat::zeros(1, 2, CV_64FC1);
	Mat delta_keypoint;
	for (int i = 0; i < Si.num_candidates; i++) {
		// Makes sure not to track already extracted keypoints
		if (extracted_keypoints.at<double>(0,i) == 0) {
			//cout << "Mistake 2" << endl;
			x_T.at<double>(0,0) = Si.Ci.at<double>(1,i); // x-coordinate in image
			x_T.at<double>(0,1) = Si.Ci.at<double>(0,i); // y-coordinate in image
			cout << "Keypoint x_T = (" << x_T.at<double>(0,0) << "," << x_T.at<double>(0,1) << ") ";
			
			delta_keypoint = KLT::trackKLTrobustly(Ii_1_gray, Ii_gray, x_T, r_T, num_iters, lambda);
			
			if (delta_keypoint.at<double>(2,0) == 1) {
				nr_keep++;
				Si.Ci.at<double>(1,i) = delta_keypoint.at<double>(0,0) + Si.Ci.at<double>(1,i); // Check if this is right 
				Si.Ci.at<double>(0,i) = delta_keypoint.at<double>(1,0) + Si.Ci.at<double>(0,i);
				
			}
			cout << "Match = " << delta_keypoint.at<double>(2,0) << " at point = (" << Si.Ci.at<double>(0,i) << "," << Si.Ci.at<double>(1,i) << ")" << endl;
			
			if (delta_keypoint.at<double>(2,0) == 0) {
				failed_candidates.at<double>(0,i) = 1;
			}
		}
		
	}

	cout << "Mistake 3" << endl;
	// Delete un-tracked candidate keypoints 
	// We just overwrite those points 
	failed_candidates = failed_candidates + extracted_keypoints;
	
	cout << "failed_candidates" << endl;
	for (int r = 0; r < failed_candidates.rows; r++) {
		for (int c = 0; c < failed_candidates.cols; c++) {
			cout << failed_candidates.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	// Non maximum suppression of candidate keypoints 
	Mat tempMat = Mat::zeros(2, countNonZero(failed_candidates), CV_64FC1); 
	cout << "Dimensions of tempMat = (" << tempMat.rows << "," << tempMat.cols << ")" << endl; 
	for (int i = 0; i < countNonZero(failed_candidates); i++) {
		if (failed_candidates.at<double>(0,i) == 1) {
			tempMat.at<double>(0,i) = Si.Ci.at<double>(0,i);
			tempMat.at<double>(1,i) = Si.Ci.at<double>(1,i);
		}
	}
	
	cout << "tempMat" << endl;
	for (int r = 0; r < tempMat.rows; r++) {
		for (int c = 0; c < tempMat.cols; c++) {
			cout << tempMat.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "Dimensions of Si.Pi = (" << Si.Pi.rows << "," << Si.Pi.cols << ")" << endl;
	Mat suprression;
	if (tempMat.cols == 0) {
		Si.Pi.copyTo(suprression);
	}
	else {
		hconcat(Si.Pi, tempMat, suprression);
	}	
	cout << "Done here " << endl;
	// Find new candidate keypoints 
	Mat t_C_W_vector = T_wc.reshape(0,T_wc.rows * T_wc.cols);
	int n = Si.num_candidates - nr_keep;
	cout << "Before Harris " << endl;
	Mat candidate_keypoints = Harris::corner(Ii, Ii_gray, n, suprression);
	cout << "After Harris" << endl;
	cout << "Dimensions of candidate_keypoints = (" << candidate_keypoints.rows << "," << candidate_keypoints.cols << ")" << endl;
	//candidate_keypoints = candidate_keypoints.t();
	vconcat(candidate_keypoints.row(1), candidate_keypoints.row(2), candidate_keypoints);
	
	int temp = 0;
	for (int i = 0; i < Si.num_candidates; i++) {
		if (failed_candidates.at<double>(0,i) > 0) {
			// Update current keypoint position
			Si.Ci.at<double>(1,i) = candidate_keypoints.at<double>(1,temp); // Check if this is right 
			Si.Ci.at<double>(0,i) = candidate_keypoints.at<double>(0,temp);
			
			// Update first coordinates of first observation of keypoint
			Si.Fi.at<double>(1,i) = candidate_keypoints.at<double>(1,temp);
			Si.Fi.at<double>(0,i) = candidate_keypoints.at<double>(0,temp);
			
			// Update camera pose at the first observation of keypoint
			for (int j = 0; j < t_C_W_vector.rows; j++) {
				Si.Ti.at<double>(j,i) = t_C_W_vector.at<double>(j,0);
			}
			
			
			temp++;
		}
	}
	
	cout << "End continuousCandidateKeypoints" << endl;
	return Si;
}

// Method for finding landmark from two keypoints and their corresponding positions
/* Input: 
 * K - 3x3 Intrinsic parameters for the camera
 * tau - 3x4 transformation matrix [R|t] for imagepoint 0
 * T_WC - 3x4 transformation matrix [R|t] for imagepoint 1 
 * keypoint0 - 2x1 matrix - the image coordinates as [u,v]
 * keypoint1 - 2x1 matrix - the image coordinates as [u,v]
 * Output: 
 * P - 4x1 Matrix which contains the landmark as [Xw, Yw, Zw, 1] in world coordinate frame
 * Needs to be scaled correctly (like mulitplied with a constant factor) in order to work.
 */
Mat findLandmark(Mat K, Mat tau, Mat T_WC, Mat keypoint0, Mat keypoint1) {
	Mat P;
	Mat Q;
	
	Mat imagepoint0 = Mat::ones(3, 1, CV_64FC1);
	imagepoint0.at<double>(0,0) = keypoint0.at<double>(0,0);
	imagepoint0.at<double>(1,0) = keypoint0.at<double>(1,0);
	Mat imagepoint1 = Mat::ones(3, 1, CV_64FC1);
	imagepoint1.at<double>(0,0) = keypoint1.at<double>(0,0);
	imagepoint1.at<double>(1,0) = keypoint1.at<double>(1,0);
	
	Mat M0 = K*tau;
	Mat M1 = K*T_WC;
	Mat v1 = M0.row(0) - imagepoint0.at<double>(0) * M0.row(2);
	Mat v2 = M0.row(1) - imagepoint0.at<double>(1) * M0.row(2);
	vconcat(v1, v2, Q);
	v1 = M1.row(0) - imagepoint1.at<double>(0) * M1.row(2);
	vconcat(v1, Q, Q);
	v2 = M1.row(1) - imagepoint1.at<double>(1) * M1.row(2);
	vconcat(v2, Q, Q);
	
	Mat S, U, VT;
	SVDecomp(Q, S, U, VT, SVD::FULL_UV);
	VT = VT.t();
	
	P = VT.col(3); 
	
	// For debug: Maybe you should not multiply with a factor -1
	P.at<double>(0,0) = -P.at<double>(0,0) / P.at<double>(3,0);
	P.at<double>(1,0) = -P.at<double>(1,0) / P.at<double>(3,0);
	P.at<double>(2,0) = -P.at<double>(2,0) / P.at<double>(3,0);
	P.at<double>(3,0) = P.at<double>(3,0) / P.at<double>(3,0);
	
	return P;
}


// Triangulate new landmarks 
tuple<state, Mat>  triangulateNewLandmarks(state Si, Mat K, Mat T_WC, double threshold_angle) {
	
	// Check if points are ready to be traingulated 
	Mat extracted_keypoints = Mat::zeros(1, Si.num_candidates, CV_64FC1);
	
	// Matrices to store valid extracted keypoints
	Mat newKeypoints = Mat::zeros(2, Si.num_candidates, CV_64FC1);
	Mat traingulated_landmark;
	Mat newLandmarks = Mat::zeros(Si.Xi.rows, Si.num_candidates, CV_64FC1);
	int temp = 0;
	
	Mat keypoint_last_occur = Mat::ones(3, 1, CV_64FC1);
	Mat keypoint_newest_occcur = Mat::ones(3, 1, CV_64FC1);
	Mat tau;
	Mat a, b; // a = previous_vector, b = current_vector;
	double length_prev_vector, length_current_vector;
	
	// Beregn vinklen mellem vektorerne current viewpoint og den første observation af keypointet
	double fraction, alpha;
	for (int i = 0; i < Si.num_candidates; i++) {
		// First occurrence of keypoint
		keypoint_last_occur.at<double>(0,0) = Si.Fi.at<double>(0,i);
		keypoint_last_occur.at<double>(1,0) = Si.Fi.at<double>(1,i);
		
		cout << "First occurence of keypoint =(" << keypoint_last_occur.at<double>(0,0) << "," << keypoint_last_occur.at<double>(1,0) << ")" << endl;
	
		// Newest occurrence of keypoint
		keypoint_newest_occcur.at<double>(0,0) = Si.Ci.at<double>(0,i);
		keypoint_newest_occcur.at<double>(1,0) = Si.Ci.at<double>(1,i);
		
		cout << "Newest occurence of keypoint =(" << keypoint_newest_occcur.at<double>(0,0) << "," << keypoint_newest_occcur.at<double>(1,0) << ")" << endl;
		
		// Finding the angle using bearing vectors 
		Mat bearing1 = K.inv() * keypoint_last_occur;
		Mat bearing2 = K.inv() * keypoint_newest_occcur;
		
		cout << "Bearing vector 1 " << bearing1 << endl;
		cout << "Bearing vector 2 " << bearing2 << endl;
		
		// Finding length of vectors 
		double length_first_vector = sqrt(pow(bearing1.at<double>(0,0),2.0) + pow(bearing1.at<double>(1,0),2.0) + pow(bearing1.at<double>(2,0),2.0));
		double length_current_vector = sqrt(pow(bearing2.at<double>(0,0),2.0) + pow(bearing2.at<double>(1,0),2.0) + pow(bearing2.at<double>(2,0),2.0));
		
		// Determine the angle
		// The angle is in radians 
		// CHANGES NEEDED HERE
		double v = bearing1.at<double>(0,0)*bearing2.at<double>(0,0)+bearing1.at<double>(1,0)*bearing2.at<double>(1,0)+bearing1.at<double>(2,0)*bearing2.at<double>(2,0); // This value should be changed
		alpha = acos((v)/(length_prev_vector * length_current_vector)) * 360/(2*M_PI);
		
		cout << "alpha = " << alpha << endl;
		
		if (alpha > threshold_angle) {
			extracted_keypoints.at<double>(0,i) = 1;
			
			// Update new keypoints 
			newKeypoints.at<double>(0,temp) = Si.Ci.at<double>(0,i);
			newKeypoints.at<double>(1,temp) = Si.Ci.at<double>(1,i);
			
			traingulated_landmark = findLandmark(K, tau, T_WC, keypoint_last_occur, keypoint_newest_occcur); // Check if tau should be changed to matrix
			
			newLandmarks.at<double>(0,temp) = traingulated_landmark.at<double>(0,0);
			newLandmarks.at<double>(1,temp) = traingulated_landmark.at<double>(1,0);
			newLandmarks.at<double>(2,temp) = traingulated_landmark.at<double>(2,0);
			newLandmarks.at<double>(3,temp) = traingulated_landmark.at<double>(3,0);
			
			temp++;
		}
	}
	
	Mat temp_newKeypoints = Mat::zeros(2, temp, CV_64FC1);
	Mat temp_newLandmarks = Mat::zeros(4, temp, CV_64FC1);
	
	for (int j = 0; j < temp; j++) {
		temp_newKeypoints.at<double>(0,j) = newKeypoints.at<double>(0,j);
		temp_newKeypoints.at<double>(1,j) = newKeypoints.at<double>(1,j);
		
		temp_newLandmarks.at<double>(0,j) = newLandmarks.at<double>(0,j);
		temp_newLandmarks.at<double>(1,j) = newLandmarks.at<double>(1,j);
		temp_newLandmarks.at<double>(2,j) = newLandmarks.at<double>(2,j);
		temp_newLandmarks.at<double>(3,j) = newLandmarks.at<double>(3,j);
		
	}
	
	// 
	
	// Append new keypoints
	hconcat(Si.Pi, temp_newKeypoints, Si.Pi);
	
	// Append new 3D landmarks 
	hconcat(Si.Xi, temp_newLandmarks, Si.Xi);
	
	return make_tuple(Si, extracted_keypoints);
}

/*
void *PrintHello(void *threadid) {
	long tid; 
	tid = (long)threadid;
	cout << "Hello World! Thread ID, " << tid << endl;
	pthread_exit(NULL);
}
*/



