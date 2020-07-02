#include <iostream>
#include "mainCamera.hpp"
#include <limits> 
#include <assert.h> 
//#include <complex.h>

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
Matrix Harris::corner(Mat src, Mat src_gray) {
	
	// Define variables
	const char* corners_window = "Corners detected";
	
	// Maybe use minMaxLoc(img, &minVal, &maxVal); for at finde max og minimum, som gemmes i værdierne minVal og maxVal. 	
	
	// Define variables related to Harris corner
	int blockSize = 9; 
	int apertureSize = 3;
	double k = 0.08;		// Magic parameter 
	int thres = 200;	
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
	bool display = false;
	if (display == false) {
		int nr_corners = 0;
		
		int keypoints_limit = 150; // Change to 200
		Matrix Corners(keypoints_limit,3); // Column1: Corner responses, Column2: Pixel i, Column3: Pixel j
		
		int CornerResponse = 0;
		int maxCornerResponse = 0;
		Corners(0,0) = 0;
		Corners(1,0) = 0;
		
		Matrix keypoints(keypoints_limit,3);
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
			keypoints(count,0) = max;
			keypoints(count,1) = y;
			keypoints(count,2) = x;
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
	}
	cout << "End of Harris " << endl;	
	Matrix emptyArray(1,3);	
	return emptyArray;
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
Matrix SIFT::FindDescriptors(Mat src_gray, Matrix keypoints) {
	
	// Simplification of SIFT
	//cout << "Error here" << endl;
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
	
	//cout << "Error 2 here" << endl;
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
	Mat I = Mat::zeros(I_R.rows, I_R.cols, CV_64FC1);
	
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
					
					I.at<double>(y,x) = I_R.at<double>(floor(warped.at<double>(0,1)),floor(warped.at<double>(0,0)));
				}
			}
		}
	} 
	return I;
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
Mat getWarpedPatch(Mat I, Mat W, Mat x_T, int r_T) {
	
	// Initialize patch
	Mat patch = Mat::zeros(2*r_T + 1, 2*r_T + 1, CV_64FC1);
	
	// Get dimensions of image 
	int max_coords_rows = I.rows;
	int max_coords_cols = I.cols;
	
	// Find the transpose
	Mat WT = W.t();
	
	Mat pre_warp = Mat::zeros(1, 3, CV_64FC1);
	for (int x = -r_T; x <= r_T; x++) {
		for (int y = -r_T; y <= r_T; y++) {
			pre_warp.at<double>(0,0) = x;
			pre_warp.at<double>(0,1) = y;
			pre_warp.at<double>(0,2) = 1;
			
			Mat warped = x_T + pre_warp * WT;
			
			if (warped.at<double>(0,0) < max_coords_cols && warped.at<double>(0,1) < max_coords_rows) {
				if (warped.at<double>(0,0) > 0 && warped.at<double>(0,1) > 0) { // It should be greater than 0 (C++ 0-indexing)

					Mat floors = Mat::zeros(warped.rows, warped.cols, CV_64FC1);
					for (int r = 0; r < floors.rows; r++) {
						for (int c = 0; c < floors.cols; c++) {
							floors.at<double>(r, c) = floor(warped.at<double>(r, c));
						}
					}
					
					Mat weights = warped - floors;
					
					double a = weights.at<double>(0,0);
					double b = weights.at<double>(0,1);
					
					double intensity = (1-b) * ((1-a) * I.at<double>(floors.at<double>(0,1)-1,floors.at<double>(0,0)-1) + a * I.at<double>(floors.at<double>(0,1)-1,floors.at<double>(0,0)));
					
					intensity = intensity + b * ((1-a) * I.at<double>(floors.at<double>(0,1),floors.at<double>(0,0)-1) + a * I.at<double>(floors.at<double>(0,1),floors.at<double>(0,0)));
					
					patch.at<double>(y + r_T, x + r_T) = intensity;
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
Mat trackKLT(Mat I_R, Mat I, Mat x_T, int r_T, int num_iters) {
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
		Mat big_IWT = getWarpedPatch(I, W, x_T, r_T + 1); // We are here 
		
		
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
		
	}
	return W;
}

Mat KLT::trackKLTrobustly(Mat I_R, Mat I, Mat keypoint, int r_T, int num_iters, double lambda) {
	
	Mat W = trackKLT(I_R, I, keypoint, r_T, num_iters);
	
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
	
	Mat Winv = trackKLT(I, I_R, reverse_keypoint, r_T, num_iters);
	
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

Mat solveQuartic(Mat factors) {
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
	double R = -Q/2 + sqrt(pow(Q,2.0)/4 + pow(P,3.0)/27);
	double U = pow(R,(1.0/3.0));
	
	cout << "Values 4 = " << P << "," << Q <<  "," << R << "," << U << endl;
	
	double y;
	if (U == 0) {
		y = -5*alpha/6 - pow(Q,(1/3));
	} 
	else {
		y = -5*alpha/6 - P/(3*U) + U;
	}
	cout << "Value y = " << y << endl;
	
	double w = sqrt(alpha+2*y);
	
	cout << "w = " << w << endl;
	
	roots.at<double>(0,0) = real(-B/(4.0*A) + 0.5*(w + sqrt(-(3.0*alpha+2.0*y+2.0*beta/w))));
	roots.at<double>(0,1) = -B/(4.0*A) + 0.5*(w - sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
	roots.at<double>(0,2) = -B/(4.0*A) + 0.5*(-w + sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
	roots.at<double>(0,3) = -B/(4.0*A) + 0.5*(-w - sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
	
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
	Mat P2 = Mat::zeros(3, 1, CV_64FC1); 
	Mat P3 = Mat::zeros(3, 1, CV_64FC1);
	// Assign values to P1, P2 and P3 
	for (int i = 0; i < 3; i++) {
		P1.at<double>(i,0) = worldPoints.at<double>(i,0);
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
		
		f3 = T * f3;
		
		// Reassign values to P1, P2 and P3 
		for (int i = 0; i < 3; i++) {
			P1.at<double>(i,0) = worldPoints.at<double>(i,1);
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
	
	return poses;
}



tuple<Mat, Mat> Localize::ransacLocalization(Mat keypoints_i, Mat corresponding_landmarks, Mat K) {
	// Transformation matrix 
	Mat transformation_matrix = Mat::zeros(3, 4, CV_64FC1);
	
	// Method parameter
	bool adaptive_ransac = false;
	
	// Other parameters 
	double num_iterations;
	int pixel_tolerance = 10; 
	int k = 3;
	
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
		matched_query_keypoints.at<double>(0,i) = keypoints_i.at<double>(1,i);
		matched_query_keypoints.at<double>(1,i) = keypoints_i.at<double>(0,i);
	}
	Mat max_num_inliers_history = Mat::zeros(50, 1, CV_64FC1); // Should probably be changed
	Mat num_iteration_history = Mat::zeros(50, 1, CV_64FC1); // Should probably be changed to 
	int max_num_inliers = 0;
	
	// RANSAC
	int i = 0;
	
	// Needed variables 
	Mat landmark_sample = Mat::zeros(3, k, CV_64FC1);
	Mat keypoint_sample = Mat::ones(3, k, CV_64FC1); // To avoid having to set last row to 1's afterwards
	
	while ( num_iterations > i ) {
		int random_nums[corresponding_landmarks.cols];
		for (int mm = 0; mm < corresponding_landmarks.cols; mm++) {
			random_nums[mm] = mm;
		}
		random_shuffle(random_nums, random_nums + corresponding_landmarks.cols);
		for (int mm = 0; mm < k;  mm++) {
			// Landmark sample 
			landmark_sample.at<double>(0,mm) = corresponding_landmarks.at<double>(0, random_nums[mm]);
			landmark_sample.at<double>(1,mm) = corresponding_landmarks.at<double>(1, random_nums[mm]);
			landmark_sample.at<double>(2,mm) = corresponding_landmarks.at<double>(2, random_nums[mm]);
			
			// Keypoint sample 
			keypoint_sample.at<double>(0,mm) = matched_query_keypoints.at<double>(0, random_nums[mm]);
			keypoint_sample.at<double>(1,mm) = matched_query_keypoints.at<double>(1, random_nums[mm]);
		}
		
		Mat normalized_bearings = K.inv() * keypoint_sample;
		
		for (int ii = 0; ii < 3; ii++) {
			double vector_norm = sqrt(pow(normalized_bearings.at<double>(0,ii),2.0) + pow(normalized_bearings.at<double>(1,ii),2.0) + pow(normalized_bearings.at<double>(2,ii),2.0));
			normalized_bearings.at<double>(0,ii) = normalized_bearings.at<double>(0,ii)/vector_norm;
			normalized_bearings.at<double>(1,ii) = normalized_bearings.at<double>(1,ii)/vector_norm;
			normalized_bearings.at<double>(2,ii) = normalized_bearings.at<double>(2,ii)/vector_norm;
		
		}
		
		Mat poses = p3p(landmark_sample, normalized_bearings);
		
		
	}	
	
	
	
	
	
	
	
	return make_tuple(transformation_matrix, best_inlier_mask);
}


