#include <iostream>
#include "mainCamera.hpp"
#include <limits> 
#include <assert.h> 

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
		
		int keypoints_limit = 200; // Change to 200
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

Mat estimateEssentialMatrix(Mat fundamental_matrix, Mat K) {
	
	Mat essential_matrix = K.t() * fundamental_matrix * K;
	
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
	
	Mat R1 = U * W * VT.t();
	Mat R2 = U * W.t() * VT.t();
	
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
			
			// Projekct in both cameras
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
/*
Mat KLT::trackKLTrobustly(Mat I1, Mat I2, Mat keypoints, int r_t, int num_iters, double lambda) {
	
}
*/




