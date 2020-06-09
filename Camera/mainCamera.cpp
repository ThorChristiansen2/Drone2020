#include <iostream>
#include "mainCamera.hpp"
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
	cout << "Inside Match descriptors " << endl;
	int n1 = descriptor1.dim1();
	int n2 = descriptor2.dim1();
	Matrix KeypointIndex1(n1,1);
	for (int i = 0; i < n1; i++) {
		KeypointIndex1(i,0) = i;
	}
	Matrix KeypointIndex2(n2,1);
	for (int i = 0; i < n2; i++) {
		KeypointIndex2(i,0) = i;
	}
	int k, j, choice;
	if (n1 < n2) {
		k = n1;
		j = n2;
		//Matrix matches(2,n1);
	}
	else {
		k = n2;
		j = n1;
		//Matrix matches(1,n2); // Maybe make a 2nd dimension for SSD
	}
	cout << "n1, n2, k, j = " << n1 << "," << n2 << "," << k << "," << j << "," << endl;
	waitKey(0);
	Matrix matches(1,k);
	cout << "the lesser term determined " << endl;
	for (int i = 0; i < k; i++) {
		double SSD = 0;
		double min = 1000;
		int match = 0;
		if (n1 < n2) {
			j = descriptor2.dim1();
		}
		else {
			j = descriptor1.dim1();
		}
		for (int l = 0; l < j; l++) { 	// Go through all the descriptors in the other matrix.
			SSD = 0;
			for (int m = 0; m < 128; m++) {
				SSD = SSD + pow((descriptor1(i,m)-descriptor2(l,m)),2);
			}
			cout << "SSD = " << SSD << endl;
			if (SSD < min) {
				match = l;
				min = SSD;
			}
		}
		j--;
		if (n1 < n2) {
		}
		else {
			
		}
		
		matches(0,i) = match;
		if (n1 < n2) {
			descriptor2[match] *= 7;
		}
		else {
			descriptor1[match] *= 7;
		}
	}
	return matches;
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
	//cout << "Rectangle : (" << x1 << "," << y1 << "," << y2-y1 << "," << x2-x1 << ")" << endl;
	//Rect region(y1, x1, x2-x1, y2-y1);
	//Rect region(x1,y1,y2-y1,x2-x1);
	Rect region(x1,y1,y2-y1,x2-x1);
	ROI = img(region);
	
	/*
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
	*/
	return ROI;
}

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
			//cout << "G-value is: " << g.at<double>(y_coor,x_coor);
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
	
	// Maybe use minMaxLoc(img, &minVal, &maxVal); for at finde max og minimum, som gemmes i værdierne minVal og maxVal. 	
	
	// Define variables related to Harris corner
	int blockSize = 2; 
	int apertureSize = 3;
	double k = 0.08;		// Magic parameter 
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
		
		int keypoints_limit = 500; // Change to 200
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
		//printfunction(Corners,nr_corners);
		
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

Matrix circularShift(Matrix histogram) {
	//cout << "Inside circularshift" << endl;
	float max = 0;
	int index = 0;
	for (int ii = 0; ii < histogram.dim2(); ii++) {
		double value = histogram(0,ii);
		//cout << "Value given" << endl;
		if (value > max) {
			max = histogram(0,ii);
			index = ii;
			//cout << "Index ii = " << index << endl;;
		}
	}
	//cout << "Max done" << endl;
	Matrix temp(1,index);
	//cout << "Temp initialized" << endl;
	//cout << "Temp dimension = " << temp.dim1() << "," << temp.dim2() << endl;
	// This could maybe be done faster 
	for (int ii = 0; ii < index; ii++) {
		double v = histogram(0,ii);
		//cout << "Value given" << endl;
		temp(0,ii) = v;
	}
	//cout << "Done with initial circulation" << endl;
	for (int ii = index; ii < histogram.dim2(); ii++) {
		histogram(0,ii-index) = histogram(0,ii);
	}
	//cout << "Done with first assignmnet" << endl;
	for (int ii = histogram.dim2()-index; ii < histogram.dim2(); ii++) {
		//cout << "ii = " << ii << endl;
		//cout << "histogram.dim2()-index = " << histogram.dim2()-index << endl;
		double v = temp(0,ii-(histogram.dim2()-index));
		//cout << "v = " << v << endl;
		histogram(0,ii) = v;
	}
	//cout << "Done with second assignment" << endl;
	//temp = histogram[0].slice(0,index); // Automatically index -1
	//histogram[0].slice(0,index) = histogram[0].slice(index);
	//histogram[0].slice(index) = temp.slice();
	//cout << "Print circulated histogram" << endl;
	for (int mm = 0; mm < histogram.dim2(); mm++) {
		//cout << "m = " << mm << " and v = ";
		double v = histogram(0,mm);
		//cout << v << endl;
	}
	//waitKey(0);
	//cout << "Done with circular shift" << endl;
	return histogram;
}

// Find SIFT Desriptors 
Matrix SIFT::FindDescriptors(Mat src_gray, Matrix keypoints) {
	
	// Simplification of SIFT
	
	// Maybe the image should be smoothed first with a Gaussian Kernel
	
	int n = keypoints.dim1();
	
	// Initialize matrix containing keypoints descriptors
	Matrix Descriptors(n,128);
	
	//cout << "Dimensions of image (rows,cols) = (" << src.rows << "," << src.cols << ")" << endl;
	//waitKey(0);
	
	
	
	// Find Image gradients
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int ksize = 1;
	int scale = 1; 
	int delta = 0; 
	int ddepth = CV_16S;
	
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	
	// Converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	
	/*
	cout << "Type of src" << endl;
	MatType(src_gray);
	cout << "Draw image" << endl;
	for (int k = 10; k<= 20; k++) {
		for (int j = 10; j<= 20; j++) {
			cout << (int) src_gray.at<uchar>(k,j) << ", ";
		}
		cout << "" << endl;
	}
	waitKey(0);
	
	cout << "Rows of Image gradients x = " << grad_x.rows << endl;
	cout << "Columns of Image gradients x = " << grad_x.cols << endl;
	cout << "Type of image grad " << endl;
	MatType(grad_x);
	cout << "Draw Image gradients X" << endl;
	for (int k = 10; k<= 20; k++) {
		for (int j = 10; j<= 20; j++) {
			cout << (int) grad_x.at<short>(k,j) << ", ";
		}
		cout << "" << endl;
	}
	waitKey(0);
	
	cout << "Rows of Image gradients y = " << grad_y.rows << endl;
	cout << "Columns of Image gradients y = " << grad_y.cols << endl;
	cout << "Type of image grad " << endl;
	MatType(grad_y);
	cout << "Draw Image gradients Y" << endl;
	for (int k = 10; k<= 20; k++) {
		for (int j = 10; j<= 20; j++) {
			cout << (int) grad_y.at<short>(k,j) << ", ";
		}
		cout << "" << endl;
	}
	waitKey(0);
	
	cout << "Rows of Image gradients x = " << abs_grad_x.rows << endl;
	cout << "Columns of Image gradients x = " << abs_grad_x.cols << endl;
	cout << "Type of abs image grad " << endl;
	MatType(abs_grad_x);
	cout << "Draw Image abs grad X" << endl;
	for (int k = 10; k<= 20; k++) {
		for (int j = 10; j<= 20; j++) {
			cout << (int) abs_grad_x.at<uchar>(k,j) << ", ";
		}
		cout << "" << endl;
	}
	waitKey(0);
	*/
	//cout << "Calculating gauss window" << endl;
	int filter_size = 16;
	float sigma = 1.5*16;
	Mat GaussWindow;
	GaussWindow = gaussWindow(filter_size, sigma);
	
	//cout << "Inside SIFT" << endl;
	for (int i = 0; i < n; i++) {
		int y = keypoints(i,1);
		int x = keypoints(i,2); 
		
		
		//circle (src, Point(x,y), 5, Scalar(7), 2,8,0);
		//cout << "Keypoint at (y = " << y << ",x = " << x << ")" << endl;
		//cout << "Intensity at keypoint :" << (int) src.at<uchar>(y,x) << endl;
		//cout << "Intensity at pixel left of keypoint :" << (int) src.at<uchar>(y,x-1) << endl;
		//cout << "Intensity at pixel right of keypoint :" << (int) src.at<uchar>(y,x+1) << endl;
		
		/*
		cout << "Draw image" << endl;
		for (int k = y-7; k<= y+8; k++) {
				for (int j = x-7; j<= x+8; j++) {
					cout << (int) src_gray.at<uchar>(k,j) << ", ";
				}
				cout << "" << endl;
		}
		*/
		
		//waitKey(0);
		
		// Patch of size 16,16
		Mat Patch_Ix, Patch_Iy;
		Patch_Ix = selectRegionOfInterest(grad_x, y-7, x-7, y+8, x+8);
		
		Patch_Iy = selectRegionOfInterest(grad_y, y-7, x-7, y+8, x+8);
		//MatType(Patch_Ix);
		
		/*
		cout << "Print Patch_Ix" << endl;
		for (int mm = 0; mm < Patch_Ix.rows; mm++) {
			for (int nn = 0; nn < Patch_Ix.cols; nn++) {
				cout << (double) Patch_Ix.at<short>(mm,nn) << ", ";
			}
			cout << "" << endl;
		}
		
		cout << "Print Patch_Iy" << endl;
		for (int mm = 0; mm < Patch_Iy.rows; mm++) {
			for (int nn = 0; nn < Patch_Iy.cols; nn++) {
				cout << (double) Patch_Iy.at<short>(mm,nn) << ", ";
			}
			cout << "" << endl;
		}
		waitKey(0);
		*/
		
		// It has to be of dimensions 16x16
		// This is the scaled gradients 
		Mat Gradients = Mat::zeros(Size(16,16),CV_64FC1);
		//cout << "Type of Gradients" << endl;
		//MatType(Gradients);
		Mat Orientations = Mat::zeros(Size(16,16),CV_64FC1);
		//cout << "Type of Orientations" << endl;
		//MatType(Orientations);
		for (int coor_y = 0; coor_y < 16; coor_y++) {
			for (int coor_x = 0; coor_x < 16; coor_x++) {
				float norm = sqrt( pow(Patch_Ix.at<short>(coor_y,coor_x),2) + pow(Patch_Iy.at<short>(coor_y,coor_x),2));
				Gradients.at<double>(coor_y,coor_x) = norm*GaussWindow.at<double>(coor_y,coor_x);
				Orientations.at<double>(coor_y,coor_x) = atan2(Patch_Iy.at<short>(coor_y,coor_x),Patch_Ix.at<short>(coor_y,coor_x));
			}
		}
		
		/*
		cout << "Print scaled Gradients" << endl;
		for (int mm = 0; mm < Gradients.rows; mm++) {
			for (int nn = 0; nn < Gradients.cols; nn++) {
				cout << Gradients.at<double>(mm,nn) << ", ";
			}
			cout << "" << endl;
		}
		
		cout << "Print Orientations" << endl;
		for (int mm = 0; mm < Orientations.rows; mm++) {
			for (int nn = 0; nn < Orientations.cols; nn++) {
				cout << Orientations.at<double>(mm,nn) << ", ";
			}
			cout << "" << endl;
		}
		*/
		//waitKey(0);
		
		
		// Maybe you should rotate the patch, so it coincides with the orientation in the strongest direction
		
		
		// Divde the 16x16 patch into subpatches of 4x4 
		Matrix descrip(1,128);
		Mat subPatchGradients, subPatchOrientations;
		int nindex = 0;
		for (int k1 = 0; k1 <= 12; k1 = k1+4) {
			for (int k2 = 0; k2 <= 12; k2 = k2 + 4) {
				//cout << "k1 = " << k1 << endl;
				//cout << "k2 = " << k2 << endl; 
				subPatchGradients = selectRegionOfInterest(Gradients, k1, k2, k1+4, k2+4);
				//cout << "Subpatch extracted " << endl;
				subPatchOrientations = selectRegionOfInterest(Orientations, k1, k2, k1+4, k2+4);
				//cout << "Orientations extracted " << endl;
				Matrix Histogram(1,8);
				//cout << "Histogram initiated " << endl;
				//MatType(subPatchGradients);
				//MatType(subPatchOrientations);
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
				for (int mm = 0; mm < Histogram.dim2(); mm++) {
					//cout << "m = " << mm << " and v = ";
					double v = Histogram(0,mm);
					//cout << v << endl;
				}
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
		//waitKey(0);
		
		/*
		cout << "Print of whole descriptor " << endl;
		for (int ii = 0; ii < 128; ii++) {
			cout << Descriptors(i,ii) << ", ";
		}
		*/
		//waitKey(0);
		//cout << "" << endl;
		// Normalizing the vector 
		double SumOfSquares = 0;
		for (int ii = 0; ii < Descriptors.dim2(); ii++) {
			SumOfSquares = SumOfSquares + Descriptors(i,ii)*Descriptors(i,ii);
		}
		//cout << "Sum of Squares: " << SumOfSquares << endl;
		/*
		for (int ii = 0; ii < Descriptors.dim2(); ii++) {
			Descriptors(i,ii) = Descriptors(i,ii)/sqrt(SumOfSquares);
			cout << Descriptors(i,ii) << ", ";
		}
		cout << "" << endl;
		*/
		
		// Normalize the vector 
		
		
		/*
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
		*/
		//Rect region(x-7,y-7,15,15);
		//cout << "Draw square " << endl;
		//rectangle(src_gray, region, Scalar(255), 1, 8, 0);
		//imshow ("image with square", src_gray);
		//waitKey(0);
	
	// Scale the norms of the gradients by multiplying a the graidents with a gaussian
	// centered in the keypoint and with Sigma_w = 1.5*16. 
	
	
	
	}
	cout << "SIFT Done " << endl;
	return Descriptors;
}



