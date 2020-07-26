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
using namespace std::chrono;
//using namespace Numeric_lib;

void *functionKLT(void *threadarg) {
   struct thread_data *my_data;
   my_data = (struct thread_data *) threadarg;
   Mat x_T = Mat::zeros(1, 2, CV_64FC1);
   
   Mat delta_keypoint;
   for (int i = 0; i < my_data->thread_mat.cols; i++) {
	   x_T.at<double>(0,0) = my_data->thread_mat.at<double>(1,i);
	   x_T.at<double>(0,1) = my_data->thread_mat.at<double>(0,i);
	   delta_keypoint = KLT::trackKLTrobustly(my_data->Ii_1_gray, my_data->Ii_gray, x_T, my_data->dwdx, KLT_r_T, KLT_num_iters, KLT_lambda);
	   double a = delta_keypoint.at<double>(0,0) + my_data->thread_mat.at<double>(1,i);
	   double b = delta_keypoint.at<double>(1,0) + my_data->thread_mat.at<double>(0,i);
	   //my_data->thread_mat.at<double>(1,i) = b;
	   //my_data->thread_mat.at<double>(0,i) = a;
	   if (a > 0 && b > 0) {
		   my_data->thread_mat.at<double>(1,i) = b;
		   my_data->thread_mat.at<double>(0,i) = a;
	   }
	   else if (a > 0) {
		   my_data->thread_mat.at<double>(1,i) = my_data->thread_mat.at<double>(0,i);
		   my_data->thread_mat.at<double>(0,i) = a; // y-coordinate in image
	   }
	   else if (b > 0) {
		   my_data->thread_mat.at<double>(0,i) = my_data->thread_mat.at<double>(1,i);
		   my_data->thread_mat.at<double>(1,i) = b; // To avoid negative coordinates // x-coordinate in image
	   }
	   
	   /*
	   if (b > 0) {
		   my_data->thread_mat.at<double>(1,i) = b; // To avoid negative coordinates // x-coordinate in image
		   my_data->thread_mat.at<double>(0,i) = my_data->thread_mat.at<double>(1,i);
	   }
	   if (a > 0) {
		   my_data->thread_mat.at<double>(0,i) = a; // y-coordinate in image
		   my_data->thread_mat.at<double>(1,i) = my_data->thread_mat.at<double>(0,i);
	   } 
	   */
	   my_data->keep_point = delta_keypoint.at<double>(2,0);
   }
   pthread_exit(NULL);
}

/* Function used to parallelize code in Harris::Corner detector
 * 
 */
void *functionHarris(void *threadarg) {
   struct harris_data *my_harris;
   my_harris = (struct harris_data *) threadarg;
   
   int nr_rows = my_harris->thread_dst.rows;
   int nr_cols = my_harris->thread_dst.cols;
   int nr_iter = my_harris->num_keypoints;
   int i, r, c, valid_keypoints;
   valid_keypoints = 0;
   int NMBOX = my_harris->thread_non_max_suppres;
   
   for (i = 0; i < nr_iter; i++) {
	   int max = 0;
	   int x = 0; 
	   int y = 0;
	   for (r = 0; r < nr_rows; r++) {
		   for (c = 0; c < nr_rows; c++) {
			   if ((double) my_harris->thread_dst.at<float>(r,c) > max) {
					max = (double) my_harris->thread_dst.at<float>(r,c) ;
					y = r;
					x = c;
						
				}
		   }
	   }
	   for (r = -NMBOX; r <  NMBOX; r++) {
		   for (c = -NMBOX; c < NMBOX; c++) {
			   my_harris->thread_dst.at<float>(y+r,x+c) = 0;
		   }
	   }
	   if (max > my_harris->threshold) {
		   my_harris->matrice.at<double>(0,valid_keypoints) = y + my_harris->left_corner_y;
		   my_harris->matrice.at<double>(1,valid_keypoints) = x + my_harris->left_corner_x;
		   valid_keypoints++;
	   }
   }
   valid_keypoints--;
   my_harris->valid_interest_points = valid_keypoints;
   
   
   pthread_exit(NULL);
}

void *functionMatch(void *threadarg) {
   struct thread_match *my_data;
   my_data = (struct thread_match *) threadarg;
   
   double min_d1 = std::numeric_limits<double>::infinity();
   double match_d1;
   double min_d2 = std::numeric_limits<double>::infinity();
   
   int nr_descriptors = my_data->n1.rows;
   
   int dimension = my_data->n1.cols;
   for (int i = 0; i < nr_descriptors; i++) {
	   double SSD = 0;
	   for (int k = 0; k < dimension; k++) {
		   SSD = SSD + pow(my_data->n1.at<double>(i,k) - my_data->n2.at<double>(0,k) ,2.0);
	   }
	   if (SSD < min_d1) {
		   // Update minimum distance to d2
		   min_d2 = min_d1;
		   
		   // Update minimum distance to d1 
		   min_d1 = SSD;
		   // Update id for match
		   match_d1 = i;
		
	   }
	   else if (SSD < min_d2) {
		   min_d2 = SSD;
	   }
   }

   
   // Make 0.8 a variable that can be tuned 
   if (min_d1/min_d2 < 0.9) {
	   my_data->is_inlier = 1;
	   my_data->lowest_distance = min_d1;
	   my_data->match_in_n1 = match_d1;
   }
   
   pthread_exit(NULL);
}





/* Function that is supposed to return a matrix of size 4xM with coordinates corresponding to
 * the upper right and lower left corners of the matrix
 * inputs:
 * int number_subimages;
 * int boundary --> Boundary that is not included in the original image
 * int height --> Height of one subpart of the image
 * int width --> Widht of one subpart of the image
 * int dim1 --> Height of the input image
 * int dim2 --> Widht of the input image
 * 
 * Outputs:
 * A matrix of 4xM with 
 */
 /*
Mat subImage(int number_subimages, int boundary, int height, int width, int dim1, int dim2) [
	Mat indicies = Mat::zeros(4, number_subimages, CV_64FC1);

	
	return indicies;
}
*/


// Match SIFT Descriptors
/* Objective: Function that matches keypoints in frame 2 to keypoints in frame 1 using the descriptors
 * for the keypoints in frame 2 and the descriptors for the keypoints in frame 1
 * Inputs:
 * descriptor1 - Matrix (Bjarnes matrix) of size n1x128
 * descriptor2 - Matrix (Bjarnes matrix) of size n2x128
 * Output:
 * Matrix of size 2 x min(n1,n2) depending on how many keypoints there are
 */
/*
Matrix SIFT::matchDescriptors(Mat descriptor1, Mat descriptor2) {

	int n1 = descriptor1.rows;	// Matrix containing descriptors for keypoints in image 0
	
	cout << "n1 is = " << n1 << endl;
	
	int n2 = descriptor2.rows;	// Matrix containing descriptors for keypoints in image 1
	
	Mat matches = Mat::zeros(2, n1, CV_64FC1);

	int NUM_THREADS = n2;
	pthread_t threads[NUM_THREADS];
	struct thread_match td[NUM_THREADS];
	int i, rc;
	for (i = 0; i < NUM_THREADS; i++) {
		td[i].descriptor_n2_id = i;
		td[i].n2 = descriptor2.row(i);
		td[i].n1 = descriptor1;
		td[i].is_inlier = 0;
		td[i].lowest_distance = 0;
		td[i].match_in_n1 = 0;
		
		
		rc = pthread_create(&threads[i], NULL, functionMatch, (void *)&td[i]);
	}
	void* ret = NULL;
	for (int k = 0; k < NUM_THREADS; k++) {
		pthread_join(threads[k], &ret);
		if (td[k].is_inlier == 1) {
			
			double distance = td[k].lowest_distance;
			double index = td[k].match_in_n1;

			if (matches.at<double>(0, index) == 0) {
				matches.at<double>(0, index) = distance;
				matches.at<double>(1, index) = index;
				
			}
			else if (distance < matches.at<double>(0,index)) {
				matches.at<double>(0, index) = distance;
				matches.at<double>(1, index) = index;
			}
		}
	}
	
	int nr_matches = 0;
	for (int h = 0; h < matches.cols; h++) {
		if (matches.at<double>(0,h) != 0) {
			nr_matches++;
		}
	}
	
	Matrix valid_matches(2, nr_matches);
	int index = 0;
	for (int q = 0; q < n1; q++) {
		if (matches.at<double>(0, q) != 0) {
			valid_matches(0,index) = q;
			valid_matches(1,index) = matches.at<double>(1, q);
			index++;
		}
	}
	
	
	return valid_matches;
}
*/

/*
// Match SIFT Descriptors - Old function that works
Matrix SIFT::matchDescriptors(Mat descriptor1, Mat descriptor2) {

	int n1 = descriptor1.rows;	// Matrix containing descriptors for keypoints in image 0
	int n2 = descriptor2.rows;	// Matrix containing descriptors for keypoints in image 1
	
	// Threshold on the euclidean distance between keypoints 
	double threshold = 200;
	
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
				SSD = SSD + pow((descriptor2.at<double>(i,k)-descriptor1.at<double>(j,k)),2.0);
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
	Matrix valid_matches(2,count);
	cout << "Count of valid matches  = " << count << endl;
	int index = 0;
	for (int i = 0; i < n2; i++) {
		if (matches(i,4) == 1) {
			valid_matches(0,index) = i;
			valid_matches(1,index) = matches(i,0);
			index++;
		}
	}
	
	
		
	return valid_matches;
}
*/

/*
// Match SIFT Descriptors - Old function
Matrix SIFT::matchDescriptors(Matrix descriptor1, Matrix descriptor2) {

	int n1 = descriptor1.dim1();	// Matrix containing descriptors for keypoints in image 0
	int n2 = descriptor2.dim1();	// Matrix containing descriptors for keypoints in image 1
	
	// Threshold on the euclidean distance between keypoints 
	double threshold = 200;
	
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
		
	return valid_matches;
}
*/

// New attempt to write code 
/*
 * 
 * 
 * 
 */
Mat SIFT::matchDescriptors(Mat descriptor1, Mat descriptor2) {

	int descriptor_length = descriptor2.cols;
	
	int n1 = descriptor1.rows;	// Matrix containing descriptors for keypoints in image 0
	int n2 = descriptor2.rows;	// Matrix containing descriptors for keypoints in image 1
	
	// To determine 
	Mat match_n2 = Mat::zeros(n2, 5, CV_64FC1);
	// 0 - Index of best match in n1 
	// 1 - SSD for best match in n1
	// 2 - Index of second best match in n1
	// 3 - SSD for second best match in n1 
	// 4 - Bool used to control whether the keypoint should be counted as an inlier
	
	Mat multiple_matches_n1 = Mat::zeros(n1, 2, CV_64FC1);
	
	int iter_n2, iter_n1, i;
	for (iter_n2 = 0; iter_n2 < n2; iter_n2++) {
		
		// Closest match
		int best_index_n1 = 0;
		double d1 = std::numeric_limits<double>::infinity();
		
		// Second closest match
		int second_best_index_n1 = 0;
		double d2 = std::numeric_limits<double>::infinity();
		
		match_n2.at<double>(iter_n2, 0) = best_index_n1;
		match_n2.at<double>(iter_n2, 1) = d1;
		match_n2.at<double>(iter_n2, 2) = second_best_index_n1;
		match_n2.at<double>(iter_n2, 3) = d2;
		
		
		for (iter_n1 = 0; iter_n1 < n1; iter_n1++) {
			double SSD = 0;
			
			// Find the SSD between descriptor in n2 and current descriptor in n1
			for (i = 0; i < descriptor_length; i++) {
				SSD = SSD + pow( descriptor2.at<double>(iter_n2,i) - descriptor1.at<double>(iter_n1,i) ,2.0);
			} 
			
			// If this SSD is less than the best match
			if ( SSD < match_n2.at<double>(iter_n2, 1) ) {
				
				// It should be also be one that is the closest to that specific desciptor in n1
				if ( multiple_matches_n1.at<double>(iter_n1,1) == 0 ) {
					
					// Update second best 
					match_n2.at<double>(iter_n2, 2) = match_n2.at<double>(iter_n2, 0);
					match_n2.at<double>(iter_n2, 3) = match_n2.at<double>(iter_n2, 1);
					
					// Update best
					match_n2.at<double>(iter_n2, 0) = iter_n1;
					match_n2.at<double>(iter_n2, 1) = SSD;
					
					// Update in multiple match
					multiple_matches_n1.at<double>(iter_n1,0) = iter_n2;
					multiple_matches_n1.at<double>(iter_n1,1) = SSD;
					
					match_n2.at<double>(iter_n2, 4) = 0;
					
				}
				else if ( SSD < multiple_matches_n1.at<double>(iter_n1,1) ) {
					// Save index of previous best 
					int temp_index = multiple_matches_n1.at<double>(iter_n1,0);
					
					// Update second best 
					match_n2.at<double>(iter_n2, 2) = match_n2.at<double>(iter_n2, 0);
					match_n2.at<double>(iter_n2, 3) = match_n2.at<double>(iter_n2, 1);
					
					// Update best
					match_n2.at<double>(iter_n2, 0) = iter_n1;
					match_n2.at<double>(iter_n2, 1) = SSD;
					
					// Update in multiple match
					multiple_matches_n1.at<double>(iter_n1,0) = iter_n2;
					multiple_matches_n1.at<double>(iter_n1,1) = SSD;
					
					match_n2.at<double>(iter_n2, 4) = 0;
					
					// Reset the previous best value
					if ( match_n2.at<double>(temp_index, 0) == iter_n1) {
						
						match_n2.at<double>(temp_index, 4) = 0;
					}
					
					
				}
				else {
					match_n2.at<double>(iter_n2, 4) = 2;
				}
				
			}
			else if ( SSD < match_n2.at<double>(iter_n2, 3) ) {
				
				// Update Second best
				match_n2.at<double>(iter_n2, 2) = iter_n1;
				match_n2.at<double>(iter_n2, 3) = SSD;
			}
				
		}
		
			
	}
	
	
	// Calculate distance ratio
	int nr_valid_matches = 0;
	int k;
	for (k = 0; k < n2; k++) {
		if (match_n2.at<double>(k, 4) != 2) {
			double distance1 = match_n2.at<double>(k, 1);
			double distance2 = match_n2.at<double>(k, 3);
			double ratio = distance1/distance2;
			if (ratio < 0.8) {
				match_n2.at<double>(k, 4) = 1;
				
				nr_valid_matches++;
			}
		}
		
	}
	
	// Make matrix with valid matches
	Mat valid_matches = Mat::zeros(2, nr_valid_matches, CV_64FC1);
	int temp_value = 0;
	for (int j = 0; j < n2; j++) {
		if (match_n2.at<double>(j, 4) == 1) {
			
			valid_matches.at<double>(0, temp_value) = match_n2.at<double>(j, 0);
			valid_matches.at<double>(1, temp_value) = j;
			
			temp_value++;
		}
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
	
	
	// Maybe use minMaxLoc(img, &minVal, &maxVal); for at finde max og minimum, som gemmes i værdierne minVal og maxVal. 	
	
	// Define variables related to Harris corner
	int blockSize = 9; 
	int apertureSize = 3;
	double k = 0.08;		// Magic parameter 
	//int thres = 200;	
	// Parameters before: blocksize = 2, aperturesize = 3, thres = 200, k = 0.04
	
	// Variables related to Non Maximum suppression 
	int NMSBox = 20;
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
		// This operation can maybe be parallelaized
		for (int c = 0; c < suppression.cols; c++) {
			if (row_nr < dst_norm.rows && col_nr < dst_norm.cols) {
				row_nr = suppression.at<double>(0,c);
				col_nr = suppression.at<double>(1,c);
				for (int i = -3; i < 4; i++) {
					for (int j = -3; j < 4; j++) {
						dst_norm.at<float>(row_nr + i, col_nr + j) = 0;
					}
				}
			}
		}
	}

	
	
	/*
	 * 
	 * NB: JUSTER FOR COORDINATES, SÅ DER SKLA LÆGGES Y OG X TIL FOR BILLEDET
	 * 
	 */
	
	//Define number of keypoints
	// possible mistake with dividing the keypoints into cols.
	Mat keypoints = Mat::zeros(2, keypoints_limit, CV_64FC1);
	
	// Create indices matrix
	int NUM_THREADS = Harris_threads;
	Mat indicies = Mat::zeros(2, Harris_threads, CV_64FC1);
	int index = 0;
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 6; c++) {
			indicies.at<double>(0,index) = r*188 + boundaries-1;			// y1
			indicies.at<double>(1,index) = c*210 + boundaries-1;			// x1
			index++;
		}
	}
	
	
	pthread_t threads[NUM_THREADS];
	struct harris_data td[NUM_THREADS];
	int i, rc, q;
	q = keypoints_limit / NUM_THREADS; // Potential mistake, when it does not become a possible division
	for (i = 0; i < NUM_THREADS; i++) {
		td[i].num_keypoints  = q;
		
		td[i].matrice = keypoints.colRange(i*q,(i+1)*q);
		td[i].threshold = 220;
		td[i].left_corner_y = indicies.at<double>(0,i);
		td[i].left_corner_x = indicies.at<double>(1,i);

		td[i].thread_dst = dst_norm.colRange(indicies.at<double>(1,i),indicies.at<double>(1,i)+210).rowRange(indicies.at<double>(0,i),indicies.at<double>(0,i)+188);

		td[i].thread_non_max_suppres = NMSBox;
		
		
		rc = pthread_create(&threads[i], NULL, functionHarris, (void *)&td[i]);
	}
	void* ret = NULL;
	vector<Mat> keypoint_container;
	for (int k = 0; k < NUM_THREADS; k++) {
		pthread_join(threads[k], &ret);
		int h = td[k].valid_interest_points;
		if (h > 0) {
			keypoint_container.push_back(td[k].matrice.colRange(0,h));
		}
	}
	Mat valid_points;
	hconcat(keypoint_container, valid_points);
	// Test REct region for at se, om du får det rigtige. 
	
	
	/*
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

				}
			}

		keypoints.at<double>(0, count) = max;
		keypoints.at<double>(1, count) = y;
		keypoints.at<double>(2, count) = x;
	
		nonMaximumSuppression(dst_norm, y-NMSBox, x-NMSBox, y+NMSBox+1, x+NMSBox+1);

	}
	*/
		
	
	//return keypoints;
	return valid_points;
	
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
//FindDescriptors(Mat src_gray, Mat keypoints)
/*
// Find SIFT Desriptors using Parallelization
void *FindDescriptors(void *threadarg) {
	// SIT = SIFT_Descriptor_thread
	struct SIT *my_data;
	my_data = (struct SIT *) threadarg;
	
	// Simplification of SIFT
	// Maybe the image should be smoothed first with a Gaussian Kernel
	int n = my_data->keypoints.cols;
	
	// Initialize matrix containing keypoints descriptors
	// Matrix Descriptors(n,128);
	Mat Descriptors = Mat::zeros(n, 128, CV_64FC1);
	
	// Find Image gradients
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int ksize = 1;
	int scale = 1; 
	int delta = 0; 
	int ddepth = CV_16S;
	
	// Find the gradients in the Sobel operator by using the OPENCV function
	Sobel(my_data->image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(my_data->image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	
	// Converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	
	int filter_size = 16;
	float sigma = 1.5*16;
	Mat GaussWindow;
	GaussWindow = gaussWindow(filter_size, sigma);
	
	// For each keypoint
	for (int i = 0; i < n; i++) {
		int y = my_data->keypoints.at<double>(1, i);
		int x = my_data->keypoints.at<double>(2, i); 

		
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
				Histogram = circularShift(Histogram);
				for (int ii = 0; ii < Histogram.dim2(); ii++) {
					//Descriptors[i].slice(nindex*8,nindex*8+ii) = Histogram(0,ii);
					//Descriptors(i,nindex*8+ii) = Histogram(0,ii);
					Descriptors.at<double>(i,nindex*8+ii) = Histogram(0,ii);
				}
				nindex++;
			}
		}
		
		// Normalizing the vector 
		double SumOfSquares = 0;
		for (int ii = 0; ii < Descriptors.cols; ii++) {
			SumOfSquares = SumOfSquares + Descriptors.at<double>(i,ii)*Descriptors.at<double>(i,ii);
		}

	// Scale the norms of the gradients by multiplying a the graidents with a gaussian
	// centered in the keypoint and with Sigma_w = 1.5*16. 

	}
	my_data->descriptors = Descriptors;
	
	pthread_exit(NULL);
}
*/


void *functionDescriptor(void *threadarg) {
   struct thread_descriptor *my_data;
   my_data = (struct thread_descriptor *) threadarg;
   
   //cout << "mistake 0 here " << endl;
   int y = my_data->thread_interest_point.at<double>(0,0);
   int x = my_data->thread_interest_point.at<double>(1,0);
		
	Mat grad_x = my_data->thread_grad_x;
	Mat grad_y = my_data->thread_grad_y;
	Mat GaussWindow = my_data->thread_Gauss_Window;
		
	// Extract a patch of size 16,16 from the image with x-gradients and y-gradients
	Mat Patch_Ix, Patch_Iy;

	Patch_Ix = selectRegionOfInterest(grad_x, y-7, x-7, y+8, x+8);
	Patch_Iy = selectRegionOfInterest(grad_y, y-7, x-7, y+8, x+8);
		
	//cout << "Mistake 1 here" << endl;
	// This is the scaled gradients 
	Mat Gradients = Mat::zeros(Size(16,16),CV_64FC1);
	// This is the orientations (angles of the gradients in radians)
	Mat Orientations = Mat::zeros(Size(16,16),CV_64FC1);

		//cout << "Test angle = " << atan2(-0.2,-5) << endl;

		for (int coor_y = 0; coor_y < 16; coor_y++) {
			for (int coor_x = 0; coor_x < 16; coor_x++) {
				double dx = Patch_Ix.at<short>(coor_y,coor_x);
				double dy = Patch_Iy.at<short>(coor_y,coor_x);
				//cout << "dx,dy = (" << dx << "," << dy << ")" << endl;
				//waitKey(0);
				double norm = sqrt( pow(dx,2.0) + pow(dy,2.0));
				//cout << "norm = " << norm << endl;
				//cout << "GaussWindow.at<double>(coor_y,coor_x) = " << GaussWindow.at<double>(coor_y,coor_x) << endl;
				Gradients.at<double>(coor_y,coor_x) = norm*GaussWindow.at<double>(coor_y,coor_x);
				Orientations.at<double>(coor_y,coor_x) = atan2(dy,dx);
				//cout << "Orientations.at<double>(coor_y,coor_x) = " << Orientations.at<double>(coor_y,coor_x) << endl;
			}
		}		
		// Maybe you should rotate the patch, so it coincides with the orientation in the strongest direction
		
		// Divde the 16x16 patch into subpatches of 4x4 
		Mat Patch_of_HOGs = Mat::zeros(16, 8, CV_64FC1);
		
		//cout << "Gradients = " << Gradients << endl;
		//cout << "Orientations = " << Orientations << endl;
		
		// Build a HOG for every 4x4 subpatch and insert that in the 16x8 matrix called Patch_of_HOGs
		//cout << "Mistake 2 here" << endl;
		int index = 0;
		for (int k1 = 0; k1 < 4; k1++) {
			for (int k2 = 0; k2 < 4; k2++) {
				Mat subPatch_gradients = Gradients.colRange(k2*4,(k2+1)*4).rowRange(k1*4,(k1+1)*4); 
				Mat subPatch_orientations = Orientations.colRange(k2*4,(k2+1)*4).rowRange(k1*4,(k1+1)*4); 
				
				//cout << "subPatch_gradients = " << subPatch_gradients << endl;
				//cout << "subPatch_orientations = " << subPatch_orientations << endl;
				
				//MatType(subPatch_gradients);
				//MatType(subPatch_orientations);
				
				for (int r = 0; r < 4; r++) {
					for (int c = 0; c < 4; c++) {
						double angle = subPatch_orientations.at<double>(r,c);
						
						if (0 <= angle && angle < M_PI/4) { // Between 0 rad and Pi/4 rad 
							Patch_of_HOGs.at<double>(index,0) = Patch_of_HOGs.at<double>(index,0) + subPatch_gradients.at<double>(r,c);
						}
						else if (M_PI/4 <= angle && angle < M_PI/2) { // Between Pi/4 rad and Pi/2 rad 
							Patch_of_HOGs.at<double>(index,1) = Patch_of_HOGs.at<double>(index,1) + subPatch_gradients.at<double>(r,c);
						}
						else if (M_PI/2 <= angle && angle < (3*M_PI)/4) { // Between Pi/2 rad and 3*Pi/4 rad 
							Patch_of_HOGs.at<double>(index,2) = Patch_of_HOGs.at<double>(index,2) + subPatch_gradients.at<double>(r,c);
						}
						else if ((3*M_PI)/4 <= angle && angle < M_PI) { // Between 3*Pi/4 rad and Pi rad 
							Patch_of_HOGs.at<double>(index,3) = Patch_of_HOGs.at<double>(index,3) + subPatch_gradients.at<double>(r,c);
						}
						else if (angle < -(3*M_PI)/4 && -M_PI <= angle) { // Between -3*Pi/4 rad and -Pi rad 
							Patch_of_HOGs.at<double>(index,4) = Patch_of_HOGs.at<double>(index,4) + subPatch_gradients.at<double>(r,c);
						}
						else if (angle < -(M_PI)/2 && -(3*M_PI)/4 <= angle) { // Between -Pi/2 rad and -3*Pi/4 rad 
							Patch_of_HOGs.at<double>(index,5) = Patch_of_HOGs.at<double>(index,5) + subPatch_gradients.at<double>(r,c);
						}
						else if (angle < -(M_PI)/4 && -(M_PI)/2 <= angle) { // Between -Pi/4 rad and -Pi/2 rad 
							Patch_of_HOGs.at<double>(index,6) = Patch_of_HOGs.at<double>(index,6) + subPatch_gradients.at<double>(r,c);
						}
						else if (angle < 0 && -(M_PI)/4 <= angle) { // Between 0 rad and -Pi/4 rad 
							Patch_of_HOGs.at<double>(index,7) = Patch_of_HOGs.at<double>(index,7) + subPatch_gradients.at<double>(r,c);
						}
					}
				}
				
				//cout << "Patch_of_HOGs = " << Patch_of_HOGs << endl;
				//waitKey(0);
				
				index++;
			}
		}
		
		//cout << "Mistake 3 here" << endl;
		// Build HOG for the entire 16x16 Patch
		Mat total_HOG = Mat::zeros(1, 8, CV_64FC1);
		for (int q = 0; q < Patch_of_HOGs.rows; q++) {
			total_HOG.at<double>(0,0) = total_HOG.at<double>(0,0) + Patch_of_HOGs.at<double>(q,0);
			total_HOG.at<double>(0,1) = total_HOG.at<double>(0,1) + Patch_of_HOGs.at<double>(q,1);
			total_HOG.at<double>(0,2) = total_HOG.at<double>(0,2) + Patch_of_HOGs.at<double>(q,2);
			total_HOG.at<double>(0,3) = total_HOG.at<double>(0,3) + Patch_of_HOGs.at<double>(q,3);
			total_HOG.at<double>(0,4) = total_HOG.at<double>(0,4) + Patch_of_HOGs.at<double>(q,4);
			total_HOG.at<double>(0,5) = total_HOG.at<double>(0,5) + Patch_of_HOGs.at<double>(q,5);
			total_HOG.at<double>(0,6) = total_HOG.at<double>(0,6) + Patch_of_HOGs.at<double>(q,6);
			total_HOG.at<double>(0,7) = total_HOG.at<double>(0,7) + Patch_of_HOGs.at<double>(q,7);
			
		}
		//cout << "Mistake 4 here" << endl;
		// Find the biggest bin in the total HOG for the entire 16x16 patch
		double max_bin = 0; 
		int max_bin_index = 0;
		for (int q = 0; q < total_HOG.cols; q++) {
			if (total_HOG.at<double>(0,q) > max_bin) {
				max_bin = total_HOG.at<double>(0,q);
				max_bin_index = q;
			}
		}
		
		/*
		cout << "total_HOG = " << total_HOG << endl;
		cout << "max_bin = " << max_bin << endl;
		cout << "max_bin_index = " << max_bin_index << endl;
		
		cout << "Patch_of_HOGs = " << Patch_of_HOGs << endl;
		
		cout << "Mistake with max_bin_index" << endl;
		*/
		if (max_bin_index == 0) {
			max_bin_index = 1;
		}
		Mat temp1 = Patch_of_HOGs.colRange(0,max_bin_index);
		Mat temp2 = Patch_of_HOGs.colRange(max_bin_index,8);
		Mat temp3;
		hconcat(temp2, temp1, temp3);
		/*
		cout << "did not reach this point" << endl;
		waitKey(0);
		*/
		//cout << "temp3 = " << endl;
		//cout << temp3 << endl;
		
		// Reshape the matrix so it goes from a 16x8 Matrix to a 1x128 matrix
		//cout << "Mistake 5 here" << endl;
		Mat descrip = temp3.reshape(0, 1);
		
		
		/*
		cout << "descrip = " << descrip << endl;
		waitKey(0);
		*/
		for (int k = 0; k < descrip.cols; k++) {
			my_data->thread_descriptor_vector.at<double>(0,k) = descrip.at<double>(0,k);
		}	
		
		
		// Normalizing the vector 
		double SumOfSquares = 0;
		for (int ii = 0; ii < descrip.cols; ii++) {
			SumOfSquares = SumOfSquares + my_data->thread_descriptor_vector.at<double>(0,ii)*my_data->thread_descriptor_vector.at<double>(0,ii);
		}
		
		
		
		
		//cout << "Mistake 6 here" << endl;
		for (int ii = 0; ii < descrip.cols; ii++) {
			my_data->thread_descriptor_vector.at<double>(0,ii) = my_data->thread_descriptor_vector.at<double>(0,ii)/sqrt(SumOfSquares);
		}
		
   
   pthread_exit(NULL);
}



// This function works - 25.07-2020
// Find SIFT Desriptors  without parallelization
Mat SIFT::FindDescriptors(Mat src_gray, Mat keypoints) {
	
	Mat src_gray_blurred;
	GaussianBlur(src_gray, src_gray_blurred, Size(3,3), 5,5);

	// Simplification of SIFT
	// Maybe the image should be smoothed first with a Gaussian Kernel
	int n = keypoints.cols;
	
	// Initialize matrix containing keypoints descriptors
	//Matrix Descriptors(n,128);
	Mat Descriptors = Mat::zeros(n, 128, CV_64FC1);
	
	// Find Image gradients
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int ksize = 1;
	int scale = 1; 
	int delta = 0; 
	int ddepth = CV_16S;
	
	// Find the gradients in the Sobel operator by using the OPENCV function
	Sobel(src_gray_blurred, grad_y, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(src_gray_blurred, grad_x, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	
	// Converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	
	int filter_size = 16;
	float sigma = 1.5*16;
	Mat GaussWindow;
	GaussWindow = gaussWindow(filter_size, sigma);
	
	
	grad_y = (-1)*grad_y;
	grad_x = (-1)*grad_x;
	
	
	int NUM_THREADS = n;
	pthread_t threads[NUM_THREADS];
	struct thread_descriptor td[NUM_THREADS];
	int i, rc;
	for (i = 0; i < NUM_THREADS; i++) {
		td[i].thread_interest_point = keypoints.colRange(i,i+1);

		td[i].thread_grad_x = grad_x;
		//cout << "td[i].thread_grad_x = " << td[i].thread_grad_x << endl;
		td[i].thread_grad_y = grad_y;
		//cout << "td[i].thread_grad_y = " << td[i].thread_grad_y << endl;
		td[i].thread_descriptor_vector = Descriptors.row(i);
		
		td[i].thread_Gauss_Window = GaussWindow;
		
		rc = pthread_create(&threads[i], NULL, functionDescriptor, (void *)&td[i]);
	}
	void* ret = NULL;
	vector<Mat> keypoint_container;
	for (int k = 0; k < NUM_THREADS; k++) {
		pthread_join(threads[k], &ret);
	}
	
	
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
	
	// Initialize patch
	Mat patch = Mat::zeros(2*r_T + 1, 2*r_T + 1, CV_64FC1);
	
	// Get dimensions of image 
	int max_coords_rows = I_new.rows;
	int max_coords_cols = I_new.cols;
	
	//cout << "Dimensions of image = (" << max_coords_rows << "," << max_coords_cols << ")" << endl;
	
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
					
					//cout << "Inside if-statements" << endl;

					Mat floors = Mat::zeros(warped.rows, warped.cols, CV_64FC1);
					
					
					for (int r = 0; r < floors.rows; r++) {
						for (int c = 0; c < floors.cols; c++) {
							floors.at<double>(r, c) = floor(warped.at<double>(r, c));
						}
					}
					
					Mat weights = warped - floors;
					
					
					double a = weights.at<double>(0,0);
					double b = weights.at<double>(0,1);
					
					
					double intensity = (1-b) * ((1-a) * I_new.at<uchar>(floors.at<double>(0,1)-1,floors.at<double>(0,0)-1) + a * I_new.at<uchar>(floors.at<double>(0,1)-1,floors.at<double>(0,0)));
					

					
					intensity = intensity + b * ((1-a) * I_new.at<uchar>(floors.at<double>(0,1),floors.at<double>(0,0)-1) + a * I_new.at<uchar>(floors.at<double>(0,1),floors.at<double>(0,0)));
					
					
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
//Mat trackKLT(Mat I_R, Mat I_new, Mat x_T, int r_T, int num_iters) {
Mat trackKLT(Mat I_R, Mat I_new, Mat x_T, Mat dwdx, int r_T, int num_iters) {
	//Mat p_hist = Mat::zeros(6, num_iters+1, CV_64FC1);
	Mat W = getSimWarp(0, 0, 0, 1);
	
	
	int temp_index = 0;
	
	/*
	for (int c = 0; c < W.cols; c++) {
		for (int r = 0; r < W.rows; r++) {
			p_hist.at<double>(temp_index, 0) = W.at<double>(r,c);
			temp_index++;
		}
	}
	*/
	
	// Get the warped patch
	Mat I_RT = getWarpedPatch(I_R, W, x_T, r_T);
		
	
	I_RT = I_RT.t();
	Mat i_R = I_RT.reshape(0,I_RT.rows * I_RT.cols);
	
	int n = 2*r_T + 1;
	/*
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
	*/
	
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
	Mat IWT_temp, IWT, IWTx, IWTy, temp_IWTx, temp_IWTy, temp_IWTx2, temp_IWTy2, didw, H, temp_delta_p, delta_p, delta_p_temp;
	for (int iter = 0; iter < num_iters; iter++) {
		Mat big_IWT = getWarpedPatch(I_new, W, x_T, r_T + 1); // We are here 		
		
		//Mat IWT_temp, IWT;
		IWT_temp = selectRegionOfInterest(big_IWT, 1, 1, big_IWT.rows-1, big_IWT.cols-1);
		IWT_temp.copyTo(IWT);
	
		IWT = IWT.t();
		Mat i = IWT.reshape(0, IWT.rows * IWT.cols);
		
		// Getting di/dp 
		//cout << "Getting di/dp" << endl;
		//Mat IWTx, IWTy, temp_IWTx, temp_IWTy, temp_IWTx2, temp_IWTy2;
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
		
		
		
		//Mat didw;
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
		H = didp.t() * didp;
		
		// Hessian matrix check
		
		temp_delta_p = didp.t() * (i_R - i);
		
		// Calculate delta_p 
		delta_p = H.inv() * (didp.t() * (i_R - i)); // Maybe problem with 
		
		// Reshape delta_p 
		delta_p_temp = delta_p.reshape(0, 3);
		
		delta_p_temp = delta_p_temp.t();
		
		W = W + delta_p_temp; // 2 = W.rows
		 
		W = W.t();
		
		// Transpose W to get the right shape for the next iteration - C++ is different from Matlab
		W = W.t();

		
	}
	return W;
}

//Mat KLT::trackKLTrobustly(Mat I_R, Mat I_new, Mat keypoint, int r_T, int num_iters, double lambda) {
Mat KLT::trackKLTrobustly(Mat I_R, Mat I_new, Mat keypoint, Mat dwdx, int r_T, int num_iters, double lambda) {
	
	Mat W = trackKLT(I_R, I_new, keypoint, dwdx, r_T, num_iters);
	
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
	
	// Brug for threading her. 
	Mat Winv = trackKLT(I_new, I_R, reverse_keypoint, dwdx, r_T, num_iters);
	
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
	
	Mat roots = Mat::zeros(1, 4, CV_64FC1);
	
	double A = factors.at<double>(0,0);
	double B = factors.at<double>(0,1);
	double C = factors.at<double>(0,2);
	double D = factors.at<double>(0,3);
	double E = factors.at<double>(0,4);
	
	//cout << "Factors = " << A << "," << B << "," << C << "," << D << "," << E << endl;
	
	double A_pw2 = A*A;
	double B_pw2 = B*B; 
	double A_pw3 = A_pw2*A;
	double B_pw3 = B_pw2*B;
	double A_pw4 = A_pw3*A;
	double B_pw4 = B_pw3*B;
	
	double alpha = -3*B_pw2/(8*A_pw2) + C/A;
	double beta = B_pw3/(8*A_pw3) - B*C/(2*A_pw2) + D/A;
	double gamma = -3*B_pw4/(256*A_pw4) + B_pw2*C/(16*A_pw3) - B*D/(4*A_pw2) + E/A;
	
	double alpha_pw2 = alpha * alpha; 
	double alpha_pw3 = alpha_pw2 * alpha;
	
	double P = -alpha_pw2/12 - gamma;
	double Q = -alpha_pw3/108 + alpha*gamma/3 - pow(beta,2.0)/8;
	
	
	std::complex<double> i_value;
	i_value = 1i;

	
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
	
	
	null_v = 0. + 0i;
	real_value = -5.0*alpha/6.0;
	if (U == null_v) {
		y = real_value - pow(Q, 1.0/3.0);
	}
	else {
		y = real_value - P/(3.*U) + U;
	}
	
	
	w = pow(alpha+2.*y, 1.0/2.0);
	
	//cout << "w = " << w << endl;
	
	std::complex<double> temp0, temp1, temp2, temp3;
	real_value = -B/(4*A);
	
	
	temp0 = real_value + 0.5*(w + pow( -(3*alpha+2.*y+2.*beta/w), 1.0/2.0));
	temp1 = real_value + 0.5*(w - pow( -(3*alpha+2.*y+2.*beta/w), 1.0/2.0));
	temp2 = real_value + 0.5*(-w + pow( -(3*alpha+2.*y-2.*beta/w), 1.0/2.0));
	temp3 = real_value + 0.5*(-w - pow( -(3*alpha+2.*y-2.*beta/w), 1.0/2.0));
	
	
	//roots.at<double>(0,0) = real(temp0);
	//roots.at<double>(0,1) = real(temp1);
	//roots.at<double>(0,0) = real(temp2);
	//roots.at<double>(0,1) = real(temp3);

	
	temp0 = real(temp0);
	temp1 = real(temp1);
	temp2 = real(temp2);
	temp3 = real(temp3);
	
	roots.at<double>(0,0) = real(temp0);
	roots.at<double>(0,1) = real(temp1);
	roots.at<double>(0,2) = real(temp2);
	roots.at<double>(0,3) = real(temp3);
	
	
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
		
		Mat Nt = N_complex.t();
		
		Mat temp = Nt*C;
		
		C = P1_complex + temp;
		
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
		
		
		R = N_complex.t() * R.t() * T_complex;
		
		
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
	int min_inlier_count = 30; // This parameter should be tuned for the implementation
	double record_inlier = 0;

		
	if (adaptive_ransac) {
		num_iterations = INFINITY;
	}
	else {
		num_iterations = 1000;
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
			//cout << "Random number = " << random_nums[mm] << endl;
			// Landmark sample 
			landmark_sample.at<double>(0,mm) = corresponding_landmarks.at<double>(0, random_nums[mm]);
			landmark_sample.at<double>(1,mm) = corresponding_landmarks.at<double>(1, random_nums[mm]);
			landmark_sample.at<double>(2,mm) = corresponding_landmarks.at<double>(2, random_nums[mm]);
			
			// Keypoint sample 
			keypoint_sample.at<double>(0,mm) = matched_query_keypoints.at<double>(0, random_nums[mm]);
			keypoint_sample.at<double>(1,mm) = matched_query_keypoints.at<double>(1, random_nums[mm]);
		}
		
		
		normalized_bearings = K.inv() * keypoint_sample;
		
		for (int ii = 0; ii < 3; ii++) {
			double vector_norm = sqrt(pow(normalized_bearings.at<double>(0,ii),2.0) + pow(normalized_bearings.at<double>(1,ii),2.0) + pow(normalized_bearings.at<double>(2,ii),2.0));
			normalized_bearings.at<double>(0,ii) = normalized_bearings.at<double>(0,ii)/vector_norm;
			normalized_bearings.at<double>(1,ii) = normalized_bearings.at<double>(1,ii)/vector_norm;
			normalized_bearings.at<double>(2,ii) = normalized_bearings.at<double>(2,ii)/vector_norm;
		
		}
		
		
		poses = p3p(landmark_sample, normalized_bearings);
	
		/*
		 * To check if some of the values are NaN
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
			record_inlier = countNonZero(is_inlier);
			R_C_W_guess.copyTo(best_R_C_W);
			t_C_W_guess.copyTo(best_t_C_W);
		}
		
		for (int alt_idx = 1; alt_idx <= 3; alt_idx++) {
			for (int k = 0; k < 3; k++) {
				R_W_C.at<double>(0,k) = poses.at<double>(0,k+1 + alt_idx*4);
				R_W_C.at<double>(1,k) = poses.at<double>(1,k+1 + alt_idx*4);
				R_W_C.at<double>(2,k) = poses.at<double>(2,k+1 + alt_idx*4);
				t_W_C.at<double>(k,0) = poses.at<double>(k, alt_idx*4);
			}
			
			R_C_W_guess = R_W_C.t();

			t_C_W_guess = -R_W_C.t()*t_W_C;
			
			points = R_C_W_guess * corresponding_landmarks + repeat(t_C_W_guess, 1, corresponding_landmarks.cols);
			projected_points = projectPoints(points, K);
			
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

			if (countNonZero(is_inlier) > record_inlier && countNonZero(is_inlier) >= min_inlier_count) {
				record_inlier = countNonZero(is_inlier);
				R_C_W_guess.copyTo(best_R_C_W);
				t_C_W_guess.copyTo(best_t_C_W);

			}
			
		}
		
		if (countNonZero(is_inlier) > max_num_inliers && countNonZero(is_inlier) >= min_inlier_count) {
			max_num_inliers = countNonZero(is_inlier);
			best_inlier_mask = is_inlier;
		}
		
		if (adaptive_ransac) {
			float division = (float) max_num_inliers/ (float) is_inlier.cols;
			//cout << "division = " << division << endl;
			float outlier_ratio = 1. - division;
			
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
	/*
	cout << "best_R_C_W" << endl;
	cout << best_R_C_W << endl;
	cout << "best_t_C_W" << endl;
	ćout << best_t_C_W << endl;
	*/
	
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
	cout << "Finding new Candidate Keypoints for the first time" << endl;
	
	// Convert the image to gray scale 
	Mat Ii_gray;
	cvtColor(Ii, Ii_gray, COLOR_BGR2GRAY);
	
		
	// Find new keypoints 
	int keypoint_max = num_candidate_keypoints + Si.k; // Select at least num_candidate_keypoints new candidate keypoints
	int dim1, dim2;
	dim1 = Ii_gray.rows;
	dim2 = Ii_gray.cols;
	
	// Make an image that is used to check whether that keypoint had already been found before. 
	Mat checkImage = Mat::zeros(dim1, dim2, CV_64FC1);
	for (int i = 0; i < Si.k; i++) {
		checkImage.at<double>(Si.Pi.at<double>(0,i),Si.Pi.at<double>(1,i)) = 1;
	}
	
	Mat Ii_resized = Ii_gray.colRange(10,dim2-10).rowRange(10,dim1-10);
	Mat temp1;
	goodFeaturesToTrack(Ii_resized, temp1, keypoint_max, 0.01, 10, noArray(), 3, true, 0.04);
	Mat keypoints_Ii = Mat::zeros(2, temp1.rows, CV_64FC1);
	
	int a, b, temp_index;
	temp_index = 0;
	
	for (int i = 0; i < keypoints_Ii.cols; i++) {
		int a = temp1.at<float>(i,1) + 10;
		int b = temp1.at<float>(i,0) + 10;
		//circle (Ii_gray, Point(b,a), 5,  Scalar(0,0,255), 2,8,0);
		if ( checkImage.at<double>(a,b) != 1 ) {
			keypoints_Ii.at<double>(0,temp_index) = a;
			keypoints_Ii.at<double>(1,temp_index) = b;
			temp_index++;
		}
	}
	keypoints_Ii = keypoints_Ii.colRange(0,30);
	
	// Update state 
	Si.num_candidates = keypoints_Ii.cols;
	
	cout << "Number of New candidate Keypoints" << endl;
	
	// Update keypoints
	keypoints_Ii.copyTo(Si.Ci); 
	
	
	// Draw keypoints Si.Pi
	Mat Ii_draw;
	Ii.copyTo(Ii_draw);
	for (int i = 0; i < Si.Pi.cols; i++) {
		circle (Ii_draw, Point(Si.Pi.at<double>(1,i),Si.Pi.at<double>(0,i)), 5,  Scalar(0,0,255), 2,8,0);
	}
	imshow("newCandidateKeypoints Ii Si.Pi", Ii_draw);
	waitKey(0);
	// Draw Candidate keypoints Si.Ci
	for (int i = 0; i < Si.Ci.cols; i++) {
		circle (Ii_draw, Point(Si.Ci.at<double>(1,i),Si.Ci.at<double>(0,i)), 5,  Scalar(255,0,0), 2,8,0);
	}
	imshow("newCandidateKeypoints Ii Si.Ci", Ii_draw);
	waitKey(0);
	
	
	// Update first observation of keypoints
	keypoints_Ii.copyTo(Si.Fi);
	
	// Update the camera poses at the first observations.
	Mat t_C_W_vector = T_wc.reshape(0,T_wc.rows * T_wc.cols);
	Mat poses = repeat(t_C_W_vector, 1, keypoints_Ii.cols);
	poses.copyTo(Si.Ti);
	
	return Si;
}

/* Name: continuousCandidateKeypoints
 * Objective: This function tracks candidate keypoints stored in the matrix Si.Ci from previous frame I^(i-1) to current frame I^i. 
 * If a candidate keypoint is successfully tracked from frame I^(i-1) to current frame I^i the coordinates of the specific keypoint
 * is updated in Si.Ci. 
 * If a candidate keypoint is not tracked successfully it is simply discarded.
 * If the number of candidate keypoints fall below a certain threshold specified as num_candidate_keypoints in mainCamera.hpp new
 * potential candidate keypoints are found in frame I^i. These points are then concatenated horizontally to the other candidate keypoints
 * 
 * Inputs:
 * Mat Ii_1 - Previous frame I^(i-1)
 * Mat Ii - Current frame I^i
 * State Si - The state which is a struct containing keypoints, corresponding landmarks, newest detection of candidate keypoints, 
 * first detection of candidate keypoints and newest detection of candidate keypoints, 
 * Mat T_wc - A matrix of size 3x4 which contains the rotation and translation
 * 
 * Output:
 * State Si - An update of state Si
 */
state continuousCandidateKeypoints(Mat Ii_1, Mat Ii, state Si, Mat T_wc) {
	cout << "continuousCandidateKeypoints " << endl;
	
	// Variables used for for loops
	int i, q;
	
	Mat Ii_1_gray, Ii_gray;
	cvtColor(Ii_1, Ii_1_gray, COLOR_BGR2GRAY);
	cvtColor(Ii, Ii_gray, COLOR_BGR2GRAY);
	
	
	// Find descriptors for candidate keypoints from previous frame
	Mat descriptors_candidates_Ii_1 = SIFT::FindDescriptors(Ii_1_gray, Si.Ci);
	
	int dim1, dim2;
	dim1 = Ii_gray.rows;
	dim2 = Ii_gray.cols;
	
	// Create a Matrix of same size as image and use it to ignorre keypoints that are already points
	Mat checkImage = Mat::zeros(dim1, dim2, CV_64FC1);
	for (i = 0; i < Si.k; i++) {
		checkImage.at<double>(Si.Pi.at<double>(0,i),Si.Pi.at<double>(1,i)) = 1;
	}
	
	// Find keypoints in new image
	Mat Ii_resized = Ii_gray.colRange(10,dim2-10).rowRange(10,dim1-10);
	Mat temp1;
	goodFeaturesToTrack(Ii_resized, temp1, 300, 0.01, 10, noArray(), 3, true, 0.04);
	Mat keypoints_Ii = Mat::zeros(2, temp1.rows, CV_64FC1);
	int a, b, temp_index;
	temp_index = 0;
	for (i = 0; i < keypoints_Ii.cols; i++) {
		a = temp1.at<float>(i,1) + 10;
		b = temp1.at<float>(i,0) + 10;
		if ( checkImage.at<double>(a,b) != 1 ) {
			keypoints_Ii.at<double>(0,temp_index) = a;
			keypoints_Ii.at<double>(1,temp_index) = b;
			temp_index++;
		}
	}
	keypoints_Ii = keypoints_Ii.colRange(0,temp_index);
	
	// Find descriptors for the new keypoints
	Mat descriptors_candidates_Ii = SIFT::FindDescriptors(Ii_gray, keypoints_Ii);
	
	// Find matches
	Mat matches = SIFT::matchDescriptors(descriptors_candidates_Ii_1, descriptors_candidates_Ii);
	Mat matches2 = SIFT::matchDescriptors(descriptors_candidates_Ii, descriptors_candidates_Ii_1);
	
	Mat valid_matches = Mat::zeros(2, matches.cols, CV_64FC1);
	temp_index = 0;
	for (i = 0; i < matches.cols; i++) {
		int index_frame0 = matches.at<double>(0,i);
		int index_frame1 = matches.at<double>(1,i);
		
		for (q = 0; q < matches2.cols; q++) {
			if (matches2.at<double>(1,q) == index_frame0) {
				if (matches2.at<double>(0,q) == index_frame1) {
					// Mutual match
					valid_matches.at<double>(0,temp_index) = index_frame0;
					valid_matches.at<double>(1,temp_index) = index_frame1;
					temp_index++;
				}
			}
		}
	}
	matches = valid_matches.colRange(0,temp_index);
	cout << "Number of mutual valid mathces = " << matches.cols << endl;
	
	vector<Mat> Ci_container;
	vector<Mat> Fi_container;
	vector<Mat> Ti_container;
	
	for (int i = 0; i < matches.cols; i++) {
		Ci_container.push_back(keypoints_Ii.col(valid_matches.at<double>(1,i)));
		Fi_container.push_back(Si.Fi.col(valid_matches.at<double>(0,i)));
		Ti_container.push_back(Si.Ti.col(valid_matches.at<double>(0,i)));
	}
	hconcat(Ci_container, Si.Ci);
	hconcat(Fi_container, Si.Fi);
	hconcat(Ti_container, Si.Ti);
	Si.num_candidates = Si.Ci.cols;
	
	
	// Draw keypoints Si.Pi
	Mat Ii_draw;
	Ii.copyTo(Ii_draw);
	for (int i = 0; i < Si.Pi.cols; i++) {
		circle (Ii_draw, Point(Si.Pi.at<double>(1,i),Si.Pi.at<double>(0,i)), 5,  Scalar(0,0,255), 2,8,0);
	}
	imshow("continuousCandidateKeypoints Ii Si.Pi", Ii_draw);
	waitKey(0);
	// Draw Candidate keypoints Si.Ci
	for (int i = 0; i < Si.Ci.cols; i++) {
		circle (Ii_draw, Point(Si.Ci.at<double>(1,i),Si.Ci.at<double>(0,i)), 5,  Scalar(255,0,0), 2,8,0);
	}
	imshow("continuousCandidateKeypoints Ii Si.Ci", Ii_draw);
	waitKey(0);
	
	if (Si.num_candidates < num_candidate_keypoints) {
		
		// Number of new candidate keypoints that should be found 
		int n = num_candidate_keypoints - Si.num_candidates; 
		
		// Start by ignorring current candidate keypoints
		for (i = 0; i < Si.num_candidates; i++) {
			checkImage.at<double>(Si.Ci.at<double>(0,i),Si.Ci.at<double>(1,i)) = 1;
		}
		
		int k = Si.k + Si.num_candidates + n;
		Mat temp2;
		goodFeaturesToTrack(Ii_resized, temp2, k, 0.01, 10, noArray(), 3, true, 0.04);
		Mat Potential_new_CI = Mat::zeros(2, temp2.rows, CV_64FC1);
		temp_index = 0;
		for (int i = 0; i < Potential_new_CI.cols; i++) {
			a = temp2.at<float>(i,1) + 10;
			b = temp2.at<float>(i,0) + 10;
			
			if ( checkImage.at<double>(a,b) != 1 ) {
				Potential_new_CI.at<double>(0,temp_index) = a;
				Potential_new_CI.at<double>(1,temp_index) = b;
				temp_index++;
			}
		}
		
		// Concatenate the newly founded keypoints
		hconcat(Potential_new_CI.colRange(0,temp_index), Si.Ci);
		hconcat(Potential_new_CI.colRange(0,temp_index), Si.Fi);
		
		Mat t_C_W_vector = T_wc.reshape(0,T_wc.rows * T_wc.cols);
		Mat poses = repeat(t_C_W_vector, 1, temp_index);
		
		// Concatenate the current poses
		hconcat(poses, Si.Ti);
		
		
	}
	
	// Update number of candidate keypoints
	Si.num_candidates = Si.Ci.cols;
	
	cout << "Number of candidate keypoints = " << Si.Ci.cols << endl;
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
Mat findLandmark(Mat K, Mat tau, Mat T_WC, Mat imagepoint0, Mat imagepoint1) {
	Mat P;
	Mat Q;
	
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
//tuple<state, Mat>  triangulateNewLandmarks(state Si, Mat K, Mat T_WC, double threshold_angle) {
state triangulateNewLandmarks(state Si, Mat K, Mat T_WC, double threshold_angle) {
	
	// Check if points are ready to be traingulated 
	Mat extracted_keypoints = Mat::zeros(1, Si.num_candidates, CV_64FC1);
	
	// Matrices to store valid extracted keypoints
	Mat newKeypoints = Mat::zeros(2, Si.num_candidates, CV_64FC1);
	Mat traingulated_landmark;
	Mat newLandmarks = Mat::zeros(Si.Xi.rows, Si.num_candidates, CV_64FC1);
	int i;
	int temp = 0;
	
	Mat keypoint_last_occur = Mat::ones(3, 1, CV_64FC1);
	Mat keypoint_newest_occcur = Mat::ones(3, 1, CV_64FC1);
	Mat tau;
	Mat a, b; // a = previous_vector, b = current_vector;
	double length_first_vector, length_current_vector;
	
	// Paralleliser det her måske
	// Beregn vinklen mellem vektorerne current viewpoint og den første observation af keypointet
	double fraction, alpha;
	for (i = 0; i < Si.num_candidates; i++) {
		// First occurrence of keypoint
		keypoint_last_occur.at<double>(0,0) = Si.Fi.at<double>(0,i); // y-coordinate
		keypoint_last_occur.at<double>(1,0) = Si.Fi.at<double>(1,i); // x-coordinate

		// Newest occurrence of keypoint
		keypoint_newest_occcur.at<double>(0,0) = Si.Ci.at<double>(0,i); // y-coordinate
		keypoint_newest_occcur.at<double>(1,0) = Si.Ci.at<double>(1,i); // x--coordinate

		// Finding the angle using bearing vectors 
		Mat bearing1 = K.inv() * keypoint_last_occur;
		Mat bearing2 = K.inv() * keypoint_newest_occcur;
		
		// Finding length of vectors 
		length_first_vector = sqrt(pow(bearing1.at<double>(0,0),2.0) + pow(bearing1.at<double>(1,0),2.0) + pow(bearing1.at<double>(2,0),2.0));
		length_current_vector = sqrt(pow(bearing2.at<double>(0,0),2.0) + pow(bearing2.at<double>(1,0),2.0) + pow(bearing2.at<double>(2,0),2.0));
		
		// Determine the angle
		// The angle is in radians 
		// CHANGES NEEDED HERE
		double v = bearing1.at<double>(0,0)*bearing2.at<double>(0,0)+bearing1.at<double>(1,0)*bearing2.at<double>(1,0)+bearing1.at<double>(2,0)*bearing2.at<double>(2,0); // This value should be changed

		if ((v/(length_first_vector * length_current_vector)) > 1) {
			alpha = acos(1) * 360/(2*M_PI);
		}
		else if ((v/(length_first_vector * length_current_vector)) < -1) {
			alpha = acos(-1) * 360/(2*M_PI);
		}
		else {
			alpha = acos((v/(length_first_vector * length_current_vector))) * 360/(2*M_PI);
		}
		
		if (alpha > threshold_angle) {
			extracted_keypoints.at<double>(0,i) = 1;
			
			// Update new keypoints 
			newKeypoints.at<double>(0,temp) = Si.Ci.at<double>(0,i);
			newKeypoints.at<double>(1,temp) = Si.Ci.at<double>(1,i);
			
			traingulated_landmark = findLandmark(K, tau, T_WC, keypoint_last_occur, keypoint_newest_occcur); // Check if tau should be changed to matrix
			
			traingulated_landmark.copyTo(newLandmarks.col(temp));
			
			temp++;
		}
	}
	
	// Append new keypoints and new landmarks 
	if (temp > 0) {
		newKeypoints = newKeypoints.colRange(0,temp);
		newLandmarks = newLandmarks.colRange(0,temp);
		
		// Append new keypoints
		hconcat(Si.Pi, newKeypoints, Si.Pi);
		
		// Append new 3D landmarks 
		hconcat(Si.Xi, newLandmarks, Si.Xi);
	}
	
	
	// Remove the candidate keypoints that have been triangulated.
	vector<Mat> Ci_container;
	vector<Mat> Fi_container;
	vector<Mat> Ti_container;
	
	for (i = 0; i < extracted_keypoints.cols; i++) {
		if (extracted_keypoints.at<double>(0,i) == 0) {
			Ci_container.push_back( Si.Ci.col(i) );
			Fi_container.push_back( Si.Fi.col(i) );
			Ti_container.push_back( Si.Ti.col(i) );
		}
	}
	hconcat(Ci_container, Si.Ci);
	hconcat(Fi_container, Si.Fi);
	hconcat(Ti_container, Si.Ti);
	Si.num_candidates = Si.Ci.cols;
	
	//return make_tuple(Si, extracted_keypoints);
	return Si;
}





