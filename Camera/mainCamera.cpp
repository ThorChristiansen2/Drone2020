#include <iostream>
#include "mainCamera.hpp"
#include <limits> 
#include <assert.h> 
//#include <complex.h>
#include <complex>
#include <iomanip>

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
//using namespace std::complex_literals;
using namespace std::chrono;
//using namespace Numeric_lib;


// ####################### drawCorners #######################
void drawCorners(Mat img, Mat keypoints, const char* frame_name) {
	for (int k = 0; k < keypoints.cols; k++) {
		double y = keypoints.at<double>(0, k);
		double x = keypoints.at<double>(1, k);
		circle (img, Point(x,y), 5, Scalar(0,0,255), 2,8,0);
	}
	imshow(frame_name, img);
	waitKey(0);
}

// ####################### randomSampling #######################
// This part of the code comes from: https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement 
unordered_set<int> pickSet(int N, int k, std::mt19937& gen)
{
    unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = uniform_int_distribution<>(1, r)(gen);

        // there are two cases.
        // v is not in candidates ==> add it
        // v is in candidates ==> well, r is definitely not, because
        // this is the first iteration in the loop that we could've
        // picked something that big.

        if (!elems.insert(v).second) {
            elems.insert(r);
        }   
    }
    return elems;
}

vector<int> pick(int N, int k) {
    random_device rd;
    mt19937 gen(rd());

    unordered_set<int> elems = pickSet(N, k, gen);

    // ok, now we have a set of k elements. but now
    // it's in a [unknown] deterministic order.
    // so we have to shuffle it:

    vector<int> result(elems.begin(), elems.end());
    shuffle(result.begin(), result.end(), gen);
    return result;
}

void *functionMatch2(void *threadarg) {
   struct thread_match2 *my_data;
   my_data = (struct thread_match2 *) threadarg;
   
   double d1 = std::numeric_limits<double>::infinity();
   double best_match_d1 = std::numeric_limits<double>::infinity();
   double d2 = std::numeric_limits<double>::infinity();
   double best_match_d2 = std::numeric_limits<double>::infinity();
   int i, j;
   
   int dimension1 = my_data->descriptors_n1.rows; 
   int dimension2 = my_data->descriptors_n1.cols;
   
   for (i = 0; i < dimension1; i++) {
	   
	   double SSD = 0;
	   
	   for (j = 0; j < dimension2; j++) {
		   SSD = SSD + pow( my_data->descriptors_n1.at<double>(i, j) - my_data->descriptor_n2.at<double>(0, j) ,2.0);
	   }
	   // If a descriptor with lower distance has been detected:
	   if ( SSD < d1 ) {
		   // Update distance for 2nd best descriptor
		   d2 = d1; 
		   best_match_d2 = best_match_d1;
		   
		   d1 = SSD;
		   best_match_d1 = i;
	   }
	   // If a descriptor with second lowest distance has been detected 
	   else if ( SSD < d2 ) {
		   d2 = SSD;
		   best_match_d2 = i;
	   }
	   
   }
   
   if ( d2 != std::numeric_limits<double>::infinity() ) {
	   double distance_ratio = d1/d2;
	   if ( distance_ratio < 0.8 ) {
		   my_data->is_inlier = 1;
		   my_data->lowest_distance = d1;
		   my_data->best_match = best_match_d1;
	   }
   }
   
   pthread_exit(NULL);
}


// New attempt to write code 
/*
 * struct thread_match2 {
		//int thread_id;
		int descriptor_n2_id;
		Mat descriptors_n1;
		Mat descriptor_n2;
		int is_inlier;
		double lowest_distance;
		int best_match;
};
 * 
 * 
 */
Mat SIFT::matchDescriptors2(Mat descriptor1, Mat descriptor2) {

	int descriptor_length = descriptor2.cols;
	
	int n1 = descriptor1.rows;	// Matrix containing descriptors for keypoints in image 0
	int n2 = descriptor2.rows;	// Matrix containing descriptors for keypoints in image 1
	
	// To determine 
	Mat matches = Mat::ones(2, n1, CV_64FC1);
	matches = (-1)*matches;
	
	int NUM_THREADS = n2;
	pthread_t threads[NUM_THREADS];
	struct thread_match2 td[NUM_THREADS];
	int i, rc;
	for (i = 0; i < NUM_THREADS; i++) {
		td[i].descriptor_n2_id = i;
		td[i].descriptors_n1 = descriptor1;
		td[i].descriptor_n2 = descriptor2.row(i);
		td[i].is_inlier = 0;
		td[i].lowest_distance = std::numeric_limits<double>::infinity();
		td[i].best_match = 0;
		
		rc = pthread_create(&threads[i], NULL, functionMatch2, (void *)&td[i]);
		if (rc) {
			cout << "unable to create thread " << rc << " and i = " << i << endl;
		}
	}
	void* ret = NULL;
	for (int k = 0; k < NUM_THREADS; k++) {
		pthread_join(threads[k], &ret);
		//cout << "thread joined " << endl;
		//cout << "td[k].is_inlier = " << td[k].is_inlier << endl; 
		if ( td[k].is_inlier == 1 ) {
			
			int n2_id = td[k].descriptor_n2_id;
			double d1 = td[k].lowest_distance;
			double the_best_match = td[k].best_match;
			
			//cout << "n2_id = " << n2_id << endl;
			//cout << "d1 = " << d1 << endl;
			//cout << "the_best_match = " << the_best_match << endl;
			
			if ( matches.at<double>(0, the_best_match) == -1 ) {
				matches.at<double>(0, the_best_match) = d1; // Update minimum distance
				matches.at<double>(1, the_best_match) = n2_id; // Update minimum distance
				
			}
			else if ( d1 <  matches.at<double>(0, the_best_match) ) {
				matches.at<double>(0, the_best_match) = d1;
				matches.at<double>(1, the_best_match) = n2_id;
			}
			
			//cout << "matches = " << matches << endl;
			//waitKey(0);
		
		}
	}

	
	
	return matches;
	
}

/*
 * This function matches keypoints by looking at multiple descriptors for keypoints.
*/
void *functionMatchAdvanced(void *threadarg) {
   struct thread_match *my_data;
   my_data = (struct thread_match *) threadarg;
   
   //cout << "start of thread " << endl;
   
   double min_d1 = std::numeric_limits<double>::infinity();
   double match_d1 = std::numeric_limits<double>::infinity();
   
   double min_d2 = std::numeric_limits<double>::infinity();
   double match_d2 = std::numeric_limits<double>::infinity();
   
   int nr_descriptors = my_data->Advan_descriptors_n1.rows;
   
   //cout << "Mistake here " << endl;
   
   //cout << "min_d1, match_d1, min_d2, match_d2 = " << min_d1 << "," <<  match_d1 << "," << min_d2 << "," << match_d2 << endl;
   //waitKey(0);
      
   int dimension = my_data->Advan_descriptors_n1.cols - 1; // Since there are only 128 columns
   for (int i = 0; i < nr_descriptors; i++) {
	   double SSD = 0;
	   double SSD1 = 0; // For vector 1
	   double SSD2 = 0; // For vector 2 
	   
	   for (int k = 0; k < dimension; k++) {
		   SSD1 = SSD1 + pow(my_data->Advan_descriptors_n1.at<double>(i,k) - my_data->Advan_descriptors_n2.at<double>(0,k), 2.0);
		   SSD2 = SSD2 + pow(my_data->Advan_descriptors_n1.at<double>(i,k) - my_data->Advan_descriptors_n2.at<double>(1,k), 2.0);
	   }
	   SSD = min(SSD1,SSD2); // Choose the lowest of the SSDs
	   
	   //cout << "SSD, SSD1, SSD2 = " << SSD << "," << SSD1 << "," << SSD2 << endl;
	   
	   if (SSD < min_d1) {
		   
		   double temp_min_d1 = min_d1;
		   double temp_match_d1 = match_d1;
		   
		   
		   if (my_data->Advan_descriptors_n1.at<double>(i,128) != temp_match_d1) {
			   // Update second best
			   match_d2 = match_d1;
			   min_d2 = min_d1;
			   
			   // Update first best
			   match_d1 = my_data->Advan_descriptors_n1.at<double>(i,128);
			   min_d1 = SSD;
			   
		   }
		   else if (my_data->Advan_descriptors_n1.at<double>(i,128) == temp_match_d1) {
			   match_d1 = my_data->Advan_descriptors_n1.at<double>(i,128);
			   min_d1 = SSD;
			   
		   }
		
	   }
	   else if (SSD < min_d2) {
		   if ( my_data->Advan_descriptors_n1.at<double>(i,128) != match_d1) {
			   match_d2 = my_data->Advan_descriptors_n1.at<double>(i,128);
			   min_d2 = SSD;
		   }
	   }
   }
   
   /*
   cout << "min_d1, min_d2 = " << min_d1 << "," << min_d2 << endl;
   cout << "match_d1, match_d2 = " << match_d1 << "," << match_d2 << endl;
   waitKey(0);
   */

  //cout << "Mistake here 2 " << endl;
   // Make 0.8 a variable that can be tuned 
   if ( min_d2 != std::numeric_limits<double>::infinity()) {
	   if (min_d1/min_d2 < 0.8) {
		   my_data->Advan_is_inlier = 1;
		   my_data->Advan_lowest_distance = min_d1;
		   my_data->Advan_best_match = match_d1;
	   }
   }
   else {
	   my_data->Advan_is_inlier = 0;
	   my_data->Advan_lowest_distance = std::numeric_limits<double>::infinity();
	   my_data->Advan_best_match = 0;
   }
   
	//cout << "my_data->Advan_is_inlier, my_data->Advan_lowest_distance, my_data->Advan_best_match = " <<  my_data->Advan_is_inlier << "," << my_data->Advan_lowest_distance << "," << my_data->Advan_best_match << endl;
   
   pthread_exit(NULL);
}




// Match SIFT Descriptors
/* Objective: Function that matches keypoints in frame 2 to keypoints in frame 1 using the descriptors
 * for the keypoints in frame 2 and the descriptors for the keypoints in frame 1
 * This function uses parallel threading and can take more than just one descriptor for each keypoint
 */
Mat SIFT::matchDescriptorsAdvanced(Mat descriptor1, Mat descriptor2) {

	int k1 = descriptor1.rows;
	int k2 = descriptor2.rows;
	
	int n1 = descriptor1.at<double>(k1-1, 129-1);	// Number of keypoints in image 0
	int n2 = descriptor2.at<double>(k2-1, 129-1);	// Number of keypoints in image 1 
	
	
	Mat matches = Mat::ones(2, n1, CV_64FC1);
	matches = (-1.0)*matches;
	
	//cout << "matches = " << matches << endl;
		
	//cout << "descriptor2 = " << descriptor2.rowRange(0,2) << endl;
	//waitKey(0);

	int NUM_THREADS = n2;
	//int NUM_THREADS = 1;
	pthread_t threads[NUM_THREADS];
	struct thread_match td[NUM_THREADS];
	int i, rc;
	
	for (i = 0; i < NUM_THREADS; i++) {
		double id = descriptor2.at<double>(i*2,128);		
		td[i].Advan_descriptor_n2_id = id;
		
		//cout << "td[i].Advan_descriptor_n2_id = " << td[i].Advan_descriptor_n2_id << endl;
		//waitKey(0);
		
		descriptor1.copyTo(td[i].Advan_descriptors_n1);
		//cout << "td[i].Advan_descriptors_n1 = " << td[i].Advan_descriptors_n1 << endl;
		//waitKey(0);
		
		Mat vector = descriptor2.rowRange(i*2,(i+1)*2);
		vector.copyTo(td[i].Advan_descriptors_n2); 
		
		//cout << "td[i].Advan_descriptors_n2 = " << td[i].Advan_descriptors_n2 << endl;
		//waitKey(0);
		
		td[i].Advan_is_inlier = 0;
		td[i].Advan_lowest_distance = std::numeric_limits<double>::infinity();
		td[i].Advan_best_match = std::numeric_limits<double>::infinity();
		
		
		rc = pthread_create(&threads[i], NULL, functionMatchAdvanced, (void *)&td[i]);
		if (rc) {
			cout << "unable to create thread " << rc << " and i = " << i << endl;
		}
	}
	

	void* ret = NULL;
	for (int k = 0; k < NUM_THREADS; k++) {
		pthread_join(threads[k], &ret);
	

		if ( td[k].Advan_is_inlier == 1 ) {

			double d1 = td[k].Advan_lowest_distance;
			double match_n1 = td[k].Advan_best_match;
			double n2_index = td[k].Advan_descriptor_n2_id;

		
			if ( matches.at<double>(0, match_n1) == -1 ) {
				matches.at<double>(0, match_n1) = d1;
				matches.at<double>(1, match_n1) = n2_index;
				//cout << "Got to this point" << endl;
			}
			else if ( d1 < matches.at<double>(0, match_n1) ) {
				matches.at<double>(0, match_n1) = d1;
				matches.at<double>(1, match_n1) = n2_index;
			}

			
		}
		
	}

	//Mat valid_matches = matches.row(1);
	
	
	return matches;
}





// New attempt to write code 
/* This funciton works as of 27.7-2020
 * This Function uses parallelthreading - but only considers one descriptor for every keypoint

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
// Find SIFT Desriptors  with parallelization
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
		//keypoints.colRange(i,i+1).copyTo( td[i].thread_interest_point );

		// Optimize by now sending both grad_x and grad_y but just gradients in general
		td[i].thread_grad_x = grad_x;
		//grad_x.copyTo( td[i].thread_grad_x );
		
		td[i].thread_grad_y = grad_y;
		//grad_y.copyTo( td[i].thread_grad_y );
	
		Mat vector = Descriptors.row(i);
		td[i].thread_descriptor_vector = Descriptors.row(i);
		//vector.copyTo(td[i].thread_descriptor_vector);
		
		td[i].thread_Gauss_Window = GaussWindow;
		//GaussWindow.copyTo( td[i].thread_Gauss_Window  );
		
		rc = pthread_create(&threads[i], NULL, functionDescriptor, (void *)&td[i]);
	}
	void* ret = NULL;
	vector<Mat> keypoint_container;
	for (int k = 0; k < NUM_THREADS; k++) {
		pthread_join(threads[k], &ret);
	}
	
	
	return Descriptors;
}

// Advanced Find descriptors
 
void *functionAdvancedDescriptor(void *threadarg) {
   struct thread_descriptorAdvanced *my_data;
   my_data = (struct thread_descriptorAdvanced *) threadarg;
   
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
		
		//cout << "Patch_of_HOGs = " << Patch_of_HOGs << endl;
		
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
		
		
		// Only possible with 2 descriptors for each keypoint
		Mat multiple_descriptors = Mat::zeros(2, 129, CV_64FC1);

	
		int nr_desciptors = 0;
		
		Mat vector = Patch_of_HOGs.reshape(0, 1);;
		double SumOfSquares = 0; 
		for (int ii = 0; ii < vector.cols; ii++) {
				SumOfSquares = SumOfSquares + vector.at<double>(0,ii)*vector.at<double>(0,ii);
		}
		
		// Find maximum number of totalHOG 
		int max_index_1 = 0;
		double max_value_1 = 0;
		for (int i = 0; i < total_HOG.cols; i++) {
			if ( total_HOG.at<double>(0,i) > max_value_1 ) {
				max_value_1 = total_HOG.at<double>(0,i);
				max_index_1 = i;
			}
		}
		total_HOG.at<double>(0, max_index_1) = 0;
		
		if ( max_index_1 == 0 ) {
			total_HOG.at<double>(0, 1) = 0;
			total_HOG.at<double>(0, 7) = 0;
		}
		else if ( max_index_1 == 7 ) {
			total_HOG.at<double>(0, 6) = 0;
			total_HOG.at<double>(0, 1) = 0;
		}
		else {
			total_HOG.at<double>(0, max_index_1+1) = 0;
			total_HOG.at<double>(0, max_index_1-1) = 0;
		}
		
		// Find second maxmum number of totalHOG
		int max_index_2 = 0;
		double max_value_2 = 0;
		for (int i = 0; i < total_HOG.cols; i++) {
			if ( total_HOG.at<double>(0,i) > max_value_2 ) {
				max_value_2 = total_HOG.at<double>(0,i);
				max_index_2 = i;
			}
		}
		
	
		
		// First maximum
		Mat temp1, temp2, temp3, descrip; 
		if ( max_index_1 == 0 ) {
			max_index_1++;
		}
		temp1 = Patch_of_HOGs.colRange(0,max_index_1);
		temp2 = Patch_of_HOGs.colRange(max_index_1,8);
		temp3;
		hconcat(temp2, temp1, temp3);
		descrip = temp3.reshape(0, 1);
					//cout << "descrip = " << descrip << endl;
		for (int ii = 0; ii < descrip.cols; ii++) {
			multiple_descriptors.at<double>(nr_desciptors,ii) = descrip.at<double>(0,ii)/sqrt(SumOfSquares);
		}
		multiple_descriptors.at<double>(0,128) = my_data->thread_descriptor_id;
		nr_desciptors++;
		
		
		// Second maximum
		if ( max_index_2 == 0 ) {
			max_index_2++;
		}
		temp1 = Patch_of_HOGs.colRange(0,max_index_2);
		temp2 = Patch_of_HOGs.colRange(max_index_2,8);
		temp3;
		hconcat(temp2, temp1, temp3);
		descrip = temp3.reshape(0, 1);
					//cout << "descrip = " << descrip << endl;
		for (int ii = 0; ii < descrip.cols; ii++) {
			multiple_descriptors.at<double>(nr_desciptors,ii) = descrip.at<double>(0,ii)/sqrt(SumOfSquares);
		}
		multiple_descriptors.at<double>(1,128) = my_data->thread_descriptor_id;
	
		
			
		my_data->thread_multiple_descriptors = multiple_descriptors;
   
   pthread_exit(NULL);
}



// This function works - 25.07-2020
// Find SIFT Desriptors  without parallelization
Mat SIFT::FindDescriptorsAdvanced(Mat src_gray, Mat keypoints) {
	
	Mat src_gray_blurred;
	GaussianBlur(src_gray, src_gray_blurred, Size(3,3), 5,5);

	// Simplification of SIFT
	// Maybe the image should be smoothed first with a Gaussian Kernel
	int n = keypoints.cols;
	
	// Initialize matrix containing keypoints descriptors
	//Matrix Descriptors(n,128);
	//Mat Descriptors = Mat::zeros(n, 128, CV_64FC1);
	
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
	//int NUM_THREADS = 1;
	pthread_t threads[NUM_THREADS];
	struct thread_descriptorAdvanced td[NUM_THREADS];
	int i, rc;
	for (i = 0; i < NUM_THREADS; i++) {
		td[i].thread_interest_point = keypoints.colRange(i,i+1);
		td[i].thread_descriptor_id = i;

		td[i].thread_grad_x = grad_x;

		td[i].thread_grad_y = grad_y;
		
		td[i].thread_Gauss_Window = GaussWindow;
		
		rc = pthread_create(&threads[i], NULL, functionAdvancedDescriptor, (void *)&td[i]);
	}
	void* ret = NULL;
	vector<Mat> descriptor_container;
	for (int k = 0; k < NUM_THREADS; k++) {
		pthread_join(threads[k], &ret);
		descriptor_container.push_back(td[k].thread_multiple_descriptors);
		
	}
	Mat Descriptors;
	vconcat(descriptor_container, Descriptors);
	
	
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
	
	/**
	% Copyright (c) 2011, Laurent Kneip, ETH Zurich
	% All rights reserved.
	% 
	% Redistribution and use in source and binary forms, with or without
	% modification, are permitted provided that the following conditions are met:
	%     * Redistributions of source code must retain the above copyright
	%       notice, this list of conditions and the following disclaimer.
	%     * Redistributions in binary form must reproduce the above copyright
	%       notice, this list of conditions and the following disclaimer in the
	%       documentation and/or other materials provided with the distribution.
	%     * Neither the name of ETH Zurich nor the
	%       names of its contributors may be used to endorse or promote products
	%       derived from this software without specific prior written permission.
	% 
	% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	% DISCLAIMED. IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY
	% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	**/
	
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
	
	/*
	std::complex<double> i_value;
	i_value = 1i;
	*/
	std::complex<double> i_value(0.0, 1.0);
	//_value = 1i;

	
	//std::complex<double> R, U, y, w, null_v;
	std::complex<double> U, y, w;
	//std::complex<double> R, U, y, w;
	double real_value, imaginary_value;
	if (pow(Q,2.0)/4 + pow(P,3.0)/27 < 0) {
		imaginary_value = sqrt(-(pow(Q,2.0)/4 + pow(P,3.0)/27));
		real_value = (-Q/2.0);
		//R = real_value + imaginary_value*i_value;
		std::complex<double> R(real_value, imaginary_value);
		U = pow(R, 1.0/3.0);
	}
	else {
		real_value = -Q/2.0 + sqrt(pow(Q,2.0)/4 + pow(P,3.0)/27);
		//R = real_value + 0i; 
		std::complex<double> R(real_value,0.0);
		U = pow(R, 1.0/3.0);
	}
	//U = pow(R, 1.0/3.0);
	
	
	//null_v = 0. + 0i;
	std::complex<double> null_v(0.0,0.0);
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
	
	/*
	if (isnan(roots.at<double>(0,0)) || isnan(roots.at<double>(0,1)) || isnan(roots.at<double>(0,2)) || isnan(roots.at<double>(0,3))) {
		//cout << "Roots = " << roots.at<double>(0,0) << ", " << roots.at<double>(0,1) << ", " << roots.at<double>(0,2) << ", " << roots.at<double>(0,3) << endl;
		//waitKey(0);
		//cout << "Factors = " << A << "," << B << "," << C << "," << D << "," << E << endl;
		//waitKey(0);
	}
	*/
	
	//Mat roots = Mat::zeros(1, 4, CV_64FC1);
	//cout << "Roots = " << roots.at<double>(0,0) << ", " << roots.at<double>(0,1) << ", " << roots.at<double>(0,2) << ", " << roots.at<double>(0,3) << endl;
	//cout << "a = " << real(a) << endl;
	//cout << "b = " << real(b) << endl;
	return roots;
}

 
Mat p3p(Mat worldPoints, Mat imageVectors) {
	
	/**
	% Copyright (c) 2011, Laurent Kneip, ETH Zurich
	% All rights reserved.
	% 
	% Redistribution and use in source and binary forms, with or without
	% modification, are permitted provided that the following conditions are met:
	%     * Redistributions of source code must retain the above copyright
	%       notice, this list of conditions and the following disclaimer.
	%     * Redistributions in binary form must reproduce the above copyright
	%       notice, this list of conditions and the following disclaimer in the
	%       documentation and/or other materials provided with the distribution.
	%     * Neither the name of ETH Zurich nor the
	%       names of its contributors may be used to endorse or promote products
	%       derived from this software without specific prior written permission.
	% 
	% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	% DISCLAIMED. IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY
	% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	%
	%
	% p3p.m
	%
	%
	%      Author: Laurent Kneip
	% Description: Compute the absolute pose of a camera using three 3D-to-2D correspondences
	%   Reference: A Novel Parametrization of the P3P-Problem for a Direct Computation of
	%              Absolute Camera Position and Orientation
	%
	%       Input: worldPoints: 3x3 matrix with corresponding 3D world points (each column is a point)
	%              imageVectors: 3x3 matrix with UNITARY feature vectors (each column is a vector)
	%      Output: poses: 3x16 matrix that will contain the solutions
	%                     form: [ 3x1 position(solution1) 3x3 orientation(solution1) 3x1 position(solution2) 3x3 orientation(solution2) ... ]
	%                     the obtained orientation matrices are defined as transforming points from the cam to the world frame
	**/
	
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
		
		//std::complex<double> cos_theta = x.at<double>(0,i) + 0i;
		std::complex<double> cos_theta(x.at<double>(0,i),0.0);
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
	double k = 3;
	int min_inlier_count = ransac_min_inlier_count; // This parameter should be tuned for the implementation
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
	
	/*
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
	
	
	Mat choosen_idx = Mat::zeros(1, 3, CV_64FC1);
	choosen_idx.at<double>(0,0) = 182-1;
	choosen_idx.at<double>(0,1) = 52-1;
	choosen_idx.at<double>(0,2) = 100-1;
	*/
	
	/*
	 * 
	 * 
	 *Make a new method for extracting random numbers 
	 * 
	 * 
	 * 
	 */

	
	while ( num_iterations-1 > i ) {
	//while (1 > i) {
		
		std::vector<int> random_idx = pick( corresponding_landmarks.cols +1 , 3);
		//cout << "random_idx = " << random_idx[0] << ", " << random_idx[1] << ", " << random_idx[2] << endl;
		Mat random_selected_idx = Mat::zeros(1, 3, CV_64FC1);
		random_selected_idx.at<double>(0,0) = random_idx[0]-1;
		random_selected_idx.at<double>(0,1) = random_idx[1]-1;
		random_selected_idx.at<double>(0,2) = random_idx[2]-1;
		//cout << "random_selected_idx = " << random_selected_idx << endl;
		
		/*
		 * // Previous random data sample
		int random_nums[corresponding_landmarks.cols];
		for (int mm = 0; mm < corresponding_landmarks.cols; mm++) {
			random_nums[mm] = mm;
		}
		random_shuffle(random_nums, random_nums + corresponding_landmarks.cols);
		cout << "random_nums[1] = " << random_nums[1] << endl;
		cout << "random_nums[2] = " << random_nums[2] << endl;
		cout << "random_nums[3] = " << random_nums[3] << endl;
		*/
		
		
		 // The real code
		for (int mm = 0; mm < k;  mm++) {
			//cout << "Random number = " << random_nums[mm] << endl;
			// Landmark sample 
			landmark_sample.at<double>(0,mm) = corresponding_landmarks.at<double>(0, random_selected_idx.at<double>(0,mm));
			landmark_sample.at<double>(1,mm) = corresponding_landmarks.at<double>(1, random_selected_idx.at<double>(0,mm));
			landmark_sample.at<double>(2,mm) = corresponding_landmarks.at<double>(2, random_selected_idx.at<double>(0,mm));
			
			// Keypoint sample 
			keypoint_sample.at<double>(0,mm) = matched_query_keypoints.at<double>(0, random_selected_idx.at<double>(0,mm));
			keypoint_sample.at<double>(1,mm) = matched_query_keypoints.at<double>(1, random_selected_idx.at<double>(0,mm));
		}
		
		//cout << "landmark_sample = " << landmark_sample << endl;
		//cout << "keypoint_sample = " << keypoint_sample << endl;
		
		/*
		// Not random
		for (int mm = 0; mm < 3;  mm++) {
			//cout << "Random number = " << random_nums[mm] << endl;
			// Landmark sample 
			landmark_sample.at<double>(0,mm) = corresponding_landmarks.at<double>(0, choosen_idx.at<double>(0,mm));
			landmark_sample.at<double>(1,mm) = corresponding_landmarks.at<double>(1, choosen_idx.at<double>(0,mm));
			landmark_sample.at<double>(2,mm) = corresponding_landmarks.at<double>(2, choosen_idx.at<double>(0,mm));
			
			// Keypoint sample 
			keypoint_sample.at<double>(0,mm) = matched_query_keypoints.at<double>(0, choosen_idx.at<double>(0,mm));
			keypoint_sample.at<double>(1,mm) = matched_query_keypoints.at<double>(1, choosen_idx.at<double>(0,mm));
		}
		*/
		
		//cout << "landmark_sample = " << landmark_sample << endl;
		//cout << "keypoint_sample = " << keypoint_sample << endl;
		
		normalized_bearings = K.inv() * keypoint_sample;
		
		for (int ii = 0; ii < 3; ii++) {
			double vector_norm = sqrt(pow(normalized_bearings.at<double>(0,ii),2.0) + pow(normalized_bearings.at<double>(1,ii),2.0) + pow(normalized_bearings.at<double>(2,ii),2.0));
			normalized_bearings.at<double>(0,ii) = normalized_bearings.at<double>(0,ii)/vector_norm;
			normalized_bearings.at<double>(1,ii) = normalized_bearings.at<double>(1,ii)/vector_norm;
			normalized_bearings.at<double>(2,ii) = normalized_bearings.at<double>(2,ii)/vector_norm;
		
		}
		//cout << "normalized_bearings = " << normalized_bearings << endl;
		
		
		poses = p3p(landmark_sample, normalized_bearings);
	
		//cout << "poses = " << poses << endl;

		
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
		
		//cout << "R_C_W_guess = " << R_C_W_guess << endl;
		//cout << "t_C_W_guess = " << t_C_W_guess << endl;
				
		
		points = R_C_W_guess * corresponding_landmarks + repeat(t_C_W_guess, 1, corresponding_landmarks.cols);
		
		//cout << "points = " << points << endl;
		
		projected_points = projectPoints(points, K);
		
		//cout << "projected_points = " << projected_points << endl;
		
		difference = matched_query_keypoints - projected_points;
		
		//cout << "difference = " << difference << endl;

		errors = difference.mul(difference);
		
		errors = errors.row(0) + errors.row(1);
		
		//cout << "errors = " << errors << endl;

		is_inlier = errors < pow(pixel_tolerance,2.0); // Remember this matrix is of type uchar 
		
		//cout << "is_inlier = " << is_inlier << endl;
		
		//cout << "countNonZero(is_inlier) = " << countNonZero(is_inlier)  << endl;

		//cout << "First iter" << endl;
		//cout << "countNonZero(is_inlier) = " << countNonZero(is_inlier) << endl;
		//waitKey(5000);
		if (countNonZero(is_inlier) > record_inlier && countNonZero(is_inlier) >= min_inlier_count) {
			//cout << "inside first record_inlier update " << endl;
			int record_int = countNonZero(is_inlier);
			record_inlier = record_int;
			R_C_W_guess.copyTo(best_R_C_W);
			t_C_W_guess.copyTo(best_t_C_W);
			is_inlier.copyTo(best_inlier_mask);
			
			//cout << "best_R_C_W = " << best_R_C_W << endl;
			//cout << "best_t_C_W = " << best_t_C_W << endl;
			//waitKey(0);
		}
		
		for (int alt_idx = 1; alt_idx <= 3; alt_idx++) {
			for (int k = 0; k < 3; k++) {
				R_W_C.at<double>(0,k) = poses.at<double>(0,k+1 + alt_idx*4);
				R_W_C.at<double>(1,k) = poses.at<double>(1,k+1 + alt_idx*4);
				R_W_C.at<double>(2,k) = poses.at<double>(2,k+1 + alt_idx*4);
				t_W_C.at<double>(k,0) = poses.at<double>(k, alt_idx*4);
			}
			
			R_C_W_guess = R_W_C.t();
			
			//cout << "alt_idx = " << alt_idx << endl;
			
			//cout << "R_C_W_guess = " << R_C_W_guess << endl;

			t_C_W_guess = -R_W_C.t()*t_W_C;
			
			//cout << "t_C_W_guess = " << t_C_W_guess << endl;
			
			points = R_C_W_guess * corresponding_landmarks + repeat(t_C_W_guess, 1, corresponding_landmarks.cols);
			
			//cout << "points = " << points << endl;
			
			projected_points = projectPoints(points, K);
			
			//cout << "projected_points = " << projected_points << endl;
			
			difference = matched_query_keypoints - projected_points;
			
			//cout << "difference = " << difference << endl;
			
			errors = difference.mul(difference);
			errors = errors.row(0) + errors.row(1);
			
			//cout << "errors = " << errors << endl;
			
			alternative_is_inlier = errors < pow(pixel_tolerance,2.0);
			
			//cout << "alternative_is_inlier = " << alternative_is_inlier << endl;
			
			//cout << "countNonZero(alternative_is_inlier) = " << countNonZero(alternative_is_inlier) << endl;

			if (countNonZero(alternative_is_inlier) > countNonZero(is_inlier) ) {
				//is_inlier = alternative_is_inlier;
				alternative_is_inlier.copyTo(is_inlier);
			} 


			if (countNonZero(is_inlier) > record_inlier && countNonZero(is_inlier) >= min_inlier_count) {
				record_inlier = countNonZero(is_inlier);
				is_inlier.copyTo(best_inlier_mask);
				R_C_W_guess.copyTo(best_R_C_W);
				t_C_W_guess.copyTo(best_t_C_W);

			}
			
		}
		
		//cout << "record_inlier = " << record_inlier << endl;
		
		/*
		if (countNonZero(is_inlier) > max_num_inliers && countNonZero(is_inlier) >= min_inlier_count) {
			max_num_inliers = countNonZero(is_inlier);
			//best_inlier_mask = is_inlier;
			is_inlier.copyTo(best_inlier_mask);
		}
		*/
		
		if (adaptive_ransac) {
			float division = (float) record_inlier/ (float) is_inlier.cols;
			//cout << "division = " << division << endl;
			float outlier_ratio = 1. - division;
			
			//cout << "outlier_ratio = " << outlier_ratio << endl;
			
			float confidence = 0.95; 
			float upper_bound_on_outlier_ratio = 0.90;
			outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio);
			num_iterations = log( 1 - confidence)/log(1-pow((1-outlier_ratio),k)); 
			
			//cout << "num_iterations = " << num_iterations << endl;
			
			//cout << "num_iterations = " << num_iterations << endl;
			
			double v = 15000;
			num_iterations = min(v, num_iterations);
			
			//cout << "num_iterations = " << num_iterations << endl;
			//cout << "num iterations after min-operation = " << num_iterations << endl;
			//waitKey(0);
		}
		
		i++;
	}	
	/*
	cout << "best_R_C_W" << endl;
	cout << best_R_C_W << endl;
	cout << "best_t_C_W" << endl;
	ćout << best_t_C_W << endl;
	*/
	
	if (record_inlier > min_inlier_count) {
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
	
	int run_code_on_drone = show_results;
	double corner_strengh = Harris_Corner_strengh;
	
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
	goodFeaturesToTrack(Ii_resized, temp1, keypoint_max, 0.01, 5, noArray(), 3, true, corner_strengh);
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
	
	//cout << "Number of New candidate Keypoints" << endl;
	
	// Update keypoints
	keypoints_Ii.copyTo(Si.Ci); 
	
	if ( run_code_on_drone == 0) {
		// Draw keypoints Si.Pi in red
		Mat Ii_draw;
		Ii.copyTo(Ii_draw);
		for (int i = 0; i < Si.Pi.cols; i++) {
			circle (Ii_draw, Point(Si.Pi.at<double>(1,i),Si.Pi.at<double>(0,i)), 5,  Scalar(0,0,255), 2,8,0);
		}
		// Draw Candidate keypoints Si.Ci in blue
		for (int i = 0; i < Si.Ci.cols; i++) {
			circle (Ii_draw, Point(Si.Ci.at<double>(1,i),Si.Ci.at<double>(0,i)), 5,  Scalar(255,0,0), 2,8,0);
		}
		imshow("newCandidateKeypoints Ii Si.Ci", Ii_draw);
		waitKey(0);
	}
	else {
		//usleep(2000000);
		for (int i = 0; i < Si.Pi.cols; i++) {
			circle (Ii, Point(Si.Pi.at<double>(1,i),Si.Pi.at<double>(0,i)), 5,  Scalar(0,0,255), 2,8,0);
		}
		for (int i = 0; i < Si.Ci.cols; i++) {
			circle (Ii, Point(Si.Ci.at<double>(1,i),Si.Ci.at<double>(0,i)), 5,  Scalar(255,0,0), 2,8,0);
		}
	}
	
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
	
	int run_code_on_drone = show_results;
	
	// Variables used for for loops
	int i, q;
	
	double corner_strengh = Harris_Corner_strengh;
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
	goodFeaturesToTrack(Ii_resized, temp1, 300, 0.01, 5, noArray(), 3, true, corner_strengh);
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
	
	// Drawing depends on whether the code is tested on the quadcopter or not 
	if ( run_code_on_drone == 0 ) {
		// Draw keypoints Si.Pi in red 
		Mat Ii_draw;
		Ii.copyTo(Ii_draw);
		for (int i = 0; i < Si.Pi.cols; i++) {
			circle (Ii_draw, Point(Si.Pi.at<double>(1,i),Si.Pi.at<double>(0,i)), 5,  Scalar(0,0,255), 2,8,0);
		}
		// Draw Candidate keypoints Si.Ci in blue
		for (int i = 0; i < Si.Ci.cols; i++) {
			circle (Ii_draw, Point(Si.Ci.at<double>(1,i),Si.Ci.at<double>(0,i)), 5,  Scalar(255,0,0), 2,8,0);
		}
		// Draw first occurence of Si.Fi in green
		for (int i = 0; i < Si.Fi.cols; i++) {
			circle (Ii_draw, Point(Si.Fi.at<double>(1,i),Si.Fi.at<double>(0,i)), 5,  Scalar(0,255,0), 2,8,0);
		}
		imshow("continuousCandidateKeypoints Ii Si.Ci", Ii_draw);
		waitKey(0);
	}
	else {
		for (int i = 0; i < Si.Pi.cols; i++) {
			circle (Ii, Point(Si.Pi.at<double>(1,i),Si.Pi.at<double>(0,i)), 5,  Scalar(0,0,255), 2,8,0);
		}
		// Draw Candidate keypoints Si.Ci
		for (int i = 0; i < Si.Ci.cols; i++) {
			circle (Ii, Point(Si.Ci.at<double>(1,i),Si.Ci.at<double>(0,i)), 5,  Scalar(255,0,0), 2,8,0);
		}
	}

	if (Si.num_candidates < num_candidate_keypoints) {
		
		// Number of new candidate keypoints that should be found 
		int n = num_candidate_keypoints - Si.num_candidates; 
		
		// Start by ignorring current candidate keypoints
		for (i = 0; i < Si.num_candidates; i++) {
			checkImage.at<double>(Si.Ci.at<double>(0,i),Si.Ci.at<double>(1,i)) = 1;
		}
		
		int k = Si.k + Si.num_candidates + n;
		Mat temp2;
		goodFeaturesToTrack(Ii_resized, temp2, k, 0.01, 5, noArray(), 3, true, 0.04);
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
		
		Mat temp_Si_Ci, temp_Si_Fi, temp_Si_Ti;
		Si.Ci.copyTo(temp_Si_Ci);
		Si.Fi.copyTo(temp_Si_Fi);
		
		// Concatenate the newly founded keypoints
		hconcat(temp_Si_Ci, Potential_new_CI.colRange(0,temp_index), Si.Ci);
		hconcat(temp_Si_Fi, Potential_new_CI.colRange(0,temp_index), Si.Fi);
		
		
		Mat t_C_W_vector = T_wc.reshape(0,T_wc.rows * T_wc.cols);
		Mat poses = repeat(t_C_W_vector, 1, temp_index);
		
		Si.Ti.copyTo(temp_Si_Ti);
		
		// Concatenate the current poses
		hconcat(temp_Si_Ti, poses, Si.Ti);
		
		
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
	
	cout << "inside findLandmark" << endl;
	
	Mat P;
	Mat Q;
	
	Mat M0 = K*tau;
	Mat M1 = K*T_WC;
	Mat v1 = M0.row(0) - imagepoint0.at<double>(0,0) * M0.row(2);
	
	Mat v2 = M0.row(1) - imagepoint0.at<double>(1,0) * M0.row(2);
	
	vconcat(v1, v2, Q);
	v1 = M1.row(0) - imagepoint1.at<double>(0,0) * M1.row(2);
	vconcat(v1, Q, Q);
	v2 = M1.row(1) - imagepoint1.at<double>(1,0) * M1.row(2);
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
state triangulateNewLandmarks(Mat I, state Si, Mat K, Mat T_WC, double threshold_angle) {
	
	cout << "triangulateNewLandmarks " << endl;
	
	// Check if points are ready to be traingulated 
	Mat extracted_keypoints = Mat::zeros(1, Si.num_candidates, CV_64FC1);
	
	// Matrices to store valid extracted keypoints
	Mat newKeypoints = Mat::zeros(2, Si.num_candidates, CV_64FC1);
	Mat triangulated_landmark;
	Mat newLandmarks = Mat::zeros(3, Si.num_candidates, CV_64FC1);
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
	//cout << "Si.num_candidates = " << Si.num_candidates << endl;
	
	if ( show_results == 0 ) {
		cout << "Before Si.Pi = " << Si.Pi << endl;
		cout << "Before Si.Ci = " << Si.Ci << endl;
		cout << "Before Si.Fi = " << Si.Fi << endl;
	}
	
	Mat I_draw_Fi, I_draw_Ci;
	I.copyTo(I_draw_Fi);
	I.copyTo(I_draw_Ci);
	
	for (i = 0; i < Si.num_candidates; i++) {
		// First occurrence of keypoint
		keypoint_last_occur.at<double>(0,0) = Si.Fi.at<double>(1,i); // x-coordinate --> Make it u
		keypoint_last_occur.at<double>(1,0) = Si.Fi.at<double>(0,i); // y-coordinate --> Make it v 
		
		if ( show_results == 0 ) {
			cout << "Si.Fi(y,x) = (" << Si.Fi.at<double>(0,i) << "," << Si.Fi.at<double>(1,i) << ")" << endl;
			circle (I_draw_Fi, Point(Si.Fi.at<double>(1,i),Si.Fi.at<double>(0,i)), 5,  Scalar(0,0,255), 2,5,0);
			imshow("Si.Fi", I_draw_Fi);
			waitKey(0);
		}

		// Newest occurrence of keypoint
		keypoint_newest_occcur.at<double>(0,0) = Si.Ci.at<double>(1,i); // x-coordinate --> Make it u 
		keypoint_newest_occcur.at<double>(1,0) = Si.Ci.at<double>(0,i); // y--coordinate --> Make it v
		
		if ( show_results == 0 ) {
			cout << "Si.Ci(y,x) = (" << Si.Ci.at<double>(0,i) << "," << Si.Ci.at<double>(1,i) << ")" << endl;
			circle (I_draw_Ci, Point(Si.Ci.at<double>(1,i),Si.Ci.at<double>(0,i)), 5,  Scalar(0,0,255), 2,5,0);
			imshow("Si.Ci", I_draw_Ci);
			waitKey(0);
		}

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
		
		cout << "alpha = " << alpha << endl;
		if (alpha > threshold_angle) {
			extracted_keypoints.at<double>(0,i) = 1;
			//cout << "Extract landmark " << endl;
			//cout << "extracted_keypoints.at<double>(0,i) = " << extracted_keypoints.at<double>(0,i) << endl;
			
			// Extract tau_vector and reshape it to a matrix
			//cout << "Si.Ti = " << Si.Ti << endl;
			//cout << "i = " << i << endl;
			Si.Ti.col(i).copyTo(tau);
			//cout << "tau = " << tau << endl;
			tau = tau.reshape(0, 3);
			
			// Update keypoints 
			newKeypoints.at<double>(0,temp) = Si.Ci.at<double>(0,i);
			newKeypoints.at<double>(1,temp) = Si.Ci.at<double>(1,i);
			
			//cout << "tau = " << tau << endl;
			
			triangulated_landmark = findLandmark(K, tau, T_WC, keypoint_last_occur, keypoint_newest_occcur); // Check if tau should be changed to matrix
			
			//cout << "traingulated_landmark = " << triangulated_landmark << endl;
			
			/*
			triangulated_landmark = triangulated_landmark.rowRange(0,3);
			cout << "traingulated_landmark = " << traingulated_landmark << endl;
			traingulated_landmark.copyTo(newLandmarks.col(temp));
			*/
			
			newLandmarks.at<double>(0,temp) = triangulated_landmark.at<double>(0,0);
			newLandmarks.at<double>(1,temp) = triangulated_landmark.at<double>(1,0);
			newLandmarks.at<double>(2,temp) = triangulated_landmark.at<double>(2,0);
			
			//cout << "newLandmarks = " << newLandmarks << endl;
			
			temp++;
		}
	}
	
	//cout << "temp = " << temp << endl;
	//cout << "newLandmarks = " << newLandmarks << endl;
	//cout << "newKeypoints = " << newKeypoints << endl;
	
	// Append new keypoints and new landmarks 
	if (temp > 0) {
		//cout << "concatenate " << endl;
		newKeypoints = newKeypoints.colRange(0,temp);
		newLandmarks = newLandmarks.colRange(0,temp);
	
		// Append new keypoints
		if (Si.Pi.cols == 0) {
			newKeypoints.copyTo( Si.Pi );
		}
		else {
			hconcat(Si.Pi, newKeypoints, Si.Pi);
		}
		
		// Append new 3D landmarks 
		if (Si.Xi.cols == 0) {
			newLandmarks.copyTo(Si.Xi);
		}
		else {
			hconcat(Si.Xi, newLandmarks, Si.Xi);
		 }
	}
	
	//cout << "Si.Pi = " << Si.Pi << endl;
	//cout << "Si.Xi = " << Si.Xi << endl;
	
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
	
	
	cout << "After Si.Pi = " << Si.Pi << endl;
	cout << "AFter Si.Ci = " << Si.Ci << endl;
	cout << "AFter Si.Fi = " << Si.Fi << endl;
	
	//return make_tuple(Si, extracted_keypoints);
	return Si;
}

// ############################# VO Initialization Pipeline #############################
tuple<state, Mat, bool> initialization(Mat I_i0, Mat I_i1, Mat K, state Si_1) {
	cout << "Begin initialization" << endl;
	
	int run_code_on_drone = show_results; // 0 : Run code on Thor's Raspberry Pi  /  1 : Run code on quadcopter
	
	double corner_strength = Harris_Corner_strengh;
	int min_inlier = min_nr_inliers_initialization;
	Mat transformation_matrix;
	bool initialization_okay;
	
	// Transform color images to gray images
	Mat I_i0_gray, I_i1_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cvtColor(I_i1, I_i1_gray, COLOR_BGR2GRAY );
	
	Mat temp0, temp1;
	int nr_interest_points = Harris_keypoints;


	
	//Mat keypoints_I_i0 = Harris::corner(I_i0, I_i0_gray, 210, emptyMatrix); // Number of maximum keypoints
	int dim1 = I_i0_gray.rows;
	int dim2 = I_i0_gray.cols;
	Mat I_i0_resized = I_i0_gray.colRange(10,dim2-10).rowRange(10,dim1-10);
		
	goodFeaturesToTrack(I_i0_resized, temp0, nr_interest_points, 0.01, 2, noArray(), 3, true, corner_strength);
	Mat keypoints_I_i0 = Mat::zeros(2, temp0.rows, CV_64FC1);
	for (int i = 0; i < keypoints_I_i0.cols; i++) {
		keypoints_I_i0.at<double>(0,i) = temp0.at<float>(i,1) + 10;
		keypoints_I_i0.at<double>(1,i) = temp0.at<float>(i,0) + 10;
	}

	if ( run_code_on_drone == 0 ) {
		Mat draw_I_i0;
		I_i0.copyTo(draw_I_i0);
		const char* text0 = "Detected corners in frame I_i0";
		drawCorners(draw_I_i0, keypoints_I_i0, text0);
		waitKey(0);
	}
	

	
	//high_resolution_clock::time_point t3 = high_resolution_clock::now();	
	
	dim1 = I_i1_gray.rows;
	dim2 = I_i1_gray.cols;
	Mat I_i1_resized = I_i1_gray.colRange(10,dim2-10).rowRange(10,dim1-10);
	goodFeaturesToTrack(I_i1_resized, temp1, nr_interest_points, 0.01, 4, noArray(), 3, true, corner_strength);
	
	Mat keypoints_I_i1 = Mat::zeros(2, temp1.rows, CV_64FC1);
	for (int i = 0; i < keypoints_I_i1.cols; i++) {
		keypoints_I_i1.at<double>(0,i) = temp1.at<float>(i,1) + 10;
		keypoints_I_i1.at<double>(1,i) = temp1.at<float>(i,0) + 10;
	}
	
	
	if ( run_code_on_drone == 0 ) {
		Mat draw_I_i1;
		I_i1.copyTo(draw_I_i1);
		const char* text1 = "Detected corners in frame I_i1";
		drawCorners(draw_I_i1, keypoints_I_i1, text1);
		waitKey(0);
	}
	
	
	


	// ######################### SIFT ######################### 
	//Finding SIFT::descriptors without parallelization 
	
	
	// cout << "Advanced method for finding keypoints " << endl;
	/*
	Mat descriptors_I_i0_Advanced = SIFT::FindDescriptorsAdvanced( I_i0_gray,  keypoints_I_i0);
	Mat descriptors_I_i1_Advanced = SIFT::FindDescriptorsAdvanced( I_i1_gray,  keypoints_I_i1);
	
	Mat matches_1_advanced = SIFT::matchDescriptorsAdvanced( descriptors_I_i0_Advanced,  descriptors_I_i1_Advanced);
	Mat matches_2_advanced = SIFT::matchDescriptorsAdvanced( descriptors_I_i1_Advanced,  descriptors_I_i0_Advanced);
	
	Mat matches = matches_1_advanced.row(1);
	Mat matches2 = matches_2_advanced.row(1);

	int temp_index;
	int temp_index_advanced = 0; 
	Mat valid_matches_advanced = Mat::zeros(2, matches.cols, CV_64FC1);
	// Fix Problem here 
	for (int i = 0; i < matches.cols; i++) {
		if ( matches.at<double>(0,i) != -1) {
			if ( matches2.at<double>(0, matches.at<double>(0,i)) == i ) {
				valid_matches_advanced.at<double>(0, temp_index_advanced) = i;
				valid_matches_advanced.at<double>(1, temp_index_advanced) = matches.at<double>(0,i);
				temp_index_advanced++;
			} 
		}
	}
	matches = valid_matches_advanced.colRange(0, temp_index_advanced);	// original
	*/
	
	// cout << "Time consuming method for finding keypoints" << endl;	
	/*
	// Time consuming 
	Mat descriptors_I_i0 = SIFT::FindDescriptors(I_i0_gray, keypoints_I_i0);
	Mat descriptors_I_i1 = SIFT::FindDescriptors(I_i1_gray, keypoints_I_i1);
	Mat matches = SIFT::matchDescriptors(descriptors_I_i0, descriptors_I_i1);
	
	// Time consuming 
	Mat matches2 = SIFT::matchDescriptors(descriptors_I_i1, descriptors_I_i0);
	
	// Not time consuming 
	Mat valid_matches = Mat::zeros(2, matches.cols, CV_64FC1);
	int temp_index = 0;
	for (int i = 0; i < matches.cols; i++) {
		int index_frame0 = matches.at<double>(0,i);
		int index_frame1 = matches.at<double>(1,i);
		
		for (int q = 0; q < matches2.cols; q++) {
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
	matches = valid_matches.colRange(0,temp_index);	// original
	*/	
	

	// Not time consuming find descriptors and match descriptors
	Mat descriptors_I_i0 = SIFT::FindDescriptors(I_i0_gray, keypoints_I_i0);
	Mat descriptors_I_i1 = SIFT::FindDescriptors(I_i1_gray, keypoints_I_i1);
	
	Mat matches = SIFT::matchDescriptors2(descriptors_I_i0, descriptors_I_i1);
	
	matches = matches.row(1);
	
	// Not Time consuming 
	Mat matches2 = SIFT::matchDescriptors2(descriptors_I_i1, descriptors_I_i0);
	
	matches2 = matches2.row(1);
	
	
	// Not time consuming 
	int temp_index;
	int temp_index_advanced = 0; 
	Mat valid_matches_advanced = Mat::zeros(2, matches.cols, CV_64FC1);
	// Fix Problem here 
	for (int i = 0; i < matches.cols; i++) {
		if ( matches.at<double>(0,i) != -1) {
			if ( matches2.at<double>(0, matches.at<double>(0,i)) == i ) {
				valid_matches_advanced.at<double>(0, temp_index_advanced) = i;
				valid_matches_advanced.at<double>(1, temp_index_advanced) = matches.at<double>(0,i);
				temp_index_advanced++;
			} 
		}
	}
	matches = valid_matches_advanced.colRange(0, temp_index_advanced);	// original
	
	/*
	high_resolution_clock::time_point t5 = high_resolution_clock::now();

	high_resolution_clock::time_point t6 = high_resolution_clock::now();
	duration<double> time_span2 = duration_cast<duration<double>>(t6-t5);
	cout << "Finding descriptors_I_i0 took = " << time_span2.count() << " seconds" << endl;
	*/
	

	// Find Point correspondences
	// Points from image 0 in row 1 and row 2 
	// Points from image 1 in row 3 and row 	

	int N = matches.cols;
	cout << "Number of matches = " << N << endl;
	
	if (N < min_inlier) {
		initialization_okay = false;
		
		return make_tuple(Si_1, transformation_matrix, initialization_okay);
	}
	
	//high_resolution_clock::time_point t11 = high_resolution_clock::now();
	
	// For plotting
	// For efficiency, you should maybe just use vectors instead of creating two new matrices
	Mat temp_points1Mat = Mat::zeros(2, N, CV_64FC1);
	Mat temp_points2Mat = Mat::zeros(2, N, CV_64FC1);
	// For fudamental matrix
	vector<Point2f> points1(N);
	vector<Point2f> points2(N);
	
	
	Mat I_i0_draw, I_i1_draw; 
	if ( run_code_on_drone == 0 ) {
		I_i0.copyTo(I_i0_draw);
		I_i1.copyTo(I_i1_draw);
	}
	
	
	for (int i = 0; i < N; i++) {
		// Be aware of differences in x and y
		
		points1[i] = Point2f(keypoints_I_i0.at<double>(0, matches.at<double>(0,i)),keypoints_I_i0.at<double>(1, matches.at<double>(0,i)));
		points2[i] = Point2f(keypoints_I_i1.at<double>(0, matches.at<double>(1,i)),keypoints_I_i1.at<double>(1, matches.at<double>(1,i)));
		
		temp_points1Mat.at<double>(0,i) = keypoints_I_i0.at<double>(0, matches.at<double>(0,i)); // y-coordinate in image 
		temp_points1Mat.at<double>(1,i) = keypoints_I_i0.at<double>(1, matches.at<double>(0,i)); // x-coordinate in image
		temp_points2Mat.at<double>(0,i) = keypoints_I_i1.at<double>(0, matches.at<double>(1,i)); // y-coordinate in image
		temp_points2Mat.at<double>(1,i) = keypoints_I_i1.at<double>(1, matches.at<double>(1,i)); // x-coordinate in image

		
		double y = keypoints_I_i1.at<double>(0, matches.at<double>(1,i));
		double x = keypoints_I_i1.at<double>(1, matches.at<double>(1,i));
		double y2 = keypoints_I_i0.at<double>(0, matches.at<double>(0,i));
		double x2 = keypoints_I_i0.at<double>(1, matches.at<double>(0,i));
		
		if (run_code_on_drone == 0) {
			line(I_i1_draw,Point(x,y),Point(x2,y2),Scalar(0,255,0),3);
			circle (I_i1_draw, Point(x,y), 5,  Scalar(0,0,255), 2,5,0);
			circle (I_i0_draw, Point(x2,y2), 5, Scalar(0,0,255), 2,5,0);
		}
		else {
			line(I_i1,Point(x,y),Point(x2,y2),Scalar(0,255,0),3);
			circle (I_i1, Point(x,y), 5,  Scalar(0,0,255), 2,8,0);
		}	
		
	}
	if ( run_code_on_drone == 0 ) {
		//imshow("Match I_i0_draw",I_i0_draw);
		//waitKey(0);
		imshow("Match I_i1_draw",I_i1_draw);
		waitKey(0);
		
	}
	
	
	
	// Find fudamental matrix 
	vector<uchar> pArray(N);
	//Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.90, 5000, pArray); // 3 can be changed to 1
	Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.90, pArray); // 3 can be changed to 1
	
	int N_inlier = countNonZero(pArray);
	cout << "Number of inlier keypoints in initializaiton = " << N_inlier << endl;
	
	// If initialization fails
	if (N_inlier < min_inlier) {
		initialization_okay = false;
		
		return make_tuple(Si_1, transformation_matrix, initialization_okay);
		
	}
	// If initialization succeeds
	else {
		initialization_okay = true;
		
		Mat points1Mat = Mat::zeros(2, N_inlier, CV_64FC1);
		Mat points2Mat = Mat::zeros(2, N_inlier, CV_64FC1);
		temp_index = 0;
		for (int i = 0; i < N; i++) {
			if ((double) pArray[i] == 1) {
				points1Mat.at<double>(0,temp_index) = temp_points1Mat.at<double>(0,i); // y-coordinate in image
				points1Mat.at<double>(1,temp_index) = temp_points1Mat.at<double>(1,i); // x-coordinate in image
				points2Mat.at<double>(0,temp_index) = temp_points2Mat.at<double>(0,i); // y-coordinate in image
				points2Mat.at<double>(1,temp_index) = temp_points2Mat.at<double>(1,i); // x-coordinate in image
				
				/*
				circle (I_i0_draw, Point(points1Mat.at<double>(1,temp_index),points1Mat.at<double>(0,temp_index)), 5, Scalar(255,0,0), 2,8,0);
				imshow("a", I_i0_draw);
				waitKey(0);
				circle (I_i1_draw, Point(points2Mat.at<double>(1,temp_index),points2Mat.at<double>(0,temp_index)), 5,  Scalar(255,0,0), 2,8,0);
				imshow("b", I_i1_draw);
				waitKey(0);
				*/
				
		
				
				temp_index++;
				
			}
		}
		 
		// Inlier keypoints after using RANSAC
		Si_1.k = N_inlier;
		
		// The realiably tracked keypoints
		Si_1.Pi = points2Mat;
			
		// Estimate Essential Matrix
		Mat essential_matrix = estimateEssentialMatrix(fundamental_matrix, K);	
		
		// Find the rotation and translation assuming the first frame is taken with the drone on the ground - At the origin
		transformation_matrix = findRotationAndTranslation(essential_matrix, K, points1Mat, points2Mat);
		cout << "Transformation matrix = " << transformation_matrix << endl;
		
		// Update State with regards to 3D (triangulated points)
		// Triangulate initial point cloud
		Mat M1 = K * (Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
		Mat M2 = K * transformation_matrix;
		Mat landmarks = linearTriangulation(points1Mat, points2Mat, M1, M2);
		
		Mat temp;
		vconcat(landmarks.row(0), landmarks.row(1), temp);
		vconcat(temp, landmarks.row(2), Si_1.Xi);
		

		
		cout << "Initializaiton" << endl;
		cout << "Number of keypoints = " << Si_1.Pi.cols << endl;
		cout << "Number of landmarks = " << Si_1.Xi.cols << endl;
	
		// return state of drone as well as transformation_matrix;
		return make_tuple(Si_1, transformation_matrix, initialization_okay);
	}
}


// ############################# VO Continuous Operation #############################
// Process Frame for continuous VO operation 
/* Arguments: 
 * Current Image I1
 * Previous Image Ii-1 (I_{i-1})
 * Previous State Si_1, which is a struct
 */
tuple<state, Mat, bool> processFrame(Mat Ii, Mat Ii_1, state Si_1, Mat K) {
	
	cout << "Images in processFrame" << endl;
	
	int run_code_on_drone = show_results; // 0 : Run code on Thor's Raspberry Pi  /  1 : Run code on quadcopter
	
	int min_inlier = min_nr_inliers_processFrame;
	double corner_strengh = Harris_Corner_strengh;
	bool processFrame_okay;
	Mat transformation_matrix, best_inlier_mask;

	// Turn the images into grayscale 
	Mat Ii_gray, Ii_1_gray;
	cvtColor(Ii, Ii_gray, COLOR_BGR2GRAY );
	cvtColor(Ii_1, Ii_1_gray, COLOR_BGR2GRAY );	
	
	if ( show_results == 0 ) {
		Mat draw_Ii_1;
		Ii_1.copyTo(draw_Ii_1);
		const char* text1 = "Detected corners in frame Ii_1";
		drawCorners(draw_Ii_1, Si_1.Pi, text1);
		waitKey(0);
	}
	
	// Find descriptors for previous frame I^i-1
	Mat descriptors_Ii_1 = SIFT::FindDescriptors(Ii_1_gray, Si_1.Pi);
	//imshow("Ii_1_gray", Ii_1_gray);
	//waitKey(0);
	
	// Find keypoints for current frame I^i
	int dim1, dim2;
	dim1 = Ii_gray.rows;
	dim2 = Ii_gray.cols;
	Mat Ii_resized = Ii_gray.colRange(10,dim2-10).rowRange(10,dim1-10);
	Mat temp1;
	goodFeaturesToTrack(Ii_resized, temp1, 300, 0.01, 10, noArray(), 3, true, corner_strengh);
	Mat keypoints_Ii = Mat::zeros(2, temp1.rows, CV_64FC1);
	for (int i = 0; i < keypoints_Ii.cols; i++) {
		keypoints_Ii.at<double>(0,i) = temp1.at<float>(i,1) + 10;
		keypoints_Ii.at<double>(1,i) = temp1.at<float>(i,0) + 10;
	}
	
	/*
	Mat draw_Ii;
	Ii.copyTo(draw_Ii);
	const char* text2 = "Detected corners in frame Ii";
	drawCorners(draw_Ii, keypoints_Ii, text2);
	waitKey(0);
	*/
	
	// Find descriptors for current frame I^i
	//imshow("Ii_gray", Ii_gray);
	//waitKey(0);
	Mat descriptors_Ii = SIFT::FindDescriptors(Ii_gray, keypoints_Ii);
	
	
	// Find matches for previous frame I^i-1 to current frame I^i
	Mat matches = SIFT::matchDescriptors2(descriptors_Ii_1, descriptors_Ii);
	matches = matches.row(1);
	
	// Find matches for current frame I^i to previous frame I^i-1
	Mat matches2 = SIFT::matchDescriptors2(descriptors_Ii, descriptors_Ii_1);
	matches2 = matches2.row(1);

	// Determine the valid matches
	int temp_index;
	int temp_index_advanced = 0; 
	Mat valid_matches_advanced = Mat::zeros(2, matches.cols, CV_64FC1);
	// Fix Problem here 
	for (int i = 0; i < matches.cols; i++) {
		if ( matches.at<double>(0,i) != -1) {
			if ( matches2.at<double>(0, matches.at<double>(0,i)) == i ) {
				valid_matches_advanced.at<double>(0, temp_index_advanced) = i;
				valid_matches_advanced.at<double>(1, temp_index_advanced) = matches.at<double>(0,i);
				temp_index_advanced++;
			} 
		}
	}
	matches = valid_matches_advanced.colRange(0, temp_index_advanced);
	cout << "Number of mutual valid mathces = " << matches.cols << endl;
	
	int N = matches.cols;
	
	if (N < min_inlier) {
		cout << "processFrame failed" << endl;
		processFrame_okay = false;
		
		return make_tuple(Si_1, transformation_matrix, processFrame_okay); 
	}
	
	// Matrix for keeping tracked keypoints and tracked corresponding landmarks 
	Mat keypoints_i = Mat::zeros(2, N, CV_64FC1);
	Mat corresponding_landmarks = Mat::zeros(3, N, CV_64FC1);
	
	
	Mat Ii_draw, Ii_1_draw; 
	Ii.copyTo(Ii_draw);
	Ii_1.copyTo(Ii_1_draw);
	
	
	
	for (int i = 0; i < N; i++) {

		// Turn the kyepoints so it becomes (u,v)
		keypoints_i.at<double>(0,i) = keypoints_Ii.at<double>(1, matches.at<double>(1,i)); // x-coordinate in image
		keypoints_i.at<double>(1,i) = keypoints_Ii.at<double>(0, matches.at<double>(1,i)); // y-coordinate in image
		
		//Si_1.Xi.col(matches(i,0)).copyTo(corresponding_landmarks.col(i));
		corresponding_landmarks.at<double>(0,i) = Si_1.Xi.at<double>(0, matches.at<double>(0,i));
		corresponding_landmarks.at<double>(1,i) = Si_1.Xi.at<double>(1, matches.at<double>(0,i));
		corresponding_landmarks.at<double>(2,i) = Si_1.Xi.at<double>(2, matches.at<double>(0,i));
		
		/*
		double y = keypoints_Ii.at<double>(0, matches.at<double>(1,i));
		double x = keypoints_Ii.at<double>(1, matches.at<double>(1,i));
		double y2 = Si_1.Pi.at<double>(0, matches.at<double>(0,i));
		double x2 = Si_1.Pi.at<double>(1, matches.at<double>(0,i));
		
		if ( run_code_on_drone == 0 ) {
			line(Ii_draw,Point(x2,y2),Point(x,y),Scalar(0,255,0),3);
			circle (Ii_draw, Point(x,y), 5,  Scalar(0,0,255), 2,8,0);
			circle (Ii_1_draw, Point(x2,y2), 5, Scalar(0,0,255), 2,8,0);
		}
		else {
			line(Ii ,Point(x2,y2),Point(x,y),Scalar(0,255,0),3);
			circle (Ii, Point(x,y), 5,  Scalar(0,0,255), 2,8,0);
			circle (Ii_1, Point(x2,y2), 5, Scalar(0,0,255), 2,8,0);
		}
		*/
		
	}
	
	
	cout << "Process Frame" << endl;
	cout << "Number of keypoints = " << keypoints_i.cols << endl;
	cout << "Number of landmarks = " << corresponding_landmarks.cols << endl;
	
	// Estimate the new pose using RANSAC and P3P algorithm 
	tie(transformation_matrix, best_inlier_mask) = Localize::ransacLocalization(keypoints_i, corresponding_landmarks, K);
	

	if ( run_code_on_drone == 0 ) {
		//cout << "best_inlier_mask " << endl;
		//cout << best_inlier_mask << endl;
		//cout << "Number of inliers = " << countNonZero(best_inlier_mask) << endl;
	}
	
	
	if (countNonZero(best_inlier_mask) < min_inlier) {
		cout << "processFrame failed" << endl;
		processFrame_okay = false;
		
		return make_tuple(Si_1, transformation_matrix, processFrame_okay); 
	}
	
	// Remove points that are determined as outliers from best_inlier_mask by using best_inlier_mask
	//cout << "best_inlier_mask" << endl;
	//cout << best_inlier_mask << endl;
	vector<Mat> keypoint_inlier;
	vector<Mat> landmark_inlier;
	for (int i = 0; i < best_inlier_mask.cols; i++) {
		if ((double) best_inlier_mask.at<uchar>(0,i) > 0) {
			keypoint_inlier.push_back(keypoints_i.col(i));
			landmark_inlier.push_back(corresponding_landmarks.col(i));
		}
	}
	hconcat(keypoint_inlier, keypoints_i);
	hconcat(landmark_inlier, corresponding_landmarks);
	
	
	// Update keypoints in state 
	Si_1.k = keypoints_i.cols;
	vconcat(keypoints_i.row(1), keypoints_i.row(0), Si_1.Pi); // Apparently you have to switch rows
	
	if ( run_code_on_drone == 0 ) {
		//imshow("matches in processFrame Ii_1_draw", Ii_1_draw);
		//waitKey(0);
		const char* text2 = "Reliable Si.Pi frame Ii";
		drawCorners(Ii_draw, Si_1.Pi, text2);
		waitKey(0);
	}
	
	corresponding_landmarks.copyTo(Si_1.Xi);
	
	//return make_tuple(Si, transformation_matrix); 
	processFrame_okay = true;
	cout << "processFrame done" << endl;
	cout << "Transformation matrix = " << transformation_matrix << endl;
	return make_tuple(Si_1, transformation_matrix, processFrame_okay); 
}



