#ifndef MAINCAMERA_HPP_INCLUDED
#define MAINCAMERA_HPP_INCLUDED 

// Libraries from opencv2
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

// Other libaries
#include "Matrix.h"
#include <math.h>
#include <tuple> 
//#include <pthread.h>
//#include <iostream>
#include "pthread.h"
#include <cstdlib>
#include <ctime>
#include <ratio>
#include <chrono>
//#include "gperftools/profiler.h"

using namespace cv;

//using namespace Numeric_lib;
using Matrix = Numeric_lib::Matrix<double,2>;
using Vector = Numeric_lib::Matrix<double,1>;

// Variables to define
// Harris corner
#define Harris_threads 30

// KLT
#define KLT_r_T 11
#define KLT_num_iters 20
#define KLT_lambda 0.6

// Num candidate keypoints
#define num_candidate_keypoints 30

// Ransac localization
#define ransac_pixel_tolerance 15 
#define ransac_min_inlier_count 10 

// triangulateNewLandmarks
#define new_landmarks_threshold_angle 20

// For the continous operation at struct is made. 
struct state {
	/* Si = (Pi, Xi, Ci, Fi, Ti)
	 * K = Number of keypoints
	 * Pi = Keypoints 														2 x K
	 * Xi = 3D landmarks 													3 x K
	 * Ci = Candidate Keypoints (The most recent observation)				2 x M
	 * Fi = The first ever observation of the keypoint						2 x M
	 * Ti = The Camera pose at the first ever observation of the keypoint 	16 x M
	 */
	int k;
	Mat Pi;
	Mat Xi; 
	
	int num_candidates;
	Mat Ci; 
	Mat Fi; 
	Mat Ti; 
};


struct harris_data {
		//int thread_id;
		Mat matrice;
		Mat thread_dst;
		double threshold;
		int num_keypoints;
		int thread_non_max_suppres;
		int left_corner_y;
		int left_corner_x;
		int valid_interest_points;
};

struct thread_data {
		//int thread_id;
		Mat Ii_1_gray;
		Mat Ii_gray;
		Mat thread_mat;
		Mat dwdx;
		int keep_point;
};

struct thread_descriptor {
		//int thread_id;
		Mat thread_interest_point;
		Mat thread_grad_x;
		Mat thread_grad_y;
		Mat thread_descriptor_vector;
		Mat thread_Gauss_Window;
};

struct thread_descriptorAdvanced {
		int thread_descriptor_id;
		Mat thread_interest_point;
		Mat thread_grad_x;
		Mat thread_grad_y;
		Mat thread_Gauss_Window;
		Mat thread_multiple_descriptors;
};

struct thread_match {
		//int thread_id;
		int descriptor_n1_id;
		int descriptor_n2_id;
		Mat descriptors_n1;
		Mat descriptors_n2;
		int is_inlier;
		double lowest_distance;
		int best_match;
};

// SIT = SIFT Descriptor thread struct
/*
struct SIT {
	Mat image_gray; // 
	Mat keypoints; 	// 2xM matrix
	Mat descriptors;
};
*/

namespace Harris {
	Mat corner(Mat src, Mat src_gray, int maxinum_keypoint, Mat suppression); 
	//void corner(Mat src, Mat src_gray, bool display);

}	// Harris Corner


namespace SIFT {
	Mat FindDescriptors(Mat src_gray, Mat keypoints);
	//Matrix matchDescriptors(Mat descriptor1, Mat descriptor2);
	Mat matchDescriptors(Mat descriptor1, Mat descriptor2);
	
	Mat FindDescriptorsAdvanced(Mat src_gray, Mat keypoints);
	Mat matchDescriptorsAdvanced(Mat descriptor1, Mat descriptor2);
	
}	// SIFT


void *FindDescriptors(void *threadarg);


namespace KLT {
	//Mat trackKLTrobustly(Mat I_R, Mat I, Mat keypoint, int r_T, int num_iters, double lambda);
	Mat trackKLTrobustly(Mat I_R, Mat I_new, Mat keypoint, Mat dwdx, int r_T, int num_iters, double lambda);
}

namespace Localize {
	std::tuple<Mat, Mat> ransacLocalization(Mat keypoints_i, Mat corresponding_landmarks, Mat K);
}


// Estimate position of camera 
Mat linearTriangulation(Mat p1, Mat p2, Mat M1, Mat M2);
Mat estimateEssentialMatrix(Mat fundamental_matrix, Mat K);
Mat findRotationAndTranslation(Mat essential_matrix, Mat K, Mat points1Mat, Mat points2Mat);

// Find new candidate Keypoints
state newCandidateKeypoints(Mat Ii, state Si, Mat T_wc);
state continuousCandidateKeypoints(Mat Ii_1, Mat Ii, state Si, Mat T_wc);

// Triangulate new candidate Keypoints
//std::tuple<state, Mat>  triangulateNewLandmarks(state Si, Mat K, Mat T_WC, double threshold_angle);
state triangulateNewLandmarks(state Si, Mat K, Mat T_WC, double threshold_angle);


// For KLT
//Mat trackKLT(Mat I_R, Mat I, Mat x_T, int r_T, int num_iters);
//Mat getSimWarp(double dx, double dy, double alpha_deg, double lambda);
//Mat warpImage(Mat I_R, Mat W);
Mat solveQuartic(Mat factors);
Mat p3p(Mat worldPoints, Mat imageVectors);
Mat Kroneckerproduct(Mat A, Mat B);
void *functionKLT(void *threadarg);

// Helper funcitons
void MatType( Mat inputMat );

// Test of funcitons 
Mat findLandmark(Mat K, Mat tau, Mat T_WC, Mat keypoint0, Mat keypoint1);

// Initialization
std::tuple<state, Mat, bool> initialization(Mat I_i0, Mat I_i1, Mat K, state Si_1);

// Process Frame 
std::tuple<state, Mat, bool> processFrame(Mat Ii, Mat Ii_1, state Si_1, Mat K);


// Test of threadid
//void *PrintHello(void *threadid);
void *PrintHello(void *threadarg);

#endif
