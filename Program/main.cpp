#include <iostream>
#include <say-hello/hello.hpp>
#include "mainCamera.hpp"
#include <unistd.h>

// Include directories for raspicam
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <raspicam/raspicam_cv.h>

/* ########################
 * Name: mainCamera.cpp
 * Made by: Thor Christiansen - s173949 
 * Date: 29.06.2020
 * Objective: The source file mainCamera.cpp contains the functions used
 * by main.cpp to treat the images - find features in the images using 
 * Harris corner etc.
 * Project: Bachelor project 2020
 * 
 * Problems: 
 * OVERSkyggende problem: Det ser ud til, at der er et problem med matricen K, som skal kalibreres.
 * Lær hvordan man SSH'er ind på raspberry pi'en
 * Units seem to be meters 
 * What is the unit of the 3D points? Is it cm? meters? other units?
 * Maybe you fuck up when you try to use KLT in the initialization 
 * See MatLab code week 3 for an efficient way to find Harris Corners 
 * When using KLT: Remember to use gray-scale images and resize the images with a factor of 4 
 * Implementation of ransac  in pose estimation in processFrame function 
 * Resize the images with a factor of 4 and also the resize the keypoints with a factor of 4 before processFrame to enhance speed
 * ########################
*/

// ####################### Raspicam Functions #######################
// Namespace and constants 
using namespace cv;
using namespace std;
const char* source_window = "Source image"; 
bool doTestSpeedOnly=false;

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
	Mat Ci; 
	Mat Fi; 
	Mat Ti; 
};

//parse command line
//returns the index of a command line param in argv. If not found, return -1
int findParam ( string param,int argc,char **argv ) {
    int idx=-1;
    for ( int i=0; i<argc && idx==-1; i++ )
        if ( string ( argv[i] ) ==param ) idx=i;
    return idx;

}
//parse command line
//returns the value of a command line param. If not found, defvalue is returned
float getParamVal ( string param,int argc,char **argv,float defvalue=-1 ) {
    int idx=-1;
    for ( int i=0; i<argc && idx==-1; i++ )
        if ( string ( argv[i] ) ==param ) idx=i;
    if ( idx==-1 ) return defvalue;
    else return atof ( argv[  idx+1] );
}

// Purpose: To set the camera properties - resolution etc.
void processCommandLine ( int argc,char **argv,raspicam::RaspiCam_Cv &Camera ) {
    Camera.set ( cv::CAP_PROP_FRAME_WIDTH,  getParamVal ( "-w",argc,argv,1280 ) );
    Camera.set ( cv::CAP_PROP_FRAME_HEIGHT, getParamVal ( "-h",argc,argv,960 ) );
    Camera.set ( cv::CAP_PROP_BRIGHTNESS,getParamVal ( "-br",argc,argv,50 ) );
    Camera.set ( cv::CAP_PROP_CONTRAST ,getParamVal ( "-co",argc,argv,50 ) );
    Camera.set ( cv::CAP_PROP_SATURATION, getParamVal ( "-sa",argc,argv,50 ) );
    Camera.set ( cv::CAP_PROP_GAIN, getParamVal ( "-g",argc,argv ,50 ) );
    Camera.set ( cv::CAP_PROP_FPS, getParamVal ( "-fps",argc,argv, 0 ) );
    if ( findParam ( "-gr",argc,argv ) !=-1 )
        Camera.set ( cv::CAP_PROP_FORMAT, CV_8UC1 );
    if ( findParam ( "-test_speed",argc,argv ) !=-1 )
        doTestSpeedOnly=true;
    if ( findParam ( "-ss",argc,argv ) !=-1 )
        Camera.set ( cv::CAP_PROP_EXPOSURE, getParamVal ( "-ss",argc,argv )  );
}

void drawCorners(Mat img, Matrix keypoints, const char* frame_name) {
	for (int k = 0; k < keypoints.dim1(); k++) {
		double x = keypoints(k,1);
		double y = keypoints(k,2);
		circle (img, Point(y,x), 5, Scalar(200), 2,8,0);
	}
	imshow(frame_name, img);
	waitKey(0);
}


// ####################### VO Initialization Pipeline #######################
tuple<state, Mat> initializaiton(Mat I_i0, Mat I_i1, Mat K, state Si_1) {
	cout << "Begin initialization" << endl;
	
	// Transform color images to gray images
	Mat I_i0_gray, I_i1_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cvtColor(I_i1, I_i1_gray, COLOR_BGR2GRAY );
	
	// Get Feature points
	Matrix keypoints_I_i0 = Harris::corner(I_i0, I_i0_gray);
	const char* text0 = "Detected corners in frame I_i0";
	//drawCorners(I_i0, keypoints_I_i0, text0);
	//waitKey(0);
	
	/*
	// ######################### KLT ######################### 
	cout << "KLT " << endl;
	Mat kpold = Mat::zeros(3, keypoints_I_i0.dim1(), CV_64FC1);
	
	//cout << "Performing KLT " << endl;
	int r_T = 15; 
	int num_iters = 50; 
	double lambda = 0.1;
	int nr_keep = 0;
	Mat delta_keypoint = Mat::zeros(3, 1, CV_64FC1);
	Mat x_T = Mat::zeros(1, 2, CV_64FC1); // Interest point
	for (int i = 0; i < keypoints_I_i0.dim1(); i++) {
		//cout << "New iteration" << endl;
		x_T.at<double>(0,0) = keypoints_I_i0(i,1);
		x_T.at<double>(0,1) = keypoints_I_i0(i,2);
		cout << "Coordinates x_T: (" << x_T.at<double>(0,0) << "," << x_T.at<double>(0,1) << ")" << endl;

		delta_keypoint = KLT::trackKLTrobustly(I_i0_gray, I_i1_gray, x_T, r_T, num_iters, lambda);
		
		if (delta_keypoint.at<double>(2,0) == 1) {
			nr_keep++;
			kpold.at<double>(2,i) = 1; // The keypoint is reliably matched
		}
		cout << "Mistake " << endl;
		cout << keypoints_I_i0(i,0);
		kpold.at<double>(0,i) = delta_keypoint.at<double>(0,0) + keypoints_I_i0(i,1); // x 
		kpold.at<double>(1,i) = delta_keypoint.at<double>(1,0) + keypoints_I_i0(i,2); // y
		cout << "kpold: (" << kpold.at<double>(0,i) << "," << kpold.at<double>(1,i) << ") and keep = " << delta_keypoint.at<double>(2,0)  << endl;
	}
	cout << "Ready to draw corners" << endl;
	for (int k = 0; k < kpold.cols; k++) {
		double x = kpold.at<double>(0,k);
		double y = kpold.at<double>(1,k);
		circle (I_i1, Point(y,x), 5, Scalar(200), 2,8,0);
	}
	imshow("Corners with outliers", I_i1);
	waitKey(0);
	
	//cout << "Done finding keypoints" << endl;
	Mat keypoints_I_i1 = Mat::zeros(2, nr_keep, CV_64FC1);
	Mat keypoints_I_i0_new = Mat::zeros(2, nr_keep, CV_64FC1);
	
	nr_keep = 0; 
	for (int j = 0; j < keypoints_I_i1.cols; j++) {
		if (kpold.at<double>(2,j) == 1) {
			keypoints_I_i1.at<double>(0, nr_keep) = kpold.at<double>(0, j);
			keypoints_I_i1.at<double>(1, nr_keep) = kpold.at<double>(1, j);
			keypoints_I_i0_new.at<double>(0, nr_keep) = keypoints_I_i0(j,1);
			keypoints_I_i0_new.at<double>(1, nr_keep) = keypoints_I_i0(j,2);
			nr_keep++;
		}
	}
	int N = nr_keep--;
	*/
	
	cout << "Matrix K " << endl;
	for (int r = 0; r < K.rows; r++) {
		for (int c = 0; c < K.cols; c++) {
			cout << K.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	
	// ######################### SIFT ######################### 
	Matrix keypoints_I_i1 = Harris::corner(I_i1, I_i1_gray);
	const char* text1 = "Detected corners in frame I_i1";
	//drawCorners(I_i1, keypoints_I_i1,text1);
	//waitKey(0);
	//cout << "Done with finding keypoints " << endl;
	// Find descriptors for Feature Points
	
	
	// Maybe use KLT instead 
	
	Matrix descriptors_I_i0 = SIFT::FindDescriptors(I_i0_gray, keypoints_I_i0);
	//cout << "descriptors_I_i0 dimensions = (" << descriptors_I_i0.dim1() << "," << descriptors_I_i0.dim2() << ")" << endl;
	
	Matrix descriptors_I_i1 = SIFT::FindDescriptors(I_i1_gray, keypoints_I_i1);
	//cout << "descriptors_I_i1 dimensions = (" << descriptors_I_i1.dim1() << "," << descriptors_I_i1.dim2() << ")" << endl;
	
	// Match descriptors 
	Matrix matches = SIFT::matchDescriptors(descriptors_I_i0, descriptors_I_i1);
	
	// Find Point correspondences
	// Points from image 0 in row 1 and row 2 
	// Points from image 1 in row 3 and row 	
	int N = matches.dim1();
	
	cout << "Number of matched keypoints N in initializaiton = " << N << endl;
	// For plotting
	// For efficiency, you should maybe just use vectors instead of creating two new matrices
	Mat temp_points1Mat = Mat::zeros(2, N, CV_64FC1);
	Mat temp_points2Mat = Mat::zeros(2, N, CV_64FC1);
	// For fudamental matrix
	vector<Point2f> points1(N);
	vector<Point2f> points2(N);
	for (int i = 0; i < N; i++) {
		
		
		// ########## KLT ##########
		/*
		 * 
		points1[i] = Point2f(keypoints_I_i1.at<double>(0,i),keypoints_I_i1.at<double>(1,i));
		points2[i] = Point2f(keypoints_I_i0_new.at<double>(0,i),keypoints_I_i0_new.at<double>(1,i));
		
		points1Mat.at<double>(0,i) = keypoints_I_i1.at<double>(0,i);
		points1Mat.at<double>(1,i) = keypoints_I_i1.at<double>(1,i); 
		points2Mat.at<double>(0,i) = keypoints_I_i0_new.at<double>(0,i);
		points2Mat.at<double>(1,i) = keypoints_I_i0_new.at<double>(1,i);
		
		
		
		double x = keypoints_I_i1.at<double>(0,i);
		double y = keypoints_I_i1.at<double>(1,i);
		double x2 = keypoints_I_i0_new.at<double>(0,i);
		double y2 = keypoints_I_i0_new.at<double>(1,i);
		
		line(I_i1,Point(y,x),Point(y2,x2),Scalar(0,255,0),3);
		circle (I_i1, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
		circle (I_i0, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
		*/
		
		
		// Config1
		//  #### SIFT #### 
		// Be aware of differences in x and y
		points1[i] = Point2f(keypoints_I_i0(matches(i,1),1),keypoints_I_i0(matches(i,1),2));
		points2[i] = Point2f(keypoints_I_i1(matches(i,0),1),keypoints_I_i1(matches(i,0),2));
		
		temp_points1Mat.at<double>(0,i) = keypoints_I_i0(matches(i,1),1);
		temp_points1Mat.at<double>(1,i) = keypoints_I_i0(matches(i,1),2); 
		temp_points2Mat.at<double>(0,i) = keypoints_I_i1(matches(i,0),1);
		temp_points2Mat.at<double>(1,i) = keypoints_I_i1(matches(i,0),2);
		
	
		double x = keypoints_I_i1(matches(i,0),1);
		double y = keypoints_I_i1(matches(i,0),2);
		double x2 = keypoints_I_i0(matches(i,1),1);
		double y2 = keypoints_I_i0(matches(i,1),2);
		line(I_i1,Point(y,x),Point(y2,x2),Scalar(0,255,0),3);
		circle (I_i1, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
		circle (I_i0, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
		
		// SIFT but switching the order of points 
		
		
		/*
		// Config2
		//  #### SIFT #### 
		// Be aware of differences in x and y
		points1[i] = Point2f(keypoints_I_i0(matches(i,1),2),keypoints_I_i0(matches(i,1),1));
		points2[i] = Point2f(keypoints_I_i1(matches(i,0),2),keypoints_I_i1(matches(i,0),1));
		
		points1Mat.at<double>(0,i) = keypoints_I_i0(matches(i,1),1);
		points1Mat.at<double>(1,i) = keypoints_I_i0(matches(i,1),2); 
		points2Mat.at<double>(0,i) = keypoints_I_i1(matches(i,0),1);
		points2Mat.at<double>(1,i) = keypoints_I_i1(matches(i,0),2);
		
	
		double x = keypoints_I_i1(matches(i,0),1);
		double y = keypoints_I_i1(matches(i,0),2);
		double x2 = keypoints_I_i0(matches(i,1),1);
		double y2 = keypoints_I_i0(matches(i,1),2);
		line(I_i1,Point(y,x),Point(y2,x2),Scalar(0,255,0),3);
		circle (I_i1, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
		circle (I_i0, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
		*/
		
		
		/*
		// Config3
		//  #### SIFT #### 
		// Be aware of differences in x and y
		points1[i] = Point2f(keypoints_I_i0(matches(i,1),1),keypoints_I_i0(matches(i,1),2));
		points2[i] = Point2f(keypoints_I_i1(matches(i,0),1),keypoints_I_i1(matches(i,0),2));
		
		points1Mat.at<double>(0,i) = keypoints_I_i0(matches(i,1),2);
		points1Mat.at<double>(1,i) = keypoints_I_i0(matches(i,1),1); 
		points2Mat.at<double>(0,i) = keypoints_I_i1(matches(i,0),2);
		points2Mat.at<double>(1,i) = keypoints_I_i1(matches(i,0),1);
		
	
		double x = keypoints_I_i1(matches(i,0),1);
		double y = keypoints_I_i1(matches(i,0),2);
		double x2 = keypoints_I_i0(matches(i,1),1);
		double y2 = keypoints_I_i0(matches(i,1),2);
		line(I_i1,Point(y,x),Point(y2,x2),Scalar(0,255,0),3);
		circle (I_i1, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
		circle (I_i0, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
		*/
		
		
		// Config4
		/*
		//  #### SIFT #### 
		// Be aware of differences in x and y
		points1[i] = Point2f(keypoints_I_i0(matches(i,1),2),keypoints_I_i0(matches(i,1),1));
		points2[i] = Point2f(keypoints_I_i1(matches(i,0),2),keypoints_I_i1(matches(i,0),1));
		
		points1Mat.at<double>(0,i) = keypoints_I_i0(matches(i,1),2);
		points1Mat.at<double>(1,i) = keypoints_I_i0(matches(i,1),1); 
		points2Mat.at<double>(0,i) = keypoints_I_i1(matches(i,0),2);
		points2Mat.at<double>(1,i) = keypoints_I_i1(matches(i,0),1);
		
	
		double x = keypoints_I_i1(matches(i,0),1);
		double y = keypoints_I_i1(matches(i,0),2);
		double x2 = keypoints_I_i0(matches(i,1),1);
		double y2 = keypoints_I_i0(matches(i,1),2);
		line(I_i1,Point(y,x),Point(y2,x2),Scalar(0,255,0),3);
		circle (I_i1, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
		circle (I_i0, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
		*/
		
		
		// SIFT but switching the order of points 
		
		
		
		
	}
	//imshow("Match",I_i1);
	//waitKey(0);
	
	// Update State  with regards to keypoints in frame Ii_1
	//Si_1.Pi = temp_points2Mat;
	
	cout << "Print of Si_1 Keypoints " << endl;
	for (int r = 0; r < Si_1.Pi.rows; r++) {
		for (int c = 0; c < Si_1.Pi.cols; c++) {
			cout << Si_1.Pi.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "Number of keypoints = " << Si_1.Pi.cols << endl;
	
	// Find fudamental matrix 
	// Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, 5000);
	//int pArray[N];
	//Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 1, 0.95, 5000, noArray()); // 1 should be changed ot 3 
	vector<uchar> pArray(N);
	Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 1, 0.95, 5000, pArray);
	
	
	cout << "Output Mask " << endl;
	for (int i = 0; i < N; i++) {
		cout << (double) pArray[i] << ", ";
	}
	cout << "" << endl;
	cout << "Nonzero elements = " << countNonZero(pArray) << endl;
	
	int N_inlier = countNonZero(pArray);
	cout << "N_inlier = " << N_inlier << endl;
	Mat points1Mat = Mat::zeros(2, N_inlier, CV_64FC1);
	Mat points2Mat = Mat::zeros(2, N_inlier, CV_64FC1);
	int temp_index = 0;
	for (int i = 0; i < N; i++) {
		if ((double) pArray[i] == 1) {
			points1Mat.at<double>(0,temp_index) = temp_points1Mat.at<double>(0,i);
			points1Mat.at<double>(0,temp_index) = temp_points1Mat.at<double>(1,i);
			points2Mat.at<double>(0,temp_index) = temp_points2Mat.at<double>(0,i);
			points2Mat.at<double>(0,temp_index) = temp_points2Mat.at<double>(1,i);
			temp_index++;
		}
	}
	
	// 
	cout << "Number of reliably matched keypoints (using RANSAC) in initializaiton = " << N_inlier << endl;
	Si_1.k = N_inlier;
	
	// Update of reliably matched keypoints
	Si_1.Pi = points2Mat;
	
	// Estimate Essential Matrix
	Mat essential_matrix = estimateEssentialMatrix(fundamental_matrix, K);	
	
	// Find position and rotation from images
	//Mat essential_matrix = (Mat_<double>(3,3) << -0.10579, -0.37558, -0.5162047, 4.39583, 0.25655, 19.99309, 0.4294123, -20.32203997, 0.023287939);
	// Find the rotation and translation assuming the first frame is taken with the drone on the ground 
	Mat transformation_matrix = findRotationAndTranslation(essential_matrix, K, points1Mat, points2Mat);
	for (int i = 0; i < transformation_matrix.rows; i++) {
		for (int j = 0; j < transformation_matrix.cols; j++) {
			cout << transformation_matrix.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	// Update State with regards to 3D (triangulated points)
	// Triangulate initial point cloud
	Mat M1 = K * (Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
	Mat M2 = K * transformation_matrix;
	Si_1.Xi = linearTriangulation(points1Mat, points2Mat, M1, M2);
	
	cout << "Print of 3D landmarks " << endl;
	for (int r = 0; r < Si_1.Xi.rows; r++) {
		for (int c = 0; c < Si_1.Xi.cols; c++) {
			cout << Si_1.Xi.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
		cout << "" << endl;
	}
	cout << "Number of landmarks = " << Si_1.Xi.cols << endl;
	
	// return state of drone as well as transformation_matrix;
	return make_tuple(Si_1, transformation_matrix);
}

Mat calibrate_matrix(Mat transformation_matrix) {
	
	Mat calib_matrix = Mat::eye(3, 4, CV_64FC1) - transformation_matrix;
	
	for (int r = 0; r < calib_matrix.rows; r++) {
		for (int c = 0; c < calib_matrix.cols; c++) {
			cout << calib_matrix.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	return calib_matrix;
}

// ####################### VO Continuous Operation #######################
// Process Frame for continuous VO operation 
/* Arguments: 
 * Current Image I1
 * Previous Image Ii-1 (I_{i-1})
 * Previous State Si_1, which is a struct
 */
tuple<state, Mat> processFrame(Mat Ii, Mat Ii_1, state Si_1, Mat K) {

	// Turn the images into grayscale 


	// new state
	state Si;

	// Variables 
	int r_T = 15; 
	int num_iters = 50; 
	double lambda = 0.1;
	int nr_keep = 0;
	Mat kpold = Mat::zeros(3, Si_1.k, CV_64FC1);
	Mat delta_keypoint;
	
	// Track every keypoint - Maybe utilize parallelization 
	Mat x_T = Mat::zeros(1, 2, CV_64FC1); // Interest point
	for (int i = 0; i < Si_1.k; i++) {
		x_T.at<double>(0,0) = Si_1.Pi.at<double>(0,i);
		x_T.at<double>(0,1) = Si_1.Pi.at<double>(1,i);

		delta_keypoint = KLT::trackKLTrobustly(Ii_1, Ii, x_T, r_T, num_iters, lambda);
		
		for (int k = 0; k < delta_keypoint.rows; k++) {

		}
		
		if (delta_keypoint.at<double>(2,0) == 1) {
			nr_keep++;
			kpold.at<double>(2,i) = 1; // The keypoint is reliably matched
		}
		kpold.at<double>(0,i) = delta_keypoint.at<double>(0,0) + Si_1.Pi.at<double>(0,i);
		kpold.at<double>(1,i) = delta_keypoint.at<double>(1,0) + Si_1.Pi.at<double>(1,i);

	}
	//Mat keypoints_i_1 = Mat::zeros(2, nr_keep, CV_64FC1);
	Mat keypoints_i = Mat::zeros(2, nr_keep, CV_64FC1);
	
	// corresponding landmarks 
	Mat corresponding_landmarks = Mat::zeros(3, nr_keep, CV_64FC1);
	
	nr_keep = 0; 
	for (int j = 0; j < Si_1.k; j++) {
		if (kpold.at<double>(2,j) == 1) {
			// Update matched keypoints 
			keypoints_i.at<double>(0, nr_keep) = kpold.at<double>(0, j);
			keypoints_i.at<double>(1, nr_keep) = kpold.at<double>(1, j);
			
			// Update landmarks 
			corresponding_landmarks.at<double>(0, nr_keep) = Si_1.Xi.at<double>(0, j);
			corresponding_landmarks.at<double>(1, nr_keep) = Si_1.Xi.at<double>(1, j);
			corresponding_landmarks.at<double>(2, nr_keep) = Si_1.Xi.at<double>(2, j);
			
			nr_keep++;
		}
	}
	cout << "Number of keypoints left = " << keypoints_i.cols << endl;
	
	// Delete landmarks for those points that were not matched 
	
	// Update keypoints in state 
	Si.k = keypoints_i.cols;
	keypoints_i.copyTo(Si.Pi); 
	corresponding_landmarks.copyTo(Si.Xi);
	
	// Estimate the new pose using RANSAC and P3P algorithm 
	Mat transformation_matrix, best_inlier_mask;
	tie(transformation_matrix, best_inlier_mask) = Localize::ransacLocalization(keypoints_i, corresponding_landmarks, K);
	
	// Remove points that are determined as outliers from best_inlier_mask 
	
	/*
	// Triangulate new points
	Mat M1 = K * prev_transformation_matrix;
	Mat M2 = K * transformation_matrix;
	Si.Xi = linearTriangulation(Si_1.Pi, Si.Pi, M1, M2 );
	*/
	
	return make_tuple(Si, transformation_matrix); 
}


// ####################### Main function #######################
int main ( int argc,char **argv ) {
	
	
	raspicam::RaspiCam_Cv Camera;
	processCommandLine ( argc,argv,Camera );
	cout<<"Connecting to camera"<<endl;
	if ( !Camera.open() ) {
		cerr<<"Error opening camera"<<endl;
		return -1;
	}
	cout<<"Connected to camera ="<<Camera.getId() <<endl;
	
	// Calibrate camera to get intrinsic parameters K 
	Mat K = (Mat_<double>(3,3) << 769.893, 0, 2.5, 0,1613.3, 4, 0, 0, 1);
	cout << "K (intrinsic matrix)" << endl;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cout << K.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	cv::Mat I_i0, I_i1, image;
	
	int nCount=getParamVal ( "-nframes",argc,argv, 100 );
	cout<<"Capturing"<<endl;
	
	// Initialization
	Camera.grab(); // You need to take an initial image in order to make the camera work
	Camera.retrieve( image ); 
	//cout << "Image captured" <<endl;
	waitKey(1000);
	
	// Initial frame 0 
	Camera.grab();
	Camera.retrieve( I_i0 ); 
	cout << "Frame I_i0 captured" <<endl;
	//imshow("Frame I_i0 displayed", I_i0);
	//waitKey(0);	// Ensures it is sufficiently far away from initial frame
	waitKey(5000);
	
	// First frame 1 
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;
	//imshow("Frame I_i1 displayed", I_i1);
	//waitKey(0
	
	
	// ADVARSEL: POTENTIEL FEJL MED HVORDAN KEYPOINTS ER STRUKTURERET PÅ. mÅSEK SKAL DER BYTTES OM PÅ RÆKKER. 
	
	// ############### VO initializaiton ###############
	// VO-pipeline: Initialization. Bootstraps the initial position. 
	state Si_1;
	Si_1.k = 0;
	Mat transformation_matrix;
	//tie(Si_1, transformation_matrix) = initializaiton(I_i0, I_i1, K2, Si_1);
	tie(Si_1, transformation_matrix) = initializaiton(I_i0, I_i1, K, Si_1);
	cout << "Transformation matrix Thor " << endl;
	for (int r = 0; r < transformation_matrix.rows; r++) {
		for (int c = 0; c < transformation_matrix.cols; c++) {
			cout << transformation_matrix.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	
	
	
	cout << "State Si_1 before initializaiton" << endl;
	cout << "Number of keypoints k = " << Si_1.k << endl;
	cout << "State Si_1" << endl;
	cout << "Keypoints of state Si = (" << Si_1.Pi.rows << "," << Si_1.Pi.cols << ")" << endl;
	cout << "Landmarks of state Si = " << Si_1.Xi.rows << "," << Si_1.Xi.cols << endl;
	for (int i = 0; i < 5; i++) {
		cout << Si_1.Xi.at<double>(0,i) << "," << Si_1.Xi.at<double>(1,i) << ","  << Si_1.Xi.at<double>(2,i) << ","  << Si_1.Xi.at<double>(3,i) << endl;
	}
	
	
	// ############### VO Continuous ###############
	bool continueVOoperation = true;
	bool pipelineBroke = false;
	bool output_T_ready = true;
	
	// Needed variables
	state Si;
	Mat Ii;
	Mat Ii_1 = I_i1;
	
	// Debug variable
	int stop = 1;
	
	while (continueVOoperation == true && pipelineBroke == false && stop < 0) {
		cout << "Begin Continuous VO operation " << endl;
		
		// Take new image 
		Camera.grab();
		Camera.retrieve( Ii );
		
		imshow("New Frame", Ii);
		waitKey(0);
		
		// Estimate pose 
		tie(Si, transformation_matrix) = processFrame(Ii, Ii_1, Si_1, K);
		cout << "Print of Transformation Matrix" << endl;
		for (int r = 0; r < transformation_matrix.rows; r++) {
			for (int c = 0; c < transformation_matrix.cols; c++) {
				cout << transformation_matrix.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		output_T_ready = true;
		
		// Submit the transformation matrix. Then set the flag low. 
		output_T_ready = false;
		
		// Resert old values 
		Ii_1 = Ii;
		Si_1 = Si;
		
		// Debug variable
		stop++;
		if (stop > 10) {
			break;
		}
		
	}
	
	cout << "VO-pipeline terminated" << endl;


	double time_=cv::getTickCount();
	
	Camera.release();
}










