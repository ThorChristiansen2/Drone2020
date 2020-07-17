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
 * Sæt angle threshold 
 * OVERSkyggende problem: Det ser ud til, at der er et problem med matricen K, som skal kalibreres.
 * Lær hvordan man SSH'er ind på raspberry pi'en
 * Units seem to be meters 
 * The units are in meters.
 * Maybe you fuck up when you try to use KLT in the initialization 
 * See MatLab code week 3 for an efficient way to find Harris Corners 
 * When using KLT: Remember to use gray-scale images // Did not work with resizing the image with a factor of 4 
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

void drawCorners(Mat img, Mat keypoints, const char* frame_name) {
	for (int k = 0; k < keypoints.cols; k++) {
		double y = keypoints.at<double>(1, k);
		double x = keypoints.at<double>(2, k);
		circle (img, Point(x,y), 5, Scalar(200), 2,8,0);
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
	
	Mat emptyMatrix;
	
	
	// Get Feature points
	Mat keypoints_I_i0 = Harris::corner(I_i0, I_i0_gray, 200, emptyMatrix); // Number of maximum keypoints
	const char* text0 = "Detected corners in frame I_i0";
	drawCorners(I_i0, keypoints_I_i0, text0);
	waitKey(0);
	
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
	
	/*
	cout << "Matrix K " << endl;
	for (int r = 0; r < K.rows; r++) {
		for (int c = 0; c < K.cols; c++) {
			cout << K.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	
	
	
	// ######################### SIFT ######################### 
	Mat keypoints_I_i1 = Harris::corner(I_i1, I_i1_gray, 200, emptyMatrix); // Number of keypoints that is looked for
	const char* text1 = "Detected corners in frame I_i1";
	drawCorners(I_i1, keypoints_I_i1, text1);
	waitKey(0);
	//cout << "Done with finding keypoints " << endl;
	// Find descriptors for Feature Points
	cout << "drawCorners found" << endl;
	
	
	// Maybe use KLT instead 
	
	Matrix descriptors_I_i0 = SIFT::FindDescriptors(I_i0_gray, keypoints_I_i0);
	//cout << "descriptors_I_i0 dimensions = (" << descriptors_I_i0.dim1() << "," << descriptors_I_i0.dim2() << ")" << endl;
	
	cout << "descriptors_I_i0 found" << endl;
	
	Matrix descriptors_I_i1 = SIFT::FindDescriptors(I_i1_gray, keypoints_I_i1);
	//cout << "descriptors_I_i1 dimensions = (" << descriptors_I_i1.dim1() << "," << descriptors_I_i1.dim2() << ")" << endl;
	
	cout << "descriptors_I_i1 found" << endl;
	
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
		points1[i] = Point2f(keypoints_I_i0.at<double>(1, matches(i,1)),keypoints_I_i0.at<double>(2, matches(i,1)));
		points2[i] = Point2f(keypoints_I_i1.at<double>(1, matches(i,0)),keypoints_I_i1.at<double>(2, matches(i,0)));
		
		temp_points1Mat.at<double>(0,i) = keypoints_I_i0.at<double>(1, matches(i,1)); // y-coordinate in image 
		temp_points1Mat.at<double>(1,i) = keypoints_I_i0.at<double>(2, matches(i,1)); // x-coordinate in image
		temp_points2Mat.at<double>(0,i) = keypoints_I_i1.at<double>(1, matches(i,0)); // y-coordinate in image
		temp_points2Mat.at<double>(1,i) = keypoints_I_i1.at<double>(2, matches(i,0)); // x-coordinate in image
		
	
		double y = keypoints_I_i1.at<double>(1, matches(i,0));
		double x = keypoints_I_i1.at<double>(2, matches(i,0));
		double y2 = keypoints_I_i0.at<double>(1, matches(i,1));
		double x2 = keypoints_I_i0.at<double>(2, matches(i,1));
		line(I_i1,Point(x,y),Point(x2,y2),Scalar(0,255,0),3);
		circle (I_i1, Point(x,y), 5,  Scalar(0,0,255), 2,8,0);
		circle (I_i0, Point(x2,y2), 5, Scalar(0,0,255), 2,8,0);
		
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
	imshow("Match",I_i1);
	waitKey(0);
	
	
	
	
	
	// Update State  with regards to keypoints in frame Ii_1
	//Si_1.Pi = temp_points2Mat;
	
	
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
			points1Mat.at<double>(0,temp_index) = temp_points1Mat.at<double>(0,i); // y-coordinate in image
			points1Mat.at<double>(1,temp_index) = temp_points1Mat.at<double>(1,i); // x-coordinate in image
			points2Mat.at<double>(0,temp_index) = temp_points2Mat.at<double>(0,i); // y-coordinate in image
			points2Mat.at<double>(1,temp_index) = temp_points2Mat.at<double>(1,i); // x-coordinate in image
			temp_index++;
		}
	}
	waitKey(5000);
	// 
	cout << "Number of reliably matched keypoints (using RANSAC) in initializaiton = " << N_inlier << endl;
	Si_1.k = N_inlier;
	
	// Update of reliably matched keypoints
	Si_1.Pi = points2Mat;
	
	cout << "Print of Si_1 Keypoints " << endl;
	for (int r = 0; r < Si_1.Pi.rows; r++) {
		for (int c = 0; c < Si_1.Pi.cols; c++) {
			cout << Si_1.Pi.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "Number of keypoints = " << Si_1.Pi.cols << endl;
	
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
	Mat landmarks = linearTriangulation(points1Mat, points2Mat, M1, M2);
	
		cout << "Print landmarks " << endl;
	for (int r = 0; r < landmarks.rows; r++) {
		for (int c = 0; c < landmarks.cols; c++) {
			cout << landmarks.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
		cout << "" << endl;
	}
	cout << "Number of landmarks = " << Si_1.Xi.cols << endl;
	Mat temp;
	vconcat(landmarks.row(0), landmarks.row(1), temp);
	vconcat(temp, landmarks.row(2), Si_1.Xi);
	
	//Si_1.Xi = linearTriangulation(points1Mat, points2Mat, M1, M2);
	
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
	Mat Ii_gray, Ii_1_gray;
	cvtColor(Ii, Ii_gray, COLOR_BGR2GRAY );
	cvtColor(Ii_1, Ii_1_gray, COLOR_BGR2GRAY );
	
	imshow("Ii_gray image", Ii_gray);
	//waitKey(0);
	imshow("Ii_1_gray image", Ii_1_gray);
	//waitKey(0);
	
	// new state
	//state Si;

	// Variables 
	int r_T = 25; // 15
	int num_iters = 25; 
	double lambda = 0.1;
	int nr_keep = 0;
	Mat kpold = Mat::zeros(3, Si_1.k, CV_64FC1);
	Mat delta_keypoint;
	
	// Track every keypoint - Maybe utilize parallelization 
	Mat x_T = Mat::zeros(1, 2, CV_64FC1); // Interest point
	cout << "Si_1.k = " << Si_1.k << endl;
	for (int i = 0; i < Si_1.k; i++) {
		x_T.at<double>(0,0) = Si_1.Pi.at<double>(1,i); // x-coordinate in image
		x_T.at<double>(0,1) = Si_1.Pi.at<double>(0,i); // y-coordiante in image
		//x_T.at<double>(0,0) = Si_1.Pi.at<double>(0,i); 
		//x_T.at<double>(0,1) = Si_1.Pi.at<double>(1,i);
		cout << "Keypoint x_T = (" << x_T.at<double>(0,0) << "," << x_T.at<double>(0,1) << ") ";

		delta_keypoint = KLT::trackKLTrobustly(Ii_1_gray, Ii_gray, x_T, r_T, num_iters, lambda);
		
		for (int k = 0; k < delta_keypoint.rows; k++) {

		}
		
		if (delta_keypoint.at<double>(2,0) == 1) {
			nr_keep++;
			kpold.at<double>(2,i) = 1; // The keypoint is reliably matched
		}
		kpold.at<double>(0,i) = delta_keypoint.at<double>(0,0) + Si_1.Pi.at<double>(1,i); // x-coordinate in image
		kpold.at<double>(1,i) = delta_keypoint.at<double>(1,0) + Si_1.Pi.at<double>(0,i); // y-coordinate in image
		cout << "Match = " << kpold.at<double>(2,i) << " at point = (" << kpold.at<double>(0,i) << "," << kpold.at<double>(1,i) << ")" << endl;

	}
	//Mat keypoints_i_1 = Mat::zeros(2, nr_keep, CV_64FC1);
	Mat keypoints_i = Mat::zeros(2, nr_keep, CV_64FC1);
	
	// corresponding landmarks 
	Mat corresponding_landmarks = Mat::zeros(3, nr_keep, CV_64FC1);
	
	nr_keep = 0; 
	for (int j = 0; j < Si_1.k; j++) {
		if (kpold.at<double>(2,j) == 1) {
			// Update matched keypoints 
			// keypoints_i.at<double>(0, nr_keep) = kpold.at<double>(0, j); // This is for debugging 
			// keypoints_i.at<double>(1, nr_keep) = kpold.at<double>(1, j);
			keypoints_i.at<double>(0, nr_keep) = kpold.at<double>(0, j); // y-coordinate in image
			keypoints_i.at<double>(1, nr_keep) = kpold.at<double>(1, j); // x-coordinate in image
			
			// Update landmarks 
			corresponding_landmarks.at<double>(0, nr_keep) = Si_1.Xi.at<double>(0, j);
			corresponding_landmarks.at<double>(1, nr_keep) = Si_1.Xi.at<double>(1, j);
			corresponding_landmarks.at<double>(2, nr_keep) = Si_1.Xi.at<double>(2, j);
			
			nr_keep++;
		}
	}
	cout << "Number of keypoints left = " << keypoints_i.cols << endl;
	waitKey(5000);
	
	cout << "Print of keypoints_i" << endl;
	for (int r = 0; r < keypoints_i.rows; r++) {
		for (int c = 0; c < keypoints_i.cols; c++) {
			cout << keypoints_i.at<double>(r,c) << ", ";
		}
		cout << "" << endl;		
	}
	cout << "" << endl;	
	cout << "Print of corresponding_landmarks" << endl;
	for (int r = 0; r < corresponding_landmarks.rows; r++) {
		for (int c = 0; c < corresponding_landmarks.cols; c++) {
			cout << corresponding_landmarks.at<double>(r,c) << ", ";
		}
		cout << "" << endl;		
	}
	cout << "" << endl;	
	waitKey(5000);
	waitKey(0);
	
	
	// Delete landmarks for those points that were not matched 
	
	
	// Estimate the new pose using RANSAC and P3P algorithm 
	Mat transformation_matrix, best_inlier_mask;
	tie(transformation_matrix, best_inlier_mask) = Localize::ransacLocalization(keypoints_i, corresponding_landmarks, K);
	
	// Remove points that are determined as outliers from best_inlier_mask by using best_inlier_mask
	
	
	
	// Update keypoints in state 
	//Si.k = keypoints_i.cols;
	Si_1.k = keypoints_i.cols;
	//vconcat(keypoints_i.row(1), keypoints_i.row(0), Si.Pi); // Apparently you have to switch rows
	vconcat(keypoints_i.row(1), keypoints_i.row(0), Si_1.Pi); // Apparently you have to switch rows
	//keypoints_i.copyTo(Si.Pi); 
	//corresponding_landmarks.copyTo(Si.Xi);
	corresponding_landmarks.copyTo(Si_1.Xi);
	
	/*
	// Triangulate new points
	Mat M1 = K * prev_transformation_matrix;
	Mat M2 = K * transformation_matrix;
	Si.Xi = linearTriangulation(Si_1.Pi, Si.Pi, M1, M2 );
	*/
	
	//return make_tuple(Si, transformation_matrix); 
	return make_tuple(Si_1, transformation_matrix); 
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
	
	/*
	// Initial frame 0 
	Camera.grab();
	Camera.retrieve( I_i0 ); 
	cout << "Frame I_i0 captured" <<endl;
	//I_i0.convertTo(I_i0, CV_64FC1);
	imshow("Frame I_i0 displayed", I_i0);
	imwrite("cam0.png", I_i0);
	waitKey(0);	// Ensures it is sufficiently far away from initial frame
	waitKey(5000);
	
	// First frame 1
	cout << "Prepare to take image 1" << endl; 
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;
	imshow("Frame I_i1 displayed", I_i1);
	imwrite("cam1.png", I_i1);
	waitKey(0);
	waitKey(5000);
	
	Mat I_i2;
	// First frame 1
	cout << "Prepare to take image 1" << endl; 
	Camera.grab();
	Camera.retrieve ( I_i2 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;
	imshow("Frame I_i1 displayed", I_i2);
	imwrite("cam1.png", I_i2);
	waitKey(0);
	waitKey(5000);
	*/
	
	
	
	
	
	// Test billeder
	I_i0 = imread("cam0.png", IMREAD_UNCHANGED);
	//I_i0.convertTo(I_i0, CV_64FC1);
	imshow("Frame I_i0 displayed", I_i0);
	waitKey(0);
	
	I_i1 = imread("cam1.png", IMREAD_UNCHANGED);
	//I_i1.convertTo(I_i1, CV_64FC1);
	imshow("Frame I_i1 displayed", I_i1);
	waitKey(0);
	
	
	
	// ADVARSEL: POTENTIEL FEJL MED HVORDAN KEYPOINTS ER STRUKTURERET PÅ. mÅSEK SKAL DER BYTTES OM PÅ RÆKKER. 
	
	
	// ############### VO initializaiton ###############
	// VO-pipeline: Initialization. Bootstraps the initial position. 
	state Si_1;
	Mat transformation_matrix;
	//tie(Si_1, transformation_matrix) = initializaiton(I_i0, I_i1, K, Si_1);
	cout << "Transformation matrix Thor " << endl;
	for (int r = 0; r < transformation_matrix.rows; r++) {
		for (int c = 0; c < transformation_matrix.cols; c++) {
			cout << transformation_matrix.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	
	
	/*
	cout << "State Si_1 before initializaiton" << endl;
	cout << "Number of keypoints k = " << Si_1.k << endl;
	cout << "State Si_1" << endl;
	cout << "Keypoints of state Si = (" << Si_1.Pi.rows << "," << Si_1.Pi.cols << ")" << endl;
	cout << "Landmarks of state Si = " << Si_1.Xi.rows << "," << Si_1.Xi.cols << endl;
	
	for (int i = 0; i < 5; i++) {
		cout << Si_1.Xi.at<double>(0,i) << "," << Si_1.Xi.at<double>(1,i) << ","  << Si_1.Xi.at<double>(2,i) << ","  << Si_1.Xi.at<double>(3,i) << endl;
	}
	*/
	
	
	/*
	// Test of System
	I_i0 = imread("0001.jpg", IMREAD_UNCHANGED);
	//I_i0.convertTo(I_i0, CV_64FC1);
	I_i1 = imread("0002.jpg", IMREAD_UNCHANGED);
	//I_i1.convertTo(I_i1, CV_64FC1);
	
	Mat K2 = Mat::zeros(3, 3, CV_64FC1);
	K2.at<double>(0,0) = 1379.74;
	K2.at<double>(0,2) = 760.35;
	K2.at<double>(1,1) = 1382.08;
	K2.at<double>(1,2) = 503.41;
	K2.at<double>(2,2) = 1;
	
	state Si_1;
	
	int N = 84;
	vector<Point2f> points1(N);
	vector<Point2f> points2(N);
	
	Mat temp_points1Mat = Mat::zeros(2, N, CV_64FC1); 
	Mat temp_points2Mat = Mat::zeros(2, N, CV_64FC1); 
	
	ifstream MyReadFile("p1.txt");	
	// Fejl i hvordan det loades ind 
	if (MyReadFile.is_open()) {
		for (int i = 0; i < N; i++) {
			MyReadFile >> temp_points1Mat.at<double>(0,i);
			MyReadFile >> temp_points1Mat.at<double>(1,i);	
		}
	}
	MyReadFile.close();
	cout << "points1Mat" << endl;
	for (int i = 0; i < temp_points1Mat.rows; i++) {
		for (int j = 0; j < temp_points1Mat.cols; j++) {
			cout << temp_points1Mat.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	
	ifstream MyRead2File("p2.txt");	
	// Fejl i hvordan det loades ind 
	if (MyRead2File.is_open()) {
		for (int i = 0; i < N; i++) {
			MyRead2File >> temp_points2Mat.at<double>(0,i);
			MyRead2File >> temp_points2Mat.at<double>(1,i);	
		}
	}
	MyRead2File.close();
	cout << "points2Mat" << endl;
	for (int i = 0; i < temp_points2Mat.rows; i++) {
		for (int j = 0; j < temp_points2Mat.cols; j++) {
			cout << temp_points2Mat.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	
	
	for (int i = 0; i < N; i++) {
		points1[i] = Point2f(temp_points1Mat.at<double>(0,i),temp_points1Mat.at<double>(1,i));
		points2[i] = Point2f(temp_points2Mat.at<double>(0,i),temp_points2Mat.at<double>(1,i));
	}
	
	cout << "2D points points1" << endl;
	for (int i = 0; i < N; i++) {
		cout << points1[i] << ", ";
	}
	
	cout << "2D points points2" << endl;
	for (int i = 0; i < N; i++) {
		cout << points2[i] << ", ";
	}
	
	
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
			points1Mat.at<double>(1,temp_index) = temp_points1Mat.at<double>(1,i);
			points2Mat.at<double>(0,temp_index) = temp_points2Mat.at<double>(0,i);
			points2Mat.at<double>(1,temp_index) = temp_points2Mat.at<double>(1,i);
			temp_index++;
		}
	}
	
	cout << "Number of reliably matched keypoints (using RANSAC) in initializaiton = " << N_inlier << endl;
	Si_1.k = N_inlier;
	
	// Update of reliably matched keypoints
	Si_1.Pi = points2Mat;
	
	cout << "Print of points1Mat " << endl;
	for (int r = 0; r < points1Mat.rows; r++) {
		for (int c = 0; c < points1Mat.cols; c++) {
			cout << points1Mat.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "Number of keypoints = " << points1Mat.cols << endl;
	
	cout << "Print of points2Mat " << endl;
	for (int r = 0; r < points2Mat.rows; r++) {
		for (int c = 0; c < points2Mat.cols; c++) {
			cout << points2Mat.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "Number of keypoints = " << points2Mat.cols << endl;
	
	// Estimate Essential Matrix
	Mat essential_matrix = estimateEssentialMatrix(fundamental_matrix, K2);	
	
	// Find position and rotation from images
	//Mat essential_matrix = (Mat_<double>(3,3) << -0.10579, -0.37558, -0.5162047, 4.39583, 0.25655, 19.99309, 0.4294123, -20.32203997, 0.023287939);
	// Find the rotation and translation assuming the first frame is taken with the drone on the ground 
	Mat transformation_matrix = findRotationAndTranslation(essential_matrix, K2, points1Mat, points2Mat);
	for (int i = 0; i < transformation_matrix.rows; i++) {
		for (int j = 0; j < transformation_matrix.cols; j++) {
			cout << transformation_matrix.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	// Update State with regards to 3D (triangulated points)
	// Triangulate initial point cloud
	Mat M1 = K2 * (Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
	//Mat M2 = K2 * transformation_matrix;
	Mat M2 = Mat::zeros(3,4, CV_64FC1);
	M2.at<double>(0,0) = 1501.945774541746;
	M2.at<double>(0,1) = -5.59695714;
	M2.at<double>(0,2) = 475.333729;
	M2.at<double>(0,3) = 1354.823757;
	M2.at<double>(1,0) = 104.776265062;
	M2.at<double>(1,1) = 1382.7595877;
	M2.at<double>(1,2) = 490.47386402700;
	M2.at<double>(1,3) = -13.020156233;
	M2.at<double>(2,0) = 0.195902438530;
	M2.at<double>(2,1) = 0.001384484989;
	M2.at<double>(2,2) = 0.980622413460;
	M2.at<double>(2,3) = -0.031844805466;
	cout << "Matrix M1 " << endl;
	for (int r = 0; r < M1.rows; r++) {
		for (int c = 0; c < M1.cols; c++) {
			cout << M1.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
		cout << "" << endl;
	}
	
	cout << "Matrix M2 " << endl;
	for (int r = 0; r < M2.rows; r++) {
		for (int c = 0; c < M2.cols; c++) {
			cout << M2.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
		cout << "" << endl;
	}
	
	
	Mat landmarks = linearTriangulation(points1Mat, points2Mat, M1, M2);
	
	cout << "Print landmarks " << endl;
	for (int r = 0; r < landmarks.rows; r++) {
		for (int c = 0; c < landmarks.cols; c++) {
			cout << landmarks.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
		cout << "" << endl;
	}
	cout << "Number of landmarks = " << landmarks.cols << endl;
	Mat temp;
	vconcat(landmarks.row(0), landmarks.row(1), temp);
	vconcat(temp, landmarks.row(2), Si_1.Xi);
	
	//Si_1.Xi = linearTriangulation(points1Mat, points2Mat, M1, M2);
	
	cout << "Print of 3D landmarks " << endl;
	for (int r = 0; r < Si_1.Xi.rows; r++) {
		for (int c = 0; c < Si_1.Xi.cols; c++) {
			cout << Si_1.Xi.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
		cout << "" << endl;
	}
	cout << "Number of landmarks = " << Si_1.Xi.cols << endl;
	*/
	
	
	
	// ############### VO Continuous ###############
	bool continueVOoperation = true;
	bool pipelineBroke = false;
	bool output_T_ready = true;
	
	// Needed variables
	state Si;
	Mat Ii;
	double threshold_angle = 20; // In degrees
	Mat extracted_keypoints = Mat::zeros(1, 100, CV_64FC1); // Remeber that 100 should be replaced by Si.num_candidates in a smart way
	
	// Mat Ii_1 = I_i1;
	// For test 
	
	
	// Debug variable
	int stop = 2;
	int iter = 0;
	Mat Ii_1 = imread("cam1.png", IMREAD_UNCHANGED);
	
	while (continueVOoperation == true && pipelineBroke == false && stop < 2) {
		cout << "Begin Continuous VO operation " << endl;
		
		/*
		// Take new image 
		Camera.grab();
		Camera.retrieve( Ii );
		
		imshow("New Frame", Ii);
		cout << "New image aquired" << endl;
		//waitKey(0);
		*/
		
		Ii = imread("cam1.png", IMREAD_UNCHANGED);
		//imshow("Continous operation frame", Ii);
		//waitKey(0);
		//Ii.convertTo(Ii, CV_64FC1);
		for (int r = 0; r < K.rows; r++) {
			for (int c = 0; c < K.cols; c++) {
				cout << K.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		
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
		
		
		// Find new candidate keypoints 
		if (iter == 0) {
			Si = newCandidateKeypoints(Ii, Si, transformation_matrix);
			iter++;
		}
		else {
			cout << "Iter 2" << endl;
			Si = continuousCandidateKeypoints(Ii_1, Ii, Si, transformation_matrix, extracted_keypoints);
			cout << "ContinuousCandidateKeypoints" << endl;
			for (int r = 0; r < Si.Ci.rows; r++) {
				for (int c = 0; c < Si.Ci.cols; c++) {
					cout << Si.Ci.at<double>(r,c) << ", ";
				}
				cout << "" << endl;
			}
			waitKey(0);
			tie(Si, extracted_keypoints) = triangulateNewLandmarks( Si, K, transformation_matrix, threshold_angle);
		}
		
		
		// Test of function newCandidateKeypoints 
		cout << "Test of function newCandidateKeypoints" << endl;
		for (int k = 0; k < Si.Pi.cols; k++) {
			double x = Si.Pi.at<double>(0, k);
			double y = Si.Pi.at<double>(1, k);
			circle (Ii, Point(y,x), 5, Scalar(0,0,255), 2,8,0);
		}
		imshow("Corners from Si.Pi", Ii);
		waitKey(0);
		for (int k = 0; k < Si.num_candidates; k++) {
			double x = Si.Ci.at<double>(0, k);
			double y = Si.Ci.at<double>(1, k);
			circle (Ii, Point(y,x), 5, Scalar(255,0,0), 2,8,0);
		}
		imshow("Corners from Si.Ci", Ii);
		waitKey(0);
		// Draw Corners from Si.Ci 
		cout << "Si.Ci" << endl;
		for (int r = 0; r < Si.Ci.rows; r++) {
			for (int c = 0; c < Si.Ci.cols; c++) {
				cout << Si.Ci.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		cout << "Si.Fi" << endl;
		for (int r = 0; r < Si.Fi.rows; r++) {
			for (int c = 0; c < Si.Fi.cols; c++) {
				cout << Si.Fi.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		cout << "Si.Ti" << endl;
		for (int r = 0; r < 6; r++) {
			for (int c = 0; c < 1; c++) {
				cout << Si.Ti.at<double>(r,c) << ", ";
			}
			cout << "" << endl;
		}
		
		
		// Resert old values 
		Ii_1.copyTo(Ii);
		Si_1 = Si;
		cout << "Update of Si.num_candidates = " << Si.num_candidates << endl;
		
		// Debug variable
		stop++;
		if (stop > 10) {
			break;
		}
		
	}
	

	
	
	cout << "VO-pipeline terminated" << endl;


	double time_=cv::getTickCount();
	
	Camera.release();
	
	
	/*
	// Test of KLT
	cout << "Test of KLT in Main function" << endl;
	Mat I1, I2, I1_gray, I2_gray;
	I1 = imread("cam1.png", IMREAD_UNCHANGED);
	I2 = imread("cam2.png", IMREAD_UNCHANGED);
	cvtColor(I1, I1_gray, COLOR_BGR2GRAY );
	cvtColor(I2, I2_gray, COLOR_BGR2GRAY );
	
	int r_T = 15; 
	int num_iters = 50; 
	double lambda = 0.1;
	int nr_keep = 0;
	Mat delta_keypoint;
	
	Mat x_T = Mat::zeros(1, 2, CV_64FC1); 
	//x_T.at<double>(0,0) = 1023; 
	//x_T.at<double>(0,1) = 316;
	//x_T.at<double>(0,0) = 1023/2.0; 
	//x_T.at<double>(0,1) = 316/2.0;
	x_T.at<double>(0,0) = 994/1.0; 
	x_T.at<double>(0,1) = 340/1.0;
	cout << "x_T = (" << x_T.at<double>(0,0) << "," << x_T.at<double>(0,1) << ")" << endl;
	circle (I1, Point(1023,316), 5,  Scalar(0,0,255), 2,8,0);
	imshow("frame I1", I1);
	imshow("frame I1_gray", I1_gray);
	waitKey(0);
	//waitKey(5000);
	
	Mat dst1, dst2;
	//resize(I2_gray, dst, Size(), 0.25, 0.25, INTER_AREA)
	resize(I1_gray, dst1, Size(), 0.5, 0.5, INTER_CUBIC);
	resize(I2_gray, dst2, Size(), 0.5, 0.5, INTER_CUBIC);
	
	
	cout << "dst dimensions = (" << dst1.rows << "," << dst1.cols << ")" << endl;
	cout << "cam1 dimensions = (" << I1_gray.rows << "," << I1_gray.cols << ")" << endl;
	
	MatType(I1_gray);
	MatType(dst1);
	
	
	cout << "I1_gray" << endl;
	for (int r = 0; r < 10; r++) {
		for (int c = 0; c < 10; c++) {
			cout << (double) I1_gray.at<uchar>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "" << endl;
	cout << "DST" << endl;
	for (int r = 0; r < 10; r++) {
		for (int c = 0; c < 10; c++) {
			cout << (double) dst1.at<uchar>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	
	//delta_keypoint = KLT::trackKLTrobustly(dst1, dst2, x_T, r_T, num_iters, lambda);
	delta_keypoint = KLT::trackKLTrobustly(I1_gray, I2_gray, x_T, r_T, num_iters, lambda);
	
	cout << "Match = " << delta_keypoint.at<double>(2,0) << " at point = (" << delta_keypoint.at<double>(0,0) << "," << delta_keypoint.at<double>(1,0) << ")" << endl;
	*/
	
	/*
	cout << "Test of test_struct" << endl;
	testStruct hej;
	
	cout << "value = " << hej.test_k << endl;
	
	
	hej.test_k = 5;
	hej.test_Mat = Mat::ones(3, 4, CV_64FC1); 
	
	cout << "value = " << hej.test_k << endl;
	for (int r = 0; r < hej.test_Mat.rows; r++) {
		for (int c = 0; c < hej.test_Mat.cols; c++) {
			cout << hej.test_Mat.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	hej = hejmeddig(hej);
	
	cout << "value = " << hej.test_k << endl; 
	
	for (int r = 0; r < hej.test_Mat.rows; r++) {
		for (int c = 0; c < hej.test_Mat.cols; c++) {
			cout << hej.test_Mat.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	/*
	 * Method to 
	cout << "P from findLandMark" << endl;
	Mat K2 = (Mat_<double>(3,3) << 359.4280, 0, 303.5964, 0, 359.4280, 92.6078, 0, 0, 1);
	Mat M0 = (Mat_<double>(3,4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	Mat M1 = (Mat_<double>(3,4) << 1, 0, 0, 0.54, 0, 1, 0, 0, 0, 0, 1, 0);
	Mat keypoint0 = (Mat_<double>(3,1) << 228, 28, 1);
	Mat keypoint1 = (Mat_<double>(3,1) << 222, 28, 1);
	Mat P = findLandmark(K2, M0, M1,  keypoint0,  keypoint1);
	
	cout << "P Coordinates" << endl;
	for (int r = 0; r < P.rows; r++) {
		for (int c = 0; c < P.cols; c++) {
			cout << P.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	*/
	/*
	// Test of test_mat
	Mat test_mat = Mat::ones(2,25, CV_64FC1);
	cout << "test_mat " << endl;
	cout << "Dimensions of test_mat = (" << test_mat.rows << "," << test_mat.cols << ")" << endl;
	
	cout << "countNonZero(test_mat) = " << countNonZero(test_mat) << endl;
	*/
	
	/*
	cout << "Test of matrix as vectors " << endl;
	Mat K2 = (Mat_<double>(3,3) << 359.4280, 0, 303.5964, 0, 359.4280, 92.6078, 0, 0, 1.0);
	
	Mat v1 = (Mat_<double>(3,1) << 228, 28, 1);
	Mat v2 = (Mat_<double>(3,1) << 222, 28, 1);
	Mat prikprodukt = v1.at<double>(0,0)*v2.at<dou;
	cout << "prikprodukt = " << prikprodukt << endl
	
	
	cout << "K2 " << endl;
	cout << K2 << endl;
	*/
	
	/*
	#define NUM_THREADS 5;
	pthread_t threads[NUM_THREADS];
	int rc;
	int i;
	
	for (i = 0; i < NUM_THREADS; i++) {
		cout << "main() : creating thread, " << i << endl;
		rc = pthread_create(&threads[i], NULL, PrintHello, (void *)i);
		
		if (rc) {
			cout << "Error:unable to create thread, " << rc << endl;
			exit(-1);
		}
	}
	pthread_exit(NULL);
	*/
	
}










