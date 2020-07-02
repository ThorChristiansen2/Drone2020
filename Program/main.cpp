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
 * Lær hvordan man SSH'er ind på raspberry pi'en
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
	//const char* text0 = "Detected corners *Thor frame I_i0";
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
	
	
	
	// ######################### SIFT ######################### 
	Matrix keypoints_I_i1 = Harris::corner(I_i1, I_i1_gray);
	const char* text1 = "Detected corners frame I_i1";
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
	
	cout << "Number of keypoints N in initializaiton = " << N << endl;
	Si_1.k = N;
	// For plotting
	// For efficiency, you should maybe just use vectors instead of creating two new matrices
	Mat points1Mat = Mat::zeros(2, N, CV_64FC1);
	Mat points2Mat = Mat::zeros(2, N, CV_64FC1);
	// For fudamental matrix
	vector<Point2f> points1(N);
	vector<Point2f> points2(N);
	for (int i = 0; i < N; i++) {
		
		
		// ########## KLT ##########
		/*
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
	
		//  #### SIFT #### 
		// Be aware of differences in x and y
		points1[i] = Point2f(keypoints_I_i0(matches(i,1),1),keypoints_I_i0(matches(i,1),2));
		points2[i] = Point2f(keypoints_I_i1(matches(i,0),1),keypoints_I_i1(matches(i,0),2));
		
		points1Mat.at<double>(0,i) = keypoints_I_i0(matches(i,1),1);
		points1Mat.at<double>(1,i) = keypoints_I_i0(matches(i,1),2); 
		points2Mat.at<double>(0,i) = keypoints_I_i1(matches(i,0),1);
		points2Mat.at<double>(1,i) = keypoints_I_i1(matches(i,0),2);
		
	
		/*
		double x = keypoints_I_i1(matches(i,0),1);
		double y = keypoints_I_i1(matches(i,0),2);
		double x2 = keypoints_I_i0(matches(i,1),1);
		double y2 = keypoints_I_i0(matches(i,1),2);
		line(I_i1,Point(y,x),Point(y2,x2),Scalar(0,255,0),3);
		circle (I_i1, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
		circle (I_i0, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
		*/
		
		
		
	}
	//imshow("Match",I_i1);
	//waitKey(0);
	
	
	// Update State  with regards to keypoints in frame Ii_1
	Si_1.Pi = points2Mat;
	
	
	// Find fudamental matrix 
	// Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, 5000);
	Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, 5000, noArray()); // 1 should be changed ot 3 
	
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
	Si_1.Xi = linearTriangulation(points1Mat, points2Mat, M1, M2 );
	
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
	Si.Pi = keypoints_i; 
	Si.Xi = corresponding_landmarks;
	
	// Estimate the new pose using RANSAC and P3P algorithm 
	Mat transformation_matrix, best_inlier_mask;
	tie(transformation_matrix, best_inlier_mask) = Localize::ransacLocalization(keypoints_i, corresponding_landmarks, K);
	
	
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
	Mat K = (Mat_<double>(3,3) << 769.893, 0, 2.5, 0,1613.3,4, 0, 0, 1);
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
	//Camera.grab(); // You need to take an initial image in order to make the camera work
	//Camera.retrieve( image ); 
	//cout << "Image captured" <<endl;
	//waitKey(1000);
	
	
	// Initial frame 0 
	Camera.grab();
	Camera.retrieve( I_i0 ); 
	cout << "Frame I_i0 captured" <<endl;
	//imshow("Frame I_i0 displayed", I_i0);
	//waitKey(0);	// Ensures it is sufficiently far away from initial frame
	
	// First frame 1 
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;
	//imshow("Frame I_i1 displayed", I_i1);
	//waitKey(0);
	
	/*
	// Test of function findRotationAndTranslation(essential_matrix, K, points1Mat, points2Mat);
	Mat E = Mat::zeros(3, 3, CV_64FC1);
	E.at<double>(0,0) = 0.0181;
	E.at<double>(0,1) = -0.1960;
	E.at<double>(0,2) = 0.0947;
	E.at<double>(1,0) = 4.0793;
	E.at<double>(1,1) = 0.0231;
	E.at<double>(1,2) = 19.2757;
	E.at<double>(2,0) = -0.1954;
	E.at<double>(2,1) = -19.6784;
	E.at<double>(2,2) = 0.0096;
	
	cout << "E matrix " << endl;
	for (int r = 0; r < E.rows; r++) {
		for (int c = 0; c < E.cols; c++) {
			cout << E.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	Mat K1 = Mat::zeros(3, 3, CV_64FC1);
	K1.at<double>(0,0) = 1379.7;
	K1.at<double>(1,1) = 1382.1;
	K1.at<double>(2,2) = 1;
	K1.at<double>(0,2) = 760.35;
	K1.at<double>(1,2) = 503.41;
	
	cout << "K1 matrix " << endl;
	for (int r = 0; r < K1.rows; r++) {
		for (int c = 0; c < K1.cols; c++) {
			cout << K1.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	
	
	string myText;
	
	ifstream MyReadFile("matches0001.txt");
	
	Mat points1 = Mat::zeros(2, 84, CV_64FC1);
	Mat points2 = Mat::zeros(2, 84, CV_64FC1);
	
	if (MyReadFile.is_open()) {
		for (int i = 0; i < 168; i++) {
			if (i < 84) {
				MyReadFile >> points1.at<double>(0,i);
			}
			else   {
				MyReadFile >> points1.at<double>(1,i-84);
			}
		}
	}
	for (int r = 0; r < points1.rows; r++) {
		for (int c = 0; c < points1.cols; c++) {
			cout << points1.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
		cout << "" << endl;
	}
	
	ifstream MyReadFile2("matches0002.txt");
	if (MyReadFile2.is_open()) {
		for (int i = 0; i < 168; i++) {
			if (i < 84) {
				MyReadFile2 >> points2.at<double>(0,i);
			}
			else   {
				MyReadFile2 >> points2.at<double>(1,i-84);
			}
		}
	}
	for (int r = 0; r < points2.rows; r++) {
		for (int c = 0; c < points2.cols; c++) {
			cout << points2.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
		cout << "" << endl;
	}
	
		
	MyReadFile.close();
	MyReadFile2.close();
	
	Mat transformation_matrix = findRotationAndTranslation(E, K1, points1, points2);
	cout << "Transformation matrix " << endl;
	for (int r = 0; r < transformation_matrix.rows; r++) {
		for (int c = 0; c < transformation_matrix.cols; c++) {
			cout << transformation_matrix.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	// Test of solveQuartic function 
	Mat factors = Mat::zeros(1, 4, CV_64FC1);
	factors.at<double>(0,0) = -107244.2653067081;
	factors.at<double>(0,1) = 403760.1352762820;
	factors.at<double>(0,2) = -495973.8974478040;
	factors.at<double>(0,3) = 187754.8091568;
	factors.at<double>(0,4) = 7865.0218773934;
	
	Mat x = solveQuartic(factors);
	
	for (int r = 0; r < x.rows; r++) {
		for (int c = 0; c < x.cols; c++) {
			cout << x.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	
	
	
	/*
	// ############### VO initializaiton ###############
	// VO-pipeline: Initialization. Bootstraps the initial position. 
	state Si_1;
	Si_1.k = 0;
	Mat transformation_matrix;
	tie(Si_1, transformation_matrix) = initializaiton(I_i0, I_i1, K, Si_1);
	cout << "Transformation matrix Thor " << endl;
	for (int r = 0; r < transformation_matrix.rows; r++) {
		for (int c = 0; c < transformation_matrix.cols; c++) {
			cout << transformation_matrix.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	/*
	cout << "State Si_1 before initializaiton" << endl;
	cout << "Number of keypoints k = " << Si_1.k << endl;
	
	
	
	
	cout << "State Si_1" << endl;
	cout << "Number of keypoints k = " << Si_1.k << endl;
	cout << "Matrix P (attribute of S) = " << Si_1.Pi.rows << "," << Si_1.Pi.cols << endl;
	cout << "Matrix X (attribute of S) = " << Si_1.Xi.rows << "," << Si_1.Xi.cols << endl;
	
	for (int i = 0; i < 5; i++) {
		cout << Si_1.Xi.at<double>(0,i) << "," << Si_1.Xi.at<double>(1,i) << ","  << Si_1.Xi.at<double>(2,i) << ","  << Si_1.Xi.at<double>(3,i) << endl;
	}
	
	
	Mat Ii_1 = I_i1;
	Mat Ii;
	
	// Continuous VO operation
	state Si;
	
	// Test of Continuous VO operation 
	Mat I_R = imread("000000.png", IMREAD_UNCHANGED);
	I_R.convertTo(I_R, CV_64FC1);
	Mat I = imread("000001.png", IMREAD_UNCHANGED);
	I.convertTo(I, CV_64FC1);
	
	Si_1.k = 3;
	Mat keypoints = Mat::zeros(2, 3, CV_64FC1);
	keypoints.at<double>(0,0) = 784;
	keypoints.at<double>(1,0) = 100;
	keypoints.at<double>(0,1) = 389;
	keypoints.at<double>(1,1) = 162;
	keypoints.at<double>(0,2) = 399;
	keypoints.at<double>(1,2) = 158;
	
	Si_1.Pi = keypoints;
	
	//tie(Si, transformation_matrix) = processFrame(I, I_R, Si_1, K, transformation_matrix);
	
	cout << "Print of State Si" << endl;
	for (int i = 0; i < Si.k; i++) {
		cout << "(" << Si.Pi.at<double>(0,i) << "," << Si.Pi.at<double>(1,i) << ")" << endl;
	}
	*/
	
	/*
	// ############### VO Continuous ###############
	bool continueVOoperation = true;
	bool pipelineBroke = false;
	bool output_T_ready = true;
	
	// Needed variables
	state Si;
	Mat Ii;
	Mat Ii_1 = I_i1;
	
	// Debug variable
	int stop = 0;
	
	while (continueVOoperation == true && pipelineBroke == false) {
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
	*/
	
	
	
	
	
	
	
	/*
	Mat offset = calibrate_matrix(transformation_matrix);
	
	transformation_matrix = transformation_matrix + offset;
	
	for (int r = 0; r < transformation_matrix.rows; r++) {
		for (int c = 0; c < transformation_matrix.cols; c++) {
			cout << transformation_matrix.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "wait statement" << endl;
	waitKey(5000);
	Mat I_i1_new;
	Camera.grab();
	Camera.retrieve( I_i1_new ); 
	
	// Updated transformation matrix
	transformation_matrix = initializaiton(I_i1, I_i1_new, K);
	
	// Updated transformation matrix
	transformation_matrix = transformation_matrix + offset;
	
	for (int r = 0; r < transformation_matrix.rows; r++) {
		for (int c = 0; c < transformation_matrix.cols; c++) {
			cout << transformation_matrix.at<double>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	/*
	Mat W = (Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	for (int i = 0; i < W.rows; i++) {
		for (int j = 0; j < W.cols; j++) {
			cout << W.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	*/

	double time_=cv::getTickCount();
	
	Camera.release();
}










