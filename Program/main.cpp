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
 * Date: 2.06.2020
 * Objective: The source file mainCamera.cpp contains the functions used
 * by main.cpp to treat the images - find features in the images using 
 * Harris corner etc.
 * Project: Bachelor project 2020
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

// ####################### VO Initialization Pipeline #######################
void drawCorners(Mat img, Matrix keypoints, const char* frame_name) {
	//cout << "Matrix dimension 1: " << keypoints.dim1() << endl;
	//cout << "Matrix dimension 2: " << keypoints.dim2() << endl;
	for (int k = 0; k < keypoints.dim1(); k++) {
		//cout << "Plot corners at (x,y)" << endl;
		double x = keypoints(k,1);
		double y = keypoints(k,2);
		circle (img, Point(y,x), 5, Scalar(200), 2,8,0);
	}
	imshow(frame_name, img);
	waitKey(0);
}


// Initialization part of VO pipeline.
float initializaiton(Mat I_i0, Mat I_i1, Mat K) {
	
	// Transform color images to gray images
	cv::Mat I_i0_gray, I_i1_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cvtColor(I_i1, I_i1_gray, COLOR_BGR2GRAY );
	
	// Get Feature points
	Matrix keypoints_I_i0 = Harris::corner(I_i0, I_i0_gray);
	const char* text0 = "Detected corners *Thor frame I_i0";
	//drawCorners(I_i0, keypoints_I_i0, text0);
	//waitKey(0);
	Matrix keypoints_I_i1 = Harris::corner(I_i1, I_i1_gray);
	const char* text1 = "Detected corners frame I_i1";
	//drawCorners(I_i1, keypoints_I_i1,text1);
	//waitKey(0);
	//cout << "Done with finding keypoints " << endl;
	// Find descriptors for Feature Points
	
	Matrix descriptors_I_i0 = SIFT::FindDescriptors(I_i0_gray, keypoints_I_i0);
	cout << "descriptors_I_i0 dimensions = (" << descriptors_I_i0.dim1() << "," << descriptors_I_i0.dim2() << ")" << endl;
	
	Matrix descriptors_I_i1 = SIFT::FindDescriptors(I_i1_gray, keypoints_I_i1);
	cout << "descriptors_I_i1 dimensions = (" << descriptors_I_i1.dim1() << "," << descriptors_I_i1.dim2() << ")" << endl;
	
	// Match descriptors 
	Matrix matches = SIFT::matchDescriptors(descriptors_I_i0, descriptors_I_i1);
	
	// Find Point correspondences
	// Points from image 0 in row 1 and row 2 
	// Points from image 1 in row 3 and row 
	
	/*
	int point_count = 100;
	vector<Point2f> points1(point_count);
	vector<Point2f> points2(point_count);
	
	for (int i = 0; i < point_count; i++) {
		points1[i] = Point2f(5,2);
		points2[i] = Point2f(3,7);
	}
	
	Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
	*/
	
	
	int N = matches.dim1();
	// For plotting
	Matrix pointCorrespondences(4,N);
	// For efficiency, you should maybe just use vectors instead of creating two new matrices
	Mat points1Mat = Mat::zeros(2, N, CV_64FC1);
	Mat points2Mat = Mat::zeros(2, N, CV_64FC1);
	// For fudamental matrix
	vector<Point2f> points1(N);
	vector<Point2f> points2(N);
	for (int i = 0; i < N; i++) {
		// Be aware of differences in x and y
		points1[i] = Point2f(keypoints_I_i0(matches(i,1),1),keypoints_I_i0(matches(i,1),2));
		points2[i] = Point2f(keypoints_I_i1(matches(i,0),1),keypoints_I_i1(matches(i,0),2));
		
		points1Mat.at<double>(0,i) = keypoints_I_i0(matches(i,1),1);
		points1Mat.at<double>(1,i) = keypoints_I_i0(matches(i,1),2); 
		points2Mat.at<double>(0,i) = keypoints_I_i1(matches(i,0),1);
		points2Mat.at<double>(1,i) = keypoints_I_i1(matches(i,0),2);
		
		//pointCorrespondences(0,i) = keypoints_I_i0(matches(i,1),1); // x
		//pointCorrespondences(1,i) = keypoints_I_i0(matches(i,1),2); // y
		//pointCorrespondences(2,i) = keypoints_I_i1(matches(i,0),1); // x2
		//pointCorrespondences(3,i) = keypoints_I_i1(matches(i,0),2); // y2
	}
	
	// Find fudamental matrix 
	Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99,5000);
	
	// Estimate Essential Matrix
	//Mat essential_matrix = estimateEssentialMatrix(fundamental_matrix, K);
	
	
	Mat essential_matrix = (Mat_<double>(3,3) << -0.10579, -0.37558, -0.5162047, 4.39583, 0.25655, 19.99309, 0.4294123, -20.32203997, 0.023287939);
	// Find the rotation and translation assuming the first frame is taken with the drone on the ground 
	Mat transformation_matrix = findRotationAndTranslation(essential_matrix, K, points1Mat, points2Mat);
	for (int i = 0; i < transformation_matrix.rows; i++) {
		for (int j = 0; j < transformation_matrix.cols; j++) {
			cout << transformation_matrix.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	// Triangulate initial point cloud
	
	
	/*
	// Plot Point corespondences
	for (int i = 0; i < matches.dim1(); i++) {
		//double x = keypoints_I_i1(matches(i,0),1);
		//double y = keypoints_I_i1(matches(i,0),2);
		//double x2 = keypoints_I_i0(matches(i,1),1);
		//double y2 = keypoints_I_i0(matches(i,1),2);
		double x = pointCorrespondences(0,i);
		double y = pointCorrespondences(1,i);
		double x2 = pointCorrespondences(2,i);
		double y2 = pointCorrespondences(3,i);
		//line(I_i1,Point(y2,x2),Point(y,x),Scalar(0,255,0),3);
		line(I_i1,Point(y,x),Point(y2,x2),Scalar(0,255,0),3);
		circle (I_i1, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
		//imshow("Matched features I1", I_i1);
		//waitKey(0);
		//circle (I_i0, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0
		//circle (I_i0, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
		//imshow("Matched features I0", I_i0);
		//waitKey(0);
		
		
		For drawing circles
		circle (I_i1, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
		imshow("Matched features I1", I_i1);
		waitKey(0);
		imshow("Matched features I0", I_i0);
		waitKey(0);
		
	}
	*/
	//imshow("Matched features I1", I_i1);
	//waitKey(0);
	
	
	
	// Find position and rotation from images
	
	
	
	// Should return pose of drone
	return 0;
}

// Function to determine data type of image contained in OpenCV Mat object.
// Example of use for an image I_i0_gray in Opencv Mat Object
// string ty = type2str(I_i0_gray.type() ); 
// printf("Matrix: %s %dx%d \n ", ty.c_str(), I_i0_gray.cols, I_i0_gray.rows );
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
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
	Mat K = Mat::ones(3,3,CV_64FC1);
	
	
	
	cv::Mat I_i0, I_i1, image;
	
	int nCount=getParamVal ( "-nframes",argc,argv, 100 );
	cout<<"Capturing"<<endl;
	
	// Initialization
	Camera.grab(); // You need to take an initial image in order to make the camera work
	Camera.retrieve( image ); 
	cout << "Image captured" <<endl;
	waitKey(1000);
	
	
	// Initial frame 0 
	Camera.grab();
	Camera.retrieve( I_i0 ); 
	cout << "Frame I_i0 captured" <<endl;
	//imshow("Frame I_i0", I_i0);
	
	//waitKey(0);	// Ensures it is sufficiently far away from initial frame
	// First frame 1 
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;
	//waitKey(0);
	
	// VO-pipeline: Initialization. Bootstraps the initial position. 
	
	
	//initializaiton(I_i0, I_i1, K);
	
	/*
	Mat W = (Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	for (int i = 0; i < W.rows; i++) {
		for (int j = 0; j < W.cols; j++) {
			cout << W.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	
	/*
	int point_count = 100;
	vector<Point2f> points1(point_count);
	vector<Point2f> points2(point_count);
	
	for (int i = 0; i < point_count; i++) {
		points1[i] = Point2f(5,2);
		points2[i] = Point2f(3,7);
	}
	
	Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
	
	cout << "Fundamental matrix" << endl;
	for (int i = 0; i < 
	*/
	
	/*
	Mat p1 = Mat::zeros(3, 2, CV_64FC1);
	Mat p2 = Mat::zeros(3, 2, CV_64FC1);
	Mat M1 = Mat::zeros(3, 4, CV_64FC1);
	M1.at<double>(0,0) = 500; 
	M1.at<double>(0,2) = 320; 
	M1.at<double>(1,1) = 500; 
	M1.at<double>(1,2) = 240;
	M1.at<double>(2,2) = 1; 
	
	
	cout << "Matrix M1" << endl;
	for (int i = 0; i < M1.rows; i++) {
		for (int j = 0; j < M1.cols; j++) {
			cout << M1.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	
	Mat M2 = Mat::zeros(3, 4, CV_64FC1);
	M2.at<double>(0,0) = 500; 
	M2.at<double>(0,2) = 320; 
	M2.at<double>(0,3) = -100;
	M2.at<double>(1,1) = 500;
	M2.at<double>(1,2) = 240;
	M2.at<double>(2,2) = 1;
	
	
	cout << "Matrix M2" << endl;
	for (int i = 0; i < M2.rows; i++) {
		for (int j = 0; j < M2.cols; j++) {
			cout << M2.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	
	
	p1.at<double>(0,0) = 4492.45639;
	p1.at<double>(1,0) = 4004.7998;
	p1.at<double>(2,0) = 14.8799;
	
	p2.at<double>(0,0) = 4392.45639;
	p2.at<double>(1,0) = 4004.799;
	p2.at<double>(2,0) = 14.879;
	
	
	
	p1.at<double>(0,1) = 626.036988;
	p1.at<double>(1,1) = 581.456;
	p1.at<double>(2,1) = 3.5127;
	
	p2.at<double>(0,1) = 526.036988;
	p2.at<double>(1,1) = 581.456011;
	p2.at<double>(2,1) = 3.5127626;
	*/
	
	
	/*
	p1(0,0) = 2341.900;
	p1(1,0) = 2067.790;
	p1(2,0) = 7.0425;
	
	p2(0,0) = 2241.900;
	p2(1,0) = 2067.790;
	p2(2,0) = 7.0425;
	*/
	
	
	//Mat P = linearTriangulation(p1, p2, M1, M2);
	
	
	
	
	/*
	 * To access a pixel element in an image, you can use two different methods.  
	 * Either you can get it by: cout << (int) I_i0_gray.at<uchar>(k,j) << ", ";
	 * Here I_i0_gray is the image, which is a matrix Mat, and "uchar" is the type 
	 * of the data in the matrix. (k,j) is the position of the pixel.
	 * Alternatively, you can use the method:
	 * Scalar Intenisty = I_i0_gray.at<uchar>(k,j);
	 * Intenisty.val[0]
	 */
	
	

	// Draw rectangle on the image 
	//rectangle(I_i0_gray, r, Scalar(255), 1, 8, 0);
	//imshow("Image I_i0_gray w. rectangle", I_i0_gray);
	//waitKey(0);
	//cout << "Rectangle drawn" << endl;
	
	// VO-pipeline: 


	double time_=cv::getTickCount();
	
	Camera.release();
}










