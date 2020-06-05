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
float initializaiton(Mat I_i0, Mat I_i1) {
	
	// Transform color images to gray images
	cv::Mat I_i0_gray, I_i1_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cvtColor(I_i1, I_i1_gray, COLOR_BGR2GRAY );
	
	// Get Feature points
	bool display = false; // Parameter that displays images
	
	Matrix keypoints_I_i0 = Harris::corner(I_i0, I_i0_gray, display);
	const char* text1 = "Detected corners *Thor frame I_i0";
	drawCorners(I_i0, keypoints_I_i0,text1);
	
	Matrix keypoints_I_i1 = Harris::corner(I_i1, I_i1_gray, display);
	const char* text2 = "Detected corners frame I_i1";
	drawCorners(I_i1, keypoints_I_i1,text2);
	
	
	// Find descriptors for Feature Points
	// Write Sift function
	Matrix descriptors_I_i0 = SIFT::FindDescriptors(I_i0, keypoints_I_i0);
	//SIFT::operator()(I_i0_gray,
	Matrix descriptors_I_i1 = SIFT::FindDescriptors(I_i0, keypoints_I_i1);
	
	// Match Feature Points
	// Using Least Squared Distance
	
	// Find position and rotation from images
	
	
	
	// Should return pose of drone
	return 0;
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
	imshow("Frame I_i0", I_i0);
	
	waitKey(2000);	// Ensures it is sufficiently far away from initial frame
	// First frame 1 
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;

	// VO-pipeline: Initialization. Bootstraps the initial position. 
	//initializaiton(I_i0, I_i1);
	Mat I_i0_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cout << "Print I_i0" << endl;
	for (int k = 4; k< 8; k++) {
		for (int j = 4; j< 7; j++) {
			cout << I_i0_gray.at<int>(k,j) << ", ";
		}
		cout << "" << endl;
	}
	
	cout << "Matrix A" << endl;
	Mat temp = I_i0_gray.colRange(4,7).rowRange(4,8);
	Mat A = Mat::eye(3,4,CV_32F);
	A.copyTo(temp);
	for (int k = 0; k< 4; k++) {
		for (int j = 0; j< 3; j++) {
			cout << A.at<int>(k,j) << ", ";
		}
		cout << "" << endl;
	}
	
	
	// VO-pipeline: 


	double time_=cv::getTickCount();
	
	Camera.release();
}










