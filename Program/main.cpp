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

// Initialization part of VO pipeline.
float initializaiton(Mat I_i0, Mat I_i1) {
	
	// Transform color images to gray images
	cv::Mat I_i0_gray, I_i1_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cvtColor(I_i1, I_i1_gray, COLOR_BGR2GRAY );
	
	// Get Feature points
	bool display = false; // Parameter that displays images
	cout << "Start finding corners" << endl;
	Matrix keypoints = Harris::corner(I_i0, I_i0_gray, display);
	cout << "Plot corners at (x,y)" << endl;
	//cout << "Matrix dimension 1: " << keypoints.dim1() << endl;
	//cout << "Matrix dimension 2: " << keypoints.dim2() << endl;
	
	for (int k = 0; k < keypoints.dim1(); k++) {
		cout << "(y = " << keypoints(k,1) << ",x = " << keypoints(k,2) << ")" << endl;
		double x = keypoints(k,1);
		double y = keypoints(k,2);
		circle (I_i0, Point(y,x), 5, Scalar(200), 2,8,0);
	}
	imshow( "Detected corners", I_i0);
	waitKey(0);
	
	
	// Display images
	if (display == true) {
		cout << "Display I_i0" <<endl;
		imshow("Image I_i0",I_i0);
		waitKey(4000);
		Harris::corner(I_i0, I_i0_gray, display);
		waitKey(4000);
		/*
		if (display == true) {
			destroyWindow("Corners detected");
		}
		cout << "Display I_i1" << endl;
		imshow("Image I_i1",I_i1);
		waitKey(4000);
		Harris::corner(I_i1, I_i1_gray, display);
		waitKey(4000);
		*/
	}
	return 0;
}



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
	//cv::Mat I_i1;
	//cv::Mat image;
	
	int nCount=getParamVal ( "-nframes",argc,argv, 100 );
	cout<<"Capturing"<<endl;
	
	// Initialization
	Camera.grab(); // You need to take an initial image in order to make the camera work
	Camera.retrieve( image ); 
	cout << "Image captured" <<endl;
	waitKey(3000);
	
	Camera.grab();
	Camera.retrieve( I_i0 ); // Frame 0
	cout << "Frame I_i0 captured" <<endl;
	imshow("Frame I_i0", I_i0);
	waitKey(0); // Wait for 3 seconds
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;
	//namedWindow( source_window);
	//imshow( source_window, I_i1);
	//waitKey(0);
	initializaiton(I_i0, I_i1);
	


	double time_=cv::getTickCount();
	
	Camera.release();
}
/*
const char* source_window = "Source image"; 


int main( int argc,char **argv ) {
	
	
	std::cout << "main Program!\n";
	hello::say_hello();

	// Start the camera
	Mat src, src_gray;
	src = imread("/home/pi/Desktop/imagetest.jpg");
	imshow("Display image",src);
	
	cvtColor(src,src_gray,	COLOR_BGR2GRAY );
	namedWindow( source_window);
	//createTrackbar( "Threshold: ", source_window, &thres, max_thresh, Harris::corner);
	imshow(source_window, src_gray);
	Harris::corner(src,src_gray);
	
	
	
	// Test af raspicam
	if ( argc==1 ) {
        cerr<<"Usage (-help for help)"<<endl;
	}
	if ( findParam ( "-help",argc,argv ) !=-1 ) {
	    showUsage();
	    return -1;
	}
	raspicam::RaspiCam_Cv Camera;
	processCommandLine ( argc,argv,Camera );
	cout<<"Connecting to camera"<<endl;
	if ( !Camera.open() ) {
	    cerr<<"Error opening camera"<<endl;
	    return -1;
	}
	cout<<"Connected to camera ="<<Camera.getId() <<endl;
	
	
	waitKey(0);
	//return 0;

}
*/









