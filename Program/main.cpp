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
 * Date: 1.06.2020
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
float initializaiton() {
	
	
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
	cv::Mat image;
	int nCount=getParamVal ( "-nframes",argc,argv, 100 );
	cout<<"Capturing"<<endl;
	
	for (int k = 0; k <= 4; k++) {
		Camera.grab();
		Camera.retrieve ( image );
		namedWindow( source_window);
		imshow(source_window,image);
		cout << "New Image displayed" <<endl;
		waitKey(3000);
		cout << "New Image terminated" <<endl;
		destroyWindow(source_window);
	}

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









