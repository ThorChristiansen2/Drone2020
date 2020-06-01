#include <iostream>
#include <say-hello/hello.hpp>
#include "mainCamera.hpp"


// Include directories for raspicam
#include <ctime>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <raspicam/raspicam_cv.h>
#include <unistd.h>

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

using namespace cv;
using namespace std;

bool doTestSpeedOnly=false;
const char* source_window = "Source image"; 
//const char* corners_window = "Corners detected";

float getParamVal ( string param,int argc,char **argv,float defvalue=-1 ) {
    int idx=-1;
    for ( int i=0; i<argc && idx==-1; i++ )
        if ( string ( argv[i] ) ==param ) idx=i;
    if ( idx==-1 ) return defvalue;
    else return atof ( argv[  idx+1] );
}

int findParam ( string param,int argc,char **argv ) {
    int idx=-1;
    for ( int i=0; i<argc && idx==-1; i++ )
        if ( string ( argv[i] ) ==param ) idx=i;
    return idx;

}


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


//     Camera.setSharpness ( getParamVal ( "-sh",argc,argv,0 ) );
//     if ( findParam ( "-vs",argc,argv ) !=-1 )
//         Camera.setVideoStabilization ( true );
//     Camera.setExposureCompensation ( getParamVal ( "-ev",argc,argv ,0 ) );


}


void showUsage() {
    cout<<"Usage: "<<endl;
    cout<<"[-gr set gray color capture]\n";
    cout<<"[-test_speed use for test speed and no images will be saved]\n";
    cout<<"[-w width] [-h height] \n[-br brightness_val(0,100)]\n";
    cout<<"[-co contrast_val (0 to 100)]\n[-sa saturation_val (0 to 100)]";
    cout<<"[-g gain_val  (0 to 100)]\n";
    cout<<"[-ss shutter_speed (0 to 100) 0 auto]\n";
    cout<<"[-fps frame_rate (0 to 120) 0 auto]\n";
    cout<<"[-nframes val: number of frames captured (100 default). 0 == Infinite lopp]\n";

    cout<<endl;
}

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
	
	/*
	// Test the Pi-camera
	int res_width = 640;
	int res_height = 480;
	std::cout << "Opening camera with :" << res_width << " x " << res_height << std::endl;
	
	piCamera.set(CAP_PROP_FRAME_WIDTH, (float)res_width );
	piCamera.set(CAP_PROP_FRAME_HEIGHT, (float)res_height );
	piCamera.set(CAP_PROP_FPS, 30);
	
	piCamera.set(CAP_PROP_FORMAT, CV_8UC3); 
	
	if(!piCamera.open()) {
		std::cout << "Did not open\n";
	}
	
	if (piCamera.isOpened()) std::cout << "RaspiCam_CV_open\n";
	*/
	
	waitKey(0);
	//return 0;

}
