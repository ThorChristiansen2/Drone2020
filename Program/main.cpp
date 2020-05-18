#include <iostream>
#include <say-hello/hello.hpp>
#include "mainCamera.hpp"

/* ########################
 * Name: mainCamera.cpp
 * Made by: Thor Christiansen - s173949 
 * Data: 18.05.2020
 * Objective: The source file mainCamera.cpp contains the functions used
 * by main.cpp to treat the images - find features in the images using 
 * Harris corner etc.
 * Project: Bachelor project 2020
 * ########################
*/

using namespace cv;
using namespace std;


const char* source_window = "Source image"; 
//const char* corners_window = "Corners detected";

int main() {
	
	
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
	
	waitKey(0);
	return 0;

}
