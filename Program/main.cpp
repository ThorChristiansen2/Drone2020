#include <iostream>
#include <say-hello/hello.hpp>
#include <mainCamera.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace cv;
using namespace std;

//const char* source_window = "Source image"; 
//const char* corners_iwndow = "Corners detected";

int main() {
	std::cout << "main Program!\n";
	hello::say_hello();
	Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	imshow("Display Window", image);
	//draw::circles();
	//cvtColor(src,srcd_gray, COLOR_BGR2GRAY);
	
	//namedWindow(source_window);
	//createTrackbar("Threshold: ", source_window, &thresh, max_thresh, corner);
	//imshow(source_window, src);
	
	//corner(0,0);
	waitKey(0);
	return 0;

}
