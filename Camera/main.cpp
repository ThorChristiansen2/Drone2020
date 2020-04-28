#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Own libraries
#include <say-hello/hello.hpp>

using namespace cv;
using namespace std;

bool running = true;

int main()
{
	std::cout << "Start of main File!\n";
	//piCamera.set(cv::CAP_PROP_FRAME_WIDTH, (float)640 );
	
	Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	imshow("Display Window", image);
	hello::say_hello();
	waitKey(0);
	return 0;
}
