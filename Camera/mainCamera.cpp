#include <iostream>
#include <mainCamera.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace cv;
using namespace std;

//Mat src, src_gray;
//int thres = 200;
//int max_thres = 255;

 
void draw::circles() {
	// Draw circles
	std::cout << "Start of main File!\n";
	//piCamera.set(cv::CAP_PROP_FRAME_WIDTH, (float)640 );
	
	Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	imshow("Display Window", image);
	waitKey(0);
}

/*
void Harris::corner(int, void*) {
	int blockSize = 2; 
	int apertureSize = 3;
	double k = 0.04; 

	Mat dst = Mat::zeros( src.size(), CV_32FC1 );
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	for (int i = 0; i < dst_norm.rows; i++) {
		for (int j = 0; j < dst_norm.cols; j++) {
			if ( (int) dst_norm.at<float>(i,j) > thres) {
				circle (dst_norm_scaled, Point(j,i), 5, Scalar(0), 2,8,0);
			}

		}

	}
	
	namedWindow( corners_window) ; 
	imshow( corners_windows, dst_norm_scaled);

}
*/





