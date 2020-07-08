#ifndef MAINCAMERA_HPP_INCLUDED
#define MAINCAMERA_HPP_INCLUDED 

// Libraries from opencv2
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

// Other libaries
#include "Matrix.h"
#include <math.h>
#include <tuple> 

using namespace cv;

//using namespace Numeric_lib;
using Matrix = Numeric_lib::Matrix<double,2>;
using Vector = Numeric_lib::Matrix<double,1>;

/*
namespace draw {

	void circles();

}	// namespace draw
*/

/*
 * Usually, you have to write cv::Mat, but since you have written 'using 
 * namespace cv', this is not necessary. 
*/ 

namespace Harris {
	Matrix corner(Mat src, Mat src_gray);
	//void corner(Mat src, Mat src_gray, bool display);

}	// Harris Corner

namespace SIFT {
	Matrix FindDescriptors(Mat src, Matrix keypoints);
	Matrix matchDescriptors(Matrix descriptor1, Matrix descriptor2);
	
}	// SIFT



namespace KLT {
	Mat trackKLTrobustly(Mat I_R, Mat I, Mat keypoint, int r_T, int num_iters, double lambda);
}

namespace Localize {
	std::tuple<Mat, Mat> ransacLocalization(Mat keypoints_i, Mat corresponding_landmarks, Mat K);
}


// Estimate position of camera 
Mat linearTriangulation(Mat p1, Mat p2, Mat M1, Mat M2);
Mat estimateEssentialMatrix(Mat fundamental_matrix, Mat K);
Mat findRotationAndTranslation(Mat essential_matrix, Mat K, Mat points1Mat, Mat points2Mat);


// For KLT
//Mat trackKLT(Mat I_R, Mat I, Mat x_T, int r_T, int num_iters);
//Mat getSimWarp(double dx, double dy, double alpha_deg, double lambda);
//Mat warpImage(Mat I_R, Mat W);
Mat solveQuartic(Mat factors);
Mat p3p(Mat worldPoints, Mat imageVectors);

// Helper funcitons
void MatType( Mat inputMat );
#endif
