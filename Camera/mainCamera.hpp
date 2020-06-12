#ifndef MAINCAMERA_HPP_INCLUDED
#define MAINCAMERA_HPP_INCLUDED 

// Libraries from opencv2
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>

// Other libaries
#include "Matrix.h"
#include <math.h>

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


Matrix linearTriangulation(Matrix p1, Matrix p2, Matrix M1, Matrix M2);





#endif
