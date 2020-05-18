#ifndef MAINCAMERA_HPP_INCLUDED
#define MAINCAMERA_HPP_INCLUDED 

// Libraries from opencv2
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

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
	void corner(Mat src, Mat src_gray);
	//void corner(int,void*);

}	// Harris Corner



#endif
