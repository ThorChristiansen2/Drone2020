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

// ####################### Raspicam Functions #######################
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

// ####################### VO Initialization Pipeline #######################
void drawCorners(Mat img, Matrix keypoints, const char* frame_name) {
	//cout << "Matrix dimension 1: " << keypoints.dim1() << endl;
	//cout << "Matrix dimension 2: " << keypoints.dim2() << endl;
	for (int k = 0; k < keypoints.dim1(); k++) {
		//cout << "Plot corners at (x,y)" << endl;
		double x = keypoints(k,1);
		double y = keypoints(k,2);
		circle (img, Point(y,x), 5, Scalar(200), 2,8,0);
	}
	imshow(frame_name, img);
	waitKey(0);
}


// Initialization part of VO pipeline.
float initializaiton(Mat I_i0, Mat I_i1) {
	
	// Transform color images to gray images
	cv::Mat I_i0_gray, I_i1_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cvtColor(I_i1, I_i1_gray, COLOR_BGR2GRAY );
	
	// Get Feature points
	bool display = false; // Parameter that displays images
	
	Matrix keypoints_I_i0 = Harris::corner(I_i0, I_i0_gray, display);
	const char* text1 = "Detected corners *Thor frame I_i0";
	drawCorners(I_i0, keypoints_I_i0, text1);
	//waitKey(1000);
	
	Matrix keypoints_I_i1 = Harris::corner(I_i1, I_i1_gray, display);
	const char* text2 = "Detected corners frame I_i1";
	drawCorners(I_i1, keypoints_I_i1,text2);
	//waitKey(1000);
	
	// Find descriptors for Feature Points
	// Write Sift function
	Matrix descriptors_I_i0 = SIFT::FindDescriptors(I_i0_gray, keypoints_I_i0);
	cout << "descriptors_I_i0 dimensions = (" << descriptors_I_i0.dim1() << "," << descriptors_I_i0.dim2() << ")" << endl;
	//waitKey(1000);
	
	Matrix descriptors_I_i1 = SIFT::FindDescriptors(I_i1_gray, keypoints_I_i1);
	cout << "descriptors_I_i1 dimensions = (" << descriptors_I_i1.dim1() << "," << descriptors_I_i1.dim2() << ")" << endl;
	//waitKey(0);
	/*
	cout << "Print of descriptors_I_i1" << endl;
	for (int i = 0; i < descriptors_I_i1.dim1(); i++) {
		for (int j = 0; j < 128; j++) {
			cout << descriptors_I_i1(i,j) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	Matrix matches = SIFT::matchDescriptors(descriptors_I_i0, descriptors_I_i1);
	
	bool valid;
	if (descriptors_I_i0.dim1() < descriptors_I_i1.dim1()) {
		valid = true;
	} 
	else {
		valid = false;
	}
	for (int i = 0; i < matches.dim2(); i++) {
		cout << "Keypoint " << i << " matches with keypoint " << matches(0,i) << endl;
		if (valid == true) {
			double x = keypoints_I_i0(i,1); // Skal måske være 0
			double y = keypoints_I_i0(i,2); // Skal måske være 1
			double x2 = keypoints_I_i1(matches(0,i),1);
			double y2 = keypoints_I_i1(matches(0,i),2);
			circle (I_i0, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
			imshow("Matched features I0", I_i0);
			waitKey(0);
			circle (I_i1, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
			imshow("Matched features I1", I_i1);
			waitKey(0);
		}
		else {
			double x = keypoints_I_i0(matches(0,i),1); // Skal måske være 0
			double y = keypoints_I_i0(matches(0,i),2); // Skal måske være 1
			double x2 = keypoints_I_i1(i,1);
			double y2 = keypoints_I_i1(i,2);
			circle (I_i0, Point(y,x), 5,  Scalar(0,0,255), 2,8,0);
			imshow("Matched features I0", I_i0);
			waitKey(0);
			circle (I_i1, Point(y2,x2), 5, Scalar(0,0,255), 2,8,0);
			imshow("Matched features I1", I_i1);
			waitKey(0);
		}
	}
	
	
	//SIFT::operator()(I_i0_gray,
	//Matrix descriptors_I_i1 = SIFT::FindDescriptors(I_i0, keypoints_I_i1);
	
	// Match Feature Points
	// Using Least Squared Distance
	
	// Find position and rotation from images
	
	
	
	// Should return pose of drone
	return 0;
}

// Function to determine data type of image contained in OpenCV Mat object.
// Example of use for an image I_i0_gray in Opencv Mat Object
// string ty = type2str(I_i0_gray.type() ); 
// printf("Matrix: %s %dx%d \n ", ty.c_str(), I_i0_gray.cols, I_i0_gray.rows );
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

// Function to determine data type of image contained in OpenCV Mat object.
// Example of use: MatType(I_i0_gray); where I_i0_gray is your Mat object.
/*
void MatType( Mat inputMat ) {
	
    int inttype = inputMat.type();

    string r, a;
    uchar depth = inttype & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (inttype >> CV_CN_SHIFT);
    switch ( depth ) {
        case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;  
        case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;  
        case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break; 
        case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break; 
        case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break; 
        case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break; 
        case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break; 
        default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break; 
    }   
    r += "C";
    r += (chans+'0');
    cout << "Mat is of type " << r << " and should be accessed with " << a << endl;
	
}
*/

/*
// Select region of interest from image 
Mat selectRegionOfInterest(Mat img, int y1, int x1, int y2, int x2) {
	Mat ROI;
	if (x1 < 0) {
		x1 = 0;
	}
	if (y1 < 0) {
		y1 = 0;
	}
	if (y2 > img.rows) {
		y2 = img.rows;
	}
	if (x2 > img.cols) {
		x2 = img.cols;
	}
	cout << "Rectangle : (" << y1 << "," << x1 << "," << x2 << "," << y2 << ")" << endl;
	//Rect region(y1, x1, x2-x1, y2-y1);
	Rect region(y1,x1,x2-x1,y2-y1);
	ROI = img(region);
	cout << "Region extracted" << endl;
	return ROI;
}
*/

// ####################### Main function #######################
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
	
	int nCount=getParamVal ( "-nframes",argc,argv, 100 );
	cout<<"Capturing"<<endl;
	
	// Initialization
	Camera.grab(); // You need to take an initial image in order to make the camera work
	Camera.retrieve( image ); 
	cout << "Image captured" <<endl;
	waitKey(1000);
	
	
	// Initial frame 0 
	Camera.grab();
	Camera.retrieve( I_i0 ); 
	cout << "Frame I_i0 captured" <<endl;
	imshow("Frame I_i0", I_i0);
	
	waitKey(1000);	// Ensures it is sufficiently far away from initial frame
	// First frame 1 
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;

	// VO-pipeline: Initialization. Bootstraps the initial position. 
	initializaiton(I_i0, I_i1);
	//MatType(I_i0);
	
	/*
	 * To access a pixel element in an image, you can use two different methods.  
	 * Either you can get it by: cout << (int) I_i0_gray.at<uchar>(k,j) << ", ";
	 * Here I_i0_gray is the image, which is a matrix Mat, and "uchar" is the type 
	 * of the data in the matrix. (k,j) is the position of the pixel.
	 * Alternatively, you can use the method:
	 * Scalar Intenisty = I_i0_gray.at<uchar>(k,j);
	 * Intenisty.val[0]
	 */
	
	/*
	//cout << Data_type_mat; 
	Mat I_i0_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cout << "Print I_i0" << endl;
	//string Data_type_mat = MatType(I_i0_gray);
	cout << "Intensity at (k = " << 0 << ",j= " << 0 << ") : " << (int) I_i0_gray.at<uchar>(0,0) << endl; 
	for (int k = 870; k<= 885; k++) {
		for (int j = 870; j<= 885; j++) {
			cout << (int) I_i0_gray.at<uchar>(k,j) << ", ";
		}
		cout << "" << endl;
	}
	
	cout << "Show image I_i0" << endl;
	imshow("Image I_i0_gray", I_i0_gray);
	
	
	cout << "Matrix A" << endl;
	
	cout << "Intensity before r (k = " << 0 << ",j= " << 0 << ") : " << (int) I_i0_gray.at<uchar>(0,0) << endl; 
	//Rect r(0,0,10,10);
	cout << "Intensity after r (k = " << 0 << ",j= " << 0 << ") : " << (int) I_i0_gray.at<uchar>(0,0) << endl; 
	// Draw rectangle on the image 
	//rectangle(I_i0_gray, r, Scalar(255), 1, 8, 0);
	//imshow("Image I_i0_gray w. rectangle", I_i0_gray);
	//waitKey(0);
	//cout << "Rectangle drawn" << endl;
	
	
	
	//Mat A = I_i0_gray(r);
	Mat A = selectRegionOfInterest(I_i0_gray,877-7,877-7,877+8,877+8);
	cout << "Dimenion of A : " << A.rows << endl;
	cout << "Dimension of A : " << A.cols << endl;
	//cout << "Start point " << A.at<double>(1,1) << endl;
	//Mat A = Mat::zeros(3,4,CV_32FC1);
	//A.copyTo(temp);
	
	for (int k = 0; k<= A.rows; k++) {
		for (int j = 0; j<= A.cols; j++) {
			cout << (int) A.at<uchar>(k,j) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	// VO-pipeline: 


	double time_=cv::getTickCount();
	
	Camera.release();
}










