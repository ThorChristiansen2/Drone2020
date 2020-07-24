#include <iostream>
#include <say-hello/hello.hpp>
#include "mainCamera.hpp"
#include <unistd.h>

// Include directories for raspicam
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <raspicam/raspicam_cv.h>
//#include "pthread.h"
//#include <cstdlib>

/* ########################
 * Name: mainCamera.cpp
 * Made by: Thor Christiansen - s173949 
 * Date: 29.06.2020
 * Objective: The source file mainCamera.cpp contains the functions used
 * by main.cpp to treat the images - find features in the images using 
 * Harris corner etc.
 * Project: Bachelor project 2020
 * 
 * Problems: 
 * Sæt angle threshold 
 * OVERSkyggende problem: Det ser ud til, at der er et problem med matricen K, som skal kalibreres.
 * Lær hvordan man SSH'er ind på raspberry pi'en
 * Units seem to be meters 
 * The units are in meters.
 * Maybe you fuck up when you try to use KLT in the initialization 
 * See MatLab code week 3 for an efficient way to find Harris Corners 
 * When using KLT: Remember to use gray-scale images // Did not work with resizing the image with a factor of 4 
 * Implementation of ransac  in pose estimation in processFrame function 
 * 
 * Optimize and make match descriptors better - see why there is something wrong with it
 * ########################
*/

// ####################### Raspicam Functions #######################
// Namespace and constants 
using namespace cv;
using namespace std;
const char* source_window = "Source image"; 
bool doTestSpeedOnly=false;
using namespace std::chrono;

//#define NUM_SIFT_THREADS 6
//#define NUM_THREADS 29

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

// ####################### Rtest of parallel threads #######################


// ####################### drawCorners #######################
void drawCorners(Mat img, Mat keypoints, const char* frame_name) {
	for (int k = 0; k < keypoints.cols; k++) {
		double y = keypoints.at<double>(0, k);
		double x = keypoints.at<double>(1, k);
		circle (img, Point(x,y), 5, Scalar(200), 2,8,0);
	}
	imshow(frame_name, img);
	waitKey(0);
}

/*
void *functionKLT(void *threadarg) {
   struct thread_data *my_data;
   my_data = (struct thread_data *) threadarg;
   Mat x_T = Mat::zeros(1, 2, CV_64FC1);
   
   Mat delta_keypoint;
   for (int i = 0; i < my_data->thread_mat.cols; i++) {
	   x_T.at<double>(0,0) = my_data->thread_mat.at<double>(1,i);
	   x_T.at<double>(0,1) = my_data->thread_mat.at<double>(0,i);
	   delta_keypoint = KLT::trackKLTrobustly(my_data->Ii_1_gray, my_data->Ii_gray, x_T, my_data->dwdx, 11, 20, 0.1);
	   double a = delta_keypoint.at<double>(0,0) + my_data->thread_mat.at<double>(1,i);
	   double b = delta_keypoint.at<double>(1,0) + my_data->thread_mat.at<double>(0,i);
	   my_data->thread_mat.at<double>(1,i) = b; // x-coordinate in image
	   my_data->thread_mat.at<double>(0,i) = a; // y-coordinate in image
	   my_data->keep_point = delta_keypoint.at<double>(2,0);
   }
   pthread_exit(NULL);
}
*/

// ####################### VO Initialization Pipeline #######################
tuple<state, Mat> initializaiton(Mat I_i0, Mat I_i1, Mat K, state Si_1) {
	cout << "Begin initialization" << endl;
	
	
	// Transform color images to gray images
	Mat I_i0_gray, I_i1_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	cvtColor(I_i1, I_i1_gray, COLOR_BGR2GRAY );
	
	Mat temp0, temp1, emptyMatrix;
	
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	//Mat keypoints_I_i0 = Harris::corner(I_i0, I_i0_gray, 210, emptyMatrix); // Number of maximum keypoints
	int dim1 = I_i0_gray.rows;
	int dim2 = I_i0_gray.cols;
	Mat I_i0_resized = I_i0_gray.colRange(10,dim2-10).rowRange(10,dim1-10);
	
	//imshow("I_i0_resized", I_i0_resized);
	//waitKey(0);
	
	goodFeaturesToTrack(I_i0_resized, temp0, 300, 0.01, 10, noArray(), 3, true, 0.04);
	Mat keypoints_I_i0 = Mat::zeros(2, temp0.rows, CV_64FC1);
	for (int i = 0; i < keypoints_I_i0.cols; i++) {
		keypoints_I_i0.at<double>(0,i) = temp0.at<float>(i,1) + 10;
		keypoints_I_i0.at<double>(1,i) = temp0.at<float>(i,0) + 10;
	}
	
	/*
	cout << "keypoints_I_i0 = " << keypoints_I_i0 << endl;
	waitKey(0);
	
	Mat temp2; 
	goodFeaturesToTrack(I_i0_gray, temp2, 200, 0.01, 10, noArray(), 3, true, 0.04);
	cout << "temp2 = " << temp2.t() << endl;
	waitKey(0);
	
	for (int i = 0; i < keypoints_I_i0.cols; i++) {
		double y1 = keypoints_I_i0.at<double>(0,i);
		double x1 = keypoints_I_i0.at<double>(1,i);
		cout << "Point1 = (" << y1 << "," << x1 << ")" << endl;
		circle (I_i0, Point(x1,y1), 5, Scalar(0,0,255), 2,8,0);	
		imshow("Test of points I_i0", I_i0);
		waitKey(0);
		double y2 = temp2.at<float>(i,1);
		double x2 = temp2.at<float>(i,0);
		cout << "Point2 = (" << y2 << "," << x2 << ")" << endl;
		circle (I_i0, Point(x2,y2), 5, Scalar(0,255,0), 2,8,0);	
		imshow("Test of points I_i0", I_i0);
		waitKey(0);
	}
	*/
	
	
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2-t1);
	cout << "Finding keypoints_I_i0 took = " << time_span.count() << " seconds" << endl;
	
	/*
	Mat draw_I_i0;
	I_i0.copyTo(draw_I_i0);
	const char* text0 = "Detected corners in frame I_i0";
	drawCorners(draw_I_i0, keypoints_I_i0, text0);
	waitKey(0);
	*/
	

	
	
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	
	//Mat keypoints_I_i1 = Harris::corner(I_i1, I_i1_gray, 210, emptyMatrix); // Number of keypoints that is looked 
	
	
	dim1 = I_i1_gray.rows;
	dim2 = I_i1_gray.cols;
	Mat I_i1_resized = I_i1_gray.colRange(10,dim2-10).rowRange(10,dim1-10);
	//imshow("I_i1_resized", I_i1_resized);
	//waitKey(0);
	goodFeaturesToTrack(I_i1_resized, temp1, 300, 0.01, 10, noArray(), 3, true, 0.04);
	Mat keypoints_I_i1 = Mat::zeros(2, temp1.rows, CV_64FC1);
	for (int i = 0; i < keypoints_I_i1.cols; i++) {
		keypoints_I_i1.at<double>(0,i) = temp1.at<float>(i,1) + 10;
		keypoints_I_i1.at<double>(1,i) = temp1.at<float>(i,0) + 10;
	}
	
	
	
	high_resolution_clock::time_point t4 = high_resolution_clock::now();
	duration<double> time_span1 = duration_cast<duration<double>>(t4-t3);
	cout << "Finding keypoints_I_i1 took = " << time_span1.count() << " seconds" << endl;
	
	/*
	Mat draw_I_i1;
	I_i1.copyTo(draw_I_i1);
	const char* text1 = "Detected corners in frame I_i1";
	drawCorners(draw_I_i1, keypoints_I_i1, text1);
	waitKey(0);
	*/
	
	

	
	
	// ######################### SIFT ######################### 
	//Finding SIFT::descriptors without parallelization 
	high_resolution_clock::time_point t5 = high_resolution_clock::now();

	Mat descriptors_I_i0 = SIFT::FindDescriptors(I_i0_gray, keypoints_I_i0);

	/*
	cout << "descriptors_I_i0" << endl;
	cout << descriptors_I_i0 << endl;
	waitKey(0);
	*/

	
	high_resolution_clock::time_point t6 = high_resolution_clock::now();
	duration<double> time_span2 = duration_cast<duration<double>>(t6-t5);
	cout << "Finding descriptors_I_i0 took = " << time_span2.count() << " seconds" << endl;
	
	
	
	high_resolution_clock::time_point t7 = high_resolution_clock::now();
	
	Mat descriptors_I_i1 = SIFT::FindDescriptors(I_i1_gray, keypoints_I_i1);
	
	
	high_resolution_clock::time_point t8 = high_resolution_clock::now();
	duration<double> time_span3 = duration_cast<duration<double>>(t8-t7);
	cout << "Finding descriptors_I_i0 took = " << time_span3.count() << " seconds" << endl;
	
	
	
	
	high_resolution_clock::time_point t9 = high_resolution_clock::now();
	
	// Match descriptors --> Optimize this part of the code because it takes the longest time to run
	// Matrix matches = SIFT::matchDescriptors(descriptors_I_i0, descriptors_I_i1);
	Mat matches = SIFT::matchDescriptors(descriptors_I_i0, descriptors_I_i1);

	Mat matches2 = SIFT::matchDescriptors(descriptors_I_i1, descriptors_I_i0);

	Mat valid_matches = Mat::zeros(2, matches.cols, CV_64FC1);
	int temp_index = 0;
	for (int i = 0; i < matches.cols; i++) {
		int index_frame0 = matches.at<double>(0,i);
		int index_frame1 = matches.at<double>(1,i);
		
		for (int q = 0; q < matches2.cols; q++) {
			if (matches2.at<double>(1,q) == index_frame0) {
				if (matches2.at<double>(0,q) == index_frame1) {
					// Mutual match
					valid_matches.at<double>(0,temp_index) = index_frame0;
					valid_matches.at<double>(1,temp_index) = index_frame1;
					temp_index++;
				}
			}
		}
	}
	matches = valid_matches.colRange(0,temp_index);
	
	
	high_resolution_clock::time_point t10 = high_resolution_clock::now();
	duration<double> time_span4 = duration_cast<duration<double>>(t10-t9);
	cout << "Finding matches took = " << time_span4.count() << " seconds" << endl;
	
	
	// Find Point correspondences
	// Points from image 0 in row 1 and row 2 
	// Points from image 1 in row 3 and row 	

	//int N = matches.dim2();
	int N = matches.cols;
	cout << "Number of matches = " << N << endl;
	
	high_resolution_clock::time_point t11 = high_resolution_clock::now();
	
	// For plotting
	// For efficiency, you should maybe just use vectors instead of creating two new matrices
	Mat temp_points1Mat = Mat::zeros(2, N, CV_64FC1);
	Mat temp_points2Mat = Mat::zeros(2, N, CV_64FC1);
	// For fudamental matrix
	vector<Point2f> points1(N);
	vector<Point2f> points2(N);
	Mat a, b; 
	I_i0.copyTo(a);
	I_i1.copyTo(b);
	
	
	for (int i = 0; i < N; i++) {
		// Config1
		//  #### SIFT #### 
		// Be aware of differences in x and y
		/* Did not work
		points1[i] = Point2f(keypoints_I_i0.at<double>(0, matches(0,i)),keypoints_I_i0.at<double>(1, matches(0,i)));
		points2[i] = Point2f(keypoints_I_i1.at<double>(0, matches(1,i)),keypoints_I_i1.at<double>(1, matches(1,i)));
		
		temp_points1Mat.at<double>(0,i) = keypoints_I_i0.at<double>(0, matches(0,i)); // y-coordinate in image 
		temp_points1Mat.at<double>(1,i) = keypoints_I_i0.at<double>(1, matches(0,i)); // x-coordinate in image
		temp_points2Mat.at<double>(0,i) = keypoints_I_i1.at<double>(0, matches(1,i)); // y-coordinate in image
		temp_points2Mat.at<double>(1,i) = keypoints_I_i1.at<double>(1, matches(1,i)); // x-coordinate in image

		
		double y = keypoints_I_i1.at<double>(0, matches(1,i));
		double x = keypoints_I_i1.at<double>(1, matches(1,i));
		double y2 = keypoints_I_i0.at<double>(0, matches(0,i));
		double x2 = keypoints_I_i0.at<double>(1, matches(0,i));
		line(I_i1,Point(x,y),Point(x2,y2),Scalar(0,255,0),3);
		circle (I_i1, Point(x,y), 5,  Scalar(0,0,255), 2,8,0);
		circle (I_i0, Point(x2,y2), 5, Scalar(0,0,255), 2,8,0);	
		*/
		
		/* This worked with old SIFT:MATCH!! 
		points1[i] = Point2f(keypoints_I_i0.at<double>(0, matches(1,i)),keypoints_I_i0.at<double>(1, matches(1,i)));
		points2[i] = Point2f(keypoints_I_i1.at<double>(0, matches(0,i)),keypoints_I_i1.at<double>(1, matches(0,i)));
		
		temp_points1Mat.at<double>(0,i) = keypoints_I_i0.at<double>(0, matches(0,i)); // y-coordinate in image 
		temp_points1Mat.at<double>(1,i) = keypoints_I_i0.at<double>(1, matches(0,i)); // x-coordinate in image
		temp_points2Mat.at<double>(0,i) = keypoints_I_i1.at<double>(0, matches(1,i)); // y-coordinate in image
		temp_points2Mat.at<double>(1,i) = keypoints_I_i1.at<double>(1, matches(1,i)); // x-coordinate in image

		
		double y = keypoints_I_i1.at<double>(0, matches(0,i));
		double x = keypoints_I_i1.at<double>(1, matches(0,i));
		double y2 = keypoints_I_i0.at<double>(0, matches(1,i));
		double x2 = keypoints_I_i0.at<double>(1, matches(1,i));
		line(I_i1,Point(x,y),Point(x2,y2),Scalar(0,255,0),3);
		circle (I_i1, Point(x,y), 5,  Scalar(0,0,255), 2,8,0);
		//imshow("Draw circle I_i1", I_i1);
		//waitKey(0);
		circle (I_i0, Point(x2,y2), 5, Scalar(0,0,255), 2,8,0);	
		//imshow("Draw circle I_i0", I_i0);
		//waitKey(0);
		*/
		
		points1[i] = Point2f(keypoints_I_i0.at<double>(0, matches.at<double>(0,i)),keypoints_I_i0.at<double>(1, matches.at<double>(0,i)));
		points2[i] = Point2f(keypoints_I_i1.at<double>(0, matches.at<double>(1,i)),keypoints_I_i1.at<double>(1, matches.at<double>(1,i)));
		
		temp_points1Mat.at<double>(0,i) = keypoints_I_i0.at<double>(0, matches.at<double>(0,i)); // y-coordinate in image 
		temp_points1Mat.at<double>(1,i) = keypoints_I_i0.at<double>(1, matches.at<double>(0,i)); // x-coordinate in image
		temp_points2Mat.at<double>(0,i) = keypoints_I_i1.at<double>(0, matches.at<double>(1,i)); // y-coordinate in image
		temp_points2Mat.at<double>(1,i) = keypoints_I_i1.at<double>(1, matches.at<double>(1,i)); // x-coordinate in image

		/*
		double y = keypoints_I_i1.at<double>(0, matches.at<double>(1,i));
		double x = keypoints_I_i1.at<double>(1, matches.at<double>(1,i));
		double y2 = keypoints_I_i0.at<double>(0, matches.at<double>(0,i));
		double x2 = keypoints_I_i0.at<double>(1, matches.at<double>(0,i));
		line(I_i1,Point(x,y),Point(x2,y2),Scalar(0,255,0),3);
		circle (I_i1, Point(x,y), 5,  Scalar(0,0,255), 2,8,0);
		//imshow("Draw circle I_i1", I_i1);
		//waitKey(0);
		circle (I_i0, Point(x2,y2), 5, Scalar(0,0,255), 2,8,0);	
		//imshow("Draw circle I_i0", I_i0);
		//waitKey(0);
		*/
		
	}
	//imshow("Match",I_i1);
	//waitKey(0);
	
	
	// Find fudamental matrix 
	vector<uchar> pArray(N);
	Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.90, 5000, pArray); // 3 can be changed to 1
	
	int N_inlier = countNonZero(pArray);
	//cout << "N_inlier = " << N_inlier << endl;
	Mat points1Mat = Mat::zeros(2, N_inlier, CV_64FC1);
	Mat points2Mat = Mat::zeros(2, N_inlier, CV_64FC1);
	temp_index = 0;
	for (int i = 0; i < N; i++) {
		if ((double) pArray[i] == 1) {
			points1Mat.at<double>(0,temp_index) = temp_points1Mat.at<double>(0,i); // y-coordinate in image
			points1Mat.at<double>(1,temp_index) = temp_points1Mat.at<double>(1,i); // x-coordinate in image
			points2Mat.at<double>(0,temp_index) = temp_points2Mat.at<double>(0,i); // y-coordinate in image
			points2Mat.at<double>(1,temp_index) = temp_points2Mat.at<double>(1,i); // x-coordinate in image
			
			/*
			circle (a, Point(points1Mat.at<double>(1,temp_index),points1Mat.at<double>(0,temp_index)), 5, Scalar(0,0,255), 2,8,0);
			imshow("a", a);
			waitKey(0);
			circle (b, Point(points2Mat.at<double>(1,temp_index),points2Mat.at<double>(0,temp_index)), 5,  Scalar(0,0,255), 2,8,0);
			imshow("b", b);
			waitKey(0);
			*/
	
			
			temp_index++;
			
		}
	}
	 
	//cout << "Number of reliably matched keypoints (using RANSAC) in initializaiton = " << N_inlier << endl;
	Si_1.k = N_inlier;
	
	// Update of reliably matched keypoints
	Si_1.Pi = points2Mat;
		
	// Estimate Essential Matrix
	Mat essential_matrix = estimateEssentialMatrix(fundamental_matrix, K);	
	
	// Find the rotation and translation assuming the first frame is taken with the drone on the ground 
	Mat transformation_matrix = findRotationAndTranslation(essential_matrix, K, points1Mat, points2Mat);
	for (int i = 0; i < transformation_matrix.rows; i++) {
		for (int j = 0; j < transformation_matrix.cols; j++) {
			cout << transformation_matrix.at<double>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	
	// Update State with regards to 3D (triangulated points)
	// Triangulate initial point cloud
	Mat M1 = K * (Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
	Mat M2 = K * transformation_matrix;
	Mat landmarks = linearTriangulation(points1Mat, points2Mat, M1, M2);
	
	//cout << "Print landmarks " << endl;
	//cout << landmarks << endl;
	//cout << "Number of landmarks = " << Si_1.Xi.cols << endl;
	Mat temp;
	vconcat(landmarks.row(0), landmarks.row(1), temp);
	vconcat(temp, landmarks.row(2), Si_1.Xi);
	
	/*
	cout << "Print of 3D landmarks " << endl;
	cout << Si_1.Xi << endl;
	cout << "Number of landmarks = " << Si_1.Xi.cols << endl
	*/
	
	cout << "Initializaiton" << endl;
	cout << "Number of keypoints = " << Si_1.Pi.cols << endl;
	cout << "Number of landmarks = " << Si_1.Xi.cols << endl;
	
	
	high_resolution_clock::time_point t12 = high_resolution_clock::now();
	duration<double> time_span5 = duration_cast<duration<double>>(t12-t11);
	cout << "rest of initialization matches took = " << time_span5.count() << " seconds" << endl;
	
	// return state of drone as well as transformation_matrix;
	return make_tuple(Si_1, transformation_matrix);
}

// ####################### VO Continuous Operation #######################
// Process Frame for continuous VO operation 
/* Arguments: 
 * Current Image I1
 * Previous Image Ii-1 (I_{i-1})
 * Previous State Si_1, which is a struct
 */
tuple<state, Mat> processFrame(Mat Ii, Mat Ii_1, state Si_1, Mat K) {

	// Turn the images into grayscale 
	Mat Ii_gray, Ii_1_gray;
	cvtColor(Ii, Ii_gray, COLOR_BGR2GRAY );
	cvtColor(Ii_1, Ii_1_gray, COLOR_BGR2GRAY );	
	
	
	Mat descriptors_Ii_1 = SIFT::FindDescriptors(Ii_1_gray, Si_1.Pi);
	
	int dim1, dim2;
	dim1 = Ii_gray.rows;
	dim2 = Ii_gray.cols;
	Mat Ii_resized = Ii_gray.colRange(10,dim2-10).rowRange(10,dim1-10);
	Mat temp1;
	goodFeaturesToTrack(Ii_resized, temp1, 300, 0.01, 10, noArray(), 3, true, 0.04);
	Mat keypoints_Ii = Mat::zeros(2, temp1.rows, CV_64FC1);
	for (int i = 0; i < keypoints_Ii.cols; i++) {
		keypoints_Ii.at<double>(0,i) = temp1.at<float>(i,1) + 10;
		keypoints_Ii.at<double>(1,i) = temp1.at<float>(i,0) + 10;
	}
	
	Mat descriptors_Ii = SIFT::FindDescriptors(Ii_gray, keypoints_Ii);
	
	Mat matches = SIFT::matchDescriptors(descriptors_Ii_1, descriptors_Ii);
	Mat matches2 = SIFT::matchDescriptors(descriptors_Ii, descriptors_Ii_1);
	
	Mat valid_matches = Mat::zeros(2, matches.cols, CV_64FC1);
	int temp_index = 0;
	for (int i = 0; i < matches.cols; i++) {
		int index_frame0 = matches.at<double>(0,i);
		int index_frame1 = matches.at<double>(1,i);
		
		for (int q = 0; q < matches2.cols; q++) {
			if (matches2.at<double>(1,q) == index_frame0) {
				if (matches2.at<double>(0,q) == index_frame1) {
					// Mutual match
					valid_matches.at<double>(0,temp_index) = index_frame0;
					valid_matches.at<double>(1,temp_index) = index_frame1;
					temp_index++;
				}
			}
		}
	}
	matches = valid_matches.colRange(0,temp_index);
	cout << "Number of mutual valid mathces = " << matches.cols << endl;
	waitKey(0);
	
	int N = matches.cols;
	/*
	Mat emptyMatrix;
	
	// Not necessary since you already have the keypoints
	//Mat keypoints_Ii_1 = Harris::corner(Ii_1, Ii_1_gray, 210, emptyMatrix);
	Mat descriptors_Ii_1 = SIFT::FindDescriptors(Ii_1_gray, Si_1.Pi);
	
	cout << "dimensions of descriptors_Ii_1 = (" << descriptors_Ii_1.rows << "," << descriptors_Ii_1.cols << ")" << endl;
	
	Mat keypoints_Ii = Harris::corner(Ii, Ii_gray, 210, emptyMatrix);
	
	
	Mat descriptors_Ii = SIFT::FindDescriptors(Ii_gray, keypoints_Ii);
	
	const char* text1 = "ProcessFrame Ii_1";
	drawCorners(Ii_1, Si_1.Pi, text1);
	waitKey(0);
	
	const char* text2 = "ProcessFrame Ii";
	drawCorners(Ii, keypoints_Ii, text2);
	waitKey(0);
	
	// Match descriptors - Should be optimized
	Mat matches = SIFT::matchDescriptors(descriptors_Ii_1, descriptors_Ii);
	
	int N = matches.cols;
	
	cout << "N is = " << N << endl;
	
	Mat keypoints_i = Mat::zeros(2, N, CV_64FC1);
	Mat corresponding_landmarks = Mat::zeros(3, N, CV_64FC1); 
	cout << "After update" << endl;
	*/
	Mat keypoints_i = Mat::zeros(2, N, CV_64FC1);
	Mat corresponding_landmarks = Mat::zeros(3, N, CV_64FC1);
	
	for (int i = 0; i < N; i++) {
		// Config1
		//  #### SIFT #### 
	
		// Turn the kyepoints so it becomes (u,v)
		keypoints_i.at<double>(0,i) = keypoints_Ii.at<double>(1, matches.at<double>(1,i)); // x-coordinate in image
		keypoints_i.at<double>(1,i) = keypoints_Ii.at<double>(0, matches.at<double>(1,i)); // y-coordinate in image
		
		//Si_1.Xi.col(matches(i,0)).copyTo(corresponding_landmarks.col(i));
		corresponding_landmarks.at<double>(0,i) = Si_1.Xi.at<double>(0, matches.at<double>(0,i));
		corresponding_landmarks.at<double>(1,i) = Si_1.Xi.at<double>(1, matches.at<double>(0,i));
		corresponding_landmarks.at<double>(2,i) = Si_1.Xi.at<double>(2, matches.at<double>(0,i));
		
		/*
		double y = keypoints_Ii.at<double>(0, matches.at<double>(1,i));
		double x = keypoints_Ii.at<double>(1, matches.at<double>(1,i));
		double y2 = Si_1.Pi.at<double>(0, matches.at<double>(0,i));
		double x2 = Si_1.Pi.at<double>(1, matches.at<double>(0,i));
		line(Ii,Point(x2,y2),Point(x,y),Scalar(0,255,0),3);
		circle (Ii, Point(x,y), 5,  Scalar(0,0,255), 2,8,0);
		circle (Ii_1, Point(x2,y2), 5, Scalar(0,0,255), 2,8,0);
		*/
	}
	/*
	imshow("matches in processFrame Ii_1", Ii_1);
	waitKey(0);
	imshow("matches in processFrame Ii", Ii);
	waitKey(0);
	*/
	
	/*
	int nr_keep = 0;
	//Mat kpold = Mat::zeros(3, Si_1.k, CV_64FC1);
	//Mat delta_keypoint;
	
	//high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	
	int NUM_THREADS = Si_1.k;
	
	int r_T = KLT_r_T;
	int n = 2*r_T + 1;
	Mat xy1 = Mat::zeros(n * n, 3, CV_64FC1);
	int temp_index = 0; 
	for (int i = -r_T; i <= r_T; i++) {
		for (int j = -r_T; j <= r_T; j++) {
			xy1.at<double>(temp_index,0) = i;
			xy1.at<double>(temp_index,1) = j;
			xy1.at<double>(temp_index,2) = 1;
			temp_index++;
		} 
	}
	
	// Find the Kroeneckerproduct 
	Mat dwdx = Kroneckerproduct(xy1, Mat::eye(2, 2, CV_64FC1));
	
	pthread_t threads[NUM_THREADS];
	struct thread_data td[NUM_THREADS];
	int i, rc;
	for (i = 0; i < NUM_THREADS; i++) {
		td[i].Ii_1_gray = Ii_1_gray;
		td[i].Ii_gray = Ii_gray;
		td[i].dwdx = dwdx;
		td[i].thread_mat = Si_1.Pi.col(i);
		rc = pthread_create(&threads[i], NULL, functionKLT, (void *)&td[i]);
	}
	void* ret = NULL;
	vector<Mat> keypoint_container;
	vector<Mat> landmark_container;
	Mat keypoints_i;
	Mat corresponding_landmarks;
	for (int k = 0; k < NUM_THREADS; k++) {
		pthread_join(threads[k], &ret);
		if (td[k].keep_point == 1) {
			keypoint_container.push_back(td[k].thread_mat);
			landmark_container.push_back(Si_1.Xi.col(k));
			//cout << "Mistake 1" << endl;
		}
		//Matcontainer.push_back(td[k].thread_mat);
		//cout << "Mistake here = " << k << endl;
	}
	//cout << "Mistake hej" << endl;
	hconcat(keypoint_container, keypoints_i);
	hconcat(landmark_container, corresponding_landmarks);
	*/
	

	
	/*
	cout << "Print of keypoints_i" << endl;
	cout << keypoints_i << endl;
	cout << "Print of corresponding_landmarks" << endl;
	cout << corresponding_landmarks << endl;	
	*/	
	cout << "Process Frame" << endl;
	cout << "Number of keypoints = " << keypoints_i.cols << endl;
	cout << "Number of landmarks = " << corresponding_landmarks.cols << endl;
	
	
	// Estimate the new pose using RANSAC and P3P algorithm 
	Mat transformation_matrix, best_inlier_mask;
	tie(transformation_matrix, best_inlier_mask) = Localize::ransacLocalization(keypoints_i, corresponding_landmarks, K);
	
	cout << "best_inlier_mask " << endl;
	cout << best_inlier_mask << endl;
	
	// Remove points that are determined as outliers from best_inlier_mask by using best_inlier_mask
	//cout << "best_inlier_mask" << endl;
	//cout << best_inlier_mask << endl;
	vector<Mat> keypoint_inlier;
	vector<Mat> landmark_inlier;
	for (int i = 0; i < best_inlier_mask.cols; i++) {
		if ((double) best_inlier_mask.at<uchar>(0,i) > 0) {
			keypoint_inlier.push_back(keypoints_i.col(i));
			landmark_inlier.push_back(corresponding_landmarks.col(i));
		}
	}
	hconcat(keypoint_inlier, keypoints_i);
	hconcat(landmark_inlier, corresponding_landmarks);
	
	
	// Update keypoints in state 
	Si_1.k = keypoints_i.cols;
	vconcat(keypoints_i.row(1), keypoints_i.row(0), Si_1.Pi); // Apparently you have to switch rows
	corresponding_landmarks.copyTo(Si_1.Xi);
	
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//duration<double> time_span = duration_cast<duration<double>>(t2-t1);
	//cout << "This process took = " << time_span.count() << " seconds.";
	
	
	//return make_tuple(Si, transformation_matrix); 
	return make_tuple(Si_1, transformation_matrix); 
}


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
	
	// Calibrate camera to get intrinsic parameters K 
	Mat K = (Mat_<double>(3,3) << 769.893, 0, 2.5, 0,1613.3, 4, 0, 0, 1);
	
	cv::Mat I_i0, I_i1, image;
	
	int nCount=getParamVal ( "-nframes",argc,argv, 100 );
	cout<<"Capturing"<<endl;
	
	// Initialization
	Camera.grab(); // You need to take an initial image in order to make the camera work
	Camera.retrieve( image ); 
	//cout << "Image captured" <<endl;
	waitKey(1000);
	
	/*
	// Initial frame 0 
	Camera.grab();
	Camera.retrieve( I_i0 ); 
	cout << "Frame I_i0 captured" <<endl;
	//I_i0.convertTo(I_i0, CV_64FC1);
	imshow("Frame I_i0 displayed", I_i0);
	waitKey(0);
	
	cout << "Size of image = " << I_i0.rows << "," << I_i0.cols << ")" << endl;
	
	//imwrite("cam0.png", I_i0);
	//waitKey(0);	// Ensures it is sufficiently far away from initial frame
	//waitKey(5000);
	
	
	// First frame 1
	cout << "Prepare to take image 1" << endl; 
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
	cout << "Frame I_i1 captured" <<endl;
	imshow("Frame I_i1 displayed", I_i1);
	waitKey(0);
	*/
	
	
	 
	
	// Test billeder
	I_i0 = imread("cam0.png", IMREAD_UNCHANGED);
	//I_i0.convertTo(I_i0, CV_64FC1);
	imshow("Frame I_i0 displayed", I_i0);
	waitKey(0);
	
	I_i1 = imread("cam1.png", IMREAD_UNCHANGED);
	//I_i1.convertTo(I_i1, CV_64FC1);
	imshow("Frame I_i1 displayed", I_i1);
	waitKey(0);
	
	
	/*
	int y1 = 3;
	int x1 = 1;
	int y2 = 5;
	int x2 = 5;
	
	cout << "Entire image" << endl;
	for (int r = 0; r < 10; r++) {
		for (int c = 0; c < 10; c++) {
			cout << (double) I_i0_gray.at<uchar>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	cout << "Subpart of I_i0" << endl;
	MatType(I_i0_gray);
	for (int r = y1; r < y2; r++) {
		for (int c = x1; c < x2; c++) {
			cout << (double) I_i0_gray.at<uchar>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	 Make a region of the proper size. 
	 
	  Start point in point (x1,y1) width = y2-y1 and height x2-x1. 
	 
	// Rect region(x1,y1,x2-x1,y2-y1);
	
	//Rect region(x1,y1,x2-x1,y2-y1);
	//Rect region(x1,y1,y2-y1,x2-x1);
	//Rect region(y1,x1,x2-x1,y2-y1);
	//Rect region(y1,x1,y2-y1,x2-x1);
	
	// NB: THIS ONLY WORKS FOR GRAY-SCALE IMAGES, WHERE THERE IS ONLY ONE CHANNEL
	//Mat subImg(I_i0, region);
	//Mat subImg = I_i0(Range(y1,y2),Range(x1,x2));
	Mat subImg = I_i0_gray.colRange(x1,x2).rowRange(y1,y2);
	cout << "subImg" << endl;
	//cout << subImg << endl;	
	MatType(subImg);
	cout << "Dimensions of subImg = (" << subImg.rows << "," << subImg.cols << ")" << endl;	
	for (int i = 0; i < subImg.rows; i++) {
		for (int j = 0; j < subImg.cols; j++) {
			cout << (double) subImg.at<uchar>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	subImg.at<uchar>(1,2) = 0;
	cout << "subImg" << endl;
	for (int i = 0; i < subImg.rows; i++) {
		for (int j = 0; j < subImg.cols; j++) {
			cout << (double) subImg.at<uchar>(i,j) << ", ";
		}
		cout << "" << endl;
	}
	cout << "I_i0" << endl;
	for (int r = y1; r < y2; r++) {
		for (int c = x1; c < x2; c++) {
			cout << (double) I_i0_gray.at<uchar>(r,c) << ", ";
		}
		cout << "" << endl;
	}
	*/
	
	/*
	// Test af Harris Corners
	Mat I_i0_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	Mat emptyMatrix;
	Mat keypoints_I_i0 = Harris::corner(I_i0, I_i0_gray, 180, emptyMatrix);
	const char* text0 = "Detected corners in frame I_i0";
	drawCorners(I_i0, keypoints_I_i0, text0);
	waitKey(0);
	*/
	/*
	Mat I_i0_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	
	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("HARRIS");
	
	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create("SIFT");
	
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	detector->detect(I_i0_gray, keypoints1);
	
	cv::Mat descriptor1;
	descriptor->compute(I_i0_gray,keypoints1, descriptor1);
	*/
	
	
	/*
	Mat corners;
	Mat I_i0_gray;
	cvtColor(I_i0, I_i0_gray, COLOR_BGR2GRAY );
	vector<uchar> pArray;
	Mat hej;
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	
	vector<KeyPoint> key_points;
	goodFeaturesToTrack(I_i0_gray, hej, 200, 0.01, 10, noArray(), 3, true, 0.08);
	key_points.push_back(hej.row(0));
	
	Mat descriptors;
	SIFT::operator()(I_i0, noArray(), key_points, descriptors, true);
	
	//Mat descriptors;
	//SIFT::operator(I_i0, noArray(), descriptors, true);
	*/
	
	/*
	high_resolution_clock::time_point t6 = high_resolution_clock::now();
	duration<double> time_span2 = duration_cast<duration<double>>(t6-t3);
	cout << "This Harris took: = " << time_span2.count() << " seconds" << endl;
	
	cout << "dimensions of hej = (" << hej.rows << "," << hej.cols << ")" << endl;
	cout << "hej = " << hej << endl;
	cout << "Point = (" << hej.row(0) << ")" << endl;
	
	Mat a;
	vector<Mat> keypoint_container;
	//cout << "a = " << a[0][0]
	keypoint_container.push_back(hej.row(0)); 
	keypoint_container.push_back(hej.row(1)); 
	hconcat(keypoint_container, a);
	cout << "a = " << a.at<float>(0,0) << endl;
	cout << "hej = " << hej.at<float>(0,0) << endl;
	MatType(a);
	Mat keypoints = Mat::zeros(2,hej.rows, CV_64FC1);
	for (int i = 0; i < keypoints.cols; i++) {
		keypoints.at<double>(0,i) = hej.at<float>(i,1);
		keypoints.at<double>(1,i) = hej.at<float>(i,0);
	}
	const char* text0 = "Test corners in frame I_i0";
	drawCorners(I_i0, keypoints, text0);
	waitKey(0);
	*/
	

	// ############### VO initializaiton ###############
	// VO-pipeline: Initialization. Bootstraps the initial position.	
	state Si_1;
	Mat transformation_matrix;
	Mat Ii_1;
	I_i1.copyTo(Ii_1);
	tie(Si_1, transformation_matrix) = initializaiton(I_i0, I_i1, K, Si_1);
	cout << "Transformation matrix " << endl;
	cout << transformation_matrix << endl;
	
	
	// ############### VO Continuous ###############
	bool continueVOoperation = true;
	bool pipelineBroke = false;
	bool output_T_ready = true;
	
	// Needed variables
	state Si;
	//Mat Ii_1 = imread("cam1.png", IMREAD_UNCHANGED);
	Mat Ii;
	double threshold_angle = new_landmarks_threshold_angle; // In degrees
	Mat extracted_keypoints = Mat::zeros(1, num_candidate_keypoints, CV_64FC1); // Remeber that 100 should be replaced by Si.num_candidates in a smart way
	
	// Mat Ii_1 = I_i1;
	// For test 
	
	
	// Debug variable
	int stop = 0;
	int iter = 0;
	
	while (continueVOoperation == true && pipelineBroke == false && stop < 1) {
		cout << "Begin Continuous VO operation " << endl;
		
		/*
		// Take new image 
		Camera.grab();
		Camera.retrieve( Ii );
		cout << "New image taken" << endl;
		*/
		
		
		/*
		imshow("New Frame", Ii);
		cout << "New image aquired" << endl;
		waitKey(0);
		*/
		
		cout << "Continuous Operation" << endl;
		cout << "Number of keypoints = " << Si_1.Pi.cols << endl;
		cout << "Number of landmarks = " << Si_1.Xi.cols << endl;
	
		
		Ii = imread("cam1.png", IMREAD_UNCHANGED);
		imshow("Continous operation frame", Ii);
		waitKey(0);
		
		
		
		// Estimate pose 
		tie(Si, transformation_matrix) = processFrame(Ii, Ii_1, Si_1, K);
		cout << "processFrame done" << endl;
		cout << "Print of Transformation Matrix" << endl;
		cout << transformation_matrix << endl;
		
		output_T_ready = true;
		
		// Submit the transformation matrix. Then set the flag low. 
		output_T_ready = false;
		
		
		// Find new candidate keypoints 
		if (iter == 0) {
			Si = newCandidateKeypoints(Ii, Si, transformation_matrix);
			iter++;
		}
		else {
			//Ii = imread("cam2.png", IMREAD_UNCHANGED);
			
			int r_T = KLT_r_T;
			int n = 2*r_T + 1;
			Mat xy1 = Mat::zeros(n * n, 3, CV_64FC1);
			int temp_index = 0; 
			for (int i = -r_T; i <= r_T; i++) {
				for (int j = -r_T; j <= r_T; j++) {
					xy1.at<double>(temp_index,0) = i;
					xy1.at<double>(temp_index,1) = j;
					xy1.at<double>(temp_index,2) = 1;
					temp_index++;
				} 
			}
			// Find the Kroeneckerproduct 
			Mat dwdx = Kroneckerproduct(xy1, Mat::eye(2, 2, CV_64FC1));
			
			Si = continuousCandidateKeypoints(Ii_1, Ii, Si, transformation_matrix, extracted_keypoints, dwdx);
			cout << "ContinuousCandidateKeypoints" << endl;
			cout << "Number of keypoints = " << Si.Ci.cols << endl;
			
			tie(Si, extracted_keypoints) = triangulateNewLandmarks( Si, K, transformation_matrix, threshold_angle);
		}
		
		
		/*
		// Test of function newCandidateKeypoints 
		cout << "Test of function newCandidateKeypoints" << endl;
		for (int k = 0; k < Si.Pi.cols; k++) {
			double x = Si.Pi.at<double>(0, k);
			double y = Si.Pi.at<double>(1, k);
			circle (Ii, Point(y,x), 5, Scalar(0,0,255), 2,8,0);
		}
		imshow("Corners from Si.Pi", Ii);
		waitKey(0);
		for (int k = 0; k < Si.num_candidates; k++) {
			double x = Si.Ci.at<double>(0, k);
			double y = Si.Ci.at<double>(1, k);
			circle (Ii, Point(y,x), 5, Scalar(255,0,0), 2,8,0);
		}
		imshow("Corners from Si.Ci", Ii);
		waitKey(0);
				
		// Draw Corners from Si.Ci 
		cout << "Si.Ci" << endl;
		cout << Si.Ci << endl;
		
		cout << "Si.Fi" << endl;
		cout << Si.Fi << endl; 
		
		cout << "Si.Ti" << endl;
		//cout << Si.Ti << endl;
		*/
		
		// Resert old values 
		//Ii_1.copyTo(Ii);
		Ii.copyTo(Ii_1);
		Si_1 = Si;
		cout << "Update of Si.num_candidates = " << Si.num_candidates << endl;
		cout << "End of ContinVO. Numb. keypoints = " << Si_1.Pi.cols << endl;
		cout << "End of ContinVO. Numb. landmarks = " << Si_1.Xi.cols << endl;
		
		// Debug variable
		stop++;
		if (stop > 10) {
			break;
		}
		
	}
	

	
	
	cout << "VO-pipeline terminated" << endl;


	double time_=cv::getTickCount();
	
	Camera.release();

	
	
	return 0;
}










