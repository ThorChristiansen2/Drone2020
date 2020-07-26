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
	imshow("Frame I_i0 displayed", I_i0);
	waitKey(0);


	// First frame 1
	cout << "Prepare to take image 1" << endl; 
	Camera.grab();
	Camera.retrieve ( I_i1 ); // Frame 1 
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
	cout << "Test af ransacLocalizaiton" << endl;
	Mat corresponding_landmarks = Mat::zeros(3, 271, CV_64FC1);
	ifstream MyRead2File("CorrespondingLandmarks.txt");	
	// Fejl i hvordan det loades ind 
	if (MyRead2File.is_open()) {
		for (int i = 0; i < 271; i++) {
			MyRead2File >> corresponding_landmarks.at<double>(0,i);
			MyRead2File >> corresponding_landmarks.at<double>(1,i);	
			MyRead2File >> corresponding_landmarks.at<double>(2,i);	
		}
	}
	MyRead2File.close();
	
	cout << "corresponding_landmarks = " << corresponding_landmarks << endl;
	waitKey(0);
	
	Mat matched_query_keypoints = Mat::zeros(2, 271, CV_64FC1);
	ifstream MyRead3File("matched_query_keypoints.txt");	
	// Fejl i hvordan det loades ind 
	if (MyRead3File.is_open()) {
		for (int i = 0; i < 271; i++) {
			MyRead2File >> matched_query_keypoints.at<double>(0,i);
			MyRead2File >> matched_query_keypoints.at<double>(1,i);	
		}
	}
	MyRead3File.close();
	
	cout << "matched_query_keypoints = " << matched_query_keypoints << endl;
	waitKey(0);
	
	Mat K2 = (Mat_<double>(3,3) << 718.8560, 0, 607.1928, 0, 718.8560, 185.2157, 0, 0, 1);
	
	Mat transformation_matrix1, best_inlier_mask;
	tie(transformation_matrix1, best_inlier_mask) = Localize::ransacLocalization(matched_query_keypoints, corresponding_landmarks, K2);
	cout << "transformation_matrix1 = " << transformation_matrix1 << endl;
	*/

	// ############### VO initializaiton ###############
	// VO-pipeline: Initialization. Bootstraps the initial position.	
	state Si_1;
	Mat transformation_matrix;
	bool initialization_okay;
	Mat Ii_1;
	I_i1.copyTo(Ii_1);
	//tie(Si_1, transformation_matrix, init_okay) = initialization(I_i0, I_i1, K, Si_1); // One variable extra 
	tie(Si_1, transformation_matrix, initialization_okay) = initialization(I_i0, I_i1, K, Si_1); // One variable extra 
	cout << "Transformation matrix " << endl;
	cout << transformation_matrix << endl; (float) transformation_matrix.at<double>(1,3);

	
	// ############### VO Continuous ###############
	bool continueVOoperation = true;
	bool pipelineBroke = false;
	bool output_T_ready = true;
	bool processFrame_okay;
	
	// Needed variables
	state Si;
	Mat Ii;
	double threshold_angle = new_landmarks_threshold_angle; // In degrees
	
	
	// Debug variable
	int stop = 0;
	int iter = 0;
	
	cout << "Begin Continuous VO operation " << endl;
	while (continueVOoperation == true && pipelineBroke == false && stop < 1) {
		cout << "Continuous Operation " << endl;

		cout << "Number of keypoints = " << Si_1.Pi.cols << endl;
		cout << "Number of landmarks = " << Si_1.Xi.cols << endl;
	
		/*
		Camera.grab();
		Camera.retrieve ( Ii );
		imshow("Continuous Operation", Ii);
		waitKey(0);
		*/
	
		
		Ii = imread("cam1.png", IMREAD_UNCHANGED);
		imshow("Continous operation frame", Ii);
		waitKey(0);
		
		
		// Estimate pose 
		tie(Si, transformation_matrix, processFrame_okay) = processFrame(Ii, Ii_1, Si_1, K);
		cout << "processFrame done" << endl;
		cout << "Print of Transformation Matrix" << endl;
		cout << transformation_matrix << endl;
		
		output_T_ready = true;
		
		// Submit the transformation matrix. Then set the flag low. 
		output_T_ready = false;
		
		
		// Find new candidate keypoints for the first itme 
		if (iter == 0) {
			cout << "Find new candidate keypoints for the first time" << endl;
			
			Si = newCandidateKeypoints(Ii, Si, transformation_matrix);
			iter++;
		}
		// Keep finding new candidate keypoints and see if candidate keypoints can become real keypoints
		else {
			cout << "Inside other continuous operation " << endl;
			
			Si = continuousCandidateKeypoints(Ii_1, Ii, Si, transformation_matrix);
			cout << "ContinuousCandidateKeypoints" << endl;
			cout << "Number of keypoints = " << Si.Ci.cols << endl;
			
			Si = triangulateNewLandmarks( Si, K, transformation_matrix, threshold_angle);
		}
		
		
		// Resert old values 
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










