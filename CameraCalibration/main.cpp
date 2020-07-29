#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/affine.hpp>

#include <opencv2/opencv.hpp>

#include <raspicam/raspicam_cv.h>

#include <ctime>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdint.h>

using namespace std; 
using namespace cv;


/* This file is written by Thor Vestergaard Christiansen (s173949)
 * by inspiration from George Lecakes - https://www.youtube.com/watch?v=HNfPbw-1e_w
 * In order to calibrate the camera, simply just run this file. 
 * When you can see that the corners are detected properly, hit the space button on
 * your computer. This saves an image and tells you how many images are saved.
 * When you have accuqired around ~ 56 images, you should hit the enter key
 * This will give you a matrix and save it in the file Ca.librationMatrixFile
 * Remember to move the picture of the corners around in the image and rotate it. 
 * 
 * 
 * 
 */


//const float calibrationSquareDimension = 0.02402f; // meters 
const float calibrationSquareDimension = 0.0221f; // meters 
const float arucoSquareDimension = 0.1016f; // meters
const Size chessboardDimensions = Size(6, 9);


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

void createArucoMarkers() {
	
	Mat outputMarker;
	
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
	
	
	for (int i = 0; i < 50; i++) {
		aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);
		ostringstream convert;
		string imageName = "4x4Marker_";
		convert << imageName << i << ".jpg";
		imwrite(convert.str(), outputMarker);
		
	}
	
}


void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners) {
	
	for (int i = 0; i < boardSize.height; i++) {
		for (int j = 0; j < boardSize.width; j++) {
			
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}	
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false) {
	
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++) {
		
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9,6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		
		if (found) {
			
			allFoundCorners.push_back(pointBuf);
			
			if (showResults) {
				drawChessboardCorners(*iter, Size(9,6), pointBuf, found);
				imshow("Looking for Corners", *iter);
				waitKey(0);
				
			}
		}
	}
	
}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients) {
	
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);
	
	vector<vector<Point3f>> worldSpaceCornerPoints(1);
	
	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	
	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);
	
	
	vector<Mat> rVectors, tVectors;
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);
	
	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
	
}


bool saveCameraCalibraiton(string name, Mat cameraMatrix, Mat distanceCoefficients) {
	
	double value;
	ofstream outStream(name);
	if (outStream) {
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;
		
		outStream << "Calibration Matrix K (Intrinsic Parameters)" << endl;
		outStream << "Rows: " << rows << endl;
		outStream << "Columns: " << columns << endl;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				
				double value = cameraMatrix.at<double>(r, c);
				
				//outStream << value << endl;
				outStream << value << ", ";
				
			}
			outStream << "" << endl;
		}
		
		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;
		
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				
				double value = distanceCoefficients.at<double>(r, c);
				
				outStream << value << endl;
				
			}
		}
		
		outStream.close();
		return true;
		
	}
	
	return false;
}



int main( int argc,char **argv ) {
	
	
	raspicam::RaspiCam_Cv Camera;
	processCommandLine ( argc,argv,Camera );
	cout<<"Connecting to camera"<<endl;

	
	//createArucoMarkers();
	Mat frame; 
	Mat drawToFrame; 
	
	Mat cameraMatrix = Mat::eye(3,3, CV_64F);
	
	Mat distanceCoefficients; 
	
	vector<Mat> savedImages; 
	
	vector<vector<Point2f>> markerCorners, rejectedCandidates; 
	
	// This should maybe be commented out
	//VideoCapture vid(0);
	
	if ( !Camera.open() ) {
		cerr<<"Error opening camera"<<endl;
		return -1;
	}
	cout<<"Connected to camera ="<<Camera.getId() << endl;
	
	int nCount=getParamVal ( "-nframes",argc,argv, 100 );
	cout<<"Capturing"<<endl;
	
	/*
	if (!vid.isOpened()) {
		return 0;
	}
	*/
	
	int framesPerSecond = 20;
	
	namedWindow("Webcam", WINDOW_AUTOSIZE);
	
	while (true) {
		
		/*
		if (!vid.read()) {
			break;
		}
		*/
		
		Camera.grab();
		Camera.retrieve( frame ); 
		//cout << "Image: frame captured" <<endl;
		
		vector<Vec2f> foundPoints;
		bool found = false;
		
		found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | 
		
		CALIB_CB_NORMALIZE_IMAGE);
		
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);
		
		if (found) {
			imshow("Webcam", drawToFrame);
		}
		else {
			imshow("Webcam", frame);
		}
		char character = waitKey(1000 / framesPerSecond);
		
		switch(character) {
			case ' ':
				// saving image
				if (found) {
					Mat temp;
					frame.copyTo(temp);
					savedImages.push_back(temp);
					cout << "Image saved and image total: " << savedImages.size()  << endl;
				}
				break;
			case 13: 
				// start callibration 
				if (savedImages.size() > 15) {
					cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
					saveCameraCalibraiton("CalibrationMatrixFile", cameraMatrix, distanceCoefficients);
				
				}
				
				
				break;
			case 27: 
				// exit
				return 0;
				break;
			
		}
		
	}
	
	
	double time_=cv::getTickCount();
	
	Camera.release();
	
	return 0;
	
	
	
	
	
	
	
	
	
	
}









