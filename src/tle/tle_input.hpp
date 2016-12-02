#ifndef __TLE_INPUT_HPP__
#define __TLE_INPUT_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>

typedef struct BOX{
	std::string category;
	int frameId,trackId,x1,y1,x2,y2;
	float confident = 1.0;
}box;

typedef struct FRAME{
	int trackerCount=0;
	cv::Mat rawFrame;
	cv::Mat maskFrame;
	cv::Mat groundtruthFrame;
	std::vector<cv::Rect> groundtruth;
	std::vector<cv::Rect> det;
	std::vector<bool> detValid;
}frameData;


class TLEInput
{
public:
	bool valid = false;
	int totalFrame;
	int maxNumTrackers;
	int currentFrameId;
	std::vector<frameData> frames;
	std::string name;
	cv::Mat returnFrame;
	cv::Scalar plotColor;		

public:
	// Constructor
	TLEInput(int TrackerNum){
		maxNumTrackers = TrackerNum;
	};

	~TLEInput(){};

	//load
	bool load(std::string videoName,std::string labelName,std::string detName);

	// Resets the input video
    void reset();
	
	// Release the input video
	void release();
    
	// Indicates if the tracking has ended
    bool isEnded();

	// Returns tracker count
	int getTrackerCount();

	// Returns the current frame Id
	int getCurrentFrameId(){
		return currentFrameId;
	}

	// Read input label file
	box readInputLabel(std::ifstream* labelFile);
	
	// Read input det file
	box readInputDet(std::ifstream* labelFile);

	// Convert box into Rect
	cv::Rect box2Rect(box in);

};

#endif
