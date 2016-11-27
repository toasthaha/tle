#ifndef __TLE_HPP__
#define __TLE_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <gflags/gflags.h>

//target tracker
#include "kcftracker.hpp"

#define DETECT_TIME_PENALTY 8
static const std::string Version = "0.4";

enum Action {
	DETECT = 0,
	TRACK  = 1
};

typedef double reward_t;
typedef std::vector<Action> ActionVect;
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
	std::vector<bool> groundtruthValid;
	std::vector<cv::Rect> det;
}frameData;

typedef struct DASHCAM{
	std::vector<frameData> frames;
}dashcam;


class TLEInterface
{
protected:
    int maxNumFrames;  		   	// Maximum number of frames for each episode
	int maxNumTrackers;			// Maximum number of trackers
	int totalFrame;				// total Frames for current input 
	int currentFrameId;			// Current Fream Id
	cv::Mat returnFrame;		// Frame returned
	cv::Scalar plotColor;		// Bounding Box Color
	dashcam input;				// input data struct
	std::vector<bool>trackerOn;	// Showing each tracker is working or not 

	//===================
	// Tracker setting 
	//===================
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = false;
	std::vector<KCFTracker> tracker;
	//===================

public:
	// Constructor
	TLEInterface(int trackerNum,bool gui){
		maxNumTrackers = trackerNum;
		tracker.resize(maxNumTrackers);
		trackerOn.resize(maxNumTrackers,false);
	};
    
	//Destructor
	~TLEInterface(){};

	//load
	bool load(std::string videoName,std::string labelName,std::string detName);

	// Resets the track
    void reset();

	// Release input
	void releaseInput();
	
    // Indicates if the tracking has ended
    bool isEnded();

    // Applies an action and returns the reward. 
    reward_t act(Action action,bool skiped=true);

    // Returns the vector of legal actions. 
    ActionVect getLegalActionSet();

	// Returns tracker count
	int getTrackerCount();

    // Returns the current game screen
    cv::Mat getScreen();

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

std::string action_to_string(int a);

#endif
