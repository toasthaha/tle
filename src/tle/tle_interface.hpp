#ifndef __TLE_HPP__
#define __TLE_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <gflags/gflags.h>

//target tracker
#include "kcftracker.hpp"

#define DETECT_TIME_PENALTY 5
static const std::string Version = "0.1";

enum Action {
	DETECT = 0,
	TRACK	= 1
};


typedef double reward_t;
typedef std::vector<Action> ActionVect;
typedef struct BOX{
	std::string category;
	int frameId,trackId,x1,y1,x2,y2;
	float confident = 1.0;
}box;

class TLEInterface
{
protected:
    reward_t episode_score; 	// Score accumulated throughout the course of an episode
    int maxNumFrames;  		   	// Maximum number of frames for each episode
	int maxNumTrackers;			// Maximum number of trackers
	std::ifstream labelFile;	// label input file
	cv::VideoCapture Cap;		// Read video
	cv::Mat currentFrame;		// Frame readded
	int currentFrameId;			// Current Fream Id
	box nextBox;				// Next frame's 1st track target
	std::vector<bool>trackerOn;	// Showing each tracker is working or not 
	bool videoEnded;			// Video End	

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
	~TLEInterface(){
		if(labelFile.is_open())
			labelFile.close();
		
		// VideoFile will release automatically 
		// by VideoCapture Destructor
		//if(Cap.isOpened())
		//	Cap.release();
	};

	//load
	bool load(std::string videoName,std::string labelName);

	// Resets the track
    void reset();

    // Indicates if the tracking has ended
    bool isEnded();

    // Applies an action and returns the reward. 
    reward_t act(Action action);

    // Returns the vector of legal actions. 
    ActionVect getLegalActionSet();

    // Returns the current game screen
    cv::Mat getScreen();

	// Returns the current frame Id
	int getCurrentFrameId(){
		return currentFrameId;
	}

	// Read input label file
	box readInputLabel();
	
	// Convert box into Rect
	cv::Rect box2Rect(box in);

};

std::string action_to_string(int a);

#endif
