#ifndef __TLE_HPP__
#define __TLE_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "tle_input.hpp"

//target tracker
#include "kcftracker.hpp"

#define DETECT_TIME_PENALTY 8
static const std::string Version = "0.5";

enum Action {
	DETECT = 0,
	TRACK  = 1
};
typedef std::vector<Action> ActionVect;
typedef double reward_t;


class TLEInterface
{
protected:
	int maxNumInput;			 // Maximum number of video load at same time
    int maxNumFrames;  		   	 // Maximum number of frames for each episode
	int maxNumTrackers;			 // Maximum number of trackers
	int targetInputId;			 // Target Input 
	int currentFrameId;			 // Current Fream Id
	std::vector<bool> trackerOn; // Showing each tracker is working or not 
	std::vector<TLEInput> input; // input data struct

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
	TLEInterface(int trackerNum,int inputNum){
		maxNumTrackers = trackerNum;
		maxNumInput    = inputNum; 
		tracker.resize(maxNumTrackers);
		trackerOn.resize(maxNumTrackers,false);
		input.resize(maxNumInput,trackerNum);
	};
    
	//Destructor
	~TLEInterface(){};
	
    // Applies an action and returns the reward. 
    reward_t act(Action action,bool skiped=true);

	// Set target video input
	void setTargetInput(int id){
		targetInputId = id;
	}

	// Check target is ready or not
	bool checkTargetInput(){
		return input[targetInputId].valid;
	}

	// Load video input
	bool load(std::string videoName,std::string labelName,std::string detName){
		return input[targetInputId].load(videoName,labelName,detName);
	};

	// Resets the input video
    void reset(){
		input[targetInputId].reset();
	};
	
	// Release the input video
	void release(){
		input[targetInputId].release();
	};
    
	// Indicates if the tracking has ended
    bool isEnded(){
		return input[targetInputId].isEnded();
	};

	// Returns tracker count
	int getTrackerCount(){
		return input[targetInputId].getTrackerCount();
	}

	// Returns the current frame Id
	int getCurrentFrameId(){
		return input[targetInputId].currentFrameId;
	}

	// Return Input video name
	std::string getName(){
		return input[targetInputId].name;
	}
	
    // Returns the vector of legal actions. 
    ActionVect getLegalActionSet();

    // Returns the target video screen
    cv::Mat getScreen();
};

std::string action_to_string(int a);

#endif
