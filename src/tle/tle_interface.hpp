#ifndef __TLE_HPP__
#define __TLE_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <gflags/gflags.h>

//target tracker
#include "kcftracker.hpp"


static const std::string Version = "0.1";

enum Action {
	DECTECT = 1,
	TRACK	= 2
};

typedef double reward_t;
typedef std::vector<Action> ActionVect;


class TLEInterface
{
protected:
    reward_t episode_score; 	// Score accumulated throughout the course of an episode
    int max_num_frames;     	// Maximum number of frames for each episode
	std::ifstream labelFile;	// label input file
	cv::VideoCapture Cap;		// Read video
	cv::Mat currentFrame;		// Frame readded
	int currentFrameNum;		// Current Fream Num
	
	//===================
	// Tracker setting 
	//===================
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = false;
	KCFTracker* tracker;
	//===================

public:
	// Constructor
	TLEInterface(){
		 tracker = new KCFTracker(HOG,FIXEDWINDOW,MULTISCALE,LAB);
	};
    
	//Destructor
	~TLEInterface(){
		delete tracker;
	};

	//load
	bool load(std::string videofile,std::string labelfile);

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

};

#endif
