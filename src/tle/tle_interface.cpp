
#include <iostream>
#include "tle_interface.hpp"

using namespace std;
using namespace cv;


// Do action and return reeard 
// First time need to use DECTECT to reset tracker 
reward_t TLEInterface::act(Action action, bool skiped){

	cv::Mat currentFrame,maskFrame,scoreFrame;
	Rect plot;

	// Set input frame pointer
	int currentFrameId = input[targetInputId].currentFrameId;
	frameData* framePtr = &input[targetInputId].frames[currentFrameId];	

	currentFrame = framePtr->rawFrame.clone();
	maskFrame    = framePtr->maskFrame.clone();
	scoreFrame   = Mat::zeros(currentFrame.size(),CV_8UC1);
	input[targetInputId].returnFrame = framePtr->rawFrame.clone();
	
	// Action
	double score = 0;
	for(int t=0; t<maxNumTrackers; t++){
		// ACTION TRACK
		if(action==TRACK){
			if(trackerOn[t]){
				plot = tracker[t].update(currentFrame);
				if(skiped==false)
					input[targetInputId].plotColor = Scalar(255,0,0);
			}
		// ACTION DETECTION
		}else if(action==DETECT){
			// Reset each tracker 
			trackerOn[t] = framePtr->detValid[t];
			if(trackerOn[t]){
				tracker[t].init( framePtr->det[t], currentFrame );
				plot = framePtr->det[t];
				if(skiped==false)
					input[targetInputId].plotColor = Scalar(0,0,255);
			}
		}

		// Plot target bounding box
		if(trackerOn[t]){
			cv::rectangle(maskFrame,plot,0,-1);
			cv::rectangle(scoreFrame,plot,1,-1);
			//cv::rectangle(input[targetInputId].returnFrame,plot,input[targetInputId].plotColor,30);
		}
	}
	
	// Calculate IoU Score
	score  = cv::sum(framePtr->groundtruthFrame &  scoreFrame).val[0];
	double total_area = cv::sum(framePtr->groundtruthFrame | scoreFrame).val[0];
	if( total_area > 0)
		score = (score/total_area)*2 - 1 ;
	else
		score = 0 ;
	
	if(action==DETECT)
		score += -DETECT_TIME_PENALTY + framePtr->trackerCount/4;

	// add mask to screene
	input[targetInputId].returnFrame.setTo(Scalar(0,0,0),maskFrame);

	// update frameId
	input[targetInputId].currentFrameId++;

	return score;
};

// Returns the current game screen
Mat TLEInterface::getScreen() {
	return input[targetInputId].returnFrame;
};

// Returns the vector of legal actions. 
ActionVect TLEInterface::getLegalActionSet() {
	ActionVect legalSet(2);
	legalSet[0] = DETECT;
	legalSet[1] = TRACK;
	cout << "ActionVect : "<<legalSet.size()<< endl;
	return legalSet;
};

std::string action_to_string(int a){
	static string tmp_action_to_string[] = {
		"DETECT",
		"TRACK"
	};
	assert( a >=0 && a<=1 );
	return tmp_action_to_string[a];
};
