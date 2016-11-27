
#include "tle_interface.hpp"
#include "math.h"

using namespace std;
using namespace cv;

// Load video file
bool TLEInterface::load(string videoName,string labelName,string detName){

	cv::VideoCapture Cap;		
	std::ifstream labelFile,detFile;	

	// reset 
	currentFrameId = 0;

	// if already open file	
	if(labelFile.is_open())
		labelFile.close();	

	// Open label file
	labelFile.open(labelName);
	if(labelFile.is_open()==false)
		return false;
	
	// if already open file	
	if(detFile.is_open())
		detFile.close();	

	// Open label file
	detFile.open(detName);
	if(detFile.is_open()==false)
		return false;

	// Open video file
	Cap.open(videoName);
	if(Cap.isOpened()==false)
		return false;
	
	// Get total frame number and initialize
	totalFrame = Cap.get(CV_CAP_PROP_FRAME_COUNT);
	input.frames.resize(totalFrame);	
	// Read video and  Background subtraction
	cv::Mat currentFrame,tempFrame;
	cv::BackgroundSubtractorMOG2 bgSubtractor(2,70,false);				

	for(int t=0; t<totalFrame;t++){
		// Read Video
		Cap.read(currentFrame);	
		input.frames[t].rawFrame = currentFrame.clone();
		// Initialzation groundtruth
		input.frames[t].det.resize(maxNumTrackers);
		input.frames[t].groundtruth.resize(maxNumTrackers);
		input.frames[t].groundtruthValid.resize(maxNumTrackers,false);
		input.frames[t].groundtruthFrame = Mat::zeros(currentFrame.size(),CV_8UC1);
		// Background subtraction
		bgSubtractor(currentFrame,tempFrame,0.5);
		input.frames[t].maskFrame = Mat::ones(currentFrame.size(),CV_8UC1);
		input.frames[t].maskFrame.setTo(0,tempFrame);
	}
	
	// Read label
	box in;
	while(!labelFile.eof()){
		in = readInputLabel(&labelFile);
		input.frames[in.frameId-1].groundtruth[in.trackId-1] = box2Rect(in);
		input.frames[in.frameId-1].groundtruthValid[in.trackId-1] = true;
		input.frames[in.frameId-1].trackerCount++;
		cv::rectangle(input.frames[in.frameId-1].groundtruthFrame,box2Rect(in),1,-1);
	}
	// EOF will get one extra count  
	input.frames[in.frameId-1].trackerCount--;

	// Read det
	while(!detFile.eof()){
		in = readInputDet(&detFile);
		input.frames[in.frameId-1].det[in.trackId-1] = box2Rect(in);
	}

	return true;
};

// Do action and return reeard 
// First time need to use DECTECT to reset tracker 
reward_t TLEInterface::act(Action action,bool skiped){


	cv::Mat currentFrame,maskFrame,scoreFrame;
	Rect result,resultTrue,plot;

	// Set input frame pointer
	frameData* framePtr = &input.frames[currentFrameId];	

	currentFrame = framePtr->rawFrame.clone();
	maskFrame    = framePtr->maskFrame.clone();
	scoreFrame = Mat::zeros(currentFrame.size(),CV_8UC1);
	returnFrame  = currentFrame.clone();
	
	int trackerCount = framePtr->trackerCount;
	
	double score = 0;
	for(int t=0; t<maxNumTrackers; t++){
		// ACTION TRACK
		if(action==TRACK){
			if(trackerOn[t]){
				result = tracker[t].update(currentFrame);
				plot = result;
				if(skiped==false)
					plotColor = Scalar(255,0,0);
			}
		// ACTION DETECTION
		}else if(action==DETECT){
			// Reset each tracker 
			trackerOn[t] = framePtr->groundtruthValid[t];
			if(trackerOn[t]){
				tracker[t].init( framePtr->groundtruth[t], currentFrame );
				plot = framePtr->groundtruth[t];
				if(skiped==false)
					plotColor = Scalar(0,0,255);
			}
		}

		// Plot target bounding box
		if(trackerOn[t]){
			cv::rectangle(maskFrame,plot,0,-1);
			cv::rectangle(returnFrame,plot,plotColor,30);
			cv::rectangle(scoreFrame,plot,1,-1);
		}
	}
	if(action==TRACK){
		score  = cv::sum(framePtr->groundtruthFrame &  scoreFrame).val[0];
	    double total_area = cv::sum(framePtr->groundtruthFrame | scoreFrame).val[0];
		if( total_area > 0)
			score = (score/total_area)*2 - 1 ;
		else
			score = 0 ;
	}
	else if(action==DETECT)
		score = -DETECT_TIME_PENALTY + trackerCount/4;

	// add mask to screene
	returnFrame.setTo(Scalar(0,0,0),maskFrame);
	
	// update frameId
	currentFrameId++;

	return score;
};


// Reset the track
void TLEInterface::reset(){
	//Set current frame id to 0
	currentFrameId = 0;
};

// Release input
void TLEInterface::releaseInput(){
	input.frames.clear();	
}

// Indicates if the tracking has ended
bool TLEInterface::isEnded() {
	return (currentFrameId>=totalFrame);
};

// Returns tracker count
int TLEInterface::getTrackerCount(){
	return input.frames[currentFrameId-1].trackerCount;
};

// Returns the current game screen
Mat TLEInterface::getScreen() {
	return returnFrame;
};

// Read Input label file
box TLEInterface::readInputLabel(std::ifstream* labelFile){
	box in;
	(*labelFile)>>in.frameId>>in.trackId>>in.category \
	  		 >>in.x1>>in.y1>>in.x2>>in.y2;		

	//cout<<in.frameId<<" "<<in.trackId<<" "<<in.category \
	<<" "<<in.x1<<","<<in.y1<<","<<in.x2<<","<<in.y2<<"\n"; 	
	return in;
}

// Read Input det file
box TLEInterface::readInputDet(std::ifstream* labelFile){
	box in;
	(*labelFile)>>in.frameId>>in.trackId>>in.category \
	  		 >>in.x1>>in.y1>>in.x2>>in.y2>>in.confident;		

	//cout<<in.frameId<<" "<<in.trackId<<" "<<in.category \
	<<" "<<in.x1<<","<<in.y1<<","<<in.x2<<","<<in.y2<<"\n"; 	
	return in;
}


// Convert box into Rect
Rect TLEInterface::box2Rect(box in){
	Rect out;
	out.x = in.x1;
	out.y = in.y1;
	out.height = in.x2-in.x1;
	out.width  = in.y2-in.y1;
	return out;
}

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

}	

