
#include "tle_interface.hpp"
#include "math.h"

using namespace std;
using namespace cv;

// Load video file
bool TLEInterface::load(string videoName,string labelName){

	cv::VideoCapture Cap;		
	std::ifstream labelFile;	

	// reset 
	currentFrameId = 0;

	// if already open file	
	if(labelFile.is_open())
		labelFile.close();	

	// Open label file
	labelFile.open(labelName);
	if(labelFile.is_open()==false)
		return false;

	// Open video file
	Cap.open(videoName);
	if(Cap.isOpened()==false)
		return false;
	
	// Get total frame number and initialize
	totalFrame = Cap.get(CV_CAP_PROP_FRAME_COUNT);
	input.frames.resize(totalFrame);	
	for(int t=0; t<totalFrame;t++){
		input.frames[t].groundtruth.resize(maxNumTrackers);
		input.frames[t].groundtruthValid.resize(maxNumTrackers,false);
	}
	
	// Read label
	box in;
	while(!labelFile.eof()){
		in = readInputLabel(&labelFile);
		input.frames[in.frameId-1].groundtruth[in.trackId-1] = box2Rect(in);
		input.frames[in.frameId-1].groundtruthValid[in.trackId-1] = true;
		input.frames[in.frameId-1].trackerCount++;
	}
	// EOF will get one extra count  
	input.frames[in.frameId-1].trackerCount--;
	

	// Read video and  Background subtraction
	cv::Mat currentFrame,tempFrame;
	cv::BackgroundSubtractorMOG2 bgSubtractor(2,50,false);				
	
	for(int t=0; t<totalFrame; t++){
		Cap.read(currentFrame);	
		input.frames[t].rawFrame = currentFrame.clone();
		//background subtraction
		bgSubtractor(currentFrame,tempFrame,0.5);
		input.frames[t].maskFrame = Mat::ones(currentFrame.size(),CV_8UC1);
		input.frames[t].maskFrame.setTo(0,tempFrame);
	}
	return true;
};

// Do action and return reeard 
// First time need to use DECTECT to reset tracker 
reward_t TLEInterface::act(Action action){


	cv::Mat currentFrame,maskFrame;
	Rect result,resultTrue,plot;

	// Set input frame pointer
	frameData* framePtr = &input.frames[currentFrameId];	

	currentFrame = framePtr->rawFrame.clone();
	maskFrame    = framePtr->maskFrame.clone();
	int trackerCount = framePtr->trackerCount;
	
	double score = 0;
	for(int t=0; t<maxNumTrackers; t++){
		// ACTION TRACK
		if(action==TRACK){
			if(trackerOn[t]){
				result = tracker[t].update(currentFrame);
				plot = result;
			}
			if(input.frames[currentFrameId].groundtruthValid[t]){
				if(trackerOn[t]){
					// using result calculating reward
					resultTrue = result & input.frames[currentFrameId].groundtruth[t];
					score  += (double)(resultTrue.area())/input.frames[currentFrameId].groundtruth[t].area() \
			 	 		    + (double)(resultTrue.area()-result.area())/result.area();
				}
				else{
					score += -1;
				}
			}	
		// ACTION DETECTION
		}else if(action==DETECT){
			// Reset each tracker 
			trackerOn[t] = framePtr->groundtruthValid[t];
			if(trackerOn[t]){
				tracker[t].init( framePtr->groundtruth[t], currentFrame );
				plot = input.frames[currentFrameId].groundtruth[t];
			}
		}

		// Plot target bounding box
		if(trackerOn[t])
			cv::rectangle(maskFrame,Point(plot.x,plot.y),\
							Point(plot.x+plot.width,plot.y+plot.height),0,-1);
			//cv::circle(returnFrame,Point(plot.x+plot.width/2,plot.y+plot.height/2),30,Scalar(255,255,255),-1);
	}
	if(action==TRACK && trackerCount>0)
		score /= trackerCount;
	else if(action==DETECT)
		score = -DETECT_TIME_PENALTY+ceil((double)trackerCount/4);

	// add mask to screen and resize
	currentFrame.setTo(Scalar(0,0,0),maskFrame);
	cv::resize(currentFrame,returnFrame,Point(84,84));
	
	// update frameId
	currentFrameId++;

	return score;
};


// Reset the track
void TLEInterface::reset(){
	//Set current frame id to 0
	currentFrameId = 0;
};

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

