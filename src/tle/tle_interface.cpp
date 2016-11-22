
#include "tle_interface.hpp"
#include "math.h"

using namespace std;
using namespace cv;

// Load video file
bool TLEInterface::load(string videoName,string labelName){

	// if already open file	
	if(labelFile.is_open())
		labelFile.close();	

	// Open label file
	labelFile.open(labelName);
	// Check if label file open sucess
	if(labelFile.is_open()==false)
		return false;

	// Open video file
	Cap.open(videoName);
	// Check if video file open sucess 
	if(Cap.isOpened()==false)
		return false;
	else
		return true;
};

// Reset the track
void TLEInterface::reset(){

	//Set current frame id to 0
	currentFrameId = 0;

	//Reset labelfile
	labelFile.clear();                 // clear fail and eof bits
	labelFile.seekg(0, std::ios::beg); // back to the start!

	//Reset videofile
	Cap.set(CV_CAP_PROP_POS_MSEC, 0);

};

// Indicates if the tracking has ended
bool TLEInterface::isEnded() {
	return videoEnded;
};

// Do action and return reeard 
// First time need to use DECTECT to reset tracker 
reward_t TLEInterface::act(Action action){

	box in;
	Rect result,resultTrue,plot;
	vector<Rect>groundtruth(maxNumTrackers);
	vector<bool>groundtruthValid(maxNumTrackers,false);

	//get Next frame
	videoEnded = !Cap.read(currentFrame);
	currentFrameId = Cap.get(CV_CAP_PROP_POS_FRAMES);
	returnFrame = currentFrame;

	//backgrouind subtraction
	bgSubtractor(currentFrame,maskFrame,0.5);
	returnFrame = currentFrame;
	Mat mask = Mat::ones(currentFrame.size(),CV_8UC1);
	mask.setTo(0,maskFrame);
	//returnFrame.setTo(Scalar(0,0,0),mask);

	// pre-fetch
	if(currentFrameId==1){
		in = readInputLabel();
		nextBox = in;
	}

	//take the pre-fetch one
	in = nextBox;

	//read GroundTruthLabel
	while (in.frameId == currentFrameId ){
		// store box into groundtruth
		groundtruth[in.trackId-1] = box2Rect(in);
		groundtruthValid[in.trackId-1] = true;
		// read next input
		if(labelFile.eof())
			return 0;
		in = readInputLabel();
	}
	nextBox = in;

	double score = 0;

	for(int t=0; t<maxNumTrackers; t++){
		// ACTION TRACK
		if(action==TRACK){
			if(trackerOn[t]){
				result = tracker[t].update(currentFrame);
				plot = result;
			}
			if(groundtruthValid[t]){
				if(trackerOn[t]){
					// using result calculating reward
					resultTrue = result & groundtruth[t];
					score  += (double)(resultTrue.area())/groundtruth[t].area() \
			 	 		    + (double)(resultTrue.area()-result.area())/result.area();
				}
				else{
					score += -1;
				}
			}	
		// ACTION DETECTION
		}else if(action==DETECT){
			// Reset tracker Count
			if(t==0)trackerCount = 0;

			// Reset each tracker 
			trackerOn[t] = groundtruthValid[t];
			if(groundtruthValid[t]){
				tracker[t].init( groundtruth[t], currentFrame );
				plot = groundtruth[t];
				trackerCount++;
			}
		}
		if(trackerOn[t])
			cv::rectangle(mask,Point(plot.x,plot.y),Point(plot.x+plot.width,plot.y+plot.height),0,-1);
			//cv::circle(returnFrame,Point(plot.x+plot.width/2,plot.y+plot.height/2),30,Scalar(255,255,255),-1);
	}
	if(action==TRACK && trackerCount>0)
		score /= trackerCount;
	else if(action==DETECT)
		score = -DETECT_TIME_PENALTY+ceil((double)trackerCount/4);


	returnFrame.setTo(Scalar(0,0,0),mask);

	return score;
};

// Returns the vector of legal actions. 
ActionVect TLEInterface::getLegalActionSet() {
	ActionVect legalSet(2);
	legalSet[0] = DETECT;
	legalSet[1] = TRACK;
	cout << "ActionVect : "<<legalSet.size()<< endl;
	return legalSet;
};

// Returns tracker count
int TLEInterface::getTrackerCount(){
	return trackerCount;
};

// Returns the current game screen
Mat TLEInterface::getScreen() {
	return returnFrame;
};

// Read Input label file
box TLEInterface::readInputLabel(){
	box in;
	labelFile>>in.frameId>>in.trackId>>in.category \
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

std::string action_to_string(int a){
	static string tmp_action_to_string[] = {
		"DETECT",
		"TRACK"
	};
	assert( a >=0 && a<=1 );
	return tmp_action_to_string[a];

}	

