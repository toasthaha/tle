
#include "tle_interface.hpp"
using namespace std;
using namespace cv;

// Load video file
bool TLEInterface::load(string videoName,string labelName){

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
	return currentFrame.empty();
};

// TODO add reward 
// Do action and return reeard 
// First time need to use DECTECT to reset tracker 
reward_t TLEInterface::act(Action action){
	Rect result,groundtruth;
	string category;
	int frameId,trackId,x1,y1,x2,y2;

	//only single object tracking now
	labelFile>>frameId>>trackId>>category>>x1>>y1>>x2>>y2;
	
	while (trackId < 1 || frameId < currentFrameId ){
		if( frameId > currentFrameId ){
			return 0;
		}
		labelFile>>frameId>>trackId>>category>>x1>>y1>>x2>>y2;
	}
	if(trackId != 1)
		return 0;

	//cout << frameId << " " << trackId << " " << category << " " << x1 << "," << y1 << "," << x2 << ","<< y2 << "\n"; 	
	
	groundtruth.x = x1;
	groundtruth.y = y1;
	groundtruth.width = x2-x1;
	groundtruth.height = y2-y1;
	
	
	if(action==TRACK){
		result = tracker->update(currentFrame);
		// using result calculating reward
		result = result & groundtruth;
		return (double)result.area()/groundtruth.area();
	}else if(action==DECTECT){
		tracker->init( groundtruth, currentFrame );
		return -5;
	}
	return -1;
};

// Returns the vector of legal actions. 
ActionVect TLEInterface::getLegalActionSet() {
	ActionVect legalSet(DECTECT,TRACK);
	return legalSet;
};

// Returns the current game screen
Mat TLEInterface::getScreen() {
	Cap >> currentFrame;
	currentFrameId = Cap.get(CV_CAP_PROP_POS_FRAMES);
	return currentFrame;
};

