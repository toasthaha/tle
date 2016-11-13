
#include "tle_interface.hpp"
using namespace std;
using namespace cv;

// Load video file
bool TLEInterface::load(string videofile,string labelfile){
	
	// open label file
	labelFile.open(labelfile);

	// open video file
	Cap.open(videofile);
	// Check is video file open sucess 
	if(Cap.isOpened()==false)
		return false;
	else
		return true;
};

// Reset the track
void TLEInterface::reset(){

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
	while (trackId !=1)
		labelFile>>frameId>>trackId>>category>>x1>>y1>>x2>>y2;
	groundtruth.x = x1;
	groundtruth.y = y1;
	groundtruth.width = x2-x1;
	groundtruth.height = y1-y2;
	
	
	if(action==TRACK){
		result = tracker->update(currentFrame);
		// using result calculating reward
		result = result | groundtruth;
		return result.area()/groundtruth.area();
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
	return currentFrame;
};

