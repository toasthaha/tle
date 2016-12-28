#include "tle_input.hpp"
using namespace std;
using namespace cv;

// Load video file
bool TLEInput::load(string videoName,string labelName,string detName){

	cv::VideoCapture Cap;		
	std::ifstream labelFile,detFile;	

	// reset 
	name = videoName;

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
	frames.resize(totalFrame);	
	
	// Read video and  Background subtraction
	cv::Mat currentFrame,tempFrame;
	cv::BackgroundSubtractorMOG2 bgSubtractor(2,70,false);				

	for(int t=0; t<totalFrame;t++){
		// Read Video
		Cap.read(currentFrame);	
		frames[t].rawFrame = currentFrame.clone();
		// Initialzation groundtruth
		frames[t].det.resize(maxNumTrackers);
		frames[t].detValid.resize(maxNumTrackers,false);
		frames[t].groundtruth.resize(maxNumTrackers);
		frames[t].groundtruthFrame = Mat::zeros(currentFrame.size(),CV_8UC1);
		// Background subtraction
		bgSubtractor(currentFrame,tempFrame,0.5);
		frames[t].maskFrame = Mat::ones(currentFrame.size(),CV_8UC1);
		frames[t].maskFrame.setTo(0,tempFrame);
	}
	// Read label
	box in;
	while(!labelFile.eof()){
		in = readInputLabel(&labelFile);
		frames[in.frameId-1].groundtruth[in.trackId-1] = box2Rect(in);
		cv::rectangle(frames[in.frameId-1].groundtruthFrame,box2Rect(in),1,-1);
	}

	// Read det
	while(!detFile.eof()){
		in = readInputDet(&detFile);
		frames[in.frameId-1].det[in.trackId-1] = box2Rect(in);
		frames[in.frameId-1].detValid[in.trackId-1] = true;
	}
	valid = true;

	return true;
};
 
// Release input
void TLEInput::release(){
	frames.clear();	
}

// Indicates if the tracking has ended
bool TLEInput::isEnded(int frameId) {
	return (frameId>=totalFrame);
};

// Read Input label file
box TLEInput::readInputLabel(std::ifstream* labelFile){
	box in;
	(*labelFile)>>in.frameId>>in.trackId>>in.category \
	  		 >>in.x1>>in.y1>>in.x2>>in.y2;		

	//cout<<in.frameId<<" "<<in.trackId<<" "<<in.category \
	<<" "<<in.x1<<","<<in.y1<<","<<in.x2<<","<<in.y2<<"\n"; 	
	return in;
}

// Read Input det file
box TLEInput::readInputDet(std::ifstream* labelFile){
	box in;
	(*labelFile)>>in.frameId>>in.trackId>>in.category \
	  		 >>in.x1>>in.y1>>in.x2>>in.y2>>in.confident;		

	//cout<<in.frameId<<" "<<in.trackId<<" "<<in.category \
	<<" "<<in.x1<<","<<in.y1<<","<<in.x2<<","<<in.y2<<"\n"; 	
	return in;
}

// Convert box into Rect
Rect TLEInput::box2Rect(box in){
	Rect out;
	out.x = in.x1;
	out.y = in.y1;
	out.height = in.x2-in.x1;
	out.width  = in.y2-in.y1;
	return out;
}


