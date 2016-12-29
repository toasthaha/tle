
#include <iostream>
#include "tle_interface.hpp"

using namespace std;
using namespace cv;

// Do action and return reeard 
// First time need to use DECTECT to reset tracker 
reward_t TLEInterface::act(Action action, bool train){

    std::vector<cv::Rect> target;
    cv::Mat currentFrame,maskFrame,scoreFrame;
    Rect plot;

    // Set input frame pointer
    frameData* framePtr = &input[targetInputId].frames[currentFrameId]; 

    currentFrame = framePtr->rawFrame.clone();
    maskFrame    = framePtr->maskFrame.clone();
    //scoreFrame   = Mat::zeros(currentFrame.size(),CV_8UC1);
    input[targetInputId].returnFrame = framePtr->rawFrame.clone();
    
    // Action
    target.clear();
    for(int t=0; t<maxNumTrackers; t++){
        // ACTION TRACK
        if(action==TRACK){
            if(trackerOn[t]){
                plot = tracker[t].update(currentFrame);
            }
        }
        // ACTION DETECTION
        else if(action==DETECT){
            // Reset each tracker 
            if( t < framePtr->det.size() ){
                trackerOn[t] = true;
                tracker[t].init( framePtr->det[t], currentFrame );
                plot = framePtr->det[t];
            }
            else 
                trackerOn[t] = false;
        }
        // Store Target
        target.push_back(plot);

        // Plot target bounding box
        if(trackerOn[t]){
            cv::rectangle(maskFrame,plot,0,-1);
            cv::rectangle(scoreFrame,plot,1,-1);
            // Draw Bounding Box
            //cv::rectangle(input[targetInputId].returnFrame,plot,cv::Scalar(0,0,255),30);
        }
    }
    

    // calculate fram buffer number
    if(action==TRACK){
        // Assume Tracker 120 FPS
        frameBufferNum += 1 - (1./120)*OUTPUT_FPS;
    }
    else{
        // Assume Detector 2 FPS
        frameBufferNum += 1 - (1./2)*OUTPUT_FPS;
    }

    // Calculate IoU Score
    double score = 0;
    for(int t=0; t < framePtr->groundtruth.size(); t++){
        double iou = 0;
        for(int idx=0 ; idx<target.size(); idx++){
                double i = (framePtr->groundtruth[t] & target[idx]).area();
                double u = (framePtr->groundtruth[t] | target[idx]).area();
                if(iou < i/u)
                    iou = i/u;
        }
        score += iou;
    }
    score = (framePtr->groundtruth.size()==0)? 0 : score/framePtr->groundtruth.size(); 
    if(train)
        score = score*2 -1; 
    
    /*
    score  = cv::sum(framePtr->groundtruthFrame &  scoreFrame).val[0];
    double total_area = cv::sum(framePtr->groundtruthFrame | scoreFrame).val[0];
    if( total_area > 0)
        score = (score/total_area)*2 - 1 ;
    else
        score = 0 ;
    */
    if(frameBufferNum < 0 )
        score += -100;

    // add mask to screen
    input[targetInputId].returnFrame.setTo(Scalar(0,0,0),maskFrame);

    // update frameId
    currentFrameId++;

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
