
#include <istream>
#include <sstream>
#include <iomanip>
#include "tle_interface.hpp"

#define PERIOD 10
int main( int argc, char* argv[]){
	
	int numinputs,numtrackers,numtrain,numvalidate;
	std::ifstream videoList("videolist.txt");
	videoList >> numtrackers;
	videoList >> numtrain;
	videoList >> numvalidate;
	numinputs = numtrain + numvalidate;
	
	TLEInterface tle(numtrackers,1);
	std::ofstream outFile("period.log");
	
	tle.setTargetInput(0);

	for(int t=0,target=0 ; t<numinputs; t++){
  		std::stringstream labelName,videoName,detName; 
	
		videoList >> target;
		labelName << "/data/cedl/dashcam/labels/";
		videoName << "/data/cedl/dashcam/videos/";
		detName   << "/data/cedl/dashcam/det/";
	  	labelName << std::setfill('0') << std::setw(6) << target << ".txt";
  		videoName << std::setfill('0') << std::setw(6) << target << ".mp4";
  		detName   << std::setfill('0') << std::setw(6) << target << "_det.txt";

		std::cout << labelName.str() << std::endl;
  		std::cout << videoName.str() << std::endl;
  		std::cout << detName.str() << std::endl;
		outFile << "case" <<  target << std::endl;

		if (tle.load(videoName.str(),labelName.str(),detName.str())==false){
			std::cout << "open file failed" << std::endl; 
			continue;
		}
		std::cout << "open file sucess" << std::endl;

		float score   = 0;
		float imScore = 0;

		for(int period=10; period < 100 ; period+=10,score=0){
			tle.act(DETECT);
			for(int frame=1;  tle.isEnded()==false; frame++){
				if(frame%period==0)
					imScore = tle.act(DETECT);
				else
					imScore = tle.act(TRACK);

				score += imScore ;
				std::cout << std::setw(15) << imScore \
    	        	      << std::setw(15) << score   << std::endl;
				//outFile   << std::setw(15) << imScore \
            		      << std::setw(15) << score   << std::endl;
			}
		
			std::cout<<"score "<<score<<std::endl;
			outFile << "period " << period << "\tscore " << score << std::endl;
			tle.reset();
		}
		tle.release();
	}
}
