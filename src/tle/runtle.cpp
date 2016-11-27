
#include <istream>
#include <sstream>
#include <iomanip>
#include "tle_interface.hpp"

#define PERIOD 15
int main( int argc, char* argv[]){
	
	bool gui = true;
	TLEInterface tle(25,gui);
	

	//for(int t=1 ; t<=976; t++){
	for(int t=2 ; t<=2; t++){
  		std::stringstream labelName,videoName,detName; 
	
		labelName << "/data/cedl/dashcam/labels/";
		videoName << "/data/cedl/dashcam/videos/";
		detName   << "/data/cedl/dashcam/det/";
	  	labelName << std::setfill('0') << std::setw(6) << t << ".txt";
  		videoName << std::setfill('0') << std::setw(6) << t << ".mp4";
  		detName   << std::setfill('0') << std::setw(6) << t << "_det.txt";

		std::cout << labelName.str() << std::endl;
  		std::cout << videoName.str() << std::endl;
  		std::cout << detName.str() << std::endl;

		if (tle.load(videoName.str(),labelName.str(),detName.str())==false){
			std::cout << "open file failed" << std::endl; 
			continue;
		}

		std::stringstream outName;
		outName  << "period" <<  PERIOD << "/" << std::setfill('0') << std::setw(6) <<t;
		std::ofstream outFile(outName.str());
		

		std::cout << "open file sucess" << std::endl;

		float score   = 0;
		float imScore = 0;

		score = tle.act(DETECT);
		for(int frame=1;  tle.isEnded()==false; frame++){
			if(frame%PERIOD==0||tle.getTrackerCount()>20)
				imScore = tle.act(DETECT);
			else
				imScore = tle.act(TRACK);

			score += imScore ;
			std::cout << std::setw(15) << imScore \
            	      << std::setw(15) << score   << std::endl;
			outFile   << std::setw(15) << imScore \
            	      << std::setw(15) << score   << std::endl;
		}

		std::cout<<"score "<<score<<std::endl;
		outFile << score << std::endl;
	

	
		tle.reset();
		tle.releaseInput();
	}
}
