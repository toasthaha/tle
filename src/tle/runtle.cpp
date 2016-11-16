
#include "tle_interface.hpp"

int main( int argc, char* argv[]){
	
	bool gui = true;
	TLEInterface tle(10,gui);
	std::string labelFile = "/users/student/mr104/toasthaha/work/dashcam/label/000002.txt";
	std::string videoFile = "/users/student/mr104/toasthaha/work/dashcam/videos/000002.mp4";
	
	if (tle.load(videoFile,labelFile)==false){
		std::cout << "open file failed" << std::endl; 
		return -1;
	}

	std::cout << "open file sucess" << std::endl;

	double score =0;
	score =  tle.act(DETECT);
	std::cout << score << std::endl;
	
	while( tle.isEnded()==false){
		score += tle.act(TRACK);
		std::cout << score << std::endl;
	}
	std::cout<<"score "<<score<<std::endl;
}
