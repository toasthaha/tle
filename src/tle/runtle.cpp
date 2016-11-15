
#include "tle_interface.hpp"

int main( int argc, char* argv[]){
	
	TLEInterface tle(10);
	std::string labelFile = "/users/student/mr104/toasthaha/work/dashcam/label/000001.txt";
	std::string videoFile = "/users/student/mr104/toasthaha/work/dashcam/videos/000001.mp4";
	
	if (tle.load(videoFile,labelFile)==false){
		std::cout << "open file failed" << std::endl; 
		return -1;
	}
	std::cout << "open file sucess" << std::endl;

	double score =0;
	tle.getScreen();	
	std::cout << "score "<< tle.act( DECTECT ) << std::endl;
	while( tle.isEnded()==false){
		tle.getScreen();
		score += tle.act(TRACK);
	}
	std::cout<<"score "<<score<<std::endl;

}
