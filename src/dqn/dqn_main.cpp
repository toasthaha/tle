#include <cmath>
#include <iostream>
#include <fstream>
#include <tle_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "prettyprint.hpp"
#include "dqn.hpp"

DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_bool(cpu, false, "Using CPU model");
DEFINE_bool(evaluate , false, "Evaluation mode: only playing a game, no updates");
DEFINE_bool(show_frame, false, "Show the current frame in CUI");
DEFINE_int32(memory, 80000, "Capacity of replay memory");
DEFINE_int32(explore, 10000, "Number of iterations needed for epsilon to reach 0.1");
DEFINE_int32(memory_threshold, 2000, "Enough amount of transitions to start learning");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_double(evaluate_with_epsilon, 0.05, "Epsilon value to be used in evaluation mode");
DEFINE_double(gamma, 0.95, "Discount factor of future rewards (0,1]");
DEFINE_double(repeat_games, 1, "Number of games played in evaluation mode");
DEFINE_string(model, "", "Model file to load");
DEFINE_string(solver, "dqn_solver.prototxt", "Solver parameter file (*.prototxt)");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return 0.1;
  }
}

// Play one episode and return the total score
double PlayOneEpisode(
    TLEInterface& tle,
	dqn::DQN& dqn,
    const double epsilon,
    const bool update) {
	
	std::deque<dqn::FrameDataSp> past_frames;
  	auto total_score = 0.0;
	auto reward = 0.0;
	Action action;
	
	total_score = tle.act(DETECT);
	for(int  frame = 1; !tle.isEnded() && frame< FLAGS_skip_frame ;frame++)
		total_score += tle.act(TRACK);

  	while(!tle.isEnded()){
		if(FLAGS_gui){
			cv::imshow("Image",tle.getScreen());
			cv::waitKey(100);	
		}
   	 	
		//Read frames and preprocess
   		past_frames.push_back( dqn::PreprocessScreen(tle.getScreen()) );
    	if (past_frames.size() > dqn::kInputFrameCount) 
        		past_frames.pop_front();
	
	    if (past_frames.size() < dqn::kInputFrameCount) {
    			// If there are not past frames enough for DQN input, just select TRACK
				for(int  frame = 0; !tle.isEnded() && frame< FLAGS_skip_frame ;frame++)
					total_score += tle.act(TRACK);
		}else{
			dqn::InputFrames input_frames;
			std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
		
			// Select Action
			action = dqn.SelectAction(input_frames, epsilon);
			auto immediate_score = tle.act(action,false);
			for(int  frame = 1; !tle.isEnded() && frame< FLAGS_skip_frame ;frame++)
					immediate_score += tle.act(TRACK);
			total_score += immediate_score;

			//reward = (immediate_score == 0)? 0 : immediate_score / std::abs(immediate_score);
			reward = immediate_score / (FLAGS_skip_frame + 8 );
			//std::cout << reward << std::endl;

			if (update) {
        		// Add the current transition to replay memory
        		const auto transition = tle.isEnded() ?
	            dqn::Transition(input_frames, action, reward, boost::none) :
        	    dqn::Transition(input_frames, action, reward, dqn::PreprocessScreen(tle.getScreen()));
	   		    
				dqn.AddTransition(transition);
        		// If the size of replay memory is enough, update DQN
       			if (dqn.memory_size() > FLAGS_memory_threshold){ 
					dqn.Update();
				}
      		}
		}
	}// end of while loop
	tle.reset();
    return total_score;
}

int main(int argc, char** argv) {
	// Initialize
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();
	google::LogToStderr();
	
	int numInputs,numTrackers,numTrain,numValidate;
	std::ifstream videoList("videolist.txt");
	videoList >> numTrackers;
	videoList >> numTrain;
	videoList >> numValidate;
	numInputs = numTrain+numValidate;
	
	TLEInterface tle(numTrackers,numInputs);
	
	// Select running mode
	if(FLAGS_cpu)
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	else
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
	
	// Get the vector of legal actions
	const auto legal_actions = tle.getLegalActionSet();
	dqn::DQN dqn(legal_actions, FLAGS_solver, FLAGS_memory, FLAGS_gamma);
	dqn.Initialize();
		
	// Load pretrained model
	if(!FLAGS_model.empty()) {
		std::cout << "Loading " << FLAGS_model << std::endl;
		dqn.LoadTrainedModel(FLAGS_model);
	 }

	// Open log file
	std::ofstream logFile("log.csv");

	// Load input file
	for(int t = 0, target=0; t< numInputs;t++){
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
	
		tle.setTargetInput(t);
		if (tle.load(videoName.str(),labelName.str(),detName.str())==false){
			std::cout << "open file failed" << std::endl; 
			continue;
		}
	}

	/*
	// Just evaluate	
	if (FLAGS_evaluate) {
		auto total_score = 0.0;
		for (auto i = 0; i < FLAGS_repeat_games; ++i) {
			std::cout << "game: " << i << std::endl;
			const auto score = \
				PlayOneEpisode(tle, dqn, FLAGS_evaluate_with_epsilon, false);
				std::cout << "score: " << score << std::endl;
				total_score += score;
		}
		std::cout << "total_score: " << total_score << std::endl;
		return 0;
	}
	*/

	// Trainning
	int episode = 0;
	while(dqn.current_iteration() < 100000){
		// Train
		for(int t=0 ; t<numTrain ; t++){
			if( tle.checkTargetInput() == true){
				std::cout << "episode: " << episode++ << std::endl;
				const auto epsilon = CalculateEpsilon(dqn.current_iteration());
				tle.setTargetInput( episode % numTrain );
				PlayOneEpisode(tle, dqn, epsilon, true);
			}
		}
		// Validate
		logFile << episode << "," << dqn.current_iteration();
		for(int t=0; t<numValidate ; t++) {
			double validate_score = 0.0;
			if( tle.checkTargetInput()==true){
				tle.setTargetInput( t+numTrain );
				validate_score += PlayOneEpisode(tle, dqn, 0.02 , false);
				logFile << "," << validate_score ;
			}
		}
		logFile << std::endl;
	}

	// Evaluate All
	logFile << std::endl << std::endl;
	for(int t=0 ; t < numInputs ; t++){
		tle.setTargetInput(t);
		const auto eval_score = PlayOneEpisode(tle, dqn, 0.02 , false);
		logFile << tle.getName() <<"," << eval_score << std::endl;
	}

};

