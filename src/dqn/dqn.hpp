#ifndef DQN_HPP_
#define DQN_HPP_

#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <tle_interface.hpp>
#include <caffe/caffe.hpp>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>
#include <opencv2/opencv.hpp>


typedef unsigned char pixel_t;

namespace dqn {

constexpr auto kCroppedFrameSize = 84;
constexpr auto kCroppedFrameDataSize = kCroppedFrameSize * kCroppedFrameSize;
constexpr auto kInputFrameCount = 4;
constexpr auto kInputDataSize = kCroppedFrameDataSize * kInputFrameCount;
constexpr auto kMinibatchSize = 32;
constexpr auto kMinibatchDataSize = kInputDataSize * kMinibatchSize;
constexpr auto kGamma = 0.95f;
constexpr auto kOutputCount = 18;

using FrameData = std::array<float, kCroppedFrameDataSize>;
using FrameDataSp = std::shared_ptr<FrameData>;
using InputFrames = std::array<FrameDataSp, 4>;
using Transition = std::tuple<InputFrames, Action, float, boost::optional<FrameDataSp>,int>;

using FramesLayerInputData = std::array<float, kMinibatchDataSize>;
using TargetLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;
using FilterLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;
using FrameNumLayerInputData = std::array<float,kMinibatchSize>;

/**
 * Deep Q-Network
 */
class DQN {
public:
  DQN(
      const ActionVect& legal_actions,
      const std::string& solver_param,
      const int replay_memory_capacity,
      const double gamma) :
        legal_actions_(legal_actions),
        solver_param_(solver_param),
        replay_memory_capacity_(replay_memory_capacity),
        gamma_(gamma),
        current_iter_(0),
        random_engine(0) {}

  /**
   * Initialize DQN. Must be called before calling any other method.
   */
  void Initialize();

  /**
   * Load a trained model from a file.
   */
  void LoadTrainedModel(const std::string& model_file);

  /**
   * Select an action by epsilon-greedy.
   */
  Action SelectAction(const InputFrames& input_frames, double epsilon,float frameNum);

  /**
   * Add a transition to replay memory
   */
  void AddTransition(const Transition& transition);

  /**
   * Update DQN using one minibatch
   */
  void Update();

  int memory_size() const { return replay_memory_.size(); }
  int current_iteration() const { return current_iter_; }
  void set_iter(int t){ current_iter_ = t;} 

private:
  using SolverSp = std::shared_ptr<caffe::Solver<float>>;
  using NetSp = boost::shared_ptr<caffe::Net<float>>;
  using BlobSp = boost::shared_ptr<caffe::Blob<float>>;
  using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float>>;

  std::pair<Action, float> SelectActionGreedily(const InputFrames& last_frames,float frameNum);
  std::vector<std::pair<Action, float>> SelectActionGreedily(
      const std::vector<InputFrames>& last_frames,const std::vector<float>& frameNum);
  void InputDataIntoLayers(
      const FramesLayerInputData& frames_data,
      const TargetLayerInputData& target_data,
      const FilterLayerInputData& filter_data,
      const FrameNumLayerInputData& frameNum_data);

  const ActionVect legal_actions_;
  const std::string solver_param_;
  const int replay_memory_capacity_;
  const double gamma_;
  int current_iter_;
  std::deque<Transition> replay_memory_;
  SolverSp solver_;
  NetSp net_;
  BlobSp q_values_blob_;
  MemoryDataLayerSp frames_input_layer_;
  MemoryDataLayerSp target_input_layer_;
  MemoryDataLayerSp filter_input_layer_;
  MemoryDataLayerSp frameNum_input_layer_;
  TargetLayerInputData dummy_input_data_;
  std::mt19937 random_engine;
};

/**
 * Preprocess an TLE screen (downsampling & grayscaling)
 */
FrameDataSp PreprocessScreen(const cv::Mat& raw_screen);

}

#endif /* DQN_HPP_ */
