//
// Created by donghoon on 8/23/22.
// 
 
#include <filesystem>

#include "../../include/raisin_6d_controller/raibo_6d_controller.hpp"

namespace controller {

raiboLearningController::raiboLearningController()
:
      encoder_(ENCNUMLAYER),
      actor_({256, 128})
      {
}

bool raiboLearningController::create(raisim::World *world, std::string& resourceDir) {
  control_dt_ = 0.01;
  communication_dt_ = 0.00025;

  auto robot = world->getObject("robot");
  raiboController_.create(reinterpret_cast<raisim::ArticulatedSystem *>(robot));

  /// load policy network parameters
  std::string network_path = resourceDir + "/controller/raisin_6d_controller/resource/jsh/";
  std::string model_itertaion = "50000";
  std::string encoder_file_name = std::string("GRU_") + model_itertaion + std::string(".txt");
  std::string actor_file_name = std::string("MLP_") + model_itertaion + std::string(".txt");
  std::string obs_mean_file_name = std::string("mean") + model_itertaion + std::string(".csv");
  std::string obs_var_file_name = std::string("var") + model_itertaion + std::string(".csv");

  encoder_.readParamFromTxt(network_path + encoder_file_name);
  actor_.readParamFromTxt(network_path + actor_file_name);

  obs_.setZero(raiboController_.getObDim());
  obsMean_.setZero(raiboController_.getObDim());
  obsVariance_.setZero(raiboController_.getObDim());

  std::string in_line;
  std::ifstream obsMean_file(network_path + obs_mean_file_name);
  std::ifstream obsVariance_file(network_path + obs_var_file_name);


  /// load observation mean and variance
  if (obsMean_file.is_open()) {
    for (int i = 0; i < obsMean_.size(); ++i) {
      std::getline(obsMean_file, in_line, '\n');
      obsMean_(i) = std::stof(in_line);
    }
  }

  if (obsVariance_file.is_open()) {
    for (int i = 0; i < obsVariance_.size(); ++i) {
      std::getline(obsVariance_file, in_line, '\n');
      obsVariance_(i) = std::stof(in_line);
    }
  }

  obsMean_file.close();
  obsVariance_file.close();

  command_.setZero(COMDIM);

  return true;
}

bool raiboLearningController::init() {
  return true;
}

bool raiboLearningController::advance() {

  raiboController_.updateObservation();
  raiboController_.advance(obsScalingAndGetAction().head(12));

  return true;
}

Eigen::VectorXf raiboLearningController::obsScalingAndGetAction() {
  /// normalize the obs
  obs_ = raiboController_.getObservation().cast<float>();

  for (int i = 0; i < obs_.size(); ++i) {
    obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
  }
  /// forward the obs to the encoder
  Eigen::Matrix<float, OBSDIM, 1> encoder_input = obs_.head(OBSDIM);
  latent_ = encoder_.forward(encoder_input);

  /// concat obs and e_out and forward to the actor
  Eigen::Matrix<float, ENCOUTDIM + OBSDIM + COMDIM, 1> actor_input;
  actor_input << latent_, obs_;

  Eigen::VectorXf action = actor_.forward(actor_input);
  return action;
}

bool raiboLearningController::reset() {
  raiboController_.reset();
  encoder_.initHidden();
  command_.setZero(COMDIM);
  return true;
}

void raiboLearningController::setCommand(const Eigen::Ref<raisim::EigenVec>& command) {
  command_ = command;
  raiboController_.setCommand(command_);
}

}
