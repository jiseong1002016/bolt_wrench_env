//
// Created by donghoon on 8/23/22.
//
#pragma once

#include "raisim/World.hpp"
#include "./helper/BasicEigenTypes.hpp"
#include "./raiboController.hpp"
#include "./helper/neuralNet.hpp"

/// TODO
#define OBSDIM 42
#define ACTDIM 12
#define COMDIM 6
#define ENCOUTDIM 128
#define ENCNUMLAYER 1

namespace controller {

class raiboLearningController {

 public:
  raiboLearningController();
  bool create(raisim::World *world, std::string& resourceDir);
  bool init();
  Eigen::VectorXf obsScalingAndGetAction();
  bool advance();
  bool reset();
  void setCommand(const Eigen::Ref<raisim::EigenVec>& command);

 private:

  raisim::RaiboLowController raiboController_;
  Eigen::VectorXf obs_;
  Eigen::VectorXf command_;
  Eigen::Matrix<float, ENCOUTDIM, 1> latent_;

  Eigen::VectorXf obsMean_;
  Eigen::VectorXf obsVariance_;
  raisim::nn::GRU<float, OBSDIM, ENCOUTDIM> encoder_;
  raisim::nn::Linear<float, ENCOUTDIM + OBSDIM + COMDIM, ACTDIM, raisim::nn::ActivationType::leaky_relu> actor_;
  bool onlyHidden_;

  double control_dt_;
  double communication_dt_;
};

}

