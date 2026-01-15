//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "./helper/BasicEigenTypes.hpp"
#include <raisim/RaisimServer.hpp>
//
// Created by donghoon on 8/11/22.
//

#ifndef _RAISIM_GYM_RAIBO_LOW_CONTROLLER_HPP
#define _RAISIM_GYM_RAIBO_LOW_CONTROLLER_HPP

namespace raisim {

class RaiboLowController {
 public:
  inline bool create(raisim::ArticulatedSystem * robot) {
    raibo_ = robot;
    gc_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_.resize(raibo_->getDOF());
    gc_init_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_init_.resize(raibo_->getDOF());

    /// Observation
    nominalJointConfig_.setZero(nJoints_);
    nominalJointConfig_ << 0, 0.63026, -1.29938, 0, 0.63026, -1.29938, 0, 0.63026, -1.29938, 0, 0.63026, -1.29938;
    jointTarget_.setZero(nJoints_);
    gc_init_ << 0, 0, 0.5225, 1, 0, 0, 0, nominalJointConfig_;
    gv_init_.setZero();

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);
    previousAction_.setZero(actionDim_);

    actionMean_ << nominalJointConfig_; /// joint target
    actionStd_ << Eigen::VectorXd::Constant(nJoints_, 0.1); /// joint target

    obDouble_.setZero(obDim_);

    /// pd controller
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(100.0);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(1.0);
    raibo_->setPdGains(jointPgain_, jointDgain_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);

    return true;
  };

  void reset() {
    command_.setZero(6);
    raibo_->getState(gc_, gv_);
    jointTarget_ = gc_.tail(nJoints_);
    previousAction_ << gc_.tail(nJoints_);
  }

  void updateStateVariables() {
    raibo_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, baseRot_);
    bodyAngVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);
  }

  bool advance(const Eigen::Ref<EigenVec> &action) {
//    raibo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
//    raibo_->setPdGains(jointPgain_, jointDgain_);
    /// action scaling
    jointTarget_ = action.cast<double>();
    jointTarget_ = jointTarget_.cwiseProduct(actionStd_);
    jointTarget_ += actionMean_;

    pTarget_.tail(nJoints_) = jointTarget_;
    raibo_->setPdTarget(pTarget_, vTarget_);

    previousAction_ = jointTarget_;

    return true;
  }

  void updateObservation() {
    updateStateVariables();

    /// body orientation
    obDouble_.head(3) = baseRot_.e().row(2).transpose();
    /// body ang vel
    obDouble_.segment(3, 3) = bodyAngVel_;
    /// joint pos
    obDouble_.segment(6, nJoints_) = gc_.tail(nJoints_);
    /// joint vel
    obDouble_.segment(18, nJoints_) = gv_.tail(nJoints_);
    /// previous action
    obDouble_.segment(30, nJoints_) = previousAction_;
    /// command
    obDouble_.tail(6) = command_;
  }

  Eigen::VectorXd getObservation() {
    return obDouble_;
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    command_ = command.cast<double>();
  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }

  void getInitState(Eigen::VectorXd &gc, Eigen::VectorXd &gv) {
    gc.resize(gcDim_);
    gv.resize(gvDim_);
    gc << gc_init_;
    gv << gv_init_;
  }

  Eigen::VectorXd getJointPGain() const { return jointPgain_; }
  Eigen::VectorXd getJointDGain() const { return jointDgain_; }
  Eigen::VectorXd getJointPTarget() const { return jointTarget_; }
  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  [[nodiscard]] double getSimDt() { return simDt_; }
  [[nodiscard]] double getConDt() { return conDt_; }
  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }

  void setSimDt(double dt) { simDt_ = dt; };
  void setConDt(double dt) { conDt_ = dt; };

  // robot configuration variables
  raisim::ArticulatedSystem *raibo_;
  Eigen::VectorXd nominalJointConfig_;
  static constexpr int nJoints_ = 12;
  static constexpr int actionDim_ = 12;
  static constexpr size_t obDim_ = 48;
  double simDt_ = .0025;
  static constexpr int gcDim_ = 19;
  static constexpr int gvDim_ = 18;

  // robot state variables
  Eigen::VectorXd gc_, gv_, gc_init_, gv_init_;
  Eigen::Vector3d bodyAngVel_;
  raisim::Mat<3, 3> baseRot_;

  // robot observation variables
  Eigen::VectorXd obDouble_;

  // control variables
  double conDt_ = 0.01;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_;
  Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
  Eigen::VectorXd jointTarget_;
  Eigen::VectorXd jointPgain_, jointDgain_;
  Eigen::VectorXd command_;

};

}

#endif //_RAISIM_GYM_RAIBO_CONTROLLER_HPP
