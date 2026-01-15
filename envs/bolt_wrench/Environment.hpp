#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "UtilityFunctions.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:
  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), resourceDir_(resourceDir), visualizable_(visualizable) {

    /// 1. 시뮬레이션 설정 로드
    simulation_dt_ = cfg["simulation_dt"].template As<double>();
    control_dt_ = cfg["control_dt"].template As<double>();
    thread_pitch_ = cfg["thread_pitch"].template As<double>();

    // 초기 자세 및 힘 값 로드
    ee_left_init_ = cfg["ee_01_left_gc"].template As<double>();
    ee_right_init_ = cfg["ee_02_right_gc"].template As<double>();
    ee_left_force_ = cfg["ee_01_left_gf"].template As<double>();
    ee_right_force_ = cfg["ee_02_right_gf"].template As<double>();

    /// 2. 월드 초기화
    world_ = std::make_unique<raisim::World>();
    world_->setTimeStep(simulation_dt_);
    world_->addGround();

    /// 3. 객체 스폰
    bolt_ = world_->addArticulatedSystem(
        resourceDir_ + "/bolt/bolt/urdf/bolt.urdf",
        "", {}, raisim::COLLISION(1), -1);
    bolt_->setName("bolt");

    wrench_ = world_->addArticulatedSystem(
        resourceDir_ + "/wrench/wrench/urdf/wrench.urdf",
        "", {}, raisim::COLLISION(2), -1);
    wrench_->setName("wrench");

    end_effector_ = world_->addArticulatedSystem(
        resourceDir_ + "/end-effector/urdf/end-effector.urdf",
        "", {}, raisim::COLLISION(3), -1);
    end_effector_->setName("end_effector");

    /// 4. 차원 설정
    gcDim_ = end_effector_->getGeneralizedCoordinateDim();
    gvDim_ = end_effector_->getDOF();
    nJoints_ = gvDim_;

    obDim_ = gcDim_ + gvDim_; 
    actionDim_ = nJoints_;
    
    obDouble_.setZero(obDim_);
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);

    /// 5. 초기 관측값 업데이트
    updateObservation();
  }

  void init() final { }

  void reset() final {
    // 1. Bolt 초기화
    Eigen::VectorXd bolt_gc = Eigen::VectorXd::Zero(bolt_->getGeneralizedCoordinateDim());
    Eigen::VectorXd bolt_gv = Eigen::VectorXd::Zero(bolt_->getDOF());
    bolt_->setState(bolt_gc, bolt_gv);

    // 2. Wrench 초기화
    Eigen::VectorXd wrench_gc = Eigen::VectorXd::Zero(wrench_->getGeneralizedCoordinateDim());
    Eigen::VectorXd wrench_gv = Eigen::VectorXd::Zero(wrench_->getDOF());
    wrench_gc.head(7) << 0.0, 0.0, 0.034, 1.0, 0.0, 0.0, 0.0;
    wrench_->setState(wrench_gc, wrench_gv);

    // 3. End-effector 초기화
    Eigen::VectorXd ee_gc = Eigen::VectorXd::Zero(end_effector_->getGeneralizedCoordinateDim());
    Eigen::VectorXd ee_gv = Eigen::VectorXd::Zero(end_effector_->getDOF());
    ee_gc.head(7) << 0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0; 

    // ★ 사용자의 Custom 초기값 적용 (인덱스는 URDF에 맞게 조정 필요)
    // 예시: 7번이 left, 11번이 right 조인트라고 가정 (이전 코드 기반)
    ee_gc[7] = ee_left_init_;  
    ee_gc[11] = ee_right_init_;
    
    end_effector_->setState(ee_gc, ee_gv);

    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    Eigen::VectorXd tau_total = action.cast<double>();
    
    if(tau_total.size() != end_effector_->getDOF()) {
        tau_total = Eigen::VectorXd::Zero(end_effector_->getDOF());
    }

    // [3] 상시 적용되는 Grasping Force 추가 (TC=true 효과)
    // 에이전트가 내는 행동(Action)에 "기본 악력"을 더해줍니다.
    // 인덱스 6, 13 등은 URDF 조인트 인덱스 확인 필수!
    tau_total[6] += ee_left_force_;
    tau_total[13] += ee_right_force_;

    /// Simulation Loop
    for (int i = 0; i < int(control_dt_ / simulation_dt_); i++) {
      const auto& gc = end_effector_->getGeneralizedCoordinate();
      const auto& gv = end_effector_->getGeneralizedVelocity();

      // Gear Coupling
      double error_p1 = gc[7] + gc[11];
      double error_v1 = gv[6] + gv[10];
      double tau_gear1 = -kp_gear_ * error_p1 - kd_gear_ * error_v1;

      double error_p2 = gc[10] + gc[14];
      double error_v2 = gv[9] + gv[13];
      double tau_gear2 = -kp_gear_ * error_p2 - kd_gear_ * error_v2;

      tau_total[6] += tau_gear1;
      tau_total[10] += tau_gear1;
      tau_total[9] += tau_gear2;
      tau_total[13] += tau_gear2;

      end_effector_->setGeneralizedForce(tau_total);
      world_->integrate();

      // Thread Pitch Logic
      double angle = bolt_->getGeneralizedCoordinate()[1]; 
      double omega = bolt_->getGeneralizedVelocity()[1]; 
      double translation = angle / (2.0 * M_PI) * thread_pitch_; 
      double velocity = omega / (2.0 * M_PI) * thread_pitch_; 

      Eigen::VectorXd b_gc = bolt_->getGeneralizedCoordinate().e();
      Eigen::VectorXd b_gv = bolt_->getGeneralizedVelocity().e();
      b_gc[0] = translation; 
      b_gv[0] = velocity; 
      bolt_->setState(b_gc, b_gv);
    }

    updateObservation();
    
    double reward = bolt_->getGeneralizedVelocity()[1]; 
    if(std::isnan(reward)) reward = 0.0;

    return reward;
  }

  /// [OBSERVE] - 수정된 부분 (.e() 추가)
  void updateObservation() {
    obDouble_.head(gcDim_) = end_effector_->getGeneralizedCoordinate().e();
    obDouble_.segment(gcDim_, gvDim_) = end_effector_->getGeneralizedVelocity().e();
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);
    if (end_effector_->getGeneralizedCoordinate()[2] < 0.05) { 
        return true;
    }
    return false;
  }
  
  void setCurriculumFactor(double factor) {
      curriculumFactor_ = factor;
  }

 private:
  raisim::ArticulatedSystem* bolt_;
  raisim::ArticulatedSystem* wrench_;
  raisim::ArticulatedSystem* end_effector_;
  
  double simulation_dt_ = 0.0001;
  double control_dt_ = 0.01;
  double thread_pitch_ = 0.002;
  double kp_gear_ = 10000.0;
  double kd_gear_ = 1.0;
  double terminalRewardCoeff_ = -10.0;
  
  std::string resourceDir_;
  bool visualizable_ = false;
  int gcDim_, gvDim_, nJoints_;
  
  double curriculumFactor_ = 1.0;

  Eigen::VectorXd obDouble_;
  Eigen::VectorXd actionMean_;
  Eigen::VectorXd actionStd_;

  // [4] 새로 추가된 변수 선언
  double ee_left_init_ = 0.0;
  double ee_right_init_ = 0.0;
  double ee_left_force_ = 0.0;
  double ee_right_force_ = 0.0;
};

} // namespace raisim