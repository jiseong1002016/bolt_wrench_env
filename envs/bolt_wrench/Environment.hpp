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

    /// [해결책] 부모의 protected 멤버인 server_를 직접 제어
    if (visualizable_) {
      // 부모 클래스에 선언된 server_ 객체를 여기서 생성합니다.
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
    }

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
    // ee_gc[7] = ee_left_init_;  
    // ee_gc[11] = ee_right_init_;
    
    end_effector_->setState(ee_gc, ee_gv);

    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    server_->focusOn(end_effector_);
    
    // [수정 1] Action은 루프 밖에서 변환해 둡니다 (Base Torque)
    Eigen::VectorXd action_command = action.cast<double>();
    // Eigen::VectorXd action_command = Eigen::VectorXd::Zero(end_effector_->getDOF()); // <-- 강제 0 할당
    
    if(action_command.size() != end_effector_->getDOF()) {
        action_command = Eigen::VectorXd::Zero(end_effector_->getDOF());
    }

    /// Simulation Loop
    for (int i = 0; i < int(control_dt_ / simulation_dt_); i++) {
      // [수정 2] 매 스텝마다 action_command를 복사하여 '현재 스텝용 토크'를 만듭니다.
      // 이렇게 해야 이전 스텝의 tau_gear가 누적되지 않습니다.
      Eigen::VectorXd tau_step = action_command;

      const auto& gc = end_effector_->getGeneralizedCoordinate();
      const auto& gv = end_effector_->getGeneralizedVelocity();

      // Gear Coupling Logic
      double error_p1 = gc[7] + gc[11];
      double error_v1 = gv[6] + gv[10];
      // [복구 추천] 누적 버그가 사라지면 kp를 다시 10000.0으로 올려도 됩니다.
      double tau_gear1 = -kp_gear_ * error_p1 - kd_gear_ * error_v1;

      double error_p2 = gc[10] + gc[14];
      double error_v2 = gv[9] + gv[13];
      double tau_gear2 = -kp_gear_ * error_p2 - kd_gear_ * error_v2;

      // [수정 3] 현재 스텝용 변수(tau_step)에 더합니다.
      tau_step[6] += tau_gear1;
      tau_step[10] += tau_gear1;
      tau_step[9] += tau_gear2;
      tau_step[13] += tau_gear2;

      // [수정 4] 최종 토크 인가
      end_effector_->setGeneralizedForce(tau_step);
      world_->integrate();

      // Thread Pitch Logic (Bolt)
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
  
  double simulation_dt_ = 0.0001; // diverse at 0.0001
  double control_dt_ = 0.01;
  double thread_pitch_ = 0.002;
  double kp_gear_ = 10000.0; // diverse at 10000.0
  double kd_gear_ = 1.0;  // diverse at 1.0
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