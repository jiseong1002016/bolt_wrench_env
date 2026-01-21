#ifndef BOLT_WRENCH_INCLUDE_JOINT_COMMAND_GENERATOR_HPP_
#define BOLT_WRENCH_INCLUDE_JOINT_COMMAND_GENERATOR_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

struct JointCommand {
  Eigen::Vector3d base_pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond base_quat = Eigen::Quaterniond::Identity();
  Eigen::Vector2d q_des = Eigen::Vector2d::Zero();
  Eigen::Vector2d v_des = Eigen::Vector2d::Zero();
};

class JointCommandGenerator {
 public:
  // [변경] Environment.hpp에서 넘겨준 초기값을 저장합니다.
  JointCommandGenerator(const Eigen::Vector3d& base_pos,
                        const Eigen::Quaterniond& base_quat,
                        double gc7_init,
                        double gc14_init)
      : base_pos_(base_pos),
        base_quat_(base_quat.normalized()),
        gc7_init_(gc7_init),
        gc14_init_(gc14_init) {
    
    // GC 7 Range: [-0.960, 0.0] -> Mid: -0.48, Amp: 0.48
    gc7_mid_ = 0.5 * (gc7_min_ + gc7_max_);
    gc7_amp_ = 0.5 * (gc7_max_ - gc7_min_);

    // GC 14 Range: [0.00, 0.96] -> Mid: 0.48, Amp: 0.48
    gc14_mid_ = 0.5 * (gc14_min_ + gc14_max_);
    gc14_amp_ = 0.5 * (gc14_max_ - gc14_min_);
  }

  void update(JointCommand& command, double sec) const {
    command.base_pos = base_pos_;
    command.base_quat = base_quat_;

    // 기본값은 초기 위치로 설정 (안전을 위해)
    command.q_des[0] = gc7_init_;
    command.q_des[1] = gc14_init_;
    // Leave v_des untouched so velocity command is not overridden here.

    if (sec < 0.0) {
      return; // 0초 이전이면 초기 자세 유지
    }

    // [궤적 생성]
    // 주파수 (Hz): 필요에 따라 조절하세요 (예: 0.5Hz = 2초 주기)
    const double omega = 2.0 * M_PI * frequency_hz_;
    
    // GC7: 0.0에서 시작하여 -0.96까지 내려갔다 옴
    // -0.48 + 0.48 * cos(0) = 0.0 (Start)
    command.q_des[0] = gc7_mid_ + gc7_amp_ * std::cos(omega * sec);

    // GC14: 0.0에서 시작하여 0.96까지 올라갔다 옴
    // 0.48 - 0.48 * cos(0) = 0.0 (Start)
    command.q_des[1] = gc14_mid_ - gc14_amp_ * std::cos(omega * sec);

    // (선택 사항) 속도 피드포워드가 필요하면 주석 해제 (PD 제어 성능 향상)
    // 미분: d/dt (A * cos(wt)) = -A * w * sin(wt)
    command.v_des[0] = -gc7_amp_ * omega * std::sin(omega * sec);
    command.v_des[1] = gc14_amp_ * omega * std::sin(omega * sec);
  }

  static constexpr int kCommandDim = 11;

  static void FromVector(const Eigen::Ref<const Eigen::VectorXd>& vec,
                         JointCommand& command) {
    if (vec.size() < kCommandDim) {
      command = JointCommand();
      return;
    }
    command.base_pos = vec.segment<3>(0);
    command.base_quat = Eigen::Quaterniond(vec[3], vec[4], vec[5], vec[6]);
    command.base_quat.normalize();
    command.q_des = vec.segment<2>(7);
    command.v_des = vec.segment<2>(9);
  }

  static void ToVector(const JointCommand& command,
                       Eigen::Ref<Eigen::VectorXd> vec) {
    if (vec.size() != kCommandDim) {
      vec.resize(kCommandDim);
    }
    vec[0] = command.base_pos.x();
    vec[1] = command.base_pos.y();
    vec[2] = command.base_pos.z();
    vec[3] = command.base_quat.w();
    vec[4] = command.base_quat.x();
    vec[5] = command.base_quat.y();
    vec[6] = command.base_quat.z();
    vec[7] = command.q_des[0];
    vec[8] = command.q_des[1];
    vec[9] = command.v_des[0];
    vec[10] = command.v_des[1];
  }

 private:
  Eigen::Vector3d base_pos_;
  Eigen::Quaterniond base_quat_;

  double gc7_init_;
  double gc14_init_;

  // Constants for Range
  const double gc7_min_ = -0.960;
  const double gc7_max_ = 0.0;
  const double gc14_min_ = 0.0;
  const double gc14_max_ = 0.96;

  double gc7_mid_;
  double gc7_amp_;
  double gc14_mid_;
  double gc14_amp_;

  double frequency_hz_ = 1; // 속도 (0.1Hz)
};

#endif  // BOLT_WRENCH_INCLUDE_JOINT_COMMAND_GENERATOR_HPP_
