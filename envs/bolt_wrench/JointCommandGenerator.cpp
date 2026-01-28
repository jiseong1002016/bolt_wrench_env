#include "include/JointCommandGenerator.hpp"
#include <iostream>

JointCommandGenerator::JointCommandGenerator(const Eigen::Vector3d& base_pos,
                                             const Eigen::Quaterniond& base_quat,
                                             double gc7_init,
                                             double gc14_init,
                                             JointCommandMode mode,
                                             double grasp_gc14_start,
                                             double grasp_gc14_target,
                                             double grasp_gc14_duration,
                                             double grasp_approach_duration,
                                             double grasp_move_duration,
                                             double grasp_rotate_duration,
                                             double grasp_rotate_radius,
                                             double gf_6,
                                             double gf_13)
    : base_pos_(base_pos),
      base_quat_(base_quat.normalized()),
      gc7_init_(gc7_init),
      gc14_init_(gc14_init),
      command_mode_(mode),
      grasp_gc14_start_(grasp_gc14_start),
      grasp_gc14_target_(grasp_gc14_target),
      grasp_gc14_duration_(grasp_gc14_duration),
      grasp_approach_duration_(grasp_approach_duration),
      grasp_move_duration_(grasp_move_duration),
      grasp_rotate_duration_(grasp_rotate_duration),
      grasp_rotate_radius_(grasp_rotate_radius),
      gf_6_(gf_6),
      gf_13_(gf_13) {
  // GC 7 Range: [-0.960, 0.0] -> Mid: -0.48, Amp: 0.48
  gc7_mid_ = 0.5 * (gc7_min_ + gc7_max_);
  gc7_amp_ = 0.5 * (gc7_max_ - gc7_min_);

  // GC 14 Range: [0.00, 0.96] -> Mid: 0.48, Amp: 0.48
  gc14_mid_ = 0.5 * (gc14_min_ + gc14_max_);
  gc14_amp_ = 0.5 * (gc14_max_ - gc14_min_);
}

void JointCommandGenerator::update(JointCommand& command,
                                   double sec) const {
  command.base_pos = base_pos_;
  command.base_quat = base_quat_;

  // 기본값은 초기 위치로 설정 (안전을 위해)
  command.q_des[0] = gc7_init_;
  command.q_des[1] = gc14_init_;
  command.tau_ff.setZero();
  // Leave v_des untouched so velocity command is not overridden here.

  if (sec < 0.0) {
    command.base_pos = base_pos_;
    command.base_quat = base_quat_;
    command.q_des[0] = gc7_init_;
    command.q_des[1] = gc14_init_;
    return;  // 0초 이전이면 초기 자세 유지
  }

  if (command_mode_ == JointCommandMode::kGrasp) {
    static double last_print_sec = -1.0;
    const bool should_print =
        (last_print_sec < 0.0) || (sec - last_print_sec >= 0.5);
    if (should_print) {
      last_print_sec = sec;
    }
    const double start = grasp_gc14_start_;
    const double target = grasp_gc14_target_;
    const double t0 = grasp_approach_duration_;
    const double t1 = t0 + grasp_gc14_duration_;
    const double t2 = t1 + grasp_move_duration_;
    const double t3 = t2 + grasp_rotate_duration_;
    // Apply feedforward torques only during moving and rotating phases.
    if (sec >= t1 && sec < t3) {
      command.tau_ff[0] = gf_6_;
      command.tau_ff[1] = gf_13_;
    }
    // Hold the base at the reset-aligned pose until the move phase.
    const Eigen::Vector3d base_hold_pos(0.0, -1.26259, 1.034);
    const Eigen::Quaterniond base_hold_quat(1.0, 0.0, 0.0, 0.0);
    const Eigen::Vector3d wrench_origin(0.0, 0.0, 0.034);
    if (sec >= 0.0 && sec < grasp_approach_duration_) {
      command.q_des[1] = start;
      command.base_pos = base_hold_pos;
      command.base_quat = base_hold_quat;
      if (should_print) {
        std::cout << "Approaching for Grasping...\n" << std::endl;
      }
    } else if (sec >= grasp_approach_duration_ &&
               sec < grasp_approach_duration_ + grasp_gc14_duration_) {
      const double denom = std::max(grasp_gc14_duration_, 1e-6);
      const double t = (sec - grasp_approach_duration_) / denom;
      command.q_des[1] = start + t * (target - start);
      command.base_pos = base_hold_pos;
      command.base_quat = base_hold_quat;
      if (should_print) {
        std::cout << "Grasping... GC14: " << command.q_des[1] << "\n" << std::endl;
      }
    } else if (sec >= t1 && sec < t2) {
      command.q_des[1] = target;
      const double denom = std::max(grasp_move_duration_, 1e-6);
      const double t = (sec - t1) / denom;
      command.base_pos = base_hold_pos + t * (wrench_origin - base_hold_pos);
      if (should_print) {
        std::cout << "Moving to Wrench Position: "
                  << command.base_pos.transpose() << "\n" << std::endl;
      }
    } else if (sec >= t2 && sec < t3) {
      command.q_des[1] = target;
      const double denom = std::max(grasp_rotate_duration_, 1e-6);
      const double t = (sec - t2) / denom;
      const double theta = 2.0 * M_PI * t;
      command.base_pos = wrench_origin +
                         Eigen::Vector3d(grasp_rotate_radius_ * std::cos(theta),
                                         grasp_rotate_radius_ * std::sin(theta),
                                         0.0);
      if (should_print) {
        std::cout << "Rotating around Wrench: "
                  << command.base_pos.transpose() << "\n" << std::endl;
      }
    } else if (sec >= t3) {
      command.q_des[1] = target;
      command.base_pos = wrench_origin +
                         Eigen::Vector3d(grasp_rotate_radius_, 0.0, 0.0);
      if (should_print) {
        std::cout << "Grasping Sequence Completed. Holding Position.\n" << std::endl;
      }
    }

    return;
  }

  if (command_mode_ == JointCommandMode::kSine) {
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
}

void JointCommandGenerator::FromVector(
    const Eigen::Ref<const Eigen::VectorXd>& vec,
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

void JointCommandGenerator::ToVector(const JointCommand& command,
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
