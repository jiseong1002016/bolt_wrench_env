#include "include/JointCommandGenerator.hpp"

JointCommandGenerator::JointCommandGenerator(const Eigen::Vector3d& base_pos,
                                             const Eigen::Quaterniond& base_quat,
                                             double gc7_init,
                                             double gc14_init,
                                             JointCommandMode mode,
                                             double grasp_gc14_start,
                                             double grasp_gc14_duration,
                                             double grasp_approach_duration)
    : base_pos_(base_pos),
      base_quat_(base_quat.normalized()),
      gc7_init_(gc7_init),
      gc14_init_(gc14_init),
      command_mode_(mode),
      grasp_gc14_start_(grasp_gc14_start),
      grasp_gc14_duration_(grasp_gc14_duration),
      grasp_approach_duration_(grasp_approach_duration) {
  // GC 7 Range: [-0.960, 0.0] -> Mid: -0.48, Amp: 0.48
  gc7_mid_ = 0.5 * (gc7_min_ + gc7_max_);
  gc7_amp_ = 0.5 * (gc7_max_ - gc7_min_);

  // GC 14 Range: [0.00, 0.96] -> Mid: 0.48, Amp: 0.48
  gc14_mid_ = 0.5 * (gc14_min_ + gc14_max_);
  gc14_amp_ = 0.5 * (gc14_max_ - gc14_min_);
}

void JointCommandGenerator::update(JointCommand& command,
                                   double sec,
                                   const GraspContext* grasp_context) const {
  command.base_pos = base_pos_;
  command.base_quat = base_quat_;

  // 기본값은 초기 위치로 설정 (안전을 위해)
  command.q_des[0] = gc7_init_;
  command.q_des[1] = gc14_init_;
  // Leave v_des untouched so velocity command is not overridden here.

  if (sec < 0.0) {
    return;  // 0초 이전이면 초기 자세 유지
  }

  if (command_mode_ == JointCommandMode::kGrasp) {
    if (sec < 10.0) {
      // Hold initial pose for the first 5 seconds.
      command.v_des[0] = 0.0;
      command.v_des[1] = 0.0;
      return;
    }
    const double approach_duration = std::max(grasp_approach_duration_, 1e-6);
    const double approach_ratio = std::min(1.0, sec / approach_duration);
    const double s = approach_ratio * approach_ratio * approach_ratio *
                     (10.0 + approach_ratio * (-15.0 + 6.0 * approach_ratio));

    command.q_des[0] = 0.0;
    command.v_des[0] = 0.0;

    const double duration = std::max(grasp_gc14_duration_, 1e-6);
    const double ratio = std::min(1.0, sec / duration);
    command.q_des[1] = grasp_gc14_start_ * (1.0 - ratio);
    command.v_des[1] = (ratio >= 1.0) ? 0.0 : (-grasp_gc14_start_ / duration);

    if (grasp_context) {
      const Eigen::Quaterniond target_quat =
          grasp_context->target_base_quat.normalized();
      const Eigen::Vector3d target_base_pos =
          grasp_context->target_center_world -
          target_quat * grasp_context->grasp_center_offset_base;

      command.base_pos = base_pos_ + s * (target_base_pos - base_pos_);
      command.base_quat = base_quat_.slerp(s, target_quat).normalized();
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
