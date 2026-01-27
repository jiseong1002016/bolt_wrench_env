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

enum class JointCommandMode {
  kSine,
  kGrasp,
};

class JointCommandGenerator {
 public:
  // [변경] Environment.hpp에서 넘겨준 초기값을 저장합니다.
  JointCommandGenerator(const Eigen::Vector3d& base_pos,
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
                        double grasp_rotate_radius);

  void update(JointCommand& command, double sec) const;

  static constexpr int kCommandDim = 11;

  static void FromVector(const Eigen::Ref<const Eigen::VectorXd>& vec,
                         JointCommand& command);

  static void ToVector(const JointCommand& command,
                       Eigen::Ref<Eigen::VectorXd> vec);

 private:
  Eigen::Vector3d base_pos_;
  Eigen::Quaterniond base_quat_;

  double gc7_init_;
  double gc14_init_;
  JointCommandMode command_mode_;
  double grasp_gc14_start_;
  double grasp_gc14_target_;
  double grasp_gc14_duration_;
  double grasp_approach_duration_;
  double grasp_move_duration_;
  double grasp_rotate_duration_;
  double grasp_rotate_radius_;

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
