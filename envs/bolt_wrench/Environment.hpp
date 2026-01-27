#pragma once

#include <cmath>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <iostream>
#if defined(__linux__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>
#include "../../RaisimGymEnv.hpp"
#include "UtilityFunctions.hpp"
#include "include/JointCommandGenerator.hpp"

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
    grasp_gc14_start_ = ee_right_init_;

    if (&cfg["use_action_command"]) {
      use_action_command_ = cfg["use_action_command"].template As<bool>();
    }
    if (&cfg["command_mode"]) {
      std::string mode = cfg["command_mode"].template As<std::string>();
      std::transform(mode.begin(),
                     mode.end(),
                     mode.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (mode == "grasp") {
        command_mode_ = JointCommandMode::kGrasp;
      } else {
        command_mode_ = JointCommandMode::kSine;
      }
    }
    if (&cfg["active_kp"]) {
      active_kp_ = cfg["active_kp"].template As<double>();
    }
    if (&cfg["active_kd"]) {
      active_kd_ = cfg["active_kd"].template As<double>();
    }
    bool base_kp_set = false;
    bool base_kd_set = false;
    if (&cfg["base_kp"]) {
      base_kp_ = cfg["base_kp"].template As<double>();
      base_kp_set = true;
    }
    if (&cfg["base_kd"]) {
      base_kd_ = cfg["base_kd"].template As<double>();
      base_kd_set = true;
    }
    if (!base_kp_set) {
      base_kp_ = active_kp_;
    }
    if (!base_kd_set) {
      base_kd_ = active_kd_;
    }
    if (&cfg["kp_gear"]) {
      kp_gear_ = cfg["kp_gear"].template As<double>();
    }
    if (&cfg["kd_gear"]) {
      kd_gear_ = cfg["kd_gear"].template As<double>();
    }
    if (&cfg["grasp_gc14_start"]) {
      grasp_gc14_start_ = cfg["grasp_gc14_start"].template As<double>();
    }
    if (&cfg["grasp_gc14_target"]) {
      grasp_gc14_target_ = cfg["grasp_gc14_target"].template As<double>();
    }
    if (&cfg["grasp_gc14_duration"]) {
      grasp_gc14_duration_ = cfg["grasp_gc14_duration"].template As<double>();
    }
    if (&cfg["grasp_approach_duration"]) {
      grasp_approach_duration_ = cfg["grasp_approach_duration"].template As<double>();
    }
    if (&cfg["grasp_move_duration"]) {
      grasp_move_duration_ = cfg["grasp_move_duration"].template As<double>();
    }
    if (&cfg["grasp_rotate_duration"]) {
      grasp_rotate_duration_ = cfg["grasp_rotate_duration"].template As<double>();
    }
    if (&cfg["grasp_rotate_radius"]) {
      grasp_rotate_radius_ = cfg["grasp_rotate_radius"].template As<double>();
    }
    if (&cfg["gravity_compensation"]) {
      gravity_compensation_ = cfg["gravity_compensation"].template As<bool>();
    }
    // Legacy position-force gains (kept for backward compatibility).
    if (&cfg["base_force_kp"]) {
      base_force_pos_kp_ = cfg["base_force_kp"].template As<double>();
    }
    if (&cfg["base_force_kd"]) {
      base_force_pos_kd_ = cfg["base_force_kd"].template As<double>();
    }
    // Position-force gains (preferred).
    if (&cfg["base_force_pos_kp"]) {
      base_force_pos_kp_ = cfg["base_force_pos_kp"].template As<double>();
    }
    if (&cfg["base_force_pos_kd"]) {
      base_force_pos_kd_ = cfg["base_force_pos_kd"].template As<double>();
    }
    if (&cfg["base_force_pos_ki"]) {
      base_force_pos_ki_ = cfg["base_force_pos_ki"].template As<double>();
    }
    if (&cfg["base_force_pos_i_max"]) {
      base_force_pos_i_max_ = cfg["base_force_pos_i_max"].template As<double>();
    }
    // Rotation-torque gains.
    if (&cfg["base_force_rot_kp"]) {
      base_force_rot_kp_ = cfg["base_force_rot_kp"].template As<double>();
    }
    if (&cfg["base_force_rot_kd"]) {
      base_force_rot_kd_ = cfg["base_force_rot_kd"].template As<double>();
    }
    if (&cfg["base_force_rot_ki"]) {
      base_force_rot_ki_ = cfg["base_force_rot_ki"].template As<double>();
    }
    if (&cfg["base_force_rot_i_max"]) {
      base_force_rot_i_max_ = cfg["base_force_rot_i_max"].template As<double>();
    }
    if (&cfg["base_force_max"]) {
      base_force_max_ = cfg["base_force_max"].template As<double>();
    }
    if (&cfg["base_torque_max"]) {
      base_torque_max_ = cfg["base_torque_max"].template As<double>();
    }

    /// 2. 월드 초기화
    world_ = std::make_unique<raisim::World>();
    world_->setTimeStep(simulation_dt_);
    world_->setGravity({0.0, 0.0, -9.81});
    world_->addGround();

    /// [해결책] 부모의 protected 멤버인 server_를 직접 제어
    if (visualizable_) {
      // 부모 클래스에 선언된 server_ 객체를 여기서 생성합니다.
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      if (!launchServerWithFallback()) {
        server_.reset();
        visualizable_ = false;
      }
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
    const auto& wrench_masses = wrench_->getMass();
    wrench_body_masses_.assign(wrench_masses.begin(), wrench_masses.end());

    end_effector_ = world_->addArticulatedSystem(
        resourceDir_ + "/end-effector/urdf/end-effector.urdf",
        "", {}, raisim::COLLISION(3), -1);
    end_effector_->setName("end_effector");
    const auto& ee_masses = end_effector_->getMass();
    ee_body_masses_.assign(ee_masses.begin(), ee_masses.end());
    base_body_idx_ = static_cast<int>(end_effector_->getBodyIdx("ee_base"));
    if (base_body_idx_ < 0 ||
        static_cast<size_t>(base_body_idx_) >= ee_body_masses_.size()) {
      base_body_idx_ = 0;
    }
    ee_frame_idx_ = end_effector_->getFrameIdxByLinkName(ee_frame_name_);
    if (ee_frame_idx_ >= end_effector_->getFrames().size()) {
      ee_frame_idx_ = 0;
    }

    ee_4_left_frame_idx_ =
        end_effector_->getFrameIdxByLinkName(ee_4_left_frame_name_);
    if (ee_4_left_frame_idx_ >= end_effector_->getFrames().size()) {
      ee_4_left_frame_idx_ = 0;
    }
    ee_4_right_frame_idx_ =
        end_effector_->getFrameIdxByLinkName(ee_4_right_frame_name_);
    if (ee_4_right_frame_idx_ >= end_effector_->getFrames().size()) {
      ee_4_right_frame_idx_ = 0;
    }

    wrench_frame_idx_ = wrench_->getFrameIdxByLinkName(wrench_frame_name_);
    if (wrench_frame_idx_ >= wrench_->getFrames().size()) {
      wrench_frame_idx_ = 0;
    }

    bolt_frame_idx_ = bolt_->getFrameIdxByLinkName(bolt_frame_name_);
    if (bolt_frame_idx_ >= bolt_->getFrames().size()) {
      bolt_frame_idx_ = 0;
    }

    /// 4. 차원 설정
    gcDim_ = end_effector_->getGeneralizedCoordinateDim();
    gvDim_ = end_effector_->getDOF();
    const int ee_obs_dim = 3 + 9;
    const int wrench_rel_dim = 3 + 4;
    const int bolt_rel_dim = 3;
    obDim_ = gcDim_ + gvDim_ + ee_obs_dim + wrench_rel_dim + bolt_rel_dim;
    actionDim_ = 12;
    
    obDouble_.setZero(obDim_);
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    pd_target_pos_.setZero(gcDim_);
    pd_target_vel_.setZero(gvDim_);

    end_effector_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    Eigen::VectorXd jointPgain = Eigen::VectorXd::Zero(gvDim_);
    Eigen::VectorXd jointDgain = Eigen::VectorXd::Zero(gvDim_);
    const int base_gc_dim = (gcDim_ == gvDim_ + 1) ? 7 : 0;
    const int joint_count = gcDim_ - base_gc_dim;
    const int base_vel_dim = gvDim_ - joint_count;
    {
      static bool printed_dims_once = false;
      if (!printed_dims_once) {
        printed_dims_once = true;
        std::cout << "[dims] gcDim=" << gcDim_
                  << " gvDim=" << gvDim_
                  << " base_gc_dim=" << base_gc_dim
                  << " base_vel_dim=" << base_vel_dim
                  << " joint_count=" << joint_count
                  << " floating_base=" << (base_gc_dim == 7 ? "true" : "false")
                  << std::endl;
      }
    }
    if (base_gc_dim == 7 && base_vel_dim > 0) {
      jointPgain.head(base_vel_dim).setConstant(base_kp_);
      jointDgain.head(base_vel_dim).setConstant(base_kd_);
    }
    if (joint_count >= 8) {
      const int idx_gv_7 = base_vel_dim + 0;
      const int idx_gv_14 = base_vel_dim + 7;
      if (idx_gv_14 < gvDim_) {
        jointPgain[idx_gv_7] = active_kp_;
        jointPgain[idx_gv_14] = active_kp_;
        jointDgain[idx_gv_7] = active_kd_;
        jointDgain[idx_gv_14] = active_kd_;
      }
    }
    end_effector_->setPdGains(jointPgain, jointDgain);
    end_effector_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    if (joint_count > 0) {
      const Eigen::Vector3d base_pos_init(0.0, 0.0, 0.15);
      const Eigen::Quaterniond base_quat_init(1.0, 0.0, 0.0, 0.0);
      joint_command_generator_ = std::make_unique<JointCommandGenerator>(
          base_pos_init,
          base_quat_init,
          ee_left_init_,
          ee_right_init_,
          command_mode_,
          grasp_gc14_start_,
          grasp_gc14_target_,
          grasp_gc14_duration_,
          grasp_approach_duration_,
          grasp_move_duration_,
          grasp_rotate_duration_,
          grasp_rotate_radius_);
      joint_command_vector_.setZero(JointCommandGenerator::kCommandDim);
      if (server_) {
        joint_command_vis_.setZero(static_cast<size_t>(joint_command_vector_.size()));
        std::vector<std::string> command_names = {
            "x", "y", "z", "qw", "qx", "qy", "qz", "gc_7", "gc_14", "gv_7", "gv_14"};
        joint_command_graph_ = server_->addTimeSeriesGraph(
            "joint_command", command_names, "time", "position");
      }
    }

    /// 5. 초기 관측값 업데이트
    updateObservation();
  }

  void init() final { }

  void reset() final {
    base_pos_error_integral_.setZero();
    base_rot_error_integral_.setZero();
    // 1. Bolt 초기화
    Eigen::VectorXd bolt_gc = Eigen::VectorXd::Zero(bolt_->getGeneralizedCoordinateDim());
    Eigen::VectorXd bolt_gv = Eigen::VectorXd::Zero(bolt_->getDOF());
    bolt_->setState(bolt_gc, bolt_gv);

    // 2. Wrench 초기화
    Eigen::VectorXd wrench_gc = Eigen::VectorXd::Zero(wrench_->getGeneralizedCoordinateDim());
    Eigen::VectorXd wrench_gv = Eigen::VectorXd::Zero(wrench_->getDOF());
    // wrench_gc.head(7) << 0.0, 0.0, 0.034, 1.0, 0.0, 0.0, 0.0;
    wrench_gc.head(7) << 0.0, -1.0, 1.034, 1.0, 0.0, 0.0, 0.0;
    wrench_->setState(wrench_gc, wrench_gv);

    // 3. End-effector 초기화
    Eigen::VectorXd ee_gc = Eigen::VectorXd::Zero(end_effector_->getGeneralizedCoordinateDim());
    Eigen::VectorXd ee_gv = Eigen::VectorXd::Zero(end_effector_->getDOF());
    ee_gc.head(7) << 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0; 

    const int base_gc_dim = (gcDim_ == gvDim_ + 1) ? 7 : 0;
    const int joint_count = gcDim_ - base_gc_dim;
    if (joint_count >= 1 && ee_gc.size() > base_gc_dim + 0) {
      ee_gc[base_gc_dim + 0] = ee_left_init_;
    }
    if (joint_count >= 8 && ee_gc.size() > base_gc_dim + 7) {
      ee_gc[base_gc_dim + 7] = ee_right_init_;
    }
    end_effector_->setState(ee_gc, ee_gv);

    if (base_gc_dim == 7) {  // end-effector posision setting based on wrench // gc < 7 initialization
      raisim::Vec<3> wrench_pos_W;
      raisim::Mat<3, 3> wrench_rot_W;
      wrench_->getFramePosition(wrench_frame_idx_, wrench_pos_W);
      wrench_->getFrameOrientation(wrench_frame_idx_, wrench_rot_W);
      const Eigen::Vector3d wrench_pos = wrench_pos_W.e();
      std::cout << "Wrench Position: " << wrench_pos.transpose() << std::endl;
      const Eigen::Matrix3d R_wrench = wrench_rot_W.e();
      std::cout << "Wrench Rotation:" << R_wrench << std::endl;
      const Eigen::Vector3d target_center_world =
          wrench_pos + R_wrench * Eigen::Vector3d(0.0, -0.1125, 0.0);  // grasp_target_offset_wrench_;
      std::cout << "Target Center World: " << target_center_world.transpose() << std::endl;

      raisim::Vec<3> ee_left_pos_W;
      raisim::Mat<3, 3> ee_left_rot_W;
      end_effector_->getFramePosition(ee_4_left_frame_idx_, ee_left_pos_W);
      end_effector_->getFrameOrientation(ee_4_left_frame_idx_, ee_left_rot_W);
      const Eigen::Vector3d left_world =
          ee_left_pos_W.e() + ee_left_rot_W.e() * grasp_point_left_local_;

      raisim::Vec<3> ee_right_pos_W;
      raisim::Mat<3, 3> ee_right_rot_W;
      end_effector_->getFramePosition(ee_4_right_frame_idx_, ee_right_pos_W);
      end_effector_->getFrameOrientation(ee_4_right_frame_idx_, ee_right_rot_W);
      const Eigen::Vector3d right_world =
          ee_right_pos_W.e() + ee_right_rot_W.e() * grasp_point_right_local_;

      const Eigen::Vector3d center_world = 0.5 * (left_world + right_world);
      const Eigen::Vector3d base_pos(ee_gc[0], ee_gc[1], ee_gc[2]);
      const Eigen::Quaterniond base_quat(ee_gc[3], ee_gc[4], ee_gc[5], ee_gc[6]);
      const Eigen::Vector3d grasp_center_offset_base =
          base_quat.toRotationMatrix().transpose() * (center_world - base_pos);
      const Eigen::Vector3d new_base_pos =
          target_center_world - base_quat * grasp_center_offset_base;
      std::cout << "New Base Position: " << new_base_pos.transpose() << std::endl;
      ee_gc[0] = new_base_pos.x();
      ee_gc[1] = new_base_pos.y();
      ee_gc[2] = new_base_pos.z();
      
      end_effector_->setState(ee_gc, ee_gv);
    }
    SetEndEffectorGeneralizedCoordinates(end_effector_, ee_left_init_, grasp_gc14_start_);
    std::cout << "[reset] ee_gc(now)="
              << end_effector_->getGeneralizedCoordinate().e().transpose()
              << std::endl;
    // updateWrenchPoseFromEndEffector();

    updateObservation();
  }

  void resetToDemoState(const Eigen::VectorXd& bolt_gc,
                        const Eigen::VectorXd& bolt_gv,
                        const Eigen::VectorXd& wrench_gc,
                        const Eigen::VectorXd& wrench_gv,
                        const Eigen::VectorXd& robot_gc,
                        const Eigen::VectorXd& robot_gv) {
    bolt_->setState(bolt_gc, bolt_gv);
    wrench_->setState(wrench_gc, wrench_gv);
    end_effector_->setState(robot_gc, robot_gv);
    updateObservation();
  }   // used in runner.py for setting demo states

  float step(const Eigen::Ref<EigenVec>& action) final {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    if (server_) {
      server_->focusOn(end_effector_);
      // server_->focusOn(wrench_);
    }

    // Timing scaffold for command generation (currently unused; kept for quick profiling).
    // const auto command_start = Clock::now();
    Eigen::VectorXd action_command = action.cast<double>();
    if (action_command.size() != actionDim_) {
      action_command = Eigen::VectorXd::Zero(actionDim_);
    }

    const int sim_steps = static_cast<int>(control_dt_ / simulation_dt_);
    const int base_gc_dim = (gcDim_ == gvDim_ + 1) ? 7 : 0;
    const int joint_count = gcDim_ - base_gc_dim;
    const double now = world_->getWorldTime();
    const double base_force_enable_time =
        grasp_approach_duration_ + grasp_gc14_duration_;
    const bool base_force_enabled = (now >= base_force_enable_time);
    const bool can_use_action_command =
        use_action_command_ &&
        action_command.size() >= JointCommandGenerator::kCommandDim;
    const bool can_use_generator = static_cast<bool>(joint_command_generator_);
    const bool has_command = can_use_action_command || can_use_generator;

    if (can_use_action_command) {  // 제어기 쓸 때 @ runner.py
      JointCommandGenerator::FromVector(
          action_command.head(JointCommandGenerator::kCommandDim), joint_command_);
    } else if (can_use_generator) {   // 데모 만들 때 @ demo_generator.py
      joint_command_generator_->update(joint_command_, now);
    }
    // const auto command_end = Clock::now();
    // [role] command_start/command_end were used to profile command generation latency.
    // std::cout << "[step] command_gen_ms="
    //           << Ms(command_end - command_start).count() << "\n";

    // if (has_command) {  
    //   JointCommandGenerator::ToVector(joint_command_, joint_command_vector_);
    //   std::cout << "[step] t=" << now << " joint_command="
    //             << joint_command_vector_.transpose() << std::endl;
    //   if (joint_command_graph_ &&
    //       joint_command_vis_.size() == joint_command_vector_.size()) {
    //     for (int i = 0; i < joint_command_vector_.size(); ++i) {
    //       joint_command_vis_[static_cast<size_t>(i)] = joint_command_vector_[i];
    //     }
    //     joint_command_graph_->addDataPoints(now, joint_command_vis_);
    //   }
    // }

    const int base_vel_dim = gvDim_ - joint_count;
    const int idx_gc_7 = base_gc_dim + 0;
    const int idx_gc_10 = base_gc_dim + 3;
    const int idx_gc_11 = base_gc_dim + 4;
    const int idx_gc_14 = base_gc_dim + 7;
    const int idx_gv_7 = base_vel_dim + 0;
    const int idx_gv_10 = base_vel_dim + 3;
    const int idx_gv_11 = base_vel_dim + 4;
    const int idx_gv_14 = base_vel_dim + 7;
    Eigen::VectorXd tau_total = Eigen::VectorXd::Zero(gvDim_);
    // Timing scaffold for the simulation loop (currently unused; kept for quick profiling).
    // const auto sim_loop_start = Clock::now();
    double integrate_ms = 0.0;
    double bolt_logic_ms = 0.0;
    for (int i = 0; i < sim_steps; i++) {  // step simulation in control_dt
      tau_total.setZero();

      const auto& gc = end_effector_->getGeneralizedCoordinate();
      const auto& gv = end_effector_->getGeneralizedVelocity();
      if (has_command) {
        pd_target_pos_ = gc.e();
        pd_target_vel_ = gv.e();
        const bool has_base = (base_gc_dim == 7 && gc.size() >= 7);
        const bool has_coupled_joints =
            (joint_count >= 8 && gc.size() > idx_gc_14 && gv.size() > idx_gv_14);

        if (has_base) {
          // Debug print once per second instead of every step.
          if (last_debug_print_time_ < 0.0 || now - last_debug_print_time_ >= 0.5) {
            last_debug_print_time_ = now;
            JointCommandGenerator::ToVector(joint_command_, joint_command_vector_);
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "[step] t=" << std::setw(9) << now << " joint_command=\n"
                      << joint_command_vector_.transpose() << std::endl;
            const Eigen::Vector3d base_pos_now(gc[0], gc[1], gc[2]);
            const Eigen::Quaterniond base_quat_now(gc[3], gc[4], gc[5], gc[6]);
            std::cout << "[step] base_pos=\n" << std::setw(10) << base_pos_now.x()
                      << " " << std::setw(10) << base_pos_now.y() << " "
                      << std::setw(10) << base_pos_now.z()
                      << " base_quat(wxyz)= " << std::setw(10) << base_quat_now.w()
                      << " " << std::setw(10) << base_quat_now.x() << " "
                      << std::setw(10) << base_quat_now.y() << " "
                      << std::setw(10) << base_quat_now.z() << std::endl;
          }
          pd_target_pos_[0] = joint_command_.base_pos.x();
          pd_target_pos_[1] = joint_command_.base_pos.y();
          pd_target_pos_[2] = joint_command_.base_pos.z();
          const Eigen::Quaterniond base_quat =
              joint_command_.base_quat.normalized();
          pd_target_pos_[3] = base_quat.w();
          pd_target_pos_[4] = base_quat.x();
          pd_target_pos_[5] = base_quat.y();
          pd_target_pos_[6] = base_quat.z();
          if (base_vel_dim > 0 && pd_target_vel_.size() >= base_vel_dim) {
            pd_target_vel_.segment(0, base_vel_dim).setZero();
          }
        }
        if (has_coupled_joints) {
          pd_target_pos_[idx_gc_7] = joint_command_.q_des[0];
          pd_target_pos_[idx_gc_14] = joint_command_.q_des[1];
          pd_target_vel_[idx_gv_7] = joint_command_.v_des[0];
          pd_target_vel_[idx_gv_14] = joint_command_.v_des[1];
        }
        end_effector_->setPdTarget(pd_target_pos_, pd_target_vel_);
      }

      // Gear Coupling Logic
      if (joint_count >= 8) {
        if (gc.size() > idx_gc_14 && gv.size() > idx_gv_14) {
          const double error_p1 = gc[idx_gc_7] + gc[idx_gc_11];
          const double error_v1 = gv[idx_gv_7] + gv[idx_gv_11];
          const double tau_gear1 = -kp_gear_ * error_p1 - kd_gear_ * error_v1;

          const double error_p2 = gc[idx_gc_10] + gc[idx_gc_14];
          const double error_v2 = gv[idx_gv_10] + gv[idx_gv_14];
          const double tau_gear2 = -kp_gear_ * error_p2 - kd_gear_ * error_v2;

          tau_total[idx_gv_7] += tau_gear1;
          tau_total[idx_gv_11] += tau_gear1;
          tau_total[idx_gv_10] += tau_gear2;
          tau_total[idx_gv_14] += tau_gear2;
        }
      }
      // ---------------------------------------------------------
      // Gravity compensation (toggle in config.yaml)
      // ---------------------------------------------------------
      end_effector_->clearExternalForcesAndTorques();
      wrench_->clearExternalForcesAndTorques();
      if (gravity_compensation_) {
        const double g = 9.81;
        for (size_t body_idx = 0; body_idx < ee_body_masses_.size(); ++body_idx) {
          end_effector_->setExternalForce(
              body_idx, {0.0, 0.0, ee_body_masses_[body_idx] * g});
        }
        for (size_t body_idx = 0; body_idx < wrench_body_masses_.size(); ++body_idx) {
          wrench_->setExternalForce(
              body_idx, {0.0, 0.0, wrench_body_masses_[body_idx] * g});
        }
      }
      // Pull the floating base toward the commanded pose using external force/torque.
      if (has_command && base_force_enabled && base_gc_dim == 7 &&
          static_cast<size_t>(base_body_idx_) < ee_body_masses_.size() &&
          gc.size() >= 7 && gv.size() >= 6) {
        const Eigen::Vector3d base_pos(gc[0], gc[1], gc[2]);
        const Eigen::Vector3d base_vel(gv[0], gv[1], gv[2]);
        const Eigen::Vector3d base_ang_vel(gv[3], gv[4], gv[5]);
        const Eigen::Vector3d pos_error = joint_command_.base_pos - base_pos;
        const Eigen::Vector3d vel_error = -base_vel;
        base_pos_error_integral_ += pos_error * simulation_dt_;
        base_pos_error_integral_ =
            base_pos_error_integral_.cwiseMax(
                Eigen::Vector3d::Constant(-base_force_pos_i_max_))
                .cwiseMin(Eigen::Vector3d::Constant(base_force_pos_i_max_));
        Eigen::Vector3d base_force =
            base_force_pos_kp_ * pos_error +
            base_force_pos_kd_ * vel_error +
            base_force_pos_ki_ * base_pos_error_integral_;
        const double force_norm = base_force.norm();
        if (force_norm > base_force_max_ && force_norm > 1e-9) {
          base_force *= (base_force_max_ / force_norm);
        }
        end_effector_->setExternalForce(
            base_body_idx_, {base_force.x(), base_force.y(), base_force.z()});

        // Rotation PD in world frame via quaternion error -> axis-angle.
        Eigen::Quaterniond base_quat(gc[3], gc[4], gc[5], gc[6]);
        base_quat.normalize();
        const Eigen::Quaterniond des_quat = joint_command_.base_quat.normalized();
        Eigen::Quaterniond q_err = des_quat * base_quat.conjugate();
        q_err.normalize();
        if (q_err.w() < 0.0) {
          q_err.coeffs() *= -1.0;
        }
        const Eigen::Vector3d qv(q_err.x(), q_err.y(), q_err.z());
        const double qv_norm = qv.norm();
        Eigen::Vector3d rot_error = Eigen::Vector3d::Zero();
        if (qv_norm > 1e-9) {
          const double angle = 2.0 * std::atan2(qv_norm, q_err.w());
          rot_error = (angle / qv_norm) * qv;
        }
        const Eigen::Vector3d ang_vel_error = -base_ang_vel;
        base_rot_error_integral_ += rot_error * simulation_dt_;
        base_rot_error_integral_ =
            base_rot_error_integral_.cwiseMax(
                Eigen::Vector3d::Constant(-base_force_rot_i_max_))
                .cwiseMin(Eigen::Vector3d::Constant(base_force_rot_i_max_));
        Eigen::Vector3d base_torque =
            base_force_rot_kp_ * rot_error +
            base_force_rot_kd_ * ang_vel_error +
            base_force_rot_ki_ * base_rot_error_integral_;
        const double torque_norm = base_torque.norm();
        if (torque_norm > base_torque_max_ && torque_norm > 1e-9) {
          base_torque *= (base_torque_max_ / torque_norm);
        }
        end_effector_->setExternalTorque(
            base_body_idx_, {base_torque.x(), base_torque.y(), base_torque.z()});
      }
      end_effector_->setGeneralizedForce(tau_total);
      const auto integrate_start = Clock::now();
      world_->integrate();
      integrate_ms += Ms(Clock::now() - integrate_start).count();

      // ---------------------------------------------------------
      // [4] Bolt Logic
      // ---------------------------------------------------------
      const auto bolt_start = Clock::now();
      double angle = bolt_->getGeneralizedCoordinate()[1]; 
      double omega = bolt_->getGeneralizedVelocity()[1]; 
      double translation = angle / (2.0 * M_PI) * thread_pitch_; 
      double velocity = omega / (2.0 * M_PI) * thread_pitch_; 

      Eigen::VectorXd b_gc = bolt_->getGeneralizedCoordinate().e();
      Eigen::VectorXd b_gv = bolt_->getGeneralizedVelocity().e();
      b_gc[0] = translation; 
      b_gv[0] = velocity; 
      bolt_->setState(b_gc, b_gv);
      bolt_logic_ms += Ms(Clock::now() - bolt_start).count();
    }
    // const auto sim_loop_end = Clock::now();
    // [role] sim_loop_start/sim_loop_end were used to profile the full control step.
    // std::cout << "[step] sim_loop_ms=" << Ms(sim_loop_end - sim_loop_start).count()
    //           << " integrate_ms=" << integrate_ms
    //           << " bolt_logic_ms=" << bolt_logic_ms << "\n";

    // Timing scaffold for observation update (currently unused; kept for quick profiling).
    // const auto obs_start = Clock::now();
    updateObservation();
    // std::cout << "[step] update_obs_ms="
    //           << Ms(Clock::now() - obs_start).count() << "\n";

    const auto& gc_now = end_effector_->getGeneralizedCoordinate();
    const double gc7_now = (gc_now.size() > idx_gc_7) ? gc_now[idx_gc_7] : 0.0;
    const double gc14_now =
        (gc_now.size() > idx_gc_14) ? gc_now[idx_gc_14] : 0.0;
    // std::cout << "[step] gc_7=" << gc7_now << " gc_14=" << gc14_now << "\n";
    
    double reward = bolt_->getGeneralizedVelocity()[1]; 
    if(std::isnan(reward)) reward = 0.0;

    return reward;
  }

  /// [OBSERVE] - 수정된 부분 (.e() 추가)
  void updateObservation() {
    obDouble_.head(gcDim_) = end_effector_->getGeneralizedCoordinate().e();
    obDouble_.segment(gcDim_, gvDim_) = end_effector_->getGeneralizedVelocity().e();

    raisim::Vec<3> ee_pos_W;
    raisim::Mat<3, 3> ee_rot_W;
    end_effector_->getFramePosition(ee_frame_idx_, ee_pos_W);
    end_effector_->getFrameOrientation(ee_frame_idx_, ee_rot_W);
    const Eigen::Vector3d ee_pos = ee_pos_W.e();
    const Eigen::Matrix3d R_ee = ee_rot_W.e();

    raisim::Vec<3> wrench_pos_W;
    raisim::Mat<3, 3> wrench_rot_W;
    wrench_->getFramePosition(wrench_frame_idx_, wrench_pos_W);
    wrench_->getFrameOrientation(wrench_frame_idx_, wrench_rot_W);
    const Eigen::Vector3d wrench_pos = wrench_pos_W.e();
    const Eigen::Matrix3d R_wrench = wrench_rot_W.e();

    raisim::Vec<3> bolt_pos_W;
    bolt_->getFramePosition(bolt_frame_idx_, bolt_pos_W);
    const Eigen::Vector3d bolt_pos = bolt_pos_W.e();

    int idx = gcDim_ + gvDim_;
    obDouble_.segment<3>(idx) = ee_pos;
    idx += 3;
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        obDouble_[idx++] = R_ee(r, c);
      }
    }

    const Eigen::Vector3d p_err_wrench = wrench_pos - ee_pos;
    obDouble_.segment<3>(idx) = p_err_wrench;
    idx += 3;

    Eigen::Quaterniond q_ee(R_ee);
    Eigen::Quaterniond q_wrench(R_wrench);
    Eigen::Quaterniond q_err = (q_wrench * q_ee.conjugate()).normalized();
    obDouble_[idx++] = q_err.w();
    obDouble_[idx++] = q_err.x();
    obDouble_[idx++] = q_err.y();
    obDouble_[idx++] = q_err.z();

    const Eigen::Vector3d p_err_bolt = bolt_pos - wrench_pos;
    obDouble_.segment<3>(idx) = p_err_bolt;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);
    return false;
  }
  
  void setCurriculumFactor(double factor) {
      curriculumFactor_ = factor;
  }

  void getDemoState(Eigen::VectorXd& bolt_gc,
                    Eigen::VectorXd& bolt_gv,
                    Eigen::VectorXd& wrench_gc,
                    Eigen::VectorXd& wrench_gv,
                    Eigen::VectorXd& robot_gc,
                    Eigen::VectorXd& robot_gv) const {
    bolt_gc = bolt_->getGeneralizedCoordinate().e();
    bolt_gv = bolt_->getGeneralizedVelocity().e();
    wrench_gc = wrench_->getGeneralizedCoordinate().e();
    wrench_gv = wrench_->getGeneralizedVelocity().e();
    robot_gc = end_effector_->getGeneralizedCoordinate().e();
    robot_gv = end_effector_->getGeneralizedVelocity().e();
  }

 private:
  raisim::ArticulatedSystem* bolt_;
  raisim::ArticulatedSystem* wrench_;
  raisim::ArticulatedSystem* end_effector_;
  
  double simulation_dt_;
  double control_dt_;
  double thread_pitch_;
  double kp_gear_;
  double kd_gear_;
  double terminalRewardCoeff_ = -10.0;
  
  std::string resourceDir_;
  bool visualizable_ = false;
  int gcDim_, gvDim_;
  
  double curriculumFactor_ = 1.0;

  Eigen::VectorXd obDouble_;
  Eigen::VectorXd actionMean_;
  Eigen::VectorXd actionStd_;

  // [4] 새로 추가된 변수 선언
  double ee_left_init_;
  double ee_right_init_;
  JointCommandMode command_mode_ = JointCommandMode::kSine;
  double grasp_gc14_start_ = 0.0;
  double grasp_gc14_target_ = 0.1337;
  double grasp_gc14_duration_ = 1.0;
  double grasp_approach_duration_ = 5.0;
  double grasp_move_duration_ = 1.0;
  double grasp_rotate_duration_ = 2.0;
  double grasp_rotate_radius_ = 0.5;

  std::string ee_frame_name_ = "ee_base";
  std::vector<double> ee_body_masses_;
  std::vector<double> wrench_body_masses_;
  size_t ee_frame_idx_ = 0;
  std::string ee_4_left_frame_name_ = "ee_4_left";
  size_t ee_4_left_frame_idx_ = 0;
  std::string ee_4_right_frame_name_ = "ee_4_right";
  size_t ee_4_right_frame_idx_ = 0;
  std::string wrench_frame_name_ = "wrench";
  size_t wrench_frame_idx_ = 0;
  std::string bolt_frame_name_ = "bolt_rev";
  size_t bolt_frame_idx_ = 0;

  const Eigen::Vector3d grasp_point_left_local_ =
      Eigen::Vector3d(0.01091, 0.0375, 0.0);
  const Eigen::Vector3d grasp_point_right_local_ =
      Eigen::Vector3d(-0.01091, 0.0375, 0.0);
  const Eigen::Vector3d grasp_target_offset_wrench_ =
      Eigen::Vector3d(0.082521, 0.012646, -7.679E-19);

  int render_port_ = 8080;
  std::unique_ptr<JointCommandGenerator> joint_command_generator_;
  raisim::TimeSeriesGraph* joint_command_graph_ = nullptr;
  raisim::VecDyn joint_command_vis_;
  JointCommand joint_command_;
  Eigen::VectorXd joint_command_vector_;
  Eigen::VectorXd pd_target_pos_;
  Eigen::VectorXd pd_target_vel_;
  double last_debug_print_time_ = -1.0;
  bool use_action_command_;
  double active_kp_ = 0.0;
  double active_kd_ = 0.0;
  double base_kp_ = 0.0;
  double base_kd_ = 0.0;
  int base_body_idx_ = 0;
  double base_force_pos_kp_ = 0.0;
  double base_force_pos_kd_ = 0.0;
  double base_force_pos_ki_ = 0.0;
  double base_force_pos_i_max_ = 0.5;
  Eigen::Vector3d base_pos_error_integral_ = Eigen::Vector3d::Zero();
  double base_force_rot_kp_ = 0.0;
  double base_force_rot_kd_ = 0.0;
  double base_force_rot_ki_ = 0.0;
  double base_force_rot_i_max_ = 0.5;
  Eigen::Vector3d base_rot_error_integral_ = Eigen::Vector3d::Zero();
  double base_force_max_ = 200.0;
  double base_torque_max_ = 200.0;
  bool gravity_compensation_ = false;

  bool launchServerWithFallback() {
#if defined(__linux__) || defined(__APPLE__)
    render_port_ = 8080;
    const auto& port_node = cfg_["render_port"];
    if (!port_node.IsNone()) {
      render_port_ = port_node.template As<int>();
    }
    const int max_tries = 10;
    for (int i = 0; i < max_tries; i++) {
      const int port = render_port_ + i;
      if (!isPortAvailable(port)) {
        continue;
      }
      updateUnityGuiPort(port);
      server_->launchServer(port);
      RSINFO("RaisimServer launched on port " << port);
      render_port_ = port;
      return true;
    }

    RSWARN("RaisimServer could not open a socket. Visualization disabled.");
    return false;
#else
    server_->launchServer();
    return true;
#endif
  }

#if defined(__linux__) || defined(__APPLE__)
  bool isPortAvailable(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return false;
    }

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    const bool ok = (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    ::close(fd);
    return ok;
  }

  void updateUnityGuiPort(int port) {
    const auto& gui_node = cfg_["unity_gui_settings"];
    std::string gui_path =
        "/home/Jiseong/raisim_ws/raisimlib/raisimUnity/linux/gui_settings.xml";
    if (!gui_node.IsNone()) {
      gui_path = gui_node.template As<std::string>();
    }

    std::ifstream in(gui_path);
    if (!in) {
      RSWARN("Failed to read RaisimUnity gui_settings.xml at " << gui_path);
      return;
    }
    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    const std::string key = "ip_port value=\"";
    const auto key_pos = content.find(key);
    if (key_pos == std::string::npos) {
      RSWARN("ip_port entry not found in " << gui_path);
      return;
    }
    const auto value_start = key_pos + key.size();
    const auto value_end = content.find('"', value_start);
    if (value_end == std::string::npos) {
      RSWARN("ip_port entry malformed in " << gui_path);
      return;
    }
    content.replace(value_start, value_end - value_start, std::to_string(port));
    std::ofstream out(gui_path, std::ios::trunc);
    if (!out) {
      RSWARN("Failed to write RaisimUnity gui_settings.xml at " << gui_path);
      return;
    }
    out << content;
  }
#endif
};

} // namespace raisim
