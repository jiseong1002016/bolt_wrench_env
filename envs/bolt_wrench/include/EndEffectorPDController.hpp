#ifndef BOLT_WRENCH_INCLUDE_END_EFFECTOR_PD_CONTROLLER_HPP_
#define BOLT_WRENCH_INCLUDE_END_EFFECTOR_PD_CONTROLLER_HPP_

#include <Eigen/Core>
#include <algorithm>

class EndEffectorPDController {
 public:
  EndEffectorPDController(int gc_dim, int gv_dim, double kp, double kd)
      : gc_dim_(gc_dim), gv_dim_(gv_dim), kp_(kp), kd_(kd) {
    base_gc_dim_ = (gc_dim_ == gv_dim_ + 1) ? 7 : 0;
    joint_count_ = gc_dim_ - base_gc_dim_;
    base_vel_dim_ = gv_dim_ - joint_count_;
  }

  void setGains(double kp, double kd) {
    kp_ = kp;
    kd_ = kd;
  }

  void computeTorques(const Eigen::Ref<const Eigen::VectorXd>& gc,
                      const Eigen::Ref<const Eigen::VectorXd>& gv,
                      const Eigen::Ref<const Eigen::Vector2d>& q_des,
                      const Eigen::Ref<const Eigen::Vector2d>& v_des,
                      Eigen::Ref<Eigen::VectorXd> tau_out) const {
    tau_out.setZero();
    if (joint_count_ < 8) {
      return;
    }

    const int idx_gc_7 = base_gc_dim_ + 0;
    const int idx_gc_14 = base_gc_dim_ + 7;
    const int idx_gv_7 = base_vel_dim_ + 0;
    const int idx_gv_14 = base_vel_dim_ + 7;

    if (idx_gc_7 >= gc.size() || idx_gc_14 >= gc.size() ||
        idx_gv_7 >= gv.size() || idx_gv_14 >= gv.size() ||
        idx_gv_14 >= tau_out.size()) {
      return;
    }

    tau_out[idx_gv_7] = kp_ * (q_des[0] - gc[idx_gc_7]) +
                        kd_ * (v_des[0] - gv[idx_gv_7]);
    tau_out[idx_gv_14] = kp_ * (q_des[1] - gc[idx_gc_14]) +
                         kd_ * (v_des[1] - gv[idx_gv_14]);
  }

 private:
  int gc_dim_ = 0;
  int gv_dim_ = 0;
  int base_gc_dim_ = 0;
  int base_vel_dim_ = 0;
  int joint_count_ = 0;
  double kp_ = 0.0;
  double kd_ = 0.0;
};

#endif  // BOLT_WRENCH_INCLUDE_END_EFFECTOR_PD_CONTROLLER_HPP_
