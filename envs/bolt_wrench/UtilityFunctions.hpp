#ifndef UTILITY_FUNCTIONS_HPP
#define UTILITY_FUNCTIONS_HPP

#include <iostream>
#include <Eigen/Dense>
#include "raisim/World.hpp"

// 시뮬레이션 설정을 저장할 구조체
struct SimulationConfig {
    double control_dt;
    double simulation_dt;
    int max_iteration;
    double thread_pitch;
    int pris_joint_index;
    bool PC;
    bool TC;
    double ee_01_left_gc;
    double ee_02_right_gc;
    double ee_01_left_gf;
    double ee_02_right_gf;
    
    // 계산된 값
    int maxSimSteps;
};

// YAML 파일 로드 함수 선언
SimulationConfig loadConfig(const std::string& filePath);

// TorqueController class declaration
class TorqueController {
public:
    explicit TorqueController(raisim::ArticulatedSystem* system);
    void applyTorque(const Eigen::VectorXd& torque);

private:
    raisim::ArticulatedSystem* system_;
};

// PositionController class declaration
class PositionController {
public:
    explicit PositionController(raisim::ArticulatedSystem* system);
    void setTargetPosition(const Eigen::VectorXd& targetPosition);
    void update();

private:
    raisim::ArticulatedSystem* system_;
    Eigen::VectorXd targetPosition_;
    double kp_ = 100.0; // Example proportional gain
    double kd_ = 10.0;  // Example derivative gain
};

void SetEndEffectorGeneralizedCoordinates(
    raisim::ArticulatedSystem* end_effector,
    double ee_01_left_gc,
    double ee_02_right_gc
);

// LPF for wrench was added here but is currently disabled per request.
// using Wrench6d = Eigen::Matrix<double, 6, 1>;
// void lowPassFilterWrench(
//     const Wrench6d& raw,
//     Wrench6d& state,
//     double cutoff_hz,
//     double dt
// );
// void lowPassFilterWrenchSafe(
//     const Wrench6d& raw,
//     Wrench6d& state,
//     double cutoff_hz,
//     double dt,
//     double force_limit,
//     double torque_limit,
//     double slew_force,
//     double slew_torque
// );

#endif // UTILITY_FUNCTIONS_HPP
