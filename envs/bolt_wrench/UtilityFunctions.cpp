#include "UtilityFunctions.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "raisim/World.hpp"
#include "../../Yaml.hpp"

// 1. YAML 파일에서 시뮬레이션 설정 로드 함수 구현
SimulationConfig loadConfig(const std::string& filePath) {
    // [수정 1] Namespace 변경 (YAML -> Yaml) 및 파싱 로직 변경
    Yaml::Node config;
    try {
        Yaml::Parse(config, filePath);
    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to parse config file: " << filePath << std::endl;
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    SimulationConfig cfg;

    // [수정 2] .as<T>() -> .template As<T>() 변경
    // Yaml.hpp 구현상 템플릿 메소드 호출 시 .template 키워드가 필요할 수 있습니다.
    cfg.control_dt = config["control_dt"].template As<double>();
    cfg.simulation_dt = config["simulation_dt"].template As<double>();
    cfg.max_iteration = config["max_iteration"].template As<int>();
    cfg.thread_pitch = config["thread_pitch"].template As<double>();
    cfg.pris_joint_index = config["pris_joint_index"].template As<int>();
    
    cfg.PC = config["PC"].template As<bool>();
    cfg.TC = config["TC"].template As<bool>();
    
    cfg.ee_01_left_gc = config["ee_01_left_gc"].template As<double>();
    cfg.ee_02_right_gc = config["ee_02_right_gc"].template As<double>();
    
    cfg.ee_01_left_gf = config["ee_01_left_gf"].template As<double>();
    cfg.ee_02_right_gf = config["ee_02_right_gf"].template As<double>();

    cfg.maxSimSteps = int(cfg.control_dt / cfg.simulation_dt);
    
    return cfg;
}

// 2. TorqueController 클래스 구현
TorqueController::TorqueController(raisim::ArticulatedSystem* system) : system_(system) {}

void TorqueController::applyTorque(const Eigen::VectorXd& torque) {
    system_->setGeneralizedForce(torque);
}

// 3. PositionController 클래스 구현
PositionController::PositionController(raisim::ArticulatedSystem* system) : system_(system) {
    targetPosition_ = system_->getGeneralizedCoordinate().e();
}

void PositionController::setTargetPosition(const Eigen::VectorXd& targetPosition) {
    targetPosition_ = targetPosition;
}

void PositionController::update() {
    Eigen::VectorXd currentPosition = system_->getGeneralizedCoordinate().e();
    Eigen::VectorXd currentVelocity = system_->getGeneralizedVelocity().e();
    Eigen::VectorXd error = targetPosition_ - currentPosition;

    Eigen::VectorXd controlInput = kp_ * error - kd_ * currentVelocity;

    system_->setGeneralizedForce(controlInput);
}

// 4. 엔드이펙터 설정 함수
void SetEndEffectorGeneralizedCoordinates(
    raisim::ArticulatedSystem* end_effector,
    double ee_01_left_gc,
    double ee_02_right_gc
) {
    double gc_7 = ee_01_left_gc;
    double gc_14 = ee_02_right_gc;

    Eigen::VectorXd positions = end_effector->getGeneralizedCoordinate().e();
    
    if (positions.size() >= 8) {
        positions.tail(8) << gc_7, -gc_14-gc_7, gc_14+gc_7, -gc_14, -gc_7, gc_14+gc_7, -gc_14-gc_7, gc_14;
    }

    end_effector->setGeneralizedCoordinate(positions);
}

void lowPassFilterWrench(
    const Wrench6d& raw,
    Wrench6d& state,
    double cutoff_hz,
    double dt
) {
    cutoff_hz = std::max(0.0, cutoff_hz);
    dt = std::max(0.0, dt);

    if (cutoff_hz <= 0.0 || dt <= 0.0) {
        state = raw;
        return;
    }

    const double omega_c = 2.0 * M_PI * cutoff_hz;
    const double alpha = (omega_c * dt) / (omega_c * dt + 1.0);
    state = alpha * raw + (1.0 - alpha) * state;
}
//
namespace {

inline Eigen::Vector3d clampNorm(const Eigen::Vector3d& v, double limit) {
    if (limit <= 0.0) {
        return v;
    }
    const double n = v.norm();
    if (n <= limit || n <= 1e-12) {
        return v;
    }
    return v * (limit / n);
}

inline Eigen::Vector3d slewLimit(
    const Eigen::Vector3d& prev,
    const Eigen::Vector3d& next,
    double max_rate,
    double dt
) {
    if (max_rate <= 0.0 || dt <= 0.0) {
        return next;
    }
    const Eigen::Vector3d delta = next - prev;
    const double max_delta = max_rate * dt;
    const double delta_norm = delta.norm();
    if (delta_norm <= max_delta || delta_norm <= 1e-12) {
        return next;
    }
    return prev + delta * (max_delta / delta_norm);
}

} // namespace

void lowPassFilterWrenchSafe(
    const Wrench6d& raw,
    Wrench6d& state,
    double cutoff_hz,
    double dt,
    double force_limit,
    double torque_limit,
    double slew_force,
    double slew_torque
) {
    Wrench6d raw_clamped = raw;
    raw_clamped.head<3>() = clampNorm(raw_clamped.head<3>(), force_limit);
    raw_clamped.tail<3>() = clampNorm(raw_clamped.tail<3>(), torque_limit);

    const Wrench6d prev_state = state;
    lowPassFilterWrench(raw_clamped, state, cutoff_hz, dt);

    Eigen::Vector3d f = slewLimit(prev_state.head<3>(), state.head<3>(), slew_force, dt);
    Eigen::Vector3d t = slewLimit(prev_state.tail<3>(), state.tail<3>(), slew_torque, dt);

    f = clampNorm(f, force_limit);
    t = clampNorm(t, torque_limit);

    state.head<3>() = f;
    state.tail<3>() = t;
}

Eigen::Vector3d ComputeMovingPhaseCorrection(
    double now,
    double t_start,
    double t_end,
    const Eigen::Vector3d& wrench_home,
    const Eigen::Vector3d& wrench_origin,
    double gain,
    double max_corr
) {
    if (t_end <= t_start) {
        return Eigen::Vector3d::Zero();
    }

    const double t = (now - t_start) / (t_end - t_start);
    const double t_clamped = std::min(1.0, std::max(0.0, t));
    const double s = 0.5 - 0.5 * std::cos(M_PI * t_clamped);  // smooth start/end

    Eigen::Vector3d err = wrench_home - wrench_origin;
    err.z() = 0.0;  // no z correction

    Eigen::Vector3d corr = gain * s * err;
    const double corr_norm = corr.norm();
    if (corr_norm > max_corr && corr_norm > 1e-12) {
        corr = corr * (max_corr / corr_norm);
    }
    return corr;
}
