#include "UtilityFunctions.hpp"
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
