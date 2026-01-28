cmake_minimum_required(VERSION 3.10)
project(raisim_gym_bolt_wrench)

set(CMAKE_CXX_STANDARD 17)

# Default to Release if no build type is specified (single-config generators).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 1. 패키지 찾기
find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# 2. 라이브러리 소스
set(src
    UtilityFunctions.cpp
    JointCommandGenerator.cpp
    rsc/sweep_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../raisim_gym.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Yaml.cpp 
)

# 3. 라이브러리 생성 (이름: bolt_wrench)
add_library(bolt_wrench SHARED ${src})

# 4. 속성 및 정의 설정 (★여기가 핵심입니다★)
# 접두사 제거 (libbolt_wrench.so -> bolt_wrench.so)
set_target_properties(bolt_wrench PROPERTIES PREFIX "")

# 컴파일러에게 매크로 값 전달
# RAISIMGYM_TORCH_ENV_NAME=bolt_wrench -> PYBIND11_MODULE(bolt_wrench, m) 생성
# ENVIRONMENT_NAME=RaisimGymEnv -> Python Class 이름 (bolt_wrench.RaisimGymEnv)
target_compile_definitions(bolt_wrench PRIVATE 
    RAISIMGYM_TORCH_ENV_NAME=bolt_wrench
    ENVIRONMENT_NAME=RaisimGymEnv
)

# 5. 인클루드 및 링크 설정
target_include_directories(bolt_wrench PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${RAISIM_INCLUDE_DIRS}
)

target_link_libraries(bolt_wrench PUBLIC 
    raisim::raisim
    Eigen3::Eigen
    pybind11::module
    OpenMP::OpenMP_CXX
)

# 6. 컴파일 옵션
if(UNIX)
  target_compile_options(bolt_wrench PRIVATE -O3 -ffast-math -g -fPIC)
endif()

# 7. Demo generator executable (disabled)
# add_executable(demo_generator
#   demo_generator.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/../../Yaml.cpp
# )
# target_include_directories(demo_generator PUBLIC
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${EIGEN3_INCLUDE_DIRS}
#   ${RAISIM_INCLUDE_DIRS}
# )
# target_link_libraries(demo_generator PUBLIC
#   raisim::raisim
#   Eigen3::Eigen
# )
# if(UNIX)
#   target_compile_options(demo_generator PRIVATE -O3 -ffast-math -g)
# endif()
