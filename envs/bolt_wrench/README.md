Bolt Wrench Environment
=======================

Kill 8080
-----
sudo fuser -k 8080/tcp

Build 
-----
cd build && cmake .. && cmake --build . && cd ..

Demo Genarate
-----
python demo_generator.py
python demo_generator.py --render

Build & Demo Generate 
-----
cd build && cmake .. && cmake --build . && cd .. && python demo_generator.py


Run
-----
python runner.py

Options
-------
- --cfg <path>     (default: ./rsc/config.yaml)
- --mode <string>  (default: train)
- --weight <path>  (default: none)

Examples
--------
- python runner.py
- python runner.py --cfg rsc/config.yaml
- python runner.py --mode train
- python runner.py --weight /path/to/checkpoint.pt

Note
----
- 100, 0.1 에서 가만히 회전

Gains (current)
---------------
- End-effector PD: active_kp_=1.0, active_kd_=0.5 (Environment.hpp)
- Gear coupling: kp_gear_=1000.0, kd_gear_=1.0 (Environment.hpp)
- PositionController: kp_=100.0, kd_=10.0 (UtilityFunctions.hpp)
- Raisim joint PD gains (joint 7/14 only): kp=1.0, kd=0.5 (Environment.hpp)

dt (current)
------
- simulation_dt: 0.0001 # 0.0025
- control_dt: 0.001 # 0.01
