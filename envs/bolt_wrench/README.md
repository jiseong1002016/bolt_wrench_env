Bolt Wrench Environment
=======================

Build
-----
1) mkdir -p build
2) cd build
3) cmake ..
4) cmake --build .
5) cd ..

Run
---
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
