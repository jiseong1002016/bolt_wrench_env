# RAIPAL-URDF
`URDF` file for RAIPAL

All relevant documentations are [here](https://docs.google.com/document/d/1QMAeDxHOptx8B7S9kiuZ11pdUJxH2UBg1TFjP7rBxus/edit?usp=sharing)

All important calculations and measurments are [here](https://docs.google.com/spreadsheets/d/1z0rITUL5gzTkE2qTLBxqRETlY42cNg-HGnbWWTu-GAM/edit?usp=sharing) (Google Sheet)

A simple demo using `raisim` can be found [here](https://github.com/iamchoking/urdf-sandbox/blob/master/src/raipal.cpp)

## Changelogs
* **2025-09-02**: First stable version published
    * Physical parameters confirmed with direct measurement
        * Forearm direct measurments pending
    * `raisim` stability confirmed (with closed-loop constraint)

* **2025-10-17**: Modularized end urdf components / `stub` module
    * Many different hardware configurations managed with "modules" (sub-xml files within urdf)
        * Use `scripts/urdf-writer.py` to generate urdf's with appropriate modules attached
    * Added `stub` module
        * 0-dof end-effector to hold the load of 5 to 10 kg.

* **2026-01-02**: [*BREAKING*] Forarm parity reversed
    * Real hardware tests revealed collision / cabling issues that led to the `urdf` forearm assemblies of right and left arms to be reveresed

<!-- ## TODO -->
