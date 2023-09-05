# satplan

Visualization tool for autonomous Earth Observation Satellite missions

---

## Installation 

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Create and activate a virtual conda environment:

```
conda create -p ./.venv python=3.8

conda activate ./.venv
```
3. Install `gfortran`. and `make`. See [here](https://fortran-lang.org/learn/os_setup/install_gfortran).

4. Run make command in terminal in repository directory:
```
make install
```
---

## User Guide

 3. You can run "python src/full_mission.py" which should work out of the box if the install is correct. Let me know if there are any issues with it. You can modify the scenario settings in full_mission.py and in create_mission.py.

 4. (optional) If desired, you can run the visualizer in steps (create, execute, process, plan, plot) by calling them individually with the desired settings.

![Exciting gif](https://github.com/bgorr/satplan/blob/main/example.gif?raw=true)