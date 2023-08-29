Step 1. Follow the installation instructions for instrupy and orbitpy (in that order). These are included as submodules. In this process you will create a conda virtual environment which you will keep for this visualizer.
Step 2. Install the following dependencies (using pip install):
    numpy
    matplotlib
    basemap
    imageio
Step 3. You can run "python src/full_mission.py" which should work out of the box if the install is correct. Let me know if there are any issues with it. You can modify the scenario settings in full_mission.py and in create_mission.py
Step 4 (optional). If desired, you can run the visualizer in steps (create, execute, process, plan, plot) by calling them individually with the desired settings.

![](https://github.com/bgorr/satplan/example.gif)
