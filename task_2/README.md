## Task 2

In detail, the goal of this task is to implement a multi-robot system in which a set of robots must move towards a set of assigned targets (one for each robot) while keeping the formation tight (i.e. the distance between each robot and the barycenter of the formation must be minimized). In a similar way to task 1, the project assignment splits task 2 in 3 subtasks:
- Subtask 2.1 - Problem setup: define the problem and a suitable cost function to solve the problem by using a custom implementation of the *Aggregative Tracking* algorithm
- Subtask 2.2 - ROS2 simulation: implement a ROS2 simulation of the multi-robot system and the algorithm developed in subtask 2.1
- Subtask 2.3 - Moving inside a corridor: extend the ROS2 simulation of subtask 2.2 to a more complex scenario in which the robots must move inside a corridor before reaching the targets

To execute subtask 2.1 or a Python version of subtask 2.3 simply run the file `main.py` inside the folder `task_2` with a recent version of Python (the code has been developed with version 3.12.4). As in task 1, the code makes use of the `ArgumentParser` library of Python to parse command line arguments to allow the user to change some of the simulation parameters. The following terminal output shows all the options available.
```bash
usage: main.py [-h] [-n NODES] [-i ITERS] [--no-plots] [--skip SKIP] [--skip-animations]

options:
  -h, --help            show this help message and exit
  -n NODES, --nodes NODES
  -i ITERS, --iters ITERS
  --no-plots
  --skip SKIP
  --skip-animations
```
To disable plots rendering you can pass the `--no-plots` flag, while to selectively skip a subtask you can pass the list of subtask to skip with the `--skip` flag, e.g. to skip subtask 1 and 3 in order to run only the subtask 1.2 you pass `--skip 1,3`. Both subtasks run a simple plot based animation at the end of the simulation computation, to disable it you can pass the `--skip-animations` flag.
By default the simulation runs with the following parameters:
- `--nodes 4`
- `--iters 40000` (this high number is actually needed only for subtask 2.3, for subtask 2.1 a lower number of iterations, such as 1000, should be enough)

To run subtask 2.2 and the ROS2 simulation of subtask 2.3 you need to have a working installation of ROS2 on your machine. The code has been developed and tested with ROS2 Humble on Ubuntu 22.04. To run the simulation you need to compile the code of the ROS2 workspace under `task_2/ros` and source it. Then, you can execute either subtask 2.2 or 2.3 by running one of the launch files provided under `task_2/ros/src/surveillance/launch`:
- for subtask 2.2 run `ros2 launch surveillance surveillance.launch.py`
- for subtask 2.3 run `ros2 launch surveillance corridor.launch.py`

Both launch files accept (and in some cases require) some parameters to be passed. In particular, the `surveillance.launch.py` file accepts the following parameters:
- nodes *(optional)*: represents the number of robots (and consequently the number of targets) to be used in the simulation (is equal to 4 by default)
- tradeoff *(required)*: a positive floating number, represents the tradeoff parameter to be used in the cost function, the higher the value the more the robots will tend to minimize the distance from their targets ignoring the constraint on the formation
- distance *(required)*: a positive floating number, represents the distance between the robots and their targets at the beginning of the simulation
- timer_period *(optional)*: a positive floating number, represents the period of the timer used to update the robots' positions (is equal to 0.01 by default)
- max_iters *(optional)*: a positive integer, represents the maximum number of iterations to run the simulation (is equal to 40000 by default)
While the `corridor.launch.py` file accepts the following parameters (all of which are optional):
- nodes: same as above
- timer_period: same as above
- max_iters: same as above (is equal to 40000 by default)
- case: an integer between 0 and 3 included, represents the case study to be run (is equal to 0 by default)
