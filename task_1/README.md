## Task 1

In order to help in reaching the final goals, the project assignment splits each task in 3 smaller subtasks. For task 1 those are:
- Subtask 1.1 - Distributed optimization: implement the *Gradient Tracking* algorithm for distributed cost-coupled optimization
- Subtask 1.2 - Centralized classification: solve a logistic regression classification problem in a centralized way by implementing some sort of gradient method
- Subtask 1.3 - Distributed classification: solve the same problem of subtask 1.2 but in a distributed way by using a modified version of the algorithm developed in subtask 1.3

To execute all the subtasks simply run the file `main.py` inside the folder `task_1` with a recent version of Python (the code has been developed with version 3.12.4). The code makes use of the `ArgumentParser` library of Python to parse command line arguments to allow the user to change some of the simulation parameters. The following terminal output shows all the options available.
```bash
usage: main.py [-h] [-n NODES] [-i ITERS] [-p MAX_POINTS] [--no-plots] [--skip SKIP]

options:
  -h, --help            show this help message and exit
  -n NODES, --nodes NODES
  -i ITERS, --iters ITERS
  -p MAX_POINTS, --max-points MAX_POINTS
  --no-plots
  --skip SKIP
```
To disable plots rendering you can pass the `--no-plots` flag, while to selectively skip a subtask you can pass the list of subtask to skip with the `--skip` flag, e.g. to skip subtask 1 and 3 in order to run only the subtask 1.2 you pass `--skip 1,3`.
By default the simulation runs with the following parameters:
- `--nodes 10`
- `--iters 1000` (however be careful that this option is only relevant for subtask 1.1 and 1.2 since the number of iterations for subtask 1.3 is not the same for all the case studies but tuned to the particular case)
- `--max-points 1000`