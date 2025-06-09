# Controllers
This folder contains the implementation of the Geometric Controller as well as optimized gains for the controller under various tasks and configurations. 

## Geometric Controller

The `GeometricController` class implements a control strategy for aerial vehicles based on geometric principles. This controller is designed to provide robust and stable tracking of position and orientation trajectories on the Special Euclidean group SE(3).

## Optimizing Controller Gains
Optimizing the controller gains is done by using the Optuna library to run Bayseian Optimization and produce an optimal set of gains based on the environment's reward function. This is done in the `gc_tuning.py` script, which has some configuration ability by choosing whether to use integral and feed-forward terms in the geometric controller. 

Running the script for the first time will create a database and save the logged parameters suggested as well as the achieved rewards in the `database_gc_tuning.sqlite3` file, which can be pointed to by Optuna-Dashboard for a live-view on training. 

### Example:
```bash
python gc_tuning.py --task Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0 --num_envs 1024 
```

## Loading an Optimized Controller
Running an optimzied controller can be done by recording the optimal gains found via Optuna and saving the parameters in a dictionary within the `gc_params.py` file. Pretrained gains are already listed in the file by the name of the IsaacLab Task with an appended name for the configuration. 

### Example:
```python
from controllers.geometric_controller import GeometricController
from controllers.gc_params import gc_params_dict

task_name  = "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0"
config_name = "Hover"
control_params_dict = gc_params_dict[task_name + "-" + config_name]["controller_params"]
vehicle = "Crazyflie" if "Crazyflie" in task_name else "AM"

# Must create environment to compute mass, inertia, etc.
env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
env = env.unwrapped

agent = GeometricController(env.num_envs, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, device=device, vehicle=gc_vehicle, **control_params_dict)
```

## Evaluating the Geometric Controller
Evaluation is done via the `rl/eval_rslrl.py` script, and by setting `--baseline true`, with an optional flag for the configuration name `--baseline_gains CONFIG_NAME`. 