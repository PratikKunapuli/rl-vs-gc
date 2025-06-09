# Envs
This folder contains environment implementations for all the tasks. Each task is defined in it's own folder, with the `folder.__init__.py` defining the registry name of the environment. Environments are implemented as IsaacLab `DirectRLEnv` environments, with a corresponding `DirectRLEnvCfg` configuration class used to modify parameters. 

## Trajectory Tracking
This is the primary environment used, and is capable of simulating both hovering and Lissajous trajectories for the following robot morphologies: 0-DOF Aerial Manipulator, Quadrotor, and Brushless Crazyflie.

Available environments:
- `Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0` - 0-DOF Aerial Manipulator vehicle
- `Isaac-AerialManipulator-QuadOnly-TrajectoryTracking-v0` - Quadrotor-only part of Aerial Manipulator vehicle
- `Isaac-BrushlessCrazyflie-TrajectoryTracking-v0`- Brushless Crazyflie robot. This is intended to be used with a differnt control mode for low-level motor dynamics. 

### Configuration Parameters:
Examples of configurations for different parameters are given below. 

#### Trajectory 
Trajectory specification is done by assuming a Lissajous curve for each x, y, z in position and yaw. This trajectory is computed, along with its derivatives, within the environment and is specified by the amplitudes, frequencies, phases, and offsets. These parameters also have a randomization range associated with them if new trajectories should be sampled randomly upon reset.
```python
# Hovering at (0, 0, 3) with no yaw
env_cfg.lissajous_amplitudes = [0.0, 0.0, 0.0, 0.0]
env_cfg.lissajous_offsets = [0.0, 0.0, 3.0, 0.0]

# Random Lissajous Tracking
env_cfg.lissajous_amplitudes_rand_ranges = [2.0, 2.0, 2.0, 2.0]
env_cfg.lissajous_frequencies_rand_ranges = [3.0, 3.0, 3.0, 3.0]
```

#### Reward
Reward shaping is done by setting various penalty and bonus terms in the environment configuration:
```python
env_cfg.lin_vel_reward_scale = -0.05
env_cfg.yaw_error_reward_scale = -4.0
```

#### Visualiation
During video recordings, it is sometimes helpful to visualize differnt information about the robot/trajectory. Using the `triad` mode will place a triad at the control frame and trajectory location. `robot` mode will visualize a full robot instead, with a history of states shown as small spheres. 

```python
env_cfg.viz_mode = "triad"

env_cfg.viz_mode = "robot"
env_cfg.viz_history_length = 100 # 100 data points
```

## Ball Catching
The Ball Catching environment simulates a task requiring the 0-DOF Aerial Manipulator to catch 5 balls which are randomly thrown in the air. This environment is intended to be used as an evaluation environment as opposed to directly training within the environment. 

