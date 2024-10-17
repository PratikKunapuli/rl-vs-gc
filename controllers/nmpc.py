import do_mpc
from casadi import *
import numpy as np
from utils import math_utilities as math_utils
import multiprocessing as mp


class NMPC:
    def __init__(self, n_envs, mass, inertia, ee_pos_transform=None, ee_ori_transform=None):
        self.n_envs = n_envs
        
        self.sim_time = 50 # total timesteps for simulation
        self.Ts = 0.1 # sampling time in seconds
        self.T_horizon = 50 # prediction horizon in timesteps 
        self.N = int(self.T_horizon/self.Ts) # total number of timesteps in the horizon
        self.mass = mass
        self.inertia = inertia
        self.gravity_vec = np.array([0, 0, 9.81])

        self.thrust_limit = 3.0
        self.moment_limit_xy = 0.5
        self.moment_limit_z = 0.025
 
        # Cost matrix for the NMPC Controller
        self.Q_pos = 10 * np.eye(3)
        self.Q_ori = 1 * np.eye(9)

        self.init_states = np.zeros((n_envs, 18, 1)) # Will be (n_envs, n_states)
        self.goal_positions = np.zeros((n_envs, 3, 1)) # Will be (n_envs, 3)
        self.goal_orientations = np.zeros((n_envs, 3, 3)) # Will be (n_envs, 9)
        self.models = []
        self.mpcs = []

        for env in range(n_envs):
            self.init_states[env] = np.vstack([np.zeros((3,1)), np.eye(3).reshape(-1, 1), np.zeros((3,1)), np.zeros((3,1))]) # Will be (n_envs, n_states)
            self.goal_positions[env] = np.zeros(3).reshape(-1, 1)
            self.goal_orientations[env] = np.eye(3)
            self.models.append(self.define_model(env))
            self.mpcs.append(self.define_mpc(env))

        # self.x0 = np.vstack([np.zeros((3,1)), np.eye(3).reshape(-1, 1), np.zeros((3,1)), np.zeros((3,1))]) # Will be (n_envs, n_states)
        # self.u0 = None
        # self.ee_in_com_frame = ee_in_com_frame
        # self.goal_pos = np.zeros(3).reshape(-1, 1)
        # self.goal_ori = np.eye(3)
        
        # self.model=self.define_model()
        # self.mpc = self.define_mpc()
        # self.estimator = do_mpc.estimator.StateFeedback(self.model)
        # self.simulator = self.define_simulator()


    def define_model(self, env_num):
        """
        Instantiate a model for the NMPC Controller. 
        This model should be the quadrotor dynamics model of the center of mass. 
        The variables are x, v, R, and omega. 
        Inputs are the thrust and torques.
        """
        model_type = "discrete"
        model = do_mpc.model.Model(model_type)

        # Define the state variables
        x = model.set_variable(var_type='_x', var_name='x', shape=(3,1))
        v = model.set_variable(var_type='_x', var_name='v', shape=(3,1))
        R = model.set_variable(var_type='_x', var_name='R', shape=(3,3))
        omega = model.set_variable(var_type='_x', var_name='omega', shape=(3,1))
        
        # Define the input variables
        thrust = model.set_variable(var_type='_u', var_name='thrust', shape=(1,1))
        moments = model.set_variable(var_type='_u', var_name='moments', shape=(3,1))

        # Define the right-hand side of the dynamics
        thrust_scaled = SX.zeros(3,1)
        thrust_scaled[2] = ((thrust + 1.0) / 2.0) * self.thrust_limit * self.mass * self.gravity_vec[2]
        moments_scaled = SX.zeros(3,1)
        moments_scaled[:2] = moments[:2] * self.moment_limit_xy
        moments_scaled[2] = moments[2] * self.moment_limit_z

        x_dot = v
        v_dot = (1/self.mass) * R @ (thrust_scaled) - self.gravity_vec
        R_dot = R @ skew(omega)
        # dt = SX(self.Ts)
        # R_dot = expm(skew(omega)*dt)
        omega_dot = inv(self.inertia) @ (moments_scaled - cross(omega, self.inertia @ omega))

        # Define the dynamics
        model.set_rhs('x', x + x_dot*self.Ts)
        model.set_rhs('v', v + v_dot*self.Ts)
        model.set_rhs('R', R + R_dot*self.Ts)
        # model.set_rhs('R', R@R_dot)
        model.set_rhs('omega', omega + omega_dot*self.Ts)

        # Define cost
        model, cost_expression = self.get_cost_expression(model, self.goal_positions[env_num], self.goal_orientations[env_num])
        model.set_expression(expr_name='cost', expr=cost_expression)

        model.setup()
        return model

    def get_cost_expression(self, model, goal_pos, goal_ori):
        """
        Define the cost function for the NMPC Controller. 
        We want the cost function to be of the end-effector pose instead of the COM
        """
        # Define the cost function
        X_pos = model.x['x'] - goal_pos
        X_ori = reshape(model.x['R'] - goal_ori, (9,1))
        cost_expression = transpose(X_pos) @ self.Q_pos @ X_pos + transpose(X_ori) @ self.Q_ori @ X_ori
        return model, cost_expression
    
    def define_mpc(self, env_num):
        mpc = do_mpc.controller.MPC(self.models[env_num])

        # Setup parameters
        setup_mpc = {
            'n_horizon': self.T_horizon,
            'n_robust': 0,
            't_step': self.Ts,
            'state_discretization': 'discrete',
            'store_full_solution': True,
            'open_loop': True,
        }
        mpc.set_param(**setup_mpc)
        mpc.settings.supress_ipopt_output()

        # Define the objective function
        mterm = self.models[env_num].aux['cost']
        lterm = self.models[env_num].aux['cost']
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(thrust=0.1, moments=0.1*np.ones((3,1)))
        
        # Set input bounds
        mpc.bounds['lower', '_u', 'thrust'] = -1.0
        mpc.bounds['upper', '_u', 'thrust'] = 1.0
        mpc.bounds['lower', '_u', 'moments'] = -1.0 * np.ones((3,1))
        mpc.bounds['upper', '_u', 'moments'] = 1.0 * np.ones((3,1))

        mpc.setup()
        return mpc
    
    def define_simulator(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.Ts)
        simulator.setup()
        return simulator
    
    def set_init_state(self):
        """Sets the initial state in all components."""
        self.mpc.x0 = self.x0
        self.simulator.x0 = self.x0
        self.estimator.x0 = self.x0
        self.mpc.u0 = np.array([[-1.0/3.0, 0.0, 0.0, 0]]).T
        self.mpc.set_initial_guess()

    def run_simulation(self):
        """Runs a closed-loop control simulation."""
        x0 = self.x0
        for k in range(self.sim_time):
            u0 = self.mpc.make_step(x0)
            y_next = self.simulator.make_step(u0)
            # y_next = self.simulator.make_step(u0, w0=10**(-4)*np.random.randn(3, 1))  # Optional Additive process noise
            x0 = self.estimator.make_step(y_next)
    
    def initialize(self, init_state, goal):
        self.goal_positions = goal[:,:3]
        self.goal_orientations = goal[:,3:].reshape(-1, 3, 3)
        self.init_states = init_state

        for env in range(self.n_envs):
            self.models[env] = self.define_model(env)
            self.mpcs[env] = self.define_mpc(env)
            self.mpcs[env].x0 = self.init_states[env]
            self.mpcs[env].u0 = np.array([[-1.0/3.0, 0.0, 0.0, 0]]).T
            self.mpcs[env].set_initial_guess()


    def get_action(self, state):
        # return self.mpc.make_step(state)
        actions = np.zeros((self.n_envs, 4, 1))
        for i in range(state.shape[0]):
            actions[i] = self.mpcs[i].make_step(state[i])
        return actions


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    mass = 0.8
    inertia = np.eye(3) * 0.001
    init_x = np.array([[0, 0, 3.0]]).T
    init_v = np.array([[0, 0, 0]]).T
    init_R = np.eye(3).reshape(-1, 1)
    init_omega = np.array([[0, 0, 0]]).T
    goal_pos = np.array([[1.0, 0, 3.0]]).T
    
    x0 = np.zeros((2, 18, 1))
    x0[0] = np.vstack((init_x, init_v, init_R, init_omega))
    x0[1] = np.vstack((np.array([[1, 1, 3.0]]).T, init_v, init_R, init_omega))

    goals = np.zeros((2, 12))
    goals[0] = np.concatenate((np.array([[1.0, 0, 3.0]]).T, np.eye(3).reshape(-1, 1))).squeeze()
    goals[1] = np.concatenate((np.array([[0, 0, 3.0]]).T, np.eye(3).reshape(-1, 1))).squeeze() 


    nmpc = NMPC(2, mass, inertia)
    nmpc.initialize(x0, goals)

    # positions = np.zeros((2, 50, 3))
    times = []
    for i in range(50):
        start = time.time()
        actions = nmpc.get_action(x0)
        end = time.time()
        times.append(end-start)
    
    


    # nmpc = NMPC(mass, inertia, x0, goal_pos)

    # goal = np.concatenate((goal_pos, np.eye(3).reshape(-1, 1)))

    # start_time = time.time()
    # nmpc.initialize(x0, goal)
    # u0 = nmpc.get_action(x0)
    # end_time = time.time()
    # print("Single step time: ", end_time-start_time)

    # # nmpc.run_simulation()
    # pos_list = []
    # action_list = []
    # times = []
    # n_query_steps = 50
    # x0 = nmpc.mpc.x0
    # pos_list.append(x0[:3])
    # for i in range(n_query_steps):
    #     start = time.time()
    #     u0 = nmpc.get_action(x0)
    #     end = time.time()
    #     # print("Step: ", i, "u: ", u0)
    #     y_next = nmpc.simulator.make_step(u0)
    #     x0 = nmpc.estimator.make_step(y_next)
    #     pos_list.append(x0[:3])
    #     action_list.append(u0)
    #     times.append(end-start)
    
    # goal = np.concatenate((np.array([[0.0, 0, 3.0]]).T, np.eye(3).reshape(-1, 1)))
    # nmpc.initialize(x0, goal)
    # for i in range(n_query_steps):
    #     u0 = nmpc.get_action(x0)
    #     y_next = nmpc.simulator.make_step(u0)
    #     x0 = nmpc.estimator.make_step(y_next)
    #     pos_list.append(x0[:3])
    #     action_list.append(u0)

    # pos_list = np.array(pos_list)
    # action_list = np.array(action_list)
    # time_axis = np.arange(pos_list.shape[0]) * nmpc.Ts
    # plt.figure()
    # plt.plot(time_axis, pos_list[:, 0], label='x')
    # plt.plot(time_axis, pos_list[:, 1], label='y')
    # plt.plot(time_axis, pos_list[:, 2], label='z')
    # plt.legend()
    # plt.grid(True)
    
    # plt.figure()
    # time_axis = np.arange(action_list.shape[0]) * nmpc.Ts
    # plt.subplot(211)
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # ax.plot(time_axis, action_list[:,0], label='thrust')
    # ax2.plot(time_axis, ((action_list[:,0] + 1.0)/2.0) * 3.0 * mass * 9.81, label='Force (N)')

    # plt.subplot(212)
    # plt.plot(time_axis, action_list[:,1], label='moment_x')
    # plt.plot(time_axis, action_list[:,2], label='moment_y')
    # plt.plot(time_axis, action_list[:,3], label='moment_z')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    print("Times: \n", times)
    print("Mean time: ", np.mean(times))


    # data = nmpc.mpc.data
    # import code; code.interact(local=locals())
    # fig, ax, graphics = do_mpc.graphics.default_plot(data)
    # graphics.plot_results()
    # plt.show()