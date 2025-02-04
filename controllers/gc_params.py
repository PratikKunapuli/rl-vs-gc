"""
Dictionary of parameters for the GC controller. 
Keys are the task, and values are dictionary of parameters for the GC controller.
"""
gc_params_dict = {
    "Isaac-AerialManipulator-0DOF-SmallArmCOM-Vehicle-Hover-v0" : {
        "log_dir": "./baseline_0dof_small_arm_com_v_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 47.319,
            "kp_pos_gain_z": 33.345,
            "kd_pos_gain_xy": 10.968,
            "kd_pos_gain_z": 11.559,
            "kp_att_gain_xy": 966.220,
            "kp_att_gain_z": 16.881,
            "kd_att_gain_xy": 41.506,
            "kd_att_gain_z": 5.049,
        },
    },

    "Isaac-AerialManipulator-0DOF-SmallArmCOM-Middle-Hover-v0" : {
        "log_dir": "./baseline_0dof_small_arm_com_middle_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 45.850,
            "kp_pos_gain_z": 47.652,
            "kd_pos_gain_xy": 10.626,
            "kd_pos_gain_z": 14.349,
            "kp_att_gain_xy": 960.003,
            "kp_att_gain_z": 17.630,
            "kd_att_gain_xy": 37.064,
            "kd_att_gain_z": 3.042,
        },
    },

    "Isaac-AerialManipulator-0DOF-SmallArmCOM-EndEffector-Hover-v0" : {
        "log_dir": "./baseline_0dof_small_arm_com_ee_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 38.907,
            "kp_pos_gain_z": 27.856,
            "kd_pos_gain_xy": 7.878,
            "kd_pos_gain_z": 9.073,
            "kp_att_gain_xy": 956.455,
            "kp_att_gain_z": 13.233,
            "kd_att_gain_xy": 33.043,
            "kd_att_gain_z": 3.794,
        },
    },

    "Isaac-AerialManipulator-0DOF-SmallArmCOM-Vehicle-TrajectoryTracking-v0" : {
        "log_dir": "./baseline_0dof_small_arm_com_v_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 47.319,
            "kp_pos_gain_z": 33.345,
            "kd_pos_gain_xy": 10.968,
            "kd_pos_gain_z": 11.559,
            "kp_att_gain_xy": 966.220,
            "kp_att_gain_z": 16.881,
            "kd_att_gain_xy": 41.506,
            "kd_att_gain_z": 5.049,
            "feed_forward": True,
        },
    },

    "Isaac-AerialManipulator-0DOF-SmallArmCOM-Middle-TrajectoryTracking-v0" : {
        "log_dir": "./baseline_0dof_small_arm_com_middle_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 45.850,
            "kp_pos_gain_z": 47.652,
            "kd_pos_gain_xy": 10.626,
            "kd_pos_gain_z": 14.349,
            "kp_att_gain_xy": 960.003,
            "kp_att_gain_z": 17.630,
            "kd_att_gain_xy": 37.064,
            "kd_att_gain_z": 3.042,
            "feed_forward": True,
        },
    },

    "Isaac-AerialManipulator-0DOF-SmallArmCOM-EndEffector-TrajectoryTracking-v0" : {
        "log_dir": "./baseline_0dof_small_arm_com_ee_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 38.907,
            "kp_pos_gain_z": 27.856,
            "kd_pos_gain_xy": 7.878,
            "kd_pos_gain_z": 9.073,
            "kp_att_gain_xy": 956.455,
            "kp_att_gain_z": 13.233,
            "kd_att_gain_xy": 33.043,
            "kd_att_gain_z": 3.794,
            "feed_forward": True,
        },
    },

    # "Isaac-AerialManipulator-0DOF-Hover-v0" : {
    #     "log_dir": "./baseline_0dof_ee_reward_tune/",
    #     "controller_params": {
    #         "kp_pos_gain_xy": 43.507,
    #         "kp_pos_gain_z": 24.167,
    #         "kd_pos_gain_xy": 9.129,
    #         "kd_pos_gain_z": 6.081,
    #         "kp_att_gain_xy": 998.777,
    #         "kp_att_gain_z": 18.230,
    #         "kd_att_gain_xy": 47.821,
    #         "kd_att_gain_z": 8.818,
    #     },
    # },

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Hover" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_no_ff/",
        "controller_params": {
            "kp_pos_gain_xy": 34.004,
            "kp_pos_gain_z": 41.063,
            "kd_pos_gain_xy": 8.978,
            "kd_pos_gain_z": 12.577,
            "kp_att_gain_xy": 920.832,
            "kp_att_gain_z": 10.302,
            "kd_att_gain_xy": 42.157,
            "kd_att_gain_z": 5.042,
            "feed_forward": False,
        },
    },

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Hover-FF" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_with_ff_hover/",
        "controller_params": {
            "kp_pos_gain_xy": 34.004,
            "kp_pos_gain_z": 41.063,
            "kd_pos_gain_xy": 8.978,
            "kd_pos_gain_z": 12.577,
            "kp_att_gain_xy": 920.832,
            "kp_att_gain_z": 10.302,
            "kd_att_gain_xy": 42.157,
            "kd_att_gain_z": 5.042,
            "feed_forward": True,
        },
    },

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_with_ff/",
        "controller_params": {
            "kp_pos_gain_xy": 43.507,
            "kp_pos_gain_z": 24.167,
            "kd_pos_gain_xy": 9.129,
            "kd_pos_gain_z": 6.081,
            "kp_att_gain_xy": 998.777,
            "kp_att_gain_z": 18.230,
            "kd_att_gain_xy": 47.821,
            "kd_att_gain_z": 8.818,
            "feed_forward": True,
        },
    },

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Integral" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_pid/",
        "controller_params": {
            "kp_pos_gain_xy": 47.080,
            "kp_pos_gain_z": 37.972,
            "kd_pos_gain_xy": 3.384,
            "kd_pos_gain_z": 3.786,
            "kp_att_gain_xy": 703.522,
            "kp_att_gain_z": 17.845,
            "kd_att_gain_xy": 22.764,
            "kd_att_gain_z": 4.021,
            "ki_pos_gain_xy": 4.173,
            "ki_pos_gain_z": 18.040,
            "ki_att_gain_xy": 149.806,
            "ki_att_gain_z": 3.992,
            "use_integral": True,
            "feed_forward": False,
        },
    },

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Hand-Tune" : {
        "log_dir": "./baseline_0dof_hand_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 45.0,
            "kp_pos_gain_z": 20.0,
            "kd_pos_gain_xy": 8.0,
            "kd_pos_gain_z": 10.0,
            "kp_att_gain_xy": 900.0,
            "kp_att_gain_z": 15.0,
            "kd_att_gain_xy": 50.0,
            "kd_att_gain_z": 10.0,
            "feed_forward": False,
        },
    },

    "Isaac-AerialManipulator-QuadOnly-TrajectoryTracking-v0" : {
        "log_dir": "./baseline_quad_only_reward_tune/",
        "controller_params": {
            "kp_pos_gain_xy": 44.142,
            "kp_pos_gain_z": 26.112,
            "kd_pos_gain_xy": 4.255,
            "kd_pos_gain_z": 11.824,
            "kp_att_gain_xy": 710.390,
            "kp_att_gain_z": 17.223,
            "kd_att_gain_xy": 34.800,
            "kd_att_gain_z": 9.672,
            "feed_forward": True,
        },
    },

    "Isaac-AerialManipulator-0DOF-LongArm-TrajectoryTracking-v0" : {
        "log_dir": "./baseline_0dof_long_arm_com_middle_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 16.954,
            "kp_pos_gain_z": 35.462,
            "kd_pos_gain_xy": 5.877,
            "kd_pos_gain_z": 13.473,
            "kp_att_gain_xy": 982.229,
            "kp_att_gain_z": 19.315,
            "kd_att_gain_xy": 38.180,
            "kd_att_gain_z": 8.487,
            "feed_forward": True,
        },
    },

    "Isaac-Crazyflie-0DOF-Hover-v0" : {
        "log_dir": "./baseline_cf_0dof/",
        "controller_params": {
            "kp_pos_gain_xy": 6.5,
            "kp_pos_gain_z": 15.0,
            "kd_pos_gain_xy": 4.0,
            "kd_pos_gain_z": 9.0,
            "kp_att_gain_xy": 544.0,
            "kp_att_gain_z": 544.0,
            "kd_att_gain_xy": 46.64,
            "kd_att_gain_z": 46.64,
        },
    },


    "Isaac-AerialManipulator-0DOF-BallCatch-v0": {
        "log_dir": "./baseline_0dof_ee_reward_tune/",
        "controller_params": {
            "kp_pos_gain_xy": 43.507,
            "kp_pos_gain_z": 24.167,
            "kd_pos_gain_xy": 9.129,
            "kd_pos_gain_z": 6.081,
            "kp_att_gain_xy": 998.777,
            "kp_att_gain_z": 18.230,
            "kd_att_gain_xy": 47.821,
            "kd_att_gain_z": 8.818,
        },
    }
}