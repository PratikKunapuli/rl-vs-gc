"""
Dictionary of parameters for the GC controller. 
Keys are the task, and values are dictionary of parameters for the GC controller.
"""
gc_params_dict = {
    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Hover" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_no_ff_hover/",
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

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-NoFF" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_no_ff_traj/",
        "controller_params": {
            "kp_pos_gain_xy": 41.564,
            "kp_pos_gain_z": 43.588,
            "kd_pos_gain_xy": 3.531,
            "kd_pos_gain_z": 3.661,
            "kp_att_gain_xy": 960.076,
            "kp_att_gain_z": 18.592,
            "kd_att_gain_xy": 40.740,
            "kd_att_gain_z": 5.729,
            "feed_forward": False,
        },
    },

    # Old Parameters
    # "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Hover-FF" : {
    #     "log_dir": "./baseline_0dof_ee_reward_tune_with_ff_hover/",
    #     "controller_params": {
    #         "kp_pos_gain_xy": 17.601,
    #         "kp_pos_gain_z": 28.995,
    #         "kd_pos_gain_xy": 5.576,
    #         "kd_pos_gain_z": 12.412,
    #         "kp_att_gain_xy": 966.060,
    #         "kp_att_gain_z": 19.171,
    #         "kd_att_gain_xy": 33.214,
    #         "kd_att_gain_z": 9.086,
    #         "feed_forward": True,
    #     },
    # },

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Hover-FF" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_with_ff_hover/",
        "controller_params": {
            "kp_pos_gain_xy": 15.213,
            "kp_pos_gain_z": 11.857,
            "kd_pos_gain_xy": 4.316,
            "kd_pos_gain_z": 4.051,
            "kp_att_gain_xy": 835.763,
            "kp_att_gain_z": 15.046,
            "kd_att_gain_xy": 39.350,
            "kd_att_gain_z": 6.979,
            "feed_forward": True,
        },
    },

    # Old parameters
    # "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0" : {
    #     "log_dir": "./baseline_0dof_ee_reward_tune_with_ff/",
    #     "controller_params": {
    #         "kp_pos_gain_xy": 43.507,
    #         "kp_pos_gain_z": 24.167,
    #         "kd_pos_gain_xy": 9.129,
    #         "kd_pos_gain_z": 6.081,
    #         "kp_att_gain_xy": 998.777,
    #         "kp_att_gain_z": 18.230,
    #         "kd_att_gain_xy": 47.821,
    #         "kd_att_gain_z": 8.818,
    #         "feed_forward": True,
    #     },
    # },

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_with_ff/",
        "controller_params": {
            "kp_pos_gain_xy": 24.733,
            "kp_pos_gain_z": 24.364,
            "kd_pos_gain_xy": 4.884,
            "kd_pos_gain_z": 8.147,
            "kp_att_gain_xy": 920.567,
            "kp_att_gain_z": 19.172,
            "kd_att_gain_xy": 37.265,
            "kd_att_gain_z": 6.960,
            "feed_forward": True,
        },
    },

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Hover-Integral" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_pid_hover/",
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

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Traj-Integral" : {
        "log_dir": "./baseline_0dof_ee_reward_tune_pid_traj/",
        "controller_params": {
            "kp_pos_gain_xy": 38.435,
            "kp_pos_gain_z": 31.333,
            "kd_pos_gain_xy": 4.136,
            "kd_pos_gain_z": 3.052,
            "kp_att_gain_xy": 962.850,
            "kp_att_gain_z": 18.503,
            "kd_att_gain_xy": 29.785,
            "kd_att_gain_z": 4.879,
            "ki_pos_gain_xy": 2.831,
            "ki_pos_gain_z": 1.224,
            "ki_att_gain_xy": 151.872,
            "ki_att_gain_z": 5.729,
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

    "Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0-Hand-Tune-FF" : {
        "log_dir": "./baseline_0dof_hand_tuned_with_ff/",
        "controller_params": {
            "kp_pos_gain_xy": 45.0,
            "kp_pos_gain_z": 20.0,
            "kd_pos_gain_xy": 8.0,
            "kd_pos_gain_z": 10.0,
            "kp_att_gain_xy": 900.0,
            "kp_att_gain_z": 15.0,
            "kd_att_gain_xy": 50.0,
            "kd_att_gain_z": 10.0,
            "feed_forward": True,
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

    "Isaac-AerialManipulator-QuadOnly-TrajectoryTracking-v0-DR-20" : {
        "log_dir": "./baseline_quad_only_reward_tune_dr_20/",
        "controller_params": {
            "kp_pos_gain_xy": 48.706,
            "kp_pos_gain_z": 37.455,
            "kd_pos_gain_xy": 4.945,
            "kd_pos_gain_z": 8.774,
            "kp_att_gain_xy": 890.717,
            "kp_att_gain_z": 17.624,
            "kd_att_gain_xy": 45.703,
            "kd_att_gain_z": 6.442,
            "feed_forward": True,
        },
    },

    "Isaac-AerialManipulator-QuadOnly-TrajectoryTracking-v0-DR-40" : {
        "log_dir": "./baseline_quad_only_reward_tune_dr_40/",
        "controller_params": {
            "kp_pos_gain_xy": 46.408,
            "kp_pos_gain_z": 42.955,
            "kd_pos_gain_xy": 4.655,
            "kd_pos_gain_z": 7.979,
            "kp_att_gain_xy": 594.617,
            "kp_att_gain_z": 12.044,
            "kd_att_gain_xy": 17.815,
            "kd_att_gain_z": 3.694,
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
    },


    ################################################################################
    "Isaac-Crazyflie-CTBR-Hover-v0": {
        "log_dir": "./baseline_cf_ctbm/",
        "controller_params": {
            "feed_forward": True,
            "control_mode": "CTBM",
            "kp_pos_gain_xy": 14.078,
            "kp_pos_gain_z": 10.447,
            "kd_pos_gain_xy": 3.478,
            "kd_pos_gain_z": 3.509,
            "kp_att_gain_xy": 955.643,
            "kp_att_gain_z": 18.182,
            "kd_att_gain_xy": 61.529,
            "kd_att_gain_z": 5.044,
        }
    },

    "Isaac-Crazyflie-CTBR-Hover-v0-DR": {
        "log_dir": "./baseline_cf_ctbm_DR/",
        "controller_params": {
            "feed_forward": True,
            "control_mode": "CTBM",
            "kp_pos_gain_xy": 13.842,
            "kp_pos_gain_z": 23.479,
            "kd_pos_gain_xy": 4.493,
            "kd_pos_gain_z": 6.897,
            "kp_att_gain_xy": 701.511,
            "kp_att_gain_z": 19.116,
            "kd_att_gain_xy": 41.828,
            "kd_att_gain_z": 8.992,
        },
    },

    "Isaac-Crazyflie-CTBR-Hover-v0-DR-20ms": {
        "log_dir": "./baseline_cf_ctbm_DR_20ms/",
        "controller_params": {
            "feed_forward": True,
            "control_mode": "CTBM",
            "kp_pos_gain_xy": 12.475,
            "kp_pos_gain_z": 9.819,
            "kd_pos_gain_xy": 3.664,
            "kd_pos_gain_z": 13.285,
            "kp_att_gain_xy": 320.810,
            "kp_att_gain_z": 19.943,
            "kd_att_gain_xy": 27.181,
            "kd_att_gain_z": 7.515,
        },
    },

    "Isaac-Crazyflie-CTBR-Hover-v0-DR-40ms": {
        "log_dir": "./baseline_cf_ctbm_DR_40ms/",
        "controller_params": {
            "feed_forward": True,
            "control_mode": "CTBM",
            "kp_pos_gain_xy": 5.657,
            "kp_pos_gain_z": 8.990,
            "kd_pos_gain_xy": 1.436,
            "kd_pos_gain_z": 8.396,
            "kp_att_gain_xy": 125.582,
            "kp_att_gain_z": 15.348,
            "kd_att_gain_xy": 11.992,
            "kd_att_gain_z": 8.995,
        },
    },

    "Isaac-Crazyflie-CTBR-Hover-v0-CTBR-NoFF": {
        "log_dir": "./baseline_cf_ctbr/",
        "controller_params": {
            "feed_forward": False,
            "control_mode": "CTBR",
            # Original
            # "kp_pos_gain_xy": 9.195,
            # "kp_pos_gain_z": 27.006,
            # "kd_pos_gain_xy": 3.780,
            # "kd_pos_gain_z": 10.797,
            # "kp_att_gain_xy": 977.197,
            # "kp_att_gain_z": 18.357,
            # "kd_att_gain_xy": 22.430,
            # "kd_att_gain_z": 0.014,
            # Tau=0.01
            # "kp_pos_gain_xy": 13.154,
            # "kp_pos_gain_z": 23.166,
            # "kd_pos_gain_xy": 5.392,
            # "kd_pos_gain_z": 6.801,
            # "kp_att_gain_xy": 870.353,
            # "kp_att_gain_z": 10.002,
            # "kd_att_gain_xy": 37.353,
            # "kd_att_gain_z": 8.887,
            # Tau=0.017
            "kp_pos_gain_xy": 9.720,
            "kp_pos_gain_z": 38.171,
            "kd_pos_gain_xy": 5.373,
            "kd_pos_gain_z": 14.012,
            "kp_att_gain_xy": 982.414,
            "kp_att_gain_z": 10.412,
            "kd_att_gain_xy": 38.580,
            "kd_att_gain_z": 8.080,
        }
    },

    "Isaac-Crazyflie-CTBR-Hover-v0-CTBR-FF": {
        "log_dir": "./baseline_cf_ctbr/",
        "controller_params": {
            "feed_forward": False,
            "control_mode": "CTBR",
            "kp_pos_gain_xy": 14.078,
            "kp_pos_gain_z": 10.447,
            "kd_pos_gain_xy": 3.478,
            "kd_pos_gain_z": 3.509,
            "kp_att_gain_xy": 955.643,
            "kp_att_gain_z": 18.182,
            "kd_att_gain_xy": 61.529,
            "kd_att_gain_z": 5.044,
        }
    },
}