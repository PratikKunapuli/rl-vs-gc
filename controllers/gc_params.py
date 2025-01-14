"""
Dictionary of parameters for the GC controller. 
Keys are the task, and values are dictionary of parameters for the GC controller.
"""
gc_params_dict = {
    "Isaac-AerialManipulator-0DOF-SmallArmCOM-V-Hover-v0" : {
        "log_dir": "./baseline_0dof_small_arm_com_v_tuned/",
        "controller_params": {
            "kp_pos_gain_xy": 43.608,
            "kp_pos_gain_z": 33.4451,
            "kd_pos_gain_xy": 10.599,
            "kd_pos_gain_z": 9.617,
            "kp_att_gain_xy": 939.198,
            "kp_att_gain_z": 16.256,
            "kd_att_gain_xy": 36.385,
            "kd_att_gain_z": 3.905,
        },
    },

    "Isaac-AerialManipulator-0DOF-Hover-v0" : {
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
}