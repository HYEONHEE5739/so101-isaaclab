from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from soarm101_lab.utils.constant import ASSETS_ROOT

SO101_FOLLOWER_ASSET_PATH = Path(ASSETS_ROOT) / "SO101" / "usd" / "so101_isaaclab.usd"
SO101_WORLD_POS = (0.0, 0.0, 0.6) 

# joint limit written in USD (degree)
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}

SO101_FOLLOWER_INITIAL_JOINT_POS = {
    "shoulder_pan": 0.0,        # 0 degree
    "shoulder_lift": -1.7453,   # -100 degree
    "elbow_flex": 1.5708,       # 90 degree
    "wrist_flex": 1.2217,       # 70 degree
    "wrist_roll": 0.0,          # 0 degree
    "gripper": 0.0,             # 0 degree (-10 degree (closed) ~ 100 degree (open))        # 0 degree
}


SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_FOLLOWER_ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.02079, 0.04, SO101_WORLD_POS[2]-0.03),    # (x, y, z) in world frame [m]
        rot=(0.0, 0.0, 0.0, 1.0),  # quaternion (x, y, z, w)
        joint_pos=SO101_FOLLOWER_INITIAL_JOINT_POS,
    ),
    actuators={
        # ROTATION (Gear: 1/191, Torque: 34.4 N-m)
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan"],
            effort_limit_sim=30,
            stiffness=55,        
            damping=0.7,         
        ),
        
        # PITCH (Gear: 1/345, Torque: 62.1 N-m - HIGHEST)
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift"],
            effort_limit_sim=30,
            stiffness=30,        
            damping=0.8,         
        ),
        
        # ELBOW (Gear: 1/191, Torque: 34.4 N-m)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_flex"],
            effort_limit_sim=30,
            stiffness=25,        
            damping=0.7,         
        ),
        
        # WRIST PITCH (Gear: 1/147, Torque: 26.5 N-m)
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex"],
            effort_limit_sim=30,
            stiffness=12,        
            damping=0.5,         
        ),
        
        # WRIST ROLL (Gear: 1/147, Torque: 26.5 N-m)
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["wrist_roll"],
            effort_limit_sim=30,
            stiffness=7,         
            damping=0.5,         
        ),
        
        # GRIPPER (Gear: 1/147, Torque: 26.5 N-m)
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=30,
            stiffness=4,         
            damping=0.3,         
        ),
    },
)