from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from soarm101_lab.assets import SO101_FOLLOWER_CFG, SO101_WORLD_POS
from soarm101_lab.utils.constant import BIN_USD_PATH
from isaaclab.sensors import FrameTransformerCfg

# so101 world pos (0.0, 0.0, 0.6) 기준으로 테이블과 픽존 위치 설정
TABLE_Z = 0.6  # 테이블 높이 (z 좌표) [m]

PICK_ZONE_SIZE = 0.15  # Pick Zone 크기 (정사각형의 한 변 길이) [m]
PICK_ZONE_HALF = PICK_ZONE_SIZE / 2.0
PICK_ZONE_BORDER = 0.002  # Pick Zone 선 두께 [m]
PICK_ZONE_CENTER = (
    SO101_WORLD_POS[0],
    SO101_WORLD_POS[1] + 0.19 + PICK_ZONE_HALF, 
    TABLE_Z
    )

BIN_A_POS = (
    SO101_WORLD_POS[0] + 0.08 + PICK_ZONE_HALF,
    SO101_WORLD_POS[1] + 0.315,
    TABLE_Z,
)

BIN_B_POS = (
    SO101_WORLD_POS[0] + 0.08 + PICK_ZONE_HALF,
    SO101_WORLD_POS[1] + 0.225,
    TABLE_Z,
)

class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Pick and Place 작업용 Scene 설정"""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )
    
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    

    # 로봇 (다른 파일에서 import)
    robot = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/tool0", name="tool0"
            ),
        ],
    )

    # 테이블
    # table = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=TABLE_USD_PATH,
    #         scale=(0.01, 0.02, 0.01),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.0)),
        
    # )
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.6),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.9),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 0.3)),
    )
    # pick_zone(15cm x 15cm 크기의 영역, 로봇이 큐브를 집는 위치) - 테이블 위에 위치하도록 설정
    pick_zone_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PickZoneTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.002, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                PICK_ZONE_CENTER[0],
                PICK_ZONE_CENTER[1] + 0.074,
                PICK_ZONE_CENTER[2],
            ),
        ),
    )

    pick_zone_bottom = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PickZoneBottom",
        spawn=pick_zone_top.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                PICK_ZONE_CENTER[0],
                PICK_ZONE_CENTER[1] - 0.074,
                PICK_ZONE_CENTER[2],
            ),
        ),
    )

    pick_zone_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PickZoneLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.002, 0.15, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                PICK_ZONE_CENTER[0] - 0.074,
                PICK_ZONE_CENTER[1],
                PICK_ZONE_CENTER[2],
            ),
        ),
    )
    pick_zone_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PickZoneRight",
        spawn=pick_zone_left.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                PICK_ZONE_CENTER[0] + 0.074,
                PICK_ZONE_CENTER[1],
                PICK_ZONE_CENTER[2],
            ),
        ),
    )

    # Bin 2개 (PickPlace 작업에서 Place할 위치로 사용) - Bin_A, Bin_B
    bin_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bin_A",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BIN_USD_PATH,
            scale=(0.7, 0.7, 0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(BIN_A_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        
    )
    
    bin_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bin_B",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BIN_USD_PATH,
            scale=(0.7, 0.7, 0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(BIN_B_POS, rot=(0.0, 0.0, 0.0, 1.0)),
    )
    
    
    # side view camera / pos:(-0.185, 0.265, 1.2)
    camera_sideview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_SideView",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            #pos=(-0.185, PICK_ZONE_CENTER[1], 0.77),
            pos=(-0.135, PICK_ZONE_CENTER[1], 0.745),
            #rot=(0.61845, 0.34281, -0.34281, -0.61845),    # (W, X, Y, Z) 58도 아래로 내려다봄
            rot=(0.62932, 0.32102, -0.32102, -0.62932),     # 54
            convention="opengl",
        ),
    )


    # Pick 대상 큐브들 (RGB로 색상 구분)
    cube_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_Red",
        spawn=sim_utils.CuboidCfg(
            size=(0.024, 0.024, 0.024),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.05,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # 빨간색
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PICK_ZONE_CENTER[0], PICK_ZONE_CENTER[1], PICK_ZONE_CENTER[2]+0.01),
        ),
    )

    cube_green = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_Green",
        spawn=sim_utils.CuboidCfg(
            size=(0.024, 0.024, 0.024),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.05,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # 초록색
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PICK_ZONE_CENTER[0]- 0.075, PICK_ZONE_CENTER[1]+ 0.075, PICK_ZONE_CENTER[2]+0.01),
        ),
    )

    cube_blue = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_Blue",
        spawn=sim_utils.CuboidCfg(
            size=(0.024, 0.024, 0.024),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.05,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # 파란색
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PICK_ZONE_CENTER[0]+ 0.075, PICK_ZONE_CENTER[1]- 0.075, PICK_ZONE_CENTER[2]+0.01),
        ),
    )