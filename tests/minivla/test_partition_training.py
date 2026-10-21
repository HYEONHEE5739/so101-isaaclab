import json
import numpy as np
import pytest
from pathlib import Path
from test_lerobot_direct import lerobot_fixture

def test_measured_limits_require_the_recorded_urdf_hash(tmp_path):
    import hashlib
    from soarm101_lab.workflow.episodes import training_coordinates
    from soarm101_lab.so101_joint_mapping import RobotCoordinates
    urdf=tmp_path/'robot.urdf'
    urdf.write_text('<robot><joint name="gripper"><limit lower="-0.174533" upper="1.74533"/></joint></robot>')
    a={'provenance':{'canonical_coordinates':RobotCoordinates().metadata()},'coordinates':{'simulation_model':{
        'robot_model':{'urdf':'robot.urdf','urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest()}}}}
    coords=training_coordinates(a,tmp_path)
    assert coords.metadata()['measured_gripper_limits_rad']==[-.174533,1.74533]
    urdf.write_text(urdf.read_text().replace('-0.174533','-1.0'))
    with pytest.raises(ValueError,match='hash'):training_coordinates(a,tmp_path)
