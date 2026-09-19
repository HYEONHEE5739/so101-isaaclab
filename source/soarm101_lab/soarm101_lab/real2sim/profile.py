"""Versioned metric profile. T_A_B maps B coordinates into A (metres, column vectors)."""

import copy
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = "so101.real2sim/2"
SOURCES = {"user_measured", "model_asset", "derived", "numerically_fitted", "provisional", "unmeasured"}
CUP_FIELDS = (
    "top_outer_diameter_m", "wall_thickness_m", "height_m",
    "bottom_outer_diameter_m", "bottom_inner_diameter_m", "bottom_thickness_m", "taper_profile",
)
ROBOT_ASSET = {
    "source": "model_asset",
    "urdf": "assets/SO101/urdf/so101_isaaclab.urdf",
    "usd": "assets/SO101/usd/so101_isaaclab.usd",
}


def provenance(node, key):
    source = node.get("provenance", {}).get(key)
    if source not in SOURCES:
        raise ValueError(f"Explicit field provenance required: {key}")
    if (node.get(key) is None) != (source == "unmeasured"):
        raise ValueError(f"{key}: unknown values must be null/unmeasured; numeric seeds must be provisional")
    return source


def scene_objects(p):
    """Only explicitly enabled objects are materialized; no invented geometry."""
    return {n: o for n, o in [("table", p["table"]), *p["objects"].items()] if o["render_enabled"]}


def require_metric_geometry(p, frame):
    """A rendering proxy is never a measured landmark constraint."""
    o = p["table"] if frame == "table" else p["objects"][frame.removeprefix("object:")]
    if o["shape"] in ("cup", "cup_proxy") or o.get("dimensions_m") is None:
        raise ValueError(f"{frame}: incomplete/proxy geometry is excluded from metric fitting")
    if provenance(o, "dimensions_m") in ("provisional", "unmeasured"):
        raise ValueError(f"{frame}: provisional geometry is excluded from metric fitting")
    if o["T_workspace"] is None:
        raise ValueError(f"{frame}: provide an explicit provisional or measured pose before fitting")


def mark_fitted(p, group):
    if group in ("side", "wrist"):
        node, key = p["cameras"][group], "T_parent_camera"
    elif group in ("robot_base", "workspace"):
        node, key = p[group], "T_world"
    else:
        node = p["table"] if group == "table" else p["objects"][group.removeprefix("object:")]
        key = "T_workspace"
    node["provenance"][key] = "numerically_fitted"


def derived_measurements(p):
    """Computed on demand, never stored as independent measurements."""
    result = {}
    for name, o in p["objects"].items():
        if o["shape"] == "cup" and o["top_outer_diameter_m"] is not None and o["wall_thickness_m"] is not None:
            result[name + ".top_inner_diameter_m"] = {
                "value": o["top_outer_diameter_m"] - 2 * o["wall_thickness_m"],
                "source": "derived",
                "formula": "top_outer_diameter_m - 2 * wall_thickness_m",
            }
    for name, relation in p.get("relations", {}).items():
        a, b = (p["objects"][n] for n in relation["objects"])
        distance = relation["center_to_center_distance_m"]
        diameter_a, diameter_b = a["top_outer_diameter_m"], b["top_outer_diameter_m"]
        if any(v is None for v in (distance, diameter_a, diameter_b)):
            continue
        result[name + ".top_edge_gap_m"] = {
            "value": distance - diameter_a / 2 - diameter_b / 2,
            "source": "derived",
            "formula": "center_to_center_distance_m - diameter_a / 2 - diameter_b / 2",
        }
    return result


def upgrade_legacy(p):
    """Conservative in-memory migration. Never rewrite old revisions or trust a blanket label."""
    if p.get("schema") != "so101.real2sim/1":
        return p
    q = copy.deepcopy(p)
    q["schema"] = SCHEMA
    q.pop("geometry_source", None)
    q["migration"] = {
        "parent_hash": digest(p),
        "note": "Legacy numbers retained as provisional; measurement/calibration evidence must be reviewed.",
    }
    q["revision"] = {"kind": "legacy_import", "parent": digest(p)}
    q["robot_asset"] = dict(ROBOT_ASSET)
    for name in ("robot_base", "workspace"):
        q[name]["provenance"] = {"T_world": "provisional"}
        q[name]["calibrated"] = False
    if "size_m" in q["workspace"]:
        q["workspace"]["provenance"]["size_m"] = "provisional"
    for c in q["cameras"].values():
        c["provenance"] = {k: "provisional" for k in ("T_parent_camera", "K", "distortion", "resolution")}
        c["intrinsics_calibrated"] = c["extrinsics_calibrated"] = False
        c["source"] = "Legacy render initialization; physical camera ground truth is unknown"
    for o in [q["table"], *q["objects"].values()]:
        o["render_enabled"] = True
        o["provenance"] = {k: "provisional" for k in ("dimensions_m", "T_workspace")}
        o["appearance"]["provenance"] = {k: "provisional" for k in ("color", "roughness")}
        if o["shape"] == "cup":
            o["shape"] = "cup_proxy"
            o["provenance"]["wall_m"] = "provisional"
            o["approximation"] = (
                "Legacy cylindrical preview: bottom radius and bottom thickness are NOT measured; "
                "excluded from fitting."
            )
    q["appearance"]["provenance"] = {"light_intensity": "provisional"}
    return q


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def transform(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Transform must be finite 4x4")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous transform")
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(t[:3, :3]), 1, atol=1e-5
    ):
        raise ValueError("Rotation must be right-handed orthonormal")
    return t


def pose6(t):
    t = transform(t)
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_rotvec()]


def matrix(x):
    t = np.eye(4)
    t[:3, 3] = x[:3]
    t[:3, :3] = Rotation.from_rotvec(x[3:]).as_matrix()
    return t


def pose(t):
    t = transform(t)
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return tuple(t[:3, 3]), tuple(q[[3, 0, 1, 2]])


def validate(p):
    if p.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}; use load() to read legacy profiles conservatively")
    if "geometry_source" in p:
        raise ValueError("Whole-profile geometry_source is forbidden; use field provenance")
    if p.get("robot_asset") != ROBOT_ASSET:
        raise ValueError("Robot geometry must reference the existing model_asset")
    if p.get("units") != "metres" or p.get("optical_axes") != "x_right_y_down_z_forward":
        raise ValueError("Wrong coordinate convention")
    for name in ["robot_base", "workspace"]:
        source = provenance(p[name], "T_world")
        transform(p[name]["T_world"])
        if p[name].get("calibrated") and source in ("provisional", "unmeasured"):
            raise ValueError(f"{name}: provisional transform cannot be calibrated")
    if "size_m" in p["workspace"]:
        provenance(p["workspace"], "size_m")
        size = np.asarray(p["workspace"]["size_m"], float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.min(size) <= 0:
            raise ValueError("Workspace marker size must be positive XY")
    for n in ["side", "wrist"]:
        c = p["cameras"][n]
        for key in ("T_parent_camera", "K", "distortion", "resolution"):
            provenance(c, key)
        for flag, keys in (
            ("intrinsics_calibrated", ("K", "distortion")),
            ("extrinsics_calibrated", ("T_parent_camera",)),
        ):
            if c.get(flag) and any(c["provenance"][k] in ("provisional", "unmeasured") for k in keys):
                raise ValueError(f"{n}: provisional camera parameters cannot be calibrated")
        transform(c["T_parent_camera"])
        k = np.asarray(c["K"], float)
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or abs(k[0, 1]) + abs(k[1, 0]) > 1e-9
        ):
            raise ValueError("Invalid zero-skew camera K")
        if len(c["resolution"]) != 2 or any(type(x) != int or x < 16 for x in c["resolution"]):
            raise ValueError("Invalid image dimensions")
        if len(c["distortion"]) not in (4, 5, 8) or not np.isfinite(c["distortion"]).all():
            raise ValueError("Use OpenCV Brown distortion (4/5/8)")
        if c["parent"] != ("world" if n == "side" else "gripper"):
            raise ValueError("Camera parent must match inspected SO101 mount")
    if "table" in p["objects"]:
        raise ValueError("table is a reserved object name")
    for n, o in [("table", p["table"]), *p["objects"].items()]:
        if not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", n):
            raise ValueError("USD-safe object name required")
        if o["shape"] not in ("box", "cylinder", "cup", "cup_proxy"):
            raise ValueError("Supported shapes: box, cylinder, partial cup, cup_proxy")
        if type(o.get("render_enabled")) is not bool:
            raise ValueError("Explicit render_enabled required")
        provenance(o, "T_workspace")
        if o["T_workspace"] is not None:
            transform(o["T_workspace"])
        if o["render_enabled"] and o["T_workspace"] is None:
            raise ValueError("Rendering requires an explicit pose, which may be provisional")
        a = o["appearance"]
        for key in ("color", "roughness"):
            provenance(a, key)
        if len(a["color"]) != 3 or not all(0 <= v <= 1 for v in a["color"]) or not 0 <= a["roughness"] <= 1:
            raise ValueError("Appearance must be in [0,1]")
        if o["shape"] == "cup":
            if "dimensions_m" in o or "wall_m" in o or "top_inner_diameter_m" in o:
                raise ValueError("Cup uses explicit top/bottom measurements; inner top diameter is derived on demand")
            for key in CUP_FIELDS:
                provenance(o, key)
                value = o[key]
                if value is not None and key != "taper_profile" and (not math.isfinite(value) or value <= 0):
                    raise ValueError(f"Invalid cup measurement: {key}")
            if o["wall_thickness_m"] is not None and o["top_outer_diameter_m"] is not None:
                if 2 * o["wall_thickness_m"] >= o["top_outer_diameter_m"]:
                    raise ValueError("Cup wall must leave an opening")
            if o["render_enabled"]:
                raise ValueError("Partial cup geometry is omitted from rendering; no cylindrical assumption is allowed")
            continue
        provenance(o, "dimensions_m")
        if o["dimensions_m"] is None:
            if o["render_enabled"]:
                raise ValueError("Unknown dimensions cannot be rendered")
            continue
        dims = np.asarray(o["dimensions_m"], float)
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.min(dims) <= 0:
            raise ValueError("Dimensions must be positive XYZ")
        if o["shape"] in ("cup_proxy", "cylinder") and not np.isclose(dims[0], dims[1]):
            raise ValueError("Cylinder/cup X and Y diameters must match")
        if o["shape"] == "cup_proxy":
            if "bottom_diameter_m" in o:
                if provenance(o, "bottom_diameter_m") != "provisional" or not 2 * o["wall_m"] < o["bottom_diameter_m"] <= dims[0]:
                    raise ValueError("Cup visual bottom diameter must be a provisional positive taper")
            if "print_texture" in o:
                tex = o["print_texture"]
                if tex.get("source") != "provisional" or not Path(tex["path"]).is_file():
                    raise ValueError("Cup texture requires existing provisional asset")
                uv = np.asarray(tex["uv_rect"], float)
                if uv.shape != (4,) or not np.isfinite(uv).all() or not (0 <= uv[0] < uv[2] <= 1 and 0 <= uv[1] < uv[3] <= 1):
                    raise ValueError("Invalid cup print UV rectangle")
            if (
                provenance(o, "dimensions_m") != "provisional"
                or provenance(o, "wall_m") != "provisional"
                or not o.get("approximation")
            ):
                raise ValueError("Cup proxy geometry must remain explicitly provisional")
            if not 0 < float(o["wall_m"]) < min(dims[0] / 2, dims[2]):
                raise ValueError("Invalid cup preview wall")
    for relation in p.get("relations", {}).values():
        provenance(relation, "center_to_center_distance_m")
        names = relation["objects"]
        if (
            len(names) != 2 or len(set(names)) != 2
            or any(n not in p["objects"] or p["objects"][n]["shape"] != "cup" for n in names)
        ):
            raise ValueError("Cup spacing must reference two distinct cups")
        if not math.isfinite(relation["center_to_center_distance_m"]) or relation["center_to_center_distance_m"] <= 0:
            raise ValueError("Invalid cup center spacing")
        if "top_edge_gap_m" in relation:
            raise ValueError("Cup edge gap is derived on demand, not an independent measurement")
    if "wrist_mount" in p:
        mount = p["wrist_mount"]
        if mount.get("parent") != "gripper" or mount.get("source") != "provisional":
            raise ValueError("Wrist mount must be provisional and attached to gripper")
        child = copy.deepcopy(p)
        child.pop("wrist_mount")
        child["objects"] = mount["parts"]
        child["relations"] = {}
        validate(child)
    provenance(p["appearance"], "light_intensity")
    if "robot_print_color" in p["appearance"]:
        provenance(p["appearance"], "robot_print_color")
        color = p["appearance"]["robot_print_color"]
        if len(color) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in color):
            raise ValueError("Invalid robot print color")
    if not math.isfinite(p["appearance"]["light_intensity"]) or p["appearance"]["light_intensity"] < 0:
        raise ValueError("Invalid light intensity")
    json.dumps(p, allow_nan=False)
    return p


def load(path):
    return validate(upgrade_legacy(json.loads(Path(path).read_text())))


def get_group(p, group):
    if group == "side":
        return p["cameras"]["side"]["T_parent_camera"]
    if group == "wrist":
        return p["cameras"]["wrist"]["T_parent_camera"]
    if group in ("robot_base", "workspace"):
        return p[group]["T_world"]
    if group == "table":
        return p["table"]["T_workspace"]
    if group.startswith("object:"):
        return p["objects"][group[7:]]["T_workspace"]
    raise ValueError("Unknown parameter group: " + group)


def set_group(p, group, t):
    t = transform(t).tolist()
    if group in ("side", "wrist"):
        p["cameras"][group]["T_parent_camera"] = t
    elif group in ("robot_base", "workspace"):
        p[group]["T_world"] = t
    elif group == "table":
        p["table"]["T_workspace"] = t
    elif group.startswith("object:"):
        p["objects"][group[7:]]["T_workspace"] = t
    else:
        raise ValueError(group)


def initial(measurements, style=None):
    """Create a revision without promoting field provenance. Legacy agent is UI-independent."""
    p = validate(copy.deepcopy(measurements))
    if style:
        for row in style["styles"]:
            target = p["table"] if row["id"] == "table" else p["objects"].get(row["id"])
            if target is None:
                raise ValueError("Agent named an unmeasured object: " + row["id"])
            if "shape" in row:
                target["shape"] = row["shape"]
            target["appearance"] = {
                "color": row["color"], "roughness": row["roughness"],
                "provenance": {"color": "provisional", "roughness": "provisional"},
            }
        p["appearance"]["light_intensity"] = style["light_intensity"]
        p["appearance"]["provenance"]["light_intensity"] = "provisional"
        p["agent_notes"] = style["notes"]
    p["revision"] = {
        "kind": "initial",
        "geometry": "field provenance preserved",
        "appearance": "agent approximation" if style else "user choices",
    }
    return validate(p)


def render_k(camera):
    # Isaac Lab 2.3 pinhole limitation: use centred square-pixel virtual camera.
    # Real images are rectified to this same K for appearance comparisons.
    w, h = camera["resolution"]
    k = np.asarray(camera["K"], float)
    f = float((k[0, 0] + k[1, 1]) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
