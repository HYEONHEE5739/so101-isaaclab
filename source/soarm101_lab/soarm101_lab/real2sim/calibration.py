"""Measured landmarks, multi-pose holdout validation and bounded numerical fitting."""

import copy
from soarm101_lab.so101_joint_mapping import require_capture_mapping
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from .profile import (
    transform, pose6, matrix, get_group, set_group, validate, digest, render_k,
    require_metric_geometry, mark_fitted,
)


def load_dataset(path):
    path = Path(path)
    data = json.loads(path.read_text())
    out = []
    for row in data["observations"]:
        row = copy.deepcopy(row)
        folder = (path.parent / row["capture"]).resolve()
        if not (folder / "COMPLETE").exists():
            raise ValueError("Incomplete capture: " + str(folder))
        state = json.loads((folder / "state.json").read_text())
        require_capture_mapping(state)
        if not state["quality"]["eligible_for_calibration"]:
            raise ValueError("Capture quality rejected: " + str(folder))
        if row["split"] not in ("train", "validation"):
            raise ValueError("Explicit train/validation split required")
        if row["view"] not in ("side", "wrist"):
            raise ValueError("Unknown camera")
        if (
            np.asarray(row["point_m"]).shape != (3,)
            or np.asarray(row["uv"]).shape != (2,)
            or not np.isfinite([*row["point_m"], *row["uv"]]).all()
        ):
            raise ValueError("Invalid measured point/pixel")
        w, h = state["active_profile"]["cameras"][row["view"]]["resolution"]
        if not (0 <= row["uv"][0] < w and 0 <= row["uv"][1] < h):
            raise ValueError("Landmark outside raw image resolution")
        row["_state"] = state
        row["capture"] = str(folder)
        out.append(row)
    if not out:
        raise ValueError("Add measured 3D/2D landmarks first")
    # Robot kinematics/calibration must be consistent; scene revisions may differ.
    ids = {r["_state"]["hardware_identity"] for r in out}
    if len(ids) != 1:
        raise ValueError("Do not pool different robot assets/motor calibrations/camera devices or resolutions")
    splits = {}
    for r in out:
        splits.setdefault(r["capture"], set()).add(r["split"])
    if any(len(s) > 1 for s in splits.values()):
        raise ValueError("Split by entire capture, not by points from the same pose")
    return out


def world_point(p, row):
    frame = row["frame"]
    s = row["_state"]
    if frame == "world":
        t = np.eye(4)
    elif frame == "workspace":
        t = transform(p["workspace"]["T_world"])
    elif frame == "table":
        require_metric_geometry(p, frame)
        t = transform(p["workspace"]["T_world"]) @ transform(p["table"]["T_workspace"])
    elif frame.startswith("object:"):
        require_metric_geometry(p, frame)
        t = transform(p["workspace"]["T_world"]) @ transform(p["objects"][frame[7:]]["T_workspace"])
    elif frame.startswith("robot:"):
        t = transform(p["robot_base"]["T_world"]) @ transform(s["links_base"][frame[6:]])
    else:
        raise ValueError("Unknown landmark frame: " + frame)
    return (t @ np.r_[row["point_m"], 1])[:3]


def camera_world(p, row):
    c = p["cameras"][row["view"]]
    t = transform(c["T_parent_camera"])
    if row["view"] == "wrist":
        t = transform(p["robot_base"]["T_world"]) @ transform(row["_state"]["links_base"]["gripper"]) @ t
    return t


def project(p, row):
    c = p["cameras"][row["view"]]
    x = (np.linalg.inv(camera_world(p, row)) @ np.r_[world_point(p, row), 1])[:3]
    uv, _ = cv2.projectPoints(
        x.reshape(1, 3), np.zeros(3), np.zeros(3), np.asarray(c["K"], float), np.asarray(c["distortion"], float)
    )
    return uv.ravel(), x[2]


def residual(p, rows):
    result = []
    for r in rows:
        uv, z = project(p, r)
        result.extend(uv - np.asarray(r["uv"]))
        # Behind-camera solutions are never accepted.
        result.append(max(0.0, 0.001 - z) * 1e4)
    return np.asarray(result)


def metric(p, rows):
    if not rows:
        return None
    values = []
    behind = 0
    for r in rows:
        uv, z = project(p, r)
        values.append(float(np.linalg.norm(uv - r["uv"])))
        behind += int(z <= 0.001)
    return {
        "points": len(values),
        "mean_px": float(np.mean(values)),
        "rmse_px": float(np.sqrt(np.mean(np.square(values)))),
        "max_px": max(values),
        "behind_camera": behind,
        "captures": len({r["capture"] for r in rows}),
    }


def evaluate(p, rows):
    return {split: metric(p, [r for r in rows if r["split"] == split]) for split in ("train", "validation")}


def relevant(group, r):
    if group == "side":
        return r["view"] == "side" and r["frame"] == "world"
    if group == "wrist":
        return r["view"] == "wrist" and not r["frame"].startswith("robot:")
    if group == "robot_base":
        return r["view"] == "side" and r["frame"].startswith("robot:")
    if group == "workspace":
        return r["view"] == "side" and r["frame"] == "workspace"
    if group == "table":
        return r["view"] == "side" and r["frame"] == "table"
    return r["view"] == "side" and r["frame"] == group


def optimize(p, rows, group, translation_bound=0.10, rotation_bound=0.35):
    """One group at a time fixes the gauge; never jointly move camera and base."""
    validate(p)
    if not all(c.get("intrinsics_calibrated") for c in p["cameras"].values()):
        raise ValueError("Calibrate both camera intrinsics first")
    if group not in ("side",) and not p["cameras"]["side"].get("extrinsics_calibrated"):
        raise ValueError("First fit side camera from world-fixed measured landmarks")
    if group == "wrist" and not p["robot_base"].get("calibrated"):
        raise ValueError("First fit robot base with fixed side camera")
    if group in ("table",) or group.startswith("object:"):
        if not p["workspace"].get("calibrated"):
            raise ValueError("First fit workspace frame")
    selected = [r for r in rows if relevant(group, r)]
    train = [r for r in selected if r["split"] == "train"]
    val = [r for r in selected if r["split"] == "validation"]
    if len(train) < 12 or len({r["capture"] for r in train}) < 3 or len(val) < 4:
        raise ValueError(
            "Need >=12 train points across >=3 captures and >=4 validation points in separate captures for " + group
        )
    if group in ("robot_base", "wrist"):
        qs = np.array([list(r["_state"]["physical_follower_measured_joint_state"].values()) for r in train])
        if np.max(np.ptp(qs[:, :5], axis=0)) < 5:
            raise ValueError("Collect robot poses spanning at least 5 degrees")
    before = evaluate(p, selected)
    x0 = pose6(get_group(p, group))
    bound = np.r_[[translation_bound] * 3, [rotation_bound] * 3]

    def candidate(x):
        c = copy.deepcopy(p)
        set_group(c, group, matrix(x))
        return c

    fit = least_squares(
        lambda x: residual(candidate(x), train),
        x0,
        bounds=(x0 - bound, x0 + bound),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=500,
        x_scale="jac",
    )
    q = candidate(fit.x)
    after = evaluate(q, selected)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-7))
    condition = float(singular[0] / max(singular[-1], 1e-15))
    # Holdout must improve too; no arbitrary global "calibrated" threshold.
    accepted = bool(
        fit.success
        and rank == 6
        and condition < 1e7
        and after["train"]["behind_camera"] == 0
        and after["validation"]["behind_camera"] == 0
        and after["train"]["rmse_px"] < before["train"]["rmse_px"]
        and after["validation"]["rmse_px"] <= before["validation"]["rmse_px"] + 1e-6
    )
    report = {
        "group": group,
        "before": before,
        "after": after,
        "accepted": accepted,
        "jacobian_rank": rank,
        "condition": condition,
        "optimizer_message": fit.message,
        "source_profile": digest(p),
        "validation": "held-out captures; geometry only",
        "at_bound": bool(np.any(np.abs(fit.x - x0) > 0.99 * bound)),
    }
    if report["at_bound"]:
        report["accepted"] = False
    if report["accepted"]:
        mark_fitted(q, group)
        if group in ("side", "wrist"):
            q["cameras"][group]["extrinsics_calibrated"] = True
        elif group in ("robot_base", "workspace"):
            q[group]["calibrated"] = True
        q["revision"] = {"kind": "optimized", "parent": digest(p), "group": group, "report": report}
    return (q if report["accepted"] else copy.deepcopy(p)), report


def calibrate_intrinsics(files, columns, rows, square_m):
    if columns < 3 or rows < 3 or square_m <= 0:
        raise ValueError("Measured board dimensions required")
    obj = np.zeros((columns * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    detected = []
    resolution = None
    for f in files:
        image = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Unreadable image: " + str(f))
        size = image.shape[::-1]
        if resolution is not None and size != resolution:
            raise ValueError("All calibration images must have the same resolution")
        resolution = size
        ok, corners = cv2.findChessboardCornersSB(image, (columns, rows))
        if ok:
            detected.append((str(f), corners.astype(np.float32)))
    if len(detected) < 10:
        raise ValueError("Need >=10 detected board views (8 train + 2 held out), with varied tilts and coverage")
    train = detected[:-2]
    val = detected[-2:]
    # Zhang homography constraints: translation-only/repeated views cannot determine K.
    constraints = []

    def vij(h, i, j):
        return np.array(
            [
                h[0, i] * h[0, j],
                h[0, i] * h[1, j] + h[1, i] * h[0, j],
                h[1, i] * h[1, j],
                h[2, i] * h[0, j] + h[0, i] * h[2, j],
                h[2, i] * h[1, j] + h[1, i] * h[2, j],
                h[2, i] * h[2, j],
            ]
        )

    for _, corners in train:
        normalized = (corners.reshape(-1, 2) - np.array(resolution) / 2) / max(resolution)
        h, _ = cv2.findHomography(obj[:, :2], normalized, method=0)
        if h is None:
            raise ValueError("Degenerate board homography")
        for v in (vij(h, 0, 1), vij(h, 0, 0) - vij(h, 1, 1)):
            constraints.append(v / max(np.linalg.norm(v), 1e-15))
    singular = np.linalg.svd(np.array(constraints), compute_uv=False)
    if singular[-2] / singular[0] < 1e-5:
        raise ValueError("Board views do not constrain intrinsics: add tilted views around both axes")
    rms, k, d, rv, tv = cv2.calibrateCamera([obj] * len(train), [c for _, c in train], resolution, None, None)
    errors = []
    for f, c in val:
        ok, r, t = cv2.solvePnP(obj, c, k, d)
        if not ok:
            raise ValueError("Held-out board pose failed")
        uv, _ = cv2.projectPoints(obj, r, t, k, d)
        errors.append(float(np.sqrt(np.mean(np.sum((uv - c) ** 2, axis=2)))))
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid calibration result")
    return {
        "K": k.tolist(),
        "distortion": d.ravel().tolist(),
        "resolution": list(resolution),
        "intrinsics_calibrated": True,
        "intrinsics_report": {
            "train_rms_px": float(rms),
            "heldout_rmse_px": errors,
            "files": [f for f, _ in detected],
            "square_m": square_m,
            "board_inner_corners": [columns, rows],
            "model": "OpenCV Brown five coefficients",
            "note": "Inspect held-out errors and lens coverage; a fitted result is not a guarantee of accuracy",
        },
    }


def image_metrics(real, sim, camera):
    """Rectify only real lens distortion; never warp image to hide geometric error."""
    if real.shape != sim.shape:
        raise ValueError("Real and Sim resolutions differ")
    k = np.asarray(camera["K"], float)
    d = np.asarray(camera["distortion"], float)
    h, w = real.shape[:2]
    mx, my = cv2.initUndistortRectifyMap(k, d, None, render_k(camera), (w, h), cv2.CV_32FC1)
    a = cv2.remap(real, mx, my, cv2.INTER_LINEAR)
    valid = (mx >= 1) & (my >= 1) & (mx < w - 2) & (my < h - 2)
    if not valid.any():
        raise ValueError("No valid rectified pixels")
    ea = (cv2.Canny(a, 80, 160) > 0) & valid
    eb = (cv2.Canny(sim, 80, 160) > 0) & valid
    edge = None
    if ea.any() and eb.any():
        da = cv2.distanceTransform((~ea).astype("uint8"), cv2.DIST_L2, 5)
        db = cv2.distanceTransform((~eb).astype("uint8"), cv2.DIST_L2, 5)
        edge = float((db[ea].mean() + da[eb].mean()) / 2)
    return {
        "appearance_rgb_mae": float(np.abs(a.astype(float) - sim)[valid].mean() / 255),
        "edge_chamfer_px": edge,
        "valid_pixels": int(valid.sum()),
        "note": "Appearance/edges diagnostic only; landmark reprojection is the geometric objective",
    }, a
