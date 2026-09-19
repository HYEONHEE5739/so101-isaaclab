"""GPT chooses tasks/styles, never metric parameters. Numerical code owns geometry."""

import base64
import copy
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
from . import calibration
from .profile import initial
from .storage import revision, atomic, new_dir


def ask(model, prompt, schema, images=()):
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not model.strip():
        raise ValueError(
            "Set OPENAI_API_KEY in GUI process and choose an API model supporting vision + structured outputs"
        )
    content = [{"type": "input_text", "text": prompt}]
    for file in images:
        from PIL import Image
        import io

        im = Image.open(file).convert("RGB")
        im.thumbnail((1600, 1600))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        content.append(
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
        )
    request = {
        "model": model,
        "reasoning": {"effort": "low"},
        "store": False,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "real2sim_decision", "strict": True, "schema": schema}},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {e.code}: {body}") from e
    if data.get("status") != "completed":
        raise RuntimeError("Agent did not complete: " + str(data.get("status")))
    chunks = [c["text"] for o in data.get("output", []) for c in o.get("content", []) if c.get("type") == "output_text"]
    if not chunks:
        raise RuntimeError("Agent returned no structured decision")
    return json.loads("".join(chunks))


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def initial_scene(measurements, images, model):
    ids = ["table", *measurements["objects"]]
    schema = object_schema(
        {
            "styles": {
                "type": "array",
                "items": object_schema(
                    {
                        "id": {"type": "string", "enum": ids},
                        "shape": {"type": "string", "enum": ["box", "cylinder", "cup"]},
                        "color": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                        "roughness": {"type": "number"},
                    }
                ),
            },
            "light_intensity": {"type": "number"},
            "notes": {"type": "string"},
        }
    )
    result = ask(
        model,
        "Describe appearance of measured scene elements from these photos. Choose box/cylinder/cup primitive types ONLY where supported by the supplied measured dimensions (cup requires measured wall_m; otherwise retain the original shape). Return colors in linear RGB [0,1], roughness [0,1], a provisional nonnegative dome light intensity and notes. Never invent metric geometry. Existing measured elements: "
        + json.dumps({n: measurements["table"] if n == "table" else measurements["objects"][n] for n in ids}),
        schema,
        images,
    )
    return initial(measurements, result)


def choose(model, p, metrics, history):
    groups = ["side", "robot_base", "wrist", "workspace", "table", *["object:" + n for n in p["objects"]]]
    schema = object_schema(
        {
            "action": {"type": "string", "enum": ["optimize", "request_capture", "stop"]},
            "group": {"type": "string", "enum": groups},
            "reason": {"type": "string"},
            "additional_capture": {"type": "string"},
        }
    )
    prompt = (
        "You supervise measured SO101 calibration. Select ONE parameter group, request additional measured captures, or stop. Do not output numerical parameters. Order: calibrated intrinsics, side world anchors, robot_base, wrist mount, workspace, table/object poses. Numerical optimizer enforces fixed gauges and held-out validation. Never repeatedly retry a rejected/unobservable group without more evidence. Metrics: "
        + json.dumps(metrics)
        + "\nHistory: "
        + json.dumps(history)
        + "\nCalibration status: "
        + json.dumps({"cameras": p["cameras"], "robot_base": p["robot_base"], "workspace": p["workspace"]})
    )
    return ask(model, prompt, schema)


def iterate(profile, rows, bus, workspace, model, max_rounds=3, notify=lambda s: None, cancel=None):
    """Bounded synchronous worker. UI thread remains responsive; runtime must be stopped."""
    if not 1 <= max_rounds <= 10:
        raise ValueError("Iteration budget must be 1..10")
    p = copy.deepcopy(profile)
    history = []
    folder = new_dir(Path(workspace) / "runs", "agent")
    try:
        for index in range(max_rounds):
            if cancel and cancel.is_set():
                break
            metrics = calibration.evaluate(p, rows)
            decision = choose(model, p, metrics, history)
            item = {"decision": decision}
            history.append(item)
            atomic(folder / "history.json", history)
            notify(decision["reason"])
            if decision["action"] != "optimize":
                break
            if cancel and cancel.is_set():
                break
            try:
                candidate, report = calibration.optimize(p, rows, decision["group"])
            except ValueError as exc:
                item["error"] = str(exc)
                notify(str(exc))
                continue
            item["optimization"] = report
            if not report["accepted"]:
                continue
            path = revision(workspace, candidate, report)
            item["candidate_profile"] = str(path)
            atomic(folder / "history.json", history)
            notify("Accepted numerical fit; rendering held-out and training poses")
            command = bus.send(
                "render_dataset", {"profile": str(path), "captures": sorted({r["capture"] for r in rows})}
            )
            render = bus.wait(command, timeout=300)
            item["render"] = render
            p = candidate
            item["profile"] = str(path)
            atomic(path.parent / "iteration.json", item)
    except Exception as exc:
        history.append({"error": str(exc)})
        raise
    finally:
        atomic(folder / "history.json", history)
    return {"profile": p, "history": history, "run": str(folder)}
