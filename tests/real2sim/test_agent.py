import copy
import io
import json
from pathlib import Path
from soarm101_lab.real2sim import agent, profile


def example():
    p = profile.load(Path(__file__).resolve().parents[2] / "docs/real2sim/measurements.example.json")
    return p


def test_responses_request_contract_and_secret_not_saved(monkeypatch, tmp_path):
    from PIL import Image

    picture = tmp_path / "photo.png"
    Image.new("RGB", (10, 10)).save(picture)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    seen = []

    def open_(req, timeout):
        payload = json.loads(req.data)
        seen.append(payload)
        assert payload["store"] is False
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["input"][0]["content"][1]["type"] == "input_image"
        return io.BytesIO(
            json.dumps(
                {"status": "completed", "output": [{"content": [{"type": "output_text", "text": '{"answer":"ok"}'}]}]}
            ).encode()
        )

    monkeypatch.setattr(agent.urllib.request, "urlopen", open_)
    result = agent.ask("user-selected-model", "prompt", agent.object_schema({"answer": {"type": "string"}}), [picture])
    assert result == {"answer": "ok"}
    assert "test-only-key" not in json.dumps(seen)


def test_initial_agent_cannot_reposition_geometry(monkeypatch):
    p = example()
    saved = copy.deepcopy(p)

    def ask(model, prompt, schema, images):
        assert not {"position", "dimensions", "mass", "friction"} & set(schema["properties"])
        return {
            "styles": [{"id": "table", "color": [0.2, 0.3, 0.4], "roughness": 0.5}],
            "light_intensity": 1700,
            "notes": "appearance only",
        }

    monkeypatch.setattr(agent, "ask", ask)
    result = agent.initial_scene(p, ["photo"], "model")
    assert result["robot_base"] == saved["robot_base"]
    assert result["objects"] == saved["objects"]
    assert result["table"]["T_workspace"] == saved["table"]["T_workspace"]


def test_iteration_applies_only_accepted_then_renders(monkeypatch, tmp_path):
    p = example()
    candidate = copy.deepcopy(p)
    candidate["workspace"]["T_world"][0][3] = 0.01
    monkeypatch.setattr(agent.calibration, "evaluate", lambda *a: {"train": {"rmse_px": 2}})
    decisions = iter(
        [
            {"action": "optimize", "group": "workspace", "reason": "measured", "additional_capture": ""},
            {"action": "stop", "group": "workspace", "reason": "done", "additional_capture": ""},
        ]
    )
    monkeypatch.setattr(agent, "choose", lambda *a: next(decisions))
    monkeypatch.setattr(agent.calibration, "optimize", lambda *a: (candidate, {"accepted": True}))

    class Bus:
        def send(self, name, payload):
            assert name == "render_dataset"
            assert Path(payload["profile"]).exists()
            return "id"

        def wait(self, *a, **k):
            return {"poses": 1}

    result = agent.iterate(p, [{"capture": "capture"}], Bus(), tmp_path, "model")
    assert result["profile"] == candidate
    assert (Path(result["run"]) / "history.json").exists()


def test_rejected_fit_never_commands_runtime(monkeypatch, tmp_path):
    p = example()
    monkeypatch.setattr(agent.calibration, "evaluate", lambda *a: {})
    monkeypatch.setattr(
        agent, "choose", lambda *a: {"action": "optimize", "group": "side", "reason": "try", "additional_capture": ""}
    )
    monkeypatch.setattr(agent.calibration, "optimize", lambda *a: (p, {"accepted": False}))

    class Bus:
        def send(self, *a):
            raise AssertionError("Rejected candidate must not be sent")

    result = agent.iterate(p, [], Bus(), tmp_path, "model", max_rounds=1)
    assert result["profile"] == p
