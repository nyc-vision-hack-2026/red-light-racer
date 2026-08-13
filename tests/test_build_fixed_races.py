import base64
import json
from pathlib import Path

import pytest

from app.rounds import assert_round_valid, prompt_view, reveal_view
from tools.build_fixed_races import (
    Sequence,
    build_round,
    call_workflow,
    normalize_predictions,
    select_starting_cars,
)


CAMERA = {
    "camera_id": "test_camera",
    "frame_width": 160,
    "frame_height": 140,
    "staging_zone": [[0, 10], [110, 10], [110, 55], [0, 55]],
    "finish_line": [[0, 100], [110, 100]],
    "travel_direction": [0, 1],
    "lane_merge_px": 24,
    "max_lateral_step_px": 25,
}


def detection(x: float, y: float, *, cls: str = "car", conf: float = 0.9) -> dict:
    return {"x": x, "y": y, "w": 20, "h": 20, "cls": cls, "conf": conf}


def test_normalize_predictions_handles_nested_roboflow_output():
    payload = {
        "outputs": [
            {
                "predictions": {
                    "predictions": [
                        {"x": 20, "y": 30, "width": 10, "height": 12, "class": "car", "confidence": 0.8},
                        {"x1": 40, "y1": 50, "x2": 55, "y2": 70, "class_name": "truck", "conf": 0.7},
                    ]
                }
            }
        ]
    }

    boxes = normalize_predictions(payload)

    assert boxes == [
        {"x": 15.0, "y": 24.0, "w": 10.0, "h": 12.0, "cls": "car", "conf": 0.8},
        {"x": 40.0, "y": 50.0, "w": 15.0, "h": 20.0, "cls": "truck", "conf": 0.7},
    ]


def test_call_workflow_uses_documented_base64_rest_shape(monkeypatch):
    image = Path(__file__)
    image_bytes = image.read_bytes()
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"outputs": []}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = call_workflow(
        image,
        api_url="https://serverless.roboflow.com/",
        api_key="secret",
        workspace_name="my workspace",
        workflow_id="cars/workflow",
        image_input_name="image",
    )

    request = captured["request"]
    body = json.loads(request.data)
    assert request.full_url == (
        "https://serverless.roboflow.com/my%20workspace/workflows/cars%2Fworkflow"
    )
    assert body == {
        "api_key": "secret",
        "inputs": {
            "image": {
                "type": "base64",
                "value": base64.b64encode(image_bytes).decode("ascii"),
            }
        },
    }
    assert captured["timeout"] == 90
    assert result == {"outputs": []}


def test_starting_cars_keeps_front_vehicle_per_lane():
    boxes = [
        detection(10, 5),   # following vehicle in the left lane
        detection(11, 27),  # front vehicle in the left lane
        detection(65, 23),  # front vehicle in the right lane
        detection(135, 25), # outside the staging zone
    ]

    selected = select_starting_cars(boxes, CAMERA)

    assert len(selected) == 2
    assert selected[0]["x"] == 11
    assert selected[1]["x"] == 65


def test_build_round_tracks_cars_and_interpolates_same_frame_finish():
    frames = [Path(f"frame_{index:02d}.jpg") for index in range(8)]
    sequence = Sequence(
        candidate_id="cand_test",
        frames=frames,
        frame_names=[path.name for path in frames],
        green_index=3,
        interval_seconds=2.0,
    )
    positions_a = [20, 20, 20, 22, 50, 75, 95, 112]
    positions_b = [20, 20, 20, 21, 45, 65, 82, 103]
    detections_by_frame = []
    for index in range(8):
        # Reverse detector order on alternating frames and add an unrelated car.
        racers = [detection(10, positions_a[index]), detection(65, positions_b[index])]
        if index % 2:
            racers.reverse()
        detections_by_frame.append(racers + [detection(135, 30 + index)])

    round_data = build_round(
        sequence,
        detections_by_frame,
        CAMERA,
        round_id="r_001",
        prompt_frames=4,
    )

    assert_round_valid(round_data)
    assert len(round_data["candidates"]) == 2
    assert round_data["winner_track_id"] == 1
    assert round_data["finish_frame_index"] == {"1": 6, "2": 6}
    assert round_data["finish_crossing_time"]["1"] < round_data["finish_crossing_time"]["2"]

    prompt = prompt_view(round_data)
    assert "winner_track_id" not in prompt
    assert "finish_crossing_time" not in prompt

    reveal = reveal_view(round_data)
    assert reveal["finish_crossing_time"] == round_data["finish_crossing_time"]


def test_interpolated_finish_rejects_an_exact_tie():
    round_data = {
        "id": "r_tie",
        "fps": 0.5,
        "frames": [f"{index}.jpg" for index in range(7)],
        "green_index": 3,
        "candidates": [
            {
                "track_id": track_id,
                "label": label,
                "boxes": [
                    {"frame": frame, "x": x, "y": 20, "w": 20, "h": 20}
                    for frame in range(4)
                ],
            }
            for track_id, label, x in ((1, "A", 10), (2, "B", 60))
        ],
        "winner_track_id": 1,
        "finish_frame_index": {"1": 5, "2": 5},
        "finish_crossing_time": {"1": 4.5, "2": 4.5},
        "finish_line": [[0, 100], [110, 100]],
    }

    with pytest.raises(ValueError, match="tie at interpolated finish time"):
        assert_round_valid(round_data)
