#!/usr/bin/env python3
"""Build game-ready races from fixed-camera images and detector boxes.

The image model has one job: return vehicle bounding boxes for each frame.
This script does the temporal work locally so a stateless hosted detector is
enough:

1. select the front vehicle in each occupied lane inside the staging zone;
2. associate those vehicles using lateral position and forward motion;
3. interpolate the first crossing of the camera's fixed finish line;
4. materialize schema-valid frames and rounds.json for review.

Roboflow mode requires ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, and
ROBOFLOW_WORKFLOW_ID. The Workflow should contain an object detector only (no
ByteTrack block). For repeatable/offline runs, pass a previously saved
detection document with --detections.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rounds import RoundValidationError, assert_round_valid  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CLASSES = {"car", "truck", "bus", "motorcycle", "vehicle"}


class RaceBuildError(ValueError):
    """A candidate sequence cannot be turned into an unambiguous race."""


@dataclass
class Sequence:
    candidate_id: str
    frames: list[Path]
    frame_names: list[str]
    green_index: int
    interval_seconds: float


@dataclass
class TrackState:
    last_frame: int
    last_box: dict[str, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    matches: int = 1


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def unit(vector: Iterable[float]) -> tuple[float, float]:
    x, y = (float(v) for v in vector)
    length = math.hypot(x, y)
    if length <= 1e-9:
        raise RaceBuildError("travel_direction must be non-zero")
    return x / length, y / length


def box_center(box: dict[str, float]) -> tuple[float, float]:
    return box["x"] + box["w"] / 2, box["y"] + box["h"] / 2


def box_leading_edge(box: dict[str, float], travel: tuple[float, float]) -> tuple[float, float]:
    tx, ty = travel
    cx, cy = box_center(box)
    if abs(tx) >= abs(ty):
        return (box["x"] + box["w"], cy) if tx >= 0 else (box["x"], cy)
    return (cx, box["y"] + box["h"]) if ty >= 0 else (cx, box["y"])


def point_in_poly(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def box_intersects_zone(
    box: dict[str, float], polygon: list[list[float]], margin: float = 10.0
) -> bool:
    x0, y0 = box["x"] - margin, box["y"] - margin
    x1 = box["x"] + box["w"] + margin
    y1 = box["y"] + box["h"] + margin
    cx, cy = box_center(box)
    samples = [
        (cx, cy),
        (box["x"], box["y"]),
        (box["x"] + box["w"], box["y"]),
        (box["x"], box["y"] + box["h"]),
        (box["x"] + box["w"], box["y"] + box["h"]),
        (cx, box["y"]),
        (cx, box["y"] + box["h"]),
    ]
    if any(point_in_poly(x, y, polygon) for x, y in samples):
        return True
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in polygon)


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["w"], b["x"] + b["w"])
    bottom = min(a["y"] + a["h"], b["y"] + b["h"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
    return intersection / union if union > 0 else 0.0


def clean_class_name(value: Any) -> str:
    return str(value or "vehicle").strip().lower().replace("_", " ")


def prediction_to_box(prediction: dict[str, Any]) -> dict[str, Any] | None:
    if all(k in prediction for k in ("x", "y", "width", "height")):
        width = float(prediction["width"])
        height = float(prediction["height"])
        x = float(prediction["x"]) - width / 2
        y = float(prediction["y"]) - height / 2
    elif all(k in prediction for k in ("x1", "y1", "x2", "y2")):
        x = float(prediction["x1"])
        y = float(prediction["y1"])
        width = float(prediction["x2"]) - x
        height = float(prediction["y2"]) - y
    elif all(k in prediction for k in ("x", "y", "w", "h")):
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["w"])
        height = float(prediction["h"])
    else:
        return None
    if width <= 1 or height <= 1:
        return None
    return {
        "x": x,
        "y": y,
        "w": width,
        "h": height,
        "cls": clean_class_name(
            prediction.get("class")
            or prediction.get("class_name")
            or prediction.get("label")
        ),
        "conf": float(prediction.get("confidence") or prediction.get("conf") or 0.0),
    }


def normalize_predictions(payload: Any) -> list[dict[str, Any]]:
    """Recursively extract common Roboflow prediction shapes without duplicates."""
    boxes: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            box = prediction_to_box(node)
            if box is not None:
                boxes.append(box)
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for box in boxes:
        key = (
            round(box["x"], 2),
            round(box["y"], 2),
            round(box["w"], 2),
            round(box["h"], 2),
            box["cls"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(box)
    return unique


def call_workflow(
    image_path: Path,
    *,
    api_url: str,
    api_key: str,
    workspace_name: str,
    workflow_id: str,
    image_input_name: str,
) -> Any:
    """Call Roboflow's workflow REST endpoint without an SDK dependency."""
    workspace = urllib.parse.quote(workspace_name, safe="")
    workflow = urllib.parse.quote(workflow_id, safe="")
    endpoint = f"{api_url.rstrip('/')}/{workspace}/workflows/{workflow}"
    body = json.dumps(
        {
            "api_key": api_key,
            "inputs": {
                image_input_name: {
                    "type": "base64",
                    "value": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                }
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RaceBuildError(f"Roboflow returned HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RaceBuildError(f"Roboflow request failed: {exc.reason}") from exc


def frame_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:20]


def load_sequences(demo_dir: Path, selected: set[str] | None = None) -> list[Sequence]:
    sequences: list[Sequence] = []
    for candidate_dir in sorted(p for p in demo_dir.glob("cand_*") if p.is_dir()):
        if selected and candidate_dir.name not in selected:
            continue
        meta = load_json(candidate_dir / "meta.json")
        before = [candidate_dir / name for name in meta["before_frames"]]
        after = [candidate_dir / name for name in meta["after_frames"]]
        if not before or not after or not all(path.is_file() for path in before + after):
            raise RaceBuildError(f"{candidate_dir.name}: incomplete frame list")

        overlaps = frame_hash(before[-1]) == frame_hash(after[0])
        if overlaps:
            frames = before + after[1:]
            after_zero = len(before) - 1
        else:
            frames = before + after
            after_zero = len(before)
        green_offset = int(meta["green_index_global"]) - int(meta["prompt_end_global"])
        green_index = after_zero + green_offset
        if not 2 <= green_index < len(frames) - 2:
            raise RaceBuildError(f"{candidate_dir.name}: invalid local green index {green_index}")
        sequences.append(
            Sequence(
                candidate_id=candidate_dir.name,
                frames=frames,
                frame_names=[str(path.relative_to(demo_dir)).replace("\\", "/") for path in frames],
                green_index=green_index,
                interval_seconds=float(meta.get("interval_seconds", 2.0)),
            )
        )
    if not sequences:
        raise RaceBuildError("no candidate sequences selected")
    return sequences


def detect_sequences(
    sequences: list[Sequence],
    *,
    api_url: str,
    api_key: str,
    workspace_name: str,
    workflow_id: str,
    image_input_name: str,
    cache_dir: Path,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {"version": 1, "sequences": {}}
    for sequence in sequences:
        frame_detections: list[list[dict[str, Any]]] = []
        print(f"detecting {sequence.candidate_id}: {len(sequence.frames)} frames")
        for index, frame in enumerate(sequence.frames):
            cache_path = cache_dir / f"{frame_hash(frame)}.json"
            if cache_path.is_file():
                raw = load_json(cache_path)
                source = "cache"
            else:
                raw = call_workflow(
                    frame,
                    api_url=api_url,
                    api_key=api_key,
                    workspace_name=workspace_name,
                    workflow_id=workflow_id,
                    image_input_name=image_input_name,
                )
                cache_path.write_text(json.dumps(raw))
                source = "api"
            boxes = normalize_predictions(raw)
            frame_detections.append(boxes)
            print(f"  [{index:02d}] {source}: {len(boxes)} boxes")
        output["sequences"][sequence.candidate_id] = {
            "frames": sequence.frame_names,
            "green_index": sequence.green_index,
            "interval_seconds": sequence.interval_seconds,
            "detections": frame_detections,
        }
    return output


def filter_detections(
    detections: list[dict[str, Any]],
    *,
    allowed_classes: set[str],
    min_confidence: float,
) -> list[dict[str, Any]]:
    filtered = [
        dict(detection)
        for detection in detections
        if clean_class_name(detection.get("cls")) in allowed_classes
        and float(detection.get("conf", 0.0)) >= min_confidence
    ]
    filtered.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    deduped: list[dict[str, Any]] = []
    for detection in filtered:
        if any(box_iou(detection, kept) >= 0.75 for kept in deduped):
            continue
        deduped.append(detection)
    return deduped


def select_starting_cars(
    detections: list[dict[str, Any]],
    camera: dict[str, Any],
    *,
    max_candidates: int = 4,
) -> list[dict[str, Any]]:
    zone = camera["staging_zone"]
    travel = unit(camera["travel_direction"])
    normal = (-travel[1], travel[0])
    margin = float(camera.get("start_margin_px", 12.0))
    lane_merge = float(camera.get("lane_merge_px", 26.0))
    eligible = [box for box in detections if box_intersects_zone(box, zone, margin)]
    if len(eligible) < 2:
        raise RaceBuildError(f"only {len(eligible)} vehicles overlap the staging zone")

    # Group detections occupying roughly the same lane and retain the vehicle
    # furthest along the travel direction (the front of that lane's queue).
    clusters: list[list[dict[str, Any]]] = []
    for box in sorted(eligible, key=lambda b: box_center(b)[0] * normal[0] + box_center(b)[1] * normal[1]):
        lateral = box_center(box)[0] * normal[0] + box_center(box)[1] * normal[1]
        if clusters:
            prior_values = [
                box_center(item)[0] * normal[0] + box_center(item)[1] * normal[1]
                for item in clusters[-1]
            ]
            if abs(lateral - sum(prior_values) / len(prior_values)) <= lane_merge:
                clusters[-1].append(box)
                continue
        clusters.append([box])

    selected = [
        max(
            cluster,
            key=lambda b: box_center(b)[0] * travel[0] + box_center(b)[1] * travel[1],
        )
        for cluster in clusters
    ]
    if len(selected) > max_candidates:
        selected = sorted(
            selected,
            key=lambda b: box_center(b)[0] * travel[0] + box_center(b)[1] * travel[1],
            reverse=True,
        )[:max_candidates]
    selected.sort(key=lambda b: box_center(b)[0])
    if len(selected) < 2:
        raise RaceBuildError("fewer than two occupied starting lanes")
    return selected


def match_cost(
    state: TrackState,
    box: dict[str, Any],
    frame_index: int,
    travel: tuple[float, float],
    camera: dict[str, Any],
) -> float | None:
    gap = frame_index - state.last_frame
    traversal_gap = abs(gap)
    if traversal_gap <= 0 or traversal_gap > int(camera.get("max_missing_frames", 2)) + 1:
        return None
    direction = 1.0 if gap > 0 else -1.0
    tx, ty = travel[0] * direction, travel[1] * direction
    nx, ny = -ty, tx
    previous = box_center(state.last_box)
    current = box_center(box)
    dx, dy = current[0] - previous[0], current[1] - previous[1]
    forward = dx * tx + dy * ty
    lateral = abs(dx * nx + dy * ny)
    max_forward = float(camera.get("max_forward_step_px", 95.0)) * traversal_gap
    max_lateral = float(camera.get("max_lateral_step_px", 32.0)) * math.sqrt(traversal_gap)
    max_backward = float(camera.get("max_backward_step_px", 10.0)) * traversal_gap
    if forward < -max_backward or forward > max_forward or lateral > max_lateral:
        return None

    prediction = (
        previous[0] + state.velocity[0] * traversal_gap,
        previous[1] + state.velocity[1] * traversal_gap,
    )
    prediction_error = math.hypot(current[0] - prediction[0], current[1] - prediction[1])
    area_a = max(1.0, state.last_box["w"] * state.last_box["h"])
    area_b = max(1.0, box["w"] * box["h"])
    size_penalty = abs(math.log(area_b / area_a)) * 12.0
    backward_penalty = max(0.0, -forward) * 2.5
    return prediction_error * 0.55 + lateral * 1.4 + size_penalty + backward_penalty - box_iou(state.last_box, box) * 12.0


def best_assignment(
    states: dict[int, TrackState],
    detections: list[dict[str, Any]],
    frame_index: int,
    travel: tuple[float, float],
    camera: dict[str, Any],
) -> dict[int, int]:
    track_ids = sorted(states)
    options: dict[int, list[tuple[int, float]]] = {}
    max_cost = float(camera.get("max_match_cost", 115.0))
    for track_id in track_ids:
        ranked = []
        for detection_index, detection in enumerate(detections):
            cost = match_cost(states[track_id], detection, frame_index, travel, camera)
            if cost is not None and cost <= max_cost:
                ranked.append((detection_index, cost))
        options[track_id] = sorted(ranked, key=lambda item: item[1])[:6]

    miss_cost = float(camera.get("miss_cost", 82.0))
    best_cost = math.inf
    best: dict[int, int] = {}

    def search(position: int, used: set[int], total: float, chosen: dict[int, int]) -> None:
        nonlocal best_cost, best
        if total >= best_cost:
            return
        if position == len(track_ids):
            best_cost = total
            best = dict(chosen)
            return
        track_id = track_ids[position]
        search(position + 1, used, total + miss_cost, chosen)
        for detection_index, cost in options[track_id]:
            if detection_index in used:
                continue
            chosen[track_id] = detection_index
            search(position + 1, used | {detection_index}, total + cost, chosen)
            chosen.pop(track_id, None)

    search(0, set(), 0.0, {})
    return best


def track_from_seed(
    detections_by_frame: list[list[dict[str, Any]]],
    seeds: list[dict[str, Any]],
    seed_frame: int,
    travel: tuple[float, float],
    camera: dict[str, Any],
) -> dict[int, dict[int, dict[str, Any]]]:
    tracks = {index + 1: {seed_frame: dict(seed)} for index, seed in enumerate(seeds)}

    for frame_range in (
        range(seed_frame + 1, len(detections_by_frame)),
        range(seed_frame - 1, -1, -1),
    ):
        states = {
            index + 1: TrackState(seed_frame, dict(seed))
            for index, seed in enumerate(seeds)
        }
        for frame_index in frame_range:
            assignment = best_assignment(states, detections_by_frame[frame_index], frame_index, travel, camera)
            for track_id, detection_index in assignment.items():
                box = dict(detections_by_frame[frame_index][detection_index])
                state = states[track_id]
                traversal_gap = abs(frame_index - state.last_frame)
                prior_center = box_center(state.last_box)
                new_center = box_center(box)
                measured_velocity = (
                    (new_center[0] - prior_center[0]) / traversal_gap,
                    (new_center[1] - prior_center[1]) / traversal_gap,
                )
                blend = 0.5 if state.matches > 1 else 0.0
                state.velocity = (
                    state.velocity[0] * blend + measured_velocity[0] * (1 - blend),
                    state.velocity[1] * blend + measured_velocity[1] * (1 - blend),
                )
                state.last_frame = frame_index
                state.last_box = box
                state.matches += 1
                tracks[track_id][frame_index] = box
    return tracks


def line_progress(
    point: tuple[float, float],
    finish_line: list[list[float]],
    travel: tuple[float, float],
) -> float:
    (ax, ay), (bx, by) = finish_line
    line_x, line_y = bx - ax, by - ay
    normal_a = (line_y, -line_x)
    normal_b = (-line_y, line_x)
    normal = normal_a if normal_a[0] * travel[0] + normal_a[1] * travel[1] > 0 else normal_b
    normal = unit(normal)
    return (point[0] - ax) * normal[0] + (point[1] - ay) * normal[1]


def crossing_for_track(
    boxes: dict[int, dict[str, Any]],
    *,
    green_index: int,
    finish_line: list[list[float]],
    travel: tuple[float, float],
) -> tuple[int, float] | None:
    observations = sorted((frame, box) for frame, box in boxes.items() if frame >= green_index)
    for (previous_frame, previous_box), (frame, box) in itertools.pairwise(observations):
        previous = line_progress(box_leading_edge(previous_box, travel), finish_line, travel)
        current = line_progress(box_leading_edge(box, travel), finish_line, travel)
        if previous < 0 <= current and current > previous:
            fraction = -previous / (current - previous)
            crossing_time = previous_frame + (frame - previous_frame) * fraction
            return frame, crossing_time
    return None


def stabilize_prompt_boxes(
    boxes: dict[int, dict[str, Any]], prompt_start: int, green_index: int
) -> None:
    available = [boxes[frame] for frame in range(prompt_start, green_index) if frame in boxes]
    if len(available) < 2:
        raise RaceBuildError("candidate was not detected in at least two prompt frames")
    median: dict[str, float] = {}
    for key in ("x", "y", "w", "h"):
        values = sorted(float(box[key]) for box in available)
        median[key] = values[len(values) // 2]
    template = dict(available[-1])
    template.update(median)
    for frame in range(prompt_start, green_index):
        original = boxes.get(frame, template)
        stable = dict(original)
        stable.update(median)
        boxes[frame] = stable
    if green_index not in boxes:
        boxes[green_index] = dict(template)


def build_round(
    sequence: Sequence,
    detections_by_frame: list[list[dict[str, Any]]],
    camera: dict[str, Any],
    *,
    round_id: str,
    prompt_frames: int = 4,
    tie_tolerance_frames: float = 0.05,
) -> dict[str, Any]:
    if len(detections_by_frame) != len(sequence.frames):
        raise RaceBuildError(
            f"detection frame count {len(detections_by_frame)} != image count {len(sequence.frames)}"
        )
    travel = unit(camera["travel_direction"])
    seed_frame = sequence.green_index - 1
    seeds = select_starting_cars(detections_by_frame[seed_frame], camera)
    tracks = track_from_seed(detections_by_frame, seeds, seed_frame, travel, camera)

    crossings: dict[int, tuple[int, float] | None] = {
        track_id: crossing_for_track(
            boxes,
            green_index=sequence.green_index,
            finish_line=camera["finish_line"],
            travel=travel,
        )
        for track_id, boxes in tracks.items()
    }
    finishers = sorted(
        ((track_id, crossing) for track_id, crossing in crossings.items() if crossing is not None),
        key=lambda item: item[1][1],
    )
    if len(finishers) < 2:
        raise RaceBuildError(f"only {len(finishers)} candidates reached the finish line")
    if finishers[1][1][1] - finishers[0][1][1] < tie_tolerance_frames:
        raise RaceBuildError("finish is too close to resolve at this frame rate")

    prompt_start = max(0, sequence.green_index - prompt_frames + 1)
    last_finish_frame = max(crossing[0] for _track_id, crossing in finishers)
    end_frame = min(len(sequence.frames) - 1, max(sequence.green_index + 2, last_finish_frame + 1))
    local_green = sequence.green_index - prompt_start

    candidates: list[dict[str, Any]] = []
    finish_frame_index: dict[str, int | None] = {}
    finish_crossing_time: dict[str, float | None] = {}
    for label_index, (track_id, boxes) in enumerate(
        sorted(tracks.items(), key=lambda item: box_center(item[1][seed_frame])[0])
    ):
        stabilize_prompt_boxes(boxes, prompt_start, sequence.green_index)
        rendered_boxes = []
        for absolute_frame in range(prompt_start, end_frame + 1):
            box = boxes.get(absolute_frame)
            if box is None:
                continue
            rendered_boxes.append(
                {
                    "frame": absolute_frame - prompt_start,
                    "x": round(float(box["x"]), 1),
                    "y": round(float(box["y"]), 1),
                    "w": round(float(box["w"]), 1),
                    "h": round(float(box["h"]), 1),
                }
            )
        candidates.append(
            {"track_id": track_id, "label": "ABCD"[label_index], "boxes": rendered_boxes}
        )
        crossing = crossings[track_id]
        finish_frame_index[str(track_id)] = (
            None if crossing is None else crossing[0] - prompt_start
        )
        finish_crossing_time[str(track_id)] = (
            None if crossing is None else round(crossing[1] - prompt_start, 4)
        )

    winner = finishers[0][0]
    result = {
        "id": round_id,
        "fps": round(1.0 / sequence.interval_seconds, 4),
        "frames": sequence.frame_names[prompt_start : end_frame + 1],
        "green_index": local_green,
        "candidates": candidates,
        "winner_track_id": winner,
        "finish_frame_index": finish_frame_index,
        "finish_crossing_time": finish_crossing_time,
        "finish_line": camera["finish_line"],
        "source_candidate": sequence.candidate_id,
    }
    try:
        assert_round_valid(result)
    except RoundValidationError as exc:
        raise RaceBuildError(str(exc)) from exc
    return result


def materialize_round(
    round_data: dict[str, Any],
    sequence: Sequence,
    demo_dir: Path,
    frames_dir: Path,
) -> dict[str, Any]:
    destination = frames_dir / round_data["id"]
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    new_names = []
    for index, relative_name in enumerate(round_data["frames"]):
        source = demo_dir / relative_name
        suffix = source.suffix.lower() if source.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
        destination_name = f"{index:03d}{suffix}"
        shutil.copy2(source, destination / destination_name)
        new_names.append(f"{round_data['id']}/{destination_name}")
    materialized = dict(round_data)
    materialized["frames"] = new_names
    assert_round_valid(materialized)
    return materialized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dir", type=Path, nargs="?", default=ROOT / "demo_set_1")
    parser.add_argument("--camera", type=Path, default=None)
    parser.add_argument("--candidate", action="append", help="Candidate id to build; repeatable")
    parser.add_argument("--detections", type=Path, help="Use a saved detection JSON instead of calling Roboflow")
    parser.add_argument("--save-detections", type=Path, default=None)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com"),
    )
    parser.add_argument("--workspace", default=os.environ.get("ROBOFLOW_WORKSPACE", ""))
    parser.add_argument("--workflow-id", default=os.environ.get("ROBOFLOW_WORKFLOW_ID", ""))
    parser.add_argument("--image-input-name", default="image")
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--classes", default=",".join(sorted(DEFAULT_CLASSES)))
    parser.add_argument("--prompt-frames", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument(
        "--round-set",
        default="roboflow",
        help="Named set under data/round_sets used by --promote (default: roboflow)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Write a named set under data/round_sets without replacing the classic set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    camera_path = args.camera or demo_dir / "camera.json"
    camera = load_json(camera_path)
    selected = set(args.candidate or []) or None
    sequences = load_sequences(demo_dir, selected)

    generated_root = ROOT / "data" / "generated" / demo_dir.name
    detections_path = args.detections or args.save_detections or generated_root / "detections.json"
    if args.detections:
        detection_document = load_json(args.detections)
    else:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key or not args.workspace or not args.workflow_id:
            raise SystemExit(
                "Set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, and ROBOFLOW_WORKFLOW_ID, "
                "or pass --detections. "
                "The Workflow must output object-detection boxes and must not use ByteTrack."
            )
        detection_document = detect_sequences(
            sequences,
            api_url=args.api_url,
            api_key=api_key,
            workspace_name=args.workspace,
            workflow_id=args.workflow_id,
            image_input_name=args.image_input_name,
            cache_dir=generated_root / "cache",
        )
        detections_path.parent.mkdir(parents=True, exist_ok=True)
        detections_path.write_text(json.dumps(detection_document, indent=2) + "\n")
        print(f"saved detections: {detections_path}")

    allowed_classes = {clean_class_name(name) for name in args.classes.split(",") if name.strip()}
    if args.promote:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.round_set):
            raise SystemExit("--round-set must contain only letters, numbers, '-' or '_'")
        set_root = ROOT / "data" / "round_sets" / args.round_set
        output_path = args.output or set_root / "rounds.json"
        frames_dir = args.frames_dir or set_root / "frames"
    else:
        output_path = args.output or generated_root / "rounds.json"
        frames_dir = args.frames_dir or generated_root / "frames"

    rounds = []
    skipped = []
    for sequence in sequences:
        record = detection_document.get("sequences", {}).get(sequence.candidate_id)
        if record is None:
            skipped.append((sequence.candidate_id, "missing detections"))
            continue
        if record.get("frames") and record["frames"] != sequence.frame_names:
            skipped.append((sequence.candidate_id, "detection frame list does not match the demo set"))
            continue
        if record.get("green_index") is not None and int(record["green_index"]) != sequence.green_index:
            skipped.append((sequence.candidate_id, "detection green index does not match the demo set"))
            continue
        raw_detections = record.get("detections", [])
        filtered = [
            filter_detections(
                frame,
                allowed_classes=allowed_classes,
                min_confidence=args.min_confidence,
            )
            for frame in raw_detections
        ]
        round_id = f"r_{len(rounds) + 1:03d}"
        try:
            round_data = build_round(
                sequence,
                filtered,
                camera,
                round_id=round_id,
                prompt_frames=args.prompt_frames,
            )
            rounds.append(materialize_round(round_data, sequence, demo_dir, frames_dir))
            print(
                f"built {round_id} from {sequence.candidate_id}: "
                f"{len(round_data['candidates'])} cars, winner={round_data['winner_track_id']}"
            )
        except RaceBuildError as exc:
            skipped.append((sequence.candidate_id, str(exc)))

    if not rounds:
        details = "; ".join(f"{candidate}: {reason}" for candidate, reason in skipped)
        raise SystemExit(f"No valid rounds built. {details}")

    payload = {
        "version": 1,
        "camera_id": camera["camera_id"],
        "rounds": rounds,
        "build": {
            "source": str(demo_dir),
            "detector": "roboflow-object-detection",
            "skipped": [{"candidate": candidate, "reason": reason} for candidate, reason in skipped],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output_path} ({len(rounds)} valid rounds, {len(skipped)} skipped)")
    if not args.promote:
        print("review this output, then rerun with --promote to replace the game's stub rounds")


if __name__ == "__main__":
    main()
