"""Round prompt/reveal splitting and validity assertions."""

from __future__ import annotations

from typing import Any


class RoundValidationError(ValueError):
    pass


def assert_round_valid(round_data: dict[str, Any]) -> None:
    """Enforce §5.3 validity rules. Raises RoundValidationError on failure."""
    rid = round_data.get("id", "?")
    candidates = round_data.get("candidates") or []
    if not (2 <= len(candidates) <= 4):
        raise RoundValidationError(f"{rid}: need 2–4 candidates, got {len(candidates)}")

    track_ids = [c["track_id"] for c in candidates]
    if len(track_ids) != len(set(track_ids)):
        raise RoundValidationError(f"{rid}: duplicate track IDs within round")

    green = round_data["green_index"]
    frames = round_data["frames"]
    if green < 2:
        raise RoundValidationError(f"{rid}: need ≥3 prompt frames (green_index={green})")
    if len(frames) - 1 - green < 2:
        raise RoundValidationError(
            f"{rid}: need ≥3 reveal frames (len={len(frames)}, green={green})"
        )

    finish = round_data.get("finish_frame_index") or {}
    # Normalize keys: JSON may store int keys as strings after serialization.
    finish_norm: dict[int, int | None] = {}
    for k, v in finish.items():
        finish_norm[int(k)] = v if v is None else int(v)

    finishers = [(tid, idx) for tid, idx in finish_norm.items() if idx is not None]
    if len(finishers) < 2:
        raise RoundValidationError(f"{rid}: need ≥2 finishers, got {len(finishers)}")

    # Low-frame-rate sources can show two cars crossing in the same image.  An
    # optional interpolated crossing time preserves their order without lying
    # about the frame in which either car crossed.  Older rounds continue to
    # use finish_frame_index as their timing source.
    crossing = round_data.get("finish_crossing_time") or {}
    crossing_norm: dict[int, float | None] = {}
    for k, v in crossing.items():
        crossing_norm[int(k)] = None if v is None else float(v)

    if crossing_norm:
        timed_finishers = [
            (tid, crossing_norm.get(tid))
            for tid, _idx in finishers
            if crossing_norm.get(tid) is not None
        ]
        if len(timed_finishers) < 2:
            raise RoundValidationError(
                f"{rid}: need ≥2 interpolated finish times, got {len(timed_finishers)}"
            )
        finishers_sorted = sorted(timed_finishers, key=lambda x: float(x[1]))
        if abs(float(finishers_sorted[0][1]) - float(finishers_sorted[1][1])) < 1e-6:
            raise RoundValidationError(
                f"{rid}: tie at interpolated finish time {finishers_sorted[0][1]}"
            )
    else:
        finishers_sorted = sorted(finishers, key=lambda x: x[1])
        if len(finishers_sorted) >= 2 and finishers_sorted[0][1] == finishers_sorted[1][1]:
            raise RoundValidationError(f"{rid}: tie at finish frame {finishers_sorted[0][1]}")

    winner = round_data.get("winner_track_id")
    if winner is None:
        raise RoundValidationError(f"{rid}: missing winner_track_id")
    expected_winner = finishers_sorted[0][0]
    if int(winner) != expected_winner:
        raise RoundValidationError(
            f"{rid}: winner_track_id={winner} but earliest finisher is {expected_winner}"
        )

    # Near-stationary check: each candidate must appear in ≥2 frames before green
    # with small centre movement (≤8px between consecutive pre-green frames).
    for cand in candidates:
        pre = [b for b in cand["boxes"] if b["frame"] < green]
        at_green = [b for b in cand["boxes"] if b["frame"] == green]
        if len(pre) < 2:
            raise RoundValidationError(
                f"{rid}: track {cand['track_id']} not near-stationary (≥2 pre-green frames)"
            )
        if not at_green:
            raise RoundValidationError(
                f"{rid}: track {cand['track_id']} missing box at green_index"
            )
        centres = [(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2) for b in pre]
        for i in range(1, len(centres)):
            dx = centres[i][0] - centres[i - 1][0]
            dy = centres[i][1] - centres[i - 1][1]
            if (dx * dx + dy * dy) ** 0.5 > 8.0:
                raise RoundValidationError(
                    f"{rid}: track {cand['track_id']} not near-stationary before green"
                )


def prompt_view(round_data: dict[str, Any], frame_url_prefix: str = "/frames/") -> dict[str, Any]:
    """
    Build the client-facing prompt payload.

    Must never include winner_track_id or finish_frame_index.
    Frames and boxes truncated to 0..green_index inclusive.
    """
    green = round_data["green_index"]
    frames = round_data["frames"][: green + 1]
    frame_urls = [
        f if f.startswith("http://") or f.startswith("https://") or f.startswith("/")
        else f"{frame_url_prefix}{f}"
        for f in frames
    ]

    candidates = []
    for cand in round_data["candidates"]:
        boxes = [b for b in cand["boxes"] if b["frame"] <= green]
        candidates.append(
            {
                "track_id": cand["track_id"],
                "label": cand["label"],
                "boxes": boxes,
            }
        )

    return {
        "id": round_data["id"],
        "fps": round_data["fps"],
        "frames": frame_urls,
        "green_index": green,
        "candidates": candidates,
    }


def reveal_view(
    round_data: dict[str, Any],
    *,
    frame_url_prefix: str = "/frames/",
    frames_complete: bool = True,
    up_to_index: int | None = None,
) -> dict[str, Any]:
    """Build the reveal payload (green_index..end, optionally truncated for live drip)."""
    green = round_data["green_index"]
    end = len(round_data["frames"]) - 1 if up_to_index is None else up_to_index
    end = max(green, min(end, len(round_data["frames"]) - 1))

    frames = round_data["frames"][green : end + 1]
    frame_urls = [
        f if f.startswith("http://") or f.startswith("https://") or f.startswith("/")
        else f"{frame_url_prefix}{f}"
        for f in frames
    ]

    candidates = []
    for cand in round_data["candidates"]:
        boxes = [b for b in cand["boxes"] if green <= b["frame"] <= end]
        # Remap frame indices stay absolute (same as source) so finish_frame_index matches.
        candidates.append(
            {
                "track_id": cand["track_id"],
                "label": cand["label"],
                "boxes": boxes,
            }
        )

    finish = {str(k): v for k, v in round_data["finish_frame_index"].items()}

    out = {
        "frames": frame_urls,
        "frames_complete": frames_complete and end >= len(round_data["frames"]) - 1,
        "candidates": candidates,
        "finish_line": round_data["finish_line"],
        "finish_frame_index": finish,
    }
    if round_data.get("finish_crossing_time"):
        out["finish_crossing_time"] = {
            str(k): v for k, v in round_data["finish_crossing_time"].items()
        }
    return out


def label_for_track(round_data: dict[str, Any], track_id: int) -> str | None:
    for cand in round_data["candidates"]:
        if int(cand["track_id"]) == int(track_id):
            return cand["label"]
    return None
