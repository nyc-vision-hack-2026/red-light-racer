"""RoundSource interface and StaticRoundSource implementation.

Architectural seams (do not simplify away):
1. Rounds pulled one at a time.
2. Guess resolution is asynchronous by contract.
3. Round data read only through RoundSource — never a global dict in handlers.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from app.rounds import (
    assert_round_valid,
    label_for_track,
    prompt_view,
    reveal_view,
)
from app.scoring import clamp_elapsed_ms, compute_points


RoundOfferStatus = Literal["ready", "waiting", "session_complete"]
ResolutionStatus = Literal["pending", "resolved", "void"]


@dataclass
class RoundOffer:
    status: RoundOfferStatus
    round: dict[str, Any] | None = None
    retry_after_ms: int | None = None


@dataclass
class Resolution:
    status: ResolutionStatus
    retry_after_ms: int | None = None
    correct: bool | None = None
    winner_track_id: int | None = None
    winner_label: str | None = None
    points: int = 0
    reveal: dict[str, Any] | None = None
    reason: str | None = None


@dataclass
class _Guess:
    track_id: int | None  # None = timed out / null guess
    elapsed_ms: int
    streak: int


@dataclass
class _Session:
    target_rounds: int
    served: list[str] = field(default_factory=list)
    counted_rounds: int = 0  # non-void rounds toward target
    guesses: dict[str, _Guess] = field(default_factory=dict)
    # First resolution poll must return pending (static dress rehearsal for live).
    resolution_polls: dict[str, int] = field(default_factory=dict)
    counted_ids: set[str] = field(default_factory=set)


class RoundSource(Protocol):
    def next_round(self, session_id: str) -> RoundOffer: ...

    def resolution(self, session_id: str, round_id: str) -> Resolution: ...


class StaticRoundSource:
    """Serves pre-computed rounds from rounds.json. Zero inference."""

    def __init__(
        self,
        rounds_path: Path,
        *,
        frame_url_prefix: str = "/frames/",
        # Live dress-rehearsal knobs — leave defaults for production.
        force_pending_first_poll: bool = True,
        pending_retry_after_ms: int = 0,
        drip_reveal: bool = False,
        drip_retry_after_ms: int = 6000,
    ) -> None:
        self._lock = threading.Lock()
        self._frame_url_prefix = frame_url_prefix
        self._force_pending_first_poll = force_pending_first_poll
        self._pending_retry_after_ms = pending_retry_after_ms
        self._drip_reveal = drip_reveal
        self._drip_retry_after_ms = drip_retry_after_ms

        raw = json.loads(Path(rounds_path).read_text())
        self._camera_id = raw.get("camera_id")
        self._rounds: dict[str, dict[str, Any]] = {}
        self._round_order: list[str] = []
        for r in raw["rounds"]:
            assert_round_valid(r)
            self._rounds[r["id"]] = r
            self._round_order.append(r["id"])
        if not self._rounds:
            raise ValueError("rounds.json contains no rounds")

        self._sessions: dict[str, _Session] = {}

    def create_session(self, target_rounds: int = 10) -> str:
        sid = f"s_{secrets.token_hex(3)}"
        with self._lock:
            self._sessions[sid] = _Session(target_rounds=target_rounds)
        return sid

    def get_prompt(self, round_id: str) -> dict[str, Any] | None:
        round_data = self._rounds.get(round_id)
        if round_data is None:
            return None
        return prompt_view(round_data, self._frame_url_prefix)

    def submit_guess(
        self,
        session_id: str,
        round_id: str,
        track_id: int | None,
        elapsed_ms: int | float,
        streak: int,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("unknown_session")
            if round_id not in self._rounds:
                raise KeyError("unknown_round")
            if round_id not in session.served:
                raise KeyError("round_not_served")
            session.guesses[round_id] = _Guess(
                track_id=track_id,
                elapsed_ms=clamp_elapsed_ms(elapsed_ms),
                streak=max(0, int(streak)),
            )
            session.resolution_polls[round_id] = 0
        return {"status": "pending", "retry_after_ms": 0}

    def next_round(self, session_id: str) -> RoundOffer:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("unknown_session")

            if session.counted_rounds >= session.target_rounds:
                return RoundOffer(status="session_complete")

            served_set = set(session.served)
            remaining = [rid for rid in self._round_order if rid not in served_set]
            if not remaining:
                return RoundOffer(status="session_complete")

            rid = remaining[0]
            session.served.append(rid)
            prompt = prompt_view(self._rounds[rid], self._frame_url_prefix)
            # Static never returns waiting — clients must still handle it for live.
            return RoundOffer(status="ready", round=prompt)

    def _mark_counted(self, session: _Session, round_id: str) -> None:
        if round_id not in session.counted_ids:
            session.counted_ids.add(round_id)
            session.counted_rounds += 1

    def resolution(self, session_id: str, round_id: str) -> Resolution:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("unknown_session")
            round_data = self._rounds.get(round_id)
            if round_data is None:
                raise KeyError("unknown_round")
            guess = session.guesses.get(round_id)
            if guess is None:
                raise KeyError("no_guess")

            polls = session.resolution_polls.get(round_id, 0) + 1
            session.resolution_polls[round_id] = polls

            # Required: first poll always pending so the live path cannot rot.
            if self._force_pending_first_poll and polls == 1:
                return Resolution(
                    status="pending",
                    retry_after_ms=self._pending_retry_after_ms,
                )

            winner_id = int(round_data["winner_track_id"])
            correct = guess.track_id is not None and int(guess.track_id) == winner_id
            points = compute_points(
                correct=correct,
                elapsed_ms=guess.elapsed_ms,
                streak=guess.streak,
            )
            self._mark_counted(session, round_id)

            green = round_data["green_index"]
            last = len(round_data["frames"]) - 1

            if self._drip_reveal:
                # poll 2 -> green+0 (just green frame), grow by 1 each poll
                up_to = min(last, green + (polls - 2))
                complete = up_to >= last
                reveal = reveal_view(
                    round_data,
                    frame_url_prefix=self._frame_url_prefix,
                    frames_complete=complete,
                    up_to_index=up_to,
                )
                return Resolution(
                    status="resolved",
                    correct=correct,
                    winner_track_id=winner_id,
                    winner_label=label_for_track(round_data, winner_id),
                    points=points,
                    reveal=reveal,
                    retry_after_ms=None if complete else self._drip_retry_after_ms,
                )

            reveal = reveal_view(
                round_data,
                frame_url_prefix=self._frame_url_prefix,
                frames_complete=True,
            )
            return Resolution(
                status="resolved",
                correct=correct,
                winner_track_id=winner_id,
                winner_label=label_for_track(round_data, winner_id),
                points=points,
                reveal=reveal,
            )


def resolution_to_dict(res: Resolution) -> dict[str, Any]:
    if res.status == "pending":
        return {
            "status": "pending",
            "retry_after_ms": res.retry_after_ms if res.retry_after_ms is not None else 2000,
        }
    if res.status == "void":
        return {
            "status": "void",
            "reason": res.reason or "no_finishers",
            "points": 0,
        }
    out: dict[str, Any] = {
        "status": "resolved",
        "correct": res.correct,
        "winner_track_id": res.winner_track_id,
        "winner_label": res.winner_label,
        "points": res.points,
        "reveal": res.reveal,
    }
    if res.retry_after_ms is not None:
        out["retry_after_ms"] = res.retry_after_ms
    return out
