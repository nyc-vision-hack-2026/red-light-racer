"""Firestore leaderboard read/write with in-memory fallback for local/dev."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any


class ScoreStore:
    def submit(self, initials: str, score: int) -> int:
        """Write score, return 1-based rank among all entries."""
        raise NotImplementedError

    def leaderboard(self, limit: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError


class MemoryScoreStore(ScoreStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []

    def submit(self, initials: str, score: int) -> int:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            self._entries.append({"initials": initials, "score": score, "ts": ts})
            ranked = sorted(self._entries, key=lambda e: (-e["score"], e["ts"]))
            for i, e in enumerate(ranked, start=1):
                if e["initials"] == initials and e["score"] == score and e["ts"] == ts:
                    return i
            return len(ranked)

    def leaderboard(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ranked = sorted(self._entries, key=lambda e: (-e["score"], e["ts"]))
            return ranked[:limit]


class FirestoreScoreStore(ScoreStore):
    def __init__(self, collection: str = "scores") -> None:
        from google.cloud import firestore

        self._db = firestore.Client()
        self._col = self._db.collection(collection)
        self._fs = firestore

    def submit(self, initials: str, score: int) -> int:
        doc = {
            "initials": initials,
            "score": score,
            "ts": self._fs.SERVER_TIMESTAMP,
        }
        self._col.add(doc)
        # Rank = 1 + number of scores strictly greater.
        greater = (
            self._col.where("score", ">", score).count().get()
        )
        # count() aggregation returns an AggregateQuerySnapshot list
        try:
            count_val = greater[0][0].value
        except Exception:
            # Fallback: fetch and count client-side
            count_val = sum(1 for _ in self._col.where("score", ">", score).stream())
        return int(count_val) + 1

    def leaderboard(self, limit: int = 20) -> list[dict[str, Any]]:
        q = self._col.order_by("score", direction=self._fs.Query.DESCENDING).limit(limit)
        out: list[dict[str, Any]] = []
        for snap in q.stream():
            d = snap.to_dict() or {}
            ts = d.get("ts")
            if hasattr(ts, "isoformat"):
                ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(ts, "strftime") else ts.isoformat()
            else:
                ts_str = str(ts) if ts else ""
            out.append(
                {
                    "initials": d.get("initials", "???"),
                    "score": int(d.get("score", 0)),
                    "ts": ts_str,
                }
            )
        return out


def build_score_store() -> ScoreStore:
    """Prefer Firestore when credentials exist; else in-memory."""
    if os.environ.get("FORCE_MEMORY_STORE") == "1":
        return MemoryScoreStore()
    try:
        return FirestoreScoreStore()
    except Exception:
        return MemoryScoreStore()
