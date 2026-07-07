"""Mock vector-graph memory database for Evolve.

This module mirrors the conceptual pipeline of **Cognee** (``add -> cognify ->
search -> prune``) and its four operations ``remember``, ``recall``,
``improve`` and ``forget``, but is implemented as a fully self-contained local
store so the engine runs with **no external services**.

Backends:
- **SQLite** indexes every "Memory Frame" and "Knowledge Cluster".
- A newline-delimited JSON **file ledger** (``frames.ledger.jsonl``) mirrors the
  frames for human-readable persistence.
- A lightweight hashed-bag-of-tokens **vector** (dim 384) per frame powers
  cosine-similarity recall, with a keyword-overlap boost.

Public pipeline surface (module-level ``store`` singleton):
- ``remember_message(content, mode, metadata)``  -> remember()
- ``recall_memory(query, top_k)``                -> recall()
- ``synthesize_memory(summarizer)``              -> improve()
- ``delete_specific_memory(frame_id)``           -> forget() (granular)
- ``forget_memory()``                            -> forget() (Nuke Protocol)
"""
from __future__ import annotations

import os
import re
import math
import shutil
import hashlib
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone

from . import utils

# Where the local memory graph lives.
DEFAULT_BASE_DIR = os.path.join(os.getcwd(), ".evolve_memory")
# Dimensionality of the hashed bag-of-tokens embedding.
VECT_DIM = 384
# Stopwords excluded from keyword extraction.
STOPWORDS = set(
    "the a an of to in on for and or but is are was were be been being with as at "
    "by this that it its from your you i we they he she our my me him her them "
    "not no so if then than too very can could should would may might will just "
    "about into over under again more most some any all each".split()
)


class MemoryStore:
    """A self-contained vector-graph memory database."""

    # ----------------------------------------------------------------- #
    # Setup
    # ----------------------------------------------------------------- #
    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        self.db_path = os.path.join(self.base_dir, "memory.db")
        self.ledger_path = os.path.join(self.base_dir, "frames.ledger.jsonl")
        self.clusters_path = os.path.join(self.base_dir, "clusters.json")
        self._init_db()

    def _connect(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _cursor(self):
        """Context-managed connection that commits & closes cleanly."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._cursor() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_frames (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    content    TEXT    NOT NULL,
                    mode       TEXT    NOT NULL,
                    timestamp  TEXT    NOT NULL,
                    embedding  TEXT    NOT NULL,
                    metadata   TEXT    NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_clusters (
                    cluster_id     INTEGER PRIMARY KEY,
                    summary        TEXT    NOT NULL,
                    mode           TEXT    NOT NULL,
                    source_frames  TEXT    NOT NULL,
                    keywords       TEXT    NOT NULL,
                    frame_count    INTEGER NOT NULL,
                    created_at     TEXT    NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_frames_mode ON memory_frames(mode)"
            )

    # ----------------------------------------------------------------- #
    # Embedding (hashed bag-of-tokens -> dense unit vector)
    # ----------------------------------------------------------------- #
    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * VECT_DIM
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % VECT_DIM
            # weight slightly by token length so meaningful terms matter more
            vec[h] += 1.0 + min(len(tok), 10) * 0.01
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    @staticmethod
    def _keyword_overlap(query: str, content: str) -> float:
        q = set(re.findall(r"[a-z0-9]+", (query or "").lower())) - STOPWORDS
        c = set(re.findall(r"[a-z0-9]+", (content or "").lower()))
        if not q:
            return 0.0
        return len(q & c) / len(q)

    @staticmethod
    def _top_keywords(text: str, k: int = 8) -> list[str]:
        toks = [t for t in re.findall(r"[a-z]{3,}", (text or "").lower()) if t not in STOPWORDS]
        return [w for w, _ in Counter(toks).most_common(k)]

    # ----------------------------------------------------------------- #
    # remember()
    # ----------------------------------------------------------------- #
    def remember_message(self, content: str, mode: str, metadata: dict | None = None) -> dict | None:
        """Ingest a message as an indexed Memory Frame (remember stage)."""
        content = (content or "").strip()
        if not content:
            return None
        embedding = self._embed(content)
        now = utils.now_iso()
        meta = dict(metadata or {})
        meta.setdefault("role", "user")
        meta.setdefault("saved_at", now)

        with self._cursor() as conn:
            cur = conn.execute(
                "INSERT INTO memory_frames (content, mode, timestamp, embedding, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (content, mode, now, utils.safe_json_dumps(embedding), utils.safe_json_dumps(meta)),
            )
            frame_id = cur.lastrowid

        frame = {
            "frame_id": frame_id,
            "content": content,
            "mode": mode,
            "timestamp": now,
            "metadata": meta,
        }

        # Mirror to the persistent file ledger.
        try:
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(utils.safe_json_dumps(frame) + "\n")
        except OSError:
            pass

        return frame

    # ----------------------------------------------------------------- #
    # recall()
    # ----------------------------------------------------------------- #
    def recall_memory(self, query: str, top_k: int = 5) -> dict:
        """Retrieve the most relevant Memory Frames + Knowledge Clusters.

        Returns a dict with ``frames`` (each carrying a ``[Frame #N]`` citation),
        ``clusters`` and a ready-to-inject ``context_text``.
        """
        qvec = self._embed(query or "")

        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_frames ORDER BY id DESC LIMIT 500"
            ).fetchall()

        scored: list[tuple[float, object]] = []
        for r in rows:
            vec = utils.safe_json_loads(r["embedding"], [])
            if not vec:
                continue
            sim = self._cosine(qvec, vec)
            boost = self._keyword_overlap(query or "", r["content"])
            score = sim + 0.25 * boost
            if score > 0.0:
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        result_frames: list[dict] = []
        context_lines: list[str] = []
        for score, r in top:
            fid = r["id"]
            citation = f"[Frame #{fid}]"
            result_frames.append(
                {
                    "frame_id": fid,
                    "citation": citation,
                    "content": r["content"],
                    "mode": r["mode"],
                    "timestamp": r["timestamp"],
                    "score": round(score, 4),
                    "metadata": utils.safe_json_loads(r["metadata"], {}),
                }
            )
            context_lines.append(
                f"{citation} (mode: {r['mode']}, {utils.short_time(r['timestamp'])}): "
                f"{utils.truncate(r['content'], 320)}"
            )

        clusters = self.get_all_clusters()
        cluster_lines: list[str] = []
        for cl in clusters:
            cluster_lines.append(
                f"[Cluster #{cl['cluster_id']}]: {utils.truncate(cl['summary'], 280)}"
            )

        context_text = ""
        if context_lines:
            context_text += "## Relevant Memory Frames\n" + "\n".join(context_lines)
        if cluster_lines:
            context_text += "\n\n## Knowledge Clusters\n" + "\n".join(cluster_lines)

        return {"frames": result_frames, "clusters": clusters, "context_text": context_text}

    # ----------------------------------------------------------------- #
    # improve()  (synthesize_memory)
    # ----------------------------------------------------------------- #
    def synthesize_memory(self, summarizer=None) -> dict:
        """Consolidate chronological frames into dense Knowledge Clusters.

        Frames are grouped by mode, then sub-clustered by vector similarity, and
        each sub-cluster is summarised (via LLM when available, else a heuristic)
        into a single high-density node — optimising downstream context length.
        """
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_frames ORDER BY id ASC"
            ).fetchall()

        if not rows:
            return {
                "clusters_created": 0,
                "clusters": [],
                "message": "No memory frames to synthesize yet.",
            }

        # Refresh: replace any previously synthesised clusters.
        with self._cursor() as conn:
            conn.execute("DELETE FROM knowledge_clusters")

        by_mode: dict[str, list] = {}
        for r in rows:
            by_mode.setdefault(r["mode"], []).append(r)

        created: list[dict] = []
        for mode, mrows in by_mode.items():
            for grp in self._group_rows_by_similarity(mrows, threshold=0.12):
                ids = [g["id"] for g in grp]
                contents = [g["content"] for g in grp]
                keywords = self._top_keywords(" ".join(contents), 8)
                summary = self._summarize(contents, mode, summarizer)
                cluster = {
                    "cluster_id": self._next_cluster_id(),
                    "summary": summary,
                    "mode": mode,
                    "source_frames": ids,
                    "keywords": keywords,
                    "created_at": utils.now_iso(),
                    "frame_count": len(ids),
                }
                self._save_cluster(cluster)
                created.append(cluster)

        return {
            "clusters_created": len(created),
            "clusters": created,
            "message": (
                f"Synthesized {len(created)} knowledge cluster(s) "
                f"from {len(rows)} frame(s) across {len(by_mode)} mode(s)."
            ),
        }

    def _group_rows_by_similarity(self, rows: list, threshold: float = 0.12) -> list[list]:
        """Greedy first-fit clustering by cosine similarity."""
        vecs = [(r, utils.safe_json_loads(r["embedding"], [])) for r in rows]
        buckets: list[list] = []
        for r, v in vecs:
            placed = False
            for b in buckets:
                if b and self._cosine(v, b[0][1]) >= threshold:
                    b.append((r, v))
                    placed = True
                    break
            if not placed:
                buckets.append([(r, v)])
        return [[x[0] for x in b] for b in buckets if b]

    def _summarize(self, contents: list[str], mode: str, summarizer=None) -> str:
        """Summarise a group of frames: caller-supplied fn -> LLM -> heuristic."""
        if summarizer:
            try:
                out = summarizer(contents, mode)
                if out and out.strip():
                    return out.strip()
            except Exception:
                pass
        try:
            from . import llm as _llm  # lazy to avoid any import cycle
            out = _llm.summarize_cluster(contents, mode)
            if out and out.strip():
                return out.strip()
        except Exception:
            pass
        keywords = self._top_keywords(" ".join(contents), 6)
        first = utils.truncate(contents[0], 150) if contents else ""
        themes = ", ".join(keywords) if keywords else "general context"
        return (
            f"[{mode}] Consolidated {len(contents)} frame(s) into one knowledge node. "
            f"Key themes: {themes}. Representative excerpt: {first}"
        )

    def _next_cluster_id(self) -> int:
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT MAX(cluster_id) AS m FROM knowledge_clusters"
            ).fetchone()
        m = row["m"] if row and row["m"] is not None else 0
        return (m or 0) + 1

    def _save_cluster(self, cluster: dict) -> None:
        with self._cursor() as conn:
            conn.execute(
                "INSERT INTO knowledge_clusters "
                "(cluster_id, summary, mode, source_frames, keywords, frame_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cluster["cluster_id"],
                    cluster["summary"],
                    cluster["mode"],
                    utils.safe_json_dumps(cluster["source_frames"]),
                    utils.safe_json_dumps(cluster["keywords"]),
                    cluster["frame_count"],
                    cluster["created_at"],
                ),
            )

    # ----------------------------------------------------------------- #
    # forget()  (granular + nuke)
    # ----------------------------------------------------------------- #
    def delete_specific_memory(self, frame_id: int) -> dict:
        """Granular node pruning — delete a single Memory Frame."""
        try:
            fid = int(frame_id)
        except (TypeError, ValueError):
            return {"deleted": 0, "frame_id": frame_id, "message": "Invalid frame id."}
        with self._cursor() as conn:
            cur = conn.execute("DELETE FROM memory_frames WHERE id = ?", (fid,))
            deleted = cur.rowcount
        return {
            "deleted": deleted,
            "frame_id": fid,
            "message": (
                f"Deleted frame #{fid}." if deleted else f"Frame #{fid} not found."
            ),
        }

    def forget_memory(self) -> dict:
        """Nuke Protocol — wipe the SQLite DB, ledger and the memory directory."""
        removed: list[str] = []
        for p in (self.db_path, self.ledger_path, self.clusters_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
                    removed.append(os.path.basename(p))
            except OSError:
                pass
        try:
            if os.path.isdir(self.base_dir):
                shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass
        # Re-initialise a fresh, empty store in place.
        os.makedirs(self.base_dir, exist_ok=True)
        self._init_db()
        return {
            "nuked": True,
            "removed": removed,
            "message": "Memory database wiped and reinitialised (Nuke Protocol complete).",
        }

    # ----------------------------------------------------------------- #
    # Read helpers
    # ----------------------------------------------------------------- #
    def _row_to_frame(self, r) -> dict:
        return {
            "frame_id": r["id"],
            "content": r["content"],
            "mode": r["mode"],
            "timestamp": r["timestamp"],
            "metadata": utils.safe_json_loads(r["metadata"], {}),
        }

    def _row_to_cluster(self, r) -> dict:
        return {
            "cluster_id": r["cluster_id"],
            "summary": r["summary"],
            "mode": r["mode"],
            "source_frames": utils.safe_json_loads(r["source_frames"], []),
            "keywords": utils.safe_json_loads(r["keywords"], []),
            "frame_count": r["frame_count"],
            "created_at": r["created_at"],
        }

    def get_all_frames(self) -> list[dict]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_frames ORDER BY id ASC"
            ).fetchall()
        return [self._row_to_frame(r) for r in rows]

    def get_all_clusters(self) -> list[dict]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_clusters ORDER BY cluster_id ASC"
            ).fetchall()
        return [self._row_to_cluster(r) for r in rows]

    def get_frame(self, frame_id: int) -> dict | None:
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT * FROM memory_frames WHERE id = ?", (frame_id,)
            ).fetchone()
        return self._row_to_frame(row) if row else None

    def get_stats(self) -> dict:
        with self._cursor() as conn:
            nframes = conn.execute(
                "SELECT COUNT(*) AS n FROM memory_frames"
            ).fetchone()["n"]
            nclusters = conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge_clusters"
            ).fetchone()["n"]
            mode_rows = conn.execute(
                "SELECT mode, COUNT(*) AS n FROM memory_frames GROUP BY mode"
            ).fetchall()
        by_mode = {r["mode"]: r["n"] for r in mode_rows}
        return {
            "frames": nframes,
            "clusters": nclusters,
            "by_mode": by_mode,
            "base_dir": self.base_dir,
        }


# Module-level singleton used across the app.
store = MemoryStore()
