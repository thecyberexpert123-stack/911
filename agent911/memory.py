"""Persistent SQLite storage for memory, experience, provenance, task state, logs, and custom capabilities."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent911.config import redact_secrets
from agent911.models import (
    ExperienceRecord,
    ImprovementProposal,
    ImprovementStatus,
    MemoryCategory,
    MemoryConfidence,
    MemoryRecord,
    SelfImprovementRunRecord,
    TaskOutcome,
    TaskState,
    new_id,
    utc_now_iso,
)


def _tokenize_keywords(text: str) -> List[str]:
    """Extract normalized alphanumeric tokens of length >= 3 for relevance matching."""
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "into", "when",
        "what", "where", "which", "have", "has", "had", "will", "would",
        "could", "should", "can", "are", "was", "were", "been", "being",
        "task", "please", "use", "using", "make", "create",
    }
    words = re.findall(r"[a-zA-Z0-9_\.]{3,}", text.lower())
    seen = set()
    tokens = []
    for w in words:
        if w not in stop_words and w not in seen:
            seen.add(w)
            tokens.append(w)
    return tokens


class PersistentMemoryStore:
    """SQLite-backed store for memories, experiences, task states, and observability logs."""

    def __init__(
        self,
        db_path: str | Path,
        secrets_to_redact: Optional[List[str]] = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.secrets_to_redact = secrets_to_redact or []
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    task_id TEXT,
                    source TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    superseded_by TEXT,
                    provenance_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_cat_key ON memories(category, key);
                CREATE INDEX IF NOT EXISTS idx_memories_task ON memories(task_id);

                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    plan_summary TEXT NOT NULL,
                    capabilities_used_json TEXT NOT NULL,
                    operations_performed_json TEXT NOT NULL,
                    actual_results_json TEXT NOT NULL,
                    verification_summary_json TEXT NOT NULL,
                    live_test_summary_json TEXT NOT NULL,
                    unexpected_failures_json TEXT NOT NULL,
                    identified_causes_json TEXT NOT NULL,
                    corrections_attempted_json TEXT NOT NULL,
                    successful_remedies_json TEXT NOT NULL,
                    remaining_limitations_json TEXT NOT NULL,
                    reusable_procedures_json TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experiences_task ON experiences(task_id);

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    original_instruction TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS execution_logs (
                    log_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    step_id TEXT,
                    capability_id TEXT,
                    operation TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT,
                    duration_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_logs_task ON execution_logs(task_id);

                CREATE TABLE IF NOT EXISTS custom_capabilities (
                    capability_id TEXT PRIMARY KEY,
                    module_path TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    live_tested INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS improvement_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS self_improvement_runs (
                    run_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    final_disposition TEXT NOT NULL,
                    run_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )
            # Perform safe column migrations for existing Phase 1 databases
            mem_cols = {
                r["name"]
                for r in conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "status" not in mem_cols:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'"
                )
            if "superseded_by" not in mem_cols:
                conn.execute("ALTER TABLE memories ADD COLUMN superseded_by TEXT")
            if "provenance_json" not in mem_cols:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '[]'"
                )

            exp_cols = {
                r["name"]
                for r in conn.execute("PRAGMA table_info(experiences)").fetchall()
            }
            if "metadata_json" not in exp_cols:
                conn.execute(
                    "ALTER TABLE experiences ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )

    def _row_to_memory(self, r: sqlite3.Row) -> MemoryRecord:
        keys = r.keys()
        return MemoryRecord(
            memory_id=r["memory_id"],
            category=MemoryCategory(r["category"]),
            key=r["key"],
            content=r["content"],
            task_id=r["task_id"],
            source=r["source"],
            evidence=json.loads(r["evidence_json"]),
            confidence=MemoryConfidence(r["confidence"]),
            status=r["status"] if "status" in keys else "ACTIVE",
            superseded_by=r["superseded_by"] if "superseded_by" in keys else None,
            provenance_history=(
                json.loads(r["provenance_json"]) if "provenance_json" in keys else []
            ),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    # ------------------------------------------------------------------
    # Memory operations (with provenance, conflict resolution, and expiration)
    # ------------------------------------------------------------------

    def store_memory(
        self,
        category: MemoryCategory | str,
        key: str,
        content: str,
        task_id: Optional[str] = None,
        source: str = "agent",
        evidence: Optional[Dict[str, Any]] = None,
        confidence: MemoryConfidence | str = MemoryConfidence.OBSERVED,
    ) -> MemoryRecord:
        """
        Store or update a memory record, deduplicating on active (category, key)
        while preserving prior conflicting values in provenance_history.
        """
        cat = MemoryCategory(category) if isinstance(category, str) else category
        conf = MemoryConfidence(confidence) if isinstance(confidence, str) else confidence
        safe_content = str(redact_secrets(content, self.secrets_to_redact))
        safe_evidence = redact_secrets(evidence or {}, self.secrets_to_redact)
        now = utc_now_iso()

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM memories
                WHERE category = ? AND status = 'ACTIVE' AND (key = ? OR content = ?)
                ORDER BY updated_at DESC LIMIT 1
                """,
                (cat.value, key, safe_content),
            ).fetchone()

            if existing is not None:
                mem_id = existing["memory_id"]
                merged_evidence = json.loads(existing["evidence_json"])
                if isinstance(safe_evidence, dict):
                    merged_evidence.update(safe_evidence)

                provenance: List[Dict[str, Any]] = json.loads(
                    existing["provenance_json"] or "[]"
                )
                if existing["content"] != safe_content:
                    provenance.append(
                        {
                            "previous_content": existing["content"],
                            "previous_source": existing["source"],
                            "previous_confidence": existing["confidence"],
                            "replaced_at": now,
                        }
                    )

                conn.execute(
                    """
                    UPDATE memories
                    SET key = ?, content = ?, task_id = COALESCE(?, task_id),
                        source = ?, evidence_json = ?, confidence = ?,
                        provenance_json = ?, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        key,
                        safe_content,
                        task_id,
                        source,
                        json.dumps(merged_evidence),
                        conf.value,
                        json.dumps(provenance),
                        now,
                        mem_id,
                    ),
                )
                return MemoryRecord(
                    memory_id=mem_id,
                    category=cat,
                    key=key,
                    content=safe_content,
                    task_id=task_id or existing["task_id"],
                    source=source,
                    evidence=merged_evidence,
                    confidence=conf,
                    status="ACTIVE",
                    superseded_by=None,
                    provenance_history=provenance,
                    created_at=existing["created_at"],
                    updated_at=now,
                )

            mem_id = new_id("mem")
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, category, key, content, task_id, source,
                    evidence_json, confidence, status, superseded_by,
                    provenance_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', NULL, '[]', ?, ?)
                """,
                (
                    mem_id,
                    cat.value,
                    key,
                    safe_content,
                    task_id,
                    source,
                    json.dumps(safe_evidence),
                    conf.value,
                    now,
                    now,
                ),
            )
            return MemoryRecord(
                memory_id=mem_id,
                category=cat,
                key=key,
                content=safe_content,
                task_id=task_id,
                source=source,
                evidence=safe_evidence,
                confidence=conf,
                status="ACTIVE",
                superseded_by=None,
                provenance_history=[],
                created_at=now,
                updated_at=now,
            )

    def supersede_memory(
        self,
        old_memory_id: str,
        new_content: str,
        reason: str = "Updated with newer evidence",
        source: str = "agent",
        confidence: MemoryConfidence | str = MemoryConfidence.VERIFIED,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryRecord]:
        """
        Explicitly mark an inaccurate or outdated memory record as SUPERSEDED and insert
        a replacement record retaining full provenance.
        """
        now = utc_now_iso()
        conf = MemoryConfidence(confidence) if isinstance(confidence, str) else confidence
        safe_content = str(redact_secrets(new_content, self.secrets_to_redact))
        safe_evidence = redact_secrets(evidence or {}, self.secrets_to_redact)

        with self._connect() as conn:
            old_row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (old_memory_id,)
            ).fetchone()
            if old_row is None:
                return None

            new_mem_id = new_id("mem")
            prev_prov = json.loads(old_row["provenance_json"] or "[]")
            prev_prov.append(
                {
                    "superseded_memory_id": old_memory_id,
                    "previous_content": old_row["content"],
                    "previous_confidence": old_row["confidence"],
                    "reason": reason,
                    "superseded_at": now,
                }
            )

            conn.execute(
                """
                UPDATE memories
                SET status = 'SUPERSEDED', superseded_by = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (new_mem_id, now, old_memory_id),
            )

            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, category, key, content, task_id, source,
                    evidence_json, confidence, status, superseded_by,
                    provenance_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', NULL, ?, ?, ?)
                """,
                (
                    new_mem_id,
                    old_row["category"],
                    old_row["key"],
                    safe_content,
                    old_row["task_id"],
                    source,
                    json.dumps(safe_evidence),
                    conf.value,
                    json.dumps(prev_prov),
                    now,
                    now,
                ),
            )
            return MemoryRecord(
                memory_id=new_mem_id,
                category=MemoryCategory(old_row["category"]),
                key=old_row["key"],
                content=safe_content,
                task_id=old_row["task_id"],
                source=source,
                evidence=safe_evidence,
                confidence=conf,
                status="ACTIVE",
                superseded_by=None,
                provenance_history=prev_prov,
                created_at=now,
                updated_at=now,
            )

    def mark_memory_obsolete(self, memory_id: str, reason: str = "Obsolete") -> bool:
        """Mark a memory record as OBSOLETE without deleting historical evidence."""
        now = utc_now_iso()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return False
            prov = json.loads(row["provenance_json"] or "[]")
            prov.append({"marked_obsolete_at": now, "reason": reason})
            conn.execute(
                """
                UPDATE memories
                SET status = 'OBSOLETE', provenance_json = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (json.dumps(prov), now, memory_id),
            )
            return True

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        confidence: Optional[MemoryConfidence | str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryRecord]:
        """Update an existing memory record by ID while recording provenance."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None

            new_content = (
                str(redact_secrets(content, self.secrets_to_redact))
                if content is not None
                else row["content"]
            )
            new_conf = (
                (
                    MemoryConfidence(confidence).value
                    if isinstance(confidence, str)
                    else confidence.value
                )
                if confidence is not None
                else row["confidence"]
            )
            merged_evidence = json.loads(row["evidence_json"])
            if evidence:
                merged_evidence.update(redact_secrets(evidence, self.secrets_to_redact))
            now = utc_now_iso()

            prov = json.loads(row["provenance_json"] or "[]")
            if new_content != row["content"]:
                prov.append(
                    {
                        "previous_content": row["content"],
                        "previous_confidence": row["confidence"],
                        "updated_at": now,
                    }
                )

            conn.execute(
                """
                UPDATE memories
                SET content = ?, confidence = ?, evidence_json = ?,
                    provenance_json = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (
                    new_content,
                    new_conf,
                    json.dumps(merged_evidence),
                    json.dumps(prov),
                    now,
                    memory_id,
                ),
            )
            return MemoryRecord(
                memory_id=memory_id,
                category=MemoryCategory(row["category"]),
                key=row["key"],
                content=new_content,
                task_id=row["task_id"],
                source=row["source"],
                evidence=merged_evidence,
                confidence=MemoryConfidence(new_conf),
                status=row["status"],
                superseded_by=row["superseded_by"],
                provenance_history=prov,
                created_at=row["created_at"],
                updated_at=now,
            )

    def retrieve_memories(
        self,
        query: Optional[str] = None,
        category: Optional[MemoryCategory | str] = None,
        key: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 10,
        include_inactive: bool = False,
    ) -> List[MemoryRecord]:
        """Retrieve relevant active memories using structured filters and keyword scoring."""
        clauses: List[str] = []
        params: List[Any] = []

        if not include_inactive:
            clauses.append("status = 'ACTIVE'")
        if category is not None:
            cat_val = (
                category.value if isinstance(category, MemoryCategory) else str(category)
            )
            clauses.append("category = ?")
            params.append(cat_val)
        if key is not None:
            clauses.append("key = ?")
            params.append(key)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM memories {where_sql} ORDER BY updated_at DESC LIMIT ?"
        params.append(max(limit * 5, 50))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = [self._row_to_memory(r) for r in rows]

        if not query:
            return records[:limit]

        tokens = _tokenize_keywords(query)
        if not tokens:
            return records[:limit]

        scored: List[tuple[int, MemoryRecord]] = []
        q_lower = query.lower()
        for rec in records:
            haystack = f"{rec.key} {rec.content}".lower()
            score = 0
            if q_lower in haystack:
                score += 5
            for tok in tokens:
                if tok in haystack:
                    score += 2
            if score > 0:
                if rec.confidence == MemoryConfidence.VERIFIED:
                    score += 1
                scored.append((score, rec))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [rec for _, rec in scored[:limit]]

    # ------------------------------------------------------------------
    # Experience operations (evidence-backed, with superseding support)
    # ------------------------------------------------------------------

    def record_experience(self, experience: ExperienceRecord) -> ExperienceRecord:
        """Persist an ExperienceRecord, avoiding duplicate entries for the same task_id."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT experience_id FROM experiences WHERE task_id = ?",
                (experience.task_id,),
            ).fetchone()
            exp_id = existing["experience_id"] if existing else experience.experience_id
            experience.experience_id = exp_id

            extra_meta = {
                "task_type": experience.task_type,
                "evidence_supporting_outcome": redact_secrets(
                    experience.evidence_supporting_outcome, self.secrets_to_redact
                ),
                "applicable_conditions": experience.applicable_conditions,
                "lesson_confidence": experience.lesson_confidence,
                "superseded": experience.superseded,
                "confirmed_causes": redact_secrets(
                    experience.confirmed_causes, self.secrets_to_redact
                ),
                "hypothesized_causes": redact_secrets(
                    experience.hypothesized_causes, self.secrets_to_redact
                ),
                "times_retrieved": experience.times_retrieved,
                "times_helpful": experience.times_helpful,
                "times_prevented_failure": experience.times_prevented_failure,
                "times_irrelevant": experience.times_irrelevant,
                "times_caused_problem": experience.times_caused_problem,
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO experiences (
                    experience_id, task_id, objective, plan_summary,
                    capabilities_used_json, operations_performed_json,
                    actual_results_json, verification_summary_json,
                    live_test_summary_json, unexpected_failures_json,
                    identified_causes_json, corrections_attempted_json,
                    successful_remedies_json, remaining_limitations_json,
                    reusable_procedures_json, outcome, tags_json,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience.experience_id,
                    experience.task_id,
                    str(redact_secrets(experience.objective, self.secrets_to_redact)),
                    str(redact_secrets(experience.plan_summary, self.secrets_to_redact)),
                    json.dumps(experience.capabilities_used),
                    json.dumps(
                        redact_secrets(
                            experience.operations_performed, self.secrets_to_redact
                        )
                    ),
                    json.dumps(
                        redact_secrets(experience.actual_results, self.secrets_to_redact)
                    ),
                    json.dumps(experience.verification_summary),
                    json.dumps(experience.live_test_summary),
                    json.dumps(
                        redact_secrets(
                            experience.unexpected_failures, self.secrets_to_redact
                        )
                    ),
                    json.dumps(
                        redact_secrets(experience.identified_causes, self.secrets_to_redact)
                    ),
                    json.dumps(
                        redact_secrets(
                            experience.corrections_attempted, self.secrets_to_redact
                        )
                    ),
                    json.dumps(
                        redact_secrets(
                            experience.successful_remedies, self.secrets_to_redact
                        )
                    ),
                    json.dumps(experience.remaining_limitations),
                    json.dumps(experience.reusable_procedures),
                    experience.outcome.value,
                    json.dumps(experience.tags),
                    json.dumps(extra_meta),
                    experience.created_at,
                ),
            )
        return experience

    def mark_experience_superseded(self, experience_id: str) -> bool:
        """Mark a historical experience lesson as superseded so it is not applied blindly."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM experiences WHERE experience_id = ?",
                (experience_id,),
            ).fetchone()
            if row is None:
                return False
            meta = json.loads(row["metadata_json"] or "{}")
            meta["superseded"] = True
            conn.execute(
                "UPDATE experiences SET metadata_json = ? WHERE experience_id = ?",
                (json.dumps(meta), experience_id),
            )
            return True

    def record_experience_usefulness(
        self,
        experience_id: str,
        retrieved: bool = False,
        helpful: bool = False,
        prevented_failure: bool = False,
        irrelevant: bool = False,
        caused_problem: bool = False,
    ) -> bool:
        """
        Update practical usefulness metrics for a retrieved ExperienceRecord (Section 5.4).
        Used to improve future experience retrieval and strategy selection.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM experiences WHERE experience_id = ?",
                (experience_id,),
            ).fetchone()
            if row is None:
                return False
            meta = json.loads(row["metadata_json"] or "{}")
            if retrieved:
                meta["times_retrieved"] = int(meta.get("times_retrieved", 0)) + 1
            if helpful:
                meta["times_helpful"] = int(meta.get("times_helpful", 0)) + 1
            if prevented_failure:
                meta["times_prevented_failure"] = (
                    int(meta.get("times_prevented_failure", 0)) + 1
                )
            if irrelevant:
                meta["times_irrelevant"] = int(meta.get("times_irrelevant", 0)) + 1
            if caused_problem:
                meta["times_caused_problem"] = (
                    int(meta.get("times_caused_problem", 0)) + 1
                )
            conn.execute(
                "UPDATE experiences SET metadata_json = ? WHERE experience_id = ?",
                (json.dumps(meta), experience_id),
            )
            return True

    def _row_to_experience(self, row: sqlite3.Row) -> ExperienceRecord:
        meta = (
            json.loads(row["metadata_json"] or "{}")
            if "metadata_json" in row.keys()
            else {}
        )
        return ExperienceRecord(
            experience_id=row["experience_id"],
            task_id=row["task_id"],
            objective=row["objective"],
            plan_summary=row["plan_summary"],
            capabilities_used=json.loads(row["capabilities_used_json"]),
            operations_performed=json.loads(row["operations_performed_json"]),
            actual_results=json.loads(row["actual_results_json"]),
            verification_summary=json.loads(row["verification_summary_json"]),
            live_test_summary=json.loads(row["live_test_summary_json"]),
            unexpected_failures=json.loads(row["unexpected_failures_json"]),
            identified_causes=json.loads(row["identified_causes_json"]),
            confirmed_causes=list(meta.get("confirmed_causes", [])),
            hypothesized_causes=list(meta.get("hypothesized_causes", [])),
            corrections_attempted=json.loads(row["corrections_attempted_json"]),
            successful_remedies=json.loads(row["successful_remedies_json"]),
            remaining_limitations=json.loads(row["remaining_limitations_json"]),
            reusable_procedures=json.loads(row["reusable_procedures_json"]),
            outcome=TaskOutcome(row["outcome"]),
            tags=json.loads(row["tags_json"]),
            task_type=meta.get("task_type", "general"),
            evidence_supporting_outcome=meta.get("evidence_supporting_outcome", {}),
            applicable_conditions=meta.get("applicable_conditions", []),
            lesson_confidence=meta.get("lesson_confidence", "OBSERVED"),
            superseded=bool(meta.get("superseded", False)),
            times_retrieved=int(meta.get("times_retrieved", 0)),
            times_helpful=int(meta.get("times_helpful", 0)),
            times_prevented_failure=int(meta.get("times_prevented_failure", 0)),
            times_irrelevant=int(meta.get("times_irrelevant", 0)),
            times_caused_problem=int(meta.get("times_caused_problem", 0)),
            created_at=row["created_at"],
        )

    def retrieve_experiences(
        self,
        query: Optional[str] = None,
        task_id: Optional[str] = None,
        outcome: Optional[TaskOutcome | str] = None,
        limit: int = 5,
        include_superseded: bool = False,
    ) -> List[ExperienceRecord]:
        """
        Retrieve relevant task experiences ranked by contextual similarity,
        verification confidence, and measured practical usefulness (Section 5.3 & 5.4).
        """
        clauses: List[str] = []
        params: List[Any] = []

        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if outcome is not None:
            out_val = outcome.value if isinstance(outcome, TaskOutcome) else str(outcome)
            clauses.append("outcome = ?")
            params.append(out_val)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM experiences {where_sql} ORDER BY created_at DESC LIMIT ?"
        params.append(max(limit * 6, 40))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = [self._row_to_experience(r) for r in rows]
        if not include_superseded:
            records = [r for r in records if not r.superseded]

        if not query:
            return records[:limit]

        tokens = _tokenize_keywords(query)
        if not tokens:
            return records[:limit]

        scored: List[tuple[int, ExperienceRecord]] = []
        q_lower = query.lower()
        for rec in records:
            searchable = " ".join(
                [
                    rec.objective,
                    rec.plan_summary,
                    " ".join(rec.capabilities_used),
                    " ".join(rec.unexpected_failures),
                    " ".join(rec.identified_causes),
                    " ".join(rec.confirmed_causes),
                    " ".join(rec.successful_remedies),
                    " ".join(rec.reusable_procedures),
                    " ".join(rec.tags),
                ]
            ).lower()
            score = 0
            if q_lower in searchable:
                score += 6
            for tok in tokens:
                if tok in searchable:
                    score += 2
            if score > 0:
                if rec.lesson_confidence == "VERIFIED_EXECUTION":
                    score += 2
                # Practical usefulness adjustments (Section 5.4)
                score += min(rec.times_helpful * 2 + rec.times_prevented_failure * 3, 8)
                score -= rec.times_caused_problem * 6 + rec.times_irrelevant * 2
                if score > 0:
                    scored.append((score, rec))

        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [rec for _, rec in scored[:limit]]

    # ------------------------------------------------------------------
    # Task persistence
    # ------------------------------------------------------------------

    def save_task(self, task: TaskState) -> None:
        """Persist full structured TaskState for continuity and interruption recovery."""
        task.touch()
        safe_dict = redact_secrets(task.to_dict(), self.secrets_to_redact)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks (
                    task_id, original_instruction, stage, outcome,
                    state_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    str(redact_secrets(task.original_instruction, self.secrets_to_redact)),
                    task.current_execution_state.value,
                    task.final_outcome.value,
                    json.dumps(safe_dict),
                    task.created_at,
                    task.updated_at,
                ),
            )

    def load_task(self, task_id: str) -> Optional[TaskState]:
        """Load a persisted TaskState by task_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            return TaskState.from_dict(json.loads(row["state_json"]))

    def list_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List summary records of recent tasks."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, original_instruction, stage, outcome, created_at, updated_at
                FROM tasks ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Execution & Structured Event Logging
    # ------------------------------------------------------------------

    def record_execution_log(
        self,
        task_id: str,
        operation: str,
        success: bool,
        step_id: Optional[str] = None,
        capability_id: Optional[str] = None,
        input_data: Any = None,
        output_data: Any = None,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> str:
        """Record a redacted observability log entry."""
        log_id = new_id("log")
        safe_in = redact_secrets(
            input_data if input_data is not None else {}, self.secrets_to_redact
        )
        safe_out = redact_secrets(
            output_data if output_data is not None else {}, self.secrets_to_redact
        )
        safe_err = redact_secrets(error, self.secrets_to_redact) if error else None

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO execution_logs (
                    log_id, task_id, step_id, capability_id, operation,
                    input_json, output_json, success, error, duration_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    task_id,
                    step_id,
                    capability_id,
                    operation,
                    json.dumps(safe_in, default=str)[:4000],
                    json.dumps(safe_out, default=str)[:4000],
                    1 if success else 0,
                    safe_err,
                    float(duration_ms),
                    utc_now_iso(),
                ),
            )
        return log_id

    def get_execution_logs(self, task_id: str) -> List[Dict[str, Any]]:
        """Retrieve all execution log records for a given task."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM execution_logs WHERE task_id = ? ORDER BY timestamp ASC",
                (task_id,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["input"] = json.loads(d.pop("input_json"))
                d["output"] = json.loads(d.pop("output_json"))
                d["success"] = bool(d["success"])
                results.append(d)
            return results

    # ------------------------------------------------------------------
    # Custom capability persistence
    # ------------------------------------------------------------------

    def save_custom_capability(
        self,
        capability_id: str,
        module_path: str,
        class_name: str,
        purpose: str,
        verified: bool,
        live_tested: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist registration metadata for an organically developed capability."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO custom_capabilities (
                    capability_id, module_path, class_name, purpose,
                    verified, live_tested, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capability_id,
                    module_path,
                    class_name,
                    purpose,
                    1 if verified else 0,
                    1 if live_tested else 0,
                    json.dumps(metadata or {}),
                    utc_now_iso(),
                ),
            )

    def list_custom_capabilities(self) -> List[Dict[str, Any]]:
        """Return all persisted custom capabilities."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM custom_capabilities ORDER BY created_at ASC"
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["verified"] = bool(d["verified"])
                d["live_tested"] = bool(d["live_tested"])
                d["metadata"] = json.loads(d.pop("metadata_json"))
                out.append(d)
            return out

    # ------------------------------------------------------------------
    # Improvement Proposals & Controlled Self-Improvement Runs (Section 6 & 7)
    # ------------------------------------------------------------------

    def save_improvement_proposal(self, proposal: ImprovementProposal) -> ImprovementProposal:
        """Persist an ImprovementProposal in SQLite."""
        proposal.updated_at = utc_now_iso()
        safe_dict = redact_secrets(proposal.to_dict(), self.secrets_to_redact)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO improvement_proposals (
                    proposal_id, title, category, priority, status,
                    proposal_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.proposal_id,
                    proposal.title,
                    proposal.category,
                    proposal.priority,
                    proposal.status.value,
                    json.dumps(safe_dict),
                    proposal.created_at,
                    proposal.updated_at,
                ),
            )
        return proposal

    def get_improvement_proposal(
        self, proposal_id: str
    ) -> Optional[ImprovementProposal]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT proposal_json FROM improvement_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                return None
            return ImprovementProposal.from_dict(json.loads(row["proposal_json"]))

    def list_improvement_proposals(
        self, status: Optional[ImprovementStatus | str] = None
    ) -> List[ImprovementProposal]:
        with self._connect() as conn:
            if status is not None:
                st_val = (
                    status.value
                    if isinstance(status, ImprovementStatus)
                    else str(status)
                )
                rows = conn.execute(
                    "SELECT proposal_json FROM improvement_proposals WHERE status = ? ORDER BY updated_at DESC",
                    (st_val,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT proposal_json FROM improvement_proposals ORDER BY updated_at DESC"
                ).fetchall()
            return [
                ImprovementProposal.from_dict(json.loads(r["proposal_json"]))
                for r in rows
            ]

    def save_self_improvement_run(
        self, run_record: SelfImprovementRunRecord
    ) -> SelfImprovementRunRecord:
        safe_dict = redact_secrets(run_record.to_dict(), self.secrets_to_redact)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO self_improvement_runs (
                    run_id, proposal_id, final_disposition, run_json, timestamp
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_record.run_id,
                    run_record.proposal_id,
                    run_record.final_disposition,
                    json.dumps(safe_dict),
                    run_record.timestamp,
                ),
            )
        return run_record

    def list_self_improvement_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_json FROM self_improvement_runs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [json.loads(r["run_json"]) for r in rows]
