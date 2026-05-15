"""
sessions.py

FastAPI router for listing and deleting interview sessions.

Endpoints:
  GET  /sessions/list?userid=...          → list all sessions with metadata/insights
  DELETE /sessions/delete                  → delete all files for a session
"""

import os
import sys
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

try:
    from botocore.exceptions import ClientError
except ImportError as exc:
    raise RuntimeError("boto3 is required. Install with pip install boto3.") from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from llm.conversation_chat import (
    _get_s3_client,
    _resolve_bucket,
    _download_json,
    build_object_key,
    build_session_prefix,
)

ENV_PATH = os.path.join(REPO_ROOT, ".env")

if load_dotenv:
    load_dotenv(ENV_PATH)


def _load_env_file(path: str) -> None:
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.environ.get(key):
                os.environ[key] = value


_load_env_file(ENV_PATH)


def _list_user_sessions(user_id: str, bucket: Optional[str] = None) -> list[str]:
    """
    List all session IDs for a user by scanning S3 prefixes under user_id/.
    Returns a list of session_id strings.
    """
    client = _get_s3_client()
    bucket_name = _resolve_bucket(bucket)
    user_prefix = f"{user_id.strip().strip('/')}/"

    paginator = client.get_paginator("list_objects_v2")
    session_ids: set[str] = set()

    for page in paginator.paginate(Bucket=bucket_name, Prefix=user_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefix = cp.get("Prefix", "")
            # prefix looks like "userid/sessionid/"
            parts = prefix.rstrip("/").split("/")
            if len(parts) >= 2:
                session_ids.add(parts[1])

    return sorted(session_ids)


def _get_session_detail(
    user_id: str,
    session_id: str,
    bucket: Optional[str] = None,
) -> dict:
    """
    Load metadata + analysis.json (fast path) for a session list item.
    Falls back to scanning conversation.json for legacy sessions that
    were created before analysis.json existed.
    """
    detail: dict = {
        "sessionid": session_id,
        "role": None,
        "knowledge_level": None,
        "created_at": None,
        "has_insights": False,
        "final_score": None,
        "overall_feedback": None,
        "strengths": [],
        "improvement_areas": [],
        "total_questions": None,
        "correct_answers": None,
    }

    # ── 1. Load metadata ───────────────────────────────────────────────────
    try:
        meta_key = build_object_key(user_id, session_id, "meta_data.json")
        metadata = _download_json(meta_key, bucket=bucket, allow_missing=True)
        if isinstance(metadata, dict):
            detail["role"] = metadata.get("Selected role")
            detail["knowledge_level"] = metadata.get("Technical Knowledge")
    except Exception:
        pass

    # ── 2. Fast path: read analysis.json ──────────────────────────────────
    analysis_loaded = False
    try:
        analysis_key = build_object_key(user_id, session_id, "analysis.json")
        analysis = _download_json(analysis_key, bucket=bucket, allow_missing=True)
        if isinstance(analysis, dict):
            detail["has_insights"] = True
            detail["final_score"] = analysis.get("final_score")
            detail["overall_feedback"] = analysis.get("overall_feedback")
            detail["strengths"] = analysis.get("strengths", [])
            detail["improvement_areas"] = analysis.get("improvement_areas", [])
            detail["total_questions"] = analysis.get("total_questions")
            detail["correct_answers"] = analysis.get("correct_answers")
            # Use the timestamp stored in analysis.json when available
            if analysis.get("time"):
                detail["created_at"] = analysis.get("time")
            analysis_loaded = True
    except Exception:
        pass

    # ── 3. Always read conversation.json for created_at (first msg time) ──
    #       and fall back for legacy sessions without analysis.json
    try:
        conv_key = build_object_key(user_id, session_id, "conversation.json")
        conversation = _download_json(conv_key, bucket=bucket, allow_missing=True)
        if isinstance(conversation, list) and conversation:
            # created_at = first message timestamp
            first = conversation[0]
            if first.get("time") and not detail["created_at"]:
                detail["created_at"] = first.get("time")

            # Legacy fallback: scan for system insights block
            if not analysis_loaded:
                for msg in reversed(conversation):
                    if (
                        isinstance(msg, dict)
                        and msg.get("sender") == "system"
                        and msg.get("type") == "insights"
                    ):
                        detail["has_insights"] = True
                        detail["final_score"] = msg.get("final_score")
                        detail["overall_feedback"] = msg.get("overall_feedback")
                        detail["strengths"] = msg.get("strengths", [])
                        detail["improvement_areas"] = msg.get("improvement_areas", [])
                        detail["total_questions"] = msg.get("total_questions")
                        detail["correct_answers"] = msg.get("correct_answers")
                        break
    except Exception:
        pass

    return detail


def _delete_session_objects(
    user_id: str,
    session_id: str,
    bucket: Optional[str] = None,
) -> int:
    """
    Delete all S3 objects under user_id/session_id/.
    Returns the count of deleted objects.
    """
    client = _get_s3_client()
    bucket_name = _resolve_bucket(bucket)
    prefix = build_session_prefix(user_id, session_id)

    paginator = client.get_paginator("list_objects_v2")
    keys_to_delete: list[dict] = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys_to_delete.append({"Key": obj["Key"]})

    if not keys_to_delete:
        return 0

    # Supabase S3-compatible storage does not support DeleteObjects (bulk delete).
    # Delete each object individually instead.
    deleted = 0
    errors = []
    for obj in keys_to_delete:
        try:
            client.delete_object(Bucket=bucket_name, Key=obj["Key"])
            deleted += 1
        except Exception as exc:
            errors.append(f"{obj['Key']}: {exc}")

    if errors:
        raise RuntimeError(f"S3 delete errors: {errors}")

    return deleted


# ── FastAPI router ──────────────────────────────────────────────────────────

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/list")
def list_sessions(userid: str = Query(..., min_length=1)) -> dict:
    """
    GET /sessions/list?userid=...
    Returns all session summaries for the user, sorted newest first.
    """
    try:
        session_ids = _list_user_sessions(userid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sessions = []
    for sid in session_ids:
        try:
            detail = _get_session_detail(userid, sid)
            sessions.append(detail)
        except Exception:
            # Include a minimal entry so the session still appears
            sessions.append({"sessionid": sid, "has_insights": False})

    # Sort: sessions with created_at descending, then those without
    def sort_key(s):
        ts = s.get("created_at") or ""
        return ts

    sessions.sort(key=sort_key, reverse=True)

    return {"userid": userid, "sessions": sessions, "total": len(sessions)}


class DeleteSessionRequest(BaseModel):
    userid: str = Field(..., min_length=1)
    sessionid: str = Field(..., min_length=1)


@router.delete("/delete")
def delete_session(payload: DeleteSessionRequest) -> dict:
    """
    DELETE /sessions/delete
    Deletes all files for the given session from S3/Supabase.
    """
    try:
        deleted_count = _delete_session_objects(payload.userid, payload.sessionid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "deleted",
        "userid": payload.userid,
        "sessionid": payload.sessionid,
        "objects_deleted": deleted_count,
    }