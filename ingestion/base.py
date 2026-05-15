import io
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from db import get_db_connection
from ingestion.resume_parsing import parse_resume_bytes
from ingestion.resume_upload import upload_resume_file
from ingestion.role_selection import select_role_and_update_metadata


router = APIRouter(prefix="/resume", tags=["resume"])


@router.post("/upload")
async def upload_resume(
    resume: UploadFile = File(...),
    selected_role: str = Form(...),
    userid: str | None = Form(None),
    email: str | None = Form(None),
) -> dict:
    if not resume.filename:
        raise HTTPException(status_code=400, detail="Resume file is required.")

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    resolved_userid = (userid or "").strip()
    if not resolved_userid:
        lookup_email = (email or "").strip()
        if not lookup_email:
            raise HTTPException(status_code=400, detail="userid or email is required.")

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT userid FROM users WHERE email = %s", (lookup_email,))
                row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="User not found.")

        resolved_userid = str(row[0])

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    dest_prefix = f"{resolved_userid}/{timestamp}/"

    try:
        data = await resume.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        upload_stream = io.BytesIO(data)
        resume_key = upload_resume_file(upload_stream, resume.filename, dest_prefix)
        metadata_key, metadata, resume_text = parse_resume_bytes(
            data,
            dest_prefix=dest_prefix,
        )
        role_key, _ = select_role_and_update_metadata(
            resume_text,
            metadata,
            selected_role=selected_role,
            dest_prefix=dest_prefix,
        )
        from llm.query_generation import generate_retrieval_query
        from rag_system.retrieval import retrieve_and_store_information

        query_key, _ = generate_retrieval_query(
            metadata_key=metadata_key,
            dest_prefix=dest_prefix,
        )
        context_key, _ = retrieve_and_store_information(
            query_key=query_key,
            dest_prefix=dest_prefix,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return {
        "resume_key": resume_key,
        "metadata_key": metadata_key,
        "role_key": role_key,
        "query_key": query_key,
        "context_key": context_key,
        "session_prefix": dest_prefix,
        "userid": resolved_userid,
    }