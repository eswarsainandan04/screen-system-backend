import io
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

try:
    from groq import Groq
except ImportError as exc:
    raise RuntimeError("groq is required. Install with pip install groq.") from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ingestion.resume_upload import upload_resume_file
ENV_PATH = os.path.join(REPO_ROOT, ".env")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_METADATA_KEY = "user1/session1/meta_data.json"
DEFAULT_QUERY_PREFIX = "user1/session1/"
DEFAULT_QUERY_FILENAME = "query.json"


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


if load_dotenv:
    load_dotenv(ENV_PATH)
_load_env_file(ENV_PATH)


def _get_env(primary: str, alternatives: tuple[str, ...] = ()) -> str | None:
    for key in (primary, *alternatives):
        value = os.getenv(key)
        if value:
            return value
    return None


def _resolve_endpoint() -> str | None:
    endpoint = _get_env("SUPABASE_S3_ENDPOINT", ("SUPABASE_ENDPOINT", "S3_ENDPOINT"))
    if endpoint:
        return endpoint

    supabase_url = _get_env("SUPABASE_URL", ("SUPABASE_PROJECT_URL",))
    if supabase_url:
        return supabase_url.rstrip("/") + "/storage/v1/s3"

    return None


def _get_s3_client():
    endpoint = _resolve_endpoint()
    if not endpoint:
        raise RuntimeError(
            "Missing SUPABASE_S3_ENDPOINT (or SUPABASE_URL for auto-derivation)."
        )

    access_key = _get_env(
        "ACCESS_KEY_ID",
        (
            "AWS_ACCESS_KEY_ID",
            "Access_key_ID",
            "SUPABASE_ACCESS_KEY_ID",
        ),
    )
    secret_key = _get_env(
        "SECRET_ACCESS_KEY",
        (
            "AWS_SECRET_ACCESS_KEY",
            "Secret_access_key",
            "SUPABASE_SECRET_ACCESS_KEY",
        ),
    )

    if not access_key or not secret_key:
        raise RuntimeError("Missing S3 access key/secret in environment.")

    region = _get_env("AWS_DEFAULT_REGION", ("AWS_REGION", "SUPABASE_S3_REGION"))

    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def _resolve_bucket(bucket: str | None) -> str:
    resolved_bucket = bucket or _get_env("SUPABASE_BUCKET", ("S3_BUCKET", "BUCKET"))
    if not resolved_bucket:
        raise RuntimeError("Missing SUPABASE_BUCKET in environment.")
    return resolved_bucket


def _download_metadata(metadata_key: str, bucket: str | None = None) -> dict:
    client = _get_s3_client()
    bucket_name = _resolve_bucket(bucket)

    response = client.get_object(Bucket=bucket_name, Key=metadata_key)
    body = response.get("Body")
    if body is None:
        raise RuntimeError("Unable to read metadata from storage.")

    data = body.read()
    if not data:
        raise RuntimeError("Downloaded metadata is empty.")

    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Metadata is not valid JSON.") from exc


def _build_prompt(metadata: dict) -> str:
    focused_profile = {
        "Selected role": metadata.get("Selected role"),
        "Technical Knowledge": metadata.get("Technical Knowledge"),
        "Role related tech": metadata.get("Role related tech"),
    }
    profile = {key: value for key, value in focused_profile.items() if value}
    if not profile:
        profile = metadata
    payload = json.dumps(profile, ensure_ascii=True, indent=2)
    return (
        "You are an AI retrieval planner for a RAG system.\n\n"
        "Generate a set of short, implementation-focused "
        "technical concept queries suitable for ChromaDB "
        "semantic vector search.\n\n"
        "Constraints:\n"
        "- Output MUST be valid JSON only.\n"
        '- Schema: {"queries": ["...", ...]}\n'
        "- 12 to 20 queries.\n"
        "- Each query is a concise phrase (2-8 words).\n"
        "- No paragraphs, no numbering, no explanations.\n"
        "- No project summaries, product descriptions, or generic buzzwords.\n"
        "- Focus on ML/DS textbook concepts and tooling tied to the profile.\n\n"
        "Use ONLY the candidate profile fields below to infer relevance.\n\n"
        "Candidate Profile:\n"
        f"{payload}\n\n"
        "Return ONLY the JSON object."
    )


def _parse_queries(raw_text: str) -> list[str]:
    text = raw_text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("Groq did not return valid JSON.")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Groq did not return valid JSON.") from exc

    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list):
        raise RuntimeError("Groq JSON must include a 'queries' list.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in queries:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)

    if not cleaned:
        raise RuntimeError("Groq returned an empty queries list.")
    return cleaned


def generate_retrieval_query(
    metadata_key: str = DEFAULT_METADATA_KEY,
    dest_prefix: str | None = None,
    filename: str | None = None,
    bucket: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    metadata = _download_metadata(metadata_key, bucket=bucket)
    prompt = _build_prompt(metadata)

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    query_text = response.choices[0].message.content.strip()
    if not query_text:
        raise RuntimeError("Groq returned an empty response.")

    queries = _parse_queries(query_text)
    query_payload = json.dumps({"queries": queries}, ensure_ascii=True, indent=2)

    query_filename = filename or DEFAULT_QUERY_FILENAME
    query_prefix = dest_prefix or os.getenv("PARSED_RESUME_PREFIX", DEFAULT_QUERY_PREFIX)
    buffer = io.BytesIO(query_payload.encode("utf-8"))

    object_key = upload_resume_file(
        buffer,
        query_filename,
        dest_prefix=query_prefix,
        bucket=bucket,
    )

    return object_key, query_payload


if __name__ == "__main__":
    key, _ = generate_retrieval_query()
    print(f"Retrieval query uploaded to {key}")
