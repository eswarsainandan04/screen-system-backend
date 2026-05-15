import mimetypes
import os

try:
    import boto3
except ImportError as exc:
    raise RuntimeError("boto3 is required. Install with pip install boto3.") from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(REPO_ROOT, ".env")
DEFAULT_UPLOAD_PREFIX = "user1/session1/"


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


def upload_resume_file(
    file_obj,
    filename: str,
    dest_prefix: str = DEFAULT_UPLOAD_PREFIX,
    bucket: str | None = None,
) -> str:
    if not filename:
        raise ValueError("Filename is required for upload.")

    dest_prefix = dest_prefix or ""
    if dest_prefix and not dest_prefix.endswith("/"):
        dest_prefix += "/"

    object_key = f"{dest_prefix}{os.path.basename(filename)}"
    client = _get_s3_client()
    bucket_name = _resolve_bucket(bucket)

    content_type, _ = mimetypes.guess_type(filename)
    extra_args = {"ContentType": content_type} if content_type else None

    if extra_args:
        client.upload_fileobj(file_obj, bucket_name, object_key, ExtraArgs=extra_args)
    else:
        client.upload_fileobj(file_obj, bucket_name, object_key)

    return object_key


def upload_resume(
    file_path: str,
    dest_prefix: str = DEFAULT_UPLOAD_PREFIX,
    bucket: str | None = None,
) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as handle:
        return upload_resume_file(handle, os.path.basename(file_path), dest_prefix, bucket)
