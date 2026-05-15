import io
import json
import os
import sys
from typing import Any

try:
	from botocore.exceptions import ClientError
except ImportError as exc:
	raise RuntimeError("boto3 is required. Install with pip install boto3.") from exc

try:
	from dotenv import load_dotenv
except ImportError:
	load_dotenv = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT in sys.path:
	sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

ENV_PATH = os.path.join(REPO_ROOT, ".env")
DEFAULT_METADATA_FILENAME = "meta_data.json"
DEFAULT_CONTEXT_FILENAME = "context.json"
DEFAULT_CONVERSATION_FILENAME = "conversation.json"


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


def _normalize_part(value: str) -> str:
	return (value or "").strip().strip("/")


def build_session_prefix(user_id: str, session_id: str) -> str:
	user_part = _normalize_part(user_id)
	session_part = _normalize_part(session_id)
	if not user_part or not session_part:
		raise ValueError("Both user_id and session_id are required.")
	return f"{user_part}/{session_part}/"


def build_object_key(user_id: str, session_id: str, filename: str) -> str:
	prefix = build_session_prefix(user_id, session_id)
	filename = (filename or "").lstrip("/")
	if not filename:
		raise ValueError("filename is required for storage key.")
	return f"{prefix}{filename}"


def _download_text(
	object_key: str,
	bucket: str | None = None,
	allow_missing: bool = False,
) -> str | None:
	client = _get_s3_client()
	bucket_name = _resolve_bucket(bucket)

	try:
		response = client.get_object(Bucket=bucket_name, Key=object_key)
	except ClientError as exc:
		error_code = exc.response.get("Error", {}).get("Code")
		if allow_missing and error_code in {"NoSuchKey", "404", "NotFound"}:
			return None
		raise

	body = response.get("Body")
	if body is None:
		raise RuntimeError("Unable to read object from storage.")

	data = body.read()
	if not data:
		if allow_missing:
			return None
		raise RuntimeError("Downloaded object is empty.")

	return data.decode("utf-8").strip()


def _download_json(
	object_key: str,
	bucket: str | None = None,
	allow_missing: bool = False,
) -> dict | list | None:
	text = _download_text(object_key, bucket=bucket, allow_missing=allow_missing)
	if text is None:
		return None
	try:
		return json.loads(text)
	except json.JSONDecodeError as exc:
		raise RuntimeError("Downloaded JSON is invalid.") from exc


def _upload_json(payload: Any, object_key: str, bucket: str | None = None) -> None:
	client = _get_s3_client()
	bucket_name = _resolve_bucket(bucket)
	encoded = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
	buffer = io.BytesIO(encoded)
	client.upload_fileobj(
		buffer,
		bucket_name,
		object_key,
		ExtraArgs={"ContentType": "application/json"},
	)


def read_metadata(user_id: str, session_id: str, bucket: str | None = None) -> dict:
	key = build_object_key(user_id, session_id, DEFAULT_METADATA_FILENAME)
	data = _download_json(key, bucket=bucket)
	if not isinstance(data, dict):
		raise RuntimeError("Metadata JSON is missing or invalid.")
	return data


def read_context(user_id: str, session_id: str, bucket: str | None = None) -> dict | list:
	key = build_object_key(user_id, session_id, DEFAULT_CONTEXT_FILENAME)
	data = _download_json(key, bucket=bucket)
	if not isinstance(data, (dict, list)):
		raise RuntimeError("Context JSON is missing or invalid.")
	return data


def load_conversation(
	user_id: str,
	session_id: str,
	bucket: str | None = None,
) -> list[dict[str, str]]:
	key = build_object_key(user_id, session_id, DEFAULT_CONVERSATION_FILENAME)
	data = _download_json(key, bucket=bucket, allow_missing=True)
	if data is None:
		return []
	if isinstance(data, list):
		cleaned: list[dict[str, str]] = []
		for item in data:
			if not isinstance(item, dict):
				continue
			sender = str(item.get("sender", "")).strip()
			message = str(item.get("message", "")).strip()
			if sender and message:
				cleaned.append({"sender": sender, "message": message})
		return cleaned
	return []


def save_conversation(
	user_id: str,
	session_id: str,
	conversation: list[dict[str, str]],
	bucket: str | None = None,
) -> str:
	key = build_object_key(user_id, session_id, DEFAULT_CONVERSATION_FILENAME)
	_upload_json(conversation, key, bucket=bucket)
	return key
