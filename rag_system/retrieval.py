import io
import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT in sys.path:
	sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

try:
	import chromadb
except ImportError as exc:
	raise RuntimeError("chromadb is required. Install with pip install chromadb.") from exc

try:
	import requests
except ImportError as exc:
	raise RuntimeError("requests is required. Install with pip install requests.") from exc

try:
	from dotenv import load_dotenv
except ImportError:
	load_dotenv = None

from ingestion.resume_upload import upload_resume_file
from rag_system.ban_patterns import BAN_PATTERNS

ENV_PATH = os.path.join(REPO_ROOT, ".env")
CHROMA_DB_PATH = os.path.join(REPO_ROOT, "chroma_db")
COLLECTIONS = ("ml_book_collection", "ds_book_collection")
DEFAULT_QUERY_KEY = "user1/session1/query.json"
DEFAULT_OUTPUT_PREFIX = "user1/session1/"
DEFAULT_OUTPUT_FILENAME = "context.json"
DEFAULT_RESULTS = 10
DEFAULT_FINAL_RESULTS = 6
DEFAULT_METADATA_KEY = "user1/session1/meta_data.json"

BAN_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in BAN_PATTERNS]
STOPWORDS = {
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"in",
	"into",
	"is",
	"it",
	"of",
	"on",
	"or",
	"that",
	"the",
	"to",
	"with",
	"using",
}


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


def _download_text(object_key: str, bucket: str | None = None) -> str:
	client = _get_s3_client()
	bucket_name = _resolve_bucket(bucket)

	response = client.get_object(Bucket=bucket_name, Key=object_key)
	body = response.get("Body")
	if body is None:
		raise RuntimeError("Unable to read retrieval query from storage.")

	data = body.read()
	if not data:
		raise RuntimeError("Downloaded retrieval query is empty.")

	return data.decode("utf-8").strip()


def _download_json(object_key: str, bucket: str | None = None) -> dict | list:
	text = _download_text(object_key, bucket=bucket)
	try:
		return json.loads(text)
	except json.JSONDecodeError as exc:
		raise RuntimeError("Downloaded JSON is invalid.") from exc


def _load_queries(query_key: str, bucket: str | None = None) -> list[str]:
	data = _download_json(query_key, bucket=bucket)
	queries = data.get("queries") if isinstance(data, dict) else data
	if not isinstance(queries, list):
		raise RuntimeError("Query JSON must include a 'queries' list.")

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
		raise RuntimeError("Query JSON contains no valid queries.")
	return cleaned


def _get_hf_headers() -> tuple[str, dict]:
	token = _get_env("HF_TOKEN", ("HUGGINGFACE_API_TOKEN", "HUGGINGFACE_TOKEN"))
	if not token:
		raise RuntimeError("Missing HF_TOKEN in environment.")

	api_url = _get_env(
		"HF_EMBEDDINGS_URL",
		(
			"HUGGINGFACE_EMBEDDINGS_URL",
			"HF_API_URL",
		),
	)
	if not api_url:
		api_url = (
			"https://router.huggingface.co/"
			"hf-inference/models/"
			"BAAI/bge-small-en-v1.5"
		)

	headers = {"Authorization": f"Bearer {token}"}
	return api_url, headers


def _generate_embedding(text: str) -> list[float]:
	api_url, headers = _get_hf_headers()
	payload = {"inputs": text}

	response = requests.post(api_url, headers=headers, json=payload, timeout=120)
	if not response.ok:
		raise RuntimeError(
			f"HF embedding request failed: {response.status_code} {response.text}"
		)

	data = response.json()
	if isinstance(data, dict) and data.get("error"):
		raise RuntimeError(f"HF embedding error: {data['error']}")

	if isinstance(data, dict) and "embeddings" in data:
		embedding = data["embeddings"]
	else:
		embedding = data

	if (
		isinstance(embedding, list)
		and embedding
		and isinstance(embedding[0], list)
	):
		if len(embedding) == 1:
			embedding = embedding[0]
		else:
			raise ValueError("Expected a single embedding, got multiple.")

	if not isinstance(embedding, list):
		raise RuntimeError("HF embedding response format is invalid.")

	return embedding


def _query_collection(collection, embedding: list[float], n_results: int) -> dict:
	return collection.query(
		query_embeddings=[embedding],
		n_results=n_results,
		include=["documents", "metadatas", "distances"],
	)


def _get_distance_threshold() -> float | None:
	value = _get_env("CHROMA_DISTANCE_THRESHOLD", ("RETRIEVAL_DISTANCE_THRESHOLD",))
	if not value:
		return None
	try:
		return float(value)
	except ValueError as exc:
		raise RuntimeError("CHROMA_DISTANCE_THRESHOLD must be a number.") from exc


def _get_min_query_term_matches() -> int:
	value = _get_env("MIN_QUERY_TERM_MATCHES", ("QUERY_TERM_MATCHES",))
	if not value:
		return 1
	try:
		minimum = int(value)
	except ValueError as exc:
		raise RuntimeError("MIN_QUERY_TERM_MATCHES must be an integer.") from exc
	return max(1, minimum)


def _is_banned(text: str) -> bool:
	for regex in BAN_REGEXES:
		if regex.search(text):
			return True
	return False


def _clean_text(text: str) -> str:
	cleaned = re.sub(r"\[IMAGE:[^\]]+\]", " ", text)
	cleaned = re.sub(r"=+\s*PAGE\s*\d+\s*=+", " ", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"https?://\S+", " ", cleaned)
	cleaned = re.sub(r"\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", cleaned)
	cleaned = re.sub(r"\bIn\[\d+\]:", " ", cleaned)
	cleaned = re.sub(r"\s+", " ", cleaned)
	return cleaned.strip()


def _extract_query_terms(query: str) -> list[str]:
	terms = re.findall(r"[A-Za-z0-9]+", query.lower())
	filtered = [term for term in terms if term not in STOPWORDS and len(term) >= 3]
	if filtered:
		return filtered
	return [term for term in terms if len(term) >= 3]


def _query_matches_text(query: str, text: str) -> bool:
	terms = _extract_query_terms(query)
	if not terms:
		return True
	required = min(_get_min_query_term_matches(), len(terms))
	text_lower = text.lower()
	count = 0
	for term in terms:
		if re.search(rf"\b{re.escape(term)}\b", text_lower):
			count += 1
			if count >= required:
				return True
	return False

def _filter_context(text: str) -> str | None:
	cleaned = _clean_text(text)
	if not cleaned:
		return None
	if len(cleaned) < 120:
		return None
	if _is_banned(cleaned):
		return None
	return cleaned


def retrieve_and_store_information(
	query_key: str = DEFAULT_QUERY_KEY,
	dest_prefix: str | None = None,
	filename: str | None = None,
	bucket: str | None = None,
) -> tuple[str, str]:
	queries = _load_queries(query_key, bucket=bucket)
	client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
	distance_threshold = _get_distance_threshold()

	output_payload: dict[str, str | list[str]] = {}
	for query in queries:
		embedding = _generate_embedding(query)
		chunks: list[str] = []
		seen: set[str] = set()

		for collection_name in COLLECTIONS:
			collection = client.get_collection(name=collection_name)
			results = _query_collection(
				collection,
				embedding,
				n_results=DEFAULT_RESULTS,
			)

			documents = results.get("documents", [[]])
			docs = documents[0] if documents else []
			distances = results.get("distances", [[]])
			dists = distances[0] if distances else []
			for idx, doc in enumerate(docs):
				if not doc:
					continue
				distance = dists[idx] if idx < len(dists) else None
				if (
					distance_threshold is not None
					and distance is not None
					and distance > distance_threshold
				):
					continue
				cleaned = _filter_context(doc)
				if not cleaned:
					continue
				if not _query_matches_text(query, cleaned):
					continue
				key = cleaned.lower()
				if key in seen:
					continue
				seen.add(key)
				chunks.append(cleaned)

		if not chunks:
			continue
		if len(chunks) == 1:
			output_payload[query] = chunks[0]
		else:
			output_payload[query] = chunks

	output_text = json.dumps(output_payload, ensure_ascii=True, indent=2) + "\n"

	output_filename = filename or DEFAULT_OUTPUT_FILENAME
	output_prefix = dest_prefix or os.getenv("PARSED_RESUME_PREFIX", DEFAULT_OUTPUT_PREFIX)
	buffer = io.BytesIO(output_text.encode("utf-8"))

	object_key = upload_resume_file(
		buffer,
		output_filename,
		dest_prefix=output_prefix,
		bucket=bucket,
	)

	return object_key, output_text


if __name__ == "__main__":
	key, _ = retrieve_and_store_information()
	print(f"Context uploaded to {key}")
