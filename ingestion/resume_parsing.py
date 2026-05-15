import io
import json
import os

try:
	from groq import Groq
except ImportError as exc:
	raise RuntimeError("groq is required. Install with pip install groq.") from exc

try:
	from PyPDF2 import PdfReader
except ImportError as exc:
	raise RuntimeError("PyPDF2 is required. Install with pip install PyPDF2.") from exc

try:
	from dotenv import load_dotenv
except ImportError:
	load_dotenv = None

from ingestion.resume_upload import upload_resume_file

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(REPO_ROOT, ".env")
DEFAULT_PARSED_PREFIX = "user1/session1/"
DEFAULT_PARSED_FILENAME = "meta_data.json"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TEXT_CHARS = 12000


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


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
	reader = PdfReader(io.BytesIO(pdf_bytes))
	chunks = []
	for page in reader.pages:
		text = page.extract_text()
		if text:
			chunks.append(text)

	extracted = "\n".join(chunks).strip()
	if not extracted:
		raise RuntimeError("Unable to extract text from the PDF.")
	return extracted


def _build_prompt(resume_text: str) -> str:
	trimmed = resume_text[:MAX_TEXT_CHARS]
	return (
		"You are an AI resume parser. Produce JSON only.\n\n"
		"Return this shape:\n"
		"{\n"
		"  \"resume\": {\n"
		"    \"skills\": [],\n"
		"    \"Technologies\": [],\n"
		"    \"projects\": [],\n"
		"    \"experience\": {\n"
		"      \"companies\": [\n"
		"        {\n"
		"          \"companyname\": \"\",\n"
		"          \"role\": \"\",\n"
		"          \"technology\": \"\",\n"
		"          \"years\": \"\"\n"
		"        }\n"
		"      ],\n"
		"      \"total_experience\": \"\"\n"
		"    },\n"
		"    \"Domains\": []\n"
		"  }\n"
		"}\n\n"
		"Rules:\n"
		"- Use concise bullet phrases in arrays (strings).\n"
		"- For experience, list companies with role, technology, and years.\n"
		"- total_experience is the total years of experience (string).\n"
		"- Only use resume content.\n"
		"- If a section is missing, use [\"Not specified\"].\n\n"
		f"RESUME TEXT:\n{trimmed}"
	)


def _coerce_metadata(parsed: dict) -> dict:
	resume = parsed.get("resume") if isinstance(parsed, dict) else None
	if not isinstance(resume, dict):
		raise RuntimeError("Parsed resume JSON missing 'resume' object.")

	def _get_value(payload: dict, *keys: str) -> object:
		for key in keys:
			if key in payload:
				return payload.get(key)
		return None

	def _list_or_default(value: object) -> list[str]:
		if isinstance(value, list) and value:
			return [str(item).strip() for item in value if str(item).strip()]
		return ["Not specified"]

	def _company_from_item(item: object) -> dict:
		if isinstance(item, dict):
			return {
				"companyname": str(
					item.get("companyname")
					or item.get("company")
					or item.get("name")
					or "Not specified"
				).strip(),
				"role": str(
					item.get("role")
					or item.get("title")
					or "Not specified"
				).strip(),
				"technology": str(
					item.get("technology")
					or item.get("technologies")
					or item.get("tech")
					or "Not specified"
				).strip(),
				"years": str(
					item.get("years")
					or item.get("duration")
					or "Not specified"
				).strip(),
			}
		if isinstance(item, str) and item.strip():
			return {
				"companyname": item.strip(),
				"role": "Not specified",
				"technology": "Not specified",
				"years": "Not specified",
			}
		return {
			"companyname": "Not specified",
			"role": "Not specified",
			"technology": "Not specified",
			"years": "Not specified",
		}

	def _normalize_experience(value: object) -> dict:
		companies_value = None
		total_value = None
		if isinstance(value, dict):
			companies_value = value.get("companies") or value.get("company") or value.get("items")
			total_value = value.get("total_experience") or value.get("total experience")
		elif isinstance(value, list):
			companies_value = value

		companies = []
		if isinstance(companies_value, list) and companies_value:
			companies = [_company_from_item(item) for item in companies_value]
		else:
			companies = [_company_from_item(None)]

		total_experience = (
			str(total_value).strip() if isinstance(total_value, str) and total_value.strip() else "Not specified"
		)
		return {
			"companies": companies,
			"total_experience": total_experience,
		}

	projects_value = _get_value(
		resume,
		"projects",
		"Projects",
		"projects & Experience",
	)
	projects = _list_or_default(projects_value)

	experience_value = _get_value(resume, "experience", "Experience")
	experience = _normalize_experience(experience_value)

	return {
		"resume": {
			"skills": _list_or_default(_get_value(resume, "skills", "Skills")),
			"Technologies": _list_or_default(_get_value(resume, "Technologies", "technologies")),
			"projects": projects,
			"experience": experience,
			"Domains": _list_or_default(_get_value(resume, "Domains", "domains")),
		},
		"Selected role": None,
		"Technical Knowledge": None,
		"Role related tech": [],
	}


def parse_resume_bytes(
	pdf_bytes: bytes,
	dest_prefix: str | None = None,
	filename: str | None = None,
	bucket: str | None = None,
	model: str | None = None,
) -> tuple[str, dict, str]:
	if not pdf_bytes:
		raise ValueError("No PDF content provided.")

	groq_api_key = os.getenv("GROQ_API_KEY")
	if not groq_api_key:
		raise RuntimeError("GROQ_API_KEY is not set.")

	resume_text = _extract_text_from_pdf_bytes(pdf_bytes)
	prompt = _build_prompt(resume_text)

	client = Groq(api_key=groq_api_key)
	response = client.chat.completions.create(
		model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
		messages=[{"role": "user", "content": prompt}],
		temperature=0.2,
	)

	parsed_text = response.choices[0].message.content.strip()
	if not parsed_text:
		raise RuntimeError("Groq returned an empty response.")

	json_start = parsed_text.find("{")
	json_end = parsed_text.rfind("}")
	if json_start == -1 or json_end == -1 or json_end <= json_start:
		raise RuntimeError("Groq returned invalid JSON for resume metadata.")

	try:
		parsed_json = json.loads(parsed_text[json_start : json_end + 1])
	except json.JSONDecodeError as exc:
		raise RuntimeError("Groq returned invalid JSON for resume metadata.") from exc

	metadata = _coerce_metadata(parsed_json)

	parsed_filename = filename or DEFAULT_PARSED_FILENAME
	parsed_prefix = dest_prefix or os.getenv("PARSED_RESUME_PREFIX", DEFAULT_PARSED_PREFIX)
	buffer = io.BytesIO(json.dumps(metadata, ensure_ascii=True, indent=2).encode("utf-8"))

	object_key = upload_resume_file(
		buffer,
		parsed_filename,
		dest_prefix=parsed_prefix,
		bucket=bucket,
	)

	return object_key, metadata, resume_text


def parse_resume_file(
	file_path: str,
	dest_prefix: str | None = None,
	filename: str | None = None,
	bucket: str | None = None,
	model: str | None = None,
) -> tuple[str, dict, str]:
	if not os.path.isfile(file_path):
		raise FileNotFoundError(f"File not found: {file_path}")

	with open(file_path, "rb") as handle:
		return parse_resume_bytes(
			handle.read(),
			dest_prefix=dest_prefix,
			filename=filename,
			bucket=bucket,
			model=model,
		)
