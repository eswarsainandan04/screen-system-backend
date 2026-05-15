import io
import json
import os

try:
    from groq import Groq
except ImportError as exc:
    raise RuntimeError("groq is required. Install with pip install groq.") from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ingestion.resume_upload import upload_resume_file

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(REPO_ROOT, ".env")
DEFAULT_METADATA_PREFIX = "user1/session1/"
DEFAULT_METADATA_FILENAME = "meta_data.json"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TEXT_CHARS = 12000
ALLOWED_ROLES = (
    "AI/ML Engineer",
    "Data Analyst",
    "Data Science Engineer",
    "Backend Engineer",
)
ALLOWED_KNOWLEDGE = ("weak", "moderate", "strong")


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


def _build_prompt(resume_text: str, selected_role: str | None) -> str:
    trimmed = resume_text[:MAX_TEXT_CHARS]
    roles = ", ".join(ALLOWED_ROLES)
    role_rule = (
        f"- Selected role must be one of: {roles}."
        if not selected_role
        else f"- Selected role is '{selected_role}'. Use it in output."
    )
    return (
        "You analyze resumes and assign a role and skill level. Output JSON only.\n\n"
        "Return this shape:\n"
        "{\n"
        "  \"Selected role\": \"\",\n"
        "  \"Technical Knowledge\": \"weak|moderate|strong\",\n"
        "  \"Role related tech\": []\n"
        "}\n\n"
        "Rules:\n"
        f"{role_rule}\n"
        "- technical_knowledge is based on skill exposure in the resume related to selected_role.\n"
        "- role_related_tech should be a list of concise tech/skills found in the resume.\n"
        "- If role_related_tech is missing, use [\"Not specified\"].\n\n"
        f"RESUME TEXT:\n{trimmed}"
    )


def _normalize_selection(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise RuntimeError("Role selection JSON must be an object.")

    selected_role = str(parsed.get("Selected role", "")).strip()
    if selected_role not in ALLOWED_ROLES:
        raise RuntimeError("Role selection JSON has an invalid selected_role.")

    technical_knowledge = str(parsed.get("Technical Knowledge", "")).strip().lower()
    if technical_knowledge not in ALLOWED_KNOWLEDGE:
        raise RuntimeError("Role selection JSON has an invalid technical_knowledge.")

    role_related_tech = parsed.get("Role related tech")
    if isinstance(role_related_tech, list) and role_related_tech:
        tech_list = [str(item).strip() for item in role_related_tech if str(item).strip()]
    else:
        tech_list = ["Not specified"]

    return {
        "Selected role": selected_role,
        "Technical Knowledge": technical_knowledge,
        "Role related tech": tech_list,
    }


def select_role_and_update_metadata(
    resume_text: str,
    metadata: dict,
    selected_role: str | None = None,
    dest_prefix: str | None = None,
    filename: str | None = None,
    bucket: str | None = None,
    model: str | None = None,
) -> tuple[str, dict]:
    if not resume_text:
        raise ValueError("Resume text is required for role selection.")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict.")
    if selected_role is not None and selected_role not in ALLOWED_ROLES:
        raise ValueError("selected_role must be a supported role.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": _build_prompt(resume_text, selected_role)}],
        temperature=0.2,
    )

    selection_text = response.choices[0].message.content.strip()
    if not selection_text:
        raise RuntimeError("Groq returned an empty response.")

    json_start = selection_text.find("{")
    json_end = selection_text.rfind("}")
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        raise RuntimeError("Groq returned invalid JSON for role selection.")

    try:
        selection_json = json.loads(selection_text[json_start : json_end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq returned invalid JSON for role selection.") from exc

    selection = _normalize_selection(selection_json)
    if selected_role is not None:
        selection["Selected role"] = selected_role
    metadata.update(selection)

    metadata_filename = filename or DEFAULT_METADATA_FILENAME
    metadata_prefix = dest_prefix or os.getenv("PARSED_RESUME_PREFIX", DEFAULT_METADATA_PREFIX)
    buffer = io.BytesIO(json.dumps(metadata, ensure_ascii=True, indent=2).encode("utf-8"))

    object_key = upload_resume_file(
        buffer,
        metadata_filename,
        dest_prefix=metadata_prefix,
        bucket=bucket,
    )

    return object_key, metadata
