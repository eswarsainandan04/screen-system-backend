import json
import os
import sys
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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

from llm.conversation_chat import (
	load_conversation,
	read_context,
	read_metadata,
	save_conversation,
)

ENV_PATH = os.path.join(REPO_ROOT, ".env")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_CONTEXT_ITEMS = 6
MAX_CONTEXT_CHARS = 380
MAX_HISTORY_MESSAGES = 10

# Interview time limits (in seconds)
INTERVIEW_MAX_SECONDS = 20 * 60          # hard 20-min cap
INTERVIEW_WRAP_UP_SECONDS = 17 * 60     # start closing at 17 min
INTERVIEW_WARNING_SECONDS = 15 * 60     # warn the candidate at 15 min


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


def _now_iso() -> str:
	"""Return current UTC time as ISO-8601 string with seconds precision."""
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session_elapsed_seconds(conversation: list[dict]) -> int:
	"""Return seconds elapsed since the first message in the conversation."""
	if not conversation:
		return 0
	first_ts = conversation[0].get("time")
	if not first_ts:
		return 0
	try:
		start = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
		now = datetime.now(timezone.utc)
		return int((now - start).total_seconds())
	except (ValueError, TypeError):
		return 0


def _clean_text(value: str, limit: int) -> str:
	text = " ".join(value.split())
	if len(text) <= limit:
		return text
	return text[: limit - 3].rstrip() + "..."


def _format_context(context: dict | list) -> str:
	items: list[tuple[str, str]] = []
	if isinstance(context, dict):
		for key, value in context.items():
			if isinstance(value, list) and value:
				sample = value[0]
			else:
				sample = value
			sample_text = str(sample).strip()
			if not sample_text:
				continue
			items.append((str(key), sample_text))
	elif isinstance(context, list):
		for idx, value in enumerate(context, start=1):
			sample_text = str(value).strip()
			if not sample_text:
				continue
			items.append((f"Context {idx}", sample_text))

	formatted: list[str] = []
	for key, sample in items[:MAX_CONTEXT_ITEMS]:
		formatted.append(f"- {key}: {_clean_text(sample, MAX_CONTEXT_CHARS)}")
	return "\n".join(formatted) if formatted else "- No context available."


def _format_conversation_for_prompt(history: list[dict]) -> str:
	"""Format the recent conversation history for the LLM prompt, including timestamps."""
	trimmed = history[-MAX_HISTORY_MESSAGES:] if history else []
	lines: list[str] = []
	for item in trimmed:
		sender = item.get("sender", "").strip().lower()
		message = item.get("message", "").strip()
		ts = item.get("time", "")
		if not message:
			continue
		label = "AI" if sender == "ai" else "User"
		time_tag = f" [{ts}]" if ts else ""
		lines.append(f"{label}{time_tag}: {message}")
	return "\n".join(lines) if lines else "No prior conversation."


def _knowledge_guidance(level: str) -> str:
	normalized = (level or "").strip().lower()
	if normalized == "weak":
		return "Use fundamentals, definitions, and simple applied examples."
	if normalized == "moderate":
		return "Use intermediate concepts with practical scenarios."
	if normalized == "strong":
		return "Use advanced questions, tradeoffs, and deeper reasoning."
	return "Use a balanced difficulty based on the resume."


def _analyze_conversation_patterns(history: list[dict]) -> str:
	"""
	Analyze past conversation to detect:
	- Topics already covered
	- Candidate's depth of answers (short/detailed)
	- Repeated weak areas (follow-up needed)
	- Answer confidence signals
	"""
	if not history:
		return "No prior conversation to analyze."

	ai_messages = [m["message"] for m in history if m.get("sender") == "ai" and m.get("message")]
	user_messages = [m["message"] for m in history if m.get("sender") == "user" and m.get("message")]

	if not user_messages:
		return "Candidate has not answered yet."

	# Rough quality signals
	avg_answer_len = sum(len(a) for a in user_messages) / max(len(user_messages), 1)
	short_answers = sum(1 for a in user_messages if len(a.split()) < 15)
	total_user = len(user_messages)

	patterns: list[str] = []
	patterns.append(f"- Questions asked so far: {len(ai_messages)}")
	patterns.append(f"- Candidate answers so far: {total_user}")
	patterns.append(f"- Average answer length: {int(avg_answer_len)} chars")
	if short_answers > total_user * 0.5:
		patterns.append("- Candidate tends to give SHORT answers — probe deeper on weak areas.")
	else:
		patterns.append("- Candidate gives reasonably detailed answers — you can advance to harder topics.")

	# Last user answer for adaptive follow-up signal
	last_answer = user_messages[-1] if user_messages else ""
	if len(last_answer.split()) < 10:
		patterns.append("- Last answer was very brief — consider a targeted follow-up on the same topic.")
	elif any(phrase in last_answer.lower() for phrase in ["i don't know", "not sure", "i'm unsure", "no idea", "i cannot"]):
		patterns.append("- Candidate expressed uncertainty in last answer — ask a simpler clarifying question.")

	return "\n".join(patterns)


def _time_management_instruction(elapsed: int) -> str:
	"""Return a time-aware instruction for the LLM based on elapsed seconds."""
	remaining = max(0, INTERVIEW_MAX_SECONDS - elapsed)
	remaining_min = remaining // 60

	if elapsed >= INTERVIEW_MAX_SECONDS:
		return (
			"TIME IS UP. The 20-minute interview has ended. "
			"Generate a polite closing statement thanking the candidate and telling them the interview is now complete. "
			"Do NOT ask any more questions."
		)
	if elapsed >= INTERVIEW_WRAP_UP_SECONDS:
		return (
			f"CLOSING PHASE: Only {remaining_min} minute(s) left. "
			"Ask one final question or wrap up the interview gracefully. "
			"After this, generate a closing statement. Keep it short."
		)
	if elapsed >= INTERVIEW_WARNING_SECONDS:
		return (
			f"FINAL STRETCH: {remaining_min} minute(s) remaining. "
			"Ask the last 1-2 key technical questions. Prioritize the most critical topics not yet covered."
		)

	minutes_used = elapsed // 60
	return (
		f"Time used: {minutes_used} min / 20 min. "
		"Maintain steady pacing — ensure breadth across key topics before time runs out."
	)


def _build_prompt(
	metadata: dict,
	context: dict | list,
	history: list[dict],
	is_intro: bool,
	elapsed_seconds: int,
) -> str:
	role = metadata.get("Selected role", "Interviewee")
	knowledge = metadata.get("Technical Knowledge", "")
	role_related = metadata.get("Role related tech") or []
	role_related_text = ", ".join([str(item) for item in role_related if str(item).strip()])
	context_text = _format_context(context)
	conversation_text = _format_conversation_for_prompt(history)
	pattern_analysis = _analyze_conversation_patterns(history)
	time_instruction = _time_management_instruction(elapsed_seconds)

	intro_rule = (
		"Start with a brief, warm self-introduction question focused on the role.\n"
		if is_intro
		else "Go directly technical. No more warm-up questions.\n"
	)

	return (
		"You are an expert technical interviewer conducting a structured 20-minute interview.\n"
		"Generate exactly ONE response — either a question or a closing statement.\n"
		"Responses must be clean, concise, and professional.\n\n"
		f"Role: {role}\n"
		f"Knowledge level: {knowledge}\n"
		f"Role related tech: {role_related_text or 'Not specified'}\n\n"
		"=== TIME MANAGEMENT ===\n"
		f"{time_instruction}\n\n"
		"=== CONVERSATION PATTERN ANALYSIS ===\n"
		f"{pattern_analysis}\n\n"
		"=== RETRIEVED CONTEXT SNIPPETS ===\n"
		f"{context_text}\n\n"
		"=== CONVERSATION HISTORY (with timestamps) ===\n"
		f"{conversation_text}\n\n"
		"=== RULES ===\n"
		"- Generate ONE question (or closing statement if time is up).\n"
		"- Keep questions concise (1 sentence when possible).\n"
		"- Adapt based on the Conversation Pattern Analysis above.\n"
		"- Avoid repeating questions already asked.\n"
		"- Use role, knowledge level, role-related tech, and context only.\n"
		"- If last user answer shows gaps, ask a focused follow-up on that topic.\n"
		"- Prefer depth over breadth; no multi-part lists.\n"
		f"- {intro_rule}"
		f"- Difficulty guidance: {_knowledge_guidance(knowledge)}\n\n"
		"Return ONLY the question text (or closing statement). No preamble."
	)


def _get_groq_client() -> Groq:
	groq_api_key = os.getenv("GROQ_API_KEY")
	if not groq_api_key:
		raise RuntimeError("GROQ_API_KEY is not set.")
	return Groq(api_key=groq_api_key)


def generate_next_question(
	user_id: str,
	session_id: str,
	user_message: str | None = None,
	bucket: str | None = None,
	model: str | None = None,
) -> tuple[str, list[dict], str, bool]:
	"""
	Returns:
	  (question_text, conversation, conversation_key, session_ended)

	session_ended is True when the 20-min cap has been hit or LLM issued a closing statement.
	"""
	metadata = read_metadata(user_id, session_id, bucket=bucket)
	context = read_context(user_id, session_id, bucket=bucket)
	conversation = load_conversation(user_id, session_id, bucket=bucket)

	now_ts = _now_iso()
	cleaned_message = user_message.strip() if user_message else ""

	if cleaned_message:
		conversation.append({"sender": "user", "message": cleaned_message, "time": now_ts})
	elif conversation:
		# No new message — return last AI question if it exists
		last = conversation[-1]
		if last.get("sender", "").lower() == "ai" and last.get("message", "").strip():
			conversation_key = save_conversation(user_id, session_id, conversation, bucket=bucket)
			elapsed = _session_elapsed_seconds(conversation)
			session_ended = elapsed >= INTERVIEW_MAX_SECONDS
			return last["message"], conversation, conversation_key, session_ended

	elapsed = _session_elapsed_seconds(conversation)
	session_ended = elapsed >= INTERVIEW_MAX_SECONDS

	is_intro = not any(m.get("sender") == "ai" for m in conversation)
	prompt = _build_prompt(metadata, context, conversation, is_intro=is_intro, elapsed_seconds=elapsed)

	client = _get_groq_client()
	response = client.chat.completions.create(
		model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
		messages=[{"role": "user", "content": prompt}],
		temperature=0.4,
	)

	question = response.choices[0].message.content.strip()
	if not question:
		raise RuntimeError("Groq returned an empty question.")

	ai_ts = _now_iso()
	conversation.append({"sender": "ai", "message": question, "time": ai_ts})
	conversation_key = save_conversation(user_id, session_id, conversation, bucket=bucket)

	# Mark session ended if time is up after saving this message
	elapsed_after = _session_elapsed_seconds(conversation)
	if elapsed_after >= INTERVIEW_MAX_SECONDS:
		session_ended = True

	return question, conversation, conversation_key, session_ended


router = APIRouter(prefix="/interview", tags=["interview"])


class InterviewRequest(BaseModel):
	userid: str = Field(..., min_length=1)
	sessionid: str = Field(..., min_length=1)
	message: str | None = None


@router.post("/next")
def next_question(payload: InterviewRequest) -> dict:
	try:
		question, conversation, conversation_key, session_ended = generate_next_question(
			user_id=payload.userid,
			session_id=payload.sessionid,
			user_message=payload.message,
		)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	return {
		"question": question,
		"conversation": conversation,
		"conversation_key": conversation_key,
		"session_ended": session_ended,
	}


@router.post("/end")
def end_interview(payload: InterviewRequest) -> dict:
	"""
	Explicitly end the interview session.
	Triggers insight generation and returns a redirect signal.
	"""
	try:
		from llm.insights import generate_insights
		result = generate_insights(
			user_id=payload.userid,
			session_id=payload.sessionid,
		)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	return {
		"status": "ended",
		"insights_key": result.get("insights_key"),
		"score": result.get("final_score"),
	}