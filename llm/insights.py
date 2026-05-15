"""
llm/insights.py

Generates post-interview insights:
  - Per-message feedback injected into conversation.json
  - Final score (0-100) with breakdown
  - Per-question Q&A evaluation by Groq
  - Saves enriched conversation.json back to S3/Supabase

Called automatically when session ends (time-cap or user ends call).
"""

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

from llm.conversation_chat import (
	load_conversation,
	read_context,
	read_metadata,
	save_conversation,
	_upload_json,
	build_object_key,
)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ENV_PATH = os.path.join(REPO_ROOT, ".env")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_ANALYSIS_FILENAME = "analysis.json"


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


def _get_groq_client() -> Groq:
	groq_api_key = os.getenv("GROQ_API_KEY")
	if not groq_api_key:
		raise RuntimeError("GROQ_API_KEY is not set.")
	return Groq(api_key=groq_api_key)


def _extract_qa_pairs(conversation: list[dict]) -> list[dict]:
	"""
	Walk the conversation and pair each AI question with the following user answer.
	Returns list of dicts: {index, question, answer, question_time, answer_time}
	"""
	pairs = []
	i = 0
	pair_index = 0
	while i < len(conversation):
		msg = conversation[i]
		if msg.get("sender", "").lower() == "ai":
			question = msg.get("message", "").strip()
			q_time = msg.get("time", "")
			# Look for next user response
			answer = ""
			a_time = ""
			if i + 1 < len(conversation) and conversation[i + 1].get("sender", "").lower() == "user":
				answer = conversation[i + 1].get("message", "").strip()
				a_time = conversation[i + 1].get("time", "")
				i += 2
			else:
				i += 1
			if question:
				pair_index += 1
				pairs.append({
					"index": pair_index,
					"question": question,
					"answer": answer or "(no answer provided)",
					"question_time": q_time,
					"answer_time": a_time,
				})
		else:
			i += 1
	return pairs


def _build_feedback_prompt(
	role: str,
	knowledge_level: str,
	context_summary: str,
	pairs: list[dict],
) -> str:
	qa_block = ""
	for p in pairs:
		qa_block += f"\nQ{p['index']}: {p['question']}\nA{p['index']}: {p['answer']}\n"

	return (
		"You are an expert technical interviewer evaluating a completed interview.\n\n"
		f"Role: {role}\n"
		f"Expected Knowledge Level: {knowledge_level}\n\n"
		"=== KNOWLEDGE CONTEXT (from RAG) ===\n"
		f"{context_summary}\n\n"
		"=== INTERVIEW Q&A ===\n"
		f"{qa_block}\n"
		"=== TASK ===\n"
		"Evaluate EACH question-answer pair and produce a JSON object.\n\n"
		"Schema:\n"
		"{\n"
		"  \"evaluations\": [\n"
		"    {\n"
		"      \"index\": 1,\n"
		"      \"question\": \"...\",\n"
		"      \"answer\": \"...\",\n"
		"      \"score\": 0-10,\n"
		"      \"correct\": true|false,\n"
		"      \"feedback\": \"concise 1-2 sentence feedback on the answer\",\n"
		"      \"ideal_answer_hint\": \"brief hint at what a strong answer would include\"\n"
		"    }\n"
		"  ],\n"
		"  \"final_score\": 0-100,\n"
		"  \"total_questions\": N,\n"
		"  \"correct_answers\": N,\n"
		"  \"overall_feedback\": \"3-4 sentence overall performance summary\",\n"
		"  \"strengths\": [\"...\"],\n"
		"  \"improvement_areas\": [\"...\"]\n"
		"}\n\n"
		"Scoring rules:\n"
		"- Each question is scored 0-10 based on correctness, depth, and clarity.\n"
		"- final_score = (sum of all scores / (total_questions * 10)) * 100, rounded to integer.\n"
		"- correct = true if score >= 6.\n"
		"- Be honest but constructive. Avoid generic praise.\n"
		"- Return ONLY valid JSON. No markdown, no preamble."
	)


def _parse_evaluation(raw: str) -> dict:
	text = raw.strip()
	# Strip markdown fences if present
	if text.startswith("```"):
		lines = text.split("\n")
		text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

	try:
		return json.loads(text)
	except json.JSONDecodeError:
		start = text.find("{")
		end = text.rfind("}")
		if start != -1 and end != -1 and end > start:
			try:
				return json.loads(text[start: end + 1])
			except json.JSONDecodeError:
				pass
	raise RuntimeError("Could not parse evaluation JSON from Groq.")


def _summarize_context(context: dict | list, max_chars: int = 800) -> str:
	if isinstance(context, dict):
		parts = []
		for k, v in context.items():
			val = v[0] if isinstance(v, list) and v else v
			parts.append(f"{k}: {str(val)[:200]}")
		return "\n".join(parts)[:max_chars]
	if isinstance(context, list):
		return "\n".join(str(item)[:200] for item in context)[:max_chars]
	return str(context)[:max_chars]


def _inject_feedback_into_conversation(
	conversation: list[dict],
	evaluations: list[dict],
) -> list[dict]:
	"""
	Inject per-message feedback into the conversation.
	For each AI message (question), attach feedback from the evaluation.
	For each user message (answer), attach score and correct flag.
	"""
	# Build lookup by question text (normalized)
	eval_by_question: dict[str, dict] = {}
	for ev in evaluations:
		key = (ev.get("question") or "").strip().lower()[:120]
		eval_by_question[key] = ev

	enriched = []
	i = 0
	while i < len(conversation):
		msg = dict(conversation[i])
		if msg.get("sender", "").lower() == "ai":
			q_key = msg.get("message", "").strip().lower()[:120]
			ev = eval_by_question.get(q_key)
			if ev:
				msg["feedback"] = ev.get("feedback", "")
				msg["ideal_answer_hint"] = ev.get("ideal_answer_hint", "")
			enriched.append(msg)

			# Attach score to the paired user message right after
			if i + 1 < len(conversation) and conversation[i + 1].get("sender", "").lower() == "user":
				user_msg = dict(conversation[i + 1])
				if ev:
					user_msg["score"] = ev.get("score", 0)
					user_msg["correct"] = ev.get("correct", False)
				enriched.append(user_msg)
				i += 2
				continue
		else:
			enriched.append(msg)
		i += 1

	return enriched


def generate_insights(
	user_id: str,
	session_id: str,
	bucket: str | None = None,
	model: str | None = None,
) -> dict:
	"""
	Main entry point.
	1. Load conversation, metadata, context
	2. Extract Q&A pairs
	3. Ask Groq to evaluate each pair and produce a score
	4. Inject per-message feedback into conversation
	5. Append final_insights block to conversation
	6. Save enriched conversation back to S3

	Returns the insights dict.
	"""
	metadata = read_metadata(user_id, session_id, bucket=bucket)
	context = read_context(user_id, session_id, bucket=bucket)
	conversation = load_conversation(user_id, session_id, bucket=bucket)

	if not conversation:
		raise RuntimeError("No conversation found — cannot generate insights.")

	role = metadata.get("Selected role", "Unknown Role")
	knowledge = metadata.get("Technical Knowledge", "moderate")
	context_summary = _summarize_context(context)

	pairs = _extract_qa_pairs(conversation)
	if not pairs:
		raise RuntimeError("No question-answer pairs found in conversation.")

	prompt = _build_feedback_prompt(role, knowledge, context_summary, pairs)
	client = _get_groq_client()
	response = client.chat.completions.create(
		model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
		messages=[{"role": "user", "content": prompt}],
		temperature=0.2,
		max_tokens=4096,
	)

	raw_text = response.choices[0].message.content.strip()
	if not raw_text:
		raise RuntimeError("Groq returned an empty evaluation.")

	evaluation = _parse_evaluation(raw_text)
	evaluations_list: list[dict] = evaluation.get("evaluations", [])

	# Inject feedback into every message
	enriched_conversation = _inject_feedback_into_conversation(conversation, evaluations_list)

	# Build final insights block — appended as a special message at end of conversation
	final_insights = {
		"sender": "system",
		"type": "insights",
		"final_score": evaluation.get("final_score", 0),
		"total_questions": evaluation.get("total_questions", len(pairs)),
		"correct_answers": evaluation.get("correct_answers", 0),
		"overall_feedback": evaluation.get("overall_feedback", ""),
		"strengths": evaluation.get("strengths", []),
		"improvement_areas": evaluation.get("improvement_areas", []),
		"evaluations": evaluations_list,
		"time": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
	}
	enriched_conversation.append(final_insights)

	# ── Save analysis.json (lightweight — no conversation messages) ─────────
	analysis_payload = {
		"final_score": final_insights["final_score"],
		"total_questions": final_insights["total_questions"],
		"correct_answers": final_insights["correct_answers"],
		"overall_feedback": final_insights["overall_feedback"],
		"strengths": final_insights["strengths"],
		"improvement_areas": final_insights["improvement_areas"],
		"evaluations": evaluations_list,
		"time": final_insights["time"],
	}
	analysis_key = build_object_key(user_id, session_id, DEFAULT_ANALYSIS_FILENAME)
	_upload_json(analysis_payload, analysis_key, bucket=bucket)

	# Save enriched conversation (with per-message feedback injected)
	insights_key = save_conversation(user_id, session_id, enriched_conversation, bucket=bucket)

	return {
		"insights_key": insights_key,
		"analysis_key": analysis_key,
		"final_score": final_insights["final_score"],
		"total_questions": final_insights["total_questions"],
		"correct_answers": final_insights["correct_answers"],
		"overall_feedback": final_insights["overall_feedback"],
		"strengths": final_insights["strengths"],
		"improvement_areas": final_insights["improvement_areas"],
		"evaluations": evaluations_list,
	}


# ── FastAPI router ──────────────────────────────────────────────────────────

router = APIRouter(prefix="/insights", tags=["insights"])


class InsightsRequest(BaseModel):
	userid: str = Field(..., min_length=1)
	sessionid: str = Field(..., min_length=1)


@router.post("/generate")
def generate_insights_endpoint(payload: InsightsRequest) -> dict:
	"""
	POST /insights/generate
	Trigger insight generation for a completed interview session.
	"""
	try:
		result = generate_insights(user_id=payload.userid, session_id=payload.sessionid)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc
	return result


@router.get("/analysis")
def get_analysis(userid: str, sessionid: str) -> dict:
	"""
	GET /insights/analysis?userid=&sessionid=
	Read directly from analysis.json — fast, no conversation scan needed.
	"""
	try:
		key = build_object_key(userid, sessionid, DEFAULT_ANALYSIS_FILENAME)
		from llm.conversation_chat import _download_json
		data = _download_json(key, allow_missing=True)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	if not isinstance(data, dict):
		raise HTTPException(status_code=404, detail="Analysis not yet generated for this session.")
	return data


@router.get("/result")
def get_insights(userid: str, sessionid: str) -> dict:
	"""
	GET /insights/result?userid=&sessionid=
	Fetch the saved insights from the enriched conversation.json.
	(Legacy — prefer /insights/analysis for speed.)
	"""
	try:
		conversation = load_conversation(userid, sessionid)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	# Find the system insights block
	for msg in reversed(conversation):
		if msg.get("sender") == "system" and msg.get("type") == "insights":
			return msg

	raise HTTPException(status_code=404, detail="Insights not yet generated for this session.")