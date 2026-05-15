import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from db import ensure_users_table, get_db_connection
from ingestion.base import router as resume_router
from llm.interview import router as interview_router
from login.login import router as login_router
from signup.signup import router as signup_router
from llm.insights import router as insights_router
from llm.sessions import router as sessions_router


app = FastAPI(title="PGAGI Backend")

allowed_origins = [
	os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=allowed_origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)


@app.on_event("startup")
def startup() -> None:
	ensure_users_table()


app.include_router(login_router)
app.include_router(signup_router)
app.include_router(resume_router)
app.include_router(interview_router)
app.include_router(insights_router)
app.include_router(sessions_router)


@app.get("/")
def health() -> dict:
	return {"status": "ok"}


@app.get("/users/profile")
def get_profile(email: str = Query(...)) -> dict:
	query = "SELECT userid, name, email FROM users WHERE email = %s"

	with get_db_connection() as conn:
		with conn.cursor() as cursor:
			cursor.execute(query, (email,))
			row = cursor.fetchone()

	if not row:
		raise HTTPException(status_code=404, detail="User not found.")

	user_id, name, email_value = row
	return {"userid": str(user_id), "name": name, "email": email_value}
