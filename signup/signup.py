import hashlib
import os
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from db import get_db_connection

router = APIRouter(prefix="/signup", tags=["signup"])


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


def _hash_password(plain: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}${derived.hex()}"


@router.post("")
def signup(payload: SignupRequest):
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be 8+ characters.")

    query = "SELECT 1 FROM users WHERE email = %s"
    insert = """
        INSERT INTO users (userid, name, email, password)
        VALUES (%s, %s, %s, %s)
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (payload.email,))
            if cursor.fetchone():
                raise HTTPException(status_code=409, detail="Email already exists.")

            user_id = uuid.uuid4()
            password_hash = _hash_password(payload.password)
            cursor.execute(
                insert,
                (str(user_id), payload.name, payload.email, password_hash),
            )
            conn.commit()

    return {"userid": str(user_id), "name": payload.name, "email": payload.email}
