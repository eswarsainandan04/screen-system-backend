import hashlib
import hmac
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from db import get_db_connection

router = APIRouter(prefix="/login", tags=["login"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def _verify_password(plain: str, stored: str) -> bool:
    if "$" not in stored:
        return False
    salt_hex, hash_hex = stored.split("$", 1)
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(hash_hex)
    derived = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, 100_000)
    return hmac.compare_digest(derived, expected)


@router.post("")
def login(payload: LoginRequest):
    query = "SELECT userid, name, email, password FROM users WHERE email = %s"

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (payload.email,))
            row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    user_id, name, email, stored_password = row
    if not _verify_password(payload.password, stored_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    return {"userid": str(user_id), "name": name, "email": email}
