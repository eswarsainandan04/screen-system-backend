import os
from typing import Iterator

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import psycopg2
except ImportError as exc:
    raise RuntimeError("psycopg2 is required. Install with pip install psycopg2-binary.") from exc


ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

if load_dotenv:
    load_dotenv(ENV_PATH)


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    return default


def get_db_connection():
    host = _get_env("POSTGRES_HOST")
    port = _get_env("POSTGRES_PORT")
    database = _get_env("POSTGRES_DB")
    user = _get_env("POSTGRES_USER")
    password = _get_env("POSTGRES_PASSWORD")
    sslmode = _get_env("POSTGRES_SSLMODE", "require")


    host = _get_env("POSTGRES_HOST")
    port = _get_env("POSTGRES_PORT")
    database = _get_env("POSTGRES_DB")
    user = _get_env("POSTGRES_USER")
    password = _get_env("POSTGRES_PASSWORD")

    print("HOST:", host)
    print("PORT:", port)
    print("DATABASE:", database)
    print("USER:", user)
    print("PASSWORD EXISTS:", bool(password))

    if not all([host, port, database, user, password]):
        raise RuntimeError("Postgres environment variables are not fully set.")

    return psycopg2.connect(
        host=host,
        port=int(port),
        dbname=database,
        user=user,
        password=password,
        sslmode=sslmode,
    )


def ensure_users_table() -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS users (
        userid UUID PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(create_sql)
            conn.commit()
