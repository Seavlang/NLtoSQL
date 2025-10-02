# app.py
# pip install -U fastapi uvicorn langchain langchain-community langchain-ollama psycopg2-binary

import os, re, time
from typing import List, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from fastapi import UploadFile, File, Form
from sqlalchemy import text
import pandas as pd
import csv, io, re


from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from fastapi import APIRouter

router = APIRouter()
# ---------- Config (envs override defaults) ----------
DB_URI = os.getenv(
    "DB_URI",
    "postgresql+psycopg2://postgres:postgres123@203.255.78.58:9002/kyunginara"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "50"))
ALLOWED_SCHEMAS = {"public"}  # keep it tight



# ---------- Singletons ----------
db = SQLDatabase.from_uri(DB_URI)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=4096)
sql_query_chain = create_sql_query_chain(llm, db)

# ---------- Models ----------
class NLQuery(BaseModel):
    question: str = Field(..., examples=[
        'Show 10 rows with use_intt_id, trsc_dt, sale_cnt, sale_amt from public."1_SELR_DALY_SUMR"'
    ])
    limit: Optional[int] = Field(DEFAULT_LIMIT, ge=1, le=10_000)
    sql_only: bool = False

class QueryResult(BaseModel):
    sql: str
    columns: List[str] = []
    rows: List[List[Any]] = []
    row_count: int = 0
    elapsed_ms: float

# ---------- Helpers (robust + safe) ----------
def strip_fences_and_extract_sql(s: str) -> str:
    s = s.strip()
    # Prefer fenced block
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1).strip()
    # Fallback to first SELECT/WITH block
    m = re.search(r"(?is)\b(select|with)\b.*", s)
    if m:
        s = m.group(0).strip()
    # Keep only first statement if semicolons appear
    if ";" in s:
        s = s.split(";")[0].strip() + ";"
    return s

def ensure_select_only(sql: str) -> None:
    if not re.match(r"(?is)^\s*(select|with)\b", sql):
        raise HTTPException(status_code=400, detail="Only SELECT/CTE queries are allowed.")
    # block multiple statements
    if re.search(r";\s*\S", sql.strip()):
        raise HTTPException(status_code=400, detail="Multiple statements are not allowed.")

def enforce_schema(sql: str, allowed: set[str]) -> None:
    # Check explicit schema.table references only (quotes optional)
    for schema in re.findall(r'(?<!:)"?([A-Za-z_]\w*)"?\s*\.', sql):
        if schema.lower() not in allowed:
            raise HTTPException(status_code=400, detail=f'Schema "{schema}" is not allowed.')

def add_limit_if_missing(sql: str, limit: int) -> str:
    return sql if re.search(r"(?is)\blimit\s+\d+\b", sql) else f"{sql.rstrip().rstrip(';')}\nLIMIT {limit};"

def generate_sql(nl_question: str, limit: int) -> str:
    guidance = (
            nl_question.strip().rstrip(".!?")
            + "\n\nReturn ONLY one SQL SELECT statement. No explanation. "
              "Use schema public only. Double-quote table names that start with digits."
    )
    raw = sql_query_chain.invoke({"question": guidance})
    sql = strip_fences_and_extract_sql(raw)
    ensure_select_only(sql)
    enforce_schema(sql, ALLOWED_SCHEMAS)
    return add_limit_if_missing(sql, limit)

# ---------- Endpoints ----------
@router.get("/model" , summary="Model info")
def health() -> dict:
    try:
        db.run("SELECT 1;")
        return {"status": "ok", "model": OLLAMA_MODEL}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tables")
def list_tables() -> dict:
    q = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema NOT IN ('pg_catalog','information_schema')
    ORDER BY 1,2;
    """
    rows = db.run(q)
    return {"tables": rows}

@router.get("/schema/{table_name}")
def table_schema(table_name: str) -> dict:
    q = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=:t
    ORDER BY ordinal_position;
    """
    with db._engine.begin() as conn:
        res = conn.execute(text(q), {"t": table_name})
        cols = [r._mapping["column_name"] for r in res]
        types = [r._mapping["data_type"] for r in res]
    return {"table": table_name, "columns": cols, "types": types}

@router.post("/query", response_model=QueryResult)
def query(body: NLQuery):
    try:
        sql = generate_sql(body.question, body.limit or DEFAULT_LIMIT)
        if body.sql_only:
            return QueryResult(sql=sql, elapsed_ms=0.0)

        t0 = time.perf_counter()
        with db._engine.begin() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [list(r) for r in result.fetchall()]
        elapsed = (time.perf_counter() - t0) * 1000.0
        return QueryResult(sql=sql, columns=columns, rows=rows, row_count=len(rows), elapsed_ms=elapsed)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/upload/csv")
async def upload_csv(
        file: UploadFile = File(..., description="CSV file"),
        table: str = Form(..., description="Target table name (letters, numbers, _)"),
        schema: str = Form("public"),
        if_exists: str = Form("replace", description="fail | replace | append"),
        delimiter: str | None = Form(None),
        encoding: str = Form("utf-8"),
        infer_types: bool = Form(False, description="If False, load everything as text"),
        preview_rows: int = Form(10, ge=1, le=200),
        preview_only: bool = Form(False, description="If True, don't write to DB—just return a preview"),
        lowercase_cols: bool = Form(True, description="Normalize column names")
):
    # --- validations ---
    if file.content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
        raise HTTPException(400, "Only CSV files are supported")
    if if_exists not in {"fail", "replace", "append"}:
        raise HTTPException(400, "if_exists must be one of: fail | replace | append")
    # Safe SQL identifier (avoid quoting headaches)
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise HTTPException(400, "Invalid table name. Use letters, numbers, underscore; start with letter/_")

    # --- detect delimiter (peek without loading full file into memory) ---
    file.file.seek(0)
    peek = file.file.read(65536).decode(encoding, errors="ignore")
    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(peek)
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","
    file.file.seek(0)

    # --- load CSV to DataFrame ---
    read_kwargs = dict(sep=delimiter)
    if not infer_types:
        read_kwargs["dtype"] = str
    df = pd.read_csv(file.file, **read_kwargs)

    # --- normalize column names (recommended for SQL) ---
    def norm(c: str) -> str:
        c = c.strip()
        c = c.replace(" ", "_")
        c = re.sub(r"[^A-Za-z0-9_]", "_", c)
        return c.lower() if lowercase_cols else c
    df.columns = [norm(c) for c in df.columns]

    # --- preview first rows ---
    preview = df.head(min(len(df), preview_rows)).to_dict(orient="records")

    if preview_only:
        return {
            "message": "Preview only, nothing written",
            "schema": schema,
            "table": table,
            "rows_in_file": int(df.shape[0]),
            "columns": list(df.columns),
            "delimiter": delimiter,
            "preview": preview,
        }

    # --- write to Postgres (via SQLAlchemy engine inside SQLDatabase) ---
    try:
        with db._engine.begin() as conn:
            df.to_sql(
                name=table,
                con=conn,
                schema=schema,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=1000,
            )
            # confirm row count
            rc = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')).scalar()
    except Exception as e:
        raise HTTPException(400, f"Load failed: {e}")

    return {
        "message": "Upload complete",
        "schema": schema,
        "table": table,
        "inserted_rows": int(rc),
        "columns": list(df.columns),
        "delimiter": delimiter,
        "preview": preview,
    }
