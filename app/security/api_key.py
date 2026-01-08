import hashlib
from datetime import datetime
from fastapi import HTTPException
from app.db.db import get_connection

def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()

def validate_api_key(raw_key: str) -> int:
    key_hash = hash_key(raw_key)

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT
            ak.id AS api_key_id,
            ak.status AS key_status,
            ak.expires_at,
            ac.status AS client_status
        FROM api_keys ak
        JOIN api_clients ac ON ac.id = ak.client_id
        WHERE ak.api_key_hash = %s
    """, (key_hash,))

    row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if row["key_status"] != "ACTIVE":
        raise HTTPException(status_code=403, detail="API key revoked")

    if row["client_status"] != "ACTIVE":
        raise HTTPException(status_code=403, detail="Client suspended")

    if row["expires_at"] and row["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=403, detail="API key expired")

    # Update last used
    cur.execute(
        "UPDATE api_keys SET last_used_at = NOW() WHERE id = %s",
        (row["api_key_id"],)
    )
    conn.commit()

    cur.close()
    conn.close()

    return row["api_key_id"]
