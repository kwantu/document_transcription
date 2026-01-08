from app.db.db import get_connection

def log_request(api_key_id, endpoint, status_code):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO api_usage_logs (api_key_id, endpoint, response_code)
        VALUES (%s, %s, %s)
    """, (api_key_id, endpoint, status_code))

    conn.commit()
    cur.close()
    conn.close()
