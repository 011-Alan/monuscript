"""
MonuScript - SQLite Persistence Layer
Manages local SQLite database operations for conversations, messages, and recognition results.
Thread-safe with connection pooling per request.
"""

import sqlite3
import json
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "monuscript.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_connection()
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

def create_conversation(title="New Conversation"):
    conv_id = str(uuid.uuid4())
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, title)
        )
    conn.close()
    return get_conversation(conv_id)

def get_conversations():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_conversation(conv_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
        (conv_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def rename_conversation(conv_id, new_title):
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_title, conv_id)
        )
    conn.close()
    return get_conversation(conv_id)

def delete_conversation(conv_id):
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.close()
    return True

def add_message(conv_id, role, content, image_url=None):
    msg_id = str(uuid.uuid4())
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, image_url) VALUES (?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, content, image_url)
        )
        conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conv_id,)
        )
    conn.close()
    return msg_id

def add_recognition_result(message_id, monument_id, score, score_type, accepted=1, metadata_snapshot=None):
    res_id = str(uuid.uuid4())
    meta_json = json.dumps(metadata_snapshot) if metadata_snapshot is not None else None
    conn = get_connection()
    with conn:
        conn.execute(
            """INSERT INTO recognition_results 
               (id, message_id, monument_id, recognition_score, score_type, accepted, metadata_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (res_id, message_id, monument_id, score, score_type, int(accepted), meta_json)
        )
    conn.close()
    return res_id

def get_conversation_history(conv_id):
    conn = get_connection()
    messages_rows = conn.execute(
        "SELECT id, conversation_id, role, content, image_url, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conv_id,)
    ).fetchall()

    results_rows = conn.execute(
        """SELECT r.id, r.message_id, r.monument_id, r.recognition_score, r.score_type, r.accepted, r.metadata_snapshot, r.created_at
           FROM recognition_results r
           JOIN messages m ON r.message_id = m.id
           WHERE m.conversation_id = ?
           ORDER BY r.created_at ASC""",
        (conv_id,)
    ).fetchall()
    conn.close()

    results_by_msg = {}
    for r in results_rows:
        mid = r["message_id"]
        meta = json.loads(r["metadata_snapshot"]) if r["metadata_snapshot"] else {}
        item = {
            "id": r["id"],
            "monument_id": r["monument_id"],
            "score": r["recognition_score"],
            "score_type": r["score_type"],
            "accepted": bool(r["accepted"]),
            "metadata": meta
        }
        if mid not in results_by_msg:
            results_by_msg[mid] = []
        results_by_msg[mid].append(item)

    history = []
    for m in messages_rows:
        mid = m["id"]
        history.append({
            "id": mid,
            "conversation_id": m["conversation_id"],
            "role": m["role"],
            "content": m["content"],
            "image_url": m["image_url"],
            "created_at": m["created_at"],
            "results": results_by_msg.get(mid, [])
        })

    return history
