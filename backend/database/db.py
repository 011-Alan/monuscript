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


def create_conversation(title="New Conversation", user_id: str = "legacy"):
    """Create a new conversation owned by `user_id`."""
    conv_id = str(uuid.uuid4())
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO conversations (id, title, user_id) VALUES (?, ?, ?)",
            (conv_id, title, user_id)
        )
    conn.close()
    return get_conversation(conv_id, user_id)


def get_conversations(user_id: str):
    """Return all conversations owned by `user_id`, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str, user_id: str):
    """Fetch a single conversation, enforcing owner check."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, user_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def rename_conversation(conv_id: str, new_title: str, user_id: str):
    """Rename a conversation; only updates rows owned by `user_id`."""
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
            (new_title, conv_id, user_id)
        )
    conn.close()
    return get_conversation(conv_id, user_id)


def delete_conversation(conv_id: str, user_id: str = "legacy"):
    """Delete a conversation only if it belongs to `user_id`."""
    conn = get_connection()
    with conn:
        conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id)
        )
    conn.close()
    return True


def add_message(conv_id: str, role: str, content: str, image_url=None, user_id: str = "legacy"):
    """Insert a message row and bump the parent conversation's updated_at timestamp."""
    msg_id = str(uuid.uuid4())
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, user_id, role, content, image_url) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, user_id, role, content, image_url)
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


def get_conversation_history(conv_id: str):
    """Return all messages (with attached recognition results) for a conversation."""
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

    results_by_msg: dict = {}
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
