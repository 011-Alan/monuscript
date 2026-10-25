"""
MonuScript - SQLite Persistence Layer
Manages local SQLite database operations for conversations, messages, and recognition results.
Thread-safe with connection pooling per request.
Guarantees parent conversation existence before message insertion to prevent foreign key violations.
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


def ensure_conversation(conv_id: str, user_id: str = "legacy", title: str = "New Conversation", created_at: str = None, updated_at: str = None):
    """
    Guarantees that a conversation row exists in SQLite before inserting child messages.
    If already present, does not overwrite title or timestamps.
    """
    if not conv_id:
        return
    conn = get_connection()
    with conn:
        row = conn.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if not row:
            if created_at and updated_at:
                conn.execute(
                    "INSERT INTO conversations (id, title, user_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (conv_id, title, user_id, created_at, updated_at)
                )
            else:
                conn.execute(
                    "INSERT INTO conversations (id, title, user_id) VALUES (?, ?, ?)",
                    (conv_id, title, user_id)
                )
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


def add_message(conv_id: str, role: str, content: str, image_url=None, user_id: str = "legacy", msg_id: str = None, created_at: str = None):
    """
    Insert a message row and bump the parent conversation's updated_at timestamp.
    Guarantees the conversation exists first.
    """
    ensure_conversation(conv_id, user_id=user_id)
    if not msg_id:
        msg_id = str(uuid.uuid4())
    conn = get_connection()
    with conn:
        if created_at:
            conn.execute(
                """INSERT OR REPLACE INTO messages
                   (id, conversation_id, user_id, role, content, image_url, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, conv_id, user_id, role, content, image_url, created_at)
            )
        else:
            conn.execute(
                """INSERT OR REPLACE INTO messages
                   (id, conversation_id, user_id, role, content, image_url)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (msg_id, conv_id, user_id, role, content, image_url)
            )
        conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conv_id,)
        )
    conn.close()
    return msg_id


def add_recognition_result(message_id, monument_id, score, score_type, accepted=1, metadata_snapshot=None, res_id=None):
    if not res_id:
        res_id = str(uuid.uuid4())
    meta_json = json.dumps(metadata_snapshot) if metadata_snapshot is not None else None
    conn = get_connection()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO recognition_results
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


def sync_conversations_batch(user_id: str, conversations_list: list, messages_dict: dict = None):
    """
    Upserts client-side conversations and messages into the SQLite database.
    Guarantees no foreign key violations and preserves timestamps.
    """
    conn = get_connection()
    synced_convs = 0
    synced_msgs = 0

    with conn:
        for c in conversations_list or []:
            c_id = c.get("id")
            title = c.get("title", "Monument Exploration")
            created_at = c.get("created_at")
            updated_at = c.get("updated_at")
            if not c_id:
                continue

            row = conn.execute("SELECT id, updated_at FROM conversations WHERE id = ?", (c_id,)).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO conversations (id, title, user_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (c_id, title, user_id, created_at, updated_at)
                )
                synced_convs += 1
            else:
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                    (title, updated_at or row["updated_at"], c_id, user_id)
                )

        if messages_dict:
            for c_id, msgs in messages_dict.items():
                row = conn.execute("SELECT id FROM conversations WHERE id = ?", (c_id,)).fetchone()
                if not row:
                    conn.execute(
                        "INSERT INTO conversations (id, title, user_id) VALUES (?, ?, ?)",
                        (c_id, "Monument Exploration", user_id)
                    )
                    synced_convs += 1

                for m in msgs:
                    m_id = m.get("id")
                    if not m_id:
                        continue
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    img_url = m.get("image_url")
                    m_created = m.get("created_at")

                    conn.execute(
                        """INSERT OR REPLACE INTO messages
                           (id, conversation_id, user_id, role, content, image_url, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))""",
                        (m_id, c_id, user_id, role, content, img_url, m_created)
                    )
                    synced_msgs += 1

                    for res in m.get("results", []):
                        r_id = res.get("id") or str(uuid.uuid4())
                        mon_id = res.get("monument_id", "unknown")
                        sc = res.get("score", 0.0)
                        st = res.get("score_type", "recognition_score")
                        acc = 1 if res.get("accepted") else 0
                        meta = json.dumps(res.get("metadata", {}))
                        conn.execute(
                            """INSERT OR REPLACE INTO recognition_results
                               (id, message_id, monument_id, recognition_score, score_type, accepted, metadata_snapshot)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (r_id, m_id, mon_id, sc, st, acc, meta)
                        )

    conn.close()
    return synced_convs, synced_msgs
