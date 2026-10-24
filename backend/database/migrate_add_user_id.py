# Migration script to add a user_id column to existing SQLite tables and back‑fill existing rows
# This script is intended to be run once (e.g. via `python backend/database/migrate_add_user_id.py`).
# It adds a TEXT column `user_id` to the three tables that store user‑specific data and
# populates it with a placeholder value for historic rows.

import sqlite3
from pathlib import Path

# Path to the SQLite DB (same as used by the app)
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "monuscript.db"

def run_migration():
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        # 1. Add user_id to conversations
        cur.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
        # Back‑fill with a placeholder identifier for existing rows
        cur.execute("UPDATE conversations SET user_id = 'legacy' WHERE user_id = ''")
        # 2. Add user_id to messages
        cur.execute("ALTER TABLE messages ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
        # Populate from the parent conversation
        cur.execute("""
            UPDATE messages
            SET user_id = (
                SELECT user_id FROM conversations WHERE conversations.id = messages.conversation_id
            )
            WHERE user_id = ''
        """)
        # 3. Add user_id to recognition_results
        cur.execute("ALTER TABLE recognition_results ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
        # Populate from the parent message
        cur.execute("""
            UPDATE recognition_results
            SET user_id = (
                SELECT user_id FROM messages WHERE messages.id = recognition_results.message_id
            )
            WHERE user_id = ''
        """)
        conn.commit()
        print("Migration completed successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()
