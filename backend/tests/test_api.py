"""
Integration Tests for MonuScript Flask API (/api/v1)
"""

import sys
import io
from pathlib import Path
import pytest
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """GET /api/v1/health returns status ok and loaded model classes."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["supported_classes"] == 6

def test_monuments_metadata_endpoint(client):
    """GET /api/v1/monuments returns list of monuments."""
    resp = client.get("/api/v1/monuments")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert len(data["monuments"]) == 6

TEST_AUTH_HEADERS = {"Authorization": "Bearer anon_test_suite_visitor"}

def test_unauthenticated_request_rejected(client):
    """Protected endpoints reject requests without authorization."""
    resp = client.get("/api/v1/conversations")
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNAUTHORIZED"

def test_missing_image_upload(client):
    """POST /api/v1/recognize with no file returns HTTP 400."""
    resp = client.post("/api/v1/recognize", data={}, content_type="multipart/form-data", headers=TEST_AUTH_HEADERS)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "MISSING_IMAGE"

def test_corrupt_image_upload(client):
    """POST /api/v1/recognize with corrupt data returns HTTP 422 with friendly message."""
    data = {
        "image": (io.BytesIO(b"this is corrupt junk not an image"), "fake.jpg")
    }
    resp = client.post("/api/v1/recognize", data=data, content_type="multipart/form-data", headers=TEST_AUTH_HEADERS)
    assert resp.status_code == 422
    res_json = resp.get_json()
    assert res_json["success"] is False
    assert "couldn't open this image" in res_json["error"]["message"].lower()

def test_conversation_lifecycle(client):
    """Test creating, listing, renaming, and deleting conversations."""
    # 1. Create
    resp = client.post("/api/v1/conversations", json={"title": "Test Taj Mahal Chat"}, headers=TEST_AUTH_HEADERS)
    assert resp.status_code == 201
    conv = resp.get_json()["conversation"]
    conv_id = conv["id"]
    assert conv["title"] == "Test Taj Mahal Chat"

    # 2. Get details
    resp = client.get(f"/api/v1/conversations/{conv_id}", headers=TEST_AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.get_json()["conversation"]["id"] == conv_id

    # 3. Rename
    resp = client.patch(f"/api/v1/conversations/{conv_id}", json={"title": "Renamed Chat"}, headers=TEST_AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.get_json()["conversation"]["title"] == "Renamed Chat"

    # 4. Delete
    resp = client.delete(f"/api/v1/conversations/{conv_id}", headers=TEST_AUTH_HEADERS)
    assert resp.status_code == 200

    # 5. Verify deleted
    resp = client.get(f"/api/v1/conversations/{conv_id}", headers=TEST_AUTH_HEADERS)
    assert resp.status_code == 404


def test_sync_conversations_batch_endpoint(client):
    """POST /api/v1/conversations/sync batch upserts client-side conversations and messages."""
    sync_payload = {
        "conversations": [
            {
                "id": "client-sync-conv-1",
                "title": "Restored Offline Conversation",
                "created_at": "2026-10-25T10:00:00Z",
                "updated_at": "2026-10-25T10:05:00Z"
            }
        ],
        "messages": {
            "client-sync-conv-1": [
                {
                    "id": "client-sync-msg-1",
                    "role": "user",
                    "content": "Analysis of India Gate",
                    "image_url": "/uploads/test.jpg",
                    "created_at": "2026-10-25T10:00:01Z"
                },
                {
                    "id": "client-sync-msg-2",
                    "role": "assistant",
                    "content": "Identified as India Gate with high confidence.",
                    "created_at": "2026-10-25T10:00:05Z",
                    "results": [
                        {
                            "id": "res-sync-1",
                            "monument_id": "india_gate",
                            "score": 0.88,
                            "score_type": "svm_probability",
                            "accepted": True,
                            "metadata": {"arch": "Triumphal Arch"}
                        }
                    ]
                }
            ]
        }
    }

    resp = client.post("/api/v1/conversations/sync", json=sync_payload, headers=TEST_AUTH_HEADERS)
    assert resp.status_code == 200
    res_data = resp.get_json()
    assert res_data["success"] is True
    assert res_data["synced_conversations"] >= 1
    assert res_data["synced_messages"] == 2

    # Verify that the synced conversation and messages can be fetched via GET /api/v1/conversations/<id>
    get_resp = client.get("/api/v1/conversations/client-sync-conv-1", headers=TEST_AUTH_HEADERS)
    assert get_resp.status_code == 200
    conv_data = get_resp.get_json()
    assert conv_data["conversation"]["title"] == "Restored Offline Conversation"
    assert len(conv_data["messages"]) == 2
    assert conv_data["messages"][0]["content"] == "Analysis of India Gate"
    assert conv_data["messages"][1]["results"][0]["monument_id"] == "india_gate"
