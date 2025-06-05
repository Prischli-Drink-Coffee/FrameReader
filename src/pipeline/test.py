from datetime import datetime, timedelta
import pytest
import random
import string
import hashlib
from copy import deepcopy
from fastapi.testclient import TestClient
from src.pipeline.server import app
from src.utils.custom_logging import setup_logging
from decimal import Decimal
import json

log = setup_logging()
client = TestClient(app)


def generate_random_data(data_type, length=8):
    if data_type == "string":
        return ''.join(random.choices(string.ascii_letters, k=length))
    elif data_type == "number":
        return random.randint(1, 1000000)
    elif data_type == "datetime":
        return datetime.utcnow()
    elif data_type == "fingerprint":
        return hashlib.sha256(f"{generate_random_data('string')}".encode()).hexdigest()
    elif data_type == "decimal":
        return Decimal(str(random.uniform(0.0, 100.0)))
    return None


def api_request(method, url, json_data=None, params=None):
    response = client.request(method, url, json=json_data, params=params)
    return response


def assert_response(response, expected_status, keys=None):
    log.info("-------------------------------------")
    assert response.status_code == expected_status, \
        f"Unexpected status code: {response.status_code}, Response: {response.text}"
    if keys:
        response_data = response.json()
        if isinstance(response_data, list):
            for item in response_data:
                for key in keys:
                    assert key in item
        else:
            for key in keys:
                assert key in response_data
        return response_data
    return None


def generate_test_data(entity_type):
    data_map = {
        "user": {
            "fingerprint_hash": generate_random_data("fingerprint")
        },
        "user_session": {
            "UserID": None,
            "JwtTokenHash": generate_random_data("string", 32),
            "ExpiresAt": datetime.utcnow() + timedelta(days=30),
            "UserAgent": "Test Browser",
            "IPAddress": "127.0.0.1",
            "IsActive": True
        },
        "video_session": {
            "UserID": None,
            "VideoURL": f"https://example.com/video_{generate_random_data('string')}.mp4",
            "ProcessingStatus": "processing",
            "StartedAt": datetime.utcnow()
        },
        "frame_annotation": {
            "VideoSessionID": None,
            "FrameTimestamp": generate_random_data("decimal"),
            "AnnotationData": {
                "objects": [{"type": "person", "confidence": 0.95}],
                "frame_number": generate_random_data("number")
            }
        }
    }
    return data_map.get(entity_type)


def setup_entity(entity_type, endpoint):
    if entity_type == "user_session":
        user_id = setup_entity("user", "server/users")
        session_data = generate_test_data("user_session")
        entity_data = {**session_data, "UserID": user_id}
    elif entity_type == "video_session":
        user_id = setup_entity("user", "server/users")
        session_data = generate_test_data("video_session")
        entity_data = {**session_data, "UserID": user_id}
    elif entity_type == "frame_annotation":
        video_session_id = setup_entity("video_session", "server/video-sessions")
        annotation_data = generate_test_data("frame_annotation")
        entity_data = {**annotation_data, "VideoSessionID": video_session_id}
    elif entity_type == "user":
        entity_data = generate_test_data("user")
        response = api_request("POST", f"/{endpoint}/get-or-create/", 
                             json_data=entity_data)
    else:
        entity_data = generate_test_data(entity_type)
        response = api_request("POST", f"/{endpoint}/", json_data=entity_data)
    
    if entity_type != "user":
        log.info(f"Creating {entity_type} with data: {entity_data}")
        response = api_request("POST", f"/{endpoint}/", json_data=entity_data)
    
    log.info(f"POST {endpoint}/ response: {response.json()}")
    response_data = assert_response(response, 200, keys=["ID" if entity_type == "user" else "id"])
    return response_data.get("ID") or response_data.get("id")


def teardown_entity(endpoint, entity_id):
    response = api_request("DELETE", f"/{endpoint}/{entity_id}")
    assert_response(response, 200)


@pytest.mark.parametrize("entity_type, endpoint, expected_keys", [
    ("user", "server/users", ["FingerprintHash", "FirstVisit", "LastActivity"]),
    ("user_session", "server/user-sessions", ["UserID", "JwtTokenHash", "ExpiresAt"]),
    ("video_session", "server/video-sessions", ["UserID", "VideoURL", "ProcessingStatus"]),
    ("frame_annotation", "server/frame-annotations", ["VideoSessionID", "FrameTimestamp", "AnnotationData"]),
])
def test_create_and_get_entity(entity_type, endpoint, expected_keys):
    log.info("-------------------------------------")
    log.info(f"entity_type: {entity_type}, endpoint: {endpoint}, expected_keys: {expected_keys}")
    
    entity_id = setup_entity(entity_type, endpoint)
    
    response = api_request("GET", f"/{endpoint}/")
    assert_response(response, 200, keys=["ID" if entity_type == "user" else "id"] + expected_keys)
    
    id_param = "user_id" if entity_type == "user" else entity_id
    get_endpoint = f"/{endpoint}/{id_param}" if entity_type != "user" else f"/{endpoint}/user_id/{entity_id}"
    response = api_request("GET", get_endpoint)
    assert_response(response, 200, keys=["ID" if entity_type == "user" else "id"] + expected_keys)
    
    teardown_entity(endpoint, entity_id)


@pytest.mark.parametrize("entity_type, endpoint, update_fields", [
    ("user", "server/users", {"total_sessions": 5}),
    ("user_session", "server/user-sessions", {"user_agent": "Updated Browser"}),
    ("video_session", "server/video-sessions", {"processing_status": "completed"}),
    ("frame_annotation", "server/frame-annotations", {"annotation_data": {"updated": True}}),
])
def test_update_entity(entity_type, endpoint, update_fields):
    log.info("-------------------------------------")
    log.info(f"entity_type: {entity_type}, endpoint: {endpoint}, update_fields: {update_fields}")
    
    entity_id = setup_entity(entity_type, endpoint)
    
    update_endpoint = f"/{endpoint}/{entity_id}"
    response = api_request("PATCH", update_endpoint, json_data=update_fields)
    assert_response(response, 200)
    
    teardown_entity(endpoint, entity_id)


def test_user_fingerprint_operations():
    fingerprint = generate_random_data("fingerprint")
    
    response = api_request("POST", "/server/users/get-or-create/", 
                         json_data={"fingerprint_hash": fingerprint})
    user_data = assert_response(response, 200, keys=["ID", "FingerprintHash"])
    user_id = user_data["ID"]
    
    response = api_request("GET", f"/server/users/fingerprint/{fingerprint}")
    assert_response(response, 200, keys=["ID", "FingerprintHash"])
    
    response = api_request("PATCH", f"/server/users/{user_id}/activity")
    assert_response(response, 200)
    
    teardown_entity("server/users", user_id)


def test_user_session_operations():
    user_id = setup_entity("user", "server/users")
    
    session_data = generate_test_data("user_session")
    session_data["UserID"] = user_id
    
    response = api_request("POST", "/server/user-sessions/", json_data=session_data)
    session_response = assert_response(response, 200, keys=["id"])
    session_id = session_response["id"]
    
    response = api_request("GET", f"/server/user-sessions/user/{user_id}")
    assert_response(response, 200)
    
    response = api_request("PATCH", f"/server/user-sessions/{session_id}/activity")
    assert_response(response, 200)
    
    response = api_request("PATCH", f"/server/user-sessions/{session_id}/deactivate")
    assert_response(response, 200)
    
    teardown_entity("server/user-sessions", session_id)
    teardown_entity("server/users", user_id)


def test_video_session_operations():
    user_id = setup_entity("user", "server/users")
    
    video_data = {
        "user_id": user_id,
        "video_url": f"https://example.com/video_{generate_random_data('string')}.mp4"
    }
    
    response = api_request("POST", "/server/video-sessions/", json_data=video_data)
    video_session = assert_response(response, 200, keys=["id"])
    video_session_id = video_session["id"]
    
    response = api_request("GET", f"/server/video-sessions/user/{user_id}")
    assert_response(response, 200)
    
    response = api_request("PATCH", f"/server/video-sessions/{video_session_id}/complete")
    assert_response(response, 200)
    
    teardown_entity("server/video-sessions", video_session_id)
    teardown_entity("server/users", user_id)


def test_frame_annotation_operations():
    video_session_id = setup_entity("video_session", "server/video-sessions")
    
    annotation_data = {
        "video_session_id": video_session_id,
        "frame_timestamp": float(generate_random_data("decimal")),
        "annotation_data": {"test": "data"}
    }
    
    response = api_request("POST", "/server/frame-annotations/", json_data=annotation_data)
    annotation = assert_response(response, 200, keys=["id"])
    annotation_id = annotation["id"]
    
    response = api_request("GET", f"/server/frame-annotations/video-session/{video_session_id}")
    assert_response(response, 200)
    
    response = api_request("GET", f"/server/frame-annotations/video-session/{video_session_id}/statistics")
    assert_response(response, 200)
    
    teardown_entity("server/frame-annotations", annotation_id)


def test_cookie_session_operations():
    response = api_request("POST", "/server/auth/session/create")
    session_data = assert_response(response, 200, keys=["user_id", "session_id"])
    
    response = api_request("GET", "/server/auth/session/validate")
    assert response.status_code in [200, 401]  # Может быть валидной или нет
    
    response = api_request("POST", "/server/auth/session/logout")
    assert_response(response, 200)


@pytest.mark.parametrize("endpoint, params", [
    ("server/users/active/count", {}),
    ("server/user-sessions/expired/", {}),
    ("server/video-sessions/processing/", {}),
    ("server/frame-annotations/cleanup/old", {"days_old": 30}),
])
def test_utility_endpoints(endpoint, params):
    response = api_request("GET" if "cleanup" not in endpoint else "DELETE", 
                         f"/{endpoint}", params=params)
    assert response.status_code in [200, 404]


if __name__ == "__main__":
    pytest.main([__file__])