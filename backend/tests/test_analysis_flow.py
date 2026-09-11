from io import BytesIO
from threading import Barrier, Thread
from uuid import UUID

from app.repositories.in_memory import InMemoryRepository


USER_A = "00000000-0000-4000-8000-00000000000a"
USER_B = "00000000-0000-4000-8000-00000000000b"


def _upload(client, user_id=USER_A):
    csv_content = b"review,date\naplikasinya bagus,2026-08-01\notp tidak masuk,2026-08-02\n"
    response = client.post(
        "/api/v1/datasets/upload",
        headers={"X-User-ID": user_id},
        files={"file": ("reviews.csv", BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_mock_upload_analysis_status_and_dashboard_flow(client):
    dataset_id = _upload(client)
    create_response = client.post(
        "/api/v1/analyses",
        headers={"X-User-ID": USER_A},
        json={
            "dataset_id": dataset_id,
            "name": "Review Agustus",
            "feedback_column": "review",
            "date_column": "date",
        },
    )
    assert create_response.status_code == 201
    analysis_id = create_response.json()["analysis_id"]
    assert UUID(analysis_id)

    start_response = client.post(f"/api/v1/analyses/{analysis_id}/start", headers={"X-User-ID": USER_A})
    assert start_response.status_code == 202
    assert start_response.json()["status"] == "processing"

    status_response = client.get(f"/api/v1/analyses/{analysis_id}/status", headers={"X-User-ID": USER_A})
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"

    dashboard_response = client.get(f"/api/v1/analyses/{analysis_id}/dashboard", headers={"X-User-ID": USER_A})
    assert dashboard_response.status_code == 200
    dashboard = dashboard_response.json()
    assert dashboard["overview"]["total_feedback"] == 2
    assert dashboard["overview"]["total_units"] == 2
    assert dashboard["overview"]["positive"] == 1
    assert dashboard["overview"]["negative"] == 1
    assert dashboard["insight"]["status"] == "skipped"


def test_analysis_is_not_visible_to_another_user(client):
    dataset_id = _upload(client)
    create_response = client.post(
        "/api/v1/analyses",
        headers={"X-User-ID": USER_A},
        json={"dataset_id": dataset_id, "name": "Private", "feedback_column": "review"},
    )
    analysis_id = create_response.json()["analysis_id"]

    response = client.get(f"/api/v1/analyses/{analysis_id}/status", headers={"X-User-ID": USER_B})
    assert response.status_code == 404


def test_invalid_columns_and_auth_are_rejected(client):
    dataset_id = _upload(client)
    response = client.post(
        "/api/v1/analyses",
        headers={"X-User-ID": USER_A},
        json={"dataset_id": dataset_id, "name": "Invalid", "feedback_column": "missing"},
    )
    assert response.status_code == 400

    missing_auth = client.get("/api/v1/analyses")
    assert missing_auth.status_code == 401
    assert missing_auth.json()["error"]["code"] == "missing_authentication"

    malformed_uuid = client.get(
        "/api/v1/analyses/not-a-uuid/status",
        headers={"X-User-ID": USER_A},
    )
    assert malformed_uuid.status_code == 422
    assert malformed_uuid.json()["error"]["code"] == "validation_error"


def test_upload_normalizes_headers_and_rejects_data_loss(client):
    response = client.post(
        "/api/v1/datasets/upload",
        headers={"X-User-ID": USER_A},
        files={"file": ("spaced.csv", BytesIO(b" review ,date\nbagus,2026-08-01\n"), "text/csv")},
    )
    assert response.status_code == 201
    assert response.json()["columns"] == ["review", "date"]
    assert response.json()["preview"][0]["review"] == "bagus"

    extra_field = client.post(
        "/api/v1/datasets/upload",
        headers={"X-User-ID": USER_A},
        files={"file": ("extra.csv", BytesIO(b"review,date\nhello,world,unexpected\n"), "text/csv")},
    )
    assert extra_field.status_code == 400
    assert extra_field.json()["error"]["code"] == "extra_csv_fields"

    duplicate_header = client.post(
        "/api/v1/datasets/upload",
        headers={"X-User-ID": USER_A},
        files={"file": ("duplicate.csv", BytesIO(b"review, review\nhello,world\n"), "text/csv")},
    )
    assert duplicate_header.status_code == 400
    assert duplicate_header.json()["error"]["code"] == "duplicate_columns"


def test_oversized_csv_field_returns_structured_client_error(client):
    content = b"review\n" + (b"a" * 100_001) + b"\n"
    response = client.post(
        "/api/v1/datasets/upload",
        headers={"X-User-ID": USER_A},
        files={"file": ("long.csv", BytesIO(content), "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_csv"


def test_analysis_claim_is_atomic():
    repository = InMemoryRepository()
    user_id = UUID(USER_A)
    dataset = repository.create_dataset(
        user_id=user_id,
        filename="reviews.csv",
        file_type="csv",
        columns=["review"],
        rows=[{"review": "bagus"}],
        preview=[{"review": "bagus"}],
    )
    analysis = repository.create_analysis(
        user_id=user_id,
        dataset_id=dataset.id,
        name="Concurrent claim",
        feedback_column="review",
        date_column=None,
    )

    barrier = Barrier(2)
    claims: list[bool] = []

    def claim() -> None:
        barrier.wait()
        _, did_claim = repository.claim_analysis_for_user(analysis.id, user_id)
        claims.append(did_claim)

    threads = [Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert claims.count(True) == 1
