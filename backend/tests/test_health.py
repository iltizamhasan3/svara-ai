def test_health_and_openapi_are_available(client):
    health_response = client.get("/api/v1/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    paths = openapi_response.json()["paths"]
    assert "/api/v1/datasets/upload" in paths
    assert "/api/v1/analyses" in paths
    assert "/api/v1/analyses/{analysis_id}/dashboard" in paths
