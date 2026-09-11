import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.repositories.in_memory import repository


@pytest.fixture(autouse=True)
def reset_repository():
    repository.clear()
    yield
    repository.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
