import json
from pathlib import Path
from uuid import uuid4

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError

from app.schemas.ai import AIResult


ROOT = Path(__file__).parents[2]


def test_valid_ai_fixture_matches_pydantic_contract():
    fixture = json.loads((ROOT / "contracts/fixtures/ai-result.valid.json").read_text())
    result = AIResult.model_validate(fixture)
    assert result.sentiment == "negative"
    assert result.topic_id == 1


def test_invalid_ai_fixture_is_rejected():
    fixture = json.loads((ROOT / "contracts/fixtures/ai-result.invalid.json").read_text())
    with pytest.raises(ValidationError):
        AIResult.model_validate(fixture)


def test_ai_contract_rejects_invalid_label_confidence_and_topic():
    base = {
        "unit_id": uuid4(),
        "source_row_number": 1,
        "text": "feedback",
        "sentiment": "positive",
        "confidence": 0.5,
        "probabilities": {"positive": 0.5, "neutral": 0.25, "negative": 0.25},
        "topic_id": -1,
        "keywords": [],
    }
    with pytest.raises(ValidationError):
        AIResult.model_validate({**base, "sentiment": "mixed"})
    with pytest.raises(ValidationError):
        AIResult.model_validate({**base, "confidence": 1.01})
    with pytest.raises(ValidationError):
        AIResult.model_validate({**base, "topic_id": -2})


def test_dashboard_schema_validates_nested_items():
    schema = json.loads((ROOT / "contracts/dashboard.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    valid = json.loads((ROOT / "contracts/fixtures/dashboard.valid.json").read_text())
    invalid = json.loads((ROOT / "contracts/fixtures/dashboard.invalid.json").read_text())

    Draft202012Validator(schema, format_checker=FormatChecker()).validate(valid)
    with pytest.raises(JSONSchemaValidationError):
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(invalid)
