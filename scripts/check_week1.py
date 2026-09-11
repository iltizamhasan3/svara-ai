import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(path: str) -> Path:
    target = ROOT / path
    if not target.exists():
        raise AssertionError(f"missing required artifact: {path}")
    return target


def main() -> None:
    required_files = [
        "README.md",
        "contracts/ai-result.schema.json",
        "contracts/dashboard.schema.json",
        "contracts/fixtures/ai-result.valid.json",
        "contracts/fixtures/ai-result.invalid.json",
        "contracts/fixtures/dashboard.valid.json",
        "contracts/fixtures/dashboard.invalid.json",
        "contracts/fixtures/seed_feedback.csv",
        "docs/SVARA_AI_AI_CONTRACT.md",
        "docs/SVARA_AI_API_CONTRACT.md",
        "docs/SVARA_AI_ACCEPTANCE_CRITERIA.md",
        "docs/SVARA_AI_DEVELOPMENT_GUIDE.md",
        "docs/SVARA_AI_WIREFRAMES.md",
        "backend/pyproject.toml",
        "backend/app/main.py",
        "backend/tests/test_analysis_flow.py",
        "frontend/package.json",
        "frontend/app/page.tsx",
        "supabase/config.toml",
        "supabase/migrations/20260911000000_initial_schema.sql",
        "supabase/seed.sql",
    ]
    for path in required_files:
        require(path)

    ai_schema = json.loads(require("contracts/ai-result.schema.json").read_text())
    required_ai_fields = set(ai_schema["required"])
    assert {"unit_id", "sentiment", "confidence", "probabilities", "topic_id", "keywords"} <= required_ai_fields
    assert ai_schema["properties"]["sentiment"]["enum"] == ["positive", "neutral", "negative"]
    assert ai_schema["properties"]["topic_id"]["minimum"] == -1
    assert ai_schema["properties"]["confidence"]["minimum"] == 0
    assert ai_schema["properties"]["confidence"]["maximum"] == 1

    valid_fixture = json.loads(require("contracts/fixtures/ai-result.valid.json").read_text())
    invalid_fixture = json.loads(require("contracts/fixtures/ai-result.invalid.json").read_text())
    assert valid_fixture["sentiment"] in {"positive", "neutral", "negative"}
    assert 0 <= valid_fixture["confidence"] <= 1
    assert invalid_fixture["confidence"] > 1

    dashboard_schema = json.loads(require("contracts/dashboard.schema.json").read_text())
    assert dashboard_schema["properties"]["topics"]["items"]["$ref"] == "#/$defs/topic"
    assert dashboard_schema["properties"]["trend"]["items"]["$ref"] == "#/$defs/trend"
    dashboard_fixture = json.loads(require("contracts/fixtures/dashboard.valid.json").read_text())
    assert dashboard_fixture["trend"][0]["sentiment"]["negative"] == 100

    api_contract = require("docs/SVARA_AI_API_CONTRACT.md").read_text()
    for endpoint in [
        "GET /health",
        "POST /datasets/upload",
        "POST /analyses",
        "POST /analyses/{analysis_id}/start",
        "GET /analyses/{analysis_id}/status",
        "GET /analyses/{analysis_id}/dashboard",
        "GET /analyses?page=1&limit=20",
    ]:
        assert endpoint in api_contract, f"missing API contract endpoint: {endpoint}"

    migration = require("supabase/migrations/20260911000000_initial_schema.sql").read_text()
    for table in [
        "profiles",
        "datasets",
        "analyses",
        "feedback_items",
        "analysis_units",
        "sentiment_predictions",
        "topic_clusters",
        "topic_keywords",
        "unit_topic_assignments",
        "issues",
        "analysis_metrics",
        "ai_insights",
    ]:
        assert f"create table public.{table}" in migration
    for constraint in [
        "analyses_dataset_owner_fk",
        "analysis_units_analysis_dataset_fk",
        "analysis_units_feedback_dataset_fk",
        "unit_topic_assignments_unit_analysis_fk",
        "unit_topic_assignments_topic_analysis_fk",
        "issues_topic_analysis_fk",
    ]:
        assert constraint in migration
    assert migration.count("enable row level security") == 12

    tracker = require("docs/SVARA_AI_Weekly Task Tracker.md").read_text()
    week1 = tracker.split("# Minggu 2", 1)[0]
    assert not re.search(r"^- \[ \]", week1, flags=re.MULTILINE), "Week 1 still has unchecked items"

    frontend_package = json.loads(require("frontend/package.json").read_text())
    assert {"next", "react", "react-dom"} <= set(frontend_package["dependencies"])
    assert {"tailwindcss", "typescript"} <= set(frontend_package["devDependencies"])

    print("Week 1 foundation checks passed")


if __name__ == "__main__":
    main()
