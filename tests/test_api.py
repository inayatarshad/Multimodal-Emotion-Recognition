"""API tests. The registry is forced onto synthetic data so these run anywhere."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from wfb.serving.app import create_app
from wfb.serving.inference import RegistryConfig


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    app = create_app(
        RegistryConfig(
            dataset="mosi",
            force_synthetic=True,
            models=("text_only", "late", "mult"),
            checkpoint_dir="outputs/_nonexistent",
            results_dir="experiments/results",
        )
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_what_is_actually_being_served(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["models_loaded"] == 3
    assert body["dataset"] == "mosi"
    assert body["dataset_source"] == "synthetic"
    # No checkpoints exist in this fixture, so the service must say so rather than
    # presenting randomly initialised models as trained.
    assert body["trained_models"] == 0
    assert body["status"] == "degraded"


def test_model_registry_lists_every_loaded_architecture(client: TestClient) -> None:
    models = client.get("/api/models").json()
    assert {m["name"] for m in models} == {"text_only", "late", "mult"}
    for model in models:
        assert model["parameters"] > 0
        assert model["trained"] is False
    text_only = next(m for m in models if m["name"] == "text_only")
    assert text_only["modalities"] == ["text"]


def test_corruption_catalogue_is_served(client: TestClient) -> None:
    catalogue = client.get("/api/corruptions").json()
    names = {c["name"] for c in catalogue}
    assert {"gaussian_noise", "asr_error", "occlusion", "zero"} <= names
    noise = next(c for c in catalogue if c["name"] == "gaussian_noise")
    assert noise["unit"] == "SNR (dB)"


def test_samples_endpoint_respects_the_limit(client: TestClient) -> None:
    samples = client.get("/api/samples?limit=5").json()
    assert len(samples) == 5
    assert samples[0]["sentiment"] in {"negative", "neutral", "positive"}


def test_samples_endpoint_rejects_an_unserved_dataset(client: TestClient) -> None:
    assert client.get("/api/samples?dataset=meld").status_code == 404


def test_clean_prediction_has_zero_delta(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    body = client.post("/api/predict", json={"sample_id": sample_id, "model": "late"}).json()
    assert body["delta"] == pytest.approx(0.0, abs=1e-6)
    assert body["prediction"] == pytest.approx(body["clean_prediction"], abs=1e-6)
    assert body["corruption_description"] == "clean"


def test_corruption_moves_the_prediction_away_from_clean(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    body = client.post(
        "/api/predict",
        json={
            "sample_id": sample_id,
            "model": "late",
            "corruption": {"text": {"type": "zero", "severity": 1.0}},
        },
    ).json()
    assert body["delta"] != pytest.approx(0.0, abs=1e-6)
    assert "zeroed" in body["corruption_description"]
    assert body["corruption_hash"]


def test_contributions_are_reported_per_modality(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    body = client.post("/api/predict", json={"sample_id": sample_id, "model": "mult"}).json()
    assert {c["modality"] for c in body["contributions"]} == {"text", "audio", "visual"}
    assert sum(c["relative"] for c in body["contributions"]) == pytest.approx(1.0, abs=1e-5)


def test_attention_is_returned_only_when_requested(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    without = client.post("/api/predict", json={"sample_id": sample_id, "model": "mult"}).json()
    assert without["attention"] == {}

    with_attention = client.post(
        "/api/predict",
        json={"sample_id": sample_id, "model": "mult", "return_attention": True},
    ).json()
    assert "text<-audio" in with_attention["attention"]
    assert len(with_attention["attention"]["text<-audio"]) > 0


def test_repeated_requests_are_served_from_cache(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    payload = {
        "sample_id": sample_id,
        "model": "late",
        "corruption": {"audio": {"type": "gaussian_noise", "severity": 0.5}},
    }
    first = client.post("/api/predict", json=payload).json()
    second = client.post("/api/predict", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["prediction"] == pytest.approx(second["prediction"])


def test_compare_runs_every_model_on_identical_input(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    body = client.post(
        "/api/compare",
        json={
            "sample_id": sample_id,
            "corruption": {"audio": {"type": "gaussian_noise", "severity": 0.4}},
        },
    ).json()
    assert len(body["results"]) == 3
    assert {r["model"] for r in body["results"]} == {"text_only", "late", "mult"}
    # Every architecture must report the same corruption — that identity is what makes
    # the side-by-side comparison meaningful.
    assert len({r["corruption_hash"] for r in body["results"]}) == 1


def test_text_only_model_is_unaffected_by_audio_corruption(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    body = client.post(
        "/api/compare",
        json={
            "sample_id": sample_id,
            "models": ["text_only", "late"],
            "corruption": {"audio": {"type": "zero", "severity": 1.0}},
        },
    ).json()
    text_only = next(r for r in body["results"] if r["model"] == "text_only")
    late = next(r for r in body["results"] if r["model"] == "late")
    assert text_only["delta"] == pytest.approx(0.0, abs=1e-6)
    assert late["delta"] != pytest.approx(0.0, abs=1e-6)


def test_unknown_sample_returns_404(client: TestClient) -> None:
    response = client.post("/api/predict", json={"sample_id": "nope", "model": "late"})
    assert response.status_code == 404


def test_unknown_model_returns_404(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    response = client.post("/api/predict", json={"sample_id": sample_id, "model": "nope"})
    assert response.status_code == 404


def test_unknown_corruption_is_a_validation_error(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    response = client.post(
        "/api/predict",
        json={
            "sample_id": sample_id,
            "model": "late",
            "corruption": {"audio": {"type": "not_a_corruption", "severity": 0.5}},
        },
    )
    assert response.status_code == 422


def test_severity_above_one_is_rejected_by_the_schema(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    response = client.post(
        "/api/predict",
        json={
            "sample_id": sample_id,
            "model": "late",
            "corruption": {"audio": {"type": "gaussian_noise", "severity": 5.0}},
        },
    )
    assert response.status_code == 422


def test_corruption_applied_to_the_wrong_modality_is_rejected(client: TestClient) -> None:
    """``asr_error`` is text-only; asking for it on audio must fail loudly."""
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    response = client.post(
        "/api/predict",
        json={
            "sample_id": sample_id,
            "model": "late",
            "corruption": {"audio": {"type": "asr_error", "severity": 0.5}},
        },
    )
    assert response.status_code == 422


def test_unknown_request_fields_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/predict", json={"sample_id": "x", "model": "late", "bogus_field": 1}
    )
    assert response.status_code == 422


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers.get("x-request-id")


def test_openapi_document_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/api/predict" in schema["paths"]
    assert "/api/results/degradation" in schema["paths"]


def test_results_endpoints_respond_even_with_no_sweeps_yet(client: TestClient) -> None:
    """A fresh clone must not 500 on the dashboard endpoints."""
    for path in (
        "/api/results/degradation",
        "/api/results/reliance",
        "/api/results/pareto",
    ):
        response = client.get(path)
        assert response.status_code == 200, path


def test_websocket_streams_comparisons(client: TestClient) -> None:
    sample_id = client.get("/api/samples?limit=1").json()[0]["id"]
    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_json(
            {
                "sample_id": sample_id,
                "models": ["late", "mult"],
                "corruption": {"visual": {"type": "occlusion", "severity": 0.6}},
            }
        )
        message = websocket.receive_json()
        assert len(message["results"]) == 2
        assert "occluded" in message["corruption_description"]


def test_websocket_reports_a_bad_sample_without_closing(client: TestClient) -> None:
    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_json({"sample_id": "nope"})
        assert "error" in websocket.receive_json()
        # The connection must survive a bad message — a slider drag should not kill it.
        websocket.send_json({"sample_id": "still-nope"})
        assert "error" in websocket.receive_json()
