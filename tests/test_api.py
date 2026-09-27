import httpx

from jev_runtime.api import create_app


async def test_decision_endpoint_and_separate_admin_auth(runtime, question):
    app = create_app(instance=runtime, api_key="data-key", admin_key="admin-key")
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        body = {
            "model": "model",
            "input": {"text": "refund"},
            "questions": [question.model_dump(mode="json")],
        }
        assert (await client.post("/v1/decisions", json=body)).status_code == 401
        result = await client.post(
            "/v1/decisions", json=body, headers={"Authorization": "Bearer data-key"}
        )
        assert result.status_code == 200
        assert result.json()["answers"]["intent"]["value"] == "billing"
        denied = await client.get("/admin/bundles", headers={"Authorization": "Bearer data-key"})
        assert denied.status_code == 401
        allowed = await client.get("/admin/bundles", headers={"Authorization": "Bearer admin-key"})
        assert allowed.status_code == 200
        assert allowed.json()["leases"] == []


async def test_systemone_preserves_semantics_and_rejects_unknown_fields(runtime):
    app = create_app(instance=runtime)
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        body = {
            "model": "model",
            "state": "refund",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Classify",
                    "criteria": {"a": "Refund", "b": "Other"},
                }
            },
        }
        response = await client.post("/v1/systemone", json=body)
        assert response.status_code == 200
        answer = response.json()["answers"]["q"]
        assert answer["choice"] == "a"
        assert answer["probability_semantics"] == "conditional_label_distribution"
        body["questions"]["q"]["invented"] = True
        response = await client.post("/v1/systemone", json=body)
        assert response.status_code == 400


async def test_unknown_bundle_is_explicit_error(runtime, question):
    app = create_app(instance=runtime)
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/decisions",
            json={
                "model": "model",
                "bundle": "test@99",
                "input": {"text": "refund"},
                "questions": [question.model_dump(mode="json")],
            },
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "bundle_not_active"
