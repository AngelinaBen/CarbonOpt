import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.dependencies import registry

@pytest.fixture(scope="module", autouse=True)
def load_models_before_tests():
    registry.load_all()

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert "xgb_regression" in data["model_status"]

def test_predict_row_0_benchmark():
    # Exact Row 0 values from CarbonOpt_XGBoost_Regression.xlsx & CarbonOpt_Master_Output.xlsx
    payload = {
        "state": "AK",
        "year": 1962,
        "residential_billion_btu": 1.353,
        "commercial_billion_btu": 2.164,
        "industrial_billion_btu": 21.479,
        "transportation_billion_btu": 14.364,
        "co2_lag_1": 5.08,
        "co2_lag_2": 4.51,
        "energy_growth_rate": 0.05,
        "co2_growth_rate": 0.04,
        "co2_million_metric_tons": 5.628,
        "co2_per_energy": 0.143
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert abs(data["predicted_co2"] - 5.6286) < 0.3
    assert data["risk_class"] == "Low"
    assert data["anomaly"] == "Normal"
    assert data["cluster"] == 0
    assert len(data["top_risk_drivers"]) > 0
    assert "residential_reduction_pct" in data["optimization"]

def test_historical_options():
    response = client.get("/historical-options")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    state_names = [item["state"] for item in data]
    assert "AK" in state_names or "TX" in state_names

def test_historical_profile_ak_1962():
    response = client.get("/historical-profile?state=AK&year=1962")
    assert response.status_code == 200
    data = response.json()
    assert "profile" in data
    assert "analysis" in data
    assert data["profile"]["state"] == "AK"
    assert data["profile"]["year"] == 1962
    assert abs(data["analysis"]["predicted_co2"] - 5.6286) < 0.1

def test_invalid_input_validation():
    payload = {
        "state": "AK",
        "year": 1962,
        "residential_billion_btu": -10.0, # Negative energy is invalid
        "commercial_billion_btu": 2.164,
        "industrial_billion_btu": 21.479,
        "transportation_billion_btu": 14.364,
        "co2_lag_1": 5.08,
        "co2_lag_2": 4.51,
        "energy_growth_rate": 0.05,
        "co2_growth_rate": 0.04
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error
