import pandas as pd
from backend.dependencies import ModelRegistry
from backend.schemas import PredictRequest

FEATURE_COLUMNS = [
    "state",
    "year",
    "residential_billion_btu",
    "commercial_billion_btu",
    "industrial_billion_btu",
    "transportation_billion_btu",
    "co2_lag_1",
    "co2_lag_2",
    "energy_growth_rate",
    "co2_growth_rate"
]

def build_feature_dataframe(req: PredictRequest) -> pd.DataFrame:
    data = {
        "state": [req.state],
        "year": [req.year],
        "residential_billion_btu": [req.residential_billion_btu],
        "commercial_billion_btu": [req.commercial_billion_btu],
        "industrial_billion_btu": [req.industrial_billion_btu],
        "transportation_billion_btu": [req.transportation_billion_btu],
        "co2_lag_1": [req.co2_lag_1],
        "co2_lag_2": [req.co2_lag_2],
        "energy_growth_rate": [req.energy_growth_rate],
        "co2_growth_rate": [req.co2_growth_rate]
    }
    return pd.DataFrame(data)[FEATURE_COLUMNS]

def predict_co2(registry: ModelRegistry, req: PredictRequest) -> float:
    df = build_feature_dataframe(req)
    raw_pred = registry.xgb_regression.predict(df)[0]
    return float(raw_pred)

def predict_risk_class(registry: ModelRegistry, req: PredictRequest) -> str:
    df = build_feature_dataframe(req)
    raw_encoded = registry.xgb_classification.predict(df)[0]
    label = registry.risk_label_encoder.inverse_transform([raw_encoded])[0]
    return str(label)
