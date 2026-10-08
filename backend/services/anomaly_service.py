from typing import Tuple
import pandas as pd
from backend.dependencies import ModelRegistry
from backend.schemas import PredictRequest

IF_FEATURES = [
    "residential_billion_btu",
    "commercial_billion_btu",
    "industrial_billion_btu",
    "transportation_billion_btu",
    "co2_million_metric_tons",
    "co2_per_energy",
    "energy_growth_rate",
    "co2_growth_rate"
]

def build_isolation_forest_dataframe(req: PredictRequest) -> pd.DataFrame:
    total_energy = (
        req.residential_billion_btu +
        req.commercial_billion_btu +
        req.industrial_billion_btu +
        req.transportation_billion_btu
    )
    
    co2_mt = req.co2_million_metric_tons if req.co2_million_metric_tons is not None else req.co2_lag_1
    
    if req.co2_per_energy is not None:
        co2_per_e = req.co2_per_energy
    else:
        co2_per_e = (co2_mt / total_energy) if total_energy > 0 else 0.0

    data = {
        "residential_billion_btu": [req.residential_billion_btu],
        "commercial_billion_btu": [req.commercial_billion_btu],
        "industrial_billion_btu": [req.industrial_billion_btu],
        "transportation_billion_btu": [req.transportation_billion_btu],
        "co2_million_metric_tons": [co2_mt],
        "co2_per_energy": [co2_per_e],
        "energy_growth_rate": [req.energy_growth_rate],
        "co2_growth_rate": [req.co2_growth_rate]
    }
    return pd.DataFrame(data)[IF_FEATURES]

def predict_anomaly(registry: ModelRegistry, req: PredictRequest) -> Tuple[str, float]:
    df = build_isolation_forest_dataframe(req)
    scaled_x = registry.isolation_scaler.transform(df)
    pred_label = registry.isolation_forest.predict(scaled_x)[0]
    score = float(registry.isolation_forest.decision_function(scaled_x)[0])
    
    anomaly_status = "Normal" if pred_label == 1 else "Anomaly"
    return anomaly_status, score
