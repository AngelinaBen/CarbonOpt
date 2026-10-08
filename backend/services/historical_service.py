from typing import List, Dict, Any, Optional
import pandas as pd
from backend.dependencies import ModelRegistry
from backend.schemas import HistoricalOptionItem, PredictRequest

def get_historical_options(registry: ModelRegistry) -> List[HistoricalOptionItem]:
    if registry.historical_dataset is None:
        return []
    
    df = registry.historical_dataset
    grouped = df.groupby("state")["year"].apply(lambda years: sorted(list(set(years)))).reset_index()
    
    options = []
    for _, row in grouped.iterrows():
        options.append(
            HistoricalOptionItem(
                state=str(row["state"]),
                years=[int(y) for y in row["year"]]
            )
        )
    options.sort(key=lambda x: x.state)
    return options

def get_historical_profile(registry: ModelRegistry, state: str, year: int) -> Optional[PredictRequest]:
    if registry.historical_dataset is None:
        return None
        
    df = registry.historical_dataset
    match = df[(df["state"] == state) & (df["year"] == year)]
    
    if match.empty:
        return None
        
    row = match.iloc[0].to_dict()
    
    co2_mt = float(row.get("co2_million_metric_tons", row.get("co2_lag_1", 0.0)))
    total_e = (
        float(row["residential_billion_btu"]) +
        float(row["commercial_billion_btu"]) +
        float(row["industrial_billion_btu"]) +
        float(row["transportation_billion_btu"])
    )
    co2_per_e = (co2_mt / total_e) if total_e > 0 else 0.0

    return PredictRequest(
        state=str(row["state"]),
        year=int(row["year"]),
        residential_billion_btu=float(row["residential_billion_btu"]),
        commercial_billion_btu=float(row["commercial_billion_btu"]),
        industrial_billion_btu=float(row["industrial_billion_btu"]),
        transportation_billion_btu=float(row["transportation_billion_btu"]),
        co2_lag_1=float(row["co2_lag_1"]),
        co2_lag_2=float(row["co2_lag_2"]),
        energy_growth_rate=float(row["energy_growth_rate"]),
        co2_growth_rate=float(row["co2_growth_rate"]),
        co2_million_metric_tons=co2_mt,
        co2_per_energy=co2_per_e
    )
