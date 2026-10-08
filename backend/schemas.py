from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    state: str = Field(..., json_schema_extra={"example": "AK"}, description="US State code (e.g. AK, TX)")
    year: int = Field(..., ge=1900, le=2100, json_schema_extra={"example": 1962}, description="Analysis year")
    residential_billion_btu: float = Field(..., ge=0.0, json_schema_extra={"example": 1.353}, description="Residential energy consumption in billion Btu")
    commercial_billion_btu: float = Field(..., ge=0.0, json_schema_extra={"example": 2.164}, description="Commercial energy consumption in billion Btu")
    industrial_billion_btu: float = Field(..., ge=0.0, json_schema_extra={"example": 21.479}, description="Industrial energy consumption in billion Btu")
    transportation_billion_btu: float = Field(..., ge=0.0, json_schema_extra={"example": 14.364}, description="Transportation energy consumption in billion Btu")
    co2_lag_1: float = Field(..., ge=0.0, json_schema_extra={"example": 5.08}, description="CO2 emissions from 1 year prior (Million Metric Tons)")
    co2_lag_2: float = Field(..., ge=0.0, json_schema_extra={"example": 4.51}, description="CO2 emissions from 2 years prior (Million Metric Tons)")
    energy_growth_rate: float = Field(..., json_schema_extra={"example": 0.05}, description="Energy consumption growth rate")
    co2_growth_rate: float = Field(..., json_schema_extra={"example": 0.04}, description="CO2 emissions growth rate")
    co2_million_metric_tons: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 5.628}, description="Current year total CO2 emissions (Mt). If omitted, inferred from co2_lag_1.")
    co2_per_energy: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 0.143}, description="CO2 per energy ratio. If omitted, derived as co2 / total_energy.")

class RiskDriverItem(BaseModel):
    feature: str = Field(..., description="Raw encoded feature name")
    clean_feature: str = Field(..., description="Human-readable feature name")
    shap_value: float = Field(..., description="SHAP feature contribution value")

class OptimizationDetails(BaseModel):
    residential_reduction_pct: float = Field(..., description="Recommended residential sector energy reduction percentage")
    commercial_reduction_pct: float = Field(..., description="Recommended commercial sector energy reduction percentage")
    industrial_reduction_pct: float = Field(..., description="Recommended industrial sector energy reduction percentage")
    transportation_reduction_pct: float = Field(..., description="Recommended transportation sector energy reduction percentage")
    baseline_co2: float = Field(..., description="Baseline predicted next-year CO2 (Mt)")
    optimized_co2: float = Field(..., description="Optimized predicted next-year CO2 (Mt)")
    estimated_reduction_mt: float = Field(..., description="Estimated CO2 reduction in Million Metric Tons")
    target_satisfied: bool = Field(..., description="Whether optimization target constraint was satisfied")

class PredictResponse(BaseModel):
    predicted_co2: float = Field(..., description="Predicted next-year CO2 emissions (Million Metric Tons)")
    risk_class: str = Field(..., description="Carbon risk classification (Low, Medium, High)")
    anomaly: str = Field(..., description="Anomaly status (Normal or Anomaly)")
    anomaly_score: float = Field(..., description="Isolation Forest anomaly decision score")
    cluster: int = Field(..., description="Learned emission profile cluster ID (0 or 1)")
    top_risk_drivers: List[RiskDriverItem] = Field(..., description="Top SHAP feature risk drivers")
    optimization: OptimizationDetails = Field(..., description="Sector-wise reduction strategy recommendations")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status (ok / error)")
    models_loaded: bool = Field(..., description="True if all models loaded successfully")
    model_status: Dict[str, str] = Field(..., description="Status breakdown of individual model files")

class HistoricalOptionItem(BaseModel):
    state: str
    years: List[int]

class HistoricalProfileResponse(BaseModel):
    profile: Dict[str, Any]
    analysis: PredictResponse
