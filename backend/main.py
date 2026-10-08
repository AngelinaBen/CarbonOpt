from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

from backend.dependencies import registry, ModelRegistry, get_registry
from backend.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    HistoricalOptionItem,
    HistoricalProfileResponse,
    OptimizationDetails
)
from backend.services.prediction_service import predict_co2, predict_risk_class
from backend.services.anomaly_service import predict_anomaly
from backend.services.clustering_service import predict_cluster
from backend.services.shap_service import get_shap_risk_drivers
from backend.services.optimization_service import run_sector_optimization
from backend.services.historical_service import get_historical_options, get_historical_profile

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all ML models and dataset on backend startup
    success = registry.load_all()
    if not success:
        print("[WARNING] CarbonOpt backend started, but some models failed to load:", registry.status)
    else:
        print("[INFO] CarbonOpt ML models and dataset loaded successfully into memory.")
    yield

app = FastAPI(
    title="CarbonOpt Industrial Intelligence API",
    description="Backend ML API for Next-Year CO2 Prediction, Risk Classification, Anomaly Detection, Profile Clustering, SHAP Drivers, and Sector Optimization.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for local frontend development
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check(reg: ModelRegistry = Depends(get_registry)):
    status_str = "ok" if reg.is_loaded else "degraded"
    return HealthResponse(
        status=status_str,
        models_loaded=reg.is_loaded,
        model_status=reg.status
    )

@app.post("/predict", response_model=PredictResponse)
def predict_integrated(req: PredictRequest, reg: ModelRegistry = Depends(get_registry)):
    if not reg.is_loaded:
        raise HTTPException(status_code=503, detail="ML models are not loaded on backend server.")
        
    try:
        co2_pred = predict_co2(reg, req)
        risk_cls = predict_risk_class(reg, req)
        anom_status, anom_score = predict_anomaly(reg, req)
        cluster_id = predict_cluster(reg, req)
        risk_drivers = get_shap_risk_drivers(reg, req, top_k=8)
        opt_details = run_sector_optimization(reg, req)
        
        return PredictResponse(
            predicted_co2=round(co2_pred, 4),
            risk_class=risk_cls,
            anomaly=anom_status,
            anomaly_score=round(anom_score, 6),
            cluster=cluster_id,
            top_risk_drivers=risk_drivers,
            optimization=opt_details
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/historical-options", response_model=List[HistoricalOptionItem])
def list_historical_options(reg: ModelRegistry = Depends(get_registry)):
    if not reg.is_loaded:
        raise HTTPException(status_code=503, detail="Historical dataset not loaded.")
    return get_historical_options(reg)

@app.get("/historical-profile", response_model=HistoricalProfileResponse)
def fetch_historical_profile(
    state: str = Query(..., description="2-letter State code"),
    year: int = Query(..., description="Year"),
    reg: ModelRegistry = Depends(get_registry)
):
    profile_req = get_historical_profile(reg, state, year)
    if not profile_req:
        raise HTTPException(status_code=404, detail=f"No historical profile found for state '{state}' and year {year}.")
        
    analysis_res = predict_integrated(profile_req, reg)
    return HistoricalProfileResponse(
        profile=profile_req.model_dump(),
        analysis=analysis_res
    )

@app.post("/optimize", response_model=OptimizationDetails)
def optimize_sector_strategy(req: PredictRequest, reg: ModelRegistry = Depends(get_registry)):
    if not reg.is_loaded:
        raise HTTPException(status_code=503, detail="ML models not loaded.")
    return run_sector_optimization(reg, req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
