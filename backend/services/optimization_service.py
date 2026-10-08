import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from backend.dependencies import ModelRegistry
from backend.schemas import PredictRequest, OptimizationDetails
from backend.services.prediction_service import FEATURE_COLUMNS

def run_sector_optimization(registry: ModelRegistry, req: PredictRequest) -> OptimizationDetails:
    base_dict = {
        "state": req.state,
        "year": req.year,
        "residential_billion_btu": req.residential_billion_btu,
        "commercial_billion_btu": req.commercial_billion_btu,
        "industrial_billion_btu": req.industrial_billion_btu,
        "transportation_billion_btu": req.transportation_billion_btu,
        "co2_lag_1": req.co2_lag_1,
        "co2_lag_2": req.co2_lag_2,
        "energy_growth_rate": req.energy_growth_rate,
        "co2_growth_rate": req.co2_growth_rate
    }
    
    baseline_df = pd.DataFrame([base_dict])[FEATURE_COLUMNS]
    baseline_co2 = float(registry.xgb_regression.predict(baseline_df)[0])
    
    def predict_with_reduction(reductions):
        scenario = base_dict.copy()
        scenario["residential_billion_btu"] *= (1.0 - reductions[0])
        scenario["commercial_billion_btu"] *= (1.0 - reductions[1])
        scenario["industrial_billion_btu"] *= (1.0 - reductions[2])
        scenario["transportation_billion_btu"] *= (1.0 - reductions[3])
        scenario_df = pd.DataFrame([scenario])[FEATURE_COLUMNS]
        return float(registry.xgb_regression.predict(scenario_df)[0])
        
    bounds = [(0.0, 0.20), (0.0, 0.20), (0.0, 0.20), (0.0, 0.20)]
    maximum_co2 = predict_with_reduction([0.20, 0.20, 0.20, 0.20])
    target_co2 = maximum_co2
    
    def objective(reductions):
        return float(np.sum(reductions))
        
    def constraint(reductions):
        return float(target_co2 - predict_with_reduction(reductions))
        
    nlc = NonlinearConstraint(constraint, 0, np.inf)
    
    try:
        res = differential_evolution(
            objective,
            bounds=bounds,
            constraints=(nlc,),
            seed=42,
            maxiter=30,
            popsize=8,
            tol=1e-4,
            polish=False,
            workers=1
        )
        optimal_reductions = res.x
        target_satisfied = bool(res.success)
    except Exception:
        optimal_reductions = np.array([0.0, 0.0, 0.0, 0.0])
        target_satisfied = False

    optimized_co2 = predict_with_reduction(optimal_reductions)
    estimated_reduction = baseline_co2 - optimized_co2
    
    return OptimizationDetails(
        residential_reduction_pct=float(round(optimal_reductions[0] * 100, 4)),
        commercial_reduction_pct=float(round(optimal_reductions[1] * 100, 4)),
        industrial_reduction_pct=float(round(optimal_reductions[2] * 100, 4)),
        transportation_reduction_pct=float(round(optimal_reductions[3] * 100, 4)),
        baseline_co2=float(round(baseline_co2, 4)),
        optimized_co2=float(round(optimized_co2, 4)),
        estimated_reduction_mt=float(round(estimated_reduction, 4)),
        target_satisfied=target_satisfied
    )
