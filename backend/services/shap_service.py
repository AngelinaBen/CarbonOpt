from typing import List
import numpy as np
import shap
from backend.dependencies import ModelRegistry
from backend.schemas import PredictRequest, RiskDriverItem
from backend.services.prediction_service import build_feature_dataframe

def get_shap_risk_drivers(registry: ModelRegistry, req: PredictRequest, top_k: int = 8) -> List[RiskDriverItem]:
    df = build_feature_dataframe(req)
    pipeline = registry.xgb_classification
    
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    X_trans = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_trans)
    
    pred_class_idx = int(classifier.predict(X_trans)[0])
    
    if isinstance(shap_values, list):
        sample_shap = shap_values[pred_class_idx][0]
    elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        sample_shap = shap_values[0, :, pred_class_idx]
    else:
        sample_shap = shap_values[0]
        
    drivers = []
    for raw_name, val in zip(feature_names, sample_shap):
        val_float = float(val)
        if raw_name.startswith("remainder__"):
            clean_name = raw_name.replace("remainder__", "")
        elif raw_name.startswith("state__state_"):
            clean_name = "State: " + raw_name.replace("state__state_", "")
        else:
            clean_name = raw_name
            
        drivers.append(
            RiskDriverItem(
                feature=str(raw_name),
                clean_feature=str(clean_name),
                shap_value=val_float
            )
        )
        
    drivers.sort(key=lambda x: abs(x.shap_value), reverse=True)
    return drivers[:top_k]
