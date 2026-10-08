from pathlib import Path
import joblib
import pandas as pd

class ModelRegistry:
    def __init__(self):
        self.xgb_regression = None
        self.xgb_classification = None
        self.risk_label_encoder = None
        self.isolation_forest = None
        self.isolation_scaler = None
        self.kmeans_clustering = None
        self.kmeans_scaler = None
        self.historical_dataset = None
        self.master_output = None
        self.is_loaded = False
        self.status = {}

    def load_all(self):
        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"
        data_dir = project_root / "data" / "processed"
        master_path = project_root / "outputs" / "final" / "CarbonOpt_Master_Output.xlsx"

        files_to_load = {
            "xgb_regression": models_dir / "xgb_regression.pkl",
            "xgb_classification": models_dir / "xgb_classification.pkl",
            "risk_label_encoder": models_dir / "risk_label_encoder.pkl",
            "isolation_forest": models_dir / "isolation_forest.pkl",
            "isolation_scaler": models_dir / "isolation_scaler.pkl",
            "kmeans_clustering": models_dir / "kmeans_clustering.pkl",
            "kmeans_scaler": models_dir / "kmeans_scaler.pkl",
        }

        all_ok = True
        for name, path in files_to_load.items():
            if not path.exists():
                self.status[name] = f"Missing file: {path.name}"
                all_ok = False
                continue
            try:
                obj = joblib.load(path)
                setattr(self, name, obj)
                self.status[name] = "Loaded"
            except Exception as e:
                self.status[name] = f"Failed to load: {str(e)}"
                all_ok = False

        # Load historical dataset
        try:
            hist_path = data_dir / "CarbonOpt_XGBoost_Regression.xlsx"
            if hist_path.exists():
                self.historical_dataset = pd.read_excel(hist_path)
                self.status["historical_dataset"] = "Loaded"
            else:
                self.status["historical_dataset"] = f"Missing file: {hist_path.name}"
                all_ok = False
        except Exception as e:
            self.status["historical_dataset"] = f"Failed to load dataset: {str(e)}"
            all_ok = False

        # Load master output dataset for reference
        try:
            if master_path.exists():
                self.master_output = pd.read_excel(master_path)
                self.status["master_output"] = "Loaded"
            else:
                self.status["master_output"] = f"Missing file: {master_path.name}"
        except Exception as e:
            self.status["master_output"] = f"Failed to load master: {str(e)}"

        self.is_loaded = all_ok
        return self.is_loaded

# Global singleton instance
registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    return registry
