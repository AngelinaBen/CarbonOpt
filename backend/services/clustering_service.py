from backend.dependencies import ModelRegistry
from backend.schemas import PredictRequest
from backend.services.anomaly_service import build_isolation_forest_dataframe

def predict_cluster(registry: ModelRegistry, req: PredictRequest) -> int:
    df = build_isolation_forest_dataframe(req)
    scaled_x = registry.kmeans_scaler.transform(df)
    cluster_id = int(registry.kmeans_clustering.predict(scaled_x)[0])
    return cluster_id
