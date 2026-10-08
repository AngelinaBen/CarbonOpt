# CarbonOpt — Industrial Carbon Intelligence & Optimization Platform

CarbonOpt is an end-to-end industrial carbon-emission analysis platform designed to execute multi-stage machine learning inference over trained regional energy consumption and emission models.

The system combines **XGBoost Regression**, **XGBoost Risk Classification**, **Isolation Forest Anomaly Detection**, **KMeans Profile Clustering**, **SHAP Explainability**, and **SciPy Sector Optimization** into a unified web application dashboard.

---

## 🏗 System Architecture

```
React Frontend (Vite + TS + Tailwind CSS)
            │
            ▼  HTTP REST (Axios / Fetch)
FastAPI Backend API (Python 3.13 / Uvicorn)
            │
    ┌───────┴──────────────────────────────────────────┐
    ▼                                                  ▼
ML Model Registry (.pkl)                    Processed Data Cache
├── xgb_regression.pkl                      ├── CarbonOpt_XGBoost_Regression.xlsx
├── xgb_classification.pkl                  ├── CarbonOpt_IsolationForest.xlsx
├── risk_label_encoder.pkl                  └── CarbonOpt_Master_Output.xlsx
├── isolation_forest.pkl & scaler
└── kmeans_clustering.pkl & scaler
```

---

## 🚀 Key Features

1. **Next-Year CO₂ Forecast**: Predicts regional emissions in Million Metric Tons ($R^2 \approx 0.994$).
2. **Carbon Risk Classification**: Categorizes risk profiles into `Low`, `Medium`, or `High` ($96\%$ validation accuracy).
3. **Isolation Forest Anomaly Scoring**: Identifies unusual statistical profile patterns ($3,003$ normal, $159$ anomalies in training set).
4. **KMeans Profile Clustering**: Assigns profiles to learned structural energy-density clusters ($K=2$).
5. **SHAP Model Risk Drivers**: Computes dynamic game-theoretic feature attributions for predictions.
6. **Sector Reduction Strategy**: Differential evolution scenario optimization identifying optimal energy reductions across Residential, Commercial, Industrial, and Transportation sectors.
7. **Historical Profile Mode**: Select State & Year from 3,162 actual historical records to run automated inference.

---

## 🛠 Project Structure

```
carbonOPT/
├── backend/
│   ├── main.py                     # FastAPI routes, CORS, lifespan model loader
│   ├── dependencies.py             # ModelRegistry singleton & data cache loader
│   ├── schemas.py                  # Pydantic request/response schemas
│   ├── requirements.txt            # Python dependencies
│   ├── services/
│   │   ├── prediction_service.py   # Regression & Classification inference
│   │   ├── anomaly_service.py      # Isolation Forest anomaly scoring
│   │   ├── clustering_service.py   # KMeans profile assignment
│   │   ├── shap_service.py         # SHAP TreeExplainer risk drivers
│   │   ├── optimization_service.py # SciPy sector reduction optimization
│   │   └── historical_service.py   # Dataset lookup by state & year
│   └── tests/
│       └── test_api.py             # Pytest test suite & benchmark verification
│
├── frontend/
│   ├── index.html
│   ├── vite.config.ts              # Vite config with React & Tailwind plugins
│   ├── package.json
│   └── src/
│       ├── api/client.ts           # Centralized API service
│       ├── types/carbonopt.ts      # TypeScript payload interfaces
│       ├── components/             # Reusable UI components (MetricCard, ShapChart, etc.)
│       └── pages/                  # LandingPage, AnalysisPage, HistoricalPage, AboutPage
│
├── models/                         # Serialized ML model pickles (.pkl)
├── data/processed/                 # Industrial energy & emission datasets (.xlsx)
└── outputs/final/                  # Master benchmark reference dataset (.xlsx)
```

---

## 🚦 Getting Started

### Prerequisites
- Python 3.10+ (Python 3.13 recommended)
- Node.js 18+ and npm

---

### Backend Setup & Execution

1. Navigate to the backend directory:
   ```bash
   cd carbonOPT
   ```
2. Activate your virtual environment:
   ```bash
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```
3. Install backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Start the FastAPI development server:
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   The API interactive docs will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### Frontend Setup & Execution

1. Open a new terminal and navigate to the `frontend/` directory:
   ```bash
   cd carbonOPT/frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open your browser and navigate to [http://localhost:5173](http://localhost:5173).

---

## 🧪 Testing & Verification

Run the automated Pytest suite to verify API endpoints, error validation, and exact numerical benchmark reproducibility against `CarbonOpt_Master_Output.xlsx`:

```bash
python -m pytest backend/tests/test_api.py -v
```

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check and model loading status |
| `POST` | `/predict` | Executes integrated ML pipeline (Regression, Classification, Anomaly, Cluster, SHAP, Optimization) |
| `GET` | `/historical-options` | Returns list of available states and years in dataset |
| `GET` | `/historical-profile` | Retrieves historical record and runs full integrated analysis |
| `POST` | `/optimize` | Standalone endpoint for sector reduction strategy optimization |

---

## 📌 Model & Data Disclaimers

- **No Online Retraining**: All models are loaded once at backend startup from `.pkl` artifacts.
- **Statistical Attribution**: SHAP risk drivers reflect statistical model weights rather than physical causal claims.
- **Scenario Optimization**: Sector reduction recommendations represent mathematical scenario analysis to support industrial carbon decision-making.
