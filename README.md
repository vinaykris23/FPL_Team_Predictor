# ⚽ FPL Team Picker

An ML-powered Fantasy Premier League assistant that scores every player using three trained regressors — **Random Forest**, **Gradient Boosting**, and **Linear Regression** — and picks an optimal 15-man squad within the official £100m budget and squad rules.

Built with **Streamlit**, **scikit-learn**, and the **official FPL API**.

Try it live if you want to just view without all the environment setups → https://vk23-fpl-teampredictor.streamlit.app/

---

## 📸 Features

- **Three ML models** trained on 20+ engineered features (xG, xA, form, FDR, history averages, etc.)
- **Composite score** fallback — no ML required, rule-based weighted formula
- **Four picking strategies**: Value, Form, Fixture, and Balanced
- **Six formations**: 4-4-2, 4-3-3, 3-5-2, 5-3-2, 4-5-1, 3-4-3
- **Live FPL API** integration with automatic CSV cache fallback
- **Interactive charts**: goal involvement, xG/xA lollipops, price vs. points scatter, position heatmap
- **Offline mode**: retrain and pick teams entirely from saved CSV cache — no internet required
- **In-app retraining**: retrain models from the sidebar without leaving the app

---

## 🗂️ Project Structure

```
fpl-team-picker/
│
├── fpl_app.py              # Streamlit UI — charts, squad display, sidebar controls
│   
│
├── fpl_pipeline.py         # Data loading, feature engineering, ML training, artefact I/O
│   
│
├── models/                     # Saved joblib artefacts (auto-created on first train)
│   ├── Random_Forest_pipeline.joblib
│   ├── Gradient_Boosting_pipeline.joblib
│   ├── Linear_Regression_pipeline.joblib
│   ├── feature_cols.joblib
│   └── player_data.joblib
│
├── data/                       # CSV cache (auto-created on first fetch)
│   ├── bootstrap_elements.csv
│   ├── bootstrap_teams.csv
│   ├── fixtures.csv
│   ├── player_features.csv
│   └── player_histories.csv
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/fpl-team-picker.git
cd fpl-team-picker
pip install -r requirements.txt
```

### 2. Train the models

Run the pipeline once to fetch live FPL data and train the three models:

```bash
# Quick train (no history — ~5 seconds)
python pipeline/fpl_pipeline.py

# Full train with player history features (~5 minutes)
python pipeline/fpl_pipeline.py --history
```

This creates the `models/` and `data/` directories automatically.

### 3. Launch the app

```bash
streamlit run app/fpl_app.py
```

The app will open at `http://localhost:8501`.

> **First run without pre-trained models?** The app detects this and shows a "Train Models Now" button — no terminal needed.

---

## 🔄 Workflow

```
FPL API ──► load_bootstrap()
         ──► load_fixtures()
         ──► load_player_histories()   ← optional, slower
              │
              ▼
         build_features()             ← 20+ engineered features
              │
              ├──► save_data_cache()  ← writes ./data/*.csv
              │
              ▼
         train_and_evaluate()         ← RF / GB / LR pipelines
              │
              ▼
         save_artefacts()             ← writes ./models/*.joblib
              │
              ▼
         Streamlit App ──► score_players() ──► pick_team()
```

---

## 🧠 ML Models & Features

### Models

| Model | Algorithm | Notes |
|---|---|---|
| Random Forest | `RandomForestRegressor` | 200 estimators, max depth 8 |
| Gradient Boosting | `GradientBoostingRegressor` | 200 estimators, lr=0.05, depth 5 |
| Linear Regression | `LinearRegression` | Baseline model |

All models use a `StandardScaler → model` sklearn `Pipeline`.

### Feature Columns

| Feature | Description |
|---|---|
| `goals_scored`, `assists`, `clean_sheets` | Season totals |
| `minutes`, `bonus`, `bps` | Playing time and bonus point system |
| `form` | FPL rolling form rating |
| `goals_per90`, `assists_per90` | Rate stats |
| `xG_per90`, `xA_per90`, `bonus_per90` | Expected stats per 90 mins |
| `fdr_score` | Fixture difficulty rating (avg next 5 GWs) |
| `availability` | Chance of playing (injury/suspension) |
| `pts_per_million` | Value metric |
| `h_avg_pts`, `h_avg_min`, `h_goals`, `h_assists`, `h_bonus` | Historical averages from player history API |

### Composite Score (no ML)

A weighted rule-based fallback:

```
score = form×2.0 + pts_per_million×1.5 + fdr_score×1.0
      + xG_per90×3.0 + xA_per90×2.0 + bonus_per90×1.0
      + availability×0.5
```

---

## 🏟️ Squad Rules

| Rule | Value |
|---|---|
| Total budget | £100m |
| Squad size | 15 players (11 starters + 4 subs) |
| Max players per club | 3 |
| Formations supported | 4-4-2, 4-3-3, 3-5-2, 5-3-2, 4-5-1, 3-4-3 |

---

## 📊 Picking Strategies

| Strategy | Logic |
|---|---|
| **Value Focus** | Ranks by `pts_per_million` |
| **Form Focus** | Ranks by FPL `form` rating |
| **Fixture Focus** | Ranks by easiest upcoming 5 fixtures (`fdr_score`) |
| **Balanced** | Weighted mix — 40% ML score, 25% value, 20% form, 15% fixture |

---

## 💾 Offline Mode

If the FPL API is unavailable, the app and pipeline fall back to the cached CSV files in `./data/` automatically. You can also force offline mode when retraining:

```bash
python pipeline/fpl_pipeline.py --cache
```

Or click **"From Cache"** in the sidebar of the Streamlit app.

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
requests
matplotlib
seaborn
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📝 Notes

- Player history fetching (`--history`) makes ~700 individual API requests and takes around 5 minutes. The result is cached to `./data/player_histories.csv` for offline use.
- A minimum of **15,000 history rows** is required for history-based features to be active. If the count is below this threshold, those features are zeroed out and the app shows a warning.
- Models and data are saved separately so you can retrain without re-fetching history, or vice versa.
- Also make sure the official Fantasy Premier League website is not blocked in your network.
---

## 📄 License

MIT
