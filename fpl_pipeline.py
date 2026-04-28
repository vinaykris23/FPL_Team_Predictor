"""
1st run this file then run the app file.
This saves models and data to their folders in the same directory.
"""

import os
import warnings
import requests
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

warnings.filterwarnings("ignore")


FPL_BASE      = "https://fantasy.premierleague.com/api"
MODELS_DIR    = "models"
DATA_DIR      = "data"              # CSV cache — fallback when API is down
MIN_HISTORY_ROWS = 15_000           # loader raises if fewer rows are returned

POSITION_MAP  = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

FEATURE_COLS  = [
    "goals_scored", "assists", "clean_sheets", "minutes", "bonus",
    "bps", "form", "goals_per90", "assists_per90",
    "xG_per90", "xA_per90", "bonus_per90", "fdr_score",
    "availability", "pts_per_million",
    "h_avg_pts", "h_avg_min", "h_goals", "h_assists", "h_bonus",
]

#CSV

def _cache_path(filename: str) -> str:

    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)


def cache_exists() -> bool:

    required = [
        "bootstrap_elements.csv",
        "bootstrap_teams.csv",
        "fixtures.csv",
        "player_features.csv",
    ]
    return all(os.path.exists(_cache_path(f)) for f in required)


def save_data_cache(
    bootstrap: dict,
    fixtures: list,
    history_df: pd.DataFrame,
    player_df: pd.DataFrame,
) -> None:

    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(bootstrap["elements"]).to_csv(
        _cache_path("bootstrap_elements.csv"), index=False)
    pd.DataFrame(bootstrap["teams"]).to_csv(
        _cache_path("bootstrap_teams.csv"), index=False)
    pd.DataFrame(fixtures).to_csv(
        _cache_path("fixtures.csv"), index=False)
    if not history_df.empty:
        history_df.to_csv(_cache_path("player_histories.csv"), index=False)
    player_df.to_csv(_cache_path("player_features.csv"), index=False)
    print(f"  CSV cache saved to ./{DATA_DIR}/")


def load_data_cache() -> tuple[dict, list, pd.DataFrame, pd.DataFrame]:

    missing = [
        f for f in ["bootstrap_elements.csv", "bootstrap_teams.csv",
                    "fixtures.csv", "player_features.csv"]
        if not os.path.exists(_cache_path(f))
    ]
    if missing:
        raise FileNotFoundError(
            f"Cache files missing: {missing}. "
            "Run with live API access first to build the cache."
        )
    bootstrap = {
        "elements": pd.read_csv(_cache_path("bootstrap_elements.csv")).to_dict("records"),
        "teams":    pd.read_csv(_cache_path("bootstrap_teams.csv")).to_dict("records"),
    }
    fixtures   = pd.read_csv(_cache_path("fixtures.csv")).to_dict("records")
    player_df  = pd.read_csv(_cache_path("player_features.csv"))
    hist_path  = _cache_path("player_histories.csv")
    history_df = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame()


    if not history_df.empty and len(history_df) < MIN_HISTORY_ROWS:
        print(
            f"  WARNING: cached history has only {len(history_df):,} rows "
            f"(minimum required: {MIN_HISTORY_ROWS:,}). "
        )
    print(f"  CSV cache loaded from ./{DATA_DIR}/")
    return bootstrap, fixtures, history_df, player_df


# api bootstraping

def load_bootstrap(use_cache_on_failure: bool = True) -> dict:
    try:
        print("  Fetching bootstrap data...")
        r = requests.get(f"{FPL_BASE}/bootstrap-static/", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if use_cache_on_failure and os.path.exists(_cache_path("bootstrap_elements.csv")):
            print(f"  API unavailable ({e}). Loading bootstrap from CSV cache...")
            return {
                "elements": pd.read_csv(_cache_path("bootstrap_elements.csv")).to_dict("records"),
                "teams":    pd.read_csv(_cache_path("bootstrap_teams.csv")).to_dict("records"),
            }
        raise


def load_fixtures(use_cache_on_failure: bool = True) -> list:

    try:
        print("  Fetching fixtures...")
        r = requests.get(f"{FPL_BASE}/fixtures/", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if use_cache_on_failure and os.path.exists(_cache_path("fixtures.csv")):
            print(f"  API unavailable ({e}). Loading fixtures from CSV cache...")
            return pd.read_csv(_cache_path("fixtures.csv")).to_dict("records")
        raise


def load_player_histories(
    player_ids: list,
    use_cache_on_failure: bool = True,
) -> pd.DataFrame:

    hist_path = _cache_path("player_histories.csv")

    def _validate_and_return(df: pd.DataFrame, source: str) -> pd.DataFrame:
        #Check row count; raise if below minimum
        if len(df) < MIN_HISTORY_ROWS:
            raise ValueError(
                f"History data from {source} has only {len(df):,} rows "
                f"— minimum required is {MIN_HISTORY_ROWS:,}."
            )
        print(f"  History rows from {source}: {len(df):,} ✓")
        return df

    # live API
    try:
        print(f"  Fetching player histories ({len(player_ids)} players)...")
        rows = []
        for i, pid in enumerate(player_ids):
            if i % 100 == 0:
                print(f"    {i}/{len(player_ids)} done...")
            try:
                r = requests.get(f"{FPL_BASE}/element-summary/{pid}/", timeout=10)
                if r.status_code != 200:
                    continue
                for gw in r.json().get("history", []):
                    gw["element"] = pid
                    rows.append(gw)
            except Exception:
                continue

        api_df = pd.DataFrame(rows)

        #check #rows
        if len(api_df) < MIN_HISTORY_ROWS:
            print(
                f"  WARNING: API returned only {len(api_df):,} history rows "
                f"(minimum: {MIN_HISTORY_ROWS:,})."
            )
            if use_cache_on_failure and os.path.exists(hist_path):
                cached = pd.read_csv(hist_path)
                if len(cached) >= MIN_HISTORY_ROWS:
                    print(
                        f"  Using CSV cache instead ({len(cached):,} rows ≥ "
                        f"{MIN_HISTORY_ROWS:,})."
                    )
                    return _validate_and_return(cached, "CSV cache")
            print(
                f"  Proceeding with {len(api_df):,} rows. "
            )
            return api_df

        return _validate_and_return(api_df, "API")

    #  API completely unreachable use csv instead
    except Exception as e:
        if use_cache_on_failure and os.path.exists(hist_path):
            print(f"  API unavailable ({e}). Loading histories from CSV cache...")
            cached = pd.read_csv(hist_path)
            if len(cached) < MIN_HISTORY_ROWS:
                print(
                    f"  WARNING: CSV cache only has {len(cached):,} rows "
                    f"(minimum: {MIN_HISTORY_ROWS:,}). "
                )
            else:
                print(f"  CSV cache: {len(cached):,} rows ✓")
            return cached
        raise


# Feature Engineering

def build_features(
    bootstrap: dict,
    history_df: pd.DataFrame,
    fixtures: list,
) -> pd.DataFrame:

    players  = pd.DataFrame(bootstrap["elements"])
    teams_df = pd.DataFrame(bootstrap["teams"])

    players["position"] = players["element_type"].map(POSITION_MAP)
    players["price"]    = players["now_cost"] / 10

    # cast all numeric columns
    num_cols = [
        "goals_scored", "assists", "clean_sheets", "minutes",
        "yellow_cards", "red_cards", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded",
        "form", "points_per_game", "selected_by_percent",
        "transfers_in", "transfers_out", "total_points",
        "value_season", "goals_conceded", "own_goals",
    ]
    for c in num_cols:
        if c in players.columns:
            players[c] = pd.to_numeric(players[c], errors="coerce")
    exist_num = [c for c in num_cols if c in players.columns]
    players[exist_num] = players[exist_num].fillna(0)

    # fillna with np.nan and then 0
    safe = players["minutes"].replace(0, np.nan)
    players["goals_per90"]     = (players["goals_scored"] / safe * 90).fillna(0)
    players["assists_per90"]   = (players["assists"]       / safe * 90).fillna(0)
    players["xG_per90"]        = (players["expected_goals"]   / safe * 90).fillna(0) \
                                  if "expected_goals" in players.columns else 0
    players["xA_per90"]        = (players["expected_assists"] / safe * 90).fillna(0) \
                                  if "expected_assists" in players.columns else 0
    players["bonus_per90"]     = (players["bonus"] / safe * 90).fillna(0)
    players["pts_per_million"] = (players["total_points"] / players["price"].replace(0, np.nan)).fillna(0)
    players["form"]            = pd.to_numeric(players["form"], errors="coerce").fillna(0)
    players["availability"]    = players["minutes"] / (players["minutes"].max() + 1)
    players["appearances"]     = (players["minutes"] / 90).apply(np.floor).astype(int)

    # next 5 fdr feature
    next_gws = [f for f in fixtures if not f.get("finished", True)][:50]
    team_fdr: dict = {}
    for f in next_gws:
        h, a = f["team_h"], f["team_a"]
        team_fdr.setdefault(h, []).append(f.get("team_h_difficulty", 3))
        team_fdr.setdefault(a, []).append(f.get("team_a_difficulty", 3))
    players["avg_fdr_next5"] = players["team"].apply(
        lambda tid: float(np.mean(team_fdr.get(tid, [3])[:5]))
    )
    players["fdr_score"] = 5 - players["avg_fdr_next5"]

    if not history_df.empty and len(history_df) >= MIN_HISTORY_ROWS:
        hcols = {
            "total_points": "h_avg_pts", "minutes": "h_avg_min",
            "goals_scored": "h_goals",   "assists": "h_assists", "bonus": "h_bonus",
        }
        for src in hcols:
            if src in history_df.columns:
                history_df[src] = pd.to_numeric(history_df[src], errors="coerce").fillna(0)
        agg_dict = {src: "mean" for src in hcols if src in history_df.columns}
        history_agg = history_df.groupby("element").agg(agg_dict).rename(columns=hcols)
        players = players.merge(history_agg, left_on="id", right_index=True, how="left")
        for dst in hcols.values():
            if dst in players.columns:
                players[dst] = players[dst].fillna(players[dst].median())
    else:
        for dst in ["h_avg_pts", "h_avg_min", "h_goals", "h_assists", "h_bonus"]:
            players[dst] = 0

    # composite score — no ML
    players["composite_score"] = (
        players["form"]            * 2.0 +
        players["pts_per_million"] * 1.5 +
        players["fdr_score"]       * 1.0 +
        players["xG_per90"]        * 3.0 +
        players["xA_per90"]        * 2.0 +
        players["bonus_per90"]     * 1.0 +
        players["availability"]    * 0.5
    )
    #filling names
    players["web_name"]  = players["web_name"].fillna(
        players["first_name"] + " " + players["second_name"]
    )
    players["team_name"] = players["team"].map(
        teams_df.set_index("id")["short_name"].to_dict()
    ).fillna("UNK")

    return players


# ML Pipelines

def build_pipelines() -> dict:

    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
            )),
        ]),
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
    }

#r2 score and mape
def train_and_evaluate(
    df: pd.DataFrame,
    pipelines: dict | None = None,
) -> tuple[dict, list]:

    if pipelines is None:
        pipelines = build_pipelines()

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].fillna(0)
    y = df["total_points"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    for name, pipeline in pipelines.items():
        print(f"  Training {name}...")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        mask = y_test > 0
        results[name] = {
            "pipeline": pipeline,
            "mape": mean_absolute_percentage_error(y_test[mask], pred[mask]),
            "r2":   r2_score(y_test, pred) * 100,
        }
        #vk
        print(f"    MAPE = {results[name]['mape']:.1f}%   R² = {results[name]['r2']:.1f}%")

    return results, feat_cols


# Joblib Save / Load

def save_artefacts(results: dict, feat_cols: list, player_df: pd.DataFrame) -> None:

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, info in results.items():
        filename = name.replace(" ", "_") + "_pipeline.joblib"
        path     = os.path.join(MODELS_DIR, filename)
        joblib.dump(info["pipeline"], path)
        print(f"  Saved  {path}")

    joblib.dump(feat_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    print(f"  Saved  {MODELS_DIR}/feature_cols.joblib")

    joblib.dump(player_df, os.path.join(MODELS_DIR, "player_data.joblib"))
    print(f"  Saved  {MODELS_DIR}/player_data.joblib")


def load_artefacts() -> tuple[dict, list, pd.DataFrame]:

    feat_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    player_df = joblib.load(os.path.join(MODELS_DIR, "player_data.joblib"))

    pipelines = {}
    for name in ["Random Forest", "Gradient Boosting", "Linear Regression"]:
        filename = name.replace(" ", "_") + "_pipeline.joblib"
        path     = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            pipelines[name] = joblib.load(path)

    return pipelines, feat_cols, player_df


def models_exist() -> bool:

    required = (
        ["feature_cols.joblib", "player_data.joblib"] +
        [n.replace(" ", "_") + "_pipeline.joblib"
         for n in ["Random Forest", "Gradient Boosting", "Linear Regression"]]
    )
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def score_players(
    df: pd.DataFrame, pipeline: Pipeline, feat_cols: list,) -> pd.DataFrame:

    df = df.copy()
    X  = df[feat_cols].fillna(0)
    df["ml_score"] = pipeline.predict(X)
    return df



def run_training(load_history: bool = False, force_cache: bool = False) -> None:

    print("\n=== FPL Pipeline — Training Run ===\n")

    if force_cache:
        # checking cache in data  folder
        print("Offline mode — loading from CSV cache...")
        bootstrap, fixtures, history_df, player_df = load_data_cache()
        print(f"  Feature matrix: {player_df.shape[0]} players × {player_df.shape[1]} features")

    else:
       # call api
        try:
            bootstrap  = load_bootstrap()
            fixtures   = load_fixtures()
            history_df = pd.DataFrame()

            if load_history:
                ids        = [p["id"] for p in bootstrap["elements"]]
                history_df = load_player_histories(ids)
                row_count  = len(history_df)
                status     = "✓" if row_count >= MIN_HISTORY_ROWS else "⚠ below minimum"
                print(f"  History rows loaded: {row_count:,} {status}")
                if row_count < MIN_HISTORY_ROWS:
                    print(
                        f"  History features will be zeroed out "
                        f"(need ≥ {MIN_HISTORY_ROWS:,} rows)."
                    )
            else:
                print("  Skipping history load (pass load_history=True to enable)")
# build the features from this
            print("\nBuilding feature matrix...")
            player_df = build_features(bootstrap, history_df, fixtures)
            print(f"  {player_df.shape[0]} players × {player_df.shape[1]} features")

# save them to csv
            print("\nSaving CSV cache...")
            save_data_cache(bootstrap, fixtures, history_df, player_df)

        except Exception as e:
            if cache_exists():
                print(f"\nAPI error: {e}")
                print("Falling back to CSV cache...")
                bootstrap, fixtures, history_df, player_df = load_data_cache()
                print(f"  Feature matrix: "
                      f"{player_df.shape[0]} players × {player_df.shape[1]} features")
            else:
                raise RuntimeError(
                    f"API unavailable and no CSV cache found.\n"
                    f"Original error: {e}\n"
                ) from e

    # each game feature
    df_train = player_df[player_df["minutes"] >= 90].copy()
    print(f"  {len(df_train)} players used for training (≥90 mins)")

    print("\nTraining pipelines...")
    results, feat_cols = train_and_evaluate(df_train)

    print("\nSaving artefacts...")
    save_artefacts(results, feat_cols, player_df)

    print("\n=== Done. All artefacts saved to ./models/ ===\n")

    # summary table
    print(f"{'Model':<22} {'MAPE':>8} {'R²':>8}")
    print("-" * 42)
    for name, info in results.items():
        print(f"{name:<22} {info['mape']:>7.1f}% {info['r2']:>7.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train FPL ML pipelines")
    parser.add_argument(
        "--history", action="store_true",
        help="Fetch full player history (~5 min)"
    )
    parser.add_argument(
        "--cache", action="store_true",
        help="Skip the API and train from saved CSV cache in ./data/"
    )
    args = parser.parse_args()
    run_training(load_history=args.history, force_cache=args.cache)
