#!/usr/bin/env python3
"""
SANCHAY (SIH26006) — Step 1 & Step 2:
1. Train & benchmark forecasting models (Without_CV vs With_CV) on chronological split (>= 2025-07-01),
   save metrics_comparison.csv, and persist best pipelines into models/.
2. Feed predicted 7-day freight rates and port delays into the MILP chartering optimizer
   (comparing Spot_Now vs. Wait_7d_Forward charters under draft, berth, and cargo constraints).
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CSV_PATH = "sanchay_ml_features_v3.csv"
XLSX_PATH = "Sanchay_Numeric_Dataset_v3.xlsx"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: Chronological Training, CV Ablation & Model Persistence
# ---------------------------------------------------------------------------
def build_pipeline(estimator, feature_cols, sample_df):
    cat_cols = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(sample_df[c])
        or pd.api.types.is_bool_dtype(sample_df[c])
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", preprocessor), ("model", estimator)])


def get_candidate_models():
    models = {
        "HistGBM": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.04, random_state=42
        )
    }
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.85, random_state=42, verbosity=0
        )
    except ImportError:
        pass
    return models


def run_step1_training():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "port_name"]).reset_index(drop=True)

    train_df = df[df["date"] <= "2025-05-31"].copy()
    test_df = df[df["date"] >= "2025-07-01"].copy()

    exclude_cols = {
        "record_id", "date", "charter_action", "recommended_alt_vessel_class",
        "true_vessel_count", "true_congestion_index",
        "freight_rate_next_7d_usd", "freight_rate_next_30d_usd", "freight_rate_next_90d_usd",
        "benchmark_next_7d_usd_mt", "benchmark_next_30d_usd_mt", "benchmark_next_90d_usd_mt",
        "target_logret_7d", "target_logret_30d", "target_logret_90d",
        "expected_delay_days", "expected_demurrage_cost_usd",
        "alt_freight_rate_usd", "potential_savings_usd", "total_freight_cost_inr",
    }

    cv_cols = {
        "cv_obs_available", "cv_modality", "cv_ship_count", "cv_mean_conf",
        "days_since_cv_obs", "anchorage_vessel_count", "port_congestion_index",
        "anchorage_7d_rolling_mean", "cv_source", "imaging_source", "satellite_cloud_cover_pct",
    } | {c for c in df.columns if c.startswith("anchorage_vessel_count_")}

    all_features = [c for c in df.columns if c not in exclude_cols]
    no_cv_features = [c for c in all_features if c not in cv_cols]

    metrics_rows = []
    saved_bundles = {}

    for target_col in ["target_logret_7d", "expected_delay_days"]:
        for feat_set_name, feat_cols in [("Without_CV", no_cv_features), ("With_CV", all_features)]:
            tr = train_df.dropna(subset=[target_col])
            te = test_df.dropna(subset=[target_col])

            cat_cols = [
                c for c in feat_cols
                if not pd.api.types.is_numeric_dtype(tr[c])
                or pd.api.types.is_bool_dtype(tr[c])
            ]
            num_cols = [c for c in feat_cols if c not in cat_cols]

            X_tr = tr[feat_cols].copy()
            X_te = te[feat_cols].copy()
            y_tr = tr[target_col]
            y_te = te[target_col]

            for c in cat_cols:
                X_tr[c] = X_tr[c].astype(str)
                X_te[c] = X_te[c].astype(str)
            for c in num_cols:
                X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").astype(np.float64)
                X_te[c] = pd.to_numeric(X_te[c], errors="coerce").astype(np.float64)

            best_mae = np.inf
            best_entry = None

            for m_name, estimator in get_candidate_models().items():
                pipe = build_pipeline(estimator, feat_cols, X_tr)
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_te)

                mae = mean_absolute_error(y_te, preds)
                rmse = root_mean_squared_error(y_te, preds)
                r2 = r2_score(y_te, preds)

                bench_mae = np.nan
                if target_col == "target_logret_7d":
                    pred_px = te["freight_benchmark_usd_mt"].values * np.exp(preds)
                    true_px = te["benchmark_next_7d_usd_mt"].values
                    bench_mae = mean_absolute_error(true_px, pred_px)

                metrics_rows.append({
                    "target": target_col,
                    "feature_set": feat_set_name,
                    "model": m_name,
                    "mae": round(float(mae), 5),
                    "rmse": round(float(rmse), 5),
                    "r2": round(float(r2), 4),
                    "bench_7d_price_mae_usd_mt": round(float(bench_mae), 4) if not np.isnan(bench_mae) else None,
                })

                if mae < best_mae:
                    best_mae = mae
                    best_entry = (m_name, pipe)

            if feat_set_name == "With_CV":
                bundle = {
                    "target": target_col,
                    "model_name": best_entry[0],
                    "pipeline": best_entry[1],
                    "feature_cols": feat_cols,
                    "val_mae": float(best_mae),
                }
                out_path = os.path.join(MODELS_DIR, f"{target_col}_best.joblib")
                joblib.dump(bundle, out_path)
                saved_bundles[target_col] = bundle

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("metrics_comparison.csv", index=False)
    print("=== STEP 1: Chronological Test Metrics (>= 2025-07-01) ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics_comparison.csv and models in ./{MODELS_DIR}/")
    return df, saved_bundles


# ---------------------------------------------------------------------------
# STEP 2: Forecast-Driven MILP Chartering Optimization
# ---------------------------------------------------------------------------
def forecast_ports(df, bundles, as_of_date=None):
    sub = df[df["date"] <= pd.to_datetime(as_of_date)].copy() if as_of_date is not None else df.copy()
    latest = sub.sort_values("date").groupby("port_name", as_index=False).last()

    rate_art = bundles["target_logret_7d"]
    delay_art = bundles["expected_delay_days"]

    latest["pred_logret_7d"] = rate_art["pipeline"].predict(latest[rate_art["feature_cols"]])
    latest["pred_benchmark_7d_usd_mt"] = (
        latest["freight_benchmark_usd_mt"] * np.exp(latest["pred_logret_7d"])
    )
    latest["pred_delay_days"] = np.clip(
        delay_art["pipeline"].predict(latest[delay_art["feature_cols"]]), 0.0, None
    )
    return latest


def solve_chartering_milp(
    forecasts_df,
    demand_by_port_mt,
    allow_spot=True,
    allow_fwd_7d=True,
    max_trips_per_option=15,
    wait_congestion_relief_days=0.5,
):
    milp_params = pd.read_excel(XLSX_PATH, sheet_name="MILP_Port_Vessel_Params")
    feas = milp_params[
        (milp_params["class_allowed_at_port"] == 1) & (milp_params["draft_feasible"] == 1)
    ].copy().reset_index(drop=True)

    fc = forecasts_df.set_index("port_name")
    feas["curr_rate_usd_mt"] = feas["port_name"].map(fc["freight_benchmark_usd_mt"]).fillna(15.0)
    feas["pred_rate_7d_usd_mt"] = feas["port_name"].map(fc["pred_benchmark_7d_usd_mt"]).fillna(15.0)
    feas["pred_delay_days"] = feas["port_name"].map(fc["pred_delay_days"]).fillna(1.0)

    demurrage = feas["demurrage_usd_day_median_last90d"].fillna(
        feas["demurrage_usd_day_median_all"]
    ).values
    cargo = feas["typical_cargo_mt"].values
    port_cost = feas["port_cost_usd"].values
    delay_now = feas["pred_delay_days"].values
    delay_7d = np.maximum(delay_now - wait_congestion_relief_days, 0.0)

    cost_spot = port_cost + demurrage * delay_now + feas["curr_rate_usd_mt"].values * cargo
    cost_fwd7 = port_cost + demurrage * delay_7d + feas["pred_rate_7d_usd_mt"].values * cargo

    n = len(feas)
    c_obj = np.concatenate([cost_spot, cost_fwd7])

    ub_spot = np.full(n, float(max_trips_per_option) if allow_spot else 0.0)
    ub_fwd7 = np.full(n, float(max_trips_per_option) if allow_fwd_7d else 0.0)
    ub = np.concatenate([ub_spot, ub_fwd7])
    lb = np.zeros(2 * n)

    A_rows, lb_rows, ub_rows = [], [], []
    active_ports = [p for p, d in demand_by_port_mt.items() if d > 0 and p in set(feas["port_name"])]

    for port in active_ports:
        mask = (feas["port_name"] == port).values.astype(float)
        A_rows.append(np.concatenate([mask * cargo, mask * cargo]))
        lb_rows.append(float(demand_by_port_mt[port]))
        ub_rows.append(np.inf)

        berth_cap = float(feas.loc[feas["port_name"] == port, "berth_capacity_vessels"].iloc[0])
        A_rows.append(np.concatenate([mask, mask]))
        lb_rows.append(0.0)
        ub_rows.append(berth_cap)

    res = milp(
        c=c_obj,
        integrality=np.ones(2 * n),
        bounds=Bounds(lb=lb, ub=ub),
        constraints=LinearConstraint(np.array(A_rows), lb=np.array(lb_rows), ub=np.array(ub_rows)),
    )

    if res.x is None:
        return pd.DataFrame(), {"status": res.message, "total_cost_usd": None}

    x = np.round(res.x).astype(int)
    feas["spot_now_trips"] = x[:n]
    feas["fwd_7d_trips"] = x[n:]
    feas["total_trips"] = feas["spot_now_trips"] + feas["fwd_7d_trips"]
    feas["allocated_cargo_mt"] = feas["total_trips"] * feas["typical_cargo_mt"]
    feas["expected_cost_usd"] = (
        feas["spot_now_trips"] * cost_spot + feas["fwd_7d_trips"] * cost_fwd7
    ).round(2)

    plan_df = feas[feas["total_trips"] > 0][
        [
            "port_name", "vessel_class", "spot_now_trips", "fwd_7d_trips",
            "typical_cargo_mt", "allocated_cargo_mt", "port_max_draft_m", "vessel_draft_m",
            "pred_delay_days", "curr_rate_usd_mt", "pred_rate_7d_usd_mt", "expected_cost_usd",
        ]
    ].reset_index(drop=True)

    return plan_df, {"status": res.message, "total_cost_usd": round(float(res.fun), 2)}


if __name__ == "__main__":
    df_all, bundles = run_step1_training()

    latest_forecasts = forecast_ports(df_all, bundles)
    print("\n=== STEP 2A: Latest Port Forecasts ===")
    print(
        latest_forecasts[
            ["port_name", "date", "anchorage_vessel_count", "freight_benchmark_usd_mt", "pred_benchmark_7d_usd_mt", "pred_delay_days"]
        ].to_string(index=False)
    )

    sample_demand = {
        "Paradip": 120000.0,
        "Visakhapatnam": 90000.0,
        "Haldia": 30000.0,
        "Mundra": 140000.0,
    }
    plan, summary = solve_chartering_milp(latest_forecasts, sample_demand)
    print(f"\n=== STEP 2B: Optimal MILP Chartering Plan ({summary['status']}) ===")
    print(f"Total Expected Cost: ${summary['total_cost_usd']:,.2f}")
    print(plan.to_string(index=False))