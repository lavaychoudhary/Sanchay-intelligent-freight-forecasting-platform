#!/usr/bin/env python3
"""
SANCHAY (SIH26006) - v2 -> v3 numeric dataset: CV alignment, leakage-safe features, MILP tables.

Run:   python sanchay_v3_pipeline.py --v2 Sanchay_Numeric_Dataset_v2_updated.xlsx --results results_1.csv
Needs: pandas numpy openpyxl

What it does (see Change_Log sheet in the output for the reasons):
  1. Reads your YOLO results.csv and uses its precision/recall to build a CV observation model
     (the numeric model must be trained on what the CV model *sees*, not on ground truth).
  2. Splits ground truth (true_vessel_count) from the CV-observed channel (cv_ship_count, revisit gaps,
     cloud loss) and rebuilds anchorage_vessel_count / port_congestion_index from the CV channel.
  3. Adds a fixed-composition freight benchmark (a real time series) + 7/30/90-day targets.
  4. Fixes freight_next_30/90d, alternative-vessel logic (cargo + draft feasibility), rolling mean, USD/INR.
  5. Writes MILP port x vessel-class parameters and a feature table with lags/rollings (past-only).
Hook:  inject_cv_observations(df, "cv_observations.csv") replaces simulated CV rows with real best.pt output.
"""
import argparse, numpy as np, pandas as pd

SEED = 42
GE_REF_CONF = 0.615320            # your real GE_1.jpg test: mean confidence (single image -> anchor only)
REVISIT_DAYS = 5                  # one usable scene per port every ~5 days (set to your imagery source)
CLOUD_LIMIT = 60                  # optical scene unusable above this cloud %
CLOUD_RECALL_LOSS = 0.5           # recall multiplier = 1 - 0.5*cloud/100  (assumption, tune on real scenes)
SAR_FALLBACK = False              # True = cloudy days still get a (weaker) SAR-based observation; needs a SAR-trained model
SAR_RECALL_FACTOR = 0.85          # assumed recall of the SAR model relative to the optical one

CLASSES = ["Handysize", "Supramax", "Panamax", "Capesize"]
TCE_A = np.array([5500, 500, 800, -1500.]); TCE_B = np.array([4.0, 6.4, 6.2, 15.5])
SPEED = np.array([12.0, 12.5, 12.5, 12.0]); CONS = np.array([23, 27, 31, 50.])
LOAD_RATE = np.array([11000, 16000, 22000, 35000.]); DIS_FAC = np.array([.8, 1.0, 1.15, 1.4])
PCOST = np.array([38000, 52000, 68000, 115000.])
CAP = np.array([36000, 60000, 78000, 150000.])             # parcel capacity per class
D_LIGHT = np.array([4.0, 4.8, 5.5, 7.0]); D_LADEN = np.array([9.6, 12.8, 14.0, 18.2])   # approx drafts (m)
DWT_REF = np.array([36000, 60000, 78000, 178000.])
PORTS = {  # discharge rate t/day, cape allowed, draft limit general / capesize (m), source of draft
 "Paradip":       (22000, 1, 16.0, 16.0, "sourced: Capesize handled with ~16 m restriction (entrance channel 17.1 m)"),
 "Visakhapatnam": (18000, 1, 14.5, 18.1, "sourced: inner harbour 14.5 m, outer harbour 18.1 m (Vizag Port Authority)"),
 "Haldia":        (12000, 0,  9.0,  9.0, "sourced: ~9 m max draft; larger ships part-load / lighter"),
 "Kandla":        (15000, 0, 14.0, 14.0, "assumed - verify with port authority"),
 "Mundra":        (30000, 1, 17.0, 17.0, "assumed - verify with port authority"),
 "Krishnapatnam": (20000, 1, 18.0, 18.0, "assumed - verify with port authority"),
 "Mormugao":      (14000, 0, 14.5, 14.5, "assumed - verify with port authority"),
 "New Mangalore": (13000, 0, 14.0, 14.0, "assumed - verify with port authority")}
REF_DIST = {"Paradip": 5750, "Visakhapatnam": 5450, "Haldia": 5900, "Kandla": 5000, "Mundra": 4950,
            "Krishnapatnam": 5300, "Mormugao": 4750, "New Mangalore": 4650}   # Richards Bay -> port (nm)

# USD/INR anchors: 2026 points from published rupee closes (Nov 2026: 93.7-94.5; record low ~96 in May); older points approximate
INR_ANCHORS = [("2022-12-01",82.4),("2023-01-01",82.7),("2023-06-30",82.0),("2023-10-15",83.2),("2023-12-29",83.2),
 ("2024-06-30",83.5),("2024-10-31",84.1),("2024-12-31",85.6),("2025-02-15",86.9),("2025-04-30",85.1),("2025-06-30",85.7),
 ("2025-08-29",88.1),("2025-09-30",88.8),("2025-11-21",89.7),("2025-12-31",90.0),("2026-02-27",91.5),("2026-04-20",92.8),
 ("2026-04-22",93.8),("2026-05-14",95.86),("2026-06-04",95.3),("2026-06-15",94.26),("2026-07-13",95.9),("2026-09-15",95.8),
 ("2026-10-12",95.5),("2026-11-05",94.47),("2026-11-06",93.91),("2026-11-09",93.93),("2026-11-11",93.69),("2026-11-12",94.04),
 ("2026-11-16",93.92),("2026-11-18",93.9)]

def det_cost(c, bdi, vl, ifo, dist, cargo, rate, delay):
    """Deterministic voyage cost per tonne (no noise, port-cost factor 1.0) - same economics used to build the data."""
    c = np.asarray(c)
    tce = np.maximum(TCE_A[c] + TCE_B[c] * bdi, 5000)
    sea = dist / (SPEED[c] * 24) * 1.45
    port_d = np.maximum(2.0, cargo / LOAD_RATE[c]) + np.clip(cargo / (rate * DIS_FAC[c]), 1.5, 10) + delay
    price = 0.75 * vl + 0.25 * ifo
    return (tce * (sea + port_d) + (sea * CONS[c] + port_d * 5.0) * price + PCOST[c]) / cargo * 1.025

def vessel_draft(c, cargo):
    c = np.asarray(c); return D_LIGHT[c] + np.clip(cargo / DWT_REF[c], 0, 1) * (D_LADEN[c] - D_LIGHT[c])

def port_limit(port, c):
    gen = np.array([PORTS[p][2] for p in port]); cape = np.array([PORTS[p][3] for p in port])
    return np.where(np.asarray(c) == 3, cape, gen)

def cv_operating_point(results_csv):
    r = pd.read_csv(results_csv); r.columns = [c.strip() for c in r.columns]
    b = r.loc[r["metrics/mAP50(B)"].idxmax()]
    return dict(precision=float(b["metrics/precision(B)"]), recall=float(b["metrics/recall(B)"]),
                map50=float(b["metrics/mAP50(B)"]), map5095=float(b["metrics/mAP50-95(B)"]), epoch=int(b["epoch"]), epochs=len(r))

def build_inr(dates, rng):
    d0 = pd.Timestamp(INR_ANCHORS[0][0]); full = pd.date_range(d0, dates.max())
    xs = np.array([(pd.Timestamp(d) - d0).days for d, _ in INR_ANCHORS]); ys = np.array([v for _, v in INR_ANCHORS])
    base = np.interp(np.arange(len(full)), xs, ys)
    e = np.zeros(len(full))
    for t in range(1, len(full)): e[t] = 0.9 * e[t - 1] + rng.normal(0, 0.0012 * np.sqrt(1 - 0.81))
    e -= np.interp(np.arange(len(full)), xs, e[xs]); s = pd.Series(np.round(base * np.exp(e), 2), index=full)
    wk = s.index.weekday >= 5; s[wk] = np.nan; s = s.ffill()
    return s

def align(v2_path, results_csv, rng, sar_fallback=SAR_FALLBACK):
    d = pd.read_excel(v2_path, sheet_name="Numeric_Dataset"); d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["port_name", "date"]).reset_index(drop=True)
    cvp = cv_operating_point(results_csv); P, R = cvp["precision"], cvp["recall"]
    out = d.copy()
    # ---- 1. ground truth vs CV channel -------------------------------------------------
    out["true_vessel_count"] = d["anchorage_vessel_count"]; out["true_congestion_index"] = d["port_congestion_index"]
    n = len(out); obs = np.zeros(n, bool)
    for p, idx in out.groupby("port_name").indices.items():
        idx = np.sort(idx); off = rng.integers(0, REVISIT_DAYS); k = off
        while k < len(idx):
            day = min(len(idx) - 1, max(0, k + (rng.integers(-1, 2) if rng.random() < 0.3 else 0)))
            obs[idx[day]] = True; k += REVISIT_DAYS
        obs[idx[0]] = True                                           # guarantee a first observation
    cloud = out["satellite_cloud_cover_pct"].values
    first = out.groupby("port_name").cumcount().values == 0
    if not sar_fallback: obs &= (cloud <= CLOUD_LIMIT) | first                 # cloudy optical scenes are lost
    sar = obs & (cloud > CLOUD_LIMIT)
    rec = np.where(sar, R * SAR_RECALL_FACTOR, np.clip(R * (1 - CLOUD_RECALL_LOSS * cloud / 100), 0.05, 1))
    tp = rng.binomial(out["true_vessel_count"].values, rec); fp = rng.poisson(tp * (1 / P - 1))
    cv_count = np.where(obs, tp + fp, np.nan)
    out["cv_obs_available"] = obs.astype(int); out["cv_ship_count"] = cv_count
    out["cv_modality"] = np.where(~obs, "none", np.where(sar, "SAR", "optical"))
    out["cv_mean_conf"] = np.where(obs, np.clip(GE_REF_CONF + rng.normal(0, 0.04, n) - 0.001 * cloud, 0.35, 0.9), np.nan).round(3)
    g = out.groupby("port_name")
    out["anchorage_vessel_count"] = g["cv_ship_count"].ffill().astype(int)          # what the model/dashboard can know at t
    out["cv_mean_conf"] = g["cv_mean_conf"].ffill()
    last_obs = out["date"].where(out["cv_obs_available"] == 1); out["_lo"] = last_obs
    out["days_since_cv_obs"] = (out["date"] - out.groupby("port_name")["_lo"].ffill()).dt.days; out.drop(columns="_lo", inplace=True)
    out["port_congestion_index"] = (out["anchorage_vessel_count"] / out["port_capacity_vessels"]).clip(upper=1.0).round(3)
    out["anchorage_7d_rolling_mean"] = out.groupby("port_name")["anchorage_vessel_count"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()).round(2)
    out = out.drop(columns=["cv_detection_confidence", "days_since_clear_optical"])
    # ---- 2. market lookups by date --------------------------------------------------------
    daily = out.groupby("date")[["bdi_index", "vlsfo_bunker_price", "ifo380_bunker_price"]].first()
    def fut(h, col): return daily[col].shift(-h).reindex(out["date"]).values
    cidx = out["charter_vessel_class"].map({c: i for i, c in enumerate(CLASSES)}).values
    rate = out["port_name"].map(lambda p: PORTS[p][0]).values
    cargo = out["cargo_weight_mt"].values.astype(float); dist = out["route_distance_nm"].values.astype(float); delay = out["expected_delay_days"].values
    now = det_cost(cidx, out["bdi_index"].values, out["vlsfo_bunker_price"].values, out["ifo380_bunker_price"].values, dist, cargo, rate, delay)
    # ---- 3. same-voyage forward freight (7/30/90d), NaN past the data end ---------------------
    for h in (7, 30, 90):
        c_h = det_cost(cidx, fut(h, "bdi_index"), fut(h, "vlsfo_bunker_price"), fut(h, "ifo380_bunker_price"), dist, cargo, rate, delay)
        out[f"freight_rate_next_{h}d_usd"] = (out["freight_rate_usd"] * c_h / now).round(2)
    # ---- 4. fixed-composition benchmark: Panamax 70k MT, Richards Bay -> port -----------------
    ref_d = out["port_name"].map(REF_DIST).values.astype(float)
    out["freight_benchmark_usd_mt"] = det_cost(np.full(n, 2), out["bdi_index"].values, out["vlsfo_bunker_price"].values, out["ifo380_bunker_price"].values, ref_d, 70000., rate, delay).round(3)
    g = out.groupby("port_name")["freight_benchmark_usd_mt"]
    for h in (7, 30, 90): out[f"benchmark_next_{h}d_usd_mt"] = g.shift(-h).round(3)
    # ---- 5. draft feasibility + feasible alternatives -----------------------------------------
    out["vessel_draft_m"] = vessel_draft(cidx, cargo).round(2)
    out["port_max_draft_m"] = port_limit(out["port_name"].values, cidx)
    out["draft_feasible"] = (out["vessel_draft_m"] <= out["port_max_draft_m"]).astype(int)
    best_sav = np.zeros(n); best_cls = np.full(n, -1)
    for c in range(4):
        cc = np.full(n, c)
        alt = det_cost(cc, out["bdi_index"].values, out["vlsfo_bunker_price"].values, out["ifo380_bunker_price"].values, dist, cargo, rate, delay)
        ok = (cc != cidx) & (cargo <= CAP[c]) & (cargo >= 0.6 * CAP[c]) & (vessel_draft(cc, cargo) <= port_limit(out["port_name"].values, cc))
        ok &= np.array([PORTS[p][1] == 1 for p in out["port_name"]]) | (c != 3)
        sav = np.where(ok, (now - alt) / now, 0); upd = sav > best_sav; best_sav = np.where(upd, sav, best_sav); best_cls = np.where(upd, c, best_cls)
    use = best_sav >= 0.02
    out["recommended_alt_vessel_class"] = np.where(use, np.array(CLASSES)[np.clip(best_cls, 0, 3)], "None (Optimal)")
    out["alt_freight_rate_usd"] = np.where(use, out["freight_rate_usd"] * (1 - best_sav), out["freight_rate_usd"]).round(2)
    out["potential_savings_usd"] = ((out["freight_rate_usd"] - out["alt_freight_rate_usd"]) * cargo).round(0)
    d7 = (out["freight_rate_next_7d_usd"] - out["freight_rate_usd"]) / out["freight_rate_usd"] + rng.normal(0, 0.01, n)
    out["charter_action"] = np.where(use, "Switch_Vessel_Class", np.where((d7 <= -0.01) & (delay <= 5.0), "Wait_3_Days", "Book_Now"))
    # ---- 6. INR + costs ---------------------------------------------------------------------
    inr = build_inr(out["date"], rng); out["usd_inr_rate"] = out["date"].map(inr).values
    out["total_freight_cost_inr"] = (out["freight_rate_usd"] * cargo * out["usd_inr_rate"]).round(0)
    return out, cvp

def inject_cv_observations(df, csv_path):
    """Replace simulated CV rows with REAL best.pt results. CSV columns: date, port_name, cv_ship_count, cv_mean_conf
    (and optionally cv_total_box_area, cv_mean_box_area). Re-derives the carried-forward features afterwards."""
    obs = pd.read_csv(csv_path, parse_dates=["date"]); df = df.merge(obs, on=["date", "port_name"], how="left", suffixes=("", "_real"))
    m = df["cv_ship_count_real"].notna()
    df.loc[m, "cv_ship_count"] = df.loc[m, "cv_ship_count_real"]; df.loc[m, "cv_mean_conf"] = df.loc[m, "cv_mean_conf_real"]
    df.loc[m, "cv_obs_available"] = 1; df["cv_source"] = np.where(m, "live_best_pt", df.get("cv_source", "simulated_from_yolo_metrics"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_real")]).sort_values(["port_name", "date"]).reset_index(drop=True)
    g = df.groupby("port_name")
    df["anchorage_vessel_count"] = g["cv_ship_count"].ffill().fillna(df["anchorage_vessel_count"]).astype(int)
    df["cv_mean_conf"] = g["cv_mean_conf"].ffill()
    df["_lo"] = df["date"].where(df["cv_obs_available"] == 1)
    df["days_since_cv_obs"] = (df["date"] - df.groupby("port_name")["_lo"].ffill()).dt.days; df = df.drop(columns="_lo")
    df["port_congestion_index"] = (df["anchorage_vessel_count"] / df["port_capacity_vessels"]).clip(upper=1.0).round(3)
    df["anchorage_7d_rolling_mean"] = df.groupby("port_name")["anchorage_vessel_count"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()).round(2)
    return df

def milp_table(df):
    rows = []
    for p, (rate, cape, gen, capl, src) in PORTS.items():
        sub = df[df.port_name == p]
        for i, c in enumerate(CLASSES):
            cs = sub[sub.charter_vessel_class == c]; lim = capl if i == 3 else gen
            typ = float(cs.cargo_weight_mt.median()) if len(cs) else float((CAP[i] + 0.45 * CAP[i]) / 2)
            dr = float(vessel_draft(np.array([i]), np.array([typ]))[0])
            rows.append(dict(port_name=p, vessel_class=c, class_allowed_at_port=int(cape == 1 or i < 3), port_max_draft_m=lim,
                             vessel_draft_m=round(dr, 2), draft_feasible=int(dr <= lim and (cape == 1 or i < 3)),
                             typical_cargo_mt=int(typ), berth_capacity_vessels=int(sub.port_capacity_vessels.iloc[0]),
                             discharge_rate_t_day=rate, port_cost_usd=int(PCOST[i]),
                             demurrage_usd_day_median_all=int(cs.demurrage_rate_usd_day.median()) if len(cs) else None,
                             demurrage_usd_day_median_last90d=int(cs[cs.date >= cs.date.max() - pd.Timedelta(days=90)].demurrage_rate_usd_day.median()) if len(cs) else None,
                             draft_source=src))
    m = pd.DataFrame(rows)
    for col in ('demurrage_usd_day_median_all', 'demurrage_usd_day_median_last90d'):
        m[col] = m[col].fillna(m.groupby('vessel_class')[col].transform('median'))
    return m

def ml_features(df):
    """Past-only features per port. Targets: benchmark_next_{7,30,90}d, expected_delay_days, charter_action."""
    f = df.sort_values(["port_name", "date"]).reset_index(drop=True).copy()
    f["month_sin"] = np.sin(2 * np.pi * f.date.dt.month / 12); f["month_cos"] = np.cos(2 * np.pi * f.date.dt.month / 12)
    f["dow"] = f.date.dt.dayofweek; f["is_monsoon_season"] = f.date.dt.month.between(6, 9).astype(int)
    g = f.groupby("port_name")
    for col in ["freight_benchmark_usd_mt", "anchorage_vessel_count", "bdi_index", "vlsfo_bunker_price"]:
        for lag in (1, 3, 7, 14): f[f"{col}_lag{lag}"] = g[col].shift(lag)
        f[f"{col}_roll7_mean"] = g[col].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
        f[f"{col}_roll7_std"] = g[col].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
        f[f"{col}_chg7"] = (f[col] / g[col].shift(7) - 1).replace([np.inf, -np.inf], np.nan)
    for h in (7, 30, 90): f[f"target_logret_{h}d"] = np.log(f[f"benchmark_next_{h}d_usd_mt"] / f["freight_benchmark_usd_mt"])
    return f.dropna(subset=["freight_benchmark_usd_mt_lag14", "freight_benchmark_usd_mt_roll7_std"]).reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--v2", default="Sanchay_Numeric_Dataset_v2_updated.xlsx")
    ap.add_argument("--results", default="results_1.csv"); ap.add_argument("--cv-obs", default=None, help="CSV of real best.pt observations")
    ap.add_argument("--out", default="Sanchay_Numeric_Dataset_v3.xlsx"); ap.add_argument("--sar-fallback", action="store_true"); a = ap.parse_args()
    rng = np.random.default_rng(SEED); df, cvp = align(a.v2, a.results, rng, a.sar_fallback)
    df["cv_source"] = "simulated_from_yolo_metrics"
    if a.cv_obs: df = inject_cv_observations(df, a.cv_obs)
    feats = ml_features(df); feats.to_csv("sanchay_ml_features_v3.csv", index=False)
    print("CV operating point:", cvp); print("rows", len(df), "| ML feature rows", len(feats), "cols", feats.shape[1])
    milp = milp_table(df)
    with pd.ExcelWriter(a.out, engine="openpyxl") as w:
        df.assign(date=df.date.dt.strftime("%Y-%m-%d")).sort_values(["date", "port_name"]).to_excel(w, sheet_name="Numeric_Dataset", index=False)
        milp.to_excel(w, sheet_name="MILP_Port_Vessel_Params", index=False)
    print("saved", a.out, "and sanchay_ml_features_v3.csv")
