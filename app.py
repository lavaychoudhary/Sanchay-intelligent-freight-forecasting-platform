"""SANCHAY dashboard: port congestion, 7-day freight forecasts, MILP chartering and ship detection."""
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from train_and_optimize import forecast_ports, run_step1_training, solve_chartering_milp

st.set_page_config(page_title="SANCHAY | Freight & Chartering", page_icon="⚓", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens: harbour navy, sea teal, buoy amber, signal coral, sea-foam paper
# ---------------------------------------------------------------------------
NAVY, NAVY2, TEAL, AMBER, CORAL = "#0B2540", "#164A75", "#0E9AA7", "#F2A541", "#E4572E"
FOAM, INK, MUTED, LINE = "#F2F6F9", "#10253A", "#5F7488", "#DCE6ED"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=Manrope:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family:'Manrope',sans-serif; color:%INK%; }
.stApp { background:%FOAM%; }
h1,h2,h3,h4 { font-family:'Bricolage Grotesque',sans-serif !important; color:%NAVY%; letter-spacing:-.01em; }
.block-container { padding-top:1.4rem; max-width:1360px; }
header[data-testid="stHeader"] { background:transparent; }

.hero { background:radial-gradient(900px 260px at 90% -30%, rgba(14,154,167,.6), transparent 60%),
        linear-gradient(120deg,%NAVY%,%NAVY2%); border-radius:18px; padding:26px 32px; color:#fff;
        position:relative; overflow:hidden; margin-bottom:18px; }
.hero:after { content:""; position:absolute; left:0; right:0; bottom:0; height:4px;
        background:linear-gradient(90deg,%TEAL%,%AMBER%); }
.hero h1 { color:#fff !important; font-size:2rem; font-weight:800; margin:0 0 4px; }
.hero p { color:#C9D8E6; margin:0; max-width:780px; line-height:1.5; }
.chip { display:inline-block; padding:3px 11px; border-radius:99px; font-size:.76rem; font-weight:600;
        margin:12px 8px 0 0; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2); color:#E8F1F8; }
.chip.ok { background:rgba(14,154,167,.32); border-color:rgba(14,154,167,.8); }
.chip.warn { background:rgba(242,165,65,.28); border-color:rgba(242,165,65,.9); }

.kpi { background:#fff; border:1px solid %LINE%; border-left:5px solid %TEAL%; border-radius:14px;
       padding:15px 18px; box-shadow:0 5px 16px rgba(11,37,64,.05); height:100%; }
.kpi.amber { border-left-color:%AMBER%; } .kpi.coral { border-left-color:%CORAL%; } .kpi.navy { border-left-color:%NAVY%; }
.kpi .l { color:%MUTED%; font-size:.8rem; font-weight:600; }
.kpi .v { font-family:'Bricolage Grotesque',sans-serif; font-size:1.65rem; font-weight:800; color:%NAVY%; line-height:1.25; }
.kpi .s { color:%MUTED%; font-size:.78rem; }

.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid %LINE%; }
.stTabs [data-baseweb="tab"] { height:44px; padding:0 18px; font-weight:600; color:%MUTED%; border-radius:10px 10px 0 0; }
.stTabs [aria-selected="true"] { color:%NAVY% !important; background:#fff; }

/* Main canvas controls & safe file uploader styling */
.stApp label, .stApp [data-testid="stWidgetLabel"] p { color:%INK% !important; font-weight:600; }
.stApp [data-baseweb="select"] > div {
    background-color:#FFFFFF !important;
    color:%INK% !important;
    border:1px solid %LINE% !important;
    border-radius:8px;
}
.stApp [data-baseweb="select"] span,
.stApp [data-baseweb="select"] div,
.stApp [data-baseweb="select"] svg {
    color:%INK% !important;
    fill:%INK% !important;
}
.stApp [data-testid="stSelectSlider"] div,
.stApp [data-testid="stSelectSlider"] span {
    color:%INK% !important;
}
.stApp [data-testid="stFileUploader"] {
    background-color:#FFFFFF !important;
    border:1px solid %LINE% !important;
    border-radius:14px !important;
    padding:14px !important;
}
.stApp [data-testid="stFileUploader"] label,
.stApp [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
    color:%INK% !important;
    font-weight:700 !important;
}
.stApp [data-testid="stFileUploader"] button {
    background-color:%NAVY% !important;
    color:#FFFFFF !important;
    border:0 !important;
    border-radius:8px !important;
    font-weight:600 !important;
}
.stApp [data-testid="stFileUploader"] button:hover {
    background-color:%TEAL% !important;
    color:#FFFFFF !important;
}

/* Sidebar styling & high-contrast inputs */
section[data-testid="stSidebar"] { background:%NAVY%; }
section[data-testid="stSidebar"] > div:first-child { color:#E6EEF5; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption { color:#E6EEF5 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color:#fff !important; }

section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color:#FFFFFF !important;
    color:%INK% !important;
    border-radius:8px;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color:%INK% !important;
    fill:%INK% !important;
}
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] span,
div[data-baseweb="popover"] div {
    background-color:#FFFFFF !important;
    color:%INK% !important;
}

section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background-color:#FFFFFF !important;
    color:%INK% !important;
    font-weight:600;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background-color:#E6EEF5 !important;
    color:%NAVY% !important;
}

.banner { border-radius:12px; padding:12px 16px; font-weight:600; color:%NAVY%; margin-bottom:12px; }
.banner.ok { background:#D3EFF1; border-left:5px solid %TEAL%; }
.banner.bad { background:#FCE3DA; border-left:5px solid %CORAL%; }
.banner.warn { background:#FDEFD6; border-left:5px solid %AMBER%; }
.note { background:#fff; border:1px solid %LINE%; border-radius:12px; padding:14px 18px; color:%INK%; font-size:.92rem; line-height:1.55; }
div[data-testid="stDataFrame"] { border:1px solid %LINE%; border-radius:12px; overflow:hidden; }
.footer { text-align:center; color:%MUTED%; font-size:.8rem; margin-top:30px; }
</style>
"""
for k, v in dict(NAVY=NAVY, NAVY2=NAVY2, TEAL=TEAL, AMBER=AMBER, CORAL=CORAL, FOAM=FOAM, INK=INK, MUTED=MUTED, LINE=LINE).items():
    CSS = CSS.replace(f"%{k}%", v)
st.markdown(CSS, unsafe_allow_html=True)

PORT_COORDS = {
    "Paradip": (20.26, 86.67), "Visakhapatnam": (17.69, 83.29), "Haldia": (22.03, 88.10),
    "Mundra": (22.74, 69.70), "Kandla": (23.03, 70.22), "Krishnapatnam": (14.25, 80.12),
    "Mormugao": (15.41, 73.80), "New Mangalore": (12.92, 74.81),
}
RATE_T, DELAY_T = "target_logret_7d", "expected_delay_days"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="First launch: training forecast models (about 15 seconds)...")
def load_models():
    paths = {t: os.path.join("models", f"{t}_best.joblib") for t in (RATE_T, DELAY_T)}

    def _load():
        return {t: joblib.load(p) for t, p in paths.items()}

    try:
        if all(os.path.exists(p) for p in paths.values()):
            return _load()
    except Exception:
        pass
    run_step1_training()
    return _load()


@st.cache_data
def load_data():
    df = pd.read_csv("sanchay_ml_features_v3.csv", parse_dates=["date"])
    return df.sort_values(["date", "port_name"]).reset_index(drop=True)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_resource
def load_yolo():
    candidates = [
        "best.pt",
        os.path.join("models", "best.pt"),
        os.path.join("weights", "best.pt"),
    ]
    weights_path = next((p for p in candidates if os.path.exists(p)), None)
    if weights_path is None:
        return None
    try:
        from ultralytics import YOLO
        return YOLO(weights_path)
    except Exception:
        return None


def solve(fc, demand, spot, fwd, relief):
    if sum(demand.values()) <= 0:
        return pd.DataFrame(), {"status": "No cargo demand entered", "total_cost_usd": None}
    try:
        return solve_chartering_milp(fc, demand, allow_spot=spot, allow_fwd_7d=fwd, wait_congestion_relief_days=relief)
    except Exception as e:
        return pd.DataFrame(), {"status": f"Solver error: {e}", "total_cost_usd": None}


bundles = load_models()
df_all = load_data()
metrics_df = load_csv("metrics_comparison.csv")
train_curve = load_csv("results.csv")
yolo_model = load_yolo()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def kpi(label, value, sub="", tone=""):
    st.markdown(f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                unsafe_allow_html=True)


def banner(text, tone="ok"):
    st.markdown(f'<div class="banner {tone}">{text}</div>', unsafe_allow_html=True)


def style(fig, h=340, title=None):
    fig.update_layout(
        height=h,
        title=dict(text=title, x=0.01, xanchor="left") if title else None,
        margin=dict(l=12, r=16, t=54 if title else 16, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=INK, size=12),
        title_font=dict(family="Bricolage Grotesque, sans-serif", size=16, color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="#E4ECF1", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#E4ECF1", zeroline=False, automargin=True)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚓ SANCHAY")
st.sidebar.caption("Scenario controls")
dates = sorted(df_all["date"].dt.strftime("%Y-%m-%d").unique())
as_of = st.sidebar.selectbox("As-of date", dates, index=len(dates) - 1)
st.sidebar.markdown("### Charter options")
allow_spot = st.sidebar.toggle("Allow spot-now charters", True)
allow_fwd = st.sidebar.toggle("Allow 7-day forward charters", True)
relief = st.sidebar.slider("7-day congestion relief (days)", 0.0, 2.0, 0.5, 0.1)
st.sidebar.markdown("### Cargo demand by port (MT)")
defaults = {"Paradip": 120000, "Visakhapatnam": 90000, "Haldia": 30000, "Mundra": 140000,
            "Kandla": 0, "Krishnapatnam": 0, "Mormugao": 0, "New Mangalore": 0}
demand = {p: float(st.sidebar.number_input(p, min_value=0, step=10000, value=v)) for p, v in defaults.items()}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
fc = forecast_ports(df_all, bundles, as_of_date=as_of).copy()
fc["rate_change_pct"] = (fc["pred_benchmark_7d_usd_mt"] / fc["freight_benchmark_usd_mt"] - 1) * 100
plan, summ = solve(fc, demand, allow_spot, allow_fwd, relief)
cost = summ.get("total_cost_usd")
total_demand = sum(demand.values())
alloc = float(plan["allocated_cargo_mt"].sum()) if not plan.empty else 0.0
alerts = fc[fc["weather_disruption"].astype(str) != "Clear"]

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
chips = (f'<span class="chip">As-of {as_of}</span>'
         f'<span class="chip">Data {dates[0]} to {dates[-1]}</span>'
         f'<span class="chip ok">Forecast models ready</span>'
         + ('<span class="chip ok">Ship detector ready</span>' if yolo_model is not None
            else '<span class="chip warn">Ship detector unavailable</span>'))
st.markdown(f"""<div class="hero"><h1>Maritime Freight Forecasting & Chartering Optimizer</h1>
<p>Track congestion at eight Indian ports, anticipate 7-day freight rates and delays, and get a cost-optimal charter plan for your cargo.</p>
{chips}</div>""", unsafe_allow_html=True)

worst = fc.loc[fc["port_congestion_index"].idxmax()]
slow = fc.loc[fc["pred_delay_days"].idxmax()]
c = st.columns(5)
with c[0]: kpi("Vessels at anchorage", f"{int(fc['anchorage_vessel_count'].sum())}", f"Across {len(fc)} ports")
with c[1]: kpi("Most congested", worst["port_name"], f"Index {worst['port_congestion_index']:.2f}", "coral")
with c[2]: kpi("Longest predicted delay", f"{slow['pred_delay_days']:.1f} days", slow["port_name"], "amber")
with c[3]: kpi("Avg 7-day rate change", f"{fc['rate_change_pct'].mean():+.1f}%", "Model forecast, indicative", "")
with c[4]: kpi("Optimized charter cost", f"${cost/1e6:,.2f}M" if cost else "n/a",
               f"${cost/alloc:,.2f} per MT" if cost and alloc else f"{total_demand:,.0f} MT demand", "navy")
st.write("")
if len(alerts):
    banner("Weather alert: " + ", ".join(f"{r.port_name} ({str(r.weather_disruption).replace('_', ' ')})" for r in alerts.itertuples()), "warn")

tab_ov, tab_fc, tab_plan, tab_cv, tab_ml = st.tabs(
    ["Overview", "Forecasts", "Chartering plan", "Ship detection", "Model performance"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_ov:
    left, right = st.columns([3, 2])
    with left:
        lat = [PORT_COORDS.get(p, (np.nan, np.nan))[0] for p in fc["port_name"]]
        lon = [PORT_COORDS.get(p, (np.nan, np.nan))[1] for p in fc["port_name"]]
        fig = go.Figure(go.Scattergeo(
            lat=lat, lon=lon, text=fc["port_name"], mode="markers+text", textposition="middle right",
            marker=dict(size=10 + fc["anchorage_vessel_count"] * 1.6, color=fc["port_congestion_index"],
                        colorscale=[[0, TEAL], [.5, AMBER], [1, CORAL]], cmin=0, cmax=max(0.7, fc["port_congestion_index"].max()),
                        line=dict(color="#fff", width=1.5), opacity=.9,
                        colorbar=dict(title="Congestion", thickness=10, len=.6)),
            customdata=np.c_[fc["anchorage_vessel_count"], fc["pred_delay_days"]],
            hovertemplate="<b>%{text}</b><br>At anchorage: %{customdata[0]}<br>Predicted delay: %{customdata[1]:.1f} d<extra></extra>"))
        fig.update_geos(scope="asia", lataxis_range=[7, 26], lonaxis_range=[65, 93], showland=True, landcolor="#E7EEF2",
                        showocean=True, oceancolor="#CFE3EE", showcountries=True, countrycolor="#B9C9D4",
                        coastlinecolor="#9FB4C2", bgcolor="rgba(0,0,0,0)")
        show(style(fig, 470, "Port congestion map (bubble size = vessels at anchorage)"))
    with right:
        d = fc.sort_values("port_congestion_index")
        fig = go.Figure(go.Bar(x=d["port_congestion_index"], y=d["port_name"], orientation="h",
                               marker=dict(color=d["port_congestion_index"], colorscale=[[0, TEAL], [.5, AMBER], [1, CORAL]]),
                               hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        show(style(fig, 470, "Congestion ranking"))

    st.markdown("#### Port trend explorer")
    t1, t2 = st.columns([1, 1], gap="large")
    port_list = sorted(fc["port_name"])
    port = t1.selectbox("Port", port_list, index=port_list.index("Paradip") if "Paradip" in port_list else 0, key="trend_port_select")
    window = t2.select_slider("History window", [60, 90, 180, 365], value=180, format_func=lambda x: f"{x} days", key="trend_window_slider")
    hist = df_all[(df_all["port_name"] == port) & (df_all["date"] <= pd.to_datetime(as_of))].tail(window)
    row = fc[fc["port_name"] == port].iloc[0]
    g1, g2 = st.columns(2, gap="large")
    with g1:
        fig = go.Figure(go.Scatter(x=hist["date"], y=hist["freight_benchmark_usd_mt"], mode="lines", name="Benchmark",
                                   line=dict(color=NAVY2, width=2.4), fill="tozeroy", fillcolor="rgba(22,74,117,.07)"))
        fig.add_trace(go.Scatter(x=[hist["date"].iloc[-1], hist["date"].iloc[-1] + pd.Timedelta(days=7)],
                                 y=[row["freight_benchmark_usd_mt"], row["pred_benchmark_7d_usd_mt"]], mode="lines+markers",
                                 name="7-day forecast", line=dict(color=AMBER, width=3, dash="dot"), marker=dict(size=9, symbol="diamond")))
        y_vals = pd.concat([hist["freight_benchmark_usd_mt"], pd.Series([row["freight_benchmark_usd_mt"], row["pred_benchmark_7d_usd_mt"]])])
        y_pad = max((y_vals.max() - y_vals.min()) * 0.15, 0.5)
        fig.update_yaxes(range=[max(0.0, y_vals.min() - y_pad), y_vals.max() + y_pad])
        show(style(fig, 340, f"{port}: freight benchmark (USD/MT)"))
    with g2:
        cap_val = float(hist["port_capacity_vessels"].iloc[-1])
        fig = go.Figure(go.Scatter(x=hist["date"], y=hist["anchorage_vessel_count"], mode="lines", name="Vessels at anchorage",
                                   line=dict(color=TEAL, width=2), fill="tozeroy", fillcolor="rgba(14,154,167,.12)"))
        fig.add_hline(y=cap_val, line=dict(color=CORAL, dash="dash"),
                      annotation_text=f"Port capacity ({int(cap_val)})", annotation_position="bottom right",
                      annotation_font=dict(color=CORAL, size=11))
        v_max = max(float(hist["anchorage_vessel_count"].max()), cap_val) * 1.15
        fig.update_yaxes(range=[0.0, max(v_max, 5.0)])
        show(style(fig, 340, f"{port}: vessels at anchorage"))

# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------
with tab_fc:
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_bar(name="Today", x=fc["port_name"], y=fc["freight_benchmark_usd_mt"], marker_color=NAVY2)
        fig.add_bar(name="In 7 days (forecast)", x=fc["port_name"], y=fc["pred_benchmark_7d_usd_mt"], marker_color=TEAL)
        fig.update_layout(barmode="group")
        show(style(fig, 340, "Freight benchmark (USD/MT)"))
    with b:
        d = fc.sort_values("pred_delay_days", ascending=False)
        fig = go.Figure(go.Bar(x=d["port_name"], y=d["pred_delay_days"], marker_color=AMBER,
                               hovertemplate="%{x}: %{y:.2f} days<extra></extra>"))
        show(style(fig, 340, "Predicted delay (days)"))

    st.markdown("#### Forecast table")
    tbl = fc[["port_name", "date", "anchorage_vessel_count", "port_congestion_index", "freight_benchmark_usd_mt",
              "pred_benchmark_7d_usd_mt", "rate_change_pct", "pred_delay_days"]].copy()
    tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(tbl, width="stretch", hide_index=True, column_config={
        "port_name": "Port", "date": "Latest data",
        "anchorage_vessel_count": st.column_config.NumberColumn("Vessels at anchorage", format="%d"),
        "port_congestion_index": st.column_config.ProgressColumn("Congestion index", min_value=0, max_value=1, format="%.2f"),
        "freight_benchmark_usd_mt": st.column_config.NumberColumn("Rate today (USD/MT)", format="%.2f"),
        "pred_benchmark_7d_usd_mt": st.column_config.NumberColumn("Rate in 7 days (USD/MT)", format="%.2f"),
        "rate_change_pct": st.column_config.NumberColumn("Rate change", format="%+.1f%%"),
        "pred_delay_days": st.column_config.NumberColumn("Predicted delay (days)", format="%.2f")})
    st.caption("The 7-day rate forecast is indicative: on the test period it does not beat a simple 'rate stays the same' baseline "
               "(see Model performance). The delay forecast is the stronger signal.")

# ---------------------------------------------------------------------------
# Chartering plan
# ---------------------------------------------------------------------------
with tab_plan:
    ok = "optimal" in str(summ["status"]).lower()
    if total_demand <= 0:
        banner("Enter cargo demand in the sidebar to build a plan.", "warn")
    elif plan.empty:
        banner(f"No feasible plan: {summ['status']}. Try enabling more charter options.", "bad")
    else:
        banner(f"Solver status: {'optimal' if ok else summ['status']}. Total expected cost ${cost:,.2f}", "ok" if ok else "warn")

    if not plan.empty:
        _, base = solve(fc, demand, True, False, relief)
        base_cost = base.get("total_cost_usd")
        k = st.columns(4)
        with k[0]: kpi("Voyages planned", int(plan["spot_now_trips"].sum() + plan["fwd_7d_trips"].sum()),
                       f"{int(plan['fwd_7d_trips'].sum())} forward, {int(plan['spot_now_trips'].sum())} spot")
        with k[1]: kpi("Cargo allocated", f"{alloc:,.0f} MT", f"Demand {total_demand:,.0f} MT", "amber")
        with k[2]: kpi("Cost per MT", f"${cost/alloc:,.2f}", "All-in, incl. port and demurrage")
        with k[3]: kpi("Saving vs spot-only", f"${base_cost - cost:,.0f}" if base_cost else "n/a",
                       f"{(base_cost - cost) / base_cost * 100:.1f}% lower" if base_cost else "Spot-only plan infeasible", "navy")
        st.write("")
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            for cls, g in plan.groupby("vessel_class"):
                fig.add_bar(name=cls, x=g["port_name"], y=g["expected_cost_usd"])
            fig.update_layout(barmode="stack", colorway=[NAVY2, TEAL, AMBER, CORAL])
            show(style(fig, 330, "Expected cost by port and vessel class (USD)"))
        with b:
            cov = plan.groupby("port_name")["allocated_cargo_mt"].sum()
            ports = [p for p, v in demand.items() if v > 0]
            fig = go.Figure()
            fig.add_bar(name="Demand", x=ports, y=[demand[p] for p in ports], marker_color="#B8C9D6")
            fig.add_bar(name="Allocated", x=ports, y=[cov.get(p, 0) for p in ports], marker_color=TEAL)
            fig.update_layout(barmode="group")
            show(style(fig, 330, "Cargo demand vs allocation (MT)"))

        st.markdown("#### Charter plan")
        st.dataframe(plan, width="stretch", hide_index=True, column_config={
            "port_name": "Port", "vessel_class": "Vessel class", "spot_now_trips": "Spot trips", "fwd_7d_trips": "7-day forward trips",
            "typical_cargo_mt": st.column_config.NumberColumn("Cargo per trip (MT)", format="%d"),
            "allocated_cargo_mt": st.column_config.NumberColumn("Allocated (MT)", format="%d"),
            "port_max_draft_m": st.column_config.NumberColumn("Port draft (m)", format="%.1f"),
            "vessel_draft_m": st.column_config.NumberColumn("Vessel draft (m)", format="%.2f"),
            "pred_delay_days": st.column_config.NumberColumn("Delay (days)", format="%.2f"),
            "curr_rate_usd_mt": st.column_config.NumberColumn("Rate now", format="$%.2f"),
            "pred_rate_7d_usd_mt": st.column_config.NumberColumn("Rate in 7d", format="$%.2f"),
            "expected_cost_usd": st.column_config.NumberColumn("Expected cost (USD)", format="$%d")})
        st.download_button("Download plan (CSV)", plan.to_csv(index=False).encode(), f"sanchay_plan_{as_of}.csv", "text/csv")

        st.markdown("#### Charter strategy comparison")
        rows = []
        for name, s, f in [("Spot only", True, False), ("7-day forward only", False, True), ("Spot and forward", True, True)]:
            _, sm = solve(fc, demand, s, f, relief)
            rows.append({"Strategy": name, "Total cost (USD)": sm.get("total_cost_usd"),
                         "Result": "Optimal" if "optimal" in str(sm["status"]).lower() else "Infeasible"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "Total cost (USD)": st.column_config.NumberColumn(format="$%d")})

# ---------------------------------------------------------------------------
# Ship detection
# ---------------------------------------------------------------------------
with tab_cv:
    if not train_curve.empty:
        last = train_curve.iloc[-1]
        m = st.columns(4)
        with m[0]: kpi("Precision", f"{last['metrics/precision(B)']:.1%}", "Detections that are real ships")
        with m[1]: kpi("Recall", f"{last['metrics/recall(B)']:.1%}", "Real ships found", "amber")
        with m[2]: kpi("mAP50", f"{last['metrics/mAP50(B)']:.3f}", "YOLOv8s, 1 class", "navy")
        with m[3]: kpi("mAP50-95", f"{last['metrics/mAP50-95(B)']:.3f}", f"After {int(last['epoch'])} epochs", "coral")
        st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Check an image")
        img = st.file_uploader("Satellite or Google Earth image (JPG or PNG)", type=["jpg", "jpeg", "png"], key="cv_image_uploader")
        if img is None:
            st.info("Upload an image to count ships in it.")
        elif yolo_model is None:
            st.warning("The ship detector is unavailable. Add best.pt next to app.py and install ultralytics.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img.name)[1] or ".jpg") as tmp:
                tmp.write(img.getbuffer())
            try:
                with st.spinner("Detecting ships..."):
                    res = yolo_model.predict(source=tmp.name, conf=0.35, iou=0.5, imgsz=768, verbose=False)[0]
                n = len(res.boxes) if res.boxes is not None else 0
                conf = float(res.boxes.conf.cpu().numpy().mean()) if n else 0.0
                st.image(res.plot()[:, :, ::-1], caption=f"{n} ships detected", width="stretch")
                q = st.columns(2)
                with q[0]: kpi("Ships detected", n)
                with q[1]: kpi("Mean confidence", f"{conf:.2f}", "", "amber")
            finally:
                os.remove(tmp.name)
    with right:
        if not train_curve.empty:
            fig = go.Figure()
            for col, name, colr in [("metrics/precision(B)", "Precision", NAVY2), ("metrics/recall(B)", "Recall", AMBER),
                                    ("metrics/mAP50(B)", "mAP50", TEAL)]:
                fig.add_scatter(x=train_curve["epoch"], y=train_curve[col], name=name, mode="lines+markers", line=dict(color=colr))
            fig.update_yaxes(range=[0.5, 0.9])
            show(style(fig, 360, "Training progress by epoch"))
            st.caption("Metrics were still rising at epoch 15, so more training epochs should improve detection further.")

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
with tab_ml:
    st.markdown("Chronological test: trained up to 2025-05-31, evaluated from 2025-07-01. "
                "*With CV* adds ship counts from satellite imagery as features.")
    if metrics_df.empty:
        st.info("metrics_comparison.csv not found.")
    else:
        mm = metrics_df.copy()
        test = df_all[df_all["date"] >= "2025-07-01"].dropna(subset=["benchmark_next_7d_usd_mt"])
        persist = float((test["benchmark_next_7d_usd_mt"] - test["freight_benchmark_usd_mt"]).abs().mean())
        best = mm.sort_values("mae").groupby(["target", "feature_set"], as_index=False).first()
        dly = best[best["target"] == DELAY_T].set_index("feature_set")["mae"]
        rate = best[best["target"] == RATE_T].set_index("feature_set")["bench_7d_price_mae_usd_mt"]
        k = st.columns(3)
        if {"Without_CV", "With_CV"} <= set(dly.index):
            with k[0]: kpi("Delay error with CV", f"{dly['With_CV']:.2f} days",
                           f"{(1 - dly['With_CV'] / dly['Without_CV']) * 100:.0f}% lower than without CV", "")
        if "With_CV" in rate.index:
            with k[1]: kpi("7-day rate error (model)", f"${rate['With_CV']:.2f}/MT", "Mean absolute error", "amber")
        with k[2]: kpi("7-day rate error (no change)", f"${persist:.2f}/MT", "Baseline: rate stays the same", "navy")
        st.write("")
        fig = go.Figure()
        for fs, colr in [("Without_CV", "#B8C9D6"), ("With_CV", TEAL)]:
            g = mm[mm["feature_set"] == fs]
            fig.add_bar(name=fs.replace("_", " "), x=[f"{t.replace('target_', '')} ({m})" for t, m in zip(g["target"], g["model"])],
                        y=g["mae"], marker_color=colr)
        fig.update_layout(barmode="group")
        show(style(fig, 320, "Test error (MAE) by feature set"))
        st.dataframe(mm, width="stretch", hide_index=True)
        st.markdown(
            '<div class="note"><b>How to read this.</b> Ship counts from imagery clearly improve the delay forecast. '
            'The 7-day rate model has a negative R² and a larger error than the no-change baseline, so treat its output as a scenario input, '
            'not a price call. The numeric history is a synthetic v3 benchmark dataset.</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">SANCHAY · Maritime Freight Forecasting & Chartering Optimizer</div>', unsafe_allow_html=True)