import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dukaan Demand Forecast",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e8e8e8;
    }

    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.4rem;
        font-weight: 600;
        color: #00e5a0;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .sub-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1rem;
        color: #888;
        margin-top: 4px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00e5a0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-unit {
        font-size: 0.85rem;
        color: #666;
    }
    .insight-box {
        background: linear-gradient(135deg, #0d2b20, #0f1f2e);
        border-left: 3px solid #00e5a0;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: linear-gradient(135deg, #2b1a0d, #2b200f);
        border-left: 3px solid #ffaa00;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #ffcc66;
    }
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #00e5a0;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #1e2130;
        padding-bottom: 6px;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #0d0f18;
        border-right: 1px solid #1e2130;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #aaa !important;
        font-size: 0.85rem !important;
    }
    .stButton > button {
        background: #00e5a0;
        color: #0f1117;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: #00ffb3;
        color: #0f1117;
    }
    .product-tag {
        display: inline-block;
        background: #1a2a22;
        border: 1px solid #00e5a040;
        color: #00e5a0;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sample Data Generator ─────────────────────────────────────────────────────
def generate_sample_data(product, weeks=16):
    """Generate realistic Pakistani retail sales data."""
    np.random.seed(hash(product) % 1000)
    
    base_sales = {
        "Atta (Flour) - 10kg": (45, 8, True),
        "Cooking Oil - 5L": (30, 5, True),
        "Sugar - 1kg": (60, 12, False),
        "Chai Patti (Tea) - 200g": (80, 15, True),
        "Doodh (Milk) - 1L": (120, 25, False),
        "Biscuits - Assorted": (90, 20, False),
        "Soap / Detergent": (40, 10, True),
        "Mobile Top-up Cards": (200, 50, False),
        "Cold Drinks - 1.5L": (55, 30, True),
        "Chips / Snacks": (110, 35, False),
    }.get(product, (50, 10, False))
    
    base, noise_level, has_trend = base_sales
    dates = [datetime.today() - timedelta(weeks=weeks-i) for i in range(weeks)]
    
    sales = []
    for i in range(weeks):
        trend = (i * 0.4) if has_trend else 0
        seasonal = base * 0.15 * np.sin(2 * np.pi * i / 4)  # ~monthly cycle
        ramadan_bump = 20 if 10 <= i <= 14 else 0  # simulate Ramadan bump
        noise = np.random.normal(0, noise_level)
        val = max(0, base + trend + seasonal + ramadan_bump + noise)
        sales.append(round(val))
    
    return pd.DataFrame({"date": dates, "units_sold": sales, "week": range(1, weeks+1)})


# ─── Forecasting Model ─────────────────────────────────────────────────────────
def forecast(df, forecast_weeks=4):
    X = df[["week"]].values
    y = df["units_sold"].values
    
    # Polynomial regression (degree 2) captures trend curves better
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)
    
    # In-sample predictions
    y_pred_train = model.predict(X)
    mae = mean_absolute_error(y, y_pred_train)
    
    # Future predictions
    last_week = df["week"].max()
    future_weeks = np.array([[last_week + i] for i in range(1, forecast_weeks + 1)])
    future_preds = model.predict(future_weeks)
    future_preds = np.maximum(future_preds, 0)  # no negative sales
    
    future_dates = [df["date"].max() + timedelta(weeks=i) for i in range(1, forecast_weeks + 1)]
    
    future_df = pd.DataFrame({
        "date": future_dates,
        "units_predicted": future_preds.round().astype(int),
        "week": [last_week + i for i in range(1, forecast_weeks + 1)]
    })
    
    return future_df, mae, y_pred_train


def reorder_suggestion(avg_daily, lead_days=1, safety_factor=1.3):
    """Simple reorder quantity with safety stock."""
    return int(avg_daily * lead_days * 7 * safety_factor)


# ─── App Layout ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛒 Dukaan Demand Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered stock prediction for Pakistani retailers — reduce waste, never run out</div>', unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Configuration</div>', unsafe_allow_html=True)
    
    product = st.selectbox(
        "Select Product",
        [
            "Atta (Flour) - 10kg",
            "Cooking Oil - 5L",
            "Sugar - 1kg",
            "Chai Patti (Tea) - 200g",
            "Doodh (Milk) - 1L",
            "Biscuits - Assorted",
            "Soap / Detergent",
            "Mobile Top-up Cards",
            "Cold Drinks - 1.5L",
            "Chips / Snacks",
        ]
    )
    
    history_weeks = st.slider("Weeks of history to use", 8, 24, 16)
    forecast_weeks = st.slider("Weeks to forecast ahead", 1, 8, 4)
    current_stock = st.number_input("Current stock (units)", min_value=0, value=50)
    cost_per_unit = st.number_input("Cost per unit (PKR)", min_value=1, value=150)
    
    st.markdown('<div class="section-header">📤 Upload Your Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (date, units_sold)", type=["csv"])
    st.caption("Or use our sample data for demo ↑")
    
    run_btn = st.button("🔮 Run Forecast")

# ── Load / Generate Data ──
if uploaded:
    try:
        df = pd.read_csv(uploaded, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["week"] = range(1, len(df) + 1)
        st.success("✅ Data loaded from your file!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = generate_sample_data(product, history_weeks)
else:
    df = generate_sample_data(product, history_weeks)

# ── Always show on load ──
future_df, mae, y_pred_train = forecast(df, forecast_weeks)
avg_weekly = df["units_sold"].mean()
reorder_qty = reorder_suggestion(avg_weekly / 7)
weeks_of_stock = current_stock / avg_weekly if avg_weekly > 0 else 0

# ── KPI Row ──
st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Weekly Sales</div>
        <div class="metric-value">{avg_weekly:.0f}</div>
        <div class="metric-unit">units / week</div>
    </div>""", unsafe_allow_html=True)

with c2:
    next_week = future_df["units_predicted"].iloc[0]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Next Week Forecast</div>
        <div class="metric-value">{next_week}</div>
        <div class="metric-unit">units predicted</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Weeks of Stock Left</div>
        <div class="metric-value">{weeks_of_stock:.1f}</div>
        <div class="metric-unit">at current sales rate</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Accuracy (MAE)</div>
        <div class="metric-value">{mae:.1f}</div>
        <div class="metric-unit">avg units error</div>
    </div>""", unsafe_allow_html=True)

# ── Main Chart ──
st.markdown('<div class="section-header">📈 Sales History + Forecast</div>', unsafe_allow_html=True)

fig = go.Figure()

# Actual sales
fig.add_trace(go.Scatter(
    x=df["date"], y=df["units_sold"],
    name="Actual Sales",
    line=dict(color="#00e5a0", width=2),
    mode="lines+markers",
    marker=dict(size=5, color="#00e5a0"),
))

# Trend line (in-sample)
fig.add_trace(go.Scatter(
    x=df["date"], y=y_pred_train,
    name="Trend",
    line=dict(color="#ffffff", width=1.5, dash="dot"),
    mode="lines",
    opacity=0.4,
))

# Forecast
fig.add_trace(go.Scatter(
    x=future_df["date"], y=future_df["units_predicted"],
    name="Forecast",
    line=dict(color="#ffaa00", width=2.5, dash="dash"),
    mode="lines+markers",
    marker=dict(size=8, symbol="diamond", color="#ffaa00"),
))

# Confidence band (±15%)
fig.add_trace(go.Scatter(
    x=list(future_df["date"]) + list(future_df["date"])[::-1],
    y=list((future_df["units_predicted"] * 1.15).round()) + list((future_df["units_predicted"] * 0.85).round())[::-1],
    fill="toself",
    fillcolor="rgba(255,170,0,0.08)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Band",
    showlegend=True,
))

# Vertical line: today
fig.add_vline(x=df["date"].max().timestamp() * 1000, line_dash="dash", line_color="#444", annotation_text="Today", annotation_font_color="#666")

fig.update_layout(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font=dict(family="IBM Plex Mono", color="#aaa"),
    legend=dict(bgcolor="#1a1d27", bordercolor="#2a2d3a", borderwidth=1),
    xaxis=dict(gridcolor="#1e2130", zeroline=False),
    yaxis=dict(gridcolor="#1e2130", zeroline=False, title="Units Sold"),
    margin=dict(l=10, r=10, t=20, b=10),
    height=380,
)

st.plotly_chart(fig, use_container_width=True)

# ── Forecast Table ──
col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.markdown('<div class="section-header">🔮 Weekly Forecast Breakdown</div>', unsafe_allow_html=True)
    display_df = future_df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%d %b %Y")
    display_df["low"] = (display_df["units_predicted"] * 0.85).round().astype(int)
    display_df["high"] = (display_df["units_predicted"] * 1.15).round().astype(int)
    display_df["cost_est (PKR)"] = display_df["units_predicted"] * cost_per_unit
    display_df = display_df.rename(columns={
        "date": "Week Starting",
        "units_predicted": "Predicted",
        "low": "Low Estimate",
        "high": "High Estimate",
    })[["Week Starting", "Predicted", "Low Estimate", "High Estimate", "cost_est (PKR)"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col_right:
    st.markdown('<div class="section-header">💡 Smart Insights</div>', unsafe_allow_html=True)
    
    # Restock warning
    if weeks_of_stock < 1.5:
        st.markdown(f'<div class="warning-box">⚠️ <b>Low Stock Alert</b><br>You have ~{weeks_of_stock:.1f} weeks of stock left. Reorder soon!</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="insight-box">✅ <b>Stock OK</b><br>~{weeks_of_stock:.1f} weeks of stock remaining at current pace.</div>', unsafe_allow_html=True)
    
    # Reorder qty
    st.markdown(f'<div class="insight-box">📦 <b>Suggested Reorder</b><br>Order <b>{reorder_qty} units</b> to cover next {forecast_weeks} weeks with safety buffer.</div>', unsafe_allow_html=True)
    
    # Trend direction
    trend_dir = "📈 Increasing" if future_df["units_predicted"].iloc[-1] > future_df["units_predicted"].iloc[0] else "📉 Decreasing"
    st.markdown(f'<div class="insight-box">🔍 <b>Demand Trend</b><br>{trend_dir} over the forecast period.</div>', unsafe_allow_html=True)
    
    total_cost = (future_df["units_predicted"].sum() * cost_per_unit)
    st.markdown(f'<div class="insight-box">💰 <b>Est. Total Procurement</b><br>PKR {total_cost:,.0f} for {forecast_weeks}-week stock.</div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#444; font-size:0.75rem; font-family: IBM Plex Mono'>Built with ❤️ to make Pakistani retail more efficient · Inspired by Tajir's mission</div>",
    unsafe_allow_html=True
)
