"""
app.py
------
Streamlit dark-theme dashboard for the Biogas DNN Surrogate model.

Modes
-----
1. Forward Predict  – move 6 sliders → get PURITY, MASSFLOW, CO2OUT, H2OUT
2. Inverse Optimize – set desired PURITY + MASSFLOW → find minimum H2 inputs

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
from optimizer import run_inverse_optimization, BOUNDS, FEATURE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Biogas Process Optimizer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for engineering dark theme
st.markdown("""
<style>
    /* ── Base dark background ── */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .main .block-container { padding-top: 1.5rem; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] label { color: #8b949e !important; font-size: 0.78rem; }

    /* ── Cards ── */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card .label { color: #8b949e; font-size: 0.82rem; letter-spacing: 0.06em; text-transform: uppercase; }
    .metric-card .value { color: #58a6ff; font-size: 2.0rem; font-weight: 700; margin-top: 4px; }
    .metric-card .unit  { color: #8b949e; font-size: 0.75rem; }

    /* ── Section headings ── */
    .section-header {
        color: #58a6ff;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin-bottom: 14px;
        margin-top: 10px;
    }

    /* ── Input row card (optimizer results) ── */
    .input-tag {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 5px 10px;
        margin: 4px;
        font-size: 0.82rem;
        color: #d1d5db;
    }
    .input-tag span { color: #34d399; font-weight: 600; }

    /* ── Status badges ── */
    .badge-success { background:#0f3d23; color:#3fb950; border:1px solid #238636;
                     border-radius:6px; padding:5px 12px; font-size:0.85rem; }
    .badge-warn    { background:#3d2a0f; color:#e3b341; border:1px solid #9e6a03;
                     border-radius:6px; padding:5px 12px; font-size:0.85rem; }

    /* ── Slider label colour ── */
    div[data-testid="stSlider"] label { color: #c9d1d9 !important; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border-radius: 6px 6px 0 0;
        color: #8b949e; padding: 8px 20px;
        border: 1px solid #30363d; border-bottom: none;
    }
    .stTabs [aria-selected="true"] { background: #1f6feb !important; color: #fff !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #238636; color: #fff;
        border: none; border-radius: 6px;
        padding: 10px 28px; font-size: 0.95rem;
        font-weight: 600; width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #2ea043; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DNN model…")
def load_artifacts():
    model    = tf.keras.models.load_model("biogas_final_resnet_surrogate.h5", compile=False)
    scaler_X = joblib.load("scaler_X.pkl")
    pt_y     = joblib.load("pt_y.pkl")
    return model, scaler_X, pt_y

model, scaler_X, pt_y = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def add_engineered(x6: np.ndarray) -> np.ndarray:
    feed, h2, f1, f2 = x6[0], x6[1], x6[2], x6[3]
    return np.append(x6, [h2 / (feed + 1e-9), f2 - f1])

def forward_predict(x6: np.ndarray):
    x8   = add_engineered(x6).reshape(1, -1)
    x_sc = scaler_X.transform(x8)
    y_sc = model.predict(x_sc, verbose=0)
    return pt_y.inverse_transform(y_sc)[0]   # [PURITY, MASSFLOW, CO2OUT, H2OUT]

def gauge(value, title, min_val, max_val, unit="", color="#58a6ff"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#c9d1d9", "size": 14}},
        number={"font": {"color": color, "size": 28}, "suffix": f" {unit}"},
        gauge={
            "axis":  {"range": [min_val, max_val], "tickcolor": "#8b949e",
                      "tickfont": {"color": "#8b949e", "size": 10}},
            "bar":   {"color": color},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [min_val, (min_val+max_val)/2], "color": "#1a2332"},
                {"range": [(min_val+max_val)/2, max_val], "color": "#0f2444"},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        font={"color": "#c9d1d9"},
    )
    return fig

def small_metric_card(label, value, unit):
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value:.6f}</div>
        <div class="unit">{unit}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT SLIDERS (shared across both tabs)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Process Inputs")
    st.markdown("---")

    labels = {
        "VARY 1 FEED": ("Feed Flow Rate",     "kg/hr",  0.000,  0.050,  0.001),
        "VARY 2 H2":   ("H₂ Feed Rate",       "kg/hr",  0.0018, 0.004,  0.0001),
        "VARY 3 F1":   ("Flash Drum 1 Press.", "bar",     8.0,    12.0,   0.5),
        "VARY 4 F2":   ("Flash Drum 2 Press.", "bar",     10.0,   14.0,   0.5),
        "VARY 5 R2":   ("Reactor 2 Temp.",     "°C",      43.0,   50.0,   0.5),
        "VARY 6 R3":   ("Reactor 3 Temp.",     "°C",      43.0,   50.0,   0.5),
    }

    slider_vals = {}
    for key, (name, unit, lo, hi, step) in labels.items():
        slider_vals[key] = st.slider(
            f"{name} ({unit})",
            min_value=float(lo),
            max_value=float(hi),
            value=float((lo + hi) / 2),
            step=float(step),
            key=f"slider_{key}",
        )

    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.75rem;text-align:center'>"
        "ResNet DNN Surrogate · R² > 0.99<br>Trained on Aspen Plus data"
        "</div>",
        unsafe_allow_html=True,
    )

x6_input = np.array([slider_vals[k] for k in FEATURE_NAMES])


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'>
    <span style='font-size:2.4rem'>⚗️</span>
    <div>
        <div style='font-size:1.6rem;font-weight:700;color:#e6edf3'>
            Biogas Process Optimizer
        </div>
        <div style='color:#8b949e;font-size:0.88rem'>
            ResNet DNN Surrogate + Inverse H₂ Minimiser &nbsp;|&nbsp;
            Powered by Aspen Plus training data
        </div>
    </div>
</div>
<hr style='border-color:#30363d;margin-bottom:0'>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔬  Forward Prediction", "🎯  Inverse Optimizer"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — FORWARD PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("")   # breathing room
    preds = forward_predict(x6_input)
    purity, massflow, co2out, h2out = preds

    # ── Primary gauges ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">PRIMARY OUTPUTS</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            gauge(purity * 100, "CH₄ Purity", 60, 100, "%", "#58a6ff"),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            gauge(massflow, "Total Mass Flow", 0.08, 0.11, "kg/day", "#3fb950"),
            use_container_width=True,
        )

    # ── Secondary outputs ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">SECONDARY OUTPUTS</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(small_metric_card("CO₂ Outlet Flow", co2out*100, "%"), unsafe_allow_html=True)
    with c2:
        st.markdown(small_metric_card("H₂ Outlet Flow",  h2out*100,  "%"), unsafe_allow_html=True)

    # ── Input summary bar chart ───────────────────────────────────────────
    st.markdown('<div class="section-header">CURRENT INPUT VECTOR</div>', unsafe_allow_html=True)

    norm_vals = [
        (slider_vals[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0])
        for k in FEATURE_NAMES
    ]
    short_names = ["FEED", "H₂", "F1 Press", "F2 Press", "R2 Temp", "R3 Temp"]
    raw_vals    = [slider_vals[k] for k in FEATURE_NAMES]
    units_list  = ["kg/hr","kg/hr","bar","bar","°C","°C"]

    bar_fig = go.Figure(go.Bar(
        x=short_names,
        y=norm_vals,
        marker_color=["#58a6ff","#f78166","#3fb950","#d2a8ff","#ffa657","#79c0ff"],
        text=[f"{v:.4f} {u}" for v, u in zip(raw_vals, units_list)],
        textposition="outside",
        textfont={"color": "#c9d1d9", "size": 11},
    ))
    bar_fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        yaxis={"title": "Normalised Value (0–1)", "gridcolor":"#21262d",
               "color":"#8b949e", "range":[0, 1.25]},
        xaxis={"color":"#8b949e"},
        height=300, margin=dict(l=40, r=20, t=20, b=40),
        font={"color":"#c9d1d9"},
        showlegend=False,
    )
    st.plotly_chart(bar_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — INVERSE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("")

    st.markdown('<div class="section-header">SET DESIRED TARGETS</div>', unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1:
        target_purity = st.number_input(
            "Target CH₄ Purity  (fraction, e.g. 0.85)",
            min_value=0.60, max_value=0.99,
            value=0.85, step=0.005, format="%.3f",
        )
        purity_tol = st.number_input(
            "Purity Tolerance  (±)",
            min_value=0.001, max_value=0.05,
            value=0.005, step=0.001, format="%.3f",
        )
    with tc2:
        target_massflow = st.number_input(
            "Target Mass Flow  (kg/day)",
            min_value=0.080, max_value=0.110,
            value=0.090, step=0.001, format="%.4f",
        )
        massflow_tol = st.number_input(
            "Mass Flow Tolerance  (±)",
            min_value=0.0005, max_value=0.010,
            value=0.002, step=0.0005, format="%.4f",
        )

    st.markdown("")
    run_btn = st.button("🚀  Run Inverse Optimisation")

    if run_btn:
        with st.spinner("Running TF gradient optimisation across 200 candidates — ~5-10 seconds…"):
            result = run_inverse_optimization(
                target_purity   = target_purity,
                target_massflow = target_massflow,
                model           = model,
                scaler_X        = scaler_X,
                pt_y            = pt_y,
                purity_tol      = purity_tol,
                massflow_tol    = massflow_tol,
            )

        # ── Status badge ─────────────────────────────────────────────────
        if result["success"]:
            st.markdown(f'<div class="badge-success">{result["message"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-warn">{result["message"]}</div>',
                        unsafe_allow_html=True)

        st.markdown("")

        # ── Optimal Inputs ────────────────────────────────────────────────
        st.markdown('<div class="section-header">OPTIMAL INPUT SETTINGS</div>',
                    unsafe_allow_html=True)

        inp = result["inputs"]
        long_labels = {
            "VARY 1 FEED": ("Feed Flow Rate",      "kg/hr"),
            "VARY 2 H2":   ("H₂ Feed Rate ★ min", "kg/hr"),
            "VARY 3 F1":   ("Flash Drum 1 Press.", "bar"),
            "VARY 4 F2":   ("Flash Drum 2 Press.", "bar"),
            "VARY 5 R2":   ("Reactor 2 Temp.",     "°C"),
            "VARY 6 R3":   ("Reactor 3 Temp.",     "°C"),
        }

        tag_html = ""
        for k, (lbl, unit) in long_labels.items():
            val = inp[k]
            tag_html += f'<div class="input-tag">{lbl}: <span>{val:.5f}</span> {unit}</div>'

        st.markdown(tag_html, unsafe_allow_html=True)
        st.markdown("")

        # ── Predicted vs Target comparison ───────────────────────────────
        st.markdown('<div class="section-header">PREDICTED vs TARGET</div>',
                    unsafe_allow_html=True)

        pred = result["predicted"]
        comp_df = pd.DataFrame({
            "Output":    ["CH₄ Purity", "Mass Flow"],
            "Target":    [target_purity, target_massflow],
            "Predicted": [pred["PURITY"], pred["MASSFLOW"]],
            "Error":     [
                abs(pred["PURITY"]   - target_purity),
                abs(pred["MASSFLOW"] - target_massflow),
            ],
        })
        comp_df_styled = comp_df.style\
            .format({"Target":"{:.5f}","Predicted":"{:.5f}","Error":"{:.6f}"})\
            .set_properties(**{
                "background-color": "#161b22",
                "color": "#e6edf3",
                "border": "1px solid #30363d",
            })\
            .set_table_styles([{
                "selector": "th",
                "props": [("background-color","#21262d"),("color","#8b949e"),
                          ("border","1px solid #30363d")]
            }])
        st.dataframe(comp_df_styled, use_container_width=True, hide_index=True)

        # ── Secondary predicted outputs ───────────────────────────────────
        st.markdown('<div class="section-header">SECONDARY PREDICTED OUTPUTS</div>',
                    unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(small_metric_card("CO₂ Outlet Flow", pred["CO2OUT"]*100, "%"),
                        unsafe_allow_html=True)
        with s2:
            st.markdown(small_metric_card("H₂ Outlet Flow",  pred["H2OUT"]*100,  "%"),
                        unsafe_allow_html=True)

        # ── Radar / spider chart: normalised optimal inputs ───────────────
        st.markdown('<div class="section-header">OPTIMAL INPUT RADAR</div>',
                    unsafe_allow_html=True)

        norms = [
            (inp[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0])
            for k in FEATURE_NAMES
        ]
        r_labels = ["FEED", "H₂", "F1 Press", "F2 Press", "R2 Temp", "R3 Temp"]
        r_vals   = norms + [norms[0]]
        r_theta  = r_labels + [r_labels[0]]

        radar_fig = go.Figure(go.Scatterpolar(
            r=r_vals, theta=r_theta,
            fill="toself",
            fillcolor="rgba(88,166,255,0.15)",
            line={"color":"#58a6ff","width":2},
            marker={"color":"#58a6ff","size":7},
        ))
        radar_fig.update_layout(
            polar={
                "bgcolor": "#161b22",
                "radialaxis": {"visible":True,"range":[0,1],
                               "color":"#8b949e","gridcolor":"#30363d"},
                "angularaxis": {"color":"#c9d1d9","gridcolor":"#30363d"},
            },
            paper_bgcolor="#0d1117",
            height=380,
            margin=dict(l=60, r=60, t=30, b=30),
            font={"color":"#c9d1d9"},
        )
        st.plotly_chart(radar_fig, use_container_width=True)

        # ── Copy-paste summary ────────────────────────────────────────────
        with st.expander("📋  Full result summary (copy-paste ready)"):
            lines = ["=== INVERSE OPTIMISATION RESULT ==="]
            lines.append(f"Target PURITY   : {target_purity:.4f}  (tol ±{purity_tol})")
            lines.append(f"Target MASSFLOW : {target_massflow:.5f}  (tol ±{massflow_tol})")
            lines.append("")
            lines.append("--- Optimal Inputs ---")
            for k, (lbl, unit) in long_labels.items():
                lines.append(f"  {lbl:25s}: {inp[k]:.6f}  {unit}")
            lines.append("")
            lines.append("--- Predicted Outputs ---")
            for out_key in ["PURITY","MASSFLOW","CO2OUT","H2OUT"]:
                lines.append(f"  {out_key:10s}: {pred[out_key]:.6f}")
            lines.append("")
            lines.append(result["message"])
            st.code("\n".join(lines), language="text")