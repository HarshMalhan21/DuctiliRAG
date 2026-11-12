import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Steel Fatigue Predictor",
                   page_icon="⚙️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] { background: #f5f6fa; }
[data-testid="stHeader"]           { background: transparent; }
[data-testid="stDecoration"]       { display: none; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 100%);
    border-radius: 20px;
    padding: 36px 36px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '⚙';
    position: absolute;
    right: -10px; top: -20px;
    font-size: 160px;
    opacity: 0.06;
    line-height: 1;
}
.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.68rem;
    font-weight: 600;
    color: #c4b5fd;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -0.5px !important;
    line-height: 1.2 !important;
    margin: 0 0 10px !important;
}
.hero p {
    color: #a5b4fc !important;
    font-size: 0.88rem !important;
    margin: 0 !important;
    font-weight: 400 !important;
}
.hero-pills {
    display: flex; gap: 8px; margin-top: 18px; flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.68rem;
    color: #c4b5fd;
    font-weight: 500;
}

/* ── input card ── */
.input-card {
    background: white;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    padding: 24px;
    margin-bottom: 16px;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f3f4f6;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #6b7280 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #4f46e5 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── sliders ── */
label {
    color: #374151 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
/* slider track fill */
[data-testid="stSlider"] [data-baseweb="slider"] div div div div {
    background: #4f46e5 !important;
}
/* slider thumb */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: white !important;
    border: 2.5px solid #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
    width: 18px !important;
    height: 18px !important;
}
/* slider tooltip — styled clean */
[data-testid="stSlider"] [data-baseweb="tooltip"] div {
    background: white !important;
    background-color: white !important;
    color: #374151 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
    padding: 2px 8px !important;
}

/* ── number inputs ── */
input[type="number"] {
    border-radius: 8px !important;
    border: 1.5px solid #e5e7eb !important;
    background: #fafafa !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
}
input[type="number"]:focus {
    border-color: #4f46e5 !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
    outline: none !important;
}

/* ── selectbox ── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border: 1.5px solid #e5e7eb !important;
    background: #fafafa !important;
    font-size: 0.85rem !important;
}

/* ── toggle ── */
[data-testid="stToggle"] p { font-size: 0.8rem !important; color: #374151 !important; }

/* ── predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 16px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.4) !important;
    letter-spacing: 0.3px !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important;
    box-shadow: 0 8px 30px rgba(79,70,229,0.5) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button p {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

/* ── result card ── */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.mpa-label  { font-size: 0.68rem; font-weight: 600; letter-spacing: 2px;
              text-transform: uppercase; color: #9ca3af; margin-bottom: 8px; }
.mpa-value  { font-size: 4rem; font-weight: 800; line-height: 1; letter-spacing: -2px; }
.mpa-unit   { font-size: 1rem; font-weight: 400; color: #9ca3af; margin-left: 4px; }
.grade-tag  { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px;
              border-radius: 24px; font-size: 0.74rem; font-weight: 700;
              letter-spacing: 0.5px; text-transform: uppercase; margin-top: 14px; }
.divider    { height: 1px; background: #f3f4f6; margin: 18px 0; }
.stat-row   { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; }
.stat-box   { background: #f9fafb; border: 1px solid #f3f4f6;
              border-radius: 10px; padding: 12px; text-align: center; }
.stat-val   { font-size: 1rem; font-weight: 700; color: #111827; }
.stat-lbl   { font-size: 0.6rem; color: #9ca3af; text-transform: uppercase;
              letter-spacing: 0.5px; margin-top: 3px; }

.chart-head { margin: 24px 0 4px; }
.chart-head h4 { font-size: 0.9rem; font-weight: 700; color: #111827; margin: 0; }
.chart-head p  { font-size: 0.76rem; color: #9ca3af; margin: 3px 0 0; }

/* scroll hint */
.scroll-hint {
    text-align: center; padding: 10px 0 4px;
    color: #9ca3af; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.3px;
    animation: bounce 1.5s infinite;
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(4px); }
}
</style>
""", unsafe_allow_html=True)

# JS injection to forcibly remove purple background from slider tooltips
st.markdown("""
<script>
function fixSliders() {
    document.querySelectorAll('[data-baseweb="tooltip"]').forEach(el => {
        el.style.setProperty('background', 'white', 'important');
        el.style.setProperty('background-color', 'white', 'important');
        el.style.setProperty('color', '#374151', 'important');
        el.style.setProperty('border', '1px solid #e5e7eb', 'important');
        el.style.setProperty('border-radius', '6px', 'important');
        el.querySelectorAll('*').forEach(child => {
            child.style.setProperty('background', 'transparent', 'important');
            child.style.setProperty('background-color', 'transparent', 'important');
            child.style.setProperty('color', '#374151', 'important');
        });
    });
}
// run on load and whenever DOM changes
fixSliders();
const obs = new MutationObserver(fixSliders);
obs.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)


# ── load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('best_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features


model, scaler, features = load_model()


# ── hero banner ───────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-tag'>ML Prediction Tool</div>
    <h1>Steel Fatigue Strength<br>Predictor</h1>
    <p>Input composition and heat treatment parameters<br>to predict rotating bending fatigue strength.</p>
</div>
""", unsafe_allow_html=True)


# ── tabbed inputs ─────────────────────────────────────────────────
# ── tabbed inputs ─────────────────────────────────────────────────
# ── tabbed inputs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🔥  Heat Treatment", "🧪  Composition", "📐  Structural"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        NT = st.slider("Normalizing Temp (°C) - NT", 825, 930, 870, 5)
        THT = st.slider("Hardening Temp (°C) - THT",   30, 870, 850, 5)
        THt = st.slider("Hardening Time (hrs) - THt",   0,  30,  30, 5)
    with c2:
        TT = st.slider("Tempering Temp (°C) - TT", 30, 680, 600, 10)
        Tt = st.slider("Tempering Time (hrs) - Tt",  0, 120, 60, 5)
        QmT = st.slider("Quench Media (°C) - QmT",  30, 140,  30, 10)
    with c3:
        THQCr = st.selectbox(
            "Cooling Rate (°C/s) - THQCr", [0, 8, 16, 24], index=1)
        carburized = st.toggle("Carburized?", value=False)
        if carburized:
            CT = st.slider("Carburize Temp (°C) - CT", 30, 930, 910, 5)
            Ct = st.slider("Carburize Time (hrs) - Ct", 0, 540, 300, 20)
            Dt = st.slider("Diffusion Time (hrs) - Dt",  0,  70,  35,  5)
        else:
            CT, Ct, Dt = 30, 0, 0

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        C = st.number_input("Carbon - C",     0.17, 0.63,
                            0.40, 0.01, format="%.2f")
        Si = st.number_input("Silicon - Si",   0.16, 2.05,
                             0.25, 0.01, format="%.2f")
    with c2:
        Mn = st.number_input("Manganese - Mn", 0.37, 1.60,
                             0.75, 0.01, format="%.2f")
        Cr = st.number_input("Chromium - Cr",  0.01, 1.17,
                             0.90, 0.01, format="%.2f")
    with c3:
        Ni = st.number_input("Nickel - Ni",    0.01, 2.78,
                             0.10, 0.05, format="%.2f")
        Mo = st.number_input("Molybdenum - Mo", 0.00,
                             0.24, 0.20, 0.01, format="%.2f")
    Cu = st.number_input("Copper (wt%) - Cu", 0.01,
                         0.26, 0.05, 0.01, format="%.2f")

with tab3:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        RedRatio = st.number_input(
            "Reduction Ratio", 240.0, 5530.0, 700.0, 50.0,   format="%.1f")
    with c2:
        dA = st.number_input("Inclusions A - dA",     0.0,
                             0.1, 0.001, 0.001, format="%.3f")
    with c3:
        dB = st.number_input("Inclusions B - dB",     0.0,
                             0.1, 0.001, 0.001, format="%.3f")
    with c4:
        dC = st.number_input("Inclusions C - dC",     0.0,
                             0.1, 0.001, 0.001, format="%.3f")

predict_btn = st.button("⚡  Predict Fatigue Strength",
                        use_container_width=True)

if predict_btn:
    st.markdown("<div class='scroll-hint'>↓ &nbsp; scroll down to see results</div>", unsafe_allow_html=True)


# ── results ───────────────────────────────────────────────────────
if predict_btn:
    sample = {
        'NT': NT, 'THT': THT, 'THt': THt, 'THQCr': THQCr,
        'CT': CT, 'Ct': Ct, 'Dt': Dt, 'QmT': QmT,
        'TT': TT, 'Tt': Tt,
        'C': C, 'Si': Si, 'Mn': Mn,
        'Ni': Ni, 'Cr': Cr, 'Cu': Cu, 'Mo': Mo,
        'RedRatio': RedRatio, 'dA': dA, 'dB': dB, 'dC': dC
    }
    sample['CE'] = C + Mn/6 + (Cr + Mo)/5 + (Ni + Cu)/15
    sample['Mo_added'] = 1 if Mo > 0 else 0
    sample['Carburized'] = 1 if CT > 30 else 0

    input_df = pd.DataFrame([sample])[features]
    input_scaled = scaler.transform(input_df)
    fatigue_pred = round(np.exp(model.predict(input_scaled)[0]), 1)

    if fatigue_pred < 450:
        grade, color, bg, border, dot = "Low",    "#ef4444", "#fff1f2", "#fecdd3", "🔴"
    elif fatigue_pred < 700:
        grade, color, bg, border, dot = "Medium", "#f59e0b", "#fffbeb", "#fde68a", "🟡"
    else:
        grade, color, bg, border, dot = "High",   "#10b981", "#ecfdf5", "#a7f3d0", "🟢"

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2, gap="large")

    with r1:
        st.markdown(f"""
        <div class='result-card' style='height:320px;display:flex;flex-direction:column;justify-content:center'>
            <div class='mpa-label'>Fatigue Strength</div>
            <div>
                <span class='mpa-value' style='color:{color}'>{fatigue_pred}</span>
                <span class='mpa-unit'>MPa</span>
            </div>
            <span class='grade-tag' style='background:{bg};color:{color};border:1.5px solid {border}'>
                {dot} {grade} Grade
            </span>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fatigue_pred,
            number={'suffix': " MPa", 'font': {
                'color': color, 'size': 28, 'family': 'Inter'}},
            title={'text': "Dataset Range Position", 'font': {
                'size': 14, 'color': '#6b7280', 'family': 'Inter'}},
            gauge={
                'axis': {
                    'range': [225, 1190],
                    'tickvals': [225, 450, 700, 1190],
                    'ticktext': ['225', '450', '700', '1190'],
                    'tickcolor': '#d1d5db',
                    'tickfont': {'color': '#9ca3af', 'size': 10, 'family': 'Inter'}
                },
                'bar': {'color': color, 'thickness': 0.28},
                'bgcolor': 'white',
                'bordercolor': '#f3f4f6',
                'steps': [
                    {'range': [225,  450], 'color': '#fff1f2'},
                    {'range': [450,  700], 'color': '#fffbeb'},
                    {'range': [700, 1190], 'color': '#ecfdf5'},
                ],
                'threshold': {
                    'line': {'color': color, 'width': 3},
                    'thickness': 0.8, 'value': fatigue_pred
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font={'family': 'Inter', 'color': '#374151'},
            height=280,
            margin=dict(t=40, b=20, l=40, r=70)
        )
        st.markdown(
            "<div class='result-card' style='padding:12px 16px'>", unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # stat row below both charts

    # feature importance
    st.markdown("""
    <div class='chart-head'>
        <h4>Feature Importance</h4>
        <p>Which inputs the model relied on most for this prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        imp = dict(zip(features, model.feature_importances_))
        top8 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:8]
        names = [x[0] for x in top8]
        vals = [x[1] for x in top8]
        colors = [color if n in ['Tt', 'CT', 'TT',
                                 'Carburized', 'NT'] else '#e0e7ff' for n in names]

        fig2 = go.Figure(go.Bar(
            x=vals, y=names, orientation='h',
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in vals],
            textposition='outside',
            textfont=dict(color='#9ca3af', size=10, family='Inter')
        ))
        fig2.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            font={'family': 'Inter', 'color': '#374151', 'size': 11},
            xaxis=dict(gridcolor='#f9fafb', color='#d1d5db', showgrid=True,
                       zeroline=False, title=''),
            yaxis=dict(color='#374151', autorange='reversed',
                       tickfont=dict(size=11, color='#374151')),
            height=280, margin=dict(t=8, b=20, l=10, r=55)
        )
        st.plotly_chart(fig2, use_container_width=True,
                        config={"displayModeBar": False})
    except Exception:
        pass
