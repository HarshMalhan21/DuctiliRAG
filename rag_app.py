from step5_rag_generator import generate_ml_failure_report, ask_metallurgy_question, summarize_document
from step4_hybrid_retriever import get_corpus_stats_fast, ingest_uploaded_pdf_to_db
from step1_shap_bridge import predict_and_create_rag_query
import plotly.express as px
import pandas as pd
import streamlit as st
import os
import warnings

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="DuctiliRAG | Metallurgical Failure Analysis Assistant", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; overflow-anchor: auto; }

    /* Force Larger, Prominent Tab Headers Across All Streamlit Versions */
    div[data-baseweb="tab-list"] {
        gap: 14px !important;
        border-bottom: 2px solid #cbd5e1 !important;
        padding-bottom: 6px !important;
    }
    button[data-baseweb="tab"] {
        height: auto !important;
        padding: 12px 24px !important;
        border-radius: 10px 10px 0 0 !important;
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-bottom: none !important;
    }
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p,
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #1e293b !important;
        line-height: 1.35 !important;
        margin: 0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #eff6ff !important;
        border: 1px solid #93c5fd !important;
        border-bottom: 4px solid #2563eb !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] div[data-testid="stMarkdownContainer"] p,
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #1d4ed8 !important;
    }

    /* Larger Widget & Metric Labels */
    div[data-testid="stWidgetLabel"] p {
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.07), rgba(59, 130, 246, 0.04));
        border: 1px solid rgba(59, 130, 246, 0.28);
        padding: 14px 16px;
        border-radius: 10px;
    }

    /* Figure Title, Explanation & Source Tag Styling */
    .fig-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .fig-explain {
        background-color: #f1f5f9;
        border-left: 4px solid #2563eb;
        color: #0f172a;
        padding: 11px 14px;
        border-radius: 6px;
        font-size: 0.96rem;
        line-height: 1.52;
        margin-top: 8px;
        margin-bottom: 6px;
    }
    .fig-source {
        display: inline-block;
        background-color: #e2e8f0;
        color: #1e293b;
        font-size: 0.90rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

for k, v in [("chat_history", []), ("last_uploaded_doc", None), ("tab1_ml_diag", None),
             ("tab1_rag_result", None), ("tab2_summary_res", None), ("tab3_qa_res", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

_, indexed_sources = get_corpus_stats_fast()


def render_figure_gallery(images_list, heading="🖼️ Extracted Technical Figures, Graphs & Visual Explanations"):
    if not images_list:
        return
    st.markdown(f"#### {heading}")
    imgs = images_list[:8]
    cols = st.columns(2)
    for idx, item in enumerate(imgs):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='fig-title'>📐 {item.get('title', 'Technical Figure')}</div>", unsafe_allow_html=True)
                st.image(item["image_path"], width="stretch")
                if item.get("explanation"):
                    st.markdown(
                        f"<div class='fig-explain'><strong>What this shows:</strong> {item['explanation']}</div>",
                        unsafe_allow_html=True
                    )
                if item.get("source_page"):
                    st.markdown(
                        f"<div class='fig-source'>📄 Source: {item['source_page']}</div>", unsafe_allow_html=True)


with st.sidebar:
    st.title("📁 Document Library")
    uploaded_pdf = st.file_uploader(
        "Upload a failure report, standard, or lecture PDF:", type=["pdf"])

    if uploaded_pdf is not None:
        if st.button("🚀 Upload & Process Document", type="primary", width="stretch"):
            prog = st.progress(0, text="Extracting text & HD figures...")
            new_doc = ingest_uploaded_pdf_to_db(
                uploaded_pdf.read(),
                uploaded_pdf.name,
                progress_callback=lambda c, t, s: prog.progress(
                    min(int((c / float(max(1, t))) * 100), 100), text=s)
            )
            prog.empty()
            st.session_state.last_uploaded_doc = new_doc
            st.success(f"Added `{new_doc}` in HD!")
            st.rerun()

    st.divider()
    st.subheader("📚 Available Reference Documents")
    for src in indexed_sources:
        st.markdown(f"• **{src.replace('_', ' ')}**")

    if st.session_state.chat_history:
        st.divider()
        st.subheader("🕒 Recent Questions")
        for idx, item in enumerate(reversed(st.session_state.chat_history[-4:]), start=1):
            st.caption(f"**{idx}.** {item['question'][:48]}...")
        if st.button("🗑️ Clear History", width="stretch"):
            st.session_state.chat_history = []
            st.rerun()

st.title("🔬 Metallurgical Fatigue Predictor & Failure Analysis Assistant")
st.markdown(
    "Predict rotating-bending fatigue endurance limits, diagnose microstructural root causes, "
    "and query technical failure literature with Vision-captioned high-definition figure extraction."
)

tab1, tab2, tab3 = st.tabs([
    "🧪 1. Fatigue Strength Predictor & Failure Diagnosis",
    "📄 2. Document Summarizer",
    "💬 3. Literature Q&A"
])

with tab1:
    st.subheader(
        "1. Enter Steel Specimen Composition & Heat-Treatment Parameters")

    with st.form("tab1_specimen_form", border=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 🧪 Chemical Composition (wt%)")
            C = st.number_input("Carbon (%C)", 0.10, 1.20, 0.35, 0.01)
            Si = st.number_input("Silicon (%Si)", 0.05, 2.00, 0.25, 0.05)
            Mn = st.number_input("Manganese (%Mn)", 0.10, 2.00, 0.75, 0.05)
            Ni = st.number_input("Nickel (%Ni)", 0.00, 4.50, 0.05, 0.05)
            Cr = st.number_input("Chromium (%Cr)", 0.00, 3.50, 0.15, 0.05)
            Mo = st.number_input("Molybdenum (%Mo)", 0.00, 1.00, 0.00, 0.05)
            Cu = st.number_input("Copper (%Cu)", 0.00, 0.50, 0.05, 0.01)
        with c2:
            st.markdown("##### 🔥 Heat Treatment Schedule")
            NT = st.number_input(
                "Normalizing Temp (NT, °C)", 750, 1050, 885, 5)
            THT = st.number_input(
                "Through-Hardening Temp (THT, °C)", 30, 1050, 855, 5)
            THt = st.number_input(
                "Through-Hardening Time (THt, min)", 0, 240, 30, 5)
            THQCr = st.number_input(
                "Quench Cooling Rate (THQCr, °C/s)", 0, 100, 8, 1)
            QmT = st.number_input(
                "Quench Medium Temp (QmT, °C)", 10, 200, 30, 5)
            TT = st.number_input("Tempering Temp (TT, °C)", 30, 700, 600, 10)
            Tt = st.number_input("Tempering Time (Tt, min)", 0, 300, 60, 10)
        with c3:
            st.markdown("##### 🛡️ Carburizing & Cleanliness (ASTM E45)")
            CT = st.number_input(
                "Carburizing Temp (CT, °C — 30 if none)", 30, 1000, 30, 10)
            Ct = st.number_input(
                "Carburizing Time (Ct, min)", 0.0, 600.0, 0.0, 10.0)
            Dt = st.number_input(
                "Diffusion Time (Dt, min)", 0.0, 300.0, 0.0, 10.0)
            RedRatio = st.number_input(
                "Forging Reduction Ratio", 100.0, 3000.0, 500.0, 50.0)
            dA = st.number_input("Type A Sulfide (dA)", 0.00, 0.30, 0.08, 0.01)
            dB = st.number_input("Type B Alumina (dB)", 0.00, 0.30, 0.05, 0.01)
            dC = st.number_input("Type C Oxide (dC)", 0.00, 0.30, 0.02, 0.01)

        b1, b2 = st.columns(2)
        fast_btn = b1.form_submit_button(
            "⚡ Predict Fatigue Strength & Key Drivers", width="stretch")
        rag_btn = b2.form_submit_button(
            "🔬 Generate Full Metallurgical Failure Analysis Report", type="primary", width="stretch")

    raw_inputs = {"NT": NT, "THT": THT, "THt": THt, "THQCr": THQCr, "CT": CT, "Ct": Ct, "Dt": Dt, "QmT": QmT,
                  "TT": TT, "Tt": Tt, "C": C, "Si": Si, "Mn": Mn, "Ni": Ni, "Cr": Cr, "Cu": Cu, "Mo": Mo,
                  "RedRatio": RedRatio, "dA": dA, "dB": dB, "dC": dC}

    if fast_btn:
        st.session_state.tab1_ml_diag = predict_and_create_rag_query(
            raw_inputs)
        st.session_state.tab1_rag_result = None

    if rag_btn:
        st.session_state.tab1_ml_diag = predict_and_create_rag_query(
            raw_inputs)
        with st.spinner("Synthesizing comprehensive 4-section metallurgical failure report..."):
            st.session_state.tab1_rag_result = generate_ml_failure_report(
                raw_inputs, ml_diag=st.session_state.tab1_ml_diag)

    if st.session_state.tab1_ml_diag is not None:
        ml_diag = st.session_state.tab1_ml_diag
        eng = ml_diag["engineered_features"]
        st.divider()
        st.subheader("2. Predicted Fatigue Performance & Key Limiting Factors")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Predicted Fatigue Limit", f"{ml_diag['predicted_mpa']} MPa")
        m2.metric("Strength Class", ml_diag["grade"].split("(")[0].strip())
        m3.metric("Carbon Equivalent (CE)", f"{eng['CE']:.3f}")
        m4.metric("Surface Carburized",
                  "Yes" if eng["Carburized"] == 1 else "No")
        m5.metric("Mo Alloyed", "Yes" if eng["Mo_added"] == 1 else "No")

        c_chart, c_flags = st.columns([1.2, 1])
        with c_chart:
            shap_df = pd.DataFrame(ml_diag["top_4_shap_drivers"])
            fig = px.bar(shap_df.iloc[::-1], x="shap_score", y="feature", orientation="h", text="value",
                         title="Primary Factors Reducing Fatigue Limit (MPa Impact)",
                         labels={
                             "shap_score": "Estimated Impact (MPa)", "feature": "Parameter"},
                         color_discrete_sequence=["#dc2626"])
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        with c_flags:
            st.markdown("##### ⚠️ Metallurgical Risk Alerts")
            if not ml_diag["metallurgical_warnings"]:
                st.success(
                    "All heat-treatment and cleanliness parameters are within standard bounds.")
            for w in ml_diag["metallurgical_warnings"]:
                st.warning(w)

    if st.session_state.tab1_rag_result is not None:
        rag_result = st.session_state.tab1_rag_result
        st.divider()
        st.subheader("3. Metallurgical Failure Analysis Report")
        if rag_result.get("error"):
            st.error(
                "⚠️ Analysis service temporarily busy. Please click the button again.")
        else:
            st.markdown(rag_result["report_markdown"])
            render_figure_gallery(
                rag_result["linked_images"], "🖼️ Reference Figures & Explanations from Cited Literature")

with tab2:
    st.subheader("📄 Executive Document Summarizer")
    def_idx = indexed_sources.index(
        st.session_state.last_uploaded_doc) if st.session_state.last_uploaded_doc in indexed_sources else 0
    selected_doc = st.selectbox(
        "Select a Document to Summarize:", indexed_sources, index=def_idx)

    if st.button("📑 Generate Executive Summary", type="primary"):
        with st.spinner(f"Generating engineering summary for `{selected_doc}`..."):
            sum_res = summarize_document(selected_doc)
            sum_res["doc_name"] = selected_doc
            st.session_state.tab2_summary_res = sum_res

    if st.session_state.tab2_summary_res is not None:
        sum_res = st.session_state.tab2_summary_res
        if sum_res.get("error"):
            st.error("⚠️ Summary generation timed out. Please try again.")
        else:
            st.markdown(sum_res["summary"])
            render_figure_gallery(
                sum_res["linked_images"], "🖼️ Key Figures, Graphs & Visual Explanations")

with tab3:
    st.subheader("💬 Technical Literature Q&A")

    with st.form("tab3_qa_form", border=False):
        f1, f2 = st.columns(2)
        doc_scope = f1.selectbox("Search Scope:", ["All"] + indexed_sources)
        mat_scope = f2.selectbox("Material Category:", [
                                 "All", "steel", "aluminum", "titanium"])
        user_q = st.text_input("Ask an engineering question:",
                               placeholder="e.g., What is standard fatigue strength and how is it tested?")
        ask_btn = st.form_submit_button(
            "🔍 Get Answer", type="primary", width="stretch")

    if ask_btn and user_q.strip():
        with st.spinner("Searching literature and synthesizing detailed answer..."):
            ans_out = ask_metallurgy_question(
                user_q, filter_material=mat_scope, filter_source=doc_scope)
            ans_out.update({"question": user_q, "scope": doc_scope})
            st.session_state.tab3_qa_res = ans_out
            if not ans_out.get("error"):
                st.session_state.chat_history.append(
                    {"question": user_q, "answer": ans_out["answer"], "scope": doc_scope})

    if st.session_state.tab3_qa_res is not None:
        ans_out = st.session_state.tab3_qa_res
        if ans_out.get("error"):
            st.error("⚠️ Service temporarily busy. Please click Get Answer again.")
        else:
            if not ans_out["confidence_passed"]:
                st.warning(
                    "🛡️ Question outside the scope of the available metallurgical documents.")
            st.markdown(ans_out["answer"])
            render_figure_gallery(
                ans_out["linked_images"], "🖼️ Figures, Graphs & Visual Explanations from Cited Pages")
            with st.expander("📑 View Cited Source Excerpts"):
                for idx, item in enumerate(ans_out["retrieved_items"], start=1):
                    c = item["chunk"]
                    st.markdown(
                        f"**Source #{idx}: `{c['metadata']['source']}` ({c['metadata']['section']})**")
                    st.info(c["text"][:600] +
                            ("..." if len(c["text"]) > 600 else ""))
