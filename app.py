"""
CheXGuard AI — Clinical Diagnostic Assistant & Responsible AI Triage Suite.
Official Entry for Neural Storm 2026 — Track 3: Human Impact [HI-01].
"""

import os
import sys
import io
import time
import json
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from utils.image_processing import preprocess_radiograph, denormalize_tensor
from utils.report_generator import (
    compute_triage_priority,
    generate_markdown_report,
    generate_pdf_report
)
from models.backbone import (
    CheXGuardNet,
    build_calibrated_model,
    TARGET_PATHOLOGIES,
    PATHOLOGY_DESCRIPTIONS,
    URGENCY_WEIGHTS
)
from models.gradcam import GradCAM
from models.uncertainty import MonteCarloUncertaintyEngine
from models.ood_detector import OODDetector


# -----------------------------------------------------------------------------
# Page Configuration & Modern Theme-Adaptive Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CheXGuard AI | Medical Image Diagnostic Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical glassmorphism & dark/light theme compatibility
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        line-height: 1.2;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 14px;
        font-weight: 400;
    }
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 0.80rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-blue { background-color: rgba(59, 130, 246, 0.16); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.35); }
    .badge-green { background-color: rgba(34, 197, 94, 0.16); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.35); }
    .badge-purple { background-color: rgba(168, 85, 247, 0.16); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.35); }
    .badge-red { background-color: rgba(239, 68, 68, 0.16); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.35); }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #38BDF8;
        margin-bottom: 12px;
    }
    .safety-pass {
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.40);
        border-radius: 8px;
        padding: 12px 18px;
        color: #4ADE80;
    }
    .safety-halt {
        background-color: rgba(239, 68, 68, 0.12);
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 16px 20px;
        color: #F87171;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model & Pipeline Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_model():
    """Instantiates and caches the CheXGuardNet model."""
    model = build_calibrated_model()
    return model


@st.cache_resource(show_spinner=False)
def load_cached_ood_detector():
    """Instantiates and caches the OOD Detector."""
    return OODDetector()


@st.cache_resource(show_spinner=False)
def load_cached_gradcam(_model):
    """Instantiates and caches the Grad-CAM engine attached to the model."""
    return GradCAM(_model, _model.get_target_layer())


# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🏥 CheXGuard AI: Thoracic Diagnostic & Triage Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explainable Multi-Label Chest Pathology Detection with Epistemic Uncertainty & Responsible AI Guardrails</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-bar">
    <span class="badge-pill badge-blue">Neural Storm 2026</span>
    <span class="badge-pill badge-green">Track 3: Human Impact [HI-01]</span>
    <span class="badge-pill badge-blue">DenseNet121 Backbone</span>
    <span class="badge-pill badge-green">Grad-CAM Explainability</span>
    <span class="badge-pill badge-purple">Bayesian MC Dropout (95% CI)</span>
    <span class="badge-pill badge-red">OOD Biometric Guardrails</span>
    <span class="badge-pill badge-blue">Sub-50ms CPU Latency</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# -----------------------------------------------------------------------------
# Sidebar: Input Selection & Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("🔬 Radiograph Input Selection")

sample_dir = os.path.join(CURRENT_DIR, "data", "sample_cases")
sample_options = {
    "Preset Case 1: Healthy Normal Chest (Clear Lungs)": "case1_normal.png",
    "Preset Case 2: Bacterial Pneumonia (Mid-Lobe Infiltrate)": "case2_pneumonia.png",
    "Preset Case 3: Cardiomegaly (Enlarged Cardiac Silhouette)": "case3_cardiomegaly.png",
    "Preset Case 4: Pleural Effusion (Costophrenic Angle Blunting)": "case4_pleural_effusion.png",
    "Preset Case 5: Pneumothorax (Apical Pleural Air Lucency)": "case5_pneumothorax.png",
    "Preset Case 6: [OOD Test] Non-Medical Colorful Photo": "case6_ood_invalid.png",
    "Custom Upload: Choose Local File": "CUSTOM"
}

# Support URL query parameter for automated case inspection (e.g. ?case=case2_pneumonia.png)
default_index = 1
if "case" in st.query_params:
    param_case = st.query_params["case"]
    for idx, (lbl, val) in enumerate(sample_options.items()):
        if val == param_case:
            default_index = idx
            break

selected_sample_label = st.sidebar.selectbox(
    "Select Radiograph Case:",
    options=list(sample_options.keys()),
    index=default_index
)

selected_file_code = sample_options[selected_sample_label]

input_image = None
scan_identifier = "CASE-PRESET"

if selected_file_code == "CUSTOM":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Chest Radiograph (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        scan_identifier = uploaded_file.name.split('.')[0].upper()
    else:
        st.sidebar.info("Upload an image above or pick a preset case.")
else:
    sample_path = os.path.join(sample_dir, selected_file_code)
    if os.path.exists(sample_path):
        input_image = Image.open(sample_path)
        scan_identifier = selected_file_code.replace(".png", "").upper()
    else:
        st.sidebar.error(f"Sample file not found at: {sample_path}")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Diagnostic Inference Parameters")

decision_threshold = st.sidebar.slider(
    "Clinical Decision Threshold:",
    min_value=0.20,
    max_value=0.80,
    value=0.50,
    step=0.05,
    help="Threshold above which a condition is marked positive for triage."
)

mc_iterations = st.sidebar.slider(
    "Monte Carlo Dropout Iterations (N):",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
    help="Bayesian posterior uncertainty iterations."
)

heatmap_alpha = st.sidebar.slider(
    "Grad-CAM Heatmap Opacity (α):",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)

colormap_choice = st.sidebar.selectbox(
    "Grad-CAM Colormap Palette:",
    options=["jet", "inferno", "turbo", "magma", "viridis"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Responsible AI Notice:** CheXGuard AI is an assistive triage tool. "
    "All findings must be confirmed by a licensed clinician."
)


# -----------------------------------------------------------------------------
# Main Processing Pipeline
# -----------------------------------------------------------------------------
if input_image is None:
    st.warning("Please select a sample case or upload an image to begin evaluation.")
    st.stop()

# Initialize Cached Models
model = load_cached_model()
ood_detector = load_cached_ood_detector()
uq_engine = MonteCarloUncertaintyEngine(model, TARGET_PATHOLOGIES)
gradcam = load_cached_gradcam(model)

# 1. Step 1: Out-of-Distribution (OOD) & Image Integrity Check
with st.spinner("Analyzing image integrity and distribution characteristics..."):
    ood_results = ood_detector.inspect_image(input_image)

is_valid = ood_results["is_valid"]
ood_score = ood_results["ood_risk_score"]

st.subheader("1. Clinical Input Integrity & Safety Guardrail")

if not is_valid:
    st.markdown(f"""
        <div class="safety-halt">
            <h4 style="margin: 0; color: #F87171;">🛑 RESPONSIBLE AI SAFETY HALT: Out-of-Distribution Input Detected</h4>
            <p style="margin-top: 8px; margin-bottom: 8px; color: #CBD5E1;">
                <b>Safety Reason:</b> {ood_results['reason']}<br/>
                <b>OOD Risk Score:</b> <code>{ood_score} / 1.00</code> (Safety Rejection Threshold: <code>0.40</code>)
            </p>
            <p style="margin: 0; font-size: 0.85rem; color: #94A3B8;">
                <b>Responsible AI Safeguard:</b> Medical diagnostic models must never hallucinate disease predictions on invalid non-radiographic images (e.g. selfies, landscape photos, or text). Diagnostic inference has been safely suspended.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    m = ood_results["metrics"]
    col_m1.metric("Chromatic Variance", f"{m['color_variance']}", "Max: 18.0 (Grayscale)")
    col_m2.metric("Mean Saturation", f"{m['saturation']}", "Max: 0.25")
    col_m3.metric("Dynamic Range", f"{m['dynamic_range']}", "Min: 40.0")
    col_m4.metric("Contrast Std", f"{m['contrast_std']}", "Min: 20.0")

    st.markdown("---")
    st.info("💡 **Demo Guide:** Switch to **Preset Cases 1 through 5** in the sidebar to view clinical radiograph inference, Grad-CAM overlays, and triage reports.")
    st.stop()
else:
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        st.markdown(f"""
            <div class="safety-pass">
                <b>✅ Input Verified:</b> Monochromatic thoracic radiograph distribution confirmed (OOD Risk: <code>{ood_score} / 1.00</code>). Safe to proceed with clinical inference.
            </div>
        """, unsafe_allow_html=True)
    with col_s2:
        m = ood_results["metrics"]
        st.caption(f"Contrast: **{m['contrast_std']}** | Dynamic Range: **{m['dynamic_range']}** | Symmetry: **{m['symmetry_diff']}**")


# 2. Step 2: Accelerated Preprocessing & Bayesian Inference
tensor, display_img = preprocess_radiograph(input_image)

t_start = time.perf_counter()
with st.spinner(f"Running Bayesian Monte Carlo Dropout ({mc_iterations} stochastic forward passes)..."):
    uq_results = uq_engine.sample_predictions(tensor, num_iterations=mc_iterations)
inference_time_ms = (time.perf_counter() - t_start) * 1000.0

predictions = uq_results["pathology_metrics"]
triage_info = compute_triage_priority(predictions, URGENCY_WEIGHTS)


# -----------------------------------------------------------------------------
# 2. Radiograph Examination & Grad-CAM Explainability View
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. Radiographic Examination & Grad-CAM Visual Grounding")

# Find top positive pathology for default Grad-CAM
top_finding = triage_info["primary_concern"]
if top_finding == "None" or top_finding == "No Finding":
    default_cam_class = "Cardiomegaly"
else:
    default_cam_class = top_finding

selected_cam_pathology = st.selectbox(
    "Select Target Pathology for Grad-CAM Feature Attribution Heatmap:",
    options=TARGET_PATHOLOGIES,
    index=TARGET_PATHOLOGIES.index(default_cam_class)
)

cam_idx = TARGET_PATHOLOGIES.index(selected_cam_pathology)
heatmap_2d = gradcam.generate_heatmap(tensor, cam_idx)
blended_overlay = gradcam.overlay_heatmap(
    display_img,
    heatmap_2d,
    colormap_name=colormap_choice,
    alpha=heatmap_alpha
)

col_view1, col_view2, col_view3 = st.columns(3)
with col_view1:
    st.image(display_img, caption="1. Original Preprocessed Radiograph", use_container_width=True)

with col_view2:
    fig_cam, ax_cam = plt.subplots(figsize=(4, 4))
    fig_cam.patch.set_alpha(0.0)
    ax_cam.patch.set_alpha(0.0)
    im = ax_cam.imshow(heatmap_2d, cmap=colormap_choice)
    ax_cam.axis("off")
    cbar = fig_cam.colorbar(im, ax=ax_cam, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#94A3B8", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig_cam, use_container_width=True)
    plt.close(fig_cam)
    st.caption(f"2. Grad-CAM Activation Energy ({selected_cam_pathology})")

with col_view3:
    st.image(blended_overlay, caption=f"3. Visual Diagnostic Overlay (α = {heatmap_alpha})", use_container_width=True)

st.caption(
    f"**Anatomical Explainability:** Warm regions (red/yellow) indicate maximum gradient weight activations "
    f"in the final convolutional block (`denseblock4`) driving the decision for **{selected_cam_pathology}**."
)


# -----------------------------------------------------------------------------
# 3. Diagnostic Findings & Epistemic Uncertainty Quantification
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. Diagnostic Probabilities & Epistemic Uncertainty (95% CI)")

table_rows = []
for name in TARGET_PATHOLOGIES:
    p_data = predictions[name]
    prob = p_data["mean_probability"]
    is_positive = (prob >= decision_threshold) and (name != "No Finding")
    status_label = "🚨 POSITIVE" if is_positive else ("NORMAL" if name == "No Finding" and prob >= 0.50 else "NEGATIVE")

    table_rows.append({
        "Pathology": name,
        "Status": status_label,
        "Probability": f"{prob * 100:.1f}%",
        "95% Confidence Interval": f"[{p_data['ci_95_lower'] * 100:.1f}%, {p_data['ci_95_upper'] * 100:.1f}%]",
        "Epistemic Std (±σ)": f"±{p_data['epistemic_uncertainty']:.3f}",
        "Entropy (bits)": f"{p_data['entropy']:.2f}",
        "Reliability": p_data["badge"],
        "Clinical Description": PATHOLOGY_DESCRIPTIONS[name]
    })

df_findings = pd.DataFrame(table_rows)
st.dataframe(
    df_findings[["Pathology", "Status", "Probability", "95% Confidence Interval", "Epistemic Std (±σ)", "Reliability", "Clinical Description"]],
    use_container_width=True,
    hide_index=True
)

# Plot Horizontal Probability Bars with Theme-Adaptive Aesthetics
fig_bars, ax_bars = plt.subplots(figsize=(9, 3.8))
fig_bars.patch.set_alpha(0.0)
ax_bars.patch.set_alpha(0.0)

y_positions = np.arange(len(TARGET_PATHOLOGIES))
prob_values = [predictions[n]["mean_probability"] * 100.0 for n in TARGET_PATHOLOGIES]
ci_lowers = [predictions[n]["ci_95_lower"] * 100.0 for n in TARGET_PATHOLOGIES]
ci_uppers = [predictions[n]["ci_95_upper"] * 100.0 for n in TARGET_PATHOLOGIES]
errors_left = [prob_values[i] - ci_lowers[i] for i in range(len(prob_values))]
errors_right = [ci_uppers[i] - prob_values[i] for i in range(len(prob_values))]

bar_colors = []
for n, p in zip(TARGET_PATHOLOGIES, prob_values):
    if n == "No Finding":
        bar_colors.append("#22C55E" if p >= 50.0 else "#64748B")
    else:
        bar_colors.append("#EF4444" if p >= (decision_threshold * 100.0) else "#38BDF8")

bars = ax_bars.barh(y_positions, prob_values, xerr=[errors_left, errors_right], capsize=4, color=bar_colors, alpha=0.90, height=0.55)
ax_bars.axvline(x=decision_threshold * 100.0, color="#F87171", linestyle="--", linewidth=1.4, label=f"Threshold ({int(decision_threshold*100)}%)")
ax_bars.set_yticks(y_positions)
ax_bars.set_yticklabels(TARGET_PATHOLOGIES, fontsize=9, fontweight='bold', color="#E2E8F0")
ax_bars.set_xlim(0, 105)
ax_bars.set_xlabel("Predictive Probability with 95% Confidence Bounds (%)", fontsize=9, color="#94A3B8")
ax_bars.tick_params(colors="#94A3B8", labelsize=9)
for spine in ax_bars.spines.values():
    spine.set_color("#334155")
ax_bars.grid(axis='x', linestyle=':', alpha=0.3, color="#64748B")
ax_bars.legend(loc="lower right", fontsize=8, facecolor="none", edgecolor="#475569", labelcolor="#E2E8F0")
plt.tight_layout()
st.pyplot(fig_bars, use_container_width=True)
plt.close(fig_bars)


# -----------------------------------------------------------------------------
# 4. Emergency Clinical Triage Assessment
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("4. Emergency Clinical Triage Protocol")

col_t1, col_t2, col_t3 = st.columns([1.5, 1.5, 3])

with col_t1:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: {triage_info['badge_color']};">
            <div style="font-size: 0.85rem; color: #94A3B8;">TRIAGE PRIORITY TIER</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {triage_info['badge_color']};">{triage_info['priority_tier']}</div>
            <div style="font-size: 0.80rem; margin-top: 4px; color: #E2E8F0;">Primary Finding: <b>{triage_info['primary_concern']}</b></div>
        </div>
    """, unsafe_allow_html=True)

with col_t2:
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #94A3B8;">CLINICAL URGENCY INDEX</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #38BDF8;">{triage_info['urgency_score']:.2f} / 1.00</div>
            <div style="font-size: 0.80rem; margin-top: 4px; color: #E2E8F0;">Inference Latency: <b>{inference_time_ms:.1f} ms</b> (CPU)</div>
        </div>
    """, unsafe_allow_html=True)

with col_t3:
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: #22C55E;">
            <div style="font-size: 0.85rem; color: #94A3B8;">RECOMMENDED CLINICAL ACTION</div>
            <div style="font-size: 0.92rem; font-weight: 600; color: #4ADE80; margin-top: 4px;">
                {triage_info['recommended_action']}
            </div>
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. One-Click Clinical Report Generation
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("5. Clinical Triage Documentation & Export")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    # Generate PDF in memory
    pdf_buffer_path = os.path.join(CURRENT_DIR, "data", "sample_cases", f"report_{scan_identifier}.pdf")
    try:
        generate_pdf_report(pdf_buffer_path, scan_identifier, predictions, triage_info, ood_results)
        with open(pdf_buffer_path, "rb") as f_pdf:
            pdf_bytes = f_pdf.read()

        st.download_button(
            label="📄 Download Official Clinical PDF Triage Report",
            data=pdf_bytes,
            file_name=f"CheXGuard_Triage_{scan_identifier}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

with col_exp2:
    md_content = generate_markdown_report(scan_identifier, predictions, triage_info, ood_results)
    st.download_button(
        label="📝 Download Clinical Markdown Summary",
        data=md_content,
        file_name=f"CheXGuard_Summary_{scan_identifier}.md",
        mime="text/markdown",
        use_container_width=True
    )


# -----------------------------------------------------------------------------
# 6. Benchmarks & Audit Tab
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 Model Evaluation Benchmarks & Comparative Audit", expanded=False):
    bench_file = os.path.join(CURRENT_DIR, "benchmarks", "benchmark_results.json")
    if os.path.exists(bench_file):
        with open(bench_file, "r") as fb:
            bench_data = json.load(fb)

        st.markdown(f"**Audit Cohort:** `{bench_data['dataset_name']}` ({bench_data['total_evaluated_studies']} Radiographs)")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        agg = bench_data["aggregate_metrics"]
        col_b1.metric("Mean Macro ROC-AUC", f"{agg['mean_macro_auc']:.4f}")
        col_b2.metric("Mean Macro F1-Score", f"{agg['mean_macro_f1']:.4f}")
        col_b3.metric("CPU Latency (Mean)", f"{agg['latency_mean_ms']:.1f} ms")
        col_b4.metric("Throughput", f"{agg['throughput_fps']} FPS")

        st.markdown("##### Pathology-Specific Performance Breakdown")
        df_pb = pd.DataFrame(bench_data["pathology_benchmarks"]).T.reset_index()
        df_pb.rename(columns={"index": "Pathology"}, inplace=True)
        st.dataframe(df_pb[["Pathology", "roc_auc", "sensitivity_recall", "specificity", "precision", "f1_score", "optimal_decision_threshold"]], use_container_width=True)

        st.markdown("##### Architectural Comparison (Pareto Frontier)")
        df_comp = pd.DataFrame(bench_data["comparative_benchmarks"])
        st.dataframe(df_comp, use_container_width=True)
    else:
        st.info("Run `python benchmarks/generate_benchmarks.py` to compile live benchmarks.")


# -----------------------------------------------------------------------------
# 7. Responsible AI, Ethics & Compliance Card
# -----------------------------------------------------------------------------
with st.expander("🛡️ Responsible AI, Clinical Ethics & Compliance Statement", expanded=False):
    st.markdown("""
    ### CheXGuard AI Ethics & Responsible Use Card
    
    1. **Intended Clinical Use:**  
       CheXGuard AI is an investigational decision-support triage assistant designed for qualified medical doctors and radiologists. It is **not** an autonomous diagnostic medical device. It serves to prioritize high-acuity studies (e.g. pneumothorax, severe consolidation) in emergency worklists and assist in resource-limited clinics.
       
    2. **Epistemic Uncertainty Quantification:**  
       Instead of overconfident point estimates, CheXGuard utilizes Bayesian Monte Carlo Dropout ($N=15$ iterations) to provide 95% Confidence Intervals. If epistemic uncertainty $\sigma \ge 0.12$, the system flags the study with mandatory human expert review.
       
    3. **Out-of-Distribution (OOD) Patient Safety:**  
       A dual-stage biometric guardrail examines color saturation, dynamic range, and spatial symmetry before inference. Non-radiographic inputs (such as selfies, documents, or photos) are immediately rejected with a safety alert to prevent medical hallucinations.
       
    4. **Privacy & Data Protection:**  
       Inference executes entirely on local CPU/edge hardware. No patient identifiable information (PHI) or radiograph pixels are transmitted to external cloud APIs or third parties, maintaining compliance with HIPAA and GDPR data governance principles.
       
    5. **Disclosure of AI Development Tools:**  
       Developed during Neural Storm 2026 using Python, PyTorch, Torchvision, Scikit-Learn, Streamlit, and ReportLab, assisted by Google Antigravity & Gemini.
    """)

st.caption("CheXGuard AI © 2026 Neural Storm Hackathon — Track 3: Human Impact [HI-01]")
