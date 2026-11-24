"""
demo_app.py -- Research & Demonstration Platform
Lightweight Topology-Aware MobileViT Network for Rural Road Extraction
Center of Excellence in Artificial Intelligence and Remote Sensing
"""

import os
import sys
import time
import numpy as np
import cv2
import networkx as nx
import streamlit as st

# Set project root
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.src.utils.graph_postprocess import connect_canopy_gaps, hysteresis_threshold
from backend.src.utils.graph_builder import build_graph_from_skeleton, simplify_graph
from backend.src.utils.layer_inspector import MultiStageInspector, STAGE_METADATA

try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

import onnxruntime as ort

# Streamlit Page Configuration - Professional Academic Presentation
st.set_page_config(
    page_title="Topological Rural Road Extraction | Research Platform",
    page_icon="️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Institutional Header */
    .institute-bar {
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .institute-name {
        font-size: 0.84rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
    }
    .project-status {
        font-size: 0.74rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        background-color: #0F172A;
        border: 1px solid #3B82F6;
        color: #60A5FA;
        letter-spacing: 0.04em;
    }

    /* Title Block */
    .title-text {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .subtitle-text {
        font-size: 1.0rem;
        font-weight: 400;
        color: #94A3B8;
        margin-bottom: 18px;
        line-height: 1.55;
    }

    /* Professional Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .metric-box {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-left: 3px solid #2563EB;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .metric-box.accent {
        border-left-color: #059669;
    }
    .metric-box.purple {
        border-left-color: #8B5CF6;
    }
    .metric-box.gold {
        border-left-color: #D97706;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F1F5F9;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.76rem;
        font-weight: 500;
        color: #10B981;
        margin-top: 4px;
    }
    .metric-sub.neutral {
        color: #94A3B8;
    }

    /* Status Bar */
    .status-strip {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 10px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .status-info {
        font-size: 0.85rem;
        color: #CBD5E1;
    }
    .status-badge {
        font-size: 0.75rem;
        font-weight: 600;
        background: #064E3B;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #059669;
    }

    /* Section Headers */
    .panel-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 4px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1E293B;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Feature Cards */
    .feature-card {
        background: #0B132B;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
        height: 100%;
    }
    .feature-card h4 {
        margin-top: 0;
        font-size: 1.05rem;
        color: #60A5FA;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .feature-card p {
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    /* Video Teleprompter Card */
    .teleprompter-box {
        background: #111827;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 14px;
    }
    .teleprompter-time {
        font-size: 0.75rem;
        font-weight: 700;
        color: #60A5FA;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .teleprompter-text {
        font-size: 0.90rem;
        color: #E2E8F0;
        margin-top: 4px;
        line-height: 1.5;
    }

    /* Academic Tables */
    .academic-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        margin-top: 10px;
    }
    .academic-table th {
        background-color: #1E293B;
        color: #E2E8F0;
        text-align: left;
        padding: 10px 14px;
        font-weight: 600;
        border-bottom: 2px solid #334155;
    }
    .academic-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #1E293B;
        color: #CBD5E1;
    }
    .academic-table tr:hover {
        background-color: #0F172A;
    }
    .highlight-row {
        background-color: rgba(37, 99, 235, 0.12) !important;
        font-weight: 600;
        color: #FFFFFF !important;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 6px;
    }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-purple { background: rgba(139, 92, 246, 0.2); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.4); }
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
</style>
""", unsafe_allow_html=True)

# Normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def show_img(image, caption=None):
    """Image renderer supporting width='stretch' with fallback."""
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)

COLORMAP_CHOICES = {
    "Viridis (Standard Scientific)": cv2.COLORMAP_VIRIDIS,
    "Turbo (High Dynamic Contrast)": cv2.COLORMAP_TURBO,
    "Plasma (High Intensity)": cv2.COLORMAP_PLASMA,
    "Inferno (Thermal Road Flow)": cv2.COLORMAP_INFERNO,
    "Magma (Deep Feature Map)": cv2.COLORMAP_MAGMA,
    "Jet (Classic Spectral)": cv2.COLORMAP_JET,
}

# --- CACHED MODEL ENGINE INITIALIZATION ---
@st.cache_resource
def get_layer_inspector():
    model_path = os.path.join(REPO_ROOT, "models", "mobilevit_v2.onnx")
    if not os.path.exists(model_path):
        return None
    return MultiStageInspector(model_path)

@st.cache_resource
def get_standard_onnx_session():
    model_path = os.path.join(REPO_ROOT, "models", "mobilevit_v2.onnx")
    if not os.path.exists(model_path):
        return None, None, None
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    return session, in_name, out_name


def preprocess_image(image_bgr, target_dim=512):
    """Normalize and format optical image for ONNX model."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    if target_dim is not None:
        new_h, new_w = target_dim, target_dim
    else:
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    chw = normed.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(chw), resized, new_h, new_w


def generate_overlay(rgb, mask, color=(0, 230, 118), alpha=0.55):
    """Generate high-contrast road boundary overlay."""
    overlay = rgb.copy()
    idx = mask > 0
    color_arr = np.array(color, dtype=np.uint8)
    overlay[idx] = ((1 - alpha) * overlay[idx] + alpha * color_arr).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay


# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### ️ Demonstration Studio")
st.sidebar.caption("Rural Road Extraction Research Engine")

# Sample scenes catalog
sample_dir = os.path.join(REPO_ROOT, "data", "samples")
sample_catalog = {
    "117991_sat.jpg": "Scene 1: Complex Road Junction & Village (ID: 117991)",
    "115714_sat.jpg": "Scene 2: Dense Forest Canopy Corridor (ID: 115714)",
    "100034_sat.jpg": "Scene 3: Rural Unpaved Arterial Track (ID: 100034)",
    "102408_sat.jpg": "Scene 4: Agricultural Terrain with Foliage (ID: 102408)",
    "100892_sat.jpg": "Scene 5: Interlocking Agricultural Trails (ID: 100892)",
    "139076_sat.jpg": "Scene 6: Mountainous Winding Ridge Track (ID: 139076)",
    "174171_sat.jpg": "Scene 7: Rural Settlement Connectivity (ID: 174171)",
    "195274_sat.jpg": "Scene 8: Fragmented Canopy Crossing (ID: 195274)",
}

input_mode = st.sidebar.radio("Data Source", ["Benchmark Satellite Tiles", "Upload Custom Tile"])

current_image_key = None
active_image_bgr = None

if input_mode == "Benchmark Satellite Tiles":
    selected_name = st.sidebar.selectbox(
        "Select Satellite Tile",
        options=list(sample_catalog.keys()),
        format_func=lambda k: sample_catalog[k],
        index=0
    )
    current_image_key = selected_name
    img_path = os.path.join(sample_dir, selected_name)
    if os.path.exists(img_path):
        active_image_bgr = cv2.imread(img_path)
    else:
        st.sidebar.error(f"File not found: {img_path}")
else:
    uploaded = st.sidebar.file_uploader("Upload Optical Raster (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif"])
    if uploaded is not None:
        raw_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        active_image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        current_image_key = f"upload_{uploaded.name}_{uploaded.size}"

st.sidebar.markdown("---")
st.sidebar.markdown("### ️ Processing Controls")

res_choice = st.sidebar.select_slider(
    "Inference Resolution",
    options=[256, 512, 1024],
    value=512,
    help="512x512 is recommended for instantaneous interactive demonstrations (~170ms); 1024x1024 runs full paper resolution."
)

enable_gap_bridging = st.sidebar.checkbox("Centerline Gap Bridging (Morphological Graph)", value=True)
enable_tta = st.sidebar.checkbox("Test-Time Augmentation (4-Flip TTA)", value=True)

with st.sidebar.expander("Threshold Tuning & Post-Processing"):
    high_threshold = st.slider("High Confidence Threshold", 0.10, 0.80, 0.35, 0.05)
    low_threshold = st.slider("Hysteresis Low Threshold", 0.05, 0.40, 0.12, 0.01)
    max_gap_distance = st.slider("Max Canopy Bridge Distance (px)", 50, 300, 220, 10)

st.sidebar.markdown("---")
show_teleprompter = st.sidebar.checkbox("️ Show Video Presenter Script / Notes", value=False)
st.sidebar.caption("Center of Excellence in AI & Remote Sensing\nTarget: National Rural Road Audit (PMGSY)")


# --- TOP MAIN HEADER ---
st.markdown("""
<div class="institute-bar">
    <div class="institute-name">Center of Excellence in Artificial Intelligence & Remote Sensing</div>
    <div class="project-status">IEEE Research Evaluation Platform</div>
</div>
<div class="title-text">Lightweight Topology-Aware MobileViT Network for Rural Road Extraction</div>
<div class="subtitle-text">
    A 1.60-Million parameter edge network combining MobileViT v2 linear attention, Strip Convolutions, Channel Shift, and centerline-Dice (clDice) loss to extract connected road networks under dense forest canopy.
</div>
""", unsafe_allow_html=True)

# Top Metric Cards
st.markdown("""
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">Model Parameters</div>
        <div class="metric-val">1.60 M</div>
        <div class="metric-sub">94.8% reduction vs U-Net (31.0M)</div>
    </div>
    <div class="metric-box purple">
        <div class="metric-label">Deployment Footprint</div>
        <div class="metric-val">6.2 MB</div>
        <div class="metric-sub neutral">ONNX Runtime on Laptop CPU (1.27s)</div>
    </div>
    <div class="metric-box accent">
        <div class="metric-label">Test IoU (623 tiles)</div>
        <div class="metric-val">61.1%</div>
        <div class="metric-sub">4-flip TTA, 95% CI [60.0 - 62.1]</div>
    </div>
    <div class="metric-box gold">
        <div class="metric-label">clDice Topology Overlap</div>
        <div class="metric-val">84.5%</div>
        <div class="metric-sub">Relaxed F1: 86.7%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Optional Presenter Teleprompter Drawer
if show_teleprompter:
    st.markdown('''
    <div class="teleprompter-box">
        <div class="teleprompter-time">PRESENTER CHEAT SHEET / SCENE 4 DEMONSTRATION TIMELINE</div>
        <div class="teleprompter-text">
            <b>0:00 - 0:45 (Problem Statement - Tab 1)</b>: Select Scene 4 (ID: 102408). Introduce rural road extraction: unpaved tracks blend with soil, tree canopies cause broken disconnections, and heavy models (>30M params) cannot run on district laptops without GPUs.<br>
            <b>0:45 - 2:00 (Methodology & Novel Contributions - Tab 2)</b>: Present our 5 core novel contributions as published in the paper: (1) First sub-2M parameter edge network (1,604,657 params, 6.2 MB) uniting MobileViT v2 linear attention, strip convs, and attention gates; (2) Connectivity-aware clDice training with our novel collapse prevention guard; (3) Tangent-guided gap bridging healing tree breaks up to 220 px; (4) Native 1024x1024 benchmark on 623 unseen test tiles; (5) Graph resilience analysis for PMGSY infrastructure.<br>
            <b>2:00 - 2:45 (Architecture Flow - Tab 3)</b>: Explain the asymmetric encoder-decoder. Highlight how Attention Gates dynamically suppress background agricultural ridges and foliage before features reach the decoder.<br>
            <b>2:45 - 4:15 (Interactive Inspection on Scene 4 - Tab 4)</b>: Demonstrate Scene 4 layer progression: Stem (strip convs picking up linear traces) &rarr; Bottleneck (global transformer attention) &rarr; Gate 2 (filtering surrounding vegetation) &rarr; Soft probability &rarr; Post-processing bridging canopy gaps to restore connectivity.<br>
            <b>4:15 - 5:15 (PMGSY Graph Resilience & Benchmarks - Tabs 5 & 6)</b>: Show the planar graph for Scene 4. Highlight that Scene 4 has the lowest resilience index (R = 0.61) in our paper, identifying critical single-point bottleneck junctions. Conclude with Table 4 results: 61.1% IoU, 86.7% Relaxed F1, 1.27s CPU latency.
        </div>
    </div>
    ''', unsafe_allow_html=True)


# --- NAVIGATION TABS ---
tab_problem, tab_methodology, tab_architecture, tab_layer_inspector, tab_graph, tab_benchmarks = st.tabs([
    "1. Problem Statement",
    "2. Methodology & MobileViT",
    "3. Network Architecture",
    "4. Interactive Layer-by-Layer Processing",
    "5. Spatial Graph & PMGSY Audit",
    "6. Benchmark Results"
])


# ==============================================================================
# TAB 1: PROBLEM STATEMENT
# ==============================================================================
with tab_problem:
    st.markdown('<div class="panel-header">The Rural Road Extraction Crisis in Satellite Remote Sensing</div>', unsafe_allow_html=True)
    
    col_p1, col_p2 = st.columns([1.1, 0.9])
    with col_p1:
        st.markdown("""
        In developing nations, rural road networks under initiatives such as **India's Pradhan Mantri Gram Sadak Yojana (PMGSY)** form the lifeline for over 800 million citizens, connecting remote habitations to hospitals, schools, and markets.
        
        However, automated road mapping from 0.5m optical satellite imagery faces severe real-world bottlenecks:
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="feature-card">
                <h4> 1. Dense Canopy Occlusion</h4>
                <p>Tropical tree cover and overhang physically block optical line-of-sight. Conventional models treat occluded segments as non-road, creating catastrophic disconnections in the vector network.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card">
                <h4> 2. Unpaved Dirt & Gravel Tracks</h4>
                <p>Unlike urban multilane highways, rural roads lack lane markings and asphalt contrast. Spectral signatures of dirt roads overlap almost identically with dry agricultural fields.</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="feature-card">
                <h4> 3. Extreme Aspect Ratio Topology</h4>
                <p>Roads occupy less than <b>3.8% of pixels</b> in a tile, with widths of 2-6 px but lengths of thousands of pixels. A single 1-pixel break invalidates shortest-path routing for emergency ambulances.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card">
                <h4> 4. The Edge Deployment Barrier</h4>
                <p>Standard Vision Transformers require quadratic memory <b>O(N²)</b>. Classic baselines (U-Net, D-LinkNet) exceed 31M parameters (118 MB), requiring high-end server GPUs that cannot run on field laptops or inspection drones.</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col_p2:
        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; padding:16px;">
            <div style="font-size:0.92rem; font-weight:700; color:#F8FAFC; margin-bottom:8px;">
                Critical Insight: Why Pixel IoU Fails for Road Networks
            </div>
            <p style="font-size:0.85rem; color:#94A3B8; line-height:1.5;">
                In traditional semantic segmentation, intersection-over-union (IoU) measures pure area overlap. A model that predicts 98% of road pixels but leaves a 2-pixel break achieves <b>98% IoU</b>, yet an autonomous vehicle or disaster relief convoy cannot cross the disconnected gap.
            </p>
            <div style="text-align:center; padding:12px; background:#1E293B; border-radius:6px; margin-top:8px;">
                <span style="color:#EF4444; font-weight:700; font-size:1.0rem;">High Pixel IoU ≠ Navigable Road Network</span><br>
                <span style="color:#10B981; font-weight:600; font-size:0.85rem;">Topology-Aware clDice Loss + Edge MobileViT solves this.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists("docs/paper/figures/fig_pipeline_stages.png"):
            st.image("docs/paper/figures/fig_pipeline_stages.png",
                     caption="Topological fragmentation: Raw probability vs hysteresis healing and tangent gap bridging.")


# ==============================================================================
# TAB 2: METHODOLOGY & MOBILEVIT
# ==============================================================================
with tab_methodology:
    st.markdown('<div class="panel-header">Methodology: Pipeline Design & Training Formulation</div>', unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### 1. Separable Linear Self-Attention (MobileViT v2 Core)")
        st.markdown(
            "Standard Multi-Head Attention scales quadratically $\\mathcal{O}(N^2 \\cdot d)$, which requires gigabytes of VRAM and fails on laptop CPUs. "
            "Our encoder uses MobileViT v2 linear separable attention scaling linearly $\\mathcal{O}(N \\cdot d)$:"
        )
        st.latex(r"\text{Context} = \sum_{j=1}^N \text{Softmax}(K)_j \cdot V_j, \quad \text{Attn}(Q) = \text{Context} \odot Q")
        st.markdown(
            "**Deployment Impact:** Enables global receptive field reasoning across the entire 1024&times;1024 satellite tile to connect separated road corridors without needing a GPU."
        )
        
        st.markdown("---")
        st.markdown("#### 2. Directional Strip Convolutions (1&times;3 &rarr; 3&times;1)")
        st.markdown(
            "Adopting the directional strip convolution concept from HPLNet (Cui et al., 2025), standard 3&times;3 isotropic convolutions are factorized into sequential 1D filters:"
        )
        st.latex(r"K_{3\times 3} \;\longrightarrow\; K_{1\times 3} \;\circ\; K_{3\times 1}")
        st.markdown(
            "**Deployment Impact:** Scans along elongated linear road geometry rather than isotropic squares, while cutting kernel parameters from 9 to 6 per channel pair (**33.3% parameter reduction**)."
        )

    with m2:
        st.markdown("#### 3. Zero-Parameter Channel Shift Operator")
        st.markdown(
            "Before each MobileViT block, 25% of feature channels are physically displaced by 2 pixels in the four cardinal directions (up, down, left, right):"
        )
        st.latex(r"Z_{\text{shift}} = \left[Z_0[\uparrow 2], \; Z_1[\downarrow 2], \; Z_2[\leftarrow 2], \; Z_3[\rightarrow 2], \; Z_{\text{pass}}\right]")
        st.markdown(
            "**Deployment Impact:** At 1/4 spatial downsampling, a 2-pixel shift grants an effective reach of &plusmn;8 pixels in the original input&mdash;matching the width of a rural road corridor&mdash;at **zero learned parameters and zero FLOPs**."
        )
        
        st.markdown("---")
        st.markdown("#### 4. Differentiable Centerline-Dice (clDice) Loss")
        st.markdown(
            "Pixel-level losses (BCE, Dice) are blind to 1-pixel breaks. To train the network for topological continuity, we incorporate soft centerline Dice (clDice):"
        )
        st.latex(r"\mathcal{L}_{\text{clDice}} = 1 - 2 \times \frac{\text{Prec}_{\text{cl}}(S(P), Y) \times \text{Rec}_{\text{cl}}(P, S(Y))}{\text{Prec}_{\text{cl}} + \text{Rec}_{\text{cl}}}")
        st.markdown(
            "where $S(\\cdot)$ is continuous soft iterative morphological skeletonization via min-max pooling. This heavily penalizes topological disconnection during backpropagation."
        )

    st.markdown("---")
    st.markdown("#### 5. Dynamic Loss Scheduling & Novel Collapse Prevention Guard")
    st.markdown(
        "To balance spatial coverage and topological connectivity, our total training loss is dynamically scheduled:"
    )
    st.latex(r"\mathcal{L}_{\text{total}} = \alpha(e)\,\mathcal{L}_{\text{BCE}} + 0.35\,\mathcal{L}_{\text{Dice}} + (0.65 - \alpha(e))\,\mathcal{L}_{\text{clDice}}")
    st.markdown("""
    * **Dynamic Cosine Schedule:** $\\alpha(e)$ decays smoothly from 0.50 to 0.15 over the first 60 epochs, allowing the network to first locate gross road trajectories before prioritizing topological continuity.
    * **Novel Collapse Prevention Guard:** In early experiments, unconstrained clDice caused catastrophic collapse at epoch 46, where the model predicted 80.9% road pixels to artificially maximize centerline recall. We solved this with a dual-mechanism guard: bounding the positive class weight to 2.0, and automatically rejecting any checkpoint predicting > 20% road pixels on the 280-tile selection set.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    if os.path.exists("docs/paper/figures/fig_system.png"):
        show_img("docs/paper/figures/fig_system.png", caption="System overview: Satellite input, MobileViT network, postprocessing, graph analysis, and soft skeleton loss.")


# TAB 3: NETWORK ARCHITECTURE
# ==============================================================================
with tab_architecture:
    st.markdown('<div class="panel-header">Detailed Model Architecture & Dimensional Flow</div>', unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns([1.1, 0.9])
    with col_a1:
        st.markdown("""
        Our model is structured as an **asymmetric encoder-decoder** with attention-gated skip connections, operating on dynamic spatial resolutions (256², 512², or 1024²).
        """)
        
        st.markdown("""
        <table class="academic-table">
            <thead>
                <tr><th>Stage</th><th>Block Operator</th><th>Channels</th><th>Resolution (for 512²)</th><th>Function</th></tr>
            </thead>
            <tbody>
                <tr><td><b>Stem</b></td><td>StripConv (1×3 → 3×1), s=2</td><td>32</td><td>256 × 256</td><td>Linear edge scanning & downsampling</td></tr>
                <tr><td><b>Encoder 1</b></td><td>Inverted Residual (IR), s=2</td><td>64</td><td>128 × 128</td><td>Localized texture extraction</td></tr>
                <tr><td><b>Encoder 2</b></td><td>Channel Shift + MobileViT v2 (×2)</td><td>64</td><td>128 × 128</td><td>Linear attention across patches</td></tr>
                <tr><td><b>Encoder 3</b></td><td>Channel Shift + MobileViT v2 (×2)</td><td>96</td><td>64 × 64</td><td>Semantic corridor discrimination</td></tr>
                <tr class="highlight-row"><td><b>Bottleneck</b></td><td>MobileViT v2 (×3), C=128</td><td>128</td><td>32 × 32</td><td>Global scene context & canopy bridging</td></tr>
                <tr><td><b>Gate 3</b></td><td>Attention Gate (ψ, Wg, Wx)</td><td>1</td><td>64 × 64</td><td>Suppresses Encoder 3 foliage noise</td></tr>
                <tr><td><b>Decoder 3</b></td><td>Bilinear Up ×2 + StripConv</td><td>96</td><td>64 × 64</td><td>First spatial reconstruction stage</td></tr>
                <tr><td><b>Gate 2</b></td><td>Attention Gate (ψ, Wg, Wx)</td><td>1</td><td>128 × 128</td><td>Focuses skip features on road junctions</td></tr>
                <tr><td><b>Decoder 2</b></td><td>Bilinear Up ×2 + StripConv</td><td>64</td><td>128 × 128</td><td>Mid-level pathway recovery</td></tr>
                <tr><td><b>Gate 1</b></td><td>Attention Gate (ψ, Wg, Wx)</td><td>1</td><td>256 × 256</td><td>High-res spatial skip boundary gate</td></tr>
                <tr><td><b>Decoder 1</b></td><td>Bilinear Up ×2 + StripConv</td><td>32</td><td>256 × 256</td><td>Boundary sharpening</td></tr>
                <tr class="highlight-row"><td><b>Head</b></td><td>Bilinear Up ×2 + 1×1 Conv + Sigmoid</td><td>1</td><td>512 × 512</td><td>Continuous road probability map P</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
    with col_a2:
        st.markdown("#### Why Attention Gates on Skip Connections?")
        st.markdown("In standard U-Net skip connections, low-level encoder features are directly concatenated with decoder features. In rural satellite scenes, this floods the decoder with false-positive edges from agricultural ridges, trees, and rivers.\n\nOur **Attention Gates** compute a soft spatial coefficient map $\alpha \in [0, 1]$ conditioned on the coarse gating signal from the deeper layer:")
        st.latex(r"\alpha = \sigma\left(\psi^T \cdot \text{SiLU}(W_g \cdot g + W_x \cdot x)\right)")
        st.markdown("This dynamically mutes non-road foliage before concatenation occurs.")

    st.markdown("<br>", unsafe_allow_html=True)
    if os.path.exists("docs/paper/figures/fig_network.png"):
        show_img("docs/paper/figures/fig_network.png", caption="Full network architecture: (a) Strip convolution, (b) Channel shift, (c) Encoder-decoder layout.")


# ==============================================================================
# TAB 4: INTERACTIVE LAYER-BY-LAYER PROCESSING (STAR HIGHLIGHT)
# ==============================================================================
with tab_layer_inspector:
    st.markdown("""
    <div class="panel-header">
        <span>Live Neural Network Layer-by-Layer Feature Inspector</span>
        <span class="badge-pill badge-green">REAL-TIME MULTI-OUTPUT ONNX ENGINE</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This section lets you inspect **exactly what happens inside the neural network at each stage**.
    Select a target satellite scene, step through each layer of the model, and observe how raw pixels are progressively transformed into abstract linear attention features, filtered by attention gates, and reconstructed into continuous road vectors.
    """)

    if active_image_bgr is None:
        st.warning("Please select or upload a satellite image from the left sidebar to activate the layer inspector.")
    else:
        # Preprocess input image
        input_tensor, current_resized_rgb, img_h, img_w = preprocess_image(active_image_bgr, target_dim=res_choice)
        
        inspector = get_layer_inspector()
        if inspector is None:
            st.error("Model file `models/mobilevit_v2.onnx` not found.")
        else:
            # Session state caching for multi-stage outputs
            cache_key_layers = f"layers_{current_image_key}_{res_choice}"
            if cache_key_layers not in st.session_state:
                with st.spinner(f"Running multi-output layer extraction on {res_choice}×{res_choice} tile..."):
                    layer_dict, layer_dur = inspector.run_multi_stage_inference(input_tensor)
                    st.session_state[cache_key_layers] = (layer_dict, layer_dur)

            layer_dict, layer_dur = st.session_state[cache_key_layers]

            # Stage selector navigation
            stage_names = [s["title"] for s in STAGE_METADATA]
            
            # Postprocessing extra stages
            post_stages = [
                "Stage 13: Hysteresis Thresholding (Th=0.35, Tl=0.12)",
                "Stage 14: Centerline Gap Healing (Canopy Bridging)",
                "Stage 15: Medial-Axis Skeleton (1-Pixel Graph)"
            ]
            all_stage_titles = stage_names + post_stages

            # Navigation UI
            nav_c1, nav_c2, nav_c3 = st.columns([1.5, 3.5, 1.5])
            
            if "current_stage_idx" not in st.session_state:
                st.session_state["current_stage_idx"] = 0

            with nav_c1:
                if st.button("Previous Layer", use_container_width=True):
                    st.session_state["current_stage_idx"] = max(0, st.session_state["current_stage_idx"] - 1)

            with nav_c3:
                if st.button("Next Layer ️", use_container_width=True):
                    st.session_state["current_stage_idx"] = min(len(all_stage_titles) - 1, st.session_state["current_stage_idx"] + 1)

            with nav_c2:
                selected_stage_title = st.selectbox(
                    "Jump to Layer / Pipeline Stage",
                    options=all_stage_titles,
                    index=st.session_state["current_stage_idx"],
                    label_visibility="collapsed"
                )
                st.session_state["current_stage_idx"] = all_stage_titles.index(selected_stage_title)

            curr_idx = st.session_state["current_stage_idx"]

            # Visual Display Options
            opt_c1, opt_c2, opt_c3, opt_c4 = st.columns(4)
            with opt_c1:
                colormap_name = st.selectbox("Activation Colormap", list(COLORMAP_CHOICES.keys()), index=0)
                selected_cmap = COLORMAP_CHOICES[colormap_name]
            with opt_c2:
                vis_mode = st.selectbox("Feature Channel Projection", ["Maximum Feature Energy (Peak Response)", "L2 Activation Magnitude", "Top Active Feature Channel", "Positive Activation Mean"], index=0)
            with opt_c3:
                overlay_alpha = st.slider("Heatmap Overlay Opacity", 0.10, 0.90, 0.55, 0.05)
            with opt_c4:
                st.metric("Total Layer Latency", f"{layer_dur*1000:.1f} ms", delta="All 12 Layers")

            st.markdown("---")

            # --- STAGE RENDERING LOGIC ---
            if curr_idx < len(STAGE_METADATA):
                meta = STAGE_METADATA[curr_idx]
                node_name = meta["node"]
                
                if node_name == "input_image":
                    # Stage 0: Input Optical Tile
                    st.markdown(f"### {meta['title']}")
                    st.caption(f"Operator: `{meta['type']}` | Shape: `(1, 3, {img_h}, {img_w})`")
                    
                    c_i1, c_i2 = st.columns(2)
                    with c_i1:
                        show_img(current_resized_rgb, caption=f"Original Optical Satellite Tile ({img_w}×{img_h} px)")
                    with c_i2:
                        st.markdown("#### Stage Overview")
                        st.markdown(f"**Description:** {meta['description']}")
                        st.markdown(f"**Remote Sensing Insight:** {meta['insight']}")
                        st.markdown("**Mathematical Formulation:**")
                        st.latex(meta['math'])
                else:
                    feat_arr = layer_dict[node_name]
                    feat_shape = feat_arr.shape
                    
                    # Generate feature visualization
                    rendered_heatmap = inspector.render_feature_visualization(
                        feat_arr,
                        mode=vis_mode,
                        colormap_id=selected_cmap,
                        target_size=(img_w, img_h)
                    )
                    
                    # Generate blended overlay
                    blended_overlay = inspector.create_blended_overlay(
                        current_resized_rgb,
                        rendered_heatmap,
                        alpha=overlay_alpha
                    )

                    st.markdown(f"### {meta['title']}")
                    st.markdown(f"""
                    <span class="badge-pill badge-blue">Operator: {meta['type']}</span>
                    <span class="badge-pill badge-green">Tensor Shape: {list(feat_shape)}</span>
                    <span class="badge-pill badge-purple">Spatial Scale: 1/{int(round(img_h / feat_shape[2]))}</span>
                    <span class="badge-pill badge-amber">Channels: {feat_shape[1]}</span>
                    """, unsafe_allow_html=True)
                    
                    # 3-Column Inspection View
                    v1, v2, v3 = st.columns(3)
                    with v1:
                        st.markdown("**1. Original Satellite Tile**")
                        show_img(current_resized_rgb, caption=f"Target Scene ({img_w}×{img_h})")
                    with v2:
                        st.markdown(f"**2. Layer Feature Heatmap ({vis_mode})**")
                        show_img(rendered_heatmap, caption=f"Activation Response: {meta['node']}")
                    with v3:
                        st.markdown("**3. Satellite + Feature Overlay**")
                        show_img(blended_overlay, caption="Spatial Correlation with Terrain & Canopy")

                    # Individual Filter Bank Explorer
                    if feat_shape[1] > 1:
                        with st.expander(f"Filter Bank Analysis: View Individual Feature Channels ({feat_shape[1]} Total)", expanded=False):
                            st.markdown(f"Inspect individual directional filter responses from `{node_name}`. Each channel represents a distinct learned filter (e.g., horizontal strip scan, vertical scan, soil contrast, canopy edge).")
                            ch_grid = inspector.render_channel_grid(feat_arr, num_channels=8, colormap_id=selected_cmap)
                            if ch_grid:
                                g_cols = st.columns(4)
                                for i, (ch_num, ch_pic, c_min, c_max) in enumerate(ch_grid):
                                    with g_cols[i % 4]:
                                        st.caption(f"Channel {ch_num} (Range: [{c_min:.1f}, {c_max:.1f}])")
                                        show_img(ch_pic)

                    st.markdown("#### Feature Representation & Mathematical Formulation")
                    st.markdown(f"**Architectural Role:** {meta['description']}")
                    st.markdown(f"**Learned Features:** {meta['insight']}")
                    st.markdown("**Layer Equation:**")
                    st.latex(meta['math'])

            else:
                # Postprocessing stages (13, 14, 15)
                prob_raw = layer_dict["road_probability"][0, 0]
                mask_hyst = hysteresis_threshold(prob_raw, high_thresh=high_threshold, low_thresh=low_threshold)
                kernel_close = np.ones((5, 5), np.uint8)
                mask_closed = cv2.morphologyEx(mask_hyst, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                
                if enable_gap_bridging:
                    final_mask = connect_canopy_gaps(
                        mask_closed, max_gap_dist=float(max_gap_distance), max_angle_deg=65.0, road_width=6
                    )
                else:
                    final_mask = mask_closed

                if HAS_SKIMAGE:
                    skel = (skeletonize(final_mask > 0) * 255).astype(np.uint8)
                else:
                    skel = final_mask

                if curr_idx == len(STAGE_METADATA):
                    # Stage 13: Hysteresis Thresholding
                    st.markdown("### Stage 13: Dual-Threshold Hysteresis Binarization")
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.markdown("**Continuous Probability P**")
                        prob_vis = cv2.applyColorMap((prob_raw * 255).astype(np.uint8), selected_cmap)
                        show_img(cv2.cvtColor(prob_vis, cv2.COLOR_BGR2RGB), caption="Raw Sigmoid Output")
                    with p2:
                        st.markdown(f"**Hysteresis Mask (Th={high_threshold}, Tl={low_threshold})**")
                        show_img(mask_hyst, caption="High-Confidence Cores + Connected Low-Threshold Tails")
                    with p3:
                        st.markdown("**Hysteresis Overlay**")
                        show_img(generate_overlay(current_resized_rgb, mask_hyst), caption="Vectorized Boundaries")
                    
                    st.markdown("""
                    <div class="feature-card">
                        <h4> Why Hysteresis Outperforms Single Thresholding</h4>
                        <p>A single threshold of 0.5 discards dimly-illuminated rural roads under tree shadows. Hysteresis accepts pixels with <b>P ≥ 0.35</b> as definite roads, and recursively retains adjacent pixels down to <b>P ≥ 0.12</b> if they are physically connected to a high-confidence road core. This prevents spurious noise while maintaining corridor continuity.</p>
                    </div>
                    """, unsafe_allow_html=True)

                elif curr_idx == len(STAGE_METADATA) + 1:
                    # Stage 14: Centerline Gap Healing
                    st.markdown("### Stage 14: Tangent-Guided Canopy Gap Healing")
                    g1, g2, g3 = st.columns(3)
                    with g1:
                        st.markdown("**Before Gap Healing (Raw Thresholded)**")
                        show_img(mask_closed, caption="Breaks present under heavy tree canopy")
                    with g2:
                        st.markdown("**After Tangent-Guided Healing**")
                        show_img(final_mask, caption=f"Healed gaps up to {max_gap_distance} px (θ ≤ 65°)")
                    with g3:
                        st.markdown("**Final Healed Overlay**")
                        show_img(generate_overlay(current_resized_rgb, final_mask), caption="Fully Connected Road Vector Network")

                    st.markdown("""
                    <div class="feature-card">
                        <h4> Topological Gap Bridging Algorithm</h4>
                        <p>Even with deep linear attention, heavy continuous forest canopy (over 50-150 meters wide) can cause local breaks in the optical probability surface. Our morphological graph postprocessor identifies dead-end terminal points, computes tangent vectors at road endpoints, and bridges colinear endpoints within distance <b>d ≤ 220 px</b> and angular deviation <b>θ ≤ 65°</b>.</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    # Stage 15: Medial-Axis Skeleton
                    st.markdown("### Stage 15: 1-Pixel Medial Axis Topological Skeleton")
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.markdown("**Road Binary Polygon**")
                        show_img(final_mask, caption="Segmented Road Footprint")
                    with s2:
                        st.markdown("**1-Pixel Centerline Skeleton**")
                        show_img(skel, caption="Skeletonized Centerline Network")
                    with s3:
                        st.markdown("**Centerline Traces on Imagery**")
                        skel_overlay = current_resized_rgb.copy()
                        skel_overlay[skel > 0] = [255, 50, 50]  # Red centerline
                        show_img(skel_overlay, caption="1-Pixel Navigation Centerlines")

                    st.markdown("""
                    <div class="feature-card">
                        <h4> Topological Significance for Autonomous Routing</h4>
                        <p>The 1-pixel skeleton extracts the mathematical centerline of each road segment. This representation strips away road width variations, producing a pure topological graph ready for graph traversal, GPS trajectory matching, and Dijkstra/A* routing.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # --- FULL PANORAMIC PIPELINE GALLERY ---
            st.markdown("---")
            with st.expander("Panoramic Pipeline Evolution: Input to Feature Extraction to Topology", expanded=False):
                st.markdown("A synchronized progression showing key milestones simultaneously:")
                
                gal_cols = st.columns(8)
                with gal_cols[0]:
                    st.caption("1. Input Satellite")
                    show_img(current_resized_rgb)
                with gal_cols[1]:
                    st.caption("2. Stem (Strip Conv)")
                    show_img(inspector.render_feature_visualization(layer_dict["/m/stem/act_1/Mul_output_0"], mode="Maximum Feature Energy (Peak Response)", colormap_id=selected_cmap, target_size=(256, 256)))
                with gal_cols[2]:
                    st.caption("3. Encoder 2 (MViT)")
                    show_img(inspector.render_feature_visualization(layer_dict["/m/enc2_mvit/fusion/fusion.2/Mul_output_0"], mode="Maximum Feature Energy (Peak Response)", colormap_id=selected_cmap, target_size=(256, 256)))
                with gal_cols[3]:
                    st.caption("4. Bottleneck Core")
                    show_img(inspector.render_feature_visualization(layer_dict["/m/bottleneck/fusion/fusion.2/Mul_output_0"], mode="Maximum Feature Energy (Peak Response)", colormap_id=selected_cmap, target_size=(256, 256)))
                with gal_cols[4]:
                    st.caption("5. Gate 2 (Attention)")
                    show_img(inspector.render_feature_visualization(layer_dict["/m/gate2/psi/psi.2/Sigmoid_output_0"], colormap_id=selected_cmap, target_size=(256, 256)))
                with gal_cols[5]:
                    st.caption("6. Decoder 1 (Synthesis)")
                    show_img(inspector.render_feature_visualization(layer_dict["/m/dec1/act_1/Mul_output_0"], mode="Maximum Feature Energy (Peak Response)", colormap_id=selected_cmap, target_size=(256, 256)))
                with gal_cols[6]:
                    st.caption("7. Soft Probability")
                    p_vis = cv2.applyColorMap((np.clip(layer_dict["road_probability"][0, 0] * 255.0, 0, 255)).astype(np.uint8), selected_cmap)
                    show_img(cv2.cvtColor(p_vis, cv2.COLOR_BGR2RGB))
                with gal_cols[7]:
                    st.caption("8. Final Road Overlay")
                    prob_end = layer_dict["road_probability"][0, 0]
                    m_end = hysteresis_threshold(prob_end, high_thresh=high_threshold, low_thresh=low_threshold)
                    show_img(generate_overlay(current_resized_rgb, m_end))


# ==============================================================================
# TAB 5: SPATIAL GRAPH & PMGSY INFRASTRUCTURE AUDIT
# ==============================================================================
with tab_graph:
    st.markdown("""
    <div class="panel-header">
        <span>PMGSY Infrastructure Auditing & Spatial Graph Resilience</span>
        <span class="badge-pill badge-purple">NATIONAL RURAL ROAD AUDITING</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Under India's **Pradhan Mantri Gram Sadak Yojana (PMGSY)**, automated satellite monitoring provides actionable infrastructure intelligence.
    Here, the predicted road skeleton is converted into a planar graph $G = (V, E)$ to calculate topological resilience and identify critical single points of failure.
    """)

    if active_image_bgr is not None:
        input_tensor, current_resized_rgb, img_h, img_w = preprocess_image(active_image_bgr, target_dim=res_choice)
        inspector = get_layer_inspector()
        
        if inspector is not None:
            layer_dict, _ = inspector.run_multi_stage_inference(input_tensor)
            prob_map = layer_dict["road_probability"][0, 0]
            
            # Post-processing
            mask_hyst = hysteresis_threshold(prob_map, high_thresh=high_threshold, low_thresh=low_threshold)
            mask_closed = cv2.morphologyEx(mask_hyst, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            if enable_gap_bridging:
                final_mask = connect_canopy_gaps(mask_closed, max_gap_dist=float(max_gap_distance), max_angle_deg=65.0)
            else:
                final_mask = mask_closed

            if HAS_SKIMAGE:
                skel = skeletonize(final_mask > 0).astype(np.uint8)
            else:
                skel = (final_mask > 0).astype(np.uint8)

            # Build NetworkX Graph
            with st.spinner("Extracting NetworkX Planar Graph & Betweenness Centrality..."):
                G = build_graph_from_skeleton(skel)
                G_simple = simplify_graph(G, min_length=4)
                num_nodes = len(G_simple.nodes())
                num_edges = len(G_simple.edges())
                
                # Betweenness Centrality
                if num_nodes > 0:
                    bc = nx.betweenness_centrality(G_simple)
                    max_bc = max(bc.values()) if bc else 0.0
                    avg_bc = float(np.mean(list(bc.values()))) if bc else 0.0
                else:
                    bc, max_bc, avg_bc = {}, 0.0, 0.0

            # Graph Metrics Bar
            g_m1, g_m2, g_m3, g_m4 = st.columns(4)
            with g_m1:
                st.metric("Graph Intersections & Terminals", f"{num_nodes} nodes")
            with g_m2:
                st.metric("Traversable Corridors", f"{num_edges} road links")
            with g_m3:
                st.metric("Max Betweenness Centrality", f"{max_bc:.3f}", help="Indicates highest single-point-of-failure criticality.")
            with g_m4:
                resilience_score = 1.0 - max_bc
                st.metric("PMGSY Network Resilience Index", f"{resilience_score*100:.1f} %", delta="Robust" if resilience_score > 0.5 else "Vulnerable")

            # Graph Visualization Canvas
            col_gr1, col_gr2 = st.columns([1.1, 0.9])
            with col_gr1:
                # Render NetworkX Graph directly on Satellite Image
                graph_vis = current_resized_rgb.copy()
                
                # Draw edges
                for u, v, data in G_simple.edges(data=True):
                    pixels = data.get("pixels", [])
                    if len(pixels) > 1:
                        pts = np.array([[p[1], p[0]] for p in pixels], dtype=np.int32)
                        cv2.polylines(graph_vis, [pts], isClosed=False, color=(0, 230, 118), thickness=2)
                    else:
                        cv2.line(graph_vis, (u[1], u[0]), (v[1], v[0]), color=(0, 230, 118), thickness=2)

                # Draw nodes with centrality color
                for node, centrality in bc.items():
                    ny, nx_coord = node
                    # Color from cyan (low) to bright red (high bottleneck criticality)
                    r = int(centrality * 255 / (max_bc + 1e-6))
                    b = 255 - r
                    node_color = (r, 50, b)
                    radius = 4 if centrality < 0.2 else 7
                    cv2.circle(graph_vis, (nx_coord, ny), radius, node_color, -1)
                    cv2.circle(graph_vis, (nx_coord, ny), radius + 1, (255, 255, 255), 1)

                show_img(graph_vis, caption="Topological Planar Graph: Green links = Traversable Roads; Red/Purple Nodes = High-Criticality Bottlenecks")

            with col_gr2:
                st.markdown("""
                <div class="feature-card">
                    <h4> PMGSY Disaster & Emergency Audit</h4>
                    <p>
                        <b>1. Monsoon Washout Vulnerability:</b><br>
                        Nodes with extreme Betweenness Centrality ($C_B > 0.3$) represent single bridges or lone access corridors. If washed out during monsoons, upstream villages are completely cut off from medical relief.
                    </p>
                    <p>
                        <b>2. Autonomous UAV Path Planning:</b><br>
                        The generated NetworkX graph can be directly exported to GeoJSON or GPS waypoints for autonomous drone delivery of medical supplies.
                    </p>
                    <p>
                        <b>3. Rural Connectivity Audit:</b><br>
                        Identifies missing road segments required to meet PMGSY all-weather connectivity targets.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if os.path.exists("docs/paper/figures/fig_graph_analysis.png"):
                    show_img("docs/paper/figures/fig_graph_analysis.png", caption="Paper analysis: Betweenness centrality & network resilience under edge attack.")


# ==============================================================================
# TAB 6: BENCHMARK RESULTS
# ==============================================================================
with tab_benchmarks:
    st.markdown('<div class="panel-header">Rigorous Empirical Evaluation (v4 Model on 623 DeepGlobe Test Tiles)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    All metrics below are measured on **623 held-out DeepGlobe test tiles** at full 1024×1024 resolution. These tiles were never seen during training or epoch selection (`figures/real/v4/paper_results/results.json`).
    """)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("""
        **Inference Settings & Accuracy Breakdown**
        <table class="academic-table">
            <thead>
                <tr><th>Setting (v4 Model)</th><th>IoU (%)</th><th>F1 (%)</th><th>Relaxed F1 (%)</th><th>clDice (%)</th><th>Road Parts / Tile</th></tr>
            </thead>
            <tbody>
                <tr><td>Single Pass, Th=0.5</td><td>59.8</td><td>74.8</td><td>85.9</td><td>84.2</td><td>17</td></tr>
                <tr class="highlight-row"><td><b>4-Flip TTA, Th=0.5</b></td><td><b>61.1</b></td><td><b>75.8</b></td><td><b>86.7</b></td><td><b>84.5</b></td><td>13</td></tr>
                <tr><td>4-Flip TTA + Hysteresis + Gap Bridging</td><td>52.6</td><td>69.0</td><td>80.7</td><td>75.8</td><td><b>2</b> (GT: 2)</td></tr>
            </tbody>
        </table>
        <p style="font-size:0.80rem; color:#94A3B8; margin-top:6px;">
            * Note: Full postprocessing achieves the exact ground truth topology count (2 connected road parts per tile), raising canopy recall from 60.3% to 75.1%.
        </p>
        """, unsafe_allow_html=True)

    with b_col2:
        st.markdown("""
        **Computational Efficiency on Intel Core i5 Laptop CPU**
        <table class="academic-table">
            <thead>
                <tr><th>Architecture</th><th>Parameters</th><th>GFLOPs</th><th>Model Size</th><th>CPU Latency / Tile</th></tr>
            </thead>
            <tbody>
                <tr><td>U-Net (Ronneberger et al.)</td><td>31.04 M</td><td>1541.4</td><td>118.4 MB</td><td>14.75 s</td></tr>
                <tr><td>D-LinkNet34 (Zhou et al.)</td><td>31.10 M</td><td>212.6</td><td>118.6 MB</td><td>2.48 s</td></tr>
                <tr><td>DeepLabV3 MobileNetV3</td><td>11.02 M</td><td>78.6</td><td>42.0 MB</td><td>1.00 s</td></tr>
                <tr><td>LR-ASPP MobileNetV3</td><td>3.22 M</td><td>15.7</td><td>12.3 MB</td><td>0.49 s</td></tr>
                <tr class="highlight-row"><td><b>Proposed (MobileViT v2 + StripConv)</b></td><td><b>1.60 M</b></td><td><b>60.7</b></td><td><b>6.2 MB</b></td><td><b>1.27 s</b></td></tr>
            </tbody>
        </table>
        <p style="font-size:0.80rem; color:#10B981; margin-top:6px;">
            * 94.8% parameter reduction vs U-Net with superior topological connectivity.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        if os.path.exists("figures/real/v4/paper_results/figures/fig_eval_qualitative.png"):
            show_img("figures/real/v4/paper_results/figures/fig_eval_qualitative.png",
                     caption="Qualitative evaluation on unseen test tiles: Raw imagery, Ground Truth, and v4 Prediction.")
    with comp_c2:
        if os.path.exists("figures/real/v4/paper_results/figures/fig_eval_ablation.png"):
            show_img("figures/real/v4/paper_results/figures/fig_eval_ablation.png",
                     caption="Ablation study: Impact of clDice loss, Strip Convolutions, Channel Shift, and Attention Gates.")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748B; font-size:0.82rem; padding-bottom:12px;">
    Center of Excellence in Artificial Intelligence and Remote Sensing • IEEE Research Project Demonstration Studio • MobileViT v2 Rural Road Extraction
</div>
""", unsafe_allow_html=True)
