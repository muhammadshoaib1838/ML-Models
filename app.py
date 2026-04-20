"""
╔══════════════════════════════════════════════════════════════════╗
║              ANIMAL CLASSIFIER AI  –  Streamlit App             ║
║  Model : Classical ML + MobileNetV2 feature extraction           ║
║  Classes: 90 animal categories                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
import tensorflow as tf
import streamlit as st
import numpy as np
import pickle
import os
import time
from PIL import Image, ImageOps
# ── Page config (MUST be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Animal Classifier AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy-import heavy libraries so Streamlit Cloud stays responsive ─
@st.cache_resource(show_spinner=False)
def load_tensorflow():
    """Load TensorFlow/Keras only once and cache it."""
    import tensorflow as tf
    build_feature_extractor()
extract_features()
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import Model
    return tf, MobileNetV2, preprocess_input, Model

@st.cache_resource(show_spinner=False)
def build_feature_extractor():
    """Build MobileNetV2 feature extractor (cached across sessions)."""
    _, MobileNetV2, _, Model = load_tensorflow()
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",          # Global Average Pooling → 1-D feature vector
    )
    base.trainable = False
    extractor = Model(inputs=base.input, outputs=base.output)
    return extractor

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained ML model and label encoder from disk."""
    model_path   = "best_model.pkl"
    encoder_path = "label_encoder.pkl"

    missing = [p for p in [model_path, encoder_path] if not os.path.exists(p)]
    if missing:
        st.error(
            f"⚠️ Missing file(s): **{', '.join(missing)}**  \n"
            "Please make sure `best_model.pkl` and `label_encoder.pkl` are in the "
            "same folder as `app.py` before running the app."
        )
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder

# ── Image pre-processing ────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Resize → RGB → (224,224) → float32 array → MobileNetV2 preprocessing.
    Returns shape: (1, 224, 224, 3)
    """
    _, _, preprocess_input, _ = load_tensorflow()

    # Ensure RGB (handle PNG with alpha, grayscale, etc.)
    img = pil_image.convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.LANCZOS)  # maintains aspect ratio crop

    arr = np.array(img, dtype=np.float32)               # (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)                   # (1, 224, 224, 3)
    arr = preprocess_input(arr)                         # MobileNetV2 normalization
    return arr

def extract_features(preprocessed: np.ndarray) -> np.ndarray:
    """Run MobileNetV2 forward pass → 1-D feature vector (1, 1280)."""
    extractor = build_feature_extractor()
    features  = extractor.predict(preprocessed, verbose=0)  # (1, 1280)
    return features

def predict(features: np.ndarray):
    """
    Run classical ML model on feature vector.
    Returns (predicted_class_name, confidence_score_or_None).
    """
    model, label_encoder = load_artifacts()

    pred_index = model.predict(features)[0]             # integer index

    # Decode label
    if hasattr(label_encoder, "inverse_transform"):
        class_name = label_encoder.inverse_transform([pred_index])[0]
    else:
        class_name = str(pred_index)

    # Confidence: supported by sklearn models with predict_proba
    confidence = None
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(features)[0]   # shape: (n_classes,)
        confidence = float(np.max(proba))

    return class_name, confidence

# ── Custom CSS ──────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        /* ── Global ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
        }

        /* ── Header banner ── */
        .hero-banner {
            background: linear-gradient(135deg, rgba(79,172,254,0.15) 0%, rgba(0,242,254,0.10) 100%);
            border: 1px solid rgba(79,172,254,0.3);
            border-radius: 20px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .hero-banner h1 {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.4rem 0;
        }
        .hero-banner p {
            color: rgba(255,255,255,0.65);
            font-size: 1.05rem;
            margin: 0;
        }

        /* ── Cards ── */
        .glass-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 1.6rem;
            backdrop-filter: blur(12px);
            margin-bottom: 1.2rem;
        }

        /* ── Prediction badge ── */
        .prediction-box {
            background: linear-gradient(135deg, rgba(79,172,254,0.2) 0%, rgba(0,242,254,0.12) 100%);
            border: 1.5px solid rgba(79,172,254,0.5);
            border-radius: 14px;
            padding: 1.4rem 1.8rem;
            text-align: center;
            margin-top: 1rem;
        }
        .prediction-box .label {
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: rgba(79,172,254,0.9);
            margin-bottom: 0.3rem;
        }
        .prediction-box .animal-name {
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
            text-transform: capitalize;
            margin: 0.2rem 0 0.5rem;
        }

        /* ── Confidence bar ── */
        .conf-wrap {
            background: rgba(0,0,0,0.25);
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
            margin: 0.6rem 0 0.3rem;
        }
        .conf-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            transition: width 1s ease;
        }
        .conf-text {
            font-size: 0.82rem;
            color: rgba(255,255,255,0.6);
            text-align: right;
        }

        /* ── Upload zone ── */
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.04) !important;
            border: 2px dashed rgba(79,172,254,0.4) !important;
            border-radius: 14px !important;
            transition: border-color 0.2s;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(79,172,254,0.9) !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: rgba(15,12,41,0.85) !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
        }

        /* ── Info tag ── */
        .tag {
            display: inline-block;
            background: rgba(79,172,254,0.15);
            color: #4facfe;
            border: 1px solid rgba(79,172,254,0.3);
            border-radius: 999px;
            padding: 2px 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 2px;
        }

        /* ── Metric tiles ── */
        .metric-tile {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .metric-tile .m-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #4facfe;
        }
        .metric-tile .m-label {
            font-size: 0.72rem;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* ── Spinner override ── */
        .stSpinner > div { border-top-color: #4facfe !important; }

        /* ── Step badge ── */
        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 26px; height: 26px;
            background: linear-gradient(135deg,#4facfe,#00f2fe);
            border-radius: 50%;
            font-size: 0.78rem;
            font-weight: 700;
            color: #0f0c29;
            margin-right: 8px;
            flex-shrink: 0;
        }

        /* scrollbar */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(79,172,254,0.3); border-radius: 3px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── Sidebar ─────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center;padding:1rem 0 1.4rem;'>
                <div style='font-size:3rem;'>🐾</div>
                <div style='font-weight:700;font-size:1.1rem;color:#fff;'>Animal Classifier</div>
                <div style='font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:4px;'>Powered by MobileNetV2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### ⚙️ Model Info")

        # Architecture tags
        st.markdown(
            """
            <div style='margin:0.5rem 0 1rem;'>
                <span class='tag'>MobileNetV2</span>
                <span class='tag'>Feature Extraction</span>
                <span class='tag'>Classical ML</span>
                <span class='tag'>90 Classes</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Input size:** 224 × 224 px")
        st.markdown("**Feature dim:** 1,280 (GAP)")
        st.markdown("**Framework:** TensorFlow + scikit-learn")

        st.markdown("---")
        st.markdown("### 📋 How to Use")
        steps = [
            "Upload an animal photo (JPG / PNG / WEBP)",
            "Wait for feature extraction (~2 s on first run)",
            "View the predicted class and confidence",
        ]
        for i, s in enumerate(steps, 1):
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;margin:0.45rem 0;color:rgba(255,255,255,0.75);font-size:0.88rem;'>"
                f"<span class='step-badge'>{i}</span>{s}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:rgba(255,255,255,0.3);text-align:center;'>"
            "Animal Classifier AI · v1.0</div>",
            unsafe_allow_html=True,
        )

# ── Main UI ──────────────────────────────────────────────────────────
def render_main():
    # ── Hero banner ──
    st.markdown(
        """
        <div class='hero-banner'>
            <h1>🐾 Animal Classifier AI</h1>
            <p>Upload any animal photo and our AI will identify the species instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metric tiles ──
    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        ("90", "Animal Classes"),
        ("MobileNetV2", "Backbone"),
        ("1 280", "Feature Dims"),
        ("224 px", "Input Size"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], tiles):
        col.markdown(
            f"<div class='metric-tile'><div class='m-value'>{val}</div>"
            f"<div class='m-label'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ──
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 📤 Upload Animal Image")
        uploaded = st.file_uploader(
            label="Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(
                pil_img,
                caption=f"{uploaded.name}  ·  {pil_img.size[0]}×{pil_img.size[1]} px",
                use_column_width=True,
            )
            st.markdown(
                f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:4px;'>"
                f"Format: {pil_img.format or uploaded.type.split('/')[-1].upper()}  ·  "
                f"Mode: {pil_img.mode}</div>",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### 🔍 Prediction")

        if not uploaded:
            st.markdown(
                "<div class='glass-card' style='text-align:center;padding:3rem 1rem;'>"
                "<div style='font-size:3rem;margin-bottom:0.6rem;'>🦁</div>"
                "<div style='color:rgba(255,255,255,0.45);font-size:0.9rem;'>"
                "Upload an image to see the prediction.</div></div>",
                unsafe_allow_html=True,
            )
            return

        # ── Run prediction ──
        with st.spinner("🧠  Extracting features & predicting …"):
            t0 = time.time()
            try:
                preprocessed = preprocess_image(pil_img)
                features      = extract_features(preprocessed)
                class_name, confidence = predict(features)
                elapsed = time.time() - t0
            except Exception as exc:
                st.error(f"❌ Prediction failed: {exc}")
                st.exception(exc)
                return

        # ── Display result ──
        animal_display = class_name.replace("_", " ").title()

        st.markdown(
            f"""
            <div class='prediction-box'>
                <div class='label'>🏆 Predicted Animal</div>
                <div class='animal-name'>{animal_display}</div>
            """,
            unsafe_allow_html=True,
        )

        if confidence is not None:
            bar_pct  = int(confidence * 100)
            bar_color = (
                "#4facfe" if confidence >= 0.70
                else "#fbbf24" if confidence >= 0.40
                else "#f87171"
            )
            st.markdown(
                f"""
                <div class='conf-wrap'>
                    <div class='conf-fill' style='width:{bar_pct}%;background:{bar_color};'></div>
                </div>
                <div class='conf-text'>Confidence: <b style='color:#fff;'>{bar_pct}%</b></div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.8rem;color:rgba(255,255,255,0.45);margin-top:4px;'>"
                "Confidence not available for this model type.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)  # close prediction-box

        # ── Inference stats ──
        st.markdown("<br>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        s1.markdown(
            f"<div class='glass-card' style='text-align:center;'>"
            f"<div style='font-size:1.3rem;font-weight:700;color:#4facfe;'>{elapsed:.2f}s</div>"
            f"<div style='font-size:0.72rem;color:rgba(255,255,255,0.45);letter-spacing:1px;'>INFERENCE TIME</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        s2.markdown(
            f"<div class='glass-card' style='text-align:center;'>"
            f"<div style='font-size:1.3rem;font-weight:700;color:#4facfe;'>"
            f"{features.shape[-1]:,}</div>"
            f"<div style='font-size:0.72rem;color:rgba(255,255,255,0.45);letter-spacing:1px;'>FEATURE DIM</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Top-k probabilities (if model supports it) ──
        model, label_encoder = load_artifacts()
        if hasattr(model, "predict_proba"):
            proba   = model.predict_proba(features)[0]
            top_k   = min(5, len(proba))
            top_idx = np.argsort(proba)[::-1][:top_k]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Top Predictions")
            for rank, idx in enumerate(top_idx, 1):
                if hasattr(label_encoder, "inverse_transform"):
                    name = label_encoder.inverse_transform([idx])[0]
                else:
                    name = str(idx)
                name_display = name.replace("_", " ").title()
                pct  = proba[idx] * 100
                fill = int(pct)
                color = "#4facfe" if rank == 1 else "rgba(255,255,255,0.35)"
                st.markdown(
                    f"""
                    <div style='margin-bottom:0.55rem;'>
                        <div style='display:flex;justify-content:space-between;
                                    font-size:0.85rem;margin-bottom:3px;'>
                            <span style='color:{"#fff" if rank==1 else "rgba(255,255,255,0.6)"};
                                         font-weight:{"600" if rank==1 else "400"};'>
                                {rank}. {name_display}</span>
                            <span style='color:{color};font-weight:600;'>{pct:.1f}%</span>
                        </div>
                        <div style='background:rgba(0,0,0,0.3);border-radius:999px;height:6px;'>
                            <div style='width:{fill}%;height:100%;border-radius:999px;
                                        background:{color};'></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ── Entry point ──────────────────────────────────────────────────────
def main():
    inject_css()
    render_sidebar()
    render_main()

if __name__ == "__main__":
    main()
