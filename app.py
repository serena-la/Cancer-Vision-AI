from pathlib import Path
import hashlib
import json

import streamlit as st
from PIL import Image

from inference import analyse_image, load_model

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "evaluation_metrics.json"

st.set_page_config(
    page_title="Cancer Vision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _image_to_png_bytes(image):
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_css():
    css_path = BASE_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css()

def section_header(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<h1 class="page-title">{title}</h1>'
        f'<p class="page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown(
        '<div class="brand-wrap"><div class="brand-icon">✳</div>'
        '<div><div class="brand-name">CANCER VISION AI</div>'
        '<div class="brand-caption">EXPLAINABLE HISTOPATHOLOGY</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Image analysis", "Model & evaluation"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-status"><span class="status-dot"></span>'
        'Research prototype</div>',
        unsafe_allow_html=True,
    )
    st.caption("ResNet-50  ·  Grad-CAM")
    st.markdown(
        '<div class="sidebar-foot">For research and educational use only. '
        'Not intended for clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )

if page == "Overview":
    section_header(
        "COMPUTATIONAL PATHOLOGY · RESEARCH",
        "See the tissue. Understand the prediction.",
        "Cancer Vision AI explores explainable deep learning for histopathology image-patch classification.",
    )
    st.markdown(
        '<div class="hero-panel"><div class="hero-kicker">AI-ASSISTED IMAGE ANALYSIS</div>'
        '<div class="hero-title">A prediction should be<br><span>understandable.</span></div>'
        '<div class="hero-copy">Upload an individual tissue image patch to obtain a model prediction '
        'and a Grad-CAM visual explanation of image regions that influenced the model.</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="feature-card"><div class="feature-symbol">01</div>'
                    '<div class="feature-title">ResNet-50</div><div class="feature-copy">'
                    'A convolutional neural network adapted for two-class image classification.</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="feature-card"><div class="feature-symbol">02</div>'
                    '<div class="feature-title">Grad-CAM</div><div class="feature-copy">'
                    'A class-specific heatmap that visualises image regions influencing the selected prediction.</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="feature-card"><div class="feature-symbol">03</div>'
                    '<div class="feature-title">Transparent evaluation</div><div class="feature-copy">'
                    'Validation metrics can be displayed separately from predictions on newly uploaded images.</div></div>',
                    unsafe_allow_html=True)
    st.markdown("### Start an analysis")
    st.write("Open **Image analysis** in the sidebar to upload a PNG, JPG or JPEG tissue patch.")
    st.info("This prototype accepts individual image patches, not whole-slide images (WSI).")

elif page == "Image analysis":
    section_header(
        "IMAGE WORKSPACE",
        "Histopathology analysis",
        "Upload one image patch. The model resizes it to 512 × 512 pixels using the training notebook's resize-and-pad procedure.",
    )
    st.markdown('<div class="notice">Research prototype · The output is an algorithmic estimate, '
                'not a medical diagnosis or a substitute for pathologist review.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a histopathology image patch",
        type=["png", "jpg", "jpeg"],
        help="Upload one individual image patch. Avoid uploading identifiable patient information.",
    )

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        image_key = hashlib.sha256(raw_bytes).hexdigest()
        if st.session_state.get("image_key") != image_key:
            st.session_state["image_key"] = image_key
            st.session_state.pop("analysis_result", None)

        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("The uploaded file could not be opened as an image.")
            st.stop()

        left, right = st.columns([1.05, 1])
        with left:
            st.markdown("#### Uploaded image")
            st.image(image, use_container_width=True)
            st.caption(f"{image.width} × {image.height} pixels · {uploaded.name}")

        with right:
            st.markdown("#### Ready for analysis")
            st.markdown(
                '<div class="upload-summary"><div class="summary-label">INPUT TYPE</div>'
                '<div class="summary-value">Individual image patch</div>'
                '<div class="summary-label">MODEL INPUT</div>'
                '<div class="summary-value">512 × 512 RGB</div>'
                '<div class="summary-label">PREPROCESSING</div>'
                '<div class="summary-value">Aspect-preserving resize · white padding · ImageNet normalisation</div></div>',
                unsafe_allow_html=True,
            )
            run = st.button("Run image analysis", type="primary", use_container_width=True)

        if run:
            try:
                with st.spinner("Loading the model and generating the Grad-CAM explanation…"):
                    model, tumor_idx = load_model()
                    result = analyse_image(image, model, tumor_idx)
                st.session_state["analysis_result"] = result
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)
                st.stop()

        result = st.session_state.get("analysis_result")
        if result is not None:
            st.markdown("---")
            st.markdown("### Analysis result")
            prediction = result["prediction"]
            probability = result["tumor_probability"]
            r1, r2, r3 = st.columns(3)
            with r1:
                metric_card("MODEL PREDICTION", prediction, "Threshold = 0.50")
            with r2:
                metric_card("TUMOUR PROBABILITY", f"{probability * 100:.1f}%", "Softmax probability for tumour class")
            with r3:
                metric_card("MODEL", "ResNet-50", "Two-class classifier")

            st.progress(float(max(0.0, min(1.0, probability))), text="Tumour-class probability")
            st.markdown("#### Image comparison")
            a, b = st.columns(2)
            with a:
                st.markdown("**Model input (512 × 512)**")
                st.image(result["model_input"], use_container_width=True)
            with b:
                st.markdown("**Grad-CAM overlay**")
                st.image(result["overlay"], use_container_width=True)
            with st.expander("View heatmap separately"):
                st.image(result["heatmap"], caption="Grad-CAM activation map", use_container_width=True)
            st.caption(
                "Grad-CAM highlights image regions that influenced the model's tumour-class score. "
                "It does not identify or prove the presence of tumour cells."
            )
            st.download_button(
                "Download Grad-CAM overlay",
                data=_image_to_png_bytes(result["overlay"]),
                file_name="cancer_vision_gradcam.png",
                mime="image/png",
            )
    else:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">＋</div>'
            '<div class="empty-title">Your image workspace is ready</div>'
            '<div class="empty-copy">Choose an image patch above to begin.</div></div>',
            unsafe_allow_html=True,
        )

elif page == "Model & evaluation":
    section_header(
        "MODEL TRANSPARENCY",
        "Model & evaluation",
        "Architecture, preprocessing and validation results associated with the saved model checkpoint.",
    )
    st.markdown("### Model information")
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("ARCHITECTURE", "ResNet-50", "ImageNet V2 initialisation during training")
    with m2:
        metric_card("INPUT SIZE", "512 × 512", "RGB image patch")
    with m3:
        metric_card("DECISION THRESHOLD", "0.50", "Tumour if P(tumour) > 0.50")

    with st.expander("Architecture and preprocessing details", expanded=True):
        st.markdown("""
- **Backbone:** torchvision ResNet-50.
- **Classification head:** Dropout (0.3), followed by a fully connected layer with 2 output units.
- **Resize and pad:** preserve aspect ratio, resize the longest side to 512 pixels, then centre the image on a 512 × 512 white RGB canvas.
- **Normalisation:** ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- **Prediction:** softmax probability for the checkpoint's `tumor` class; tumour prediction when probability is greater than 0.50.
- **Explanation:** Grad-CAM applied to `model.layer4[-1]`.
        """)
    st.markdown("### Validation evaluation")
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            st.caption(
                f"Validation split: {metrics.get('n_samples', '—')} patches · "
                f"Threshold: {metrics.get('threshold', 0.5):.2f}"
            )
            display_order = [
                "Accuracy (%)", "Balanced accuracy (%)", "AUC (ROC)",
                "Average precision (PR)", "Precision (tumor)",
                "Recall / Sensitivity", "Specificity", "F1 (tumor)",
                "F1 (macro)", "MCC",
            ]
            rows = [{"Metric": k, "Value": metrics["metrics"][k]} for k in display_order
                    if k in metrics.get("metrics", {})]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            cm = metrics.get("confusion_matrix")
            if cm:
                st.markdown("#### Confusion matrix")
                import pandas as pd
                labels = metrics.get("confusion_labels", ["normal", "tumor"])
                st.dataframe(
                    pd.DataFrame(cm, index=[f"Actual {x}" for x in labels],
                                 columns=[f"Predicted {x}" for x in labels]),
                    use_container_width=True,
                )
            st.caption("These are validation-set results saved by evaluate.py, not a performance estimate for the uploaded image.")
        except Exception as exc:
            st.warning(f"Could not read evaluation_metrics.json: {exc}")
    else:
        st.info(
            "Validation metrics have not been generated yet. Run `python evaluate.py` "
            "after placing your validation dataset in `data/val/normal` and `data/val/tumor`."
        )

    st.markdown("### Interpretation and limitations")
    st.markdown("""
- The displayed probability is the model's softmax output, not a clinically calibrated probability of disease.
- Grad-CAM is a visual explanation of model activation, not a tumour segmentation mask or proof of malignancy.
- Performance depends on the training dataset, staining, scanner, tissue preparation and image distribution.
- Validation metrics describe the supplied validation split only; they do not establish clinical performance or generalisability.
- This tool is intended for research and education. It must not be used to make patient-care decisions.
    """)

st.markdown(
    '<div class="app-footer">CANCER VISION AI · RESEARCH PROTOTYPE<br>'
    'Not for clinical diagnosis, screening or treatment decisions.</div>',
    unsafe_allow_html=True,
)
