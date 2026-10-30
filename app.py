"""Streamlit UI for the Herbal Plant Classifier with an educational plant guide."""
from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from plant_info import get_plant_info

ROOT = Path(__file__).resolve().parent
LOCAL_MODEL = ROOT / "artifacts" / "best_model.pt"
MODEL_URL = (
    "https://github.com/sumeet1402/herbal-plant-classifier-with-info/"
    "releases/latest/download/best_model.pt"
)
METRICS_PATH = ROOT / "results" / "metrics.json"

def recorded_test_accuracy() -> str:
    """Read the latest committed held-out test accuracy, when available."""
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return f"{float(metrics['test_accuracy']):.2%}"
    except (OSError, ValueError, KeyError, TypeError):
        return "See evaluation report"

TEST_ACCURACY_LABEL = recorded_test_accuracy()

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

st.set_page_config(
    page_title="Herbal Plant Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_model_path() -> Path:
    """Use a local checkpoint, or download the latest published GitHub Release asset."""
    if LOCAL_MODEL.is_file():
        return LOCAL_MODEL

    cache_dir = Path(tempfile.gettempdir()) / "herbal-plant-classifier-with-info"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / "best_model.pt"
    if cached_path.is_file() and cached_path.stat().st_size > 1_000_000:
        return cached_path

    partial_path = cache_dir / "best_model.pt.partial"
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "Herbal-Plant-Classifier-Streamlit-App"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with partial_path.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)
        if partial_path.stat().st_size < 1_000_000:
            raise RuntimeError("The downloaded model is smaller than expected.")
        partial_path.replace(cached_path)
        return cached_path
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


@st.cache_resource(show_spinner="Loading the trained classifier…")
def load_classifier(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return model, class_names, preprocess, checkpoint


with st.sidebar:
    st.title("🌿 Herbal AI")
    st.caption("Computer vision · Transfer learning")
    st.divider()
    st.subheader("About this model")
    st.write(
        "ResNet18 trained with transfer learning on the DIMPSAR medicinal plant image dataset."
    )
    st.metric("Supported classes", "40")
    st.metric("Recorded test accuracy", TEST_ACCURACY_LABEL)
    st.caption(
        "Recorded on the held-out test split. Results can differ on new photos; this score is not a safety guarantee."
    )
    st.divider()
    st.subheader("Responsible use")
    st.write(
        "This is an educational image-classification demo, not a botanical authority or medical tool. "
        "Never use its prediction alone to decide whether a plant is safe to eat, touch, prepare, or use medicinally."
    )

st.title("Herbal Plant Classifier")
st.write(
    "Upload a clear plant or leaf photo to see the model's most likely class, top-five scores, and an educational plant guide."
)
st.caption(
    "Best results usually come from a single, clearly visible plant or leaf with good lighting."
)

try:
    model_path = resolve_model_path()
    model, class_names, preprocess, checkpoint = load_classifier(str(model_path))
except Exception as exc:
    st.warning("The trained model could not be loaded yet.")
    st.write(
        "Confirm that the training workflow has published a model release, then refresh this page. "
        "The app downloads the model from the latest public GitHub Release automatically."
    )
    with st.expander("Technical details"):
        st.code(f"Model URL: {MODEL_URL}\n{type(exc).__name__}: {exc}")
    st.stop()

upload_col, result_col = st.columns([1, 1], gap="large")
with upload_col:
    st.subheader("1 · Upload an image")
    uploaded = st.file_uploader(
        "Choose a plant image",
        type=["jpg", "jpeg", "png"],
        help="JPG or PNG image files.",
    )
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Your uploaded image", use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("The uploaded file could not be opened as an image. Please try another JPG or PNG.")
            st.stop()
    else:
        st.info("Your image preview will appear here after you upload a file.")

with result_col:
    st.subheader("2 · View prediction")
    if uploaded is None:
        st.write("The predicted class and confidence scores will appear here.")
    else:
        image_tensor = preprocess(image).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(image_tensor), dim=1)[0]
        count = min(5, len(class_names))
        scores, indexes = torch.topk(probabilities, k=count)
        predicted_class = class_names[int(indexes[0])]
        confidence = float(scores[0])

        st.success("Prediction generated")
        st.markdown(f"### {predicted_class.replace('_', ' ')}")
        st.metric("Top prediction score", f"{confidence:.1%}")
        st.caption(
            "This score is the model's relative confidence, not a guarantee of correctness."
        )
        st.write("**Top five classes**")
        for score, index in zip(scores.tolist(), indexes.tolist()):
            label = class_names[int(index)].replace("_", " ")
            st.write(f"{label} · {score:.1%}")
            st.progress(float(score))

        st.divider()
        st.subheader("3 · About this plant")
        plant = get_plant_info(predicted_class)
        st.write(f"**Common name:** {plant['common_name']}")
        if plant["botanical"]:
            st.write(f"**Possible botanical name:** *{plant['botanical']}*")
        st.write(plant["about"])

        with st.expander("Traditional or reported uses", expanded=True):
            st.write(plant["traditional_use"])
        with st.expander("Food use: what should I know?", expanded=True):
            st.write(plant["food_context"])
        st.warning(f"**Safety caution:** {plant['safety']}")
        st.error(
            "Do not eat, brew, apply, or take a plant as medicine because of this AI result. "
            "Confirm identity with a qualified local botanist and rely on a trusted food source. "
            "The guide is educational and may not cover every species, plant part, or interaction."
        )
        if plant["source_url"]:
            st.markdown(f"**Further reading:** [Open plant reference]({plant['source_url']})")
        st.markdown(
            f"**General plant-safety guidance:** [Poison Control plant guide]({plant['general_safety_url']})"
        )

st.divider()
with st.expander("Model and evaluation details"):
    st.write(f"**Architecture:** {checkpoint.get('architecture', 'ResNet18')}")
    st.write(f"**Number of classes:** {len(class_names)}")
    st.write(f"**Best validation epoch:** {checkpoint.get('best_epoch', 'Not recorded')}")
    val_acc = checkpoint.get("best_validation_accuracy")
    if val_acc is not None:
        st.write(f"**Best validation accuracy:** {float(val_acc):.2%}")
    st.write(f"**Recorded test accuracy:** {TEST_ACCURACY_LABEL} (from results/metrics.json when available).")
    st.markdown(
        "**Dataset:** [DIMPSAR Medicinal Plant Classification]("
        "https://huggingface.co/datasets/Project-AgML/DIMPSAR_medicinal_plant_classification)."
    )
    st.markdown(
        "**Source code and model releases:** [GitHub repository]("
        "https://github.com/sumeet1402/herbal-plant-classifier-with-info)."
    )
    st.caption(
        "Plant guide entries are educational summaries and require independent verification. "
        "Common names may be region-specific and an image model cannot confirm the exact species."
    )
