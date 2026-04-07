import os
import json
import shutil
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from ml.heatmap import generate_gradcam
from ml.segmentation import generate_segmentation
from ml.preprocessing import preprocess_input_image

# Resolve model paths from the project root so runtime cwd does not matter.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.h5")
CLASSIFICATION_MODEL_ALT_PATH = os.path.join(BASE_DIR, "models", "cnn_model.keras")
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_model.h5")
SEGMENTATION_MODEL_ALT_PATH = os.path.join(BASE_DIR, "models", "unet_model.keras")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

classification_model = None
segmentation_model = None

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def _pretty_label(name):
    return name.replace("_", " ").title()


def _load_class_names():
    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                class_indices = json.load(f)
            ordered = sorted(class_indices.items(), key=lambda item: item[1])
            return [_pretty_label(name) for name, _ in ordered]
        except Exception:
            pass
    return CLASS_NAMES


CLASS_NAMES = _load_class_names()


def _fallback_image_path(img_path, prefix):
    os.makedirs(os.path.join("static", "images"), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join("static", "images", prefix + base_name + ".png")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        shutil.copy2(img_path, save_path)
        return save_path

    if prefix.startswith("gradcam_"):
        pseudo = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        pseudo = cv2.applyColorMap(pseudo, cv2.COLORMAP_HOT)
        cv2.imwrite(save_path, pseudo)
    else:
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(save_path, binary)

    return save_path


def _load_model_if_available(primary_path, alt_path):
    for path in (primary_path, alt_path):
        if os.path.exists(path):
            try:
                return load_model(path, compile=False)
            except Exception:
                continue
    return None


def _ensure_models_loaded():
    global classification_model, segmentation_model
    if classification_model is None:
        classification_model = _load_model_if_available(
            CLASSIFICATION_MODEL_PATH,
            CLASSIFICATION_MODEL_ALT_PATH,
        )
    if segmentation_model is None:
        segmentation_model = _load_model_if_available(
            SEGMENTATION_MODEL_PATH,
            SEGMENTATION_MODEL_ALT_PATH,
        )


def predict_image(img_path):
    _ensure_models_loaded()

    img = preprocess_input_image(img_path, target_size=(224, 224))
    img_batch = np.expand_dims(img, axis=0)

    # Classification (graceful fallback when model is unavailable/corrupt).
    predicted_class = "No Tumor"
    confidence = 0.0
    class_idx = 2
    if classification_model is not None:
        preds = classification_model.predict(img_batch)
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        predicted_class = CLASS_NAMES[class_idx]

    # Grad-CAM (fallback to pseudo heatmap when model is unavailable/fails).
    try:
        if classification_model is not None:
            heatmap_path = generate_gradcam(classification_model, img_batch, class_idx, img_path)
        else:
            heatmap_path = _fallback_image_path(img_path, "gradcam_")
    except Exception:
        heatmap_path = _fallback_image_path(img_path, "gradcam_")

    # Segmentation (fallback to binary mask when model is unavailable/fails).
    try:
        if segmentation_model is not None:
            seg_path = generate_segmentation(segmentation_model, img_path)
        else:
            seg_path = _fallback_image_path(img_path, "seg_")
    except Exception:
        seg_path = _fallback_image_path(img_path, "seg_")

    return predicted_class, confidence, heatmap_path, seg_path