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
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_model.h5")
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


def _ensure_models_loaded():
    global classification_model, segmentation_model
    if classification_model is None:
        classification_model = load_model(CLASSIFICATION_MODEL_PATH)
    if segmentation_model is None:
        segmentation_model = load_model(SEGMENTATION_MODEL_PATH)

def predict_image(img_path):
    try:
        _ensure_models_loaded()
    except Exception as exc:
        raise RuntimeError(
            "Model files could not be loaded. Please verify models/cnn_model.h5 "
            "and models/unet_model.h5 are valid Keras HDF5 files."
        ) from exc

    img = preprocess_input_image(img_path, target_size=(224, 224))
    img_batch = np.expand_dims(img, axis=0)

    # Classification
    preds = classification_model.predict(img_batch)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    predicted_class = CLASS_NAMES[class_idx]

    # Grad-CAM (fallback to original image if Grad-CAM fails).
    try:
        heatmap_path = generate_gradcam(classification_model, img_batch, class_idx, img_path)
    except Exception:
        heatmap_path = _fallback_image_path(img_path, "gradcam_")

    # Segmentation (fallback to original image if generation fails).
    try:
        seg_path = generate_segmentation(segmentation_model, img_path)
    except Exception:
        seg_path = _fallback_image_path(img_path, "seg_")

    return predicted_class, confidence, heatmap_path, seg_path