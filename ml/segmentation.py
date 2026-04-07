import os
import numpy as np
import cv2
from ml.preprocessing import preprocess_input_image

def generate_segmentation(model, img_path):
    """Generate binary UNet mask (black/white only)."""
    img = preprocess_input_image(img_path, target_size=(224, 224))
    img_batch = np.expand_dims(img, axis=0)

    pred_mask = model.predict(img_batch)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = np.squeeze(pred_mask)

    # Resize to original using nearest-neighbor to preserve binary values.
    orig_img = cv2.imread(img_path)
    mask_resized = cv2.resize(
        pred_mask,
        (orig_img.shape[1], orig_img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    seg_filename = os.path.join("static", "images", "seg_" + base_name + ".png")
    cv2.imwrite(seg_filename, mask_resized)

    return seg_filename