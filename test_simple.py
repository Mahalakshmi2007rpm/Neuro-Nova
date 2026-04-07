import tensorflow as tf
import glob
import os

print("Loading model...")
model = tf.keras.models.load_model("models/cnn_model.keras")
print(f"Model loaded! Output shape: {model.output_shape}")

print("\nTesting predictions...")
# Get one med image from each class
for tumor_type in ["glioma", "meningioma", "no_tumor", "pituitary"]:
    folder = f"dataset/{tumor_type}"
    images = glob.glob(f"{folder}/**/*.jpg", recursive=True)
    if images:
        print(f"\n{tumor_type.upper()}: {len(images)} images available")
