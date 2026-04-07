import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

print("[*] Loading dataset...")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("[*] Building MobileNetV2 model...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[*] Training for 3 epochs...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    verbose=2
)

print("\n[*] Saving model...")
os.makedirs("models", exist_ok=True)

# Save in new .keras format (more reliable)
model.save("models/cnn_model.keras", save_format='keras')
print("[OK] Saved models/cnn_model.keras")

# Also save as H5 for legacy compatibility
model.save("models/cnn_model.h5")
print("[OK] Saved models/cnn_model.h5")

# Verify by loading
print("\n[*] Verifying save...")
try:
    from tensorflow.keras.models import load_model
    test = load_model("models/cnn_model.keras")
    print("[OK] .keras format verified!")
except Exception as e:
    print(f"❌ .keras error: {e}")

try:
    test = load_model("models/cnn_model.h5", compile=False)
    print("[OK] .h5 format verified!")
except Exception as e:
    print(f"❌ .h5 error: {e}")

# Save class indices
with open("models/class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_data.class_indices, f, indent=2)
print("[OK] Saved class_indices.json")

print("\n[DONE] Training complete!")
