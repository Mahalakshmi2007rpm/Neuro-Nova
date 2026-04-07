import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

print("[SETUP] Creating optimized model with large batch training...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset/", target_size=(224,224), batch_size=64, 
    class_mode='categorical', subset='training'
)
val_data = datagen.flow_from_directory(
    "dataset/", target_size=(224,224), batch_size=64,
    class_mode='categorical', subset='validation'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[TRAIN] Starting with limited steps...")
steps_per_epoch = min(10, len(train_data))
validation_steps = min(5, len(val_data))

model.fit(train_data, validation_data=val_data, epochs=2, 
          steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, verbose=1)

os.makedirs("models", exist_ok=True)

print("[SAVE] Saving to .keras format...")
try:
    model.save("models/cnn_model.keras", save_format='keras', overwrite=True)
    print("[OK] Saved .keras")
except Exception as e:
    print(f"[ERROR] .keras save failed: {e}")

print("[SAVE] Saving to .h5 format...")
try:
    model.save("models/cnn_model.h5", overwrite=True)
    print("[OK] Saved .h5")
except Exception as e:
    print(f"[ERROR] .h5 save failed: {e}")

print("[VERIFY] Loading models to verify...")
try:
    from tensorflow.keras.models import load_model
    m = load_model("models/cnn_model.keras")
    print("[OK] .keras verified!")
except Exception as e:
    print(f"[FAIL] .keras: {str(e)[:80]}")

try:
    m = load_model("models/cnn_model.h5", compile=False)
    print("[OK] .h5 verified!")
except Exception as e:
    print(f"[FAIL] .h5: {str(e)[:80]}")

with open("models/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("[OK] Class indices saved")
print("[DONE] Complete!")
