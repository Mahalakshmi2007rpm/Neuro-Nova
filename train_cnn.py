import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# 📁 Dataset path
DATASET_PATH = "dataset/"
EPOCHS = int(os.getenv("CNN_EPOCHS", "15"))
STEPS_PER_EPOCH = os.getenv("CNN_STEPS_PER_EPOCH")
VALIDATION_STEPS = os.getenv("CNN_VALIDATION_STEPS")
USE_PRETRAINED = os.getenv("CNN_PRETRAINED", "1") == "1"

if STEPS_PER_EPOCH is not None:
    STEPS_PER_EPOCH = int(STEPS_PER_EPOCH)
if VALIDATION_STEPS is not None:
    VALIDATION_STEPS = int(VALIDATION_STEPS)

# 🔄 Data Augmentation + Normalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# 📥 Training Data
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# 📥 Validation Data
val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 🧠 Load Pretrained Model
base_model = MobileNetV2(
    weights='imagenet' if USE_PRETRAINED else None,
    include_top=False,
    input_shape=(224,224,3)
)

# ❌ Freeze base model
base_model.trainable = False

# 🧠 Build Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# ⚙️ Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🛑 Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 🚀 Train Model
print("🚀 Training Started Lakshmi...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stop]
)

# 💾 Save Model
os.makedirs("models", exist_ok=True)
model.save("models/cnn_model.h5")
with open("models/class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_data.class_indices, f, indent=2)
print("✅ Model Saved Successfully in models/cnn_model.h5")