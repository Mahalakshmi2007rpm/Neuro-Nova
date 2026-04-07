import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

print("📁 Loading dataset...")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
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

print("🧠 Building MobileNetV2 model...")
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

print("🚀 Training for 5 epochs...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    verbose=1
)

# 💾 Save Model - Critical section
print("\n💾 Saving model to models/cnn_model.h5...")
os.makedirs("models", exist_ok=True)

try:
    model.save("models/cnn_model.h5")
    print("✅ Model saved successfully!")
    
    # Verify the save
    from tensorflow.keras.models import load_model
    test_load = load_model("models/cnn_model.h5", compile=False)
    print("✅ Model verified - loaded successfully!")
    
    # Save class indices
    with open("models/class_indices.json", "w", encoding="utf-8") as f:
        json.dump(train_data.class_indices, f, indent=2)
    print("✅ Class indices saved!")
    
except Exception as e:
    print(f"❌ ERROR saving model: {e}")
    import traceback
    traceback.print_exc()
