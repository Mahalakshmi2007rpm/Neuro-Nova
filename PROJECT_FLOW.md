# NEURO-NOVA: Complete Project Flow & Architecture

## Overview
Neuro-Nova is a **Brain Tumor Classification & Segmentation System** using deep learning. It's a Flask-based web application that takes MRI images, classifies tumors (Glioma, Meningioma, Pituitary, No Tumor), generates Grad-CAM visualizations, and produces segmentation masks.

---

## 🏗️ PROJECT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│              (Flask Routes + HTML Templates)                 │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├─ LOGIN/SIGNUP (Authentication)
           ├─ DASHBOARD (Main Page)
           ├─ UPLOAD (Image Upload Interface)
           └─ RESULT (Display Predictions)
           │
┌──────────▼──────────────────────────────────────────────────┐
│                  BUSINESS LOGIC LAYER                         │
│                 (Flask App & ML Pipeline)                     │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├─ IMAGE PREPROCESSING (opencv, numpy)
           ├─ CNN CLASSIFICATION (MobileNetV2)
           ├─ GRAD-CAM VISUALIZATION (TensorFlow)
           └─ UNET SEGMENTATION
           │
┌──────────▼──────────────────────────────────────────────────┐
│                   DATA LAYER                                  │
│            (Models, Database, Datasets)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 DETAILED FLOW WITH CODE

### STEP 1: APPLICATION STARTUP
**File:** [app.py](app.py)

```python
# Initialize Flask app
from flask import Flask
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

# Setup upload folder
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", 
                          os.path.join(tempfile.gettempdir(), "neuro_nova_uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple in-memory user database
USERS = {
    "lakshmi123": "12345",
    "admin": "admin"
}
```

---

### STEP 2: USER AUTHENTICATION FLOW
**Route:** `/login` and `/signup`

```python
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Verify credentials
        if username in USERS and USERS[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid username or password"
    
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Register new user
        if username in USERS:
            error = "User already exists"
        else:
            USERS[username] = password
            return redirect(url_for('login'))
    
    return render_template('signup.html', error=error)
```

**Template:** [templates/login.html](templates/login.html)
- Modern gradient UI with blue theme
- Two-panel layout (brand + form)
- Form validation for username/password

---

### STEP 3: DASHBOARD & NAVIGATION
**Route:** `/dashboard`

```python
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', username=session['username'])
```

**Template:** [templates/dashboard.html](templates/dashboard.html)
- Welcome screen after login
- Navigation to upload page
- Logout functionality

---

### STEP 4: IMAGE UPLOAD & INFERENCE
**Route:** `/upload` (POST)

```python
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        file = request.files.get('file')
        
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # 🔥 CORE INFERENCE PIPELINE
            from ml.model import predict_image
            pred_class, confidence, heatmap_path, seg_path = predict_image(filepath)
            
            # Prepare result data
            result = {
                "filename": filename,
                "pred_class": pred_class,
                "confidence": confidence,
                "heatmap": os.path.basename(heatmap_path),
                "segmentation": os.path.basename(seg_path)
            }
            
            return render_template('result.html', result=result)
    
    return render_template('upload.html', error=error)
```

**Template:** [templates/upload.html](templates/upload.html)
- Drag-and-drop image upload interface
- File validation
- Loading indicator

---

## 🧠 STEP 5: ML INFERENCE PIPELINE

### 5A. MAIN PREDICTION FUNCTION
**File:** [ml/model.py](ml/model.py)

```python
def predict_image(img_path):
    """
    Complete inference pipeline:
    1. Load image & preprocess
    2. Run CNN classification
    3. Generate Grad-CAM heatmap
    4. Generate UNet segmentation
    """
    
    _ensure_models_loaded()
    
    # 🔹 PREPROCESSING
    img = preprocess_input_image(img_path, target_size=(224, 224))
    img_batch = np.expand_dims(img, axis=0)  # Add batch dimension: (1, 224, 224, 3)
    
    # 🔹 CLASSIFICATION (CNN)
    predicted_class = "No Tumor"
    confidence = 0.0
    class_idx = 2
    
    if classification_model is not None:
        preds = classification_model.predict(img_batch)  # Shape: (1, 4)
        class_idx = int(np.argmax(preds))  # Get highest probability index
        confidence = float(np.max(preds))  # Get probability value
        predicted_class = CLASS_NAMES[class_idx]  # 0=Glioma, 1=Meningioma, 2=No Tumor, 3=Pituitary
    
    # 🔹 GRAD-CAM VISUALIZATION
    try:
        if ENABLE_GRADCAM and classification_model is not None:
            heatmap_path = generate_gradcam(classification_model, img_batch, class_idx, img_path)
        else:
            heatmap_path = _fallback_image_path(img_path, "gradcam_")
    except Exception:
        heatmap_path = _fallback_image_path(img_path, "gradcam_")
    
    # 🔹 SEGMENTATION (UNet)
    try:
        if ENABLE_SEGMENTATION and segmentation_model is not None:
            seg_path = generate_segmentation(segmentation_model, img_path)
        else:
            seg_path = _fallback_image_path(img_path, "seg_")
    except Exception:
        seg_path = _fallback_image_path(img_path, "seg_")
    
    return predicted_class, confidence, heatmap_path, seg_path
```

### 5B. IMAGE PREPROCESSING
**File:** [ml/preprocessing.py](ml/preprocessing.py)

```python
import cv2
import numpy as np

def preprocess_input_image(img_path, target_size=(224, 224)):
    """
    Prepare image for model input:
    1. Read image from file
    2. Convert BGR → RGB
    3. Resize to target size
    4. Normalize to [0, 1] range
    """
    
    # 1. Read image
    img = cv2.imread(img_path)  # Shape: (height, width, 3), BGR format
    
    # 2. Convert color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize
    img = cv2.resize(img, target_size)  # Shape: (224, 224, 3)
    
    # 4. Normalize
    img = img / 255.0  # Convert uint8 [0, 255] → float32 [0, 1]
    
    return img
```

### 5C. CNN CLASSIFICATION MODEL
**File:** [ml/model.py](ml/model.py) - Model Loading

```python
# Model paths
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASSIFICATION_MODEL_PATH = "models/cnn_model.h5"
CLASSIFICATION_MODEL_ALT_PATH = "models/cnn_model.keras"

# Load model once (lazy loading)
classification_model = None

def _ensure_models_loaded():
    global classification_model
    if classification_model is None:
        classification_model = _load_model_if_available(
            CLASSIFICATION_MODEL_PATH,
            CLASSIFICATION_MODEL_ALT_PATH,
        )

def _load_model_if_available(primary_path, alt_path):
    """Try to load model from primary or alternative path"""
    for path in (primary_path, alt_path):
        if os.path.exists(path):
            try:
                return load_model(path, compile=False)
            except Exception:
                continue
    return None
```

### 5D. GRAD-CAM VISUALIZATION
**File:** [ml/heatmap.py](ml/heatmap.py)

```python
import tensorflow as tf

def generate_gradcam(model, img_tensor, class_idx, original_img_path):
    """
    Generate Grad-CAM heatmap to visualize which regions influenced prediction.
    
    Steps:
    1. Find last convolutional layer
    2. Compute gradients of target class w.r.t. conv outputs
    3. Weight conv feature maps by gradients
    4. Apply colormap and save
    """
    
    # 1️⃣ FIND LAST CONV LAYER
    last_conv_layer = None
    
    for layer in model.layers[::-1]:
        # Handle nested models (e.g., MobileNet)
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers[::-1]:
                if "conv" in sub_layer.name.lower():
                    last_conv_layer = sub_layer
                    break
        if last_conv_layer is not None:
            break
    
    # Fallback: find any conv layer
    if last_conv_layer is None:
        for layer in model.layers[::-1]:
            if "conv" in layer.name.lower():
                last_conv_layer = layer
                break
    
    # 2️⃣ CREATE GRADIENT MODEL
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs[0]]
    )
    
    # 3️⃣ COMPUTE GRADIENTS
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]  # Get score for target class
    
    # Backprop to get gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # 4️⃣ WEIGHT CONV FEATURES BY GRADIENTS
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Average gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    
    # 5️⃣ NORMALIZE AND APPLY COLORMAP
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_u8 = np.uint8(np.clip(heatmap * 255.0, 0, 255))
    
    # Apply HOT colormap: black → red → yellow → white
    output = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_HOT)
    
    # 6️⃣ SAVE OUTPUT
    os.makedirs("static/images", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_img_path))[0]
    save_path = os.path.join("static/images", "gradcam_" + base_name + ".png")
    cv2.imwrite(save_path, output)
    
    return save_path
```

### 5E. UNET SEGMENTATION
**File:** [ml/segmentation.py](ml/segmentation.py)

```python
def generate_segmentation(model, img_path):
    """
    Generate binary mask to show tumor location.
    
    Steps:
    1. Preprocess image
    2. Predict mask from UNet
    3. Apply threshold (0.5)
    4. Convert to binary (0 or 255)
    5. Resize to original dimensions
    """
    
    # 1️⃣ PREPROCESS
    img = preprocess_input_image(img_path, target_size=(224, 224))
    img_batch = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
    
    # 2️⃣ PREDICT
    pred_mask = model.predict(img_batch)[0]  # Shape: (224, 224, 1)
    
    # 3️⃣ THRESHOLD & BINARIZE
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Values: 0 or 255
    pred_mask = np.squeeze(pred_mask)  # Shape: (224, 224)
    
    # 4️⃣ RESIZE TO ORIGINAL SIZE
    orig_img = cv2.imread(img_path)
    mask_resized = cv2.resize(
        pred_mask,
        (orig_img.shape[1], orig_img.shape[0]),
        interpolation=cv2.INTER_NEAREST  # Preserve binary values
    )
    
    # 5️⃣ SAVE
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    seg_filename = os.path.join("static", "images", "seg_" + base_name + ".png")
    cv2.imwrite(seg_filename, mask_resized)
    
    return seg_filename
```

---

## 📚 STEP 6: MODEL TRAINING

### Training Script: [train_final.py](train_final.py)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# 1️⃣ LOAD & PREPARE DATA
print("[*] Loading dataset...")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
)

train_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 2️⃣ BUILD MODEL (Transfer Learning)
print("[*] Building MobileNetV2 model...")
base_model = MobileNetV2(
    weights='imagenet',  # Pre-trained on ImageNet
    include_top=False,   # Remove classification head
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze base weights

# Add custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Convert (7, 7, 1280) → (1280,)
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # 50% dropout
    layers.Dense(4, activation='softmax')  # 4 classes
])

# 3️⃣ COMPILE
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4️⃣ TRAIN
print("[*] Training for 3 epochs...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    verbose=2
)

# 5️⃣ SAVE MODELS
print("\n[*] Saving model...")
os.makedirs("models", exist_ok=True)

# Save in .keras format (Keras 3 native)
model.save("models/cnn_model.keras", save_format='keras')

# Also save as H5 (backward compatibility)
model.save("models/cnn_model.h5")

# 6️⃣ SAVE CLASS MAPPING
with open("models/class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_data.class_indices, f, indent=2)
```

**Output:** [models/class_indices.json](models/class_indices.json)
```json
{
  "glioma": 0,
  "meningioma": 1,
  "no_tumor": 2,
  "pituitary": 3
}
```

---

## 📂 STEP 7: DISPLAY RESULTS

### Result Page: [templates/result.html](templates/result.html)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Neuro Nova | Result</title>
</head>
<body>
    <div class="result-container">
        <!-- Prediction Result -->
        <div class="prediction-card">
            <h2>Classification Result</h2>
            <p><strong>Tumor Type:</strong> {{ result.pred_class }}</p>
            <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence * 100) }}%</p>
        </div>
        
        <!-- Grad-CAM Visualization -->
        <div class="visualization-card">
            <h3>Grad-CAM Heatmap (Explainability)</h3>
            <p>Highlights regions that influenced the prediction</p>
            <img src="{{ url_for('static', filename='images/' + result.heatmap) }}" 
                 alt="Grad-CAM">
        </div>
        
        <!-- UNet Segmentation -->
        <div class="segmentation-card">
            <h3>Tumor Segmentation</h3>
            <p>Binary mask showing tumor location</p>
            <img src="{{ url_for('static', filename='images/' + result.segmentation) }}" 
                 alt="Segmentation">
        </div>
        
        <!-- Navigation -->
        <a href="{{ url_for('upload') }}">Analyze Another Image</a>
        <a href="{{ url_for('logout') }}">Logout</a>
    </div>
</body>
</html>
```

---

## 📁 FOLDER STRUCTURE & PURPOSES

```
neuro-nova/
│
├── 📄 app.py                          # Main Flask application & routes
├── 📄 config.py                       # Configuration constants
├── 📄 requirements.txt                # Python dependencies
├── 📄 Procfile                        # Render deployment config
├── 📄 runtime.txt                     # Python version for Render
│
├── 📂 ml/                             # Core ML modules
│   ├── __init__.py
│   ├── model.py                       # Main inference: predict_image()
│   ├── preprocessing.py               # Image: BGR→RGB, resize, normalize
│   ├── heatmap.py                     # Grad-CAM visualization
│   └── segmentation.py                # UNet binary segmentation
│
├── 📂 models/                         # Trained models & metadata
│   ├── cnn_model.h5                   # Classification (MobileNetV2)
│   ├── cnn_model.keras                # Classification (Keras format)
│   ├── unet_model.h5                  # Segmentation (UNet)
│   └── class_indices.json             # {class_name: index} mapping
│
├── 📂 dataset/                        # Training data
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   │   └── notumor/
│   └── pituitary/
│
├── 📂 database/                       # User database
│   └── database.db                    # SQLite (future)
│
├── 📂 templates/                      # HTML pages
│   ├── login.html                     # Authentication page
│   ├── signup.html                    # User registration
│   ├── dashboard.html                 # Welcome after login
│   ├── upload.html                    # Image upload interface
│   └── result.html                    # Display predictions
│
├── 📂 static/                         # Static assets
│   ├── css/
│   │   └── style.css                  # Styling
│   ├── js/
│   │   └── script.js                  # Client-side logic
│   ├── images/                        # Generated outputs (Grad-CAM, masks)
│   ├── heatmaps/                      # Alternative heatmap storage
│   └── uploads/                       # User uploads
│
├── 📂 uploads/                        # Temporary file storage
│
├── 📄 train_final.py                  # Final training script (MobileNetV2)
├── 📄 train_fast.py                   # Quick training
├── 📄 train_cnn.py                    # CNN training variant
├── 📄 train_cnn_quick.py              # Fast CNN variant
├── 📄 train_unet.py                   # UNet segmentation training
│
└── 📄 test_*.py                       # Inference tests
```

---

## 🔄 COMPLETE REQUEST→RESPONSE FLOW DIAGRAM

```
┌─────────────────────────────────────┐
│ 1. USER VISITS app.com              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. REDIRECTED TO /login             │
│    (login.html rendered)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. USER ENTERS CREDENTIALS          │
│    POST /login                      │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
    ✅ VALID     ❌ INVALID
        │             │
        ▼             ▼
  Redirect to    Show error
  /dashboard     Re-render
        │        login.html
        │             │
        └──────┬──────┘
               │
               ▼
    ┌──────────────────────────┐
    │ 4. DASHBOARD LOADED      │
    │    (dashboard.html)      │
    │    w/ username           │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ 5. NAVIGATE TO /upload   │
    │    (upload.html)         │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ 6. DRAG & DROP MRI IMG   │
    │    (jpg/png accepted)    │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ 7. POST /upload          │
    │    file saved to UPLOADS │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────────────────────────┐
    │ 8. ML PIPELINE (predict_image)               │
    │                                              │
    │    [Preprocess] ──→ [CNN Classify]           │
    │                              │               │
    │                    (Glioma, Meningioma,     │
    │                     No Tumor, Pituitary)    │
    │                              │               │
    │                    ┌─────────┴─────────┐    │
    │                    ▼                   ▼    │
    │            [Grad-CAM]           [Segmentation]
    │            (heatmap)             (binary mask)
    │                                              │
    └──────────┬───────────────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────────┐
    │ 9. RETURN RESULTS                            │
    │    - Tumor class                             │
    │    - Confidence %                            │
    │    - Heatmap image path                      │
    │    - Segmentation mask path                  │
    └──────────┬───────────────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────────┐
    │ 10. RENDER result.html                       │
    │     Display:                                 │
    │     - Classification result                  │
    │     - Grad-CAM visualization                 │
    │     - Tumor segmentation mask                │
    │     - "Analyze Another" button               │
    └──────────┬───────────────────────────────────┘
               │
               ├─→ [Analyze Another] → /upload
               └─→ [Logout] → /login
```

---

## 🔧 DATA STRUCTURES & MODELS

### CNN Model Architecture (MobileNetV2)
```
Input: (224, 224, 3)
  ↓
[MobileNetV2 Base] (Pre-trained on ImageNet)
  - Frozen weights (Transfer Learning)
  - Output: (7, 7, 1280)
  ↓
[Global Average Pooling] → (1280,)
  ↓
[Batch Normalization]
  ↓
[Dense Layer] 128 units, ReLU
  ↓
[Dropout] 50%
  ↓
[Dense Layer] 4 units, Softmax
  ↓
Output: [P(Glioma), P(Meningioma), P(No Tumor), P(Pituitary)]
```

### UNet Segmentation Model
```
Input: (224, 224, 3)
  ↓
[Encoder-Decoder Architecture]
  - Downsampling path
  - Upsampling path with skip connections
  ↓
Output: (224, 224, 1) - Binary mask (0.0-1.0)
  ↓
Threshold > 0.5 → Binary (0 or 255)
```

---

## 🚀 DEPLOYMENT SETUP

### [Procfile](Procfile) - Render Deployment
```
web: gunicorn app:app
```

### [runtime.txt](runtime.txt) - Python Version
```
python-3.11.x
```

### [requirements.txt](requirements.txt) - Dependencies
```
Flask==3.1.3
tensorflow==2.21.0
keras==3.12.1
opencv-python==4.13.0.92
numpy==2.2.6
Werkzeug==3.1.8
gunicorn==23.0.0
... (40+ more packages)
```

---

## 📊 KEY FEATURES

| Feature | How It Works |
|---------|------------|
| **Image Preprocessing** | BGR→RGB, resize to 224×224, normalize to [0,1] |
| **Classification** | MobileNetV2 (ImageNet pre-trained) + custom head |
| **Explainability** | Grad-CAM heatmap shows decision-influencing regions |
| **Segmentation** | UNet binary mask to pinpoint tumor location |
| **Authentication** | Session-based login/signup (in-memory USERS dict) |
| **File Handling** | Secure filename validation, temp directory storage |
| **Error Handling** | Graceful fallbacks if models fail or unavailable |

---

## 🎯 REQUEST EXAMPLE

**User uploads:** `brain_mri.jpg`

**Processing:**
1. Save to temp folder
2. Preprocess (resize, normalize)
3. CNN predicts: "Glioma" with 92.5% confidence
4. Grad-CAM highlights suspicious region (red/yellow)
5. UNet segments tumor location (white mask on black)

**Response JSON:**
```json
{
  "filename": "brain_mri.jpg",
  "pred_class": "Glioma",
  "confidence": 0.925,
  "heatmap": "gradcam_brain_mri.png",
  "segmentation": "seg_brain_mri.png"
}
```

**Rendered in browser:**
- Tumor Classification: **Glioma (92.50%)**
- Heatmap showing affected region
- Binary segmentation mask

---

## 📝 SUMMARY

**Neuro-Nova** is a complete ML-to-web pipeline:
- **Input:** MRI brain image
- **Process:** CNN classification → Grad-CAM explanation → UNet segmentation
- **Output:** Tumor type, confidence, visualizations
- **Delivery:** Flask web app with authentication and responsive UI

