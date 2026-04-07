import glob
import os
from ml.model import predict_image

tumor_types = ["glioma", "meningioma", "no_tumor", "pituitary"]
print("[TEST] Tumor Detection Results:\n")
print("-" * 70)

for tumor_type in tumor_types:
    pattern = f"dataset/{tumor_type}/**/*.jpg"
    images = glob.glob(pattern, recursive=True)
    
    if images:
        test_img = images[0]
        pred_class, confidence, heatmap_path, seg_path = predict_image(test_img)
        
        actual = tumor_type.replace("_", " ").title()
        match = "CORRECT" if pred_class.lower() == actual.lower() else "WRONG"
        
        print(f"Input Type: {actual}")
        print(f"  Test Image: {os.path.basename(test_img)}")
        print(f"  Model Prediction: {pred_class}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Result: {match}")
        print("-" * 70)
