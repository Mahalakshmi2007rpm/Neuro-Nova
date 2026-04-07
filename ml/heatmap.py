import os
import numpy as np
import tensorflow as tf
import cv2

def generate_gradcam(model, img_tensor, class_idx, original_img_path):

    # Force model to build for nested/transfer models.
    _ = model(img_tensor)

    # Handle nested models (MobileNet / transfer learning case).
    last_conv_layer = None

    for layer in model.layers[::-1]:
        # If it's a nested model (like MobileNet)
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers[::-1]:
                if "conv" in sub_layer.name.lower():
                    last_conv_layer = sub_layer
                    break
        if last_conv_layer is not None:
            break

    # Fallback if no nested conv layer is found.
    if last_conv_layer is None:
        for layer in model.layers[::-1]:
            if "conv" in layer.name.lower():
                last_conv_layer = layer
                break

    if last_conv_layer is None:
        raise ValueError("Your model has no convolution layer; Grad-CAM cannot be generated.")

    print("Using layer:", last_conv_layer.name)

    # Create gradient model.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs[0]]
    )

    # Gradient computation.
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize and stretch contrast for stronger color variation.
    heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_u8 = np.uint8(np.clip(heatmap * 255.0, 0, 255))
    heatmap_u8 = cv2.normalize(heatmap_u8, None, 0, 255, cv2.NORM_MINMAX)

    # Use HOT colormap: black -> red -> yellow -> white.
    output = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_HOT)

    # Save output image.
    os.makedirs("static/images", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_img_path))[0]
    save_path = os.path.join("static/images", "gradcam_" + base_name + ".png")
    cv2.imwrite(save_path, output)

    return save_path