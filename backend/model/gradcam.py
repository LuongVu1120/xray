import numpy as np
import cv2
import tensorflow as tf


def generate_gradcam(model: tf.keras.Model, img_array: np.ndarray, class_idx: int):
    """
    Generate Grad-CAM heatmap for the given image and class index.

    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, 224, 224, 3)
        class_idx: Predicted class index

    Returns:
        superimposed: Heatmap overlaid on original image (BGR numpy array)
        severity: Max activation intensity (0.0 - 1.0)
    """
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        # Fallback: use a known DenseNet layer name
        last_conv_layer = "conv5_block16_2_conv"

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()

    # Normalize
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    if cam.max() > 0:
        cam = cam / cam.max()

    severity = float(np.percentile(cam, 95))

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(np.uint8(img_array[0] * 255), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)

    return superimposed, severity


def generate_demo_gradcam(img_array: np.ndarray, class_idx: int):
    """
    Generate a realistic-looking demo Grad-CAM when no model is loaded.
    Used in DEMO_MODE only.
    """
    h, w = 224, 224
    cam = np.zeros((h, w), dtype=np.float32)

    # Simulate lung region activation (0=Normal, 1=Other pattern, 2=Pneumonia)
    if class_idx == 2:  # Pneumonia - lower right region
        cx, cy, rx, ry = int(w * 0.65), int(h * 0.60), 55, 50
    elif class_idx == 1:  # Scattered
        cx, cy, rx, ry = int(w * 0.40), int(h * 0.45), 70, 65
    else:  # Normal - minimal
        cx, cy, rx, ry = int(w * 0.50), int(h * 0.50), 30, 30

    for y in range(h):
        for x in range(w):
            val = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            cam[y, x] = max(0, 1.0 - val)

    cam = cam / (cam.max() + 1e-8)
    severity = float(np.percentile(cam, 95))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(np.uint8(img_array[0] * 255), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)

    return superimposed, severity
