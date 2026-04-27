import numpy as np
import cv2
import tensorflow as tf


def _rgb01_to_bgr_uint8(img_array: np.ndarray) -> np.ndarray:
    rgb = np.clip(img_array[0] * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _last_conv_output(model: tf.keras.Model):
    """Last conv tensor, including inside nested sub-models (e.g. DenseNet base)."""
    for layer in reversed(model.layers):
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D,
            ),
        ):
            return layer.output
        if isinstance(layer, tf.keras.Model):
            inner = _last_conv_output(layer)
            if inner is not None:
                return inner
    return None


def demo_gradcam_class_index(pred_idx: int, num_classes: int) -> int:
    """
    Map model class index to generate_demo_gradcam pattern (0=Normal, 1=Other, 2=Pneumonia).
    Binary models use indices 0/1 → demo patterns 0 and 2.
    """
    if num_classes >= 3:
        return min(pred_idx, 2)
    return 0 if pred_idx == 0 else 2


def _overlay(img_array: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, float]:
    cam = np.maximum(cam, 0)
    if cam.shape != (224, 224):
        cam = cv2.resize(cam, (224, 224))
    if cam.max() > 0:
        cam = cam / cam.max()
    severity = float((cam > 0.5).mean())  # tỷ lệ vùng activation cao thay cho percentile dễ saturate

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_bgr = _rgb01_to_bgr_uint8(img_array)
    superimposed = cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)
    return superimposed, severity


def generate_gradcam(model: tf.keras.Model, img_array: np.ndarray, class_idx: int):
    """Grad-CAM gốc (Selvaraju et al., 2017)."""
    conv_out = _last_conv_output(model)
    if conv_out is None:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM")

    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[conv_out, model.output])
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()

    return _overlay(img_array, cam)


def generate_gradcam_pp(model: tf.keras.Model, img_array: np.ndarray, class_idx: int):
    """
    Grad-CAM++ (Chattopadhay et al., 2018).
    Sharper localization hơn Grad-CAM gốc, đặc biệt khi có nhiều vùng phát hiện.
    """
    conv_out = _last_conv_output(model)
    if conv_out is None:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM++")

    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[conv_out, model.output])
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_outputs, predictions = grad_model(img_tensor)
                # exp(score) cho công thức Grad-CAM++
                score = tf.exp(predictions[:, class_idx])
            grads = tape1.gradient(score, conv_outputs)
        grads2 = tape2.gradient(grads, conv_outputs)
    grads3 = tape3.gradient(grads2, conv_outputs)

    grads = grads if grads is not None else tf.zeros_like(conv_outputs)
    grads2 = grads2 if grads2 is not None else tf.zeros_like(conv_outputs)
    grads3 = grads3 if grads3 is not None else tf.zeros_like(conv_outputs)

    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1, 2))
    denom = 2.0 * grads2 + grads3 * global_sum
    denom = tf.where(tf.abs(denom) < 1e-8, tf.ones_like(denom), denom)
    alpha = grads2 / denom
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1).numpy()

    return _overlay(img_array, cam)


def generate_demo_gradcam(img_array: np.ndarray, class_idx: int):
    """Heatmap mô phỏng khi không có model (DEMO_MODE=true)."""
    h, w = 224, 224
    cam = np.zeros((h, w), dtype=np.float32)

    if class_idx == 2:  # Pneumonia
        cx, cy, rx, ry = int(w * 0.65), int(h * 0.60), 55, 50
    elif class_idx == 1:
        cx, cy, rx, ry = int(w * 0.40), int(h * 0.45), 70, 65
    else:
        cx, cy, rx, ry = int(w * 0.50), int(h * 0.50), 30, 30

    yy, xx = np.mgrid[0:h, 0:w]
    val = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    cam = np.clip(1.0 - val, 0.0, None).astype(np.float32)

    return _overlay(img_array, cam)
