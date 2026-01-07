import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

def build_gradcam_model(model, input_shape):
    """Build a model that returns both the last conv layer output and predictions."""
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    last_conv_output = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_output = x

    if last_conv_output is None:
        raise ValueError("No Conv2D layer found in the model for Grad-CAM.")

    grad_model = tf.keras.Model(inputs=inp, outputs=[last_conv_output, x])
    return grad_model

def make_gradcam_heatmap(img_array, model, img_size=(96, 96)):
    """Generate Grad-CAM heatmap for an image."""
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)
    
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    grad_model = build_gradcam_model(model, (img_size[0], img_size[1], 3))
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()

def overlay_gradcam_on_image(img_uint8, heatmap, alpha=0.5, cmap='jet'):
    """Overlay Grad-CAM heatmap on original image."""
    h = tf.image.resize(
        heatmap[..., np.newaxis],
        (img_uint8.shape[0], img_uint8.shape[1])
    ).numpy().squeeze()
    
    colored = (cm.get_cmap(cmap)(h)[..., :3] * 255).astype(np.uint8)
    overlay = (alpha * colored + (1 - alpha) * img_uint8).astype(np.uint8)
    return overlay

def save_gradcam_visualization(img_uint8, heatmap, overlay, predicted_class, confidence, save_path):
    """Save Grad-CAM visualization as an image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    try:
        axes[0].imshow(img_uint8)
        axes[0].set_title(f'Original Image\nPredicted: {predicted_class}\nConfidence: {confidence:.2%}')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap\nModel Attention')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay\nAttention Regions')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150, format='png')
    finally:
        plt.close(fig)
    
    return save_path
