# How Prediction Works - Step by Step

## Overview
The prediction system uses a Convolutional Neural Network (CNN) trained to classify satellite images as either **"clear"** or **"pollution_like"** based on atmospheric haze.

## The Complete Flow

### Step 1: Image Upload & Preprocessing
```
User uploads image → Django receives file
    ↓
Decode image (JPEG/PNG) → Convert to TensorFlow tensor
    ↓
Resize to 96x96 pixels → Model expects this size
    ↓
Convert to numpy array (uint8, range 0-255)
```

### Step 2: Model Architecture (What Happens Inside)

The model processes the image through these layers:

1. **Data Augmentation** (only during training, skipped during inference)
   - Random flips, rotations, contrast adjustments
   - Actually SKIPPED when `model.predict()` is called (augmentation only in training)

2. **Rescaling Layer**
   - Converts pixel values from 0-255 → 0.0-1.0
   - `Rescaling(1./255)` divides each pixel by 255

3. **Convolutional Layers (Feature Extraction)**
   ```
   Conv2D(32 filters) → ReLU → MaxPooling
   Conv2D(64 filters) → ReLU → MaxPooling  
   Conv2D(128 filters) → ReLU → MaxPooling
   ```
   - Each Conv2D layer detects patterns (edges, textures, features)
   - MaxPooling reduces size and keeps important features
   - These layers learn to detect haze/pollution indicators

4. **Flatten Layer**
   - Converts 2D feature maps → 1D vector
   - Example: (height, width, channels) → (features)

5. **Dense Layers (Classification)**
   ```
   Dropout(0.2) → Regularization
   Dense(64 neurons) → ReLU activation
   Dense(2 neurons) → Softmax activation
   ```
   - Final dense layer outputs 2 probabilities
   - Softmax ensures they sum to 1.0

### Step 3: Model Output

The model outputs a probability distribution:
```python
predictions = model.predict(img_array, verbose=0)[0]
# Example output: [0.85, 0.15]
#                  ↑      ↑
#              "clear" "pollution_like"
```

### Step 4: Interpretation

```python
# Find the highest probability
predicted_class_idx = np.argmax(predictions)
# Example: predictions = [0.85, 0.15] → idx = 0

# Get class name
predicted_class = CLASS_NAMES[predicted_class_idx]
# CLASS_NAMES = ["clear", "pollution_like"]
# idx = 0 → "clear"

# Get confidence (the probability)
confidence = predictions[predicted_class_idx]
# confidence = 0.85 = 85% confident it's "clear"
```

### Step 5: Return Results

Returns JSON with:
- **predicted_class**: "clear" or "pollution_like"
- **confidence**: probability value (0.0 to 1.0)
- **probabilities**: both class probabilities
  ```json
  {
    "clear": 0.85,
    "pollution_like": 0.15
  }
  ```

## What the Model Learned

During training, the model learned:
- **"Clear" images**: High edge density, sharp features, good contrast
- **"Pollution_like" images**: Lower edge density, hazy appearance, reduced contrast

The model uses the same logic as the training labels:
- Computes edge density using Sobel edge detection
- Lower edge density = more haze = pollution-like
- Higher edge density = clear visibility = clear

## Example Prediction

**Input**: Satellite image (96x96x3 pixels)

**Model Processing**:
1. Rescale: [0-255] → [0.0-1.0]
2. Extract features via convolutions
3. Classify via dense layers
4. Output probabilities: [0.92, 0.08]

**Result**:
- Predicted: "clear"
- Confidence: 92%
- Interpretation: Image shows good visibility with minimal haze

## Code Location

- **Model Loading**: `ml_app/views.py` → `get_model()`
- **Prediction Logic**: `ml_app/views.py` → `predict_upload()`
- **Model Architecture**: `main.py` → `build_model()`
- **Training**: `main.py` (model.fit())

