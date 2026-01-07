import os
import shutil
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.static import serve
from PIL import Image
import io
import logging
import matplotlib
matplotlib.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Match the training script's settings
IMG_SIZE = (128, 128)
PREDICTION_THRESHOLD = 0.35 
CLASS_NAMES = ["clear", "pollution_like"]

def get_paths():
    """Get model and plot paths from Django settings."""
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'earthsearch_preview_haze_model.keras')
    PLOTS_DIR = os.path.join(settings.BASE_DIR, 'plots')
    PLOTS_MEDIA_DIR = os.path.join(settings.MEDIA_ROOT, 'plots')
    return MODEL_PATH, PLOTS_DIR, PLOTS_MEDIA_DIR

def sync_plots():
    """Sync plots from plots/ directory to media/plots/ directory."""
    try:
        _, PLOTS_DIR, PLOTS_MEDIA_DIR = get_paths()
        os.makedirs(PLOTS_MEDIA_DIR, exist_ok=True)
        if os.path.exists(PLOTS_DIR):
            for plot_file in os.listdir(PLOTS_DIR):
                if plot_file.endswith('.png'):
                    src = os.path.join(PLOTS_DIR, plot_file)
                    dst = os.path.join(PLOTS_MEDIA_DIR, plot_file)
                    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                        shutil.copy2(src, dst)
    except Exception:
        pass

_model = None

def get_model():
    """Load and cache the model."""
    global _model
    if _model is None:
        MODEL_PATH, _, _ = get_paths()
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
        else:
            _model = None
    return _model

def index(request):
    sync_plots()
    model = get_model()
    model_loaded = model is not None
    
    _, PLOTS_DIR, PLOTS_MEDIA_DIR = get_paths()
    plot_files = []
    if os.path.exists(PLOTS_DIR):
        plot_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
    elif os.path.exists(PLOTS_MEDIA_DIR):
        plot_files = [f for f in os.listdir(PLOTS_MEDIA_DIR) if f.endswith('.png')]
    
    context = {
        'model_loaded': model_loaded,
        'plot_files': plot_files,
        'plot_count': len(plot_files),
    }
    return render(request, 'ml_app/index.html', context)

def graphs(request):
    sync_plots()
    _, PLOTS_DIR, PLOTS_MEDIA_DIR = get_paths()
    plot_files = []
    if os.path.exists(PLOTS_DIR):
        plot_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
    elif os.path.exists(PLOTS_MEDIA_DIR):
        plot_files = [f for f in os.listdir(PLOTS_MEDIA_DIR) if f.endswith('.png')]
    
    training_plots = [f for f in plot_files if 'training' in f.lower()]
    confusion_plots = [f for f in plot_files if 'confusion' in f.lower()]
    viz_plots = [f for f in plot_files if any(x in f.lower() for x in ['gradcam', 'topk', 'montage', 'misclass'])]
    other_plots = [f for f in plot_files if f not in training_plots and f not in confusion_plots and f not in viz_plots]
    
    plot_categories = {
        'Training': training_plots,
        'Confusion Matrix': confusion_plots,
        'Visualizations': viz_plots,
    }
    
    context = {
        'plot_files': plot_files,
        'plot_categories': plot_categories,
        'other_plots': other_plots,
    }
    return render(request, 'ml_app/graphs.html', context)

@csrf_exempt
def predict_upload(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    model = get_model()
    if model is None:
        return JsonResponse({'error': 'Model not found. Please train the model first using the Train page or run main.py.'}, status=404)
    
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file provided'}, status=400)
    
    try:
        image_file = request.FILES['image']
        image_data = image_file.read()
        
        try:
            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.uint8)
            
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {img_array.shape}. Expected (H, W, 3)")
                
        except Exception as e:
            try:
                img_bytes = tf.constant(image_data)
                try:
                    img_tf = tf.io.decode_jpeg(img_bytes, channels=3, dct_method='INTEGER_FAST')
                except Exception:
                    try:
                        img_tf = tf.io.decode_png(img_bytes, channels=3)
                    except Exception:
                        try:
                            img_tf = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
                        except Exception:
                            raise ValueError(f"Unsupported image format. Error: {str(e)}")
                
                img_tf = tf.image.resize(img_tf, IMG_SIZE, method="bilinear")
                img_array = img_tf.numpy().astype(np.uint8)
                
            except Exception as tf_e:
                raise ValueError(
                    f"Could not decode image. Please ensure it's a valid JPEG, PNG, GIF, or BMP. "
                    f"PIL error: {str(e)}, TensorFlow error: {str(tf_e)}"
                )
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape: {img_array.shape}")
        
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        if model is None:
            return JsonResponse({'error': 'Model failed to load. Please check if the model file exists.'}, status=500)
        
        # Corrected Prediction Logic
        pollution_confidence = model.predict(img_array_expanded, verbose=0).flatten()[0]

        if pollution_confidence >= PREDICTION_THRESHOLD:
            predicted_class = "pollution_like"
            confidence = float(pollution_confidence)
        else:
            predicted_class = "clear"
            confidence = 1.0 - float(pollution_confidence)

        probabilities = {
            "clear": 1.0 - float(pollution_confidence),
            "pollution_like": float(pollution_confidence)
        }
        
        gradcam_path = None
        try:
            from ml_app.gradcam_utils import make_gradcam_heatmap, overlay_gradcam_on_image, save_gradcam_visualization
            
            original_img = img_array.copy()
            img_for_gradcam = img_array_expanded.astype(np.float32) / 255.0
            
            heatmap = make_gradcam_heatmap(img_for_gradcam, model, IMG_SIZE)
            overlay = overlay_gradcam_on_image(original_img, heatmap, alpha=0.5)
            
            import uuid
            from datetime import datetime
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'predictions'), exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            gradcam_filename = f'gradcam_{timestamp}_{unique_id}.png'
            gradcam_path_full = os.path.join(settings.MEDIA_ROOT, 'predictions', gradcam_filename)
            
            save_gradcam_visualization(
                original_img, heatmap, overlay, 
                predicted_class, confidence, 
                gradcam_path_full
            )
            
            gradcam_path = f'/media/predictions/{gradcam_filename}'
            
        except Exception as e:
            error_msg = f"Warning: Grad-CAM visualization failed: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            logger = logging.getLogger(__name__)
            logger.warning(error_msg)
        
        return JsonResponse({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'raw_predictions': probabilities,
            'gradcam_path': gradcam_path
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return JsonResponse({
            'error': str(e),
            'details': error_details if settings.DEBUG else None
        }, status=500)

def predict(request):
    model = get_model()
    context = {
        'model_loaded': model is not None,
        'class_names': CLASS_NAMES,
    }
    return render(request, 'ml_app/predict.html', context)

def train_page(request):
    from ml_app.training import training_state
    context = {
        'training_status': training_state['status'],
    }
    return render(request, 'ml_app/train.html', context)

@csrf_exempt
def start_training(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    from ml_app.training import train_model_in_background, training_state
    
    if training_state['status'] == 'running':
        return JsonResponse({'error': 'Training already in progress'}, status=400)
    
    epochs = int(request.POST.get('epochs', 50))
    batch_size = int(request.POST.get('batch_size', 64))
    learning_rate = float(request.POST.get('learning_rate', 0.001))
    optimizer = request.POST.get('optimizer', 'adam').lower()
    image_size = int(request.POST.get('image_size', 128))
    dropout_rate = float(request.POST.get('dropout_rate', 0.2))
    train_split = float(request.POST.get('train_split', 70)) / 100.0
    val_split = float(request.POST.get('val_split', 15)) / 100.0
    early_stopping_patience = int(request.POST.get('early_stopping_patience', 5))
    page_limit = int(request.POST.get('page_limit', 100))
    
    test_split = 1.0 - train_split - val_split
    if test_split < 0.05:
        return JsonResponse({'error': 'Train and Validation splits cannot exceed 95% total'}, status=400)
    
    train_model_in_background(
        epochs=epochs,
        batch_size=batch_size,
        img_size=(image_size, image_size),
        learning_rate=learning_rate,
        optimizer=optimizer,
        dropout_rate=dropout_rate,
        train_split=train_split,
        val_split=val_split,
        early_stopping_patience=early_stopping_patience,
        page_limit=page_limit
    )
    
    return JsonResponse({
        'success': True,
        'message': 'Training started'
    })

@csrf_exempt
def get_training_progress(request):
    from ml_app.training import training_state
    import copy
    state = copy.deepcopy(training_state)
    return JsonResponse(state)
