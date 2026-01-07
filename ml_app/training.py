import os
import time
import json
import threading
import numpy as np
import tensorflow as tf
from django.conf import settings

training_state = {
    'status': 'idle', 
    'progress': 0,
    'epochs': 0,
    'current_epoch': 0,
    'log': [],
    'final_accuracy': None,
}

class TrainingLogCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        training_state['current_epoch'] = epoch + 1
        training_state['progress'] = (epoch + 1) / training_state['epochs'] * 100
        log_entry = {k: round(v, 4) for k, v in logs.items()}
        log_entry['epoch'] = epoch + 1
        training_state['log'].append(log_entry)

    def on_train_begin(self, logs=None):
        training_state['status'] = 'running'
        training_state['log'] = []
        training_state['final_accuracy'] = None

    def on_train_end(self, logs=None):
        training_state['status'] = 'completed'
        if logs and 'accuracy' in logs:
            training_state['final_accuracy'] = round(logs['accuracy'], 4)

def train_model(epochs, batch_size, img_size, learning_rate, optimizer, dropout_rate, train_split, val_split, early_stopping_patience, page_limit):
    from ml_app.views import get_paths
    from ml_app.training_utils import stac_search, load_previews, pollution_proxy_labels

    training_state['epochs'] = epochs

    try:
        X_all = load_previews(["sentinel-2-l2a"], page_limit=page_limit)
    except Exception as e:
        training_state['status'] = f'error'
        training_state['log'].append({'error': str(e)})
        return

    y_all, _ = pollution_proxy_labels(X_all, img_size=img_size, percentile_thresh=60)
    y_all = y_all.astype(np.float32) # Use float32 for binary_crossentropy

    idx = np.random.permutation(len(X_all))
    n = len(idx)
    tr = int(train_split * n)
    va = int((train_split + val_split) * n)

    x_train, y_train = X_all[idx[:tr]], y_all[idx[:tr]]
    x_val, y_val = X_all[idx[tr:va]], y_all[idx[tr:va]]
    x_test, y_test = X_all[idx[va:]], y_all[idx[va:]]

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomZoom(0.1)
    ])

    def create_model(dropout_rate=0.2):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(*img_size, 3)),
            augment,
            tf.keras.layers.Rescaling(1./255),

            tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate if dropout_rate > 0 else 0),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    model = create_model(dropout_rate)

    opt_map = {
        'adam': tf.keras.optimizers.Adam(learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate),
    }
    
    model.compile(
        optimizer=opt_map.get(optimizer, tf.keras.optimizers.Adam(learning_rate)),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [TrainingLogCallback()]
    if early_stopping_patience > 0:
        early = tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience, restore_best_weights=True, monitor="val_loss")
        callbacks.append(early)
        
    model.fit(
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(512).batch(batch_size),
        validation_data=tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        epochs=epochs,
        callbacks=callbacks,
        verbose=0
    )

    MODEL_PATH, _, _ = get_paths()
    model.save(MODEL_PATH)

def train_model_in_background(*args, **kwargs):
    thread = threading.Thread(target=train_model, args=args, kwargs=kwargs)
    thread.start()
