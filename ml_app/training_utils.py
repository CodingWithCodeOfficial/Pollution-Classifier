import os
import json
import urllib.request
import urllib.error
import random
import numpy as np
import tensorflow as tf

IMG_SIZE = (96, 96)

def stac_search_paginated(api_url, collections, bbox, datetime, page_limit=100, max_pages=10):
    """STAC search with pagination."""
    features = []
    body = {
        "collections": collections,
        "bbox": bbox,
        "datetime": datetime,
        "limit": int(page_limit)
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers={"Content-Type":"application/json"})

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            page = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = "<no body>"
        raise RuntimeError(f"STAC search HTTP {e.code}: {e.reason}\nServer says: {err_body}")

    features.extend(page.get("features", []))

    next_href = None
    for link in page.get("links", []):
        if link.get("rel") == "next":
            next_href = link.get("href"); break

    pages_fetched = 1
    while next_href and pages_fetched < max_pages:
        try:
            with urllib.request.urlopen(next_href, timeout=30) as resp:
                page = json.loads(resp.read().decode("utf-8"))
                features.extend(page.get("features", []))
                next_href = None
                for link in page.get("links", []):
                    if link.get("rel") == "next":
                        next_href = link.get("href"); break
                pages_fetched += 1
        except Exception:
            break

    return features

def collect_preview_assets(feature):
    """Return PNG/JPEG preview asset URLs from a STAC feature."""
    urls = []
    assets = feature.get("assets", {})
    for key, meta in assets.items():
        href = meta.get("href", "")
        typ  = meta.get("type", "")
        roles = meta.get("roles", []) or []
        is_image = isinstance(typ, str) and (typ.startswith("image/png") or typ.startswith("image/jpeg"))
        likely_preview = any(r in roles for r in ["thumbnail","overview","visual","quicklook","browse"]) \
                         or key.lower() in ("thumbnail","overview","visual","quicklook","browse")
        if is_image and likely_preview and href.startswith("http"):
            urls.append(href)
    return urls

def fetch_bytes(url, max_retries=2):
    """Fetch image bytes from URL with retry logic."""
    for attempt in range(max_retries+1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
                if len(data) < 4:
                    raise ValueError("Image data too short")
                
                is_jpeg = data[:3] == b'\xff\xd8\xff'
                is_png = data[:4] == b'\x89PNG'
                is_gif = data[:6] in [b'GIF87a', b'GIF89a']
                is_bmp = data[:2] == b'BM'
                is_jp2 = len(data) >= 12 and data[4:8] == b'jP  ' or (len(data) >= 12 and data[:4] == b'\x00\x00\x00\x0c' and data[4:8] == b'jP  ')
                
                if is_jp2:
                    return data
                
                if is_jpeg or is_png or is_gif or is_bmp:
                    return data
                else:
                    return None
        except Exception as e:
            if attempt < max_retries: 
                import time
                time.sleep(0.75)
            else:
                return None
    return None

def decode_image_to_uint8(img_bytes):
    """Decode image bytes to uint8 array with robust error handling."""
    try:
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        return tf.image.resize(img, IMG_SIZE, method="bilinear").numpy().astype(np.uint8)
    except Exception:
        pass
    
    try:
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method='INTEGER_FAST')
        return tf.image.resize(img, IMG_SIZE, method="bilinear").numpy().astype(np.uint8)
    except Exception:
        pass
    
    try:
        img = tf.io.decode_png(img_bytes, channels=3)
        return tf.image.resize(img, IMG_SIZE, method="bilinear").numpy().astype(np.uint8)
    except Exception:
        pass
    
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)
        return img_array
    except Exception:
        pass
    
    return None

def build_preview_dataset(collections, bbox, datetime_range, page_limit=100):
    """Build dataset from STAC previews."""
    API_SEARCH = "https://earth-search.aws.element84.com/v1/search"
    features = stac_search_paginated(API_SEARCH, collections, bbox, datetime_range, page_limit=page_limit, max_pages=10)
    imgs = []
    skipped_count = 0
    for feat in features:
        urls = collect_preview_assets(feat)
        random.shuffle(urls)
        for u in urls[:1]:
            b = fetch_bytes(u)
            if b is None: 
                skipped_count += 1
                continue
            arr = decode_image_to_uint8(b)
            if arr is None:
                skipped_count += 1
                continue
            imgs.append(arr)
    
    if len(imgs) == 0:
        raise RuntimeError("No decodable image preview assets found in these collections.")
    
    if skipped_count > 0:
        print(f"Note: Skipped {skipped_count} unsupported image format(s) (e.g., JPEG2000)")
    
    X = np.stack(imgs, axis=0)
    return X

def haze_proxy_labels(X_uint8):
    """Generate haze proxy labels."""
    x = tf.convert_to_tensor(X_uint8, dtype=tf.float32) / 255.0
    gray = tf.image.rgb_to_grayscale(x)
    sob = tf.image.sobel_edges(gray)
    gx, gy = sob[..., 0], sob[..., 1]
    mag = tf.sqrt(tf.square(gx) + tf.square(gy))
    density = tf.reduce_mean(mag, axis=[1,2,3]).numpy()
    thresh = np.median(density)
    y = (density < thresh).astype(np.int64)
    return y, density, thresh

def build_model(input_shape=(96, 96, 3), num_classes=2, learning_rate=0.001, 
                optimizer='adam', dropout_rate=0.2):
    """Build the CNN model with customizable parameters."""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def make_tf_ds(x, y, batch=64, shuffle=True):
    """Create TensorFlow dataset."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle: 
        ds = ds.shuffle(len(x), seed=42)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
