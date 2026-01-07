# Django Web App - Pollution Detector

A Django-based web application for viewing machine learning model results and making predictions on satellite images.

## Features

- 📊 **View Graphs**: Display all training curves, confusion matrices, and visualizations
- 🔮 **Make Predictions**: Upload images to classify them as "clear" or "pollution-like"
- 🎨 **Modern UI**: Beautiful, responsive interface

## Setup Instructions

### 1. Install Dependencies

Make sure you have Python 3.8+ installed, then install the required packages:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install Django tensorflow numpy Pillow matplotlib
```

### 2. Prepare Your Files

Make sure you have:
- `earthsearch_preview_haze_model.keras` - Your trained model file (in the project root)
- `plots/` directory - Contains your generated visualization PNG files

### 3. Run Database Migrations (First Time Only)

```bash
python manage.py migrate
```

### 4. Create a Superuser (Optional - for admin access)

```bash
python manage.py createsuperuser
```

### 5. Copy Plots to Media Directory

The app will automatically copy plots from the `plots/` directory to `media/plots/` when the server starts. If you need to manually sync them, the views.py will handle it automatically on first request.

### 6. Run the Development Server

```bash
python manage.py runserver
```

The server will start on `http://127.0.0.1:8000/` by default.

### 7. Access the Web App

Open your browser and navigate to:
- **Home**: http://127.0.0.1:8000/
- **Graphs**: http://127.0.0.1:8000/graphs/
- **Predict**: http://127.0.0.1:8000/predict/
- **Admin** (if created superuser): http://127.0.0.1:8000/admin/

## Usage

### Viewing Graphs

1. Go to the "Graphs & Visualizations" page
2. All plots from your `plots/` directory will be displayed
3. Plots are organized by category (Training, Confusion Matrix, Visualizations)
4. Click "View Full Size" to see the full-resolution image

### Making Predictions

1. Go to the "Predict" page
2. Upload an image (drag & drop or click to select)
3. Click "Predict" to get the classification
4. Results show:
   - Predicted class (clear or pollution-like)
   - Confidence score
   - Probabilities for both classes

## Project Structure

```
.
├── pollution_detector/     # Django project settings
├── ml_app/                 # Main Django app
├── templates/ml_app/       # HTML templates
├── static/ml_app/          # CSS and JavaScript
├── media/plots/            # Plot images (auto-generated)
├── plots/                  # Source plot directory
├── manage.py               # Django management script
└── requirements.txt        # Python dependencies
```

## Notes

- The model file (`earthsearch_preview_haze_model.keras`) must be in the project root directory
- Plots from `plots/` directory will be automatically copied to `media/plots/` on server startup
- For production deployment, configure proper static/media file serving (not included in this setup)

