# Quick Start - Django Pollution Detector

## Step 1: Make sure dependencies are installed

```bash
pip install -r requirements.txt
```

## Step 2: Run migrations (first time only)

```bash
python manage.py migrate
```

## Step 3: Start the server

```bash
python manage.py runserver
```

Or use the script:
```bash
./run_server.sh
```

## Step 4: Open in browser

Once you see:
```
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

Open your browser and go to:
**http://127.0.0.1:8000/**

## Troubleshooting

If you see a blank page:
1. Check the terminal for error messages
2. Open browser Developer Tools (F12) and check the Console tab for errors
3. Make sure the server is actually running (you should see the startup message)
4. Try accessing http://127.0.0.1:8000/ directly (not localhost)

If you see "Page not found":
- Make sure you're going to http://127.0.0.1:8000/ (with the trailing slash)

If CSS doesn't load:
- The page should still show content, just without styling
- Make sure DEBUG = True in settings.py (it should be)
- Try: python manage.py collectstatic

