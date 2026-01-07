#!/bin/bash
# Run Django development server

echo "Activating virtual environment..."
source .venv/bin/activate
echo "Starting Django server..."
echo "Once started, open: http://127.0.0.1:8000/"
echo ""
python3 manage.py runserver
