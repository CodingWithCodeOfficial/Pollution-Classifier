"""
ASGI config for pollution_detector project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_detector.settings')

application = get_asgi_application()

