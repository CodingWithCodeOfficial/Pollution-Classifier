from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class MlAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_app'
    
    def ready(self):
        import os
        if os.environ.get('RUN_MAIN') != 'true':
            return
        
        from ml_app import startup
        
        logger.info("🌍 Pollution Detector App Starting...")
        startup.initialize_model_and_data()
