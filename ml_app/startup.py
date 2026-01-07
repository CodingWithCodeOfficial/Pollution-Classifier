import os
import logging
import sys
from django.conf import settings
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def initialize_model_and_data():
    """Initialize the model and generate plots by running main.py logic. Always runs, even if files exist."""
    try:
        print("🚀 Starting model initialization...")
        logger.info("🚀 Starting model initialization...")
        
        print("🔧 Running main.py to train model and generate plots...")
        print("   This may take a few minutes...")
        logger.info("🔧 Running main.py to train model and generate plots...")
        
        import threading
        
        def run_main():
            try:
                main_py_path = os.path.join(settings.BASE_DIR, 'main.py')
                
                if not os.path.exists(main_py_path):
                    print("⚠️ main.py not found. Skipping initialization.")
                    logger.warning("⚠️ main.py not found. Skipping initialization.")
                    return
                
                import importlib.util
                spec = importlib.util.spec_from_file_location("main", main_py_path)
                main_module = importlib.util.module_from_spec(spec)
                
                base_dir_str = str(settings.BASE_DIR)
                if base_dir_str not in sys.path:
                    sys.path.insert(0, base_dir_str)
                
                spec.loader.exec_module(main_module)
                
                print("✓ Model training and plot generation completed!")
                logger.info("✓ Model training and plot generation completed!")
                
            except Exception as e:
                error_msg = f"❌ Error running main.py: {str(e)}"
                print(error_msg)
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
        
        thread = threading.Thread(target=run_main, daemon=True)
        thread.start()
        print("   Initialization started in background...")
        logger.info("   Initialization started in background...")
        
    except Exception as e:
        error_msg = f"❌ Error during initialization: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
