# config.py
import os

def get_config():
    config = {
        "is_web": False,
        "storage_path": "recordings",
        "model_path": "models"
    }
    
    # Detect web environment
    if "streamlit.app" in os.environ.get("SERVER_SOFTWARE", ""):
        config["is_web"] = True
        config["storage_path"] = "/tmp/recordings"  # Use temp storage on web
        config["model_path"] = "/tmp/models"
    
    return config
