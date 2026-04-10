import logging
from utils.load_yaml_helper import load_yaml
from pathlib import Path

settings = load_yaml("project_setup/settings.yaml")
log_file_path = settings['logging']['path']

def setup_logging(log_file=log_file_path):
    # Ensure the directory exists
    root = Path(__file__).parent.parent / "logs"
    if not root.exists():
        root.mkdir(parents=False, exist_ok=True)

    logger = logging.getLogger() # Note: Setting up the ROOT logger here
    logger.setLevel(logging.DEBUG)

    # Console Handler (Stream)
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    console_fmt = logging.Formatter('%(levelname)s - [%(filename)s: %(lineno)s] - %(message)s')
    debug_handler.setFormatter(console_fmt)

    # File Handler
    validation_path = root / log_file
    validation_handler = logging.FileHandler(validation_path)
    validation_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter('%(asctime)s - [%(filename)s: %(lineno)s] - %(levelname)s - \n%(message)s\n')
    validation_handler.setFormatter(file_fmt)

    # Add handlers
    logger.addHandler(debug_handler)
    logger.addHandler(validation_handler)
    
    return logger

# In your main script:
if __name__ == "__main__":
    log = setup_logging()
    log.info("Logger is configured and ready.")

