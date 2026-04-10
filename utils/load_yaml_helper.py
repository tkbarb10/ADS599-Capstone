import yaml
from pathlib import Path

def load_yaml(relative_path: str):
    root = Path(__file__).parent.parent
    file_path = root / relative_path
    with open(file_path, "r") as f:
        file = yaml.safe_load(f)
    return file