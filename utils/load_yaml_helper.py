import yaml
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def load_yaml(relative_path: str):
    file_path = _ROOT / relative_path
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_sql_dir() -> Path:
    """Return the absolute path to the sql_scripts directory from settings.yaml."""
    settings = load_yaml("project_setup/settings.yaml")
    return _ROOT / settings['sql_dir']