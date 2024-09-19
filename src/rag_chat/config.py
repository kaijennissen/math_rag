import yaml

from rag_chat.project_root import ROOT


def load_config():
    with open(ROOT / "config/config.yaml", "r") as file:
        return yaml.safe_load(file)
