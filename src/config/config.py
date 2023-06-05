import os

from pathlib import Path

import ruamel.yaml as yaml


config_dir = str(Path(__file__).parent.absolute())
config_file_path = os.path.join(config_dir, "config.yaml")
CONFIG: dict = yaml.safe_load(open(config_file_path))
