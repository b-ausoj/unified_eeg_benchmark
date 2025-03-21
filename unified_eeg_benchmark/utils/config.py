import json
import os

_config = None

def load_config():
    global _config
    if _config is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))  
        config_path = os.path.join(script_dir, 'config.json')
        with open(config_path, "r") as config_file:
            _config = json.load(config_file)
    return _config

def get_config_value(key):
    config = load_config()
    return config.get(key)
