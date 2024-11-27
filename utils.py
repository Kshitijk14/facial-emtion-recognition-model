import logging
import yaml
import os


def setup_logger(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_params(params_path):
    params_path = os.path.abspath(params_path)
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)
