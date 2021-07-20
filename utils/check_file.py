import os

def check_if_file_exists(file_path:str) -> bool:
    return os.path.isfile(file_path)