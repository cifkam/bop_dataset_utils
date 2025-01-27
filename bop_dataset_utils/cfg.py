from pathlib import Path

LOCAL_DATA_DIR=None

def set_LOCAL_DATA_DIR(dir):
    global LOCAL_DATA_DIR
    assert dir.exists(), f"directory does not exist: {dir}"
    LOCAL_DATA_DIR = Path(dir)