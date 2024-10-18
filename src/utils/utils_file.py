from pathlib import Path
import os


def make_checkpoint_dir(dir_checkpoint):
    path = Path(dir_checkpoint)
    # remove folder if it exists
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
