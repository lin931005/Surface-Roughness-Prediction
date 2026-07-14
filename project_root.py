from pathlib import Path
import os

ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent))


def path(*parts):
    return ROOT.joinpath(*parts)


def str_path(*parts):
    return str(path(*parts))
