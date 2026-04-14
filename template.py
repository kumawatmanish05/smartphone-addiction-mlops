import os
from pathlib import Path

# Project name
project_name = "smartphone-addiction-mlops"

# List of files and directories
list_of_files = [
    f"{project_name}/data/raw/.gitkeep",
    f"{project_name}/data/processed/.gitkeep",

    f"{project_name}/notebooks/eda.ipynb",

    f"{project_name}/src/__init__.py",
    f"{project_name}/src/preprocessing.py",
    f"{project_name}/src/feature_engineering.py",
    f"{project_name}/src/train.py",
    f"{project_name}/src/predict.py",

    f"{project_name}/models/model.pkl",

    f"{project_name}/app/main.py",

    f"{project_name}/tests/__init__.py",

    f"{project_name}/requirements.txt",
    f"{project_name}/README.md",
    f"{project_name}/.gitignore",
    f"{project_name}/config.yaml",
]

# Create structure
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if not exists
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    # Create file if not exists or empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass