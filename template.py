import os 
from pathlib import Path 
import logging 

# logging over print() gives timestamps — useful for debugging pipeline runs
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s: ')

project_name = 'frauddetection'

# each notebook has a single responsibility — mirrors production ML workflow
list_of_files = [
    '.github/workflows/.gitkeep',
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/config/configuration.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/entity/config_entity.py',
    f'src/{project_name}/constant/__init__.py',
    'config/config.yaml',
    'params.yaml',
    'schema.yaml',
    'main.py',
    'Dockerfile',
    'setup.py',
    'data/README.md',                          
    'notebooks/01_eda.ipynb',                  
    'notebooks/02_timeseries.ipynb',           
    'notebooks/03_feature_engineering.ipynb',  
    'notebooks/04_modelling.ipynb',            
    'notebooks/05_evaluation.ipynb',           
    'requirements.txt',
    'README.md',
]

for filepath in list_of_files:
    # Path() normalises separators across Windows and Linux
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != '':
        # exist_ok=True prevents crash if directory already exists on re-runs
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    # only create file if missing or empty — protects existing work from being overwritten
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass 
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")