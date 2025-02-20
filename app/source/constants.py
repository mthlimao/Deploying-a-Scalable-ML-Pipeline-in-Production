from pathlib import Path

STARTER_PATH = Path(__file__).resolve().parent
DATA_PATH = STARTER_PATH.parent / 'data'
MODEL_PATH = STARTER_PATH.parent / 'model'

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]