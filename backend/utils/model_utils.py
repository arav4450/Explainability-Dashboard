import pickle
import pandas as pd
from typing import Any

# Load model from a given file path
def load_model(model_path: str) -> Any:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Preprocess input data for the model
def preprocess_data(data: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    Ensures the input data has the correct columns expected by the model.
    Args:
        data: Uploaded dataset in DataFrame format
        model_features: List of features the model expects
    """
    # Check for missing columns and fill with zeros 
    for feature in model_features:
        if feature not in data.columns:
            data[feature] = 0  

    # Ensure correct column order
    data = data[model_features]
    
    return data
