from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from utils.model_utils import load_model, preprocess_data
from utils.explainers import generate_shap_explanation, generate_lime_explanation
from fastapi.responses import JSONResponse

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI()

model = None

# Endpoint to upload and load the model
@app.post("/upload_model/")
async def upload_model(file: UploadFile = File(...)):
    global model
    file_path = os.path.join(MODEL_DIR, file.filename) 
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    model = load_model(file_path)
    return {"filename": file.filename, "message": "Model uploaded and loaded successfully"}

# Endpoint to upload and save the data
@app.post("/upload_data/")
async def upload_data(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)
    try:
        file.file.seek(0)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        return {"message": f"File '{file.filename}' uploaded successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")
    


# Endpoint to generate SHAP explanations
@app.post("/explain_shap/")
async def explain_shap(filename: str):
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    try:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Data file not found.")
        df = pd.read_csv(file_path)
        df = preprocess_data(df, model.feature_names_in_)
        shap_values,explainer = generate_shap_explanation(model, df)

        # Convert SHAP values to JSON-compatible format (list)
        #shap_values_json = [shap_values.values.tolist() for _ in shap_values]
        
        # Return SHAP values and base value (expected value)
        response_data = {
            "shap_values": shap_values.values.tolist(),
            "base_value": explainer.expected_value.tolist(),
            "feature_names": df.columns.tolist()
        }
    
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to generate LIME explanations
@app.post("/explain_lime/")
async def explain_lime(filename: str):
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    
    try:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Data file not found.")
        df = pd.read_csv(file_path)
        df = preprocess_data(df, model.feature_names_in_)
        explanation = generate_lime_explanation(model, df)
        return {"explanation": explanation}  # Ensure the 'explanation' key is returned
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
