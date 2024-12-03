import streamlit as st
import requests
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO


st.title("Model Explainability Dashboard")

# Upload model
uploaded_model = st.file_uploader("Upload a trained model:", type=["pkl"])
if uploaded_model:
    files = {'file': uploaded_model}
    response = requests.post("http://localhost:8000/upload_model/", files=files)
    st.success(response.json()["message"])

# Upload data
uploaded_data = st.file_uploader("Upload dataset for explainability:", type=["csv"])
if uploaded_data:

    file_copy = BytesIO(uploaded_data.getvalue())
    data = pd.read_csv(file_copy)
    st.write("Uploaded Data Preview:")
    st.write(data.head())

    files = {'file': uploaded_data}
    response = requests.post("http://localhost:8000/upload_data/", files=files)
    st.success(response.json()["message"])

    # Choose explanation method
    explanation_method = st.selectbox("Choose explanation method:", ["SHAP", "LIME"])
    
    # Generate explanation based on selection
    if st.button(f"Generate {explanation_method} Explanation"):
        endpoint = (
            "http://localhost:8000/explain_shap/" if explanation_method == "SHAP"
            else "http://localhost:8000/explain_lime/"
        )
        
        response = requests.post(endpoint, params={"filename": uploaded_data.name})
        
        if response.status_code != 200:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            explanation_data = response.json()
            if explanation_method == "SHAP":
                # Extract SHAP values and feature names
                    shap_values = np.array(explanation_data["shap_values"])
                    feature_names = explanation_data["feature_names"]
                    
                    # Plot SHAP summary plot
                    st.subheader("SHAP Summary Plot:")
                    shap.summary_plot(shap_values, data[feature_names], show=False)
                    st.pyplot(plt.gcf())
            elif explanation_method == "LIME":
                st.components.v1.html(explanation_data["explanation"], height=900)


        
