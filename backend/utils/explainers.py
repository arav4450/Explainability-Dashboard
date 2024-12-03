import shap
import lime
import lime.lime_tabular
import pandas as pd


def generate_shap_explanation(model,data):
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    return shap_values,explainer

def generate_lime_explanation(model,data):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=data.values,
        mode="classification"
    )
    explanation = explainer.explain_instance(data.iloc[0], model.predict_proba)
    return explanation.as_html()

