<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Model Explainability Dashboard 🚀</h1>

<p>This project provides a dashboard for explaining black-box machine learning models using <strong>SHAP (SHapley Additive exPlanations)</strong> and <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong>. Built with <strong>Streamlit</strong> for the frontend and <strong>FastAPI</strong> for the backend, it allows users to upload data, select an explainability method, and visualize feature contributions.</p>

<hr>

<h2>✨ <strong>Features</strong></h2>
<ul>
    <li><strong>Upload CSV Files:</strong> Easily upload datasets for model explanation.</li>
    <li><strong>SHAP and LIME Support:</strong> Choose between SHAP and LIME for different types of explanations.</li>
    <li><strong>Streamlit Interface:</strong> User-friendly and intuitive web interface.</li>
</ul>

<h2>📦 <strong>Installation</strong></h2>

<h3><strong>1. Clone the Repository</strong></h3>
<pre><code>git clone https://github.com/arav4450/Explainability-Dashboard.git
cd Explainability-Dashboard</code></pre>

<h3><strong>2. Create a Virtual Environment</strong></h3>
<pre><code>python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`</code></pre>

<h3><strong>3. Install Dependencies</strong></h3>
<p>We use pip-tools for dependency management.</p>
<h4>Compile and Install:</h4>
<pre><code>pip install pip-tools
pip-compile dev.in
pip-sync dev.txt</code></pre>

<hr>

<h2>🚀 <strong>Running the Project</strong></h2>

<h3>1. Start the FastAPI Backend</h3>
<pre><code>cd backend
uvicorn main:app --reload</code></pre>

<h3>2. Start the Streamlit Frontend</h3>
<pre><code>cd frontend
streamlit run app.py</code></pre>

<hr>



</body>
</html>
