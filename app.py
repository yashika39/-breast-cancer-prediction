import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Load dataset and train model
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# App title
st.title("🧠 Breast Cancer Prediction App")
st.write("Enter tumor features below and click Predict.")

# User input form
input_data = []
for feature in data.feature_names:
    value = st.number_input(feature, min_value=0.0, max_value=100.0, step=0.1)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    prediction = model.predict([input_data])[0]
    st.success("Prediction: **Benign**" if prediction == 1 else "Prediction: **Malignant**")
