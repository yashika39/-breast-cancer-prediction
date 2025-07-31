import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Sidebar model selection
st.sidebar.title("🔧 Model Settings")
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "SVM", "KNN"])

# Model setup
def get_model(name):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif name == "Logistic Regression":
        return LogisticRegression(max_iter=10000)
    elif name == "SVM":
        return SVC(probability=True)
    elif name == "KNN":
        return KNeighborsClassifier()
    return RandomForestClassifier()

model = get_model(model_choice)
model.fit(X, y)

# Title
st.title("🧠 Enhanced Breast Cancer Prediction App")
st.write("Enter tumor characteristics or upload a CSV file to make predictions.")

# CSV uploader
uploaded_file = st.file_uploader("📁 Upload CSV file for batch prediction", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    probs = model.predict_proba(df)
    df['Prediction'] = np.where(preds == 1, 'Benign', 'Malignant')
    df['Confidence (%)'] = np.round(np.max(probs, axis=1) * 100, 2)
    st.write(df)
else:
    # Manual input
    st.subheader("🔬 Input Tumor Features Manually")
    input_data = [st.number_input(f"{feature}", min_value=0.0, max_value=100.0, step=0.1) for feature in data.feature_names]
    input_array = np.array(input_data).reshape(1, -1)
    
    if st.button("🔍 Predict"):
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0]
        diagnosis = "Benign" if pred == 1 else "Malignant"
        st.success(f"Prediction: **{diagnosis}**")
        st.info(f"Confidence: {prob[1]*100:.2f}% for Benign, {prob[0]*100:.2f}% for Malignant")
        st.markdown("---")
        st.markdown("**🧾 Explanation:**")
        st.markdown("- **Benign**: Not cancerous (less serious).")
        st.markdown("- **Malignant**: Cancerous and may spread (more serious).")
        st.warning("⚠️ This is a machine learning prediction — not a medical diagnosis.")

# Feature importance
if model_choice == "Random Forest":
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=X.columns[indices], ax=ax)
    plt.title("Feature Importances")
    st.pyplot(fig)

# Retrain model
if st.button("🔄 Retrain Model"):
    model.fit(X, y)
    st.success("Model retrained!")

