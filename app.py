pip install pickle-mixin
import streamlit as st
import pandas as pd
import requests
import pickle

# URL of the trained RandomForestClassifier model file on GitHub
model_url = "https://github.com/PratulG/churn/raw/main/best_model.pkl"

# Download the model file from GitHub
response = requests.get(model_url)
response.raise_for_status()

# Save the model file locally
with open("best_model.pkl", "wb") as file:
    file.write(response.content)

# Load the downloaded model file
try:
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Set up the Streamlit app
st.title('Customer Churn Prediction')

# Create input fields for feature values
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance', min_value=0, value=1000)
active_member = st.selectbox('Active Member', ['Yes', 'No'])
tenure = st.number_input('Tenure', min_value=0, value=5)
products_number = st.number_input('Number of Products', min_value=1, value=1)

# Create a DataFrame with the user input
input_data = pd.DataFrame({
    'age': [age],
    'balance': [balance],
    'active_member': [1 if active_member == 'Yes' else 0],
    'tenure': [tenure],
    'products_number': [products_number]
})

# Make predictions using the loaded model
prediction = model.predict(input_data)
prediction_prob = model.predict_proba(input_data)[:, 1]

# Display the prediction result
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="title">Customer Churn Prediction</div>', unsafe_allow_html=True)

st.markdown('<div class="label">Age:</div>', unsafe_allow_html=True)
st.markdown(f'<div>{age}</div>', unsafe_allow_html=True)

st.markdown('<div class="label">Balance:</div>', unsafe_allow_html=True)
st.markdown(f'<div>{balance}</div>', unsafe_allow_html=True)

st.markdown('<div class="label">Active Member:</div>', unsafe_allow_html=True)
st.markdown(f'<div>{active_member}</div>', unsafe_allow_html=True)

st.markdown('<div class="label">Tenure:</div>', unsafe_allow_html=True)
st.markdown(f'<div>{tenure}</div>', unsafe_allow_html=True)

st.markdown('<div class="label">Number of Products:</div>', unsafe_allow_html=True)
st.markdown(f'<div>{products_number}</div>', unsafe_allow_html=True)

st.markdown('<div class="prediction">Prediction:</div>', unsafe_allow_html=True)
if prediction[0] == 0:
    st.markdown('<div class="prediction">The customer is likely to <strong>stay</strong>.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="prediction">The customer is likely to <strong>churn</strong>.</div>', unsafe_allow_html=True)

st.markdown(f'<div class="prediction">Churn Probability: {prediction_prob[0]:.2%}</div>', unsafe_allow_html=True)
