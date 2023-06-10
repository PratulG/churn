import streamlit as st
import pandas as pd
import requests
import joblib

# URL of the trained RandomForestClassifier model file on GitHub
model_url = "https://github.com/PratulG/churn/raw/main/best_model.joblib"

# Download the model file from GitHub
response = requests.get(model_url)
response.raise_for_status()

# Save the downloaded model file locally
with open("best_model.joblib", "wb") as file:
    file.write(response.content)

# Load the downloaded model file
model = joblib.load("best_model.joblib")


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
st.header('Customer Churn Prediction')
st.subheader('Input Features')

st.markdown(f'Age: {age}')
st.markdown(f'Balance: {balance}')
st.markdown(f'Active Member: {active_member}')
st.markdown(f'Tenure: {tenure}')
st.markdown(f'Number of Products: {products_number}')

st.subheader('Prediction')
if prediction[0] == 0:
    st.markdown('The customer is likely to **stay**.')
else:
    st.markdown('The customer is likely to **churn**.')

st.subheader('Churn Probability')
st.markdown(f'{prediction_prob[0]*100:.2f}%')
