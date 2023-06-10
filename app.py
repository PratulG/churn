import streamlit as st
import pandas as pd
import joblib

# Load the trained RandomForestClassifier model
model = joblib.load('best_model.pkl')

# Set up the Streamlit app
st.set_page_config(page_title='Customer Churn Prediction', layout='centered')

# Apply custom CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: blue;
    }
    .container {
        max-width: 500px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        font-size: 30px;
        color: blue;
        margin-bottom: 30px;
    }
    .label {
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
if prediction[0] == 0:
    st.markdown('**Prediction:** The customer is likely to **stay**.')
else:
    st.markdown('**Prediction:** The customer is likely to **churn**.')
st.markdown(f'Churn Probability: {prediction_prob[0]:.2%}')
