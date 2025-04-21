
import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('mushroom_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Title
st.title("🍄 Mushroom Edibility Predictor")

# User input form
features = {}
for feature in list(label_encoders.keys())[1:]:  # skip 'class'
    options = label_encoders[feature].classes_
    features[feature] = st.selectbox(f"Select {feature}", options)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([features])
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])
    result = model.predict(input_df)
    output = label_encoders['class'].inverse_transform(result)[0]
    st.success(f"The mushroom is **{output.upper()}**")
