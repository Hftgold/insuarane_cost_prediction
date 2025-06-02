# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:47:34 2025

@author: JANVI TYAGI
"""
import streamlit as st
import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open(r"C:\Users\JANVI TYAGI\Desktop\ml deploying\trained_model.sav", 'rb'))

# Prediction function
def predict_insurance_cost(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit user interface
def main():
    st.title("Insurance Cost Prediction.....")
    st.write("Enter the following details to predict the insurance cost:")

    # Input fields
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ("Male", "Female"))  # male: 0, female: 1
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ("Yes", "No"))  # yes: 0, no: 1
    region = st.selectbox("Region", ("Southeast", "Southwest", "Northeast", "Northwest"))  
    # southeast: 0, southwest: 1, northeast: 2, northwest: 3

    # Encode based on how the model was trained
    sex_encoded = 0 if sex == "Male" else 1
    smoker_encoded = 0 if smoker == "Yes" else 1
    region_mapping = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
    region_encoded = region_mapping[region]

    # Combine input data
    input_features = [age, sex_encoded, bmi, children, smoker_encoded, region_encoded]

    # Predict and display result
    if st.button("Predict Insurance Cost"):
        cost = predict_insurance_cost(input_features)
        st.success(f"Estimated Insurance Cost: USD {cost:.2f}")

if __name__ == '__main__':
    main()
