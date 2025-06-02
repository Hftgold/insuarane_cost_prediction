# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:35:28 2025

@author: JANVI TYAGI
"""

import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open(r"C:\Users\JANVI TYAGI\Desktop\ml deploying\trained_model.sav", 'rb'))

# Input data
input_data = (31, 1, 25.74, 0, 1, 0)

# Convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as the model expects input in 2D
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
prediction = loaded_model.predict(input_data_reshaped)
print('The insurance cost is USD', prediction[0])
