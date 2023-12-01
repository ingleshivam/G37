# prediction.py
import numpy as np

def predict_diabetes_status(classifier, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = classifier.predict(input_data_reshaped)
    return prediction[0]
