import pandas as pd
import numpy as np
from scipy.fft import fft
import joblib

def deployment(new_data_path, model_path):
    """
    Function to deploy the model on new data
    
    Parameters:
    -----------
    new_data_path : str
        Path to the CSV file containing the new data
    model_path : str
        Path to the saved model (.joblib)
    
    Returns:
    --------
    predictions : array
        Predictions for the new data
    probabilities : array
        Probabilities associated with each prediction (if available)
    """
    
    # Load the new data
    new_data = pd.read_csv(new_data_path, header=0, index_col=0)
    new_label = new_data['label']
    
    # Load the model
    model = joblib.load(model_path)
    
    # Check if it is a temporal or frequency model
    is_frequency_model = 'frequency' in model_path
    
    def time_to_frequency(v_out_str):
        N = len(v_out_str.split(','))
        v_out_list = list(map(float, v_out_str.split(',')))
        frequency_domain = fft(v_out_list)
        frequency_domain = np.abs(frequency_domain[:N//2+1])
        return frequency_domain
    
    def convert_v_out(value):
        suffixes = {'a': 1e-18, 'z': 1e-21, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'c': 1e-2, 'd': 1e-1, '': 1, 'da': 1e1, 'h': 1e2, 'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, 'E': 1e18}
        if isinstance(value, str):
            parts = value.split(',')
        else:
            return value
        converted_parts = []
        for part in parts:
            if part[-1].isalpha():
                num, suffix = part[:-1], part[-1]
                try:
                    converted_parts.append(float(num) * suffixes[suffix])
                except ValueError:
                    converted_parts.append(float('nan'))
            else:
                try:
                    converted_parts.append(float(part))
                except ValueError:
                    converted_parts.append(float('nan'))
        return ','.join(map(str, converted_parts))

    def apply_value_conversion(data, output):
        df = data
        for col in df.columns:
            df[col] = df[col].apply(convert_v_out)
        return df
    
    new_data = apply_value_conversion(new_data, 'V_out')

    # Prepare the data according to the model type
    if is_frequency_model:
        # Transform the data into the frequency domain
        X_new = new_data['V_out'].apply(time_to_frequency)
        X_new = X_new.apply(lambda x: list(x)).tolist()
        X_new = pd.DataFrame(X_new)
    else:
        X_new = new_data['V_out'].apply(lambda x: list(map(float, x.split(',')))).tolist()
        X_new = pd.DataFrame(X_new)
        X_new = new_data['V_out'].apply(lambda x: list(map(float, x.split(',')))).tolist()
    
    # Make predictions
    predictions = model.predict(X_new)
    
    # Get probabilities if the model allows it
    try:
        probabilities = model.predict_proba(X_new)
    except AttributeError:
        probabilities = None
        print("Note: This model does not provide probabilities for predictions")
    
    # Create a DataFrame with the results
    results = pd.DataFrame({
        'prediction': predictions
    })

    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            results[f'probability_class_{i}'] = probabilities[:, i]
    
    # Add last column to results
    
    results['reel_label'] = new_label
    # Add performance_difference_* to results if it differs from 0
    for col in new_data.columns:
        if col.startswith('percentage_difference_') and new_data[col].sum() != 0:
            results[col] = new_data[col]
    # Save the results
    results.to_csv('predictions_results.csv')
    print("Predictions have been saved in 'predictions_results.csv'")
    
    return predictions, probabilities
