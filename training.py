import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.fft import fft
import joblib  
import numpy as np

def training(dataset):
    """ function to train the model """

    data = pd.read_csv(dataset, header=0, index_col=0)

    def time_to_frequency(v_out_str):
        N = len(v_out_str.split(','))

        # Convert the string to a list of floats
        v_out_list = list(map(float, v_out_str.split(',')))
        # Apply Fourier Transform
        frequency_domain = fft(v_out_list)
        frequency_domain = np.abs(frequency_domain[:N//2+1])  # Only take positive frequencies
  
        return frequency_domain

    # Add frequency-domain features to the dataset
    data['V_out_frequency'] = data['V_out'].apply(time_to_frequency)

    # Convert the time-domain and frequency-domain data into lists of floats
    X_time = data['V_out'].apply(lambda x: list(map(float, x.split(','))))
    X_freq = data['V_out_frequency'].apply(lambda x: list(x))  # Already in frequency-domain as floats
    y = data['label']  # Assuming you have a 'label' column for classification

    # Split the data for time-domain
    X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time.tolist(), y, test_size=0.2, random_state=42)
    
    # Split the data for frequency-domain
    X_train_freq, X_test_freq, y_train_freq, y_test_freq = train_test_split(X_freq.tolist(), y, test_size=0.2, random_state=42)

    # Define a generalized grid search for time and frequency data
    param_grid = [
        {
            'scaler': [StandardScaler(), MinMaxScaler()],
            'classifier': [SVC()],
            'classifier__C': [0.1, 1, 10]
        },
        {
            'scaler': [StandardScaler(), MinMaxScaler()],
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [50, 100, 200]
        }
    ]

    # Create a pipeline for both datasets (time and frequency)
    def run_grid_search(X_train, y_train, X_test, y_test, domain):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Placeholder, will be replaced in GridSearchCV
            ('classifier', SVC())  # Placeholder, will be replaced in GridSearchCV
        ])

        # Run the grid search
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get the best model and score
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the model for the current domain (time or frequency)
        model_filename = f"best_model_{domain}.joblib"
        joblib.dump(best_model, model_filename)
        print(f"Saved the best {domain} model as {model_filename}")
        
        # Return the best score, accuracy, and model filename for comparison
        return best_score, accuracy, model_filename

    # Run grid search on time-domain data
    print("Running grid search on time-domain data...")
    best_score_time, accuracy_time, model_time_filename = run_grid_search(X_train_time, y_train_time, X_test_time, y_test_time, domain='time')

    # Run grid search on frequency-domain data
    print("Running grid search on frequency-domain data...")
    best_score_freq, accuracy_freq, model_freq_filename = run_grid_search(X_train_freq, y_train_freq, X_test_freq, y_test_freq, domain='frequency')

    # Compare and save the final best model
    if accuracy_time > accuracy_freq:
        print(f"The time-domain model performed better with accuracy: {accuracy_time}")
        final_model_filename = model_time_filename
    else:
        print(f"The frequency-domain model performed better with accuracy: {accuracy_freq}")
        final_model_filename = model_freq_filename

    print(f"The final best model is saved as: {final_model_filename}")

    return final_model_filename

