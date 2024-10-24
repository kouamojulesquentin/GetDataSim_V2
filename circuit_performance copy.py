"""
Filename: circuit_performance.py
Description: This module contains functions to calculate various performance metrics
             of a circuit based on component values from a dataset. It includes
             calculations for resonance frequency, bandwidth, quality factor,
             and multiple resonance frequencies if applicable.

Author: Jules Quentin KOUAMO
Creation Date: 2024-09-20
Last Modified: 2024-10-15

Dependencies:
    - numpy
    - pandas
    - scipy (unused, but kept for potential future use)
"""

import numpy as np
import pandas as pd
from scipy import signal

def label_data_test(dataset, output_name):
    """
    Label the data based on the performance metrics of the circuit.
    
    Args:
        dataset (pd.DataFrame): A DataFrame containing the component values for each circuit.
        output_name (list): A list of strings containing the column names for the component values in the dataset.
    """
    # Read the dataset
    dataset = pd.read_csv(dataset, header=0, index_col=0)
    labeled_data = pd.DataFrame()

    component_1 = 89.0
    component_2 = 15e-3
    component_3 = 95e-6
    component_1_tolerance = 0.05
    component_2_tolerance = 0.10
    component_3_tolerance = 0.20

    # Calculate min and max values for each component based on tolerance
    component_1_min = component_1 - (component_1 * component_1_tolerance)
    component_1_max = component_1 + (component_1 * component_1_tolerance)
    component_2_min = component_2 - (component_2 * component_2_tolerance)
    component_2_max = component_2 + (component_2 * component_2_tolerance)
    component_3_min = component_3 - (component_3 * component_3_tolerance)
    component_3_max = component_3 + (component_3 * component_3_tolerance)

    def label_row(row):
        # Label each row as 'Pass' or 'Fail' 
        if (component_1_min <= row[output_name[0]] <= component_1_max and 
            component_2_min <= row[output_name[1]] <= component_2_max and 
            component_3_min <= row[output_name[2]] <= component_3_max):
            return 'Pass'
        else:
            return 'Fail'

    # Apply labeling function to each row
    labeled_data['label'] = dataset.apply(label_row, axis=1)

    # Combine the original dataset with the labels
    final = pd.concat([dataset[output_name], labeled_data], axis=1)
    final.to_csv('datas/final_database_test.csv')

    # Calculate and print pass/fail ratios
    pass_ratio = labeled_data['label'].value_counts(normalize=True).get('Pass', 0)
    fail_ratio = labeled_data['label'].value_counts(normalize=True).get('Fail', 0)
    print(f"Pass ratio: {pass_ratio:.2f}")
    print(f"Fail ratio: {fail_ratio:.2f}")
