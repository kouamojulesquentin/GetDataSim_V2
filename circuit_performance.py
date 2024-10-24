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

def compute_performance(dataset, param_names):
    """
    Calculate the performance metrics of a circuit based on component values. This function is specific to the circuit.
    
    Args:
        dataset (pd.DataFrame): A DataFrame containing the component values for each circuit.
        param_names (list): A list of strings containing the column names for the component values in the dataset.
    """
    # Read the dataset
    dataset = pd.read_csv(dataset, header=0, index_col=0)
    performance = pd.DataFrame()

    # Calculate performance metrics
    performance['resonance_frequency'] = 1 / (2 * np.pi * np.sqrt(dataset[param_names[1]] * dataset[param_names[2]]))
    performance['bandwidth'] = dataset[param_names[0]] / (2 * np.pi * dataset[param_names[1]])
    performance['quality_factor'] = performance['resonance_frequency'] / performance['bandwidth']
    performance['cutoff_frequency_1'] = performance['resonance_frequency'] - (performance['bandwidth'] / 2)
    performance['cutoff_frequency_2'] = performance['resonance_frequency'] + (performance['bandwidth'] / 2)

    # Combine the original dataset with the performance metrics
    final = pd.concat([dataset, performance], axis=1)
    final.to_csv('datas/final_database.csv')

def compute_tolerance_range():
    # Define component values and their tolerances
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

    num_simulations = 1000000

    # Generate random samples for each component within their tolerance range
    component_1_samples = np.random.uniform(component_1_min, component_1_max, num_simulations)
    component_2_samples = np.random.uniform(component_2_min, component_2_max, num_simulations)
    component_3_samples = np.random.uniform(component_3_min, component_3_max, num_simulations)

    def calculate_performance(component_1, component_2, component_3):
        # Calculate performance metrics for given component values
        resonance_frequency = 1 / (2 * np.pi * np.sqrt(component_2 * component_3))
        bandwidth = component_1 / (2 * np.pi * component_2)
        quality_factor = resonance_frequency / bandwidth
        cutoff_frequency_1 = resonance_frequency - (bandwidth / 2)
        cutoff_frequency_2 = resonance_frequency + (bandwidth / 2)
        return resonance_frequency, bandwidth, quality_factor, cutoff_frequency_1, cutoff_frequency_2

    # Calculate performance metrics for all samples
    performances = np.array([calculate_performance(c1, c2, c3) for c1, c2, c3 in zip(component_1_samples, component_2_samples, component_3_samples)])

    # Determine min and max values for each performance metric
    resonance_frequency_min, resonance_frequency_max = performances[:, 0].min(), performances[:, 0].max()
    bandwidth_min, bandwidth_max = performances[:, 1].min(), performances[:, 1].max()
    quality_factor_min, quality_factor_max = performances[:, 2].min(), performances[:, 2].max()
    cutoff_frequency_1_min, cutoff_frequency_1_max = performances[:, 3].min(), performances[:, 3].max()
    cutoff_frequency_2_min, cutoff_frequency_2_max = performances[:, 4].min(), performances[:, 4].max()

    # Print the tolerance ranges
    print(f"Resonance Frequency: {resonance_frequency_min:.6f} - {resonance_frequency_max:.6f}")
    print(f"Bandwidth: {bandwidth_min:.6f} - {bandwidth_max:.6f}")
    print(f"Quality Factor: {quality_factor_min:.6f} - {quality_factor_max:.6f}")
    print(f"Cutoff Frequency 1: {cutoff_frequency_1_min:.6f} - {cutoff_frequency_1_max:.6f}")
    print(f"Cutoff Frequency 2: {cutoff_frequency_2_min:.6f} - {cutoff_frequency_2_max:.6f}")

    return {
        'resonance_frequency': (resonance_frequency_min, resonance_frequency_max),
        'bandwidth': (bandwidth_min, bandwidth_max),
        'quality_factor': (quality_factor_min, quality_factor_max),
        'cutoff_frequency_1': (cutoff_frequency_1_min, cutoff_frequency_1_max),
        'cutoff_frequency_2': (cutoff_frequency_2_min, cutoff_frequency_2_max)
    }

def label_data(dataset, output_name):
    """
    Label the data based on the performance metrics of the circuit.
    
    Args:
        dataset (pd.DataFrame): A DataFrame containing the component values for each circuit.
        output_name (list): A list of strings containing the column names for the component values in the dataset.
    """
    # Read the dataset
    dataset = pd.read_csv(dataset, header=0, index_col=0)
    labeled_data = pd.DataFrame()

    # Compute tolerance ranges for performance metrics
    tolerance_range = compute_tolerance_range()

    def label_row(row):
        # Label each row as 'Pass' or 'Fail' based on tolerance ranges
        if (tolerance_range['resonance_frequency'][0] < row['resonance_frequency'] < tolerance_range['resonance_frequency'][1] and
            tolerance_range['bandwidth'][0] < row['bandwidth'] < tolerance_range['bandwidth'][1] and
            tolerance_range['quality_factor'][0] < row['quality_factor'] < tolerance_range['quality_factor'][1] and
            tolerance_range['cutoff_frequency_1'][0] < row['cutoff_frequency_1'] < tolerance_range['cutoff_frequency_1'][1] and
            tolerance_range['cutoff_frequency_2'][0] < row['cutoff_frequency_2'] < tolerance_range['cutoff_frequency_2'][1]):
            return 'Pass'
        else:
            return 'Fail'

    # Apply labeling function to each row
    labeled_data['label'] = dataset.apply(label_row, axis=1)

    # Combine the original dataset with the labels
    final = pd.concat([dataset[output_name], labeled_data], axis=1)
    final.to_csv('datas/final_database.csv')

    # Calculate and print pass/fail ratios
    pass_ratio = labeled_data['label'].value_counts(normalize=True)['Pass']
    fail_ratio = labeled_data['label'].value_counts(normalize=True)['Fail']
    print(f"Pass ratio: {pass_ratio:.2f}")
    print(f"Fail ratio: {fail_ratio:.2f}")
