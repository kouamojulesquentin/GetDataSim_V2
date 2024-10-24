"""
Filename: transform_data.py
Description: This module contains functions for transforming simulation data:
             - to_csv: Converts raw simulation output to CSV format
             - clean_data: Cleans the CSV data by removing quotes and specific lines
             - transpose_csv: Transposes the CSV data for further processing

Author: Jules Quentin KOUAMO
Creation Date: 2024 - 09 - 20
Last Modified: 2024 - 10 - 11

Dependencies:
    - csv
"""

import csv
import pandas as pd

def to_csv(input_file, output_file, num_params, param_names):
    """
    Convert raw simulation output to CSV format.

    Args:
        input_file (str): Path to the input file
        output_file (str): Path to the output CSV file
        num_params (int): Number of parameters
        param_names (list): List of parameter names
    """
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        is_data_section = False
        param_values = {param_names[i]: [] for i in range(num_params-1)}
        last_param_values = []

        for line in infile:
            if 'time (s)' in line:
                is_data_section = True
                continue
            
            for i in range(num_params-1):
                if f'{param_names[i]} =' in line:
                    param_value = line.split('=')[1].strip()
                    param_values[param_names[i]].append(param_value)
                    break
            
            if f'{param_names[-1]}' in line and is_data_section:
                last_param_values = line.split()[1:]
                continue
            
            if all(param_values.values()) and last_param_values:
                writer.writerow(['C1'])
                
                param_combined = ",".join(f"{' '.join(param_values[param_names[i]][-1] for i in range(num_params - 1))} {last_value}" 
                                        for last_value in last_param_values)
                
                writer.writerow(['0', param_combined])

                param_values = {param_names[i]: [] for i in range(num_params-1)}
                last_param_values = []

            if is_data_section:
                tokens = line.split()
                if tokens:
                    writer.writerow(tokens)

def clean_data_1(csv_file):
    """
    Clean the CSV data by removing quotes and lines containing '='.

    Args:
        output_file (str): Path to the CSV file to clean
    """
    with open(csv_file, 'r') as file:
        lines = file.readlines()
    
    lines = [line.replace('"', '') for line in lines]
    lines = [line for line in lines if '=' not in line]

    with open(csv_file, 'w') as file:
        file.writelines(lines)

def transpose_csv(input_file, output_file):
    """
    Transpose the CSV data for further processing.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output transposed CSV file
    """
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    data = rows
    transposed_data = []

    section_started = False
    current_section_data = []

    for row in data:
        if 'C1' in row:
            if section_started and current_section_data:
                transposed_section = list(zip(*current_section_data))
                transposed_data.append(transposed_section)
                transposed_data.append([])

            section_started = True
            current_section_data = []
        else:
            if section_started:
                current_section_data.append(row)
   
    if current_section_data:
        transposed_section = list(zip(*current_section_data))
        transposed_data.append(transposed_section)
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for section in transposed_data:
            if section:
                for row in section:
                    writer.writerow(row)

def clean_data_2(dataset):
    """
    Remove lines containing time-like data from the dataset.

    Args:
        dataset (str): Path to the CSV file to clean
    """
    with open(dataset, 'r') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        # Check if the line contains a sequence of numbers followed by 'm'
        if not any(part[:-1].isdigit() and part.endswith('m') for part in line.split(',')):
            cleaned_lines.append(line)

    with open(dataset, 'w') as file:
        file.writelines(cleaned_lines)




def transform_data(dataset, param_names, output_name):
    """
    Transform the CSV file into a dataset. The CSV has no index or column names.
    The first column of the dataset contains values separated by spaces.
    Transform this column into new columns based on the space-separated values.

    Args:
        dataset (str): Path to the CSV file to transform
        param_names (list): List of parameter names
        output_name (str): Name for the combined output column
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset, header=None, low_memory=False)

    # Split the first column into multiple columns
    new_columns = df[0].str.split(' ', expand=True)
    new_columns.columns = param_names  # Correct assignment for column names

    # Drop the original first column
    df = df.drop(columns=[0])

    # Concatenate the new columns (parameters) with the rest of the DataFrame
    df = pd.concat([new_columns, df], axis=1)

    # Combine the remaining columns into a single column (output_name)
    df[output_name] = df.iloc[:, len(param_names):].apply(lambda row: ','.join(row.dropna().astype(str)), axis=1)

    # Keep only the parameter columns and the combined output column
    df = pd.concat([new_columns, df[output_name]], axis=1)

    # Save the transformed DataFrame back to the CSV file
    df.to_csv(dataset)

    print(f"Data transformed and saved to {dataset}")
    print(f"Shape :  {df.shape}")



def convert_values(value):
    """
    Convert a value with a suffix to its numeric equivalent or to float.

    Args:
        value (str): The value to convert

    Returns:
        float: The converted numeric value
    """
    suffixes = {
        'a': 1e-18, 'z': 1e-21, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
        'm': 1e-3, 'c': 1e-2, 'd': 1e-1, '': 1, 'da': 1e1, 'h': 1e2, 'k': 1e3,
        'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, 'E': 1e18
    }

    parts = value.split(',')
    try:
        if len(parts) > 1:
            # Handle values with comma-separated parts
            return float(','.join(parts))
        else:
            # Check for suffixes
            for suffix, multiplier in suffixes.items():
                if value.endswith(suffix):
                    numeric_part = value[:-len(suffix)]
                    return float(numeric_part) * multiplier
            # If no suffix found, convert to float
            return float(value)
    except ValueError:
        # If conversion fails, return the original value
        return value

def convert_v_out(value):
    suffixes = {'a': 1e-18,'z': 1e-21, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'c': 1e-2, 'd': 1e-1, '': 1, 'da': 1e1, 'h': 1e2, 'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, 'E': 1e18}
    parts = value.split(',')
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

def apply_value_conversion(dataset,output):
    """
    Read a CSV file, apply value conversion to all columns, and save the result.

    Args:
        
        output_file (str): Path to save the output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(dataset, header=0, index_col=0)

    # Apply the conversion function to all columns
    for column in df.columns:
        df[column] = df[column].astype(str).apply(convert_values)
    df[output]    =  df[output].apply(convert_v_out)

    # Save the converted data to a new CSV file
    df.to_csv(dataset)


