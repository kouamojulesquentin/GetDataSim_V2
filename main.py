"""
Filename: main.py
Description: Main script for collecting data from simulations.
             This script handles user interaction, OCN file generation,
             simulation execution, and result processing.

Author: Jules Quentin KOUAMO
Creation Date: 2024 - 09 - 20
Last Modified: 2024 - 10 - 11

Usage: Run this script directly to start the simulation process.
       Python 3.x is required.

Dependencies:
    - generate_ocn_file.py
    - transform_data.py
    - execute_scripts.py
"""

import os
import time
import random
from generate_ocn_file import generate_ocn_file
from transform_data import to_csv, clean_data_1, transpose_csv, clean_data_2, transform_data, apply_value_conversion
from execute_scripts import execute_ocean
from circuit_performance import compute_performance, compute_tolerance_range,label_data
from training import training

def main():
    """ Main function to collect data from simulations. """
    random.seed(time.time())
    
    # Variable Declaration
    # design_dir = input("Enter the absolute path of your Lib: ")
    # design = []
    # print("Enter the names for the design (e.g., Lib_name cell_name schematic):")
    # for _ in range(3):
    #     design.append(input())
    outputs_name = []
    print("Enter the input and output to observe in this order (e.g., V_in V_out):")
    for _ in range(2):
        outputs_name.append(input())
    num_params = int(input("Enter the number of parameters: "))
    # if num_params <= 0 or num_params > 50:
    #     print("The number of parameters must be between 1 and 50.")
    #     return
    param_names = []
    # param_values = []
    # param_tolerance = []
    for i in range(num_params):
        param_name = input(f"Enter the name of parameter {i + 1}: ")
    #     param_value = float(input(f"Enter the value for parameter {param_name}: "))
    #     param_tol = float(input(f"Enter the tolerance (as percentage) for parameter {param_name}: "))
        param_names.append(param_name)
    #     param_values.append(param_value)
    #     param_tolerance.append(param_tol)
    # number_of_simulations = int(input("Enter the number of simulations: "))
    # data_path = "datas/collect_data.ocn"
    current_dir = os.getcwd()
   
    simulation_result_output = f"{current_dir}/datas/{outputs_name[1]}.txt"
    simulation_result_input = f"{current_dir}/datas/{outputs_name[0]}.txt"
    cleaned_datas_output = f"{current_dir}/datas/cleaned_datas_output.csv"
    database_output = f"{current_dir}/datas/database_output.csv"
    cleaned_datas_input = f"{current_dir}/datas/cleaned_datas_input.csv"
    database_input = f"{current_dir}/datas/database_input.csv"
    final_database = f"{current_dir}/datas/final_database.csv"
    
    # Generate .ocn file
    #generate_ocn_file(data_path, design, num_params, param_names, param_values, outputs_name, number_of_simulations, param_tolerance, current_dir)
    
    # Execute the ocean script and wait for it to finish
    #execute_ocean(data_path, design_dir, current_dir)
    
    # Process the simulation results Output
    to_csv(simulation_result_output, cleaned_datas_output, num_params, param_names)
    clean_data_1(cleaned_datas_output)
    transpose_csv(cleaned_datas_output, database_output)
    clean_data_2(database_output)
    transform_data(database_output,param_names, outputs_name[1])
    apply_value_conversion(database_output, outputs_name[1])

    # Process the simulation results Input
    to_csv(simulation_result_input, cleaned_datas_input, num_params, param_names)
    clean_data_1(cleaned_datas_input)
    transpose_csv(cleaned_datas_input, database_input)
    clean_data_2(database_input)
    transform_data(database_input,param_names, outputs_name[0])
    apply_value_conversion(database_input, outputs_name[0])


    #compute performance
    compute_performance(database_output,param_names)

    #compute tolerance range
    compute_tolerance_range()

    # Label the data
    label_data(final_database,outputs_name[1])

    # train the model
    training(final_database)


if __name__ == "__main__":
    main()
