"""
Filename: generate_ocn_file.py
Description: This module contains the function to generate .ocn files for simulation.
             The function `generate_ocn_file` creates a .ocn file with the specified parameters,
             design details, and simulation setup. It supports parametric analysis with
             tolerance variations and exports the results.

Author: Jules Quentin KOUAMO
Creation Date: 2024 - 09 - 20
Last Modified: 2024 - 10 - 11

Dependencies:
    - random
"""

import random

def generate_ocn_file(path, design, num_params, param_names, param_values, outputs_name, number_of_simulations, param_tolerance, current_dir):
    """
    Generate an .ocn file for simulation with the given parameters.

    Args:
        path (str): Path to save the .ocn file
        design (list): List containing design details [design_name, library_name, cell_name]
        num_params (int): Number of parameters
        param_names (list): List of parameter names
        param_values (list): List of parameter values
        outputs_name (list): List of output names [input_name, output_name]
        number_of_simulations (int): Number of simulations to run
        param_tolerance (list): List of tolerances for each parameter
        current_dir (str): Current working directory
    """
    try:
        with open(path, 'w') as file:
            # Writing the simulator and design details
            file.write("simulator('spectre)\n")
            file.write(f'design("{design[0]}" "{design[1]}" "{design[2]}")\n\n')
            
            # Analysis setup (transient analysis)
            file.write("analysis('tran ?stop \"10\")\n")
            
            # Defining parameter variables
            for i in range(num_params):
                file.write(f'desVar("{param_names[i]}" {param_values[i]:.9g})\n')
            
            # Performing parametric analysis with tolerance variations
            for i in range(num_params):
                file.write(f'paramAnalysis("{param_names[i]}" ?values \'(')
                for _ in range(number_of_simulations):
                    variation = ((random.random() * 2 * param_tolerance[i]) - param_tolerance[i])
                    simulated_value = param_values[i] * (1 + variation / 100.0)
                    file.write(f'{simulated_value:.9g} ')
                file.write(")\n")
            
            # Closing all parametric analysis sections
            file.write(")" * num_params)
            
            # Running the parametric analysis and setting up result export
            file.write("\nparamRun()\n")
            file.write("selectResult('tran)\n")
            for output in outputs_name:
                file.write(f'ocnPrint(getData("/{output}") ?output "{current_dir}/datas/{output}.txt" ?from 0 ?to 10 ?step 0.01)\n')
            
            # Plotting the input and output waveforms
            file.write(f'plot(getData("/{outputs_name[0]}") getData("/{outputs_name[1]}"))\n')
        
        print(f"The file has been successfully generated at {path}")
    except IOError:
        print("Error opening the file.")
