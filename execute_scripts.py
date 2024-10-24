"""
Filename: execute_scripts.py
Description: This module contains the function to execute Ocean scripts.

Author: Jules Quentin KOUAMO
Creation Date: 2024 - 09 - 20
Last Modified: 2024 - 10 - 11

Dependencies:
    - os
    - subprocess
"""

import os
import subprocess

def execute_ocean(path, design_dir, current_dir):
    """ 
    Execute the Ocean command in the specified design directory and wait for it to finish.
    
    Args:
        path (str): Relative path to the .ocn file
        design_dir (str): Directory where Ocean should be executed
        current_dir (str): Current working directory
    """
    try:
        # Change to the specified design directory
        os.chdir(design_dir)
        
        # Create the full Ocean command using the relative path of the .ocn file
        full_path = os.path.join(current_dir, path)
        command = f"ocean -restore {full_path}; exit"  # Add `; exit` to quit the shell after execution
        
        # Launch the Ocean command
        process = subprocess.Popen(command, shell=True)
        
        print(f"Ocean script started in the background with PID: {process.pid}")
        
        # Wait for the process to terminate
        process.wait()
        print("Ocean script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing Ocean script: {e}")
    finally:
        # Return to the original directory
        os.chdir(current_dir)
