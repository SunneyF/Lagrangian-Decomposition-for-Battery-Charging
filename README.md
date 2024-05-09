# Accelerating Lagrangian decomposition using time-block reformulation

## Overview
This repository supports the scientific research presented in the manuscript titled "Accelerating Lagrangian Decomposition with Time-Block Reformulations for Large-Scale Charging of Swappable Batteries," authored by Sunney Fotedar, Jiaming Wu, Balázs Kulcsár, and Rebecka Jörnsten.

## Required Files

To ensure that the scripts in this repository function correctly, please make sure that the following Python files are present in your working directory:

- `helper_function.py`: Contains utility functions used across multiple scripts. Import this file using `import helper_function as hf`.
- `Solve_YS_New_test.py`: Contains the `solve_ys` function which is crucial for solving part of the model.
- `Solve_YS_Original.py`: Includes the `solveYS_original` function, a variant of the model solving function.
- `Full_Model_fix_y.py`: This file contains several important models and functions:
  - `Full_Model_x_y`
  - `Full_Model_BinP`
  - `model_Px`
  - `MinCostFlowNetwork_ORTools_News`
  - `create_reference_matrix_and_count_switches`

Make sure all the above files are downloaded and present in the directory from which you run the scripts. This ensures that all imports and module functions work seamlessly.

### Main Script

```bash
python Final_ergodic.py

## Data Files
Data for this project is stored in JSON format. The files are named according to the template 'input_data_file_B_N_gamma', where `B` is the number of batteries, `N` is the number of ports, and `gamma` is a specific parameter related to the charging process.

### JSON File Structure
The JSON files contain several key pieces of data crucial for the computations and simulations described in the paper:

- **`B_ell`**: A list of lists, where each sublist contains the batteries allocated to each demand window for processing.
- **`alpha`**: A scalar value representing the alpha parameter used for the last demand window (`L`).
- **`elec_cost`**: A list of electricity costs, containing `T` entries listed in chronological order.
- **`n_l`**: A list containing the last time period used for each demand window.
- **`L`**: A scalar representing the number of demand windows.
- **`p`**: A dictionary where each key-value pair corresponds to a battery and its associated charging time in hours.
- **`w`**: A lexicographic coefficient used in the optimization model.

## Usage
The JSON data files are integral for running the simulations and computational models discussed in the manuscript. Users are advised to reference the specific data file relevant to their simulation scenario by adhering to the naming convention outlined above.

## Additional Information
For further details on the methodology, simulation results, and theoretical background, please refer to the full text of the manuscript. This README provides only a brief overview of the data structure and its usage within the broader context of the research project.



The main script that orchestrates the model's execution is `Final_ergodic.py`. To run this script, ensure that you have all the necessary files mentioned above in your working directory. You can execute the script from your command line using the following command:


