# This script contains the functions required to run the hierarchical black hole merging simulation.

# The functions include:
# - get_OS(): Get the operating system of the user.
# - get_post_merge_params(input_params: str, op_system: str): Calculate the parameters of the remnant blackhole using the gwkik2 executable.
# - get_bh_data(): Get the Black Hole data from the file named 'bhlist'.
# - select_random_bh(bh_mass: np.array, kerr_param: np.array, num_samples: int): Randomly select Black Hole parameters from the given data.
# - extract_data(simulation_output: str): Extract relevant data from the simulation output.
# - convert_to_str(input_params_dict: dict): Convert a dictionary of input parameters to a space-separated string.
# - simulate_hierarchical_merging(op_system: str, bh_mass: np.array, kerr_param: np.array, total_generations: int, escape_velocity: dict, nuclear_escape=False, young_escape=False, globular_escape=False): Run a series of merging events constrained by the escape conditions.
# - run_n_simulations(op_system: str, bh_mass: np.array, kerr_param: np.array, n_simulations: int, max_generation: int, escape_velocity: dict, nuclear_escape: bool, young_escape: bool, globular_escape: bool): Runs a specified number of simulations of hierarchical black hole mergers.
# - get_inherent_merge_probability(n_sim: int, overall_result: dict, n_max_gen: int): Calculates the inherent merge probability for each generation in the simulations.
# - plot_merge_prob(inherent_probabilities: list, gen_count: int, n_max_gen: int, cluster_environment: str): Plot the merge probability for each generation.
# - plot_weighted_merge_prob(gen_distribution: list, n_max_gen: int, cluster_environment: str, q_vals: np.array, m1_vals: np.array): Plot the weighted merge probability for each generation.
# - each_gen_data(overall_result: dict, n_max_gen: int): Extract data for each generation in the simulations.
# - save_complete_simulation(file_path, result): Save the complete simulation data to a pickle file.
# - save_concat_simulation_data(filename: str, generation_data: dict): Save the simulation data to a text file with clear column headers.
# - save_each_gen_params(gen_data: dict, cluster_environment): Save the data for each generation in the simulations.


# ------------------------------------------------------------------------------------------------------------

# Importing required libraries

import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import platform
import pickle


def get_OS():
    """
    Get the operating system of the user.

    Returns:
    - str: The operating system of the user.

    Notes:
    - This function uses the platform.system() attribute to determine the operating system.
    - It returns a string representing the operating system (e.g., 'Linux', 'Windows', 'Mac').

    """
    
    # Get the operating system name
    os_name = platform.system().lower()

    # Return the operating system name
    if os_name == 'windows':
        return 'Windows'
    elif os_name[:5] == 'linux':
        return 'Linux-based'
    if os_name == 'darwin':
        return 'MacOS'
    else:
        return 'Unknown OS'

#----------------------------------------------------------------------------------------#

def get_post_merge_params(input_params: str, op_system: str) -> str:
    """
    Calculates the parameters of the remnant blackhole using the gwkik2 executable.

    Parameters:
    - input_params (str): A string containing the input values for the black holes' parameters.

    Returns:
    - str: The standard output (stdout) from the calculation.

    Notes:
    - This function constructs a command to run the gwkik2 executable with the provided input parameters.
    - The command is executed using subprocess.run(), capturing the output in text format.
    - The function returns the standard output (stdout) from the simulation.

    """
    
    if op_system == 'Windows':
        # Construct the input command (FOR WINDOWS)
        command = f'echo {input_params} | .\\gwkik2'

    elif op_system in ['Linux-based', 'MacOS']:
        # Construct the input command (FOR LINUX)
        command = f'echo {input_params} | ./gwkik2'
    
    else:
        raise ValueError('Operating system not supported! Use either Linux or Windows or MacOS')

    # Execute the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Return the standard output (stdout) from the simulation
    return result.stdout

#----------------------------------------------------------------------------------------#

def get_bh_data():
    """
    Get the Black Hole data from the file named 'bhlist'.

    Returns:
    - np.array: An array containing the Black Hole data (mass and Kerr parameter).

    Notes:
    - This function reads the data from the file named 'bhlist' and returns it as a NumPy array.
    - The file is assumed to have two columns: Mass and Kerr parameter, respectively. (with no headers)
    - The data is loaded using np.loadtxt() and returned as an array.

    """
    
    extensions = [".dat", ".txt", ".csv", ".json", ".yaml"]  # List of extensions to check

    # Try to load the data from the file with different extensions
    try:
        for ext in extensions:
            try:
                bh_data = np.loadtxt(f'bhlist{ext}')
                break

            except:
                continue

    except FileNotFoundError:
        # If neither file exists, return an empty array
        print('The file "bhlist" was not found with any of these extensions:', ', '.join(extensions))
        print('The file bhlist.dat does not exist in the directory.'\
            'You can either rename your own file to "bhlist" '\
            'or create a file with two columns: Mass and Kerr parameter, respectively (no header required).'\
            'The simulation further takes random values from this file as the parameters for the black holes.')
        print('Create a file with the required format and save it as "bhlist.dat" or "bhlist.txt" in the directory and try again!')
        return np.array([])

    # Copy data to shuffle
    data = bh_data.copy()
    np.random.shuffle(data)

    # Extract the columns for BH Mass and Kerr Parameter
    bh_mass = data[:, 0]
    kerr_param = data[:, 1]

    # Return the loaded data as a NumPy array
    return bh_mass, kerr_param

#----------------------------------------------------------------------------------------#

def select_random_bh(bh_masses: np.array, kerr_params: np.array, num_samples: int) -> tuple:
    """
    Randomly select Black Hole parameters from the given data.

    Parameters:
    - bh_mass (np.array): An array containing the Black Hole masses.
    - kerr_param (np.array): An array containing the Kerr parameters.

    Returns:
    - Random Black Hole mass and corresponding kerr parameter.

    Notes:
    - This function randomly selects a pair of Black Hole parameters from the given data.
    - The indices are chosen randomly, and the corresponding values are returned as a dictionary.

    """

    # Randomly select two indices
    idx = np.random.randint(low = 0, high = len(bh_masses), size=num_samples)

    # Extract the BH Mass and Kerr Parameter for the selected indices
    sample_bh_mass = bh_masses[idx]
    sample_kerr_param = kerr_params[idx]

    return sample_bh_mass, sample_kerr_param

#----------------------------------------------------------------------------------------#

def extract_data(simulation_output: str) -> np.array:
    """
    Extract relevant data from the simulation output.

    Parameters:
    - simulation_output (str): The standard output (stdout) from the simulation.

    Returns:
    - np.array: An array containing the extracted data, including:
      * Total Black Hole Mass after merging (bh_mass_fin)
      * Kick Velocity of the merged Black Hole (kick_velocity)
      * Effective Spin Parameter (xeff)
      * Kerr Parameter (afin)
      * Spin Parameter (sfin)
      * Polar Angle (thfin)

    Notes:
    - The function assumes that the simulation output contains specific information in a certain format.
    - It splits the output into lines, extracts the values for the merged Black Hole, and calculates relevant parameters.
    - The kick velocity is calculated from the component velocities (vprp1, vprp2, vpar).
    - The extracted values are returned as a NumPy array.

    """
    
    # Split the output into lines
    output_lines = simulation_output.strip().split('\n')

    # Extract values for Merged BH
    merged_bh_values = output_lines[-1].split()[:]
    bh_mass_fin = sum([float(output_lines[1:3][i].split()[0]) for i in range(2)])

    # Convert values to float and assign to variables
    vprp1, vprp2, vpar, xeff, afin, sfin, thfin = map(float, merged_bh_values)

    # Calculate kick velocity
    kick_velocity = np.sqrt(vprp1**2 + vprp2**2 + vpar**2)

    # Return the extracted data as a NumPy array
    return np.array([bh_mass_fin, kick_velocity, xeff, afin, sfin, thfin])

#----------------------------------------------------------------------------------------#

def convert_to_str(input_params_dict: dict) -> str:
    """
    Convert a dictionary of input parameters to a space-separated string. 
    This is done to make the input compatible with the calculation's required input type.

    Parameters:
    - input_params_dict (dict): A dictionary containing input parameters.

    Returns:
    - str: A space-separated string representing the input parameters.

    Notes:
    - This function takes a dictionary of input parameters and converts it into a string.
    - The values in the dictionary are extracted and joined with spaces to form the resulting string.

    Example:
    >>> input_params = {'param1': 10, 'param2': 0.5, 'param3': 'example'}
    >>> str_input = convert_to_str(input_params)
    >>> print(str_input)
    "10 0.5 example"
    """
    
    # Extract values from the dictionary and join them with spaces
    str_input = ' '.join(map(str, input_params_dict.values()))

    # Return the resulting string
    return str_input

#----------------------------------------------------------------------------------------#

def simulate_hierarchical_merging(op_system: str, bh_masses: np.array, kerr_params: np.array, total_generations: int, 
                                  escape_velocity: dict, initial_params: dict,
                                    nuclear_escape=False, young_escape=False, globular_escape=False) -> tuple:

    """
        Run a series of merging events constrained by the escape conditions.

        Parameters:
        - bh_mass (np.array): An array containing the Black Hole masses.
        - kerr_param (np.array): An array containing the Kerr parameters.
        - total_simulations (int): The total number of simulations to run.
        - nuclear_escape (bool): Flag to include nuclear escape condition.
        - young_escape (bool): Flag to include young escape condition.
        - globular_escape (bool): Flag to include globular cluster escape condition.

        Returns:
        - dict: A dictionary containing evolutionary information for each simulation, including:
        * 'bh_mass': List of Black Hole masses at each simulation step.
        * 'kick_vel': List of Kick Velocities at each simulation step.
        * 'xeff': List of xeff values at each simulation step.
        * 'afin': List of afin values at each simulation step.
        * 'sfin': List of sfin values at each simulation step.
        * 'thfin': List of thfin values at each simulation step.

        - np.array: An array containing the m1 values for each simulation. (M1 is the maximum of the two BH masses at each step)
        - np.array: An array containing the Q values for each simulation. (Q is the mass ratio of the two BHs at each step)

        Notes:
        - The function runs simulations iteratively, updating input parameters for each iteration.
        - Escape conditions can be set for nuclear, young, and globular cluster escape.
        - The function returns a dictionary containing the evolution information for each simulation step.
        - If an escape condition is met, the function stops and returns the available data up to that point.
    """

    # Generate the initial input parameters
    sample_bh_mass, sample_kerr_param = select_random_bh(bh_masses, kerr_params, num_samples=2)

    theta = np.random.randint(0, 180, size=2)
    phi = np.random.randint(0, 360, size=2)


    # Create a dictionary to store the evolutionary information of the parameters for each merger
    evol_info = {
        'bh_mass': [],
        'kick_vel': [],
        'xeff': [],
        'afin': [],
        'sfin': [],
        'thfin': []
    }

    first_input = convert_to_str(initial_params)
    str_input = first_input

    escape_velocity_conditions = {
        'Nuclear': nuclear_escape,
        'Young': young_escape,
        'Globular': globular_escape
    }
    
    m1 = []
    q = []
    
    # Calculting the initial M1 and Q values
    m1.append(max(initial_params['m1'], initial_params['m2']))
    q.append(m1[0] / max(initial_params['m1'], initial_params['m2']))


    # Initialize the number of simulations and generations
    num_generation = 2 # because we already have generation 1 black holes

    # Start merging the black holes until the escape conditions are met or the total number of maximum allowed generations is reached
    while num_generation <= total_generations:
        
        # Start the simulation and extract the data
        sim_output = extract_data(get_post_merge_params(str_input, op_system))
        bh_mass_fin, kick_vel, xeff, afin, sfin, thfin = map(float, sim_output)

        # Append the values
        evol_info['bh_mass'].append(bh_mass_fin)
        evol_info['kick_vel'].append(kick_vel)
        evol_info['xeff'].append(xeff)
        evol_info['afin'].append(afin)
        evol_info['sfin'].append(sfin)
        evol_info['thfin'].append(thfin)
        
        # Append the M1 and Q values
        previous_mass = float(str_input.split()[0])
        new_mass = float(str_input.split()[1])
        
        m1.append(max(previous_mass, new_mass))
        q.append(min(previous_mass, new_mass) / max(previous_mass, new_mass))
        
        # Check for escape conditions
        for escape_type, escape_condition in escape_velocity_conditions.items():
            if kick_vel > escape_velocity[escape_type] and escape_condition:
                return evol_info, np.array(m1), np.array(q)

        # Increment the number of generations
        num_generation += 1

        # Generate new input parameters
        sample_bh_mass, sample_kerr_param = select_random_bh(bh_masses, kerr_params, num_samples=1)
        theta2 = np.random.randint(0, 180, size=1)
        phi2 = np.random.randint(0, 360, size=1)

        # Update the input parameters
        new_input_info = {
            'm1': bh_mass_fin,
            'm2': sample_bh_mass[0],
            's1': sfin,
            'theta1': thfin,
            'phi1': xeff,
            's2': -20.0,
            'theta2': theta2[0],
            'phi2': phi2[0],
            'a1': afin,
            'a2': sample_kerr_param[0]
        }

        # Convert the input parameters from dictionary to a string
        str_input = convert_to_str(new_input_info)

    return evol_info, np.array(m1), np.array(q)

#----------------------------------------------------------------------------------------#

def plot_one_sim(result: dict, cluster_environment: str, output_file_path: str) -> None:
    # Use gridspec to create different evolution plots
    fig = plt.figure(figsize=(12, 10), dpi = 300)
    gs = fig.add_gridspec(3, 2)

    fig.suptitle(f'Hierarchical Merging in {cluster_environment} Cluster environment', fontsize = 16)

    # Plot the evolution of the BH Mass
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(result['bh_mass'], color = 'teal', linewidth = 2.0, alpha = 0.9)
    # ax1.plot(np.linspace(0, len(result['bh_mass']), 30), np.linspace(0, max(result['bh_mass']), 30),\
    #          color = 'red', linestyle = '--', linewidth = 1.5, alpha = 0.9)
    ax1.set_ylabel(r'Black Hole Mass ($M_{\odot}$)')
    ax1.set_title('Evolution of Black Hole Mass')


    # Plot the evolution of the Kick Velocity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(result['kick_vel'], color = 'teal', linewidth = 2.0, alpha = 0.9)
    ax2.set_ylabel('Kick Velocity')
    ax2.set_title('Evolution of Kick Velocity')

    # Plot the evolution of the Kerr Parameter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(result['afin'], color = 'teal', linewidth = 2.0, alpha = 0.9)
    # ax3.axhline(y=np.mean(result['afin'][-10:]), color = 'red', linestyle = '--', linewidth = 1.0, alpha = 0.9)
    # ax3.text(len(result['afin']) - 20, 1.02, 'Kerr Parameter ~ {:.2f}'.format(np.mean(result['afin'][-10:])),\
    #          fontsize = 10, bbox=dict(facecolor='white', alpha=0.5))
    ax3.set_ylim(min(result['afin']), max(result['afin']) + 0.1)
    ax3.set_ylabel('Kerr Parameter')
    ax3.set_title('Evolution of Kerr Parameter')


    # Plot the evolution of the Effective Spin Parameter
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(result['xeff'], color = 'teal', linewidth = 2.0, alpha = 0.9)
    # ax4.axhline(y=np.mean(result['xeff'][-10:]), color = 'red', linestyle = '--', linewidth = 1.0, alpha = 0.9)
    # ax4.text(len(result['xeff']) - 20, 0.7, 'Eff. Spin Parameter ~ {:.2f}'.format(np.mean(result['xeff'][-10:])),\
            # fontsize = 10, bbox=dict(facecolor='white', alpha=0.5))
    ax4.set_ylabel('Effective Spin Parameter')
    ax4.set_title('Evolution of Effective Spin Parameter')


    # Plot the evolution of the Inclination Angle 
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(result['thfin'], color = 'teal', linewidth = 2.0, alpha = 0.9)
    ax5.set_xlabel('Number of Simulations')
    ax5.set_ylabel('Inclination Angle')
    ax5.set_title('Evolution of Inclination angle')


    # Plot the evolution of the Spin Angular Momentum
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(result['sfin'], color = 'teal', linewidth = 2.0, alpha = 0.9)
    ax6.set_xlabel('Number of Simulations')
    ax6.set_ylabel('Spin Angular Momentum')
    ax6.set_title('Evolution of Spin Angular Momentum')


    fig.tight_layout()
    plt.savefig(f'{output_file_path}/Plots_{cluster_environment}/Evolution_{cluster_environment}_cluster.png', dpi = 300)

#----------------------------------------------------------------------------------------#

def run_n_simulations(op_system: str, bh_mass: np.array, kerr_param: np.array, n_simulations: int, max_generation: int,\
                      escape_velocity: dict, initial_params: dict,\
                       nuclear_escape: bool, young_escape: bool, globular_escape: bool) -> tuple:
    """
    Runs a specified number of simulations of hierarchical black hole mergers.

    Args:
        - bh_mass (np.array): An array containing the Black Hole masses.
        - kerr_param (np.array): An array containing the Kerr parameters.
        - n_simulations (int): The number of simulations to run.
        - max_generation (int): The maximum number of generations in a single simulation.
        - nuclear_escape (bool): Flag to include nuclear escape condition.
        - young_escape (bool): Flag to include young escape condition.
        - globular_escape (bool): Flag to include globular cluster escape condition.

    Returns:
        tuple: A tuple containing the following elements:
            - overall_result (dict): A dictionary containing simulation data for each
                generation across all simulations (averaged). Keys represent parameter names
                (e.g., 'bh_mass', 'kick_vel'), and values are lists containing averaged data points
                for each generation over all simulations.
            - overall_result_extend (dict): A dictionary containing simulation data for each
                generation across all simulations (concatenated). Keys represent parameter names,
                and values are lists containing all data points for each parameter from all generations
                and all simulations.
            - m1_vals (list): A list containing the initial primary black hole masses (m1)
                for each simulation.
            - q_vals (list): A list containing the mass ratios (q) for each simulation.
    """

    # Dictionary to store all data points for each generation across all simulations
    overall_result = {
        'bh_mass': [],
        'kick_vel': [],
        'xeff': [],
        'afin': [],
        'sfin': [],
        'thfin': []
    }

    # Dictionary to store all data points concatenated across simulations and generations
    overall_result_extend = {
            'bh_mass': [],
            'kick_vel': [],
            'xeff': [],
            'afin': [],
            'sfin': [],
            'thfin': []
        }

    # List of parameters to extract from simulation results
    extract_params = ['bh_mass', 'kick_vel', 'xeff', 'afin', 'sfin', 'thfin']

    # Lists to store initial primary masses (m1) and mass ratios (q)
    m1_vals = []
    q_vals = []

    # Loop through the desired number of simulations using a progress bar
    for i in tqdm(range(n_simulations)):
        # Run a single hierarchical merging simulation
        result = simulate_hierarchical_merging(op_system, bh_mass, kerr_param, total_generations = max_generation, \
                                               escape_velocity=escape_velocity, initial_params=initial_params,\
                                               nuclear_escape=nuclear_escape, young_escape=young_escape, globular_escape=globular_escape)
        
        # Extract simulation data (evolution of parameters)
        evolution_result = result[0]

        # Store initial primary mass (m1) and mass ratio (q) for this simulation
        m1_vals.append(result[1])
        q_vals.append(result[2])
        
        # Accumulate data for each parameter across simulations
        for param in extract_params:
            overall_result[param].append(evolution_result[param])
            overall_result_extend[param].extend(evolution_result[param])

    return overall_result, overall_result_extend, m1_vals, q_vals
    
#----------------------------------------------------------------------------------------#

def get_inherent_merge_probability(n_sim: int, overall_result: dict, n_max_gen: int) -> list:

    """
    Calculates the inherent merge probability for each generation in the simulations.

    Args:
        n_sim (int): The number of simulations performed.
        overall_result (dict): A dictionary containing simulation data.
            Expected key is 'bh_mass' with values as lists representing black hole masses for
            each generation in a simulation.
        n_max_gen (int): The maximum number of generations considered (inclusive).

    Returns:
        list: A list of length n_max_gen + 1, where each element represents the
            number of simulations that ended at the corresponding generation (index).
    """

    # Initialize a list to store the count for each generation
    gen_count = [0] * (n_max_gen-1)

    gen_distribution = []

    # Loop through black hole mass lists (representing generations) in each simulation
    for simulation_data in overall_result['bh_mass']:

        # Get the generation length (number of elements) for the current simulation
        generation_length = len(simulation_data)

        # Increment the count for the corresponding generation
        gen_count[generation_length-1] += 1

        # Append the maximum generation in the list of gen_distribution
        gen_distribution.append(generation_length)

    
    # Inherent merge probability is the count for each generation normalized by total simulations
    # (assuming equal probability for each simulation)
    inherent_probabilities = [count / n_sim for count in gen_count]

    return inherent_probabilities, gen_count, gen_distribution

#----------------------------------------------------------------------------------------#

def plot_merge_prob(inherent_probabilities: list, gen_dist: int, n_max_gen: int, cluster_environment: str) -> None:

    """
    Generates a histogram and line plot to visualize the merging probabilities 
    across generations for a hierarchical black hole merger simulation.

    Parameters:
        inherent_probabilities (list): A list containing the inherent merging 
                                       probability for each generation (except the first).
        gen_count (int): The generation at which the escape condition was met 
                         (or the maximum number of generations if not met).
        n_max_gen (int): The maximum number of generations allowed in the simulation.
        cluster_environment (str): The type of cluster environment simulated 
                                     (e.g., 'Nuclear', 'Young', 'Globular').

    Returns:
        None

    This function creates a plot with the following elements:
        - A histogram showing the distribution of the generation at which escape 
          conditions were met across multiple simulations.
        - A line plot showing the inherent merging probability for each generation 
          (except the first).
        - Title: Descriptive title mentioning the cluster environment.
        - X-axis label: "Generations"
        - Y-axis label: "Probability"
        - The plot is saved as a PNG image.
    """

    labels = ['Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth'][:n_max_gen-1]

    plt.figure(dpi = 200)
    plt.hist(gen_dist, bins=np.arange(0.5, n_max_gen+0.5), density = True, edgecolor='black', align='mid', color='teal')
    plt.plot(np.arange(1, n_max_gen), inherent_probabilities, 'o-', color = 'orange')
    plt.title(f'Merging probabilities in {cluster_environment} cluster environment', fontsize = 11)
    plt.xticks(np.arange(1, n_max_gen), labels, fontsize = 9)
    plt.xlabel('Generations', fontsize = 10)
    plt.ylabel('Probability', fontsize = 10)

    plt.savefig(f'Plots_{cluster_environment}/Merging_probabilities_{cluster_environment}.png', dpi = 200)

#----------------------------------------------------------------------------------------#

def plot_weighted_merge_prob(gen_distribution: list, n_max_gen: int, cluster_environment: str, q_vals: np.array, m1_vals: np.array) -> None:

    """
    Generates a weighted histogram to visualize the merging probabilities 
    across generations for a hierarchical black hole merger simulation, 
    taking into account the final mass ratio (q) and primary black hole mass (M1) 
    of each simulation.

    Parameters:
        gen_distribution (list): A list containing the generation at which the 
                                 escape condition was met (or the maximum number 
                                 of generations if not met) across multiple simulations.
        n_max_gen (int): The maximum number of generations allowed in the simulation.
        cluster_environment (str): The type of cluster environment simulated 
                                     (e.g., 'Nuclear', 'Young', 'Globular').
        q_vals (np.array): A NumPy array containing the mass ratio (q) values 
                           for each generation (including the first) of each simulation.
        m1_vals (np.array): A NumPy array containing the primary black hole mass (M1) 
                            values for each generation (including the first) of each simulation.

    Returns:
        None

    This function creates a weighted histogram with the following elements:
        - The x-axis shows the generation at which escape conditions were met.
        - The y-axis shows the probability density.
        - Each simulation outcome contributes to the histogram based on a weight 
          calculated using a formula (alpha and beta are user-defined constants) 
          that considers the final mass ratio (q) and primary black hole mass (M1) 
          of that particular simulation. This weighting reflects the 
          detection probability based on these factors (Ref. Fishbach & Holz, 2017, ApJ, 851, L25).
        - Title: Descriptive title mentioning the cluster environment.
        - X-axis label: "Generations"
        - Y-axis label: "Probability"
        - The plot is saved as a PNG image.
    """

    last_qs = np.array([q_val[-1] for q_val in q_vals])
    last_m1s = np.array([m1_val[-1] for m1_val in m1_vals])

    labels = ['Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth'][:n_max_gen-1]

    # Calculate weights based on the detection volume formula
    alpha = 2.2
    beta = 0.72

    # Volume fraction V is proportional to m_1^alpha * q^beta (Ref. Fishbach & Holz, 2017, ApJ, 851, L25)
    weights = np.power(last_m1s, alpha) * np.power(last_qs, beta)

    plt.figure(dpi = 200)
    plt.hist(gen_distribution, bins = np.arange(0.5, n_max_gen+0.5), weights = weights, density = True, edgecolor='black', align='mid', color='coral')
    plt.xticks(np.arange(1, n_max_gen), labels, fontsize = 9)
    plt.xlabel('Generations', fontsize = 10)
    plt.ylabel('Probability', fontsize = 10)

    plt.title(f'Weighted merging probabilities in {cluster_environment} cluster environment', fontsize = 11)
    
    plt.savefig(f'Plots_{cluster_environment}/Weighted_Merging_probabilities_{cluster_environment}.png', dpi = 200)

#----------------------------------------------------------------------------------------#

def each_gen_data(overall_result: dict, n_max_gen: int) -> dict:

    """
    Organizes simulation data from a dictionary (`overall_result`) into a 
    structured format separated by generation.

    Parameters:
        overall_result (dict): A dictionary containing the overall simulation results, 
                                where keys are parameter names (e.g., 'bh_mass', 'kick_vel') 
                                and values are lists containing data points for all generations.
        n_max_gen (int): The maximum number of generations specified in the simulation.

    Returns:
        dict: A dictionary with keys representing generation numbers (e.g., 'Gen_1', 'Gen_2') 
              and values as dictionaries containing lists of data for each parameter 
              extracted from the corresponding generation in the overall results.

    This function helps analyze simulation results by separating data for each generation. 
    It handles cases where there might be fewer generations than expected by filling 
    missing entries with `np.nan`, to keep the length of all the arrays same.
    """

    extract_params = ['bh_mass', 'kick_vel', 'xeff', 'afin', 'sfin', 'thfin']

    def create_generation_dictionaries(n_generations: int) -> dict:
        """
        This function creates a dictionary structure with keys for each generation
        and empty dictionaries as values to store simulation data.

        Args:
            n_generations (int): The number of generations specified by the user.

        Returns:
            dict: A dictionary with keys as generation numbers (1 to n_generations)
                and values as empty dictionaries for storing data.
        """

        generation_data = {}
        for generation in range(2, n_generations + 1):
            generation_data[f"Gen_{generation}"] = {
                'bh_mass': [],
                'kick_vel': [],
                'xeff': [],
                'afin': [],
                'sfin': [],
                'thfin': []
            }
        return generation_data


    generation_params = create_generation_dictionaries(n_max_gen)

    for i, generation in enumerate(list(generation_params.keys())):
        for param in extract_params:
            generation_params[generation][param].append([lists[i] if i < len(lists) else np.nan for lists in overall_result[param]])

    return generation_params

#----------------------------------------------------------------------------------------#

def save_complete_simulation(file_path, result) -> None:

    # Save the dictionary to a pickle file
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(result, pickle_file)

#----------------------------------------------------------------------------------------#

def save_concat_simulation_data(filename: str, generation_data: dict) -> None:
    """
    Saves the simulation data (dictionary) to a text file with clear column headers.

    Args:
        filename (str): The name of the file to save the data to.
        generation_data (dict): A dictionary containing simulation data for each generation.
            Keys represent parameter names (e.g., 'bh_mass', 'kick_vel'), and values are lists
            containing data points for each simulation.

    Raises:
        ValueError: If the generation data dictionary is empty.
    """

    # Check for empty generation data
    if not generation_data:
        raise ValueError("Generation data dictionary is empty. No data to save.")
    
    # Extract columns from the generation data
    columns = [generation_data[param] for param in ['bh_mass', 'kick_vel', 'xeff', 'afin', 'sfin', 'thfin']]

    # Stacking all the columns together
    arr = np.vstack(columns).T

    # Save the array to a text file
    np.savetxt(filename, arr, header='\t'.join(['bh_mass', 'kick_vel', 'xeff', 'afin', 'sfin', 'thfin']), delimiter='\t', comments='')

#----------------------------------------------------------------------------------------#

def save_each_gen_params(gen_data: dict, cluster_environment) -> None:

    # Save the data for each generation
    for gen in list(gen_data.keys()):
        save_concat_simulation_data(f'Results_data_{cluster_environment}/each_gen_data_{gen}.txt', gen_data[gen])

#----------------------------------------------------------------------------------------#

def plot_correlations(data_array: np.array, concat_array: np.array, cluster_env: str, n_max_gen: int) -> None:

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(4, 4)

    fig.subplots_adjust(hspace=0, wspace = 0)

    # fig.suptitle('Nuclear Cluster environment', fontsize=16)

    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(n_max_gen-1):
        ax1.hist(data_array[i, :, 0], bins=25, alpha=1, label=f'Generation {i+2}', histtype='step', density = True)
    ax1.tick_params(axis='y', labelleft=True, labelright=False, left=True, right=False)
    # ax1.set_yticks([0.02, 0.04, 0.06])
    # ax1.legend(bbox_to_anchor=(0.01, 1.4), loc = 'upper left', ncol = 4, fontsize = 11)


    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(concat_array[:, 0], concat_array[:, 2], s=1, alpha=0.3, color = 'steelblue')
    ax2.set_ylabel(r'$\chi_{eff}$')
    ax2.tick_params(axis='both',  which='both', left=True, right=True, top = True, bottom = True, direction = 'in')


    ax3 = fig.add_subplot(gs[1, 1])
    for i in range(n_max_gen-1):
        ax3.hist(data_array[i, :, 2], bins=25, alpha=1, label=f'Generation {i+2}', histtype='step', density = True)
    ax3.tick_params(axis='y',  which='both', labelleft=False, labelright=True, left=False, right=True)
    # ax3.set_yticks([2, 4, 6, 8])


    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(concat_array[:, 0], concat_array[:, 3], s=1, alpha=0.3, color = 'steelblue')
    ax4.set_ylabel('Eff. spin parameter')
    ax4.tick_params(axis='both',  which='both', left=True, right=True, top = True, bottom = True, direction = 'in')


    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(concat_array[:, 2], concat_array[:, 3], s=1, alpha=0.3, color = 'steelblue')
    ax5.tick_params(axis='both',  which='both', labelleft=False, labelright=False, left=True, right=True, top = True, bottom = True, direction = 'in')


    ax6 = fig.add_subplot(gs[2, 2])
    for i in range(n_max_gen-1):
        ax6.hist(data_array[i, :, 3], bins=25, alpha=1, label=f'Generation {i+2}', histtype='step', density = True)
    ax6.tick_params(axis='y', which = 'both', labelleft=False, labelright=True, left=False, right=True, top = True, bottom = True, direction = 'in')
    # ax6.set_yticks([5, 10, 15])


    ax7 = fig.add_subplot(gs[3, 0])
    ax7.scatter(concat_array[:, 0], concat_array[:, 1], s=1, alpha=0.3, color = 'steelblue')
    ax7.set_xlabel('BH Mass [M$_\odot$]')
    ax7.set_ylabel('Kick velocity [km/s]')
    ax7.tick_params(axis='both', which='both', left=True, right=True, top = True, bottom = True, direction = 'in')


    ax8 = fig.add_subplot(gs[3, 1])
    ax8.scatter(concat_array[:, 2], concat_array[:, 1], s=0.5, alpha=0.3, color = 'steelblue')
    ax8.set_xlabel(r'$\chi_{eff}$')
    ax8.tick_params(axis='both',  which='both', labelleft=False, labelright=False, left=True, right=True, top = True, bottom = True, direction = 'in')


    ax9 = fig.add_subplot(gs[3, 2])
    ax9.scatter(concat_array[:, 3], concat_array[:, 1], s=1, alpha=0.3, color = 'steelblue')
    ax9.set_xlabel('Eff. spin parameter')
    ax9.tick_params(axis='both', which='both', labelleft=False, labelright=False, left=True, right=True, top = True, bottom = True, direction = 'in')


    ax10 = fig.add_subplot(gs[3, 3])
    for i in range(n_max_gen-1):
        ax10.hist(data_array[i, :, 1], bins=25, alpha=1, label=f'Generation {i+2}', histtype='step', density = True)
    ax10.set_xlabel('Kick velocity [km/s]')
    ax10.tick_params(axis='both',  which='both', labelleft=False, labelright=True, left=False, right=True)

    ax10.legend(bbox_to_anchor=(0, 4.05), loc = 'upper left', ncol = 1, fontsize = 11)

    plt.savefig(f'Plots_{cluster_env}/Param_correlation_{cluster_env}.png', dpi = 300)

