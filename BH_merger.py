import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from BH_BlackBox import *
import time
import shutil


script_description = "This script simulates n number of black hole mergers and stores the physical parameters for each generation of the hierarchical merger "\
                        "in different star cluster environments: Open/Young cluster, Globular cluster, Nuclear cluster. "


parser = argparse.ArgumentParser(description=script_description)

# Arguments for the directories
parser.add_argument('-path', type=str, default=os.getcwd(), help='Path to the directory containing the gwkik2 code (Default: Current working directory)')
parser.add_argument('-output', type=str, default=os.getcwd(), help='Path to the output directory (Default: Current working directory)')
parser.add_argument('-n_sim', type=int, help='Number of simulations to run')
parser.add_argument('-n_max_gen', type=int, default=4, help='Number of maximum mergers that should occur in a simulation (Default: 4)')

# Argument for the system
parser.add_argument('-cluster_env', type=str, help='Star cluster environment: [Open, Globular, Nuclear]')

# # Arguments for the black holes' (initial) parameters
# parser.add_argument('-m1', type=float, default=30, help='Mass of Black hole 1')
# parser.add_argument('-m2', type=float, default=30, help='Mass of Black hole 2')
# parser.add_argument('-s1', type=float, default=0, help='Spin (dimensional) of Black hole 1')
# parser.add_argument('-s2', type=float, default=0, help='Spin (dimensional) of Black hole 2')
# parser.add_argument('-theta1', type=float, default=0, help='Polar angle of Black hole 1 (in degrees)')
# parser.add_argument('-theta2', type=float, default=0, help='Polar angle of Black hole 2 (in degrees)')
# parser.add_argument('-phi1', type=float, default=0, help='Azimuthal angle of Black hole 1 (in degrees)')
# parser.add_argument('-phi2', type=float, default=0, help='Azimuthal angle of Black hole 2 (in degrees)')
# parser.add_argument('-a1', type=float, default=0.1, help='Kerr parameter (dimensionless spin) of Black hole 1')
# parser.add_argument('-a2', type=float, default=0.1, help='Kerr parameter (dimensionless spin) of Black hole 2')

args = parser.parse_args()

path = args.path
output = args.output
n_sim = args.n_sim
n_max_gen = args.n_max_gen
cluster_env = args.cluster_env.lower()

# m1 = args.m1
# m2 = args.m2
# s1 = args.s1
# s2 = args.s2
# th1 = args.theta1
# th2 = args.theta2
# ph1 = args.phi1
# ph2 = args.phi2
# a1 = args.a1
# a2 = args.a2

# Check if the path exists
if os.path.exists(path):
    os.chdir(path)

else:    
    print('The path does not exist. Please enter a valid path.')
    exit()

# Check if the output directory exists
if not os.path.exists(output):
    print('The output directory does not exist. Creating the directory...')
    os.makedirs(output)

# Check if the number of max generations is valid
if n_max_gen < 1:
    print('The number of maximum generations should be greater than 1.')
    exit()

elif n_max_gen > 10:
    print('The number of maximum generations should be less than 10.')
    exit()

# Check if the number of given simulations is valid
if n_sim < 1:
    print('The number of maximum generations should be greater than 1.')
    exit()


# Check if the inputs are proper
if cluster_env.lower() in ['open', 'young', 'globular', 'nuclear']:
    pass

else:
    print('Invalid cluster environment. Please enter a valid cluster environment: [Open (or Young), Globular, Nuclear]')
    exit()


# Get the operating system
op_sys = get_OS()

print('\n--------------------------------------------------\n'\
      'Python script initiated\n'\
        '--------------------------------------------------\n')

# time.sleep(1)

print("\nThis python script simulates hierarchical merging in different star cluster environements. The workflow of this script is as follows:\n"\
      "1. Choose random black hole mass and its corresponding kerr parameter from 'bhlist' file.\n"\
        "2. Calculate the post merger parameters using some equations written in the fortran code (gwkik2.f)\n"\
        "3. Randomly choose another black hole from the 'bhlist' file and merger it with the remnant black hole from the previous merger.\n"\
        "4. Continue this until either of two conditions are met:\n"\
            "\t-Kick velocity of the remnant black hole exceeds the escape velocity of the cluster environment.\n"\
            "\t-The hierarchical merging reaches the maximum generation given by the user.\n"\
        "5. This continues for n number of simulations.\n"\
        "6. Plots are produced to:\n"\
            "\t-Check the inherent probability of getting upto a specific generation of black hole in different cluster environments.\n"\
            "\t-Weight this probability by the observed volume fraction of the merger events from diffferent GW observatories.\n"\
            "\t-See how parameters vary with increasing black hole generation.\n"\
            "\t-Know the correlations between black hole parameters.\n"\
        )

input("Press Enter to continue...\n")

print("\nIt uses grkick code, which is code written in Fortran to simulate the black hole mergers. \n"\
                    "The references to the equations used in the grkick code is given in the file 'gwkick.f90'.\n\n"\
                    "--------------------------------------"\
                    "\nNOTE: Before running the script, make sure that the gwkik2 code is compiled and the executable is in the same directory as this script.\n"\
                    "--------------------------------------\n\n")

print('******************** Program initiates ********************')
print(f'Operating system: {op_sys}')
print(f'Number of simulations: {n_sim}')
print(f'Number of maximum mergers/generations in one simulation: {n_max_gen}')
print(f'Star cluster environment: {cluster_env}\n')

input('Press Enter to continue...')

# Import the data from the bhlist file
bh_mass, bh_kerr = get_bh_data()

print('\nBlack hole data imported successfully!')

print('\nSelecting random black hole masses and Kerr parameters...')
# Get random black hole masses and Kerr parameters
sample_bh_mass, sample_kerr_param = select_random_bh(bh_mass, bh_kerr, num_samples = 2)

print('\nRandom selection of Black Hole parameters successful!')


##### Define the remaining input parameters #####
theta1, theta2 = 90.0, 0.0
phi1, phi2 = 0.0, 0.0

initial_params = {
    'm1': sample_bh_mass[0], # Mass of Black Hole 1
    'm2': sample_bh_mass[1], # Mass of Black Hole 2
    's1': -20.0, # Spin of Black Hole 1 (with dimension)
    'theta1': theta1, # Polar angle of Black Hole 1
    'phi1': phi1, # Azimuthal angle of Black Hole 1
    's2': -20.0, # Spin of Black Hole 2 (with dimension)
    'theta2': theta2, # Polar angle of Black Hole 2
    'phi2': phi2, # Azimuthal angle of Black Hole 2
    'a1': sample_kerr_param[0], # Kerr parameter of Black Hole 1 (dimensionless spin)
    'a2': sample_kerr_param[1] # Kerr parameter of Black Hole 2 (dimensionless spin)
}



################### CHANGE THE ESCAPE CONDITIONS HERE ###################
escape_velocity = {
    'Nuclear': 500,
    'Young': 30,
    'Globular': 100,
}



print('\n********Initial parameters for the first black hole merger********')
print(initial_params)

print('\n\n********Escape velocities of different star cluster environments [km/s]********\n', escape_velocity)

print('\n You can change the magnitudes in the script.')

temp1 = input('Press Enter to continue...\nor to exit, type "exit"')
if temp1.lower() == 'exit':
    exit()

print(f'\nStarting the simulation for {cluster_env} cluster environment...')

# Check the cluster environment
nuclear_escape, young_escape, globular_escape = False, False, False

if cluster_env.lower() == 'nuclear':
    nuclear_escape = True

elif cluster_env.lower() in ['open', 'young']:
    young_escape = True

elif cluster_env.lower() == 'globular':
    globular_escape = True



# Create a directory Results to save the files
if os.path.exists(f'{output}/Results_data_{cluster_env}'):
    shutil.rmtree(f'{output}/Results_data_{cluster_env}')
    os.mkdir(f'{output}/Results_data_{cluster_env}')

else:
    os.mkdir(f'{output}/Results_data_{cluster_env}')


# Create a directory in Results to save the plots
if os.path.exists(f'{output}/Plots_{cluster_env}'):
    shutil.rmtree(f'{output}/Plots_{cluster_env}')
    os.mkdir(f'{output}/Plots_{cluster_env}')

else:
    os.mkdir(f'{output}/Plots_{cluster_env}')


if nuclear_escape:
    # Run the simulation
    result_one_sim, m1_one_sim, q1_one_sim = simulate_hierarchical_merging(op_sys, bh_mass, bh_kerr, n_max_gen, escape_velocity, nuclear_escape=nuclear_escape, young_escape=young_escape, globular_escape=globular_escape)

    print('One simulation completed successfully!\nYou can see the plots in the Results directory.')

    # Save the results
    plot_one_sim(result_one_sim, cluster_env)


sim_result, sim_result_extended_array, m1_vals, q_vals = run_n_simulations(op_sys, bh_mass, bh_kerr, n_simulations= n_sim, max_generation=n_max_gen, escape_velocity=escape_velocity,\
                                                                            nuclear_escape=nuclear_escape, young_escape=young_escape, globular_escape=globular_escape)
print('\nSimulation completed successfully!')

os.chdir(f'{output}')

# Save the results
print('Saving the results...')
save_complete_simulation(f'Results_data_{cluster_env}/Simulation_param_evolution_{cluster_env}.pkl', sim_result)
save_concat_simulation_data(f'Results_data_{cluster_env}/Concatenated_param_evol_{cluster_env}.csv', sim_result_extended_array)
save_complete_simulation(f'Results_data_{cluster_env}/M1_vals_{cluster_env}.pkl', m1_vals)
save_complete_simulation(f'Results_data_{cluster_env}/q_vals_{cluster_env}.pkl', q_vals)
print('Results saved successfully!\n')

# Plot the inherent merging probabilities for different generations in the star cluster environment
inherent_merge_probability, gens_count, gens_distrbution = get_inherent_merge_probability(n_sim, sim_result, n_max_gen)

plot_merge_prob(inherent_merge_probability, gens_count, n_max_gen, cluster_env)

plot_weighted_merge_prob(gens_distrbution, n_max_gen, cluster_env, q_vals, m1_vals)

print('\nPlots saved successfully!')


# Get black hole parameters for each generation of the complete simulation
generation_data = each_gen_data(sim_result, n_max_gen)

# Save the parameters for each generation in the simulation
print('\nSaving the parameters for each generation of the complete simulation...')
save_each_gen_params(generation_data, cluster_env)
print('Parameters saved successfully!')

print('\n\n*************************** Understand the results ****************************\n')

print("\nResults are stored in a dictionary format to access them as you wish, and also in .csv format, which contains the concatenated list parameter evolution.\n"\
      "For example, the first column represents the mass of all the remnant black holes across the simulation, irrespective of its generation.\n"\
        "M1 is the maximum of the two BH masses at each step/merger.\n"\
        "Q is the mass ratio of the two merging black hole at every merger.")

print('\nExiting the script...')





