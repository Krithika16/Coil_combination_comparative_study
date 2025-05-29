# Description: This script is used to analyze the SNR sensitivity of the various coil combination techniques
import sys
sys.path.insert(0, '../')

import numpy as np
import os
import bssfp

if __name__ == '__main__':

    # Load in the simulation
    filepath_phantom = '/rds/general/user/kb4317/home/coil_combination_project/simulated_phantom_data'
    os.chdir(filepath_phantom)

    # Load in the data
    phantom = np.load('simulated_phantom_12pcbssfp_coil_images.npy')

    # List of different std dev values and their corresponding SNR values - these standard deviations were calculated based on the
    # function in BLAH
    std_dev_list = [0.006, 0.003, 0.002, 0.0015, 0.0012, 0.001, 0.00087, 0.00075, 0.00067, 0.0006]
    num_realizations = 100
    SNR = np.arange(10,101, 10)

    for s in range(len(std_dev_list)):
        for n in range(num_realizations):

            # Add noise to the data
            noisy_phantom = bssfp.add_noise_gaussian(phantom, mu=0,sigma=std_dev_list[s])

            #Save the data
            filepath_save = filepath_phantom + '/SNR_' + str(SNR[s])
            os.chdir(filepath_save)
            np.save('noisy_phantom_SNR_' + str(SNR[s]) + '_std_dev_' + str(std_dev_list[s]) +'_realization_'+ 
                    str(n) + '.npy', noisy_phantom)