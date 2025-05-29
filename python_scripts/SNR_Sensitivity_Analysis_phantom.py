# Description: This script is used to analyze the SNR sensitivity of the various coil combination techniques
import numpy as np
import os
import important_functions, robustcc, adaptive_combine, twix_recon

if __name__ == '__main__':

    # Write down the filepath prefix to the data and to save the RMSE data
    filepath = '/rds/general/user/kb4317/home/coil_combination_project/simulated_phantom_data'
    os.chdir(filepath)
    gt_image = np.load("simulated_phantom_12pcbssfp_coil_images.npy")

    # List of different coil combination techniques
    coil_combination_techniques = ['espirit','srcc','frcc','adaptive_combine']

    # List of different SNR values
    SNR = np.arange(10,101,10)
    std_dev_list = [0.006, 0.003, 0.002, 0.0015, 0.0012, 0.001, 0.00087, 0.00075, 0.00067, 0.0006]
    num_realizations = 100

    #Set up for array job - send each job (a certain coil combination technique used to combine
    #data with a certain SNR level) to a different node - should be 40 nodes in total
    counter = 0
    array_index = int(os.environ['PBS_ARRAY_INDEX'])

    for s in range(len(SNR)):
        for c in range(len(coil_combination_techniques)):
            counter = counter + 1
            if counter == array_index:

                bias = []
                variance = []

                #Also save the phase data I get so that I can recalculate the RMSE later if I want to
                #phase_0th_coil_array = np.empty((2000,12,100), dtype = 'float64')
                #phase_coil_combined_array = np.empty((2000,12,100), dtype = 'float64')

                for r in range(num_realizations):
                    #Go to the relevant file directory and load in the file
                    filepath_data = filepath + '/SNR_' + str(SNR[s])
                    os.chdir(filepath_data)

                    noisy_phantom = np.load('noisy_phantom_SNR_' + str(SNR[s]) + '_std_dev_' + str(std_dev_list[s]) +'_realization_'+ str(r) + '.npy')

                    # Perform the coil combination

                    #should be debugged
                    if coil_combination_techniques[c] == 'espirit':
                        esp, coil_combined_image = twix_recon.do_espirit_recon_bssfp(noisy_phantom)

                    #working
                    elif coil_combination_techniques[c] == 'srcc':
                        coil_combined_image = robustcc.robustcc(noisy_phantom, coil_axis = -2, pc_axis = -1)
                        # Need to shift the phase cycle axis to the end
                        coil_combined_image = np.moveaxis(coil_combined_image, [0,1,2], [0,2,1])
                        print(coil_combined_image.shape)
                    
                    #working
                    elif coil_combination_techniques[c] == 'frcc':
                        coil_combined_image = robustcc.robustcc(noisy_phantom, method = 'full', coil_axis = -2, pc_axis = -1)
                        # Need to shift the phase cycle axis to the end
                        coil_combined_image = np.moveaxis(coil_combined_image, [0,1,2], [0,2,1])
                        print(coil_combined_image.shape)
                    
                    #working
                    elif coil_combination_techniques[c] == 'adaptive_combine':
                        coil_combined_image = adaptive_combine.bssfp_adaptive_combine(noisy_phantom)

                    # Calculate the RMSE - working
                    bias_temp, variance_temp, phase_0th_coil, phase_coil_combined = important_functions.bias_variance_simulated_phantom(gt_image, coil_combined_image)
                    bias.append(bias_temp)
                    variance.append(variance_temp) 

                    #should be debugged
                    #phase_0th_coil_array[:,:,r] = phase_0th_coil
                    #phase_coil_combined_array[:,:,r] = phase_coil_combined

                # Save the RMSE and phase data
                os.chdir(filepath)
                np.save('bias1_' + coil_combination_techniques[c] + '_SNR_' + str(SNR[s]) + '.npy', np.array(bias)) #should be 100 elements long
                np.save('variance1_' + coil_combination_techniques[c] + '_SNR_' + str(SNR[s]) + '.npy', np.array(variance)) #should be 100 elements long
                # Note that this phase data is in radians and phase unwrapping has already been conducted. The only thing that hasn't
                # been done is that the phase of the 0th coil has not been subtracted from the phase of the combined coil - this 
                # subtraction is done to calculate the RMSE and should be done to replicate the RMSE results.
                #np.save('phase_0th_coil_' + coil_combination_techniques[c] + '_SNR_' + str(SNR[s]) + '.npy', phase_0th_coil_array)
                #np.save('phase_coil_combined_' + coil_combination_techniques[c] + '_SNR_' + str(SNR[s]) + '.npy', phase_coil_combined_array)                