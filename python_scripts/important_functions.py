
#import all the libraries I need to run the functions in this page
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import EllipseModel
from sklearn.preprocessing import minmax_scale



def fitEllipse(complex_data):
    
    """
    Function that fits an ellipse to complex phase-cycled bSSFP signal data.
    
    Parameters
    ----------
    complex_data : array_like
        Complex phase-cycled bSSFP signal data of the shape (npcs,), where npcs = number of 
        phase cycles. These complex signals should have already been multiplied by a constant (or normalized) 
        to ensure the data is not too small, as otherwise the ellipse fitting function might not find a solution. 
    
    Returns
    -------
    xc : float
        x-value of centroid of ellipse
    yc : float
        y-value of centroid of ellipse
    ma_axis : float
        major axis of the ellipse
    mi_axis : float
        minor axis of the ellipse
    theta : float
        tilt of the ellipse in radians
        
    Notes
    -----
    Requires the user to import the following:
    
        import numpy as np
        from skimage.measure import EllipseModel
    
    """
    
    realP = complex_data.real
    imagP = complex_data.imag
    
    ellPoints = np.vstack([realP, imagP])
    ellPoints = np.moveaxis(ellPoints, 1,0)
    ell = EllipseModel()
    ell.estimate(ellPoints)
    xc, yc, a, b, theta = ell.params
    
    if a < b:
        ma_axis = b
        mi_axis = a
        
    else:
        ma_axis = a
        mi_axis = b
    
    #print("center = ",  (xc, yc))
    #print("angle of rotation in degrees = ",  theta*180/np.pi)
    #print("major and minor axes = ", (ma_axis, mi_axis))
    
    return xc, yc, a, b, theta

def rotate_ellipses_in_dataset(complex_pc_bssfp_data):
    """
    Takes in elliptical phase-cycled bSSFP signals, where the ellipses are rotated in complex plane. It finds the center
    point and rotates the ellipses around the origin so that the center point lies on the positive horizontal axis. 
    
    Argument:
    ---------
    complex_pc_bssfp_data: (npcs, n_samples). Note that the real and imaginary axes should be fixed. These are ellipses in 
                           complex plane that have been rotated around the origin
    
    Returns: 
    --------
    rotated_complex_pc_bssfp_data: (npcs, n_samples). The ellipses have been rotated back to their original position
    """
    
    rotated_complex_pc_bssfp_data = np.empty(complex_pc_bssfp_data.shape, dtype = 'complex')
    
    for i in range(complex_pc_bssfp_data.shape[1]):
        xc, yc, _, _, _ = fitEllipse(complex_pc_bssfp_data[:,i])

        center_point = xc + 1j*yc
        angle = np.angle(center_point)

        rotated_complex_pc_bssfp_data[:,i] = complex_pc_bssfp_data[:,i] * np.exp(-1j * angle)
    return rotated_complex_pc_bssfp_data

def returnComplexImage(arr_mg, arr_ph):
    
    """
    Function that takes in magnitude and phase images and returns a complex image.
    
    Parameters
    ----------
    arr_mg : array_like
        Magnitude image of the shape (l, w), where l/w is the length/width of the image (l can be equal to w).
        
    arr_ph : array_like
        Phase image of the shape (l,w), where l/w is the length/width of the image(l can be equal to w). 
        The DICOM phase images must already be normalized to be between -pi and pi before inputting into this 
        function. 
    
    Returns
    -------
    image : array_like
        Complex image of shape (l,w) calculated using the magnitude and phase data.
        
    Notes
    -----
    Requires the user to import the following line to use the EllipseModel function:
    
        import numpy as np
    
    """
    image = np.empty(np.shape(arr_mg), dtype = complex)

    for i in range(len(arr_ph)):
        for l in range(len(arr_ph)):
            image[i, l] = arr_mg[i,l] * (math.cos(arr_ph[i,l]) + math.sin(arr_ph[i,l])*1j)
            
    return image

def displayImage(image, pc):
    
    """
    Function that takes in a complex phase-cycled bSSFP image and the number of phase cycles and uses these to 
    plot the magnitude and phase images side by side for each phase cycle. 
    
    Parameters
    ----------
    image : array_like
        Complex phase-cycled bSSFP image (l,w).
        
    pc : int
        Number of phase cycles
    
    Returns
    -------
    None
        
    Notes
    -----
    Requires the user to import the following:
    
        import numpy as np
        import matplotlib.plot as plt
    
    """
    
    fig, axs = plt.subplots(1,2, figsize=(8,8))
    axs[0].imshow(np.angle(image), cmap="gray")
    axs[0].set_title('Calculated Phase Image for phase cycle ' + str(pc))
    
    axs[1].imshow(np.abs(image), cmap="gray")
    axs[1].set_title('Calculated Magnitude Image for phase cycle ' + str(pc))

    fig.tight_layout()
    return
   
def phaseUnwrapping(phaseData):
    """
    This function performs phase unwrapping, something that is needed due to how phase is calculated by np.angle.
    If there is a sharp spike in the phase where it sharply increases or decreases to +/-pi, then the phase for 
    that phase cycle is adjusted for by 2pi. 
    
    Parameters:
        phaseData: this vector contains the phase data across the phase cycles for one voxel. Expected dimensions
                   are (npcs,), where npcs is the number of phase cycles.
        
    Returns:
        unwrappedPhaseData: this vector contains the unwrapped phase data across the phase cycles for one voxel.
                            Same dimensions as phaseData. 
    
    """
    unwrappedPhaseData = np.copy(phaseData)
    for i in range(unwrappedPhaseData.shape[0]-1):
        diff = unwrappedPhaseData[i+1]-unwrappedPhaseData[i]
        
        if diff > 3.5:
            unwrappedPhaseData[i+1] = unwrappedPhaseData[i+1] - 2*np.pi
        elif diff < -3.5:
            unwrappedPhaseData[i+1] = unwrappedPhaseData[i+1] + 2*np.pi
            
    return unwrappedPhaseData

def split_complex_data(data):
    #The data that is input into this function is 2-dimensional --> the first dimension is the number of sets of data
    #while the second dimension is the number of complex signals for each set of data (due to the various images taken from 
    #phase-cycled bSSFP). Note: this assumes that we are only dealing with a voxel, and NOT with a full image
    split_data = np.zeros([data.shape[0],2*data.shape[1]]) #Multiplying the second dimension by two as that will split complex data into real and imaginary parts

    #Here, we are looping through the data for each phase cycle and splitting the components into 2
    for i in range(data.shape[1]):
        split_data[:,2*i] = data[:,i].real
        split_data[:,2*i+1] = data[:,i].imag
    return split_data 

def combine_into_complex_data(data):
    """
    This function takes in a vector of real and imaginary values and combines them to form a new vector of complex numbers. 
    Essentially performs the opposite function of split_complex_data().
    
    Parameters:
        data: This is a vector/array containing the real and imaginary values in adjacent indices. Expected dimensions are (r,n),
        where r is the number of samples of data and n is the length of the vector containing both the real and imaginary
        values. Note that first number is expected to be real value 
        
    
    Returns:
        complex_data: This is a new vector containing the newly combined complex numbers. Expected dimensions are (r, n/2).
    """
    
    complex_data = np.empty((data.shape[0], int(data.shape[1]/2)), dtype = 'complex')
    
    indices = np.arange(0, data.shape[1], 2)
    
    t = 0
    for i in range(complex_data.shape[1]):
        complex_data[:,i] = data[:,2*i] + (data[:,2*i+1]*1j)
    
    return complex_data

def minmaxScaling(data):
    
    """
    Finds the minmax normalization of the elliptical phase-cycled bSSFP signals. 
    
    Parameters:
        data: This is the phase-cycled bSSFP data where the real and the imaginary parts have been split up. The expected 
        dimensions are (r, 2n), where r is the number of data samples and n is the number of phase cycles. 
        
    Returns:
        scaled_data: This is the minmax normalized data. Expected dimensions are (r, 2n).
    """
    
    scaled_data = minmax_scale(np.reshape(data, (data.shape[0]*data.shape[1],)), feature_range = (-1,1), axis = 0)
    scaled_data = np.reshape(scaled_data, np.shape(data))
    
    return scaled_data

def minmaxnorm_complex_data(data):
    """
    This function takes the minmax normalization of complex data.
    
    Parameters:
        data: This is the vector/array of phase-cycled bSSFP complex data that needs to be minmax normalized. Expected
        dimensions are (n,), where n is the number of phase cycles. 
        
    Returns:
        scaled_complex_data: This is the vector/array of minmax normalized phase-cycled bSSFP complex data. Same dimensions as data.
    """

    cdata = np.reshape(data, (1, data.shape[0]))
    
    sig_RI = split_complex_data(cdata)

    scaled_data = minmaxScaling(sig_RI)

    scaled_complex_data = combine_into_complex_data(scaled_data)
    
    return scaled_complex_data

def add_noise_to_invivo_signals(in_vivo_signals, noise_signals, std_dev):
    """
    Takes in in-vivo signals and noise scan signals and adds bivariate gaussian noise to reduce SNR
    
    Arguments:
    ----------
    - in_vivo_signals: (n_samples, npcs). Complex signals
    - noise_signals: (n_samples,). Complex signals
    - std_dev: float. 
    
    Returns:
    --------
    - low_SNR_in_vivo_signals: (n_samples, npcs). Complex signals
    - low_SNR_noise_in_vivo_signals: (n_samples,). Complex signals
    
    """
    
    low_SNR_in_vivo_signals = bssfp.add_noise_gaussian(in_vivo_signals, mu=0, sigma=std_dev)
    low_SNR_noise_in_vivo_signals = bssfp.add_noise_gaussian(noise_signals, mu=0, sigma=std_dev)
    
    return low_SNR_in_vivo_signals, low_SNR_noise_in_vivo_signals

def calculate_SNR_invivo_signals(in_vivo_signals, noise_signals):
    
    """
    Takes in in-vivo signals and noise scan signals and calculates the SNR
    
    Arguments:
    ----------
    - in_vivo_signals: (n_samples, npcs). Complex signals
    - noise_signals: (n_samples,). Complex signals
    
    Returns:
    --------
    - low_SNR_in_vivo_signals: (n_samples, npcs). Complex signals
    - low_SNR_noise_in_vivo_signals: (n_samples,). Complex signals
    """
    
    std_dev_rayleigh = np.std(np.abs(noise_signals))

    #Step 2: convert rayleigh std dev to gaussian std dev
    std_dev_gaussian = convert_rayleighStdDev_to_gaussianStdDev(std_dev_rayleigh)

    #Step 3: ROI-average the full cartilage region
    averaged_signal = np.average(in_vivo_signals, axis = 0)

    magnitude_per_phase_cycle = np.abs(averaged_signal) #Finding the average cartilage magnitude per phase cycle

    SNR_per_phase_cycle = magnitude_per_phase_cycle/std_dev_gaussian

    average_SNR = np.average(SNR_per_phase_cycle)    
    
    return average_SNR

def convert_rayleighStdDev_to_gaussianStdDev(rayleigh_std_dev):
    
    gaussian_std_dev = rayleigh_std_dev / np.sqrt(2-(np.pi/2))
    
    return gaussian_std_dev


def crop_zeropad_kspace(kspace, crop_row_len, crop_col_len):
    middle_kspace_row = kspace.shape[0]/2
    middle_kspace_col = kspace.shape[1]/2
    
    idx_row = np.arange(middle_kspace_row - int(crop_row_len/2), middle_kspace_row + int(crop_row_len/2), 1)
    idx_col = np.arange(middle_kspace_col - crop_col_len/2, middle_kspace_col + crop_col_len/2, 1)
    
    idx_row = [int(idx_row[i]) for i in range(len(idx_row))]
    idx_col = [int(idx_col[i]) for i in range(len(idx_col))]
    
    #create a mask where the cropped region is 1 and the rest is 0
    

    mask = np.zeros(kspace.shape)

    for i in range(len(idx_row)):
        mask[idx_row[i], idx_col] = 1
 
    
    kspace_cropped_padded = np.multiply(kspace, mask)
    
    low_res_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_cropped_padded, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    
    return low_res_img


def segment_image_and_getROI_with_startend_idx(bssfp_im, startRow, endRow, startCol, endCol):
    """
    This function segments a phase-cycled bSSFP image using the start and ending row/column indices.
    This assumes that the corresponding start and end row and column values describe
    a rectangle to be segmented
    """
  
    row_indices = []
    col_indices = []

    #Get all the row and column indices within the ROI
    for l in range(len(startRow)):
        temp_row = np.arange(startRow[l], endRow[l], 1)
        temp_col = np.arange(startCol[l], endCol[l], 1)
        
        for r in temp_row:
            for c in temp_col:
                row_indices.append(r)
                col_indices.append(c)

    
    signals = bssfp_im[row_indices, col_indices, :]
    
    return signals, row_indices, col_indices


def get_SNR_from_pc_bssfp_img_method1(pc_bssfp, row_indices_bg, col_indices_bg, row_indices_c, col_indices_c,
                                     coil_num, show_text = True):

    #Step 1: get the rayleigh noise standard deviation from the noise scan magnitude image 
    std_dev_rayleigh = np.std(np.abs(pc_bssfp[row_indices_bg, col_indices_bg, coil_num, :]))

    #Step 2: convert rayleigh std dev to gaussian std dev
    std_dev_gaussian = convert_rayleighStdDev_to_gaussianStdDev(std_dev_rayleigh)
    
    #Step 3: ROI-average the full cartilage region
    cartilage_signals = pc_bssfp[row_indices_c, col_indices_c, coil_num, :]
    averaged_signal = np.average(cartilage_signals, axis = 0)
    
    magnitude_per_phase_cycle = np.abs(averaged_signal) #Finding the average cartilage magnitude per phase cycle
    SNR_per_phase_cycle = magnitude_per_phase_cycle/std_dev_gaussian
    
    average_SNR = np.average(SNR_per_phase_cycle)
    if show_text == True:
        print("The average SNR across all the phase cycles for a cartilage ROI is: " + str(average_SNR))
    
    return average_SNR

def RMSE_simulated_phantom(noisy_phantom, coil_combined_data):
    """
    This function calculates the RMSE between the phase of the 0th coil data and the phase of the coil-combined data 
    within the region [90:140, 200:240].

    Arguments:
    noisy_phantom: (nr, nc, coil, pc)  - complex array
    coil_combined_data: (nr, nc, pc) - complex array

    Returns:
    RMSE: float
    """
    rows = np.arange(90, 140, 1)
    cols = np.arange(200,240,1)

    #Make empty arrays to store the phase data
    phase_0th_coil = []
    phase_coil_combined = []

    #Get the phase of the 0th coil and coil combined data and do phase unwrapping on the phase data
    angle_0th_coil = np.angle(noisy_phantom[90:140, 200:240,0,:])
    angle_coil_combined = np.angle(coil_combined_data[90:140, 200:240,:])

    for r in range(len(rows)):
        for c in range(len(cols)):
            phase_0th_coil.append(phaseUnwrapping(angle_0th_coil[r, c, :])) #single coil phase unwrapping
            phase_coil_combined.append(phaseUnwrapping(angle_coil_combined[r, c, :])) #coil combined phase unwrapping

    phase_0th_coil = np.array(phase_0th_coil)
    phase_coil_combined = np.array(phase_coil_combined)
     
    #Calculate the RMSE
    phase_0th_coil_zeroline = np.transpose(np.subtract(np.transpose(phase_0th_coil), phase_0th_coil[:,0]))
    phase_coil_combined_zeroline = np.transpose(np.subtract(np.transpose(phase_coil_combined), phase_coil_combined[:,0]))
    RMSE = np.sqrt(np.mean((phase_0th_coil_zeroline - phase_coil_combined_zeroline)**2))
        

    return RMSE, phase_0th_coil, phase_coil_combined

def bias_variance_invivo(pre_coil_combined_data, coil_combined_data, row_indices_c, col_indices_c):
    """
    This function calculates the bias and variance of the difference between the phase of the 0th 
    coil data and the phase of the coil-combined data within the region [90:140, 200:240].

    Arguments:
    pre_coil_combined_data: (nr, nc, coil, pc)  - complex array
    coil_combined_data: (nr, nc, pc) - complex array
    row_indices_c: list
    col_indices_c: list

    Returns:
    bias    : float
    variance: float
    """

    #Make empty arrays to store the phase data
    phase_0th_coil = []
    phase_coil_combined = []

    #Get the phase of the 0th coil and coil combined data and do phase unwrapping on the phase data
    angle_0th_coil = np.angle(pre_coil_combined_data[row_indices_c, col_indices_c, 7,:])
    angle_coil_combined = np.angle(coil_combined_data[row_indices_c, col_indices_c,:])

    for r in range(len(row_indices_c)):
        phase_0th_coil.append(phaseUnwrapping(angle_0th_coil[r, :])) #single coil phase unwrapping
        phase_coil_combined.append(phaseUnwrapping(angle_coil_combined[r, :])) #coil combined phase unwrapping

    phase_0th_coil = np.array(phase_0th_coil)
    phase_coil_combined = np.array(phase_coil_combined)
     
    #Calculate the RMSE
    phase_0th_coil_zeroline = np.transpose(np.subtract(np.transpose(phase_0th_coil), phase_0th_coil[:,0]))
    phase_coil_combined_zeroline = np.transpose(np.subtract(np.transpose(phase_coil_combined), phase_coil_combined[:,0]))
    bias = np.mean(np.abs(phase_0th_coil_zeroline - phase_coil_combined_zeroline))
    variance = np.var(np.abs(phase_0th_coil_zeroline - phase_coil_combined_zeroline))
    
    return bias, variance

def bias_variance_simulated_phantom(noisy_phantom, coil_combined_data):
    """
    This function calculates the bias and variance of the difference between the phase of the 0th 
    coil data and the phase of the coil-combined data within the region [90:140, 200:240].

    Arguments:
    noisy_phantom: (nr, nc, coil, pc)  - complex array
    coil_combined_data: (nr, nc, pc) - complex array

    Returns:
    bias    : float
    variance: float
    """
    rows = np.arange(90, 140, 1)
    cols = np.arange(200,240,1)

    #Make empty arrays to store the phase data
    phase_0th_coil = []
    phase_coil_combined = []

    #Get the phase of the 0th coil and coil combined data and do phase unwrapping on the phase data
    angle_0th_coil = np.angle(noisy_phantom[90:140, 200:240,0,:])
    angle_coil_combined = np.angle(coil_combined_data[90:140, 200:240,:])

    for r in range(len(rows)):
        for c in range(len(cols)):
            phase_0th_coil.append(phaseUnwrapping(angle_0th_coil[r, c, :])) #single coil phase unwrapping
            phase_coil_combined.append(phaseUnwrapping(angle_coil_combined[r, c, :])) #coil combined phase unwrapping

    phase_0th_coil = np.array(phase_0th_coil)
    phase_coil_combined = np.array(phase_coil_combined)
     
    #Calculate the bias and variance
    phase_0th_coil_zeroline = np.transpose(np.subtract(np.transpose(phase_0th_coil), phase_0th_coil[:,0]))
    phase_coil_combined_zeroline = np.transpose(np.subtract(np.transpose(phase_coil_combined), phase_coil_combined[:,0]))
    bias = np.mean(np.abs(phase_0th_coil_zeroline - phase_coil_combined_zeroline))
    variance = np.var(np.abs(phase_0th_coil_zeroline - phase_coil_combined_zeroline))
        

    return bias, variance, phase_0th_coil, phase_coil_combined