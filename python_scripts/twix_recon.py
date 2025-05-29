import numpy as np
#import mapvbvd
import os
import espirit
import matplotlib.pyplot as plt


def get_raw_data(dat_file_path, dimensionality):
    twixObj = mapvbvd.mapVBVD(dat_file_path)
    sqzDims = twixObj.image.sqzDims
    twixObj.image.squeeze = True
    data = twixObj.image['']

    linIndex = sqzDims.index('Lin')
    print(linIndex)
    data = np.moveaxis(data, linIndex, 1)
    print(data.shape)
    sqzDims.insert(0, sqzDims.pop(linIndex))
    print(data.shape)
    print(sqzDims)
    
    if dimensionality == '3D':
        print('hi')
        dataIm = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(data, axes=(0, 1, 3)), axes=(0, 1, 3)), axes=(0, 1, 3))
    else:
        dataIm = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    return dataIm, data

def get_masked_raw_data(dat_file_path, dimensionality, acceleration_factor = 2):
    twixObj = mapvbvd.mapVBVD(dat_file_path)
    sqzDims = twixObj.image.sqzDims
    twixObj.image.squeeze = True
    data = twixObj.image['']

    linIndex = sqzDims.index('Lin')
    print(linIndex)
    data = np.moveaxis(data, linIndex, 1)
    print(data.shape)
    sqzDims.insert(0, sqzDims.pop(linIndex))
    print(data.shape)
    print(sqzDims)

    #get a mask for each slice
    mask = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype = bool)
    
    temp = np.arange(0, mask.shape[0], acceleration_factor)

    for i in range(mask.shape[2]):  
        mask[temp,:,i] = 1 #set every acceleration_factor-th row to 1

    data_masked = np.empty(data.shape, dtype = 'complex')

    #apply mask to data
    for i in range(data.shape[3]):
        data_masked[:,:,:,i] = np.multiply(data[:,:,:,i], mask)

    #Truncate the data to remove the rows with 0s
    temp2 = np.arange(1, data.shape[0]+1, acceleration_factor)
    data_masked_truncated = np.delete(data_masked, temp2, axis = 0)

    if dimensionality == '3D':
        print('hi')
        dataIm_aliased = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(data_masked_truncated, axes=(0, 1, 3)), axes=(0, 1, 3)), axes=(0, 1, 3))
    else:
        dataIm_aliased = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(data_masked_truncated, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    return dataIm_aliased

def get_band_free_bssfp_image(bssfp_img):
    # get a band-free image for each coil by combining the different phase-cycled images


    band_free_img = np.zeros(bssfp_img[:,:,:,0].shape, dtype = 'complex')

    for c in range(bssfp_img.shape[2]):
        realP = np.sum(bssfp_img[:,:,c,:].real, axis = 2)
        imagP = np.sum(bssfp_img[:,:,c,:].imag, axis = 2)
        band_free_img[:,:,c] = realP + (1j*imagP)


    band_free_img = np.reshape(band_free_img, (band_free_img.shape[0], band_free_img.shape[1], 1, band_free_img.shape[2]))
    
    return band_free_img

def do_espirit_recon_bssfp(bssfp_img, esp_filepath = 0):
    
    #get band-free image
    band_free_img = get_band_free_bssfp_image(bssfp_img)
    
    #convert band-free image into k-space
    fourier_data_bandFree = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(band_free_img, axes = (0,1)), axes = (0,1)), axes = (0,1))
    
    #get espirit operator
    if esp_filepath == 0:
        esp = espirit.espirit(fourier_data_bandFree, 6, 24, 0.01, 0.9925)
    else:
        esp = np.load('esp_filepath')
    
    #do espirit reconstruction
    ip_pc = np.empty((bssfp_img.shape[0], bssfp_img.shape[1], bssfp_img.shape[3]), dtype = 'complex')

    for i in range(bssfp_img.shape[3]):
        dataImTemp = np.reshape(bssfp_img[:,:,:,i], (bssfp_img.shape[0], bssfp_img.shape[1], 1, bssfp_img.shape[2]))
        ip, _, _ = espirit.espirit_proj(dataImTemp, esp)
        ip   = np.squeeze(ip)
        ip_pc[:,:,i] = ip[:,:,0]

        plt.figure()
        plt.imshow(np.abs(ip_pc[:,:,i]), cmap = 'gray')
        plt.title(str(i))
        plt.colorbar()
    
    return esp, ip_pc

def do_espirit_recon_noise_scan(noise_scan, esp):
    
    #convert noise scan into k-space
    fourier_data_bandFree = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(noise_scan, axes = (0,1)), axes = (0,1)), axes = (0,1))
    
    #do espirit reconstruction
    ip_pc_noise = np.empty((noise_scan.shape[0], noise_scan.shape[1]), dtype = 'complex')

    dataImTemp = np.reshape(noise_scan, (noise_scan.shape[0], noise_scan.shape[1], 1, noise_scan.shape[2]))
    ip_noise, _, _ = espirit.espirit_proj(dataImTemp, esp)
    ip_noise   = np.squeeze(ip_noise)
    ip_pc_noise[:,:] = ip_noise[:,:, 0]
    
    return ip_pc_noise

def reorder_interleaved_gre_field_mapping_imgs(gre_imgs):
    
    slice_num = gre_imgs.shape[3]
    
    odd_scans = gre_imgs[:,:, :, 0:int(slice_num/2), :]
    even_scans = gre_imgs[:,:, :, int(slice_num/2):, :]
    even_idx = np.arange(0, slice_num, 2) #Note the reordering only works when the even indexes are this vector and not the one below
    odd_idx = np.arange(1, slice_num+1, 2)

    interleaved_scans = np.empty(gre_imgs.shape, dtype = 'complex')
    interleaved_scans[:,:,:,even_idx, :] = even_scans
    interleaved_scans[:,:,:,odd_idx, :] = odd_scans
    
    return interleaved_scans


def do_espirit_recon_gre_field_mapping(gre_imgs):
    echo_1 = gre_imgs[:,:,:,0]
    echo_2 = gre_imgs[:,:,:,1]
        
    echo_1_recon = do_basic_espirit_recon(echo_1)
    echo_2_recon = do_basic_espirit_recon(echo_2)
    
    return echo_1_recon, echo_2_recon
    
    
    
def do_basic_espirit_recon(img):
    
    fourier_data = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img, axes = (0,1)), axes = (0,1)), axes = (0,1))
    print(fourier_data.shape)
    
    fourier_data = np.reshape(fourier_data, (fourier_data.shape[0], fourier_data.shape[1], 1, fourier_data.shape[2]))
    
    esp = espirit.espirit(fourier_data, 6, 24, 0.01, 0.9925)
    
    ip_pc = np.empty((img.shape[0], img.shape[1]), dtype = 'complex')

    dataImTemp = np.reshape(img, (img.shape[0], img.shape[1], 1, img.shape[2]))
    ip, _, _ = espirit.espirit_proj(dataImTemp, esp)
    ip   = np.squeeze(ip)
    ip_pc[:,:] = ip[:,:, 0]
    
    return ip_pc


def calculate_b0_map(echo_1, echo_2):
    TE1 = 4.92*10**-3 # in seconds
    TE2 = 7.38*10**-3 # in seconds
    
    echo_1_conj = np.conj(echo_1)
    
    #equation based on https://web.stanford.edu/class/rad229/Notes/F1-QuantitativeSequences.pdf
    # https://school-brainhack.github.io/project/b0_field_mapping/ - tutorial on B0 mapping sequences
    numerator = np.angle(np.multiply(echo_2, echo_1_conj))
    denominator = 2*np.pi*(TE2-TE1)
    
    delta_f = numerator/denominator
    
    return delta_f


# Function that takes in exponential data points and fits an exponential curve to it using non-linear fitting
def fit_exponential(x, y):
    from scipy.optimize import curve_fit

    # Define the function to fit to the data
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the data using non-linear least squares
    popt, pcov = curve_fit(func, x, y)

    # Return the fitted parameters
    return popt

#function that takes in an array and outputs a same sized boolean array, where True is where the original array has a number greater than 0 and False is when the original array has a value of 0
def get_boolean_array(arr):
    bool_arr = np.zeros(arr.shape, dtype = bool)
    bool_arr[arr > 0] = True
    return bool_arr


# Get B0 map from the 2D twix raw data
def get_B0_map_from_raw_data(twix_data):

    B0_raw_data = get_raw_data(twix_data, '2D')

    #Do ESPIRIT reconstruction
    echo_1_recon, echo_2_recon = do_espirit_recon_gre_field_mapping(B0_raw_data)
    
    #Get B0 map
    B0_map = calculate_b0_map(echo_1_recon, echo_2_recon)

    return B0_map



