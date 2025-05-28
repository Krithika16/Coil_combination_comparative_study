import scipy
import numpy as np
from scipy import signal

"""
Credits: Zimu Huo wrote ifft2c(), fft2c(), smooth(), adaptive_combine() based on: 
Walsh DO, Gmitro AF, Marcellin MW. Adaptive reconstruction of phased array MR imagery. 
Magn Reson Med. 2000 May;43(5):682-90. 
doi: 10.1002/(sici)1522-2594(200005)43:5<682::aid-mrm10>3.0.co;2-g. PMID: 10800033.

Krithika Balaji wrote the bssfp_adaptive_combine() function to combine the phase-cycled bSSFP data using the adaptive combine functions.
"""

def ifft2c(F, axis = (0,1)):
    x,y = (axis)
    tmp0 = np.fft.ifftshift(np.fft.ifftshift(F, axes=(x,)), axes=(y,))
    tmp1 = np.fft.ifft(np.fft.ifft(tmp0, axis = x), axis = y)
    f = np.fft.fftshift(np.fft.fftshift(tmp1, axes=(x,)), axes=(y,))
    return f * F.shape[x]* F.shape[y]
 
def fft2c(f, axis = (0,1)):
    x,y = (axis)
    tmp0 = np.fft.fftshift(np.fft.fftshift(f, axes=(x,)), axes=(y,))
    tmp1 = np.fft.fft(np.fft.fft(tmp0, axis = x), axis = y)
    F = np.fft.ifftshift(np.fft.ifftshift(tmp1, axes=(x,)), axes=(y,))
    return F / f.shape[x]/ f.shape[y]

def smooth(img, box=5):
    t_real = np.zeros(img.shape)
    t_imag = np.zeros(img.shape)
    scipy.ndimage.filters.uniform_filter(img.real,size=box,output=t_real)
    scipy.ndimage.filters.uniform_filter(img.imag,size=box,output=t_imag)
    simg = t_real + 1j*t_imag
    return simg

def adaptive_combine(data, ks = 9, smoothing = 5):
    [ny, nx, nc] = data.shape
    image = ifft2c(data)
    ks = ks
    R = np.zeros([ny, nx, nc, nc], dtype = complex)
    for i in range(nc): 
        for j in range(nc):
            R[:,:,i,j] += smooth(signal.convolve2d(image[...,i]*image[...,j].conj(), np.ones([ks, ks]), mode='same'), smoothing)
    recon = np.zeros([ny,nx], dtype = complex)
    for y in range(ny):
        for x in range(nx):
            U, S, VT = np.linalg.svd(R[y,x,:,:],full_matrices=False)
            recon[y,x] = U[:,0].conj().T@image[y,x,:]
    return recon

def bssfp_adaptive_combine(data):
    """
    Combines the phase-cycled bSSFP coil data using the adaptive combine function
    
    Argument:
    - data: (row, col, coil, pc). This is in the image domain
    
    Return:
    - AC_data: (row, col, pc). This is in the image domain
    """
    
    AC_data = np.empty((data.shape[0], data.shape[1], data.shape[3]), dtype = 'complex')
    
    #Convert into the Fourier domain
    fourier_data = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(data, axes = (0,1)), axes = (0,1)), axes = (0,1))
    
    for i in range(AC_data.shape[2]):
        AC_data[:,:,i] = adaptive_combine(fourier_data[:,:,:,i])
    
    return AC_data