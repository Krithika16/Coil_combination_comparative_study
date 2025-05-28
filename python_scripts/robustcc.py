'''Robust coil combination for bSSFP.'''

import numpy as np
from tqdm import tqdm

"""
Credits: Nicholas McKibben wrote the below code and uploaded it to https://github.com/mckib2/ssfp/tree/main/ssfp.
"""

def robustcc(
        data, method='simple', mask=None, coil_axis=-1, pc_axis=-2):
    '''Robust elliptcal-model-preserving coil combination for bSSFP.
    Parameters
    ----------
    data : array_like
        Complex bSSFP image space data to be coil combined.
    method : str, {'simple', 'full'}, optional
        The method of phase estimation.  method='simple' is very fast
        and does a good job.
    mask : array_like or None, optional
        Only used for method='full'.  Gives mask of which pixels to
        do fit on.  Does not have coil, phase-cycle dimensions.
    coil_axis : int, optional
        Dimension holding the coil data.
    pc_axis : int, optional
        Dimension holding the phase-cycle data.
    Returns
    -------
    res : array_like
        Complex coil-combined data that preserves elliptical
        relationships between phase-cycle pixels.
    Notes
    -----
    Implements the method described in [1]_.  This coil combination
    method preserves elliptical relationships between phase-cycle
    pixels for more efficient computation, e.g., gs_recon which
    reconstructs one coil at a time ([2]_).
    References
    ----------
    .. [1] N. McKibben, G. Tarbox, E. DiBella, and N. K. Bangerter,
           "Robust Coil Combination for bSSFP Elliptical Signal
           Model," Proceedings of the 28th Annual Meeting of the
           ISMRM; Sydney, NSW, Australia, 2020.
    .. [2] Xiang, Qing‐San, and Michael N. Hoff. "Banding artifact
           removal for bSSFP imaging with an elliptical signal model."
           Magnetic resonance in medicine 71.3 (2014): 927-933.
    '''
    # Put coil and phase-cycle axes where we expect them - kr: this moves the phase
    # cycles to the second last position and the coil axis to the last position
    data = np.moveaxis(data, (coil_axis, pc_axis), (-1, -2))

    # Use SOS (root sum of squares) for magnitude estimate
    mag = np.sqrt(np.sum(np.abs(data)**2, axis=-1))

    # kr: Choose the method for phase estimation and use the appropriate function
    if method == 'simple':
        # simple coil phase estimate is fast:
        print('simple')
        phase = _simple_phase(data)
    elif method == 'full':
        # Use the full solution:
        phase = _full_phase(data, mask)
    else:
        raise NotImplementedError()

    # kr: They use the root sum of squares technique on all the coil images to get
    # the magnitude estimate and then they calculate the
    # phase of the coil combined complex-valued signals using either the simple
    # or the full phase functions. They then use the magnitude and phase to form
    # the new COMPLEX coil combined phase-cycled images. They also move the coil
    # and phase cycle axes to their original locations (ie the locations they
    # were originally input in)
    return np.moveaxis(
        mag*np.exp(1j*phase), (-1, -2), (coil_axis, pc_axis))

def _simple_phase(data):
    '''Simple strategy: choose phase from best coil ellipse.'''

    # Assume the best coil is the one with max median value

    # kr: np.argmax returns the indices of the maximum values along an axis.
    # In this line, first, the median of the MAGNITUDE of the complex data is
    # being found (basically, looking at the median intensity of the image) along the
    # coil axis for each voxel location. Then, the indices of the maximum median value of the magnitude
    # is being found along the coil dimensions. Looking at the median to ensure
    # we don't get bogged down by outliers.
    idx = np.argmax(
        np.median(np.abs(data), axis=-2, keepdims=True), axis=-1)

    print(idx)

    # kr: np.take_along_axis takes values from the input array by matching a 1D index
    # and the actual data slices along an axis. Here, we are taking the indices
    # of the maximum median magnitude of the image along the coil axis and
    # extracting the ANGLES of the corresponding complex values in the original
    # image in the coil direction.
    return np.take_along_axis(
        np.angle(data), idx[..., None], axis=-1).squeeze()

def _full_phase(data, mask=None):
    '''Do pixel-by-pixel ellipse registration.'''

    # Worst case is to do all pixels:
    if mask is None:
        # kr: Here, making a mask of ones of the same size of the input image, with
        # the exception of the coil and pc axes (because the mask should be the same
        # along the coil channel). This mask of 1s actually is a boolean array of
        # all Trues.
        mask = np.ones(data.shape[:-2], dtype=bool)

    # Do pixel-by-pixel phase estimation

    # kr: created an empty array of shape that is the same as original image with
    # exception of the coil axis as the phase array is the coil combined phase array
    phase = np.empty(data.shape[:-1])

    # kr: tqdm shows a smart progress meter - to use, wrap any iterable with tqdm(iterable)
    # and that should show a progress meter when running the code. Iterating through
    # the sum of the flattened mask number of times. idx is acquired from
    # np.argwhere, where the function finds the indices of the non-zero mask elements
    for idx in tqdm(
            # kr: np.argwhere finds the indices of the mask that are non-zero
            np.argwhere(mask),
            total=np.prod(np.sum(mask.flatten())),
            leave=False):

        # Register all coil ellipses to a single reference --> kr: size is (# pcs, # coils)
        # This is done for each pixel in the image - eg (12,8) array created for each pixel
        # extracts the pixel-wise phase cycled signals for each coil and stores it
        coil_ellipses = data[tuple(idx) + (slice(None), slice(None))]

        # Take reference ellipse to be the one with greatest
        # median value - similar to simple phase RCC
        ref_idx = np.argmax(np.median(np.abs(
            coil_ellipses), axis=-1, keepdims=True), axis=-1)
        ref = np.take_along_axis(
            coil_ellipses, ref_idx[..., None], axis=-1).squeeze()

        # Do coil by coil registration
        # kr: create empty array of the same size as the pixel-wise coil ellipses
        # array. This new array will store the registered ellipses, which would
        # still be there.
        reg_ellipses = np.empty(
            coil_ellipses.shape, dtype=coil_ellipses.dtype)
        W = np.empty(coil_ellipses.shape[-1]) # kr: weights given for each coil
        # kr: need to give weights as not all coils give ellipses (coil not
        # sensitive to specific parts)

        # kr: looping across each coil.
        for cc in range(coil_ellipses.shape[-1]):

            # kr: np.linalg.lstsq()[0] returns the least squares solution to
            # the
            T = np.linalg.lstsq( #this is th ellipse fitting
                coil_ellipses[..., cc][:, None],
                ref, rcond=None)[0]
            W[cc] = np.abs(T)**2 # save the weights
            T = np.exp(1j*np.angle(T)) # just rotate, no scaling
            reg_ellipses[..., cc] = T*coil_ellipses[..., cc]

        # Take the weighted average to the composite ellipse
        phase[tuple(idx) + (slice(None),)] = np.average(
            np.angle(reg_ellipses), axis=-1, weights=W)

    # make sure undefined values are set to 0; we have to do this
    # since we allocated with np.empty
    phase[~mask] = 0 # pylint: disable=E1130
    return phase

if __name__ == '__main__':
    pass
