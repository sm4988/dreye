import numpy as np
from axolotl import utils
import dreye


def preprocess_baden(raw_image):
    """
    # Input: Baden raw numpy file
    # Output: photon flux at each pixel (opsins x pixels)
    """

    raw_image = np.power(10, raw_image)  # undo log10 compression
    raw_image = utils.interp_to_finites(raw_image, axis=1)  # interpolate nans
    wls = dreye.Domain(start=200, end=999, interval=1.0, units='nm')  # wavelength domain

    # irradiance to photon flux
    spectra = dreye.Signals(values=raw_image, domain=wls, units='spectralirradiance', domain_units='nm')
    spectra_ue = spectra.to('uE')
    data = spectra_ue.magnitude  # (opsin, pixel)

    return data, wls


def spiral_ccw(A):
    A = np.array(A)
    out = []
    while A.size:
        out.append(A[0][::-1])  # first row reversed
        A = A[1:][::-1].T  # cut off first row and rotate clockwise
    return np.concatenate(out)


def base_spiral(nrow, ncol):
    return spiral_ccw(np.arange(nrow * ncol).reshape(nrow, ncol))[::-1]


def to_spiral(A):
    A = np.array(A)
    B = np.empty_like(A)
    B.flat[base_spiral(*A.shape)] = A.flat
    return B
