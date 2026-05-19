import numpy as np
import scipy as sp

def density_to_pressure(rho):
    return rho - np.ones_like(rho)

def pressure_to_wav_file(p, filename):
    p = p - np.min(p)
    p = p / np.max(p)
    p = (p * 65535).astype(np.uint16)
    sp.io.wavfile.write(filename, 44100, p)
