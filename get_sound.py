import numpy as np
import scipy as sp

def density_to_pressure(rho):
    return rho - np.ones_like(rho)

def pressure_to_wav_file(p, filename):
    p = p - np.min(p)
    p = p / np.max(p)
    p = np.floor(p * 32767).astype(np.int16)
    sp.io.wavfile.write(filename, 44100, p)

def density_to_wav_file(rho, filename):
    p = density_to_pressure(rho)
    pressure_to_wav_file(p, filename)