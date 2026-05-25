import numpy as np
import scipy as sp
from pathlib import Path

def density_to_pressure(rho):
    return rho - np.ones_like(rho)

def pressure_to_wav_file(p, path):
    p = p - np.mean(p)
    p = p / np.max(np.abs(p))
    p = np.floor(p * 32767).astype(np.int16)
    path = Path(path)
    print(f"Writing wav file to: {path.resolve()}")
    sp.io.wavfile.write(path, 44100, p)
    print(f"File written successfully: {path.resolve().exists()}")

def density_to_wav_file(rho, path):
    p = density_to_pressure(rho)
    pressure_to_wav_file(p, path)
