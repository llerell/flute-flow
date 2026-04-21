import numpy as np
import os
from scipy.io import wavfile


def save_to_wav(p: np.array, name: str):
    if not os.path.exists("sounds"):
        os.makedirs("sounds")
        
    p = p - np.mean(p)
    p = p / np.max(np.abs(p))
    wavfile.write(f"sounds/{name}.wav", 44100, (p * 32767).astype(np.int16))

if __name__ == "__main__":
    # Example usage
    t = np.linspace(0, 2, 44100 * 2)  # 2 seconds at 44100 Hz
    p = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    save_to_wav(p, "example_sound")