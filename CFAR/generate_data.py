import numpy as np
import pandas as pd
import time
import os

# Check if data needs to be generated
if not os.path.exists("radarData.csv"):
    # Data generation code
    Fs = 3e6  # Sampling frequency in Hz
    T = 30  # Duration of the signal in seconds
    t = np.arange(0, T, 1 / Fs)  # Time vector
    target_distances = [1000, 2000, 3000, 4000, 5000]
    target_speeds = [20, -15, 30, 25, -10]
    target_reflectivities = [0.55, 0.8, 0.6, 0.7, 0.9]
    radar_frequency = 5.6e9
    c = 3e8
    tx_signal = np.cos(2 * np.pi * radar_frequency * t)
    rx_signal = np.zeros_like(t)
    for dist, speed, reflectivity in zip(target_distances, target_speeds, target_reflectivities):
        time_delay = 2 * dist / c
        doppler_shift = 2 * radar_frequency * (speed / c)
        echo = reflectivity * np.cos(2 * np.pi * radar_frequency * (t - time_delay) + 2 * np.pi * doppler_shift * t)
        rx_signal += echo
    rx_signal += 0.03 * np.random.randn(*rx_signal.shape)
    np.savetxt('radarData.csv', np.column_stack((t, rx_signal)), delimiter=',')
