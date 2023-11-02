import numpy as np
import matplotlib.pyplot as plt

with np.load("firing_rate.npz") as rates:
    t = rates['time']
    data = rates['tavg']

S_e = data[:, 0, 233, 0]
S_i = data[:, 1, 233, 0]
R_e = data[:, 2, 233, 0]
R_i = data[:, 3, 233, 0]

plt.figure()
plt.plot(t, S_e, color='r', label='S_E')
plt.plot(t, S_i, color='b', label='S_I')
plt.title('Synaptic gating variables')
plt.legend()
plt.show()

plt.figure()
plt.plot(t, R_e, color='r', label='r_E')
plt.plot(t, R_i, color='b', label='r_I')
plt.title('Firing rates')
plt.legend()
plt.show()