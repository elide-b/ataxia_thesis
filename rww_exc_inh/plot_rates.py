import matplotlib.pyplot as plt
import numpy as np

with np.load("firing_rate_normal.npz") as rates:
    t = rates['time']
    data = rates['tavg']

S_e = data[:, 0, :, 0]
S_i = data[:, 1, :, 0]
R_e = data[:, 2, :, 0]
R_i = data[:, 3, :, 0]

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

print(np.mean(np.mean(R_e[100:, :], axis=1)))
print(np.mean(np.mean(R_i[100:, :], axis=1)))
