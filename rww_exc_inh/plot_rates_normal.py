import numpy as np
import matplotlib.pyplot as plt

with np.load("firing_rate_normal.npz") as rates:
    t = rates['time']
    data = rates['tavg']
    a_range = rates['a']
    Ji_range = rates['J_i']

# Matrix access in TVB style format
# S_e = data[:, 0, :, 0]
# S_i = data[:, 1, :, 0]
# R_e = data[:, 2, :, 0]
# R_i = data[:, 3, :, 0]


# Matrix access in parameter-sweep format
for ia, a in enumerate(a_range):
    for iJi, Ji in enumerate(Ji_range):
        S_e = data[ia, iJi, 0]  # a, J_i, state variable
        S_i = data[ia, iJi, 1]
        R_e = data[ia, iJi, 2]
        R_i = data[ia, iJi, 3]
        plt.figure()
        for regS_e in S_e:
            plt.plot(t, regS_e, color='r', label='S_E')
        for regS_i in S_i:
            plt.plot(t, regS_i, color='b', label='S_I')
        plt.title('Synaptic gating variables for a='+str(a)+' J_i='+str(Ji))
        plt.legend()

        plt.figure()
        for regR_e in R_e:
            plt.plot(t, regR_e, color='r', label='r_E')
        plt.title('Firing rates E for a='+str(a)+' J_i='+str(Ji))
        plt.legend()

        plt.figure()
        for regR_i in R_i:
            plt.plot(t, regR_i, color='b', label='r_I')
        plt.title('Firing rates I for a='+str(a)+' J_i='+str(Ji))
        plt.legend()
        plt.show()

print(np.mean(np.mean(R_e[100:, :], axis=1)))
print(np.mean(np.mean(R_i[100:, :], axis=1)))
