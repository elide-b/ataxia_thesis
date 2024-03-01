import numpy as np
from matplotlib import pyplot as plt

from plots import plot_3d

with np.load("firing_rate_normal.npz") as rates:
    t = rates['time']
    data = rates['tavg']
    a_range = rates['a_range']
    Ji_range = rates['Ji_range']

# Matrix access in TVB style format
# S_e = data[:, 0, :, 0]
# S_i = data[:, 1, :, 0]
# R_e = data[:, 2, :, 0]
# R_i = data[:, 3, :, 0]

Mean_re = []
Mean_ri = []

# Matrix access in parameter-sweep format
for ia, a in enumerate(a_range):
    for iJi, Ji in enumerate(Ji_range):
        S_e = data[ia, iJi, 0]  # a, J_i, state variable, regions, time
        S_i = data[ia, iJi, 1]
        R_e = data[ia, iJi, 2]
        R_i = data[ia, iJi, 3]
        print("E activity for a = " + str(a) + " activity for J_i = " + str(Ji) + " = ",
              np.mean(np.mean(R_e[100:, :], axis=1)))
        print("I activity for a = " + str(a) + " activity for J_i = " + str(Ji) + " = ",
              np.mean(np.mean(R_i[100:, :], axis=1)))
        Mean_re.append(np.mean(np.mean(R_e[100:, :])))
        Mean_ri.append(np.mean(np.mean(R_i[100:, :])))
        plt.figure()
        for regS_e in S_e:
            plt.plot(t, regS_e, color='r')
        for regS_i in S_i:
            plt.plot(t, regS_i, color='b')
            plt.title('Synaptic gating for a= ' + str(a) + ' J_i= ' + str(Ji))

        plt.figure()
        for regR_e in R_e:
            plt.plot(t, regR_e)
        plt.title('Firing rates E for a= ' + str(a) + ' J_i= ' + str(Ji))

        plt.figure()
        for regR_i in R_i:
            plt.plot(t, regR_i)
        plt.title('Firing rates I for a= ' + str(a) + ' J_i= ' + str(Ji))
        plt.show()

plot_3d(x=a_range, y=Ji_range, z=Mean_re, z2=Mean_ri, xaxis='Global Coupling', yaxis='Inhibitory feedback',
        zaxis='Mean firing rates', multiple=True)
