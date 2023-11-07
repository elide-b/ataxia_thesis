import numpy as np
import matplotlib.pyplot as plt

"""3D parameter plotting"""

with np.load("opt_parameters_healthy.npz") as rates:
    t = rates['time']
    data = rates['tavg']
    a_range = rates['a']
    Ji_range = rates['J_i']
    wp_range = rates['w_p']
    wi_range = rates['W_i']

for ia, a in enumerate(a_range):
    for iJi, Ji in enumerate(Ji_range):
        for iwp, wp in enumerate(wp_range):
            for iwi, wi in enumerate(wi_range):
                S_e = data[ia, iJi, iwp, iwi, 0]  # a, J_i, w_p, W_i, state variable
                S_i = data[ia, iJi, iwp, iwi, 1]
                R_e = data[ia, iJi, iwp, iwi, 2]
                R_i = data[ia, iJi, iwp, iwi, 3]
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
