In order to obtain the firing rates of the regions of the brain, we modified the original code of the Reduced Wong Wang
model (contained in rww_model_mary.py).

S = synaptic gating variable, which describes the effectiveness of synaptic transmission.\
R = firing rate, which represents the average rate at which neurons in a population are generating action potentials.

From the temporal average data, to access them for the inhibitory (i) and excitatory (e) population:\
S_e = data[:, 0, :, 0]\
S_i = data[:, 1, :, 0]\
R_e = data[:, 2, :, 0]\
R_i = data[:, 3, :, 0]