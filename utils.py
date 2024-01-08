import numpy as np
from scipy.io import loadmat
import pandas as pd
from plots import plot_fc, plot_bold_timeseries
import matplotlib.pyplot as plt
from tvb.simulator.lab import *

from rww_exc_inh.rww_model_mary import *


def compute_FC_mouse(bold_dir, mouse_id, labels_file, plot_bold=False, plot_FC=False):
    data_dir = os.path.join(bold_dir, mouse_id + '_WT_tsc2_smoothed_n172.mat')
    data_mat = loadmat(data_dir)
    bold_data = data_mat[mouse_id + '_WT_tsc2_smoothed']
    regions_labels = list(pd.read_excel(io=labels_file)['NAME'])
    regions_labels_tot = ['Right ' + reg for reg in regions_labels] + ['Left ' + reg for reg in regions_labels]

    # calculates timeseries and bold
    output_folder_ts = 'Timeseries'
    os.makedirs(output_folder_ts, exist_ok=True)
    bold_data = pd.DataFrame(bold_data)
    bold_data.columns = regions_labels_tot
    if plot_bold is True:
        plot_bold_timeseries(bold_ts=bold_data, mouse_id=mouse_id, show=True).write_html(
            output_folder_ts + '/BOLD_timeseries_' + mouse_id + '.html')

    # calculates empirical FC
    output_folder_fc = 'Empirical_FC'
    os.makedirs(output_folder_fc, exist_ok=True)
    fc_emp = np.corrcoef(bold_data, rowvar=False)
    if plot_FC is True:
        plot_fc(FC_mat=fc_emp, regions_labels=bold_data.columns, mouse_id=mouse_id, show=True).write_html(
            output_folder_fc + '/fc_' + mouse_id + '.html')

    return bold_data, fc_emp


def compute_FC_exp(conn, a, J_i, w_p, W_i, plot_bold=True, plot_FC=True):
    nreg = len(conn.weights)
    rww = ReducedWongWangExcIOInhI(J_i=np.array(J_i), w_p=np.array(w_p), W_i=np.array(W_i))
    sim = simulator.Simulator(
        model=rww,
        connectivity=conn,
        coupling=coupling.Scaling(a=np.array(a)),
        monitors=(monitors.TemporalAverage(period=1.0),
                  monitors.Bold(period=15)),
        integrator=integrators.HeunStochastic(
            noise=noise.Additive(nsig=np.array([0.015 ** 2 / 2])),
            dt=0.1
        ),
    )
    sim.initial_conditions = np.random.uniform(0.0, 0.2, size=(2, 4, nreg, 1))
    sim.configure()
    (t1, tavg), (t2, bold) = sim.run(simulation_length=1000)

    # calculates timeseries and bold
    output_folder_ts = 'Timeseries'
    os.makedirs(output_folder_ts, exist_ok=True)
    if plot_bold is True:
        plot_bold_timeseries(bold_ts=bold, show=True).write_html(
            output_folder_ts + '/BOLD_timeseries_experimental.html')

    # calculates empirical FC
    output_folder_fc = 'Experimental_FC'
    os.makedirs(output_folder_fc, exist_ok=True)
    fc_exp = np.corrcoef(bold, rowvar=False)
    if plot_FC is True:
        plot_fc(FC_mat=fc_exp, regions_labels=conn.region_labels, show=True).write_html(
            output_folder_fc + '/fc_experimental.html')

    return bold, fc_exp


def compute_FCD(bold_dir, mouse_id, labels_file, plot_FCD=True):
    # windowed dynamical FC
    # define window size
    tau = 60  # [s] window size
    w_step = 20  # [s] window step

    # t_range = np.arange(0,bold_data.shape[0],1)
    FC_windows = []
    triu_FC_windows = []

    bold_data, fc_emp = compute_FC_mouse(bold_dir, mouse_id, labels_file)
    for t in range(0, bold_data.shape[0], w_step):
        bold_window = bold_data.iloc[t:t + tau]
        fc = np.corrcoef(bold_window, rowvar=False)
        FC_windows.append(fc)
        triu_FC_windows.append(np.triu(fc))

    n_windows = len(triu_FC_windows)
    fcd = np.zeros(shape=(n_windows, n_windows))

    for i in range(n_windows):
        for j in range(n_windows):
            fcd[i][j] = np.corrcoef(triu_FC_windows[i].ravel(), triu_FC_windows[j].ravel())[0, 1]

    if plot_FCD is True:
        plt.imshow(fcd, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Functional Dynamical Connectivity Matrix')
        plt.show()

    return fcd


def compute_FCD_exp(conn, a, J_i, w_p, W_i, plot_FCD=True):
    tau = 60  # [s] window size
    w_step = 20  # [s] window step

    # t_range = np.arange(0,bold_data.shape[0],1)
    FC_windows = []
    triu_FC_windows = []

    bold_data, fc_exp = compute_FC_exp(conn, a, J_i, w_p, W_i)
    for t in range(0, bold_data.shape[0], w_step):
        bold_window = bold_data.iloc[t:t + tau]
        fc = np.corrcoef(bold_window, rowvar=False)
        FC_windows.append(fc)
        triu_FC_windows.append(np.triu(fc))

    n_windows = len(triu_FC_windows)
    fcd = np.zeros(shape=(n_windows, n_windows))

    for i in range(n_windows):
        for j in range(n_windows):
            fcd[i][j] = np.corrcoef(triu_FC_windows[i].ravel(), triu_FC_windows[j].ravel())[0, 1]

    if plot_FCD is True:
        plt.imshow(fcd, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Functional Dynamical Connectivity Matrix')
        plt.show()

    return fcd
