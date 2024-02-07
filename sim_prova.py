from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import noise
from tvb.simulator.coupling import Linear, Scaling
from tvb.simulator.integrators import EulerDeterministic, HeunStochastic
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh
from tvb.simulator.monitors import Bold, TemporalAverage
from tvb.simulator.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
import os
import pandas as pd
from scipy.stats import ks_2samp



def load_emp(MOUSE_ID='ag171031a'):
    # dir for empirical data
    BOLD_DIR = './dataset fMRI/Data_to_share/BOLD_ts'
    REGIONS_LABELS_FILE = './dataset fMRI/Data_to_share/macro_atlas_n_172_rois_excel_final.xlsx'

    # load labels
    regions_labels = list(pd.read_excel(io=REGIONS_LABELS_FILE)['NAME'])
    regions_labels_tot = ['Right ' + reg for reg in regions_labels] + ['Left ' + reg for reg in regions_labels]
    # load bold data
    data_dir = os.path.join(BOLD_DIR, MOUSE_ID + '_WT_tsc2_smoothed_n172.mat')
    data_mat = loadmat(data_dir)
    bold_data = data_mat[MOUSE_ID + '_WT_tsc2_smoothed']
    bold_data = pd.DataFrame(bold_data)
    # empirical static FC
    fc_emp = np.corrcoef(bold_data, rowvar=False)

    return regions_labels_tot, fc_emp, bold_data


def plot_connectivity(conn):
    plt.title('Connectivty weigths')
    plt.imshow(conn.weights)
    plt.colorbar()
    plt.show()


def plot_fc(FC_mat, show=True):
    plt.title('Simulated FC')
    plt.imshow(FC_mat)
    plt.colorbar()
    if show:
        plt.show()


def prepare_conn(conn, normalize=True, log_trans=True, percitile=99, dt=0.1, show=True):
    # Normalize between 0 and 1
    if normalize:
        conn.weights = conn.weights / np.max(conn.weights)
    # Apply log transform
    if log_trans:
        w = conn.weights.copy()
        w[np.isnan(w)] = 0.0  # zero nans
        w0 = w <= 0  # zero weights
        wp = w > 0  # positive weights
        w /= w[wp].min()  # divide by the minimum to have a minimum of 1.0
        w *= np.exp(1)  # multiply by e to have a minimum of e
        w[wp] = np.log(w[wp])  # log positive values
        w[w0] = 0.0  # zero zero values (redundant)
        conn.weights = w
    # Scale for the given percentile
    conn.weights /= np.percentile(conn.weights, percitile)

    # tract lenghts check
    conn.speed = np.array(3.)
    conn.tract_lengths = np.maximum(conn.speed * dt, conn.tract_lengths)
    if show:
        plot_connectivity(conn)

    return conn


def simulate(conn,
             dt=0.1,
             G=1,
             sigma=0.015,
             bold_resolution=1000,
             sim_len=1000):  # 1869 * 1000 to match gozzi acquisitions
    simulator = Simulator()
    simulator.connectivity = conn
    simulator.model = ReducedWongWangExcInh()
    simulator.model.dt = dt
    simulator.coupling = Scaling(a=np.array(G))
    simulator.initial_conditions = (0.001) * np.ones((2, 2, len(conn.weights), 1))
    simulator.integrator = HeunStochastic(dt=dt)
    simulator.integrator.noise = noise.Additive(nsig=np.array([(sigma ** 2) / 2]))
    mon_bold = Bold(period=dt * bold_resolution)
    simulator.monitors = (mon_bold,)
    simulator.configure()
    start_time = time.time()
    (t, bold), = simulator.run(simulation_length=sim_len)
    end_time = time.time()
    print('Simulation time: ', end_time - start_time)
    return t, bold


def compute_fcd(bold_data, tau=60, w_step=20):
    FC_windows = []
    triu_FC_windows = []

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
    return fcd


def common_regions(regions1, regions2):
    common_regs = [reg for reg in regions1 if reg in regions2]
    ncommon = len(common_regs)
    inds1 = [list(regions1).index(reg) for reg in common_regs]
    inds2 = [list(regions2).index(reg) for reg in common_regs]

    return inds1, inds2


def predictive_power(conn1, conn2, regions1=None, regions2=None):
    if regions1 is None and regions2 is None:
        assert conn1.shape == conn2.shape
        ncommon = conn1.shape[0]
        inds1 = inds2 = np.r_[:ncommon]
    else:
        inds1, inds2 = common_regions(regions1, regions2)
        ncommon = len(inds1)

    conn1 = conn1[inds1, :][:, inds1]
    conn2 = conn2[inds2, :][:, inds2]

    offdiag_mask = ~np.eye(ncommon, dtype=bool)

    pp = np.corrcoef(conn1[offdiag_mask],
                     conn2[offdiag_mask])[1, 0]

    return pp


INPUT_PATH = './gozzi/results/merged_oh'
conn = Connectivity()
conn.weights = np.loadtxt(INPUT_PATH + '/weights_merged_oh.txt')
conn.centres = np.loadtxt(INPUT_PATH + '/centres_merged_oh.txt')
conn.region_labels = np.genfromtxt(INPUT_PATH + '/region_labels_merged_oh.txt', dtype=str, delimiter='\n')
conn.tract_lengths = np.loadtxt(INPUT_PATH + '/tract_lengths_merged_oh.txt')

final_conn = prepare_conn(conn, show=False)
t_sim, bold_sim = simulate(conn)
regions_gozzi, fc_emp, bold_emp = load_emp()


#fc_sim = np.corrcoef(bold[:, 0, :, 0], rowvar=False)
#plot_fc(fc_sim, show=True)
#corr_fc = predictive_power(fc_sim, fc_emp, conn.region_labels, regions_gozzi)
#print(corr_fc)

# levare da region gozzi il corrispondente del caudoputamen e sostituirlo

bold_emp = bold_emp[:len(t_sim)]    # just to try with a very short simulation
to_delate = ['Right Hippocampo-amygdalar transition area', 'Left Hippocampo-amygdalar transition area']
inds_to_delate = []
for i, reg in enumerate(regions_gozzi):
    if reg in to_delate:
        inds_to_delate.append(i)

bold_emp = bold_emp.drop(bold_emp.columns[inds_to_delate], axis=1)

bold_sim_common = pd.DataFrame(bold_sim[:,0,:,0])

drop_inds = []
for i, reg in enumerate(conn.region_labels):
    if reg not in regions_gozzi:
        drop_inds.append(i)


bold_sim_common = bold_sim_common.drop(bold_sim_common.columns[drop_inds], axis=1)
fcd_emp = compute_fcd(bold_emp)
fcd_sim = compute_fcd(bold_sim_common)
test = ks_2samp(fcd_emp,fcd_sim)