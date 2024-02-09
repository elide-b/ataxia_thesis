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
from connectivity import load_mousebrain


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
    return regions_labels_tot, bold_data

def plot_connectivity(conn, show=True, save = False):
    plt.title('Connectivty weigths')
    plt.imshow(conn.weights)
    plt.colorbar()
    if save:
        plt.savefig('conn.png')
    if show:
        plt.show()

def plot_fc(FC_mat, show=False,save = True):
    plt.title('Simulated FC')
    plt.imshow(FC_mat)
    plt.colorbar()
    if save:
        plt.savefig('fc.png')
    if show:
        plt.show()

def plot_timeseries(t,data,show=False, save = True):
    plt.plot(t,data)
    if save:
        plt.savefig('timeseries.png')
    if show:
        plt.show()

def prepare_conn(conn, normalize=True, log_trans=True, percitile=99, dt=0.1, show=True):

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

    # Normalize between 0 and 1
    if normalize:
        conn.weights = conn.weights / np.max(conn.weights)
    # Scale for the given percentile
    #conn.weights /= np.percentile(conn.weights, percitile)

    # tract lenghts check
    conn.speed = np.array(3.)
    conn.tract_lengths = np.maximum(conn.speed * dt, conn.tract_lengths)
    if show:
        plot_connectivity(conn)

    return conn


def simulate(conn,
             dt=0.1,
             G=1.,
             sigma=0.015,
             bold_resolution=1000,
             tavg_resolution=1.,
             sim_len=1000):  # 1869 * 1000 to match gozzi acquisitions
    simulator = Simulator()
    simulator.connectivity = conn
    simulator.model = ReducedWongWangExcInh()
    simulator.model.dt = dt
    simulator.coupling = Scaling(a=np.array(G))
    simulator.initial_conditions = (0.001) * np.random.uniform(size=(2, 2, len(conn.weights), 1))
    simulator.integrator = HeunStochastic(dt=dt)
    simulator.integrator.noise = noise.Additive(nsig=np.array([(sigma ** 2) / 2]))
    mon_bold = Bold(period=dt * bold_resolution)
    #mon_tavg = TemporalAverage(period=dt*tavg_resolution)
    simulator.monitors = (mon_bold,)
    simulator.configure()
    print('Starting sim...')
    start_time = time.time()
    (t1, bold), = simulator.run(simulation_length=sim_len)
    end_time = time.time()
    print('End sim!')
    print('Simulation time: ', end_time - start_time)
    return t1, bold

fcds = []

def compute_fcd(bold_data, tau=60, w_step=20):
    global fcds
    FC_windows = []
    triu_FC_windows = []

    for t in range(0, bold_data.shape[0], w_step):
        bold_window = bold_data.iloc[t:t + tau]
        fc = np.corrcoef(bold_window, rowvar=False)
        FC_windows.append(fc)
        triu_FC_windows.append(np.triu(fc))

    n_windows = len(triu_FC_windows)

    fcd = np.zeros(shape=(n_windows, n_windows))

    print("WINDOW DATA:", len(triu_FC_windows))

    for i in range(n_windows):
        for j in range(n_windows):
            fcd[i][j] = np.corrcoef(triu_FC_windows[i].ravel(), triu_FC_windows[j].ravel())[0, 1]
    fcds.append(triu_FC_windows)
    return fcd

def common_regions(regions1, regions2):
    common_regs = [reg for reg in regions1 if reg in regions2]
    ncommon = len(common_regs)
    print('Common regions FC emp vs sim: ', ncommon)
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

    #empFC_wb = np.triu(conn1, 1)  # triangle matrix
    #simFC_wb = np.triu(conn2, 1)
    #pcc_wb = np.corrcoef(empFC_wb.ravel(), simFC_wb.ravel())[0, 1]

    return pp


INPUT_PATH = './gozzi/results/merged_oh'
conn = Connectivity()
conn.weights = np.loadtxt(INPUT_PATH + '/weights_merged_oh.txt')
conn.centres = np.loadtxt(INPUT_PATH + '/centres_merged_oh.txt')
conn.region_labels = np.genfromtxt(INPUT_PATH + '/region_labels_merged_oh.txt', dtype=str, delimiter='\n')
conn.tract_lengths = np.loadtxt(INPUT_PATH + '/tract_lengths_merged_oh.txt')

final_conn = prepare_conn(conn, show=False, normalize=True, log_trans=True)
#plot_connectivity(final_conn,show=False,save=True)


transient = 4
t1, bold = simulate(final_conn, sim_len=1000*1869, dt=1, G=0.01, bold_resolution=1000)
fc_sim = np.corrcoef(bold[transient:, 0, :, 0], rowvar=False)
print('Point BOLD: ', bold.shape[0])
plot_fc(fc_sim, show=True)
#np.save("bold.npy", bold)
#np.save("time.npy", t1)
#plot_timeseries(t1[4:],bold[4:,0,:,0], show=True)
#plot_timeseries(t2[4:],tavg[4:,0,:,0],show=True)


regions_gozzi, bold_emp = load_emp()

# empirical static FC
fc_emp = np.corrcoef(bold_emp, rowvar=False)
corr_fc = predictive_power(fc_sim, fc_emp, final_conn.region_labels, regions_gozzi)
print('Predictive power sim vs emp (Static FC):', corr_fc)


# levare da region gozzi il corrispondente del caudoputamen e sostituirlo con la relativa area di gozzi (sostituito in
# region_labels_oh_merged.txt)

bold_emp = bold_emp[:len(t1)]  # just to try with a very short simulation
to_delate = ['Right Hippocampo-amygdalar transition area', 'Left Hippocampo-amygdalar transition area']
inds_to_delate = []
for i, reg in enumerate(regions_gozzi):
    if reg in to_delate:
        inds_to_delate.append(i)

bold_emp = bold_emp.drop(bold_emp.columns[inds_to_delate], axis=1)

bold_sim_common = pd.DataFrame(bold[:, 0, :, 0])

drop_inds = []
for i, reg in enumerate(conn.region_labels):
    if reg not in regions_gozzi:
        drop_inds.append(i)

bold_sim_common = bold_sim_common.drop(bold_sim_common.columns[drop_inds], axis=1)

fcd_emp = compute_fcd(bold_emp)
fcd_sim = compute_fcd(bold_sim_common)
plot_fc(fcd_sim, show=True, save=False)
n = fcd_sim.shape[0]
fcd_sim = fcd_sim[np.triu_indices(n, 1)]
fcd_emp = fcd_emp[np.triu_indices(n, 1)]

print("Are we crazy?")
print("Shapes:", fcd_sim.shape, fcd_emp.shape)
print("Mean:", fcd_sim.mean(), fcd_emp.mean())

# Compute Kolmogorov distance
KS_1d = ks_2samp(fcd_emp.ravel(), fcd_sim.ravel())
KS = ks_2samp(fcd_emp, fcd_sim)
print("1d problem?", KS, KS_1d)
import plotly.graph_objs as go

go.Figure([go.Histogram(x=fcd_sim, name=f"Simulated FCD ({fcd_sim.shape})"),
           go.Histogram(x=fcd_emp, name=f"Empirical FCD ({fcd_emp.shape})")]).write_html(f"ks_{KS[0]}.html")
print('Kolmogorov distance FCD sim vs emp: ', KS)
