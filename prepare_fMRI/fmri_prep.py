import sys

sys.path.append('/Users/marialauradegrazia/Desktop/tesi_elide_robin/better_tesi')
from utils import compute_FCD

BOLD_DIR = '../dataset fMRI/Data_to_share/BOLD_TS'
MOUSE_ID = 'ag171031a'
REGIONS_LABELS_FILE = '../dataset fMRI/Data_to_share/macro_atlas_n_172_rois_excel_final.xlsx'

compute_FCD(BOLD_DIR, MOUSE_ID, REGIONS_LABELS_FILE, plot_FCD=True)
