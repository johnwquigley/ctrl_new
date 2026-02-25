from matplotlib.pyplot import *


style.use(['dark_background', 'bmh'])
rc('axes', facecolor='k')
rc('figure', facecolor='k', figsize=(10, 6), dpi=100)  # (17, 10)
rc('savefig', bbox='tight')
rc('axes', labelsize=36, titlesize=36)
rc('legend', fontsize=24)
rc('text', usetex=True)
rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('lines', markersize=10)