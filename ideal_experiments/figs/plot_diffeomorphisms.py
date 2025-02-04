import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.plot_utils import create_diffeomorphism_plots
from utils.experiment import diffeomorphisms

create_diffeomorphism_plots(diffeomorphisms)