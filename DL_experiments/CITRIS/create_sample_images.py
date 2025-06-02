import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams["figure.figsize"] = [10, 5]

data_dir ='data/causal3d_time_dep_all7_conthue_01_coarse'
fig_dir = os.path.join(data_dir, 'figs')
causal3d_dataset = dict(
    np.load(os.path.join(data_dir, 'test.npz'))
)
print(causal3d_dataset.keys())

seen = defaultdict(int)

for i in range(10000):
    print(causal3d_dataset['imgs'][i].shape)
    shape = causal3d_dataset['shape_latents'][i][0]
    if seen[shape] <= 5:
        plt.imsave(os.path.join(fig_dir, f'sample_{shape}_{i+1}.png'), causal3d_dataset['imgs'][i])
        seen[shape] += 1