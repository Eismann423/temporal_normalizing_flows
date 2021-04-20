#Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
sns.set()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#from diffusioncharacterization.ctrw.random_walks import advection_diffusion_random_walk
from src.temporal_normalizing_flows.neural_flow import neural_flow
from src.temporal_normalizing_flows.latent_distributions import gaussian
from src.temporal_normalizing_flows.preprocessing import prepare_data

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass


#%% Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path_source = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\raw\csv"

maxLength = 0
for subdir, dirs, files in os.walk(path_source):
    for filename in files:
        xls = pd.read_csv(os.path.join(subdir, filename))
        if np.array([xls.loc[:, 'X'].values]).shape[1] > maxLength:
            maxLength = np.array([xls.loc[:, 'X'].values]).shape[1]
            name = os.path.join(subdir, filename)

force = []
for subdir, dirs, files in os.walk(path_source):
    for filename in files:
        xls = pd.read_csv(os.path.join(subdir, filename))
        x = np.array([xls.loc[:, 'X'].values])
        y = np.array([xls.loc[:, 'Y'].values])
        z = np.array([xls.loc[:, 'Z'].values])

        x.resize(maxLength)
        y.resize(maxLength)
        z.resize(maxLength)

        force.append(x.tolist())
        force.append(y.tolist())
        force.append(z.tolist())

vforce = np.vstack(force)
vforce_T = vforce.T

freq = .008
endTime = len(max(vforce, key=len)) * freq
time = np.arange(0, endTime, freq)
print(time.shape)
print(vforce_T.shape)


#%% Time-dependent neural flow
x_sample = np.linspace(-25, 200, 1000)
t_sample = time
dataset = prepare_data(vforce_T, time, x_sample, t_sample)

flow = neural_flow(gaussian)
flow.train(dataset, 10000)

px, pz, jacob, z = flow.sample(dataset)

plt.contourf(px)
plt.xlabel('x')
plt.ylabel('t')