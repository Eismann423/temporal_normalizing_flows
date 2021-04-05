#Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()

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

file = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\raw\csv\9 white LR first.csv"
xls = pd.read_csv(file)
# print(xls)
x_force = xls.loc[:,'X'].values
print(x_force.size)
time = np.arange(0, x_force.size - 1, .008)

#%% Time-dependent neural flow
x_sample = np.linspace(-15, 15, 1000)
t_sample = time
dataset = prepare_data(x_force, time, x_sample, t_sample)