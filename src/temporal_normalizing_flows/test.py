#Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
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


# Making random walk data

# We start by creating a dataset:


def advection_diffusion_random_walk(walk_params, traj_params, initial_conditions):
    num_steps, num_walkers, dt = walk_params
    Diff, v = traj_params

    steps = np.random.normal(loc=v*dt, scale=np.sqrt(2*Diff*dt), size=(num_steps, num_walkers))
    trajectory = np.concatenate((initial_conditions, initial_conditions + np.cumsum(np.array(steps), axis=0)), axis=0)
    time = np.arange(num_steps + 1) * dt

    return time, trajectory

#%%

walk_params = [99, 500, 0.05]  # timesteps, walkers, stepsize
traj_params = [2.0, 0.0]       # Diffusion coefficient, velocity
initial_conditions = np.random.normal(loc=1.5, scale=0.5, size=(1, walk_params[1]))

time, position = advection_diffusion_random_walk(walk_params, traj_params, initial_conditions)


# We plot all the positions:

plt.figure(figsize=(8, 5))
plt.plot(time, position)

plt.xlabel('t')
plt.ylabel('x')
plt.xlim([0, 5])
plt.show()


# And make an estimate of the density:

frame = 50
sns.distplot(position[:, frame], bins='auto')
plt.title('t={}'.format(time[frame]))
plt.show()


# Temporal Normalizing Flow

# Now we make an estimate using the temporal normalizing flow. We first select the grid on which we calculate and prepare the dataset;

#%% Time-dependent neural flow
x_sample = np.linspace(-15, 15, 1000)
t_sample = time
dataset = prepare_data(position, time, x_sample, t_sample)


# Now we create a flow and train it:

flow = neural_flow(gaussian)
flow.train(dataset, 10000)


# We get our results by sampling the dataset:

px, pz, jacob, z = flow.sample(dataset)

#%%

plt.contourf(px)
plt.xlabel('x')
plt.ylabel('t')