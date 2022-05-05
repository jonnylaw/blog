# %% Simulate a regression model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_data(xs):
    a = 5
    b = 3
    mean = lambda x: a * np.sin(x + b)
    data = [(x, np.random.normal(loc=mean(x), scale=1, size=1))
            for x in xs]
    x, y = zip(*data)
    return np.array(x), np.array(y)


xs = np.random.uniform(0, 10, size=100)
x, y = generate_data(xs)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

plt.scatter(x, y)

# %%
import torch.nn as nn
import torch
import torch.nn.functional as F

class NonLinearRegression(nn.Module):
    def __init__(self, dropout, hidden_size, hidden_layers):
        super(NonLinearRegression, self).__init__()
        self.input = nn.Linear(1, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList(
          [nn.Linear(hidden_size, hidden_size) for i in range(hidden_layers)])
        self.dropouts = nn.ModuleList(
          [nn.Dropout(dropout) for i in range(hidden_layers)])
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.dropout(x)
        for l, d in zip(self.linears, self.dropouts):
          x = l(x)
          x = F.relu(x)
          x = d(x)
        x = self.output(x)
        return x

# %% train the model determinisitically
model = NonLinearRegression(dropout=0.1, hidden_size=64, hidden_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(500):
    x_in = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_in = torch.tensor(y_train, dtype=torch.float32)
    y_pred = model(x_in)
    loss = F.mse_loss(y_pred, y_in)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% In domain predictions
x = torch.tensor(np.linspace(0, 10, 100), dtype=torch.float32).unsqueeze(-1)
y_preds = model(x)

y_preds

plt.figure();
x_plot = x.detach().numpy().squeeze()
plt.plot(x_plot, 5 * np.sin(x_plot + 3), 'b:', label=r'$f(x) = 5\,\sin(x + 3)$')
plt.plot(x_train, y_train, 'b.', markersize=10, label='Observations')
plt.plot(x_plot, y_preds.detach().squeeze(), 'r.', label='Prediction')
plt.legend()
plt.savefig('nn_no_uncertainty.png', dpi=300)

# %% Make predictions
x_ood = torch.tensor(np.arange(-5, 15, 0.1), dtype=torch.float32).unsqueeze(-1)
y_preds = model(x_ood)

y_preds

# %%
plt.figure();
x = np.linspace(0, 10, 100)
x_ood_plot = x_ood.detach().numpy().squeeze()
plt.plot(x_ood_plot, 5 * np.sin(x_ood_plot + 3), 'b:', label=r'$f(x) = 5\,\sin(x + 3)$')
plt.plot(x_train, y_train, 'b.', markersize=10, label='Observations')
plt.plot(x_ood_plot, y_preds.detach().squeeze(), 'r.', label='Prediction')
plt.legend()
plt.savefig('ood_prediction.png', dpi=300)

# %%
def get_probability_interval(preds, interval):
  lower_int = (1. - interval) / 2
  upper_int = interval + lower_int

  # Calculate percentiles and mean
  lower = np.quantile(preds, lower_int, axis=0)
  mean = np.mean(preds, axis=0)
  upper = np.quantile(preds, upper_int, axis=0)
    
  return lower, mean, upper

def plot_predictions(x_true, y_true, x_in, y_pred, interval=0.89):
    lower, mean, upper = get_probability_interval(y_pred, interval)
    print(lower.shape, mean.shape, upper.shape)
    plt.scatter(x_true, y_true, label="Observed", color='b')
    plt.plot(x_in, 5 * np.sin(x_in + 3), 'b:', label=r'$f(x) = 5\,\sin(x + 3)$')
    plt.plot(x_in, mean, 'r-', label="Prediction")
    plt.fill(
        np.concatenate([x_in, x_in[::-1]]),
        np.concatenate([lower, upper[::-1]]),
        alpha=.5, fc='r', ec='None', label='89% CI');
    plt.legend(loc="upper right")

# %% MC Dropout predictions
model.train()

def get_predictions(model, x, n):
  # Make multiple predictions
  mus = [model(x) for _ in range(n)]

  # Sample from the observation distribution using each mean and the global sigma
  return torch.stack(mus)

x_plot = torch.tensor(np.linspace(-5, 15, 100), dtype=torch.float32).unsqueeze(-1)
y_preds = get_predictions(model, x_plot, n=1000)

# get_probability_interval(y_preds.detach().numpy(), 0.89)

plot_predictions(x_train, y_train, x_plot.detach().numpy().squeeze(), y_preds.detach().numpy())
plt.savefig('mc-dropout-regression.png', dpi=300)

# %% Define the model
import torch.nn as nn
import torch
import torch.nn.functional as F
import pyro
from pyro.nn.module import PyroModule, PyroSample
import pyro.distributions as dist

class NonLinearRegression(PyroModule):
    def __init__(self, dropout, hidden_size, hidden_layers):
        super(NonLinearRegression, self).__init__()
        self.input = PyroModule[nn.Linear](1, hidden_size)
        self.dropout = PyroModule[nn.Dropout](dropout)
        self.linears = PyroModule[nn.ModuleList](
          [PyroModule[nn.Linear](hidden_size, hidden_size) for i in range(hidden_layers)])
        self.dropouts = PyroModule[nn.ModuleList](
          [PyroModule[nn.Dropout](dropout) for i in range(hidden_layers)])
        self.output = PyroModule[nn.Linear](hidden_size, 1)

        for m in self.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, PyroSample(prior=dist.Normal(0, 1)
                                                    .expand(value.shape)
                                                    .to_event(value.dim())))
        
    def forward(self, x, y=None):
        sigma = pyro.sample('sigma', dist.InverseGamma(3, 1))
        x = self.input(x)
        x = F.relu(x)
        x = self.dropout(x)
        for l, d in zip(self.linears, self.dropouts):
          x = l(x)
          x = F.relu(x)
          x = d(x)
        mean = self.output(x)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
        return mean

regression = NonLinearRegression(0.01, 16, 1)

# %%
from pyro.nn import PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO

guide = AutoDiagonalNormal(regression)
adam = pyro.optim.AdamW({"lr": 0.003})
svi = SVI(regression, guide, adam, loss=Trace_ELBO())

# %%
from tqdm import tqdm
x_in = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
y_in = torch.tensor(y_train, dtype=torch.float32)
pyro.clear_param_store()
num_iters = 3000
for j in tqdm(range(num_iters), total=num_iters):
    # calculate the loss and take a gradient step
    loss = svi.step(x_in, y_in)

# %% Plot the posterior predictive distribution
from pyro.infer import Predictive

x_in = torch.sort(torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1), dim=0)[0]
predictive = Predictive(regression, guide=guide, num_samples=100)
samples = predictive(x_in)

def get_probability_interval(preds, interval):
  lower_int = (1. - interval) / 2
  upper_int = interval + lower_int

  # Calculate percentiles and mean
  lower = np.quantile(preds, lower_int, axis=0)
  mean = np.mean(preds, axis=0)
  upper = np.quantile(preds, upper_int, axis=0) 
    
  return lower, mean, upper

def plot_predictions(x_true, y_true, x_in, y_pred, interval=0.89):
    lower, mean, upper = get_probability_interval(y_pred, interval)
    plt.scatter(x_true, y_true, label="Observed")
    plt.errorbar(x_in.squeeze(), mean, yerr=np.array(lower[0], upper[0]), fmt="r.", alpha=0.5, label=f"{interval * 100}% CI")
    plt.legend(loc="upper left")
    plt.savefig('posterior_predictive.png', dpi=300)

plot_predictions(x_test, y_test, x_in, samples["obs"].squeeze().numpy(), interval=0.95)

# %%
import pandas as pd

predictive = Predictive(regression, guide=guide, num_samples=100)
posterior_samples = predictive(x_in)

samples_df = pd.DataFrame(posterior_samples["obs"].squeeze()).T
samples_df['y_value'] = y_test
samples_df.set_index('y_value', inplace=True)

# Pivot the dataframe to x_test, y_pred
samples_df = samples_df.stack()
samples_df = pd.DataFrame(samples_df, columns=['y_pred'], index=samples_df.index)

# Convert the index to a column
samples_df.reset_index(inplace=True)

samples_df['y_pred'] = samples_df['y_pred'].astype(float)
samples_df['y_value'] = round(samples_df['y_value'].astype(float), 2)

# %% Use plotnine
from plotnine import *

fig = ggplot(samples_df) + \
    geom_density(aes(x = 'y_pred')) + \
    geom_vline(aes(xintercept = 'y_value'), linetype='dashed') + \
    facet_wrap("~ y_value", scales="free_x") + \
    theme_minimal()

fig.save('marginal_posteriors.png', dpi=300)

# %% Using NUTS to fit the network
from pyro.infer import NUTS, MCMC

x_in = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
y_in = torch.tensor(y_train, dtype=torch.float32)
kernel = NUTS(regression)
mcmc = MCMC(kernel, num_samples=1000, warmup_steps=500)
mcmc.run(x_in, y_in)
samples = mcmc.get_samples()

# %% posterior diagnostics
mcmc.diagnostics()

# %% Plot the posterior predictive distribution using an evenly spaced set of test points
x_plot = torch.tensor(np.arange(0, 10, 0.1), dtype=torch.float32)
predictive = Predictive(regression, posterior_samples=samples)
posterior_samples = predictive(x_plot.reshape((100, 1)))

plot_predictions(x_test, y_test, x_plot, posterior_samples["obs"].squeeze().numpy(), interval=0.95)


# %% Plot the marginal posterior predictive distributions for each test point
import pandas as pd
from plotnine import *

x_test_in = torch.tensor(x_test.reshape((20, 1)), dtype=torch.float32)
predictive = Predictive(regression, posterior_samples=samples)
posterior_samples = predictive(x_test_in)

samples_df = pd.DataFrame(posterior_samples["obs"].squeeze().numpy()).T
samples_df['y_value'] = y_test
samples_df.set_index('y_value', inplace=True)

# Pivot the dataframe to x_test, y_pred
samples_df = samples_df.stack()
samples_df = pd.DataFrame(samples_df, columns=['y_pred'], index=samples_df.index)

# Convert the index to a column
samples_df.reset_index(inplace=True)

samples_df['y_value'] = round(samples_df['y_value'].astype(float), 2)

samples_df

# %%
fig, plot = ggplot(samples_df) + \
    geom_density(aes(x = 'y_pred')) + \
    geom_vline(aes(xintercept = 'y_value'), linetype='dashed') + \
    facet_wrap("~ y_value", scales="free_x") + \
    theme_minimal()

fig.savefig('marginal_posteriors.png', dpi=300)

# %%
